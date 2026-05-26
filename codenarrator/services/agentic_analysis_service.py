"""
Agentic analysis loop: an Ollama model drives file exploration via tool calls
instead of hardcoded heuristic scoring. Drop-in replacement for run_analysis_loop —
returns the same dict shape so the route needs no changes to its response handling.
"""
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import ollama

from codenarrator.services.analysis_snapshot_service import (
    _compute_dependency_graph_summary,
    _copy_state,
    _inspect_file,
    _newly_explored_file,
    _record_dependency_edge,
    _record_inspected_fact,
    _reduce_unknowns,
    _refresh_candidates_for_signal,
    _refine_summary,
    _resolve_internal_import,
    _resolved_import_targets,
    _update_confidence,
)
from codenarrator.services.repo_scanner import scan_repository
from codenarrator.core.config import settings

LOGGER = logging.getLogger(__name__)

OLLAMA_MODEL = settings.OLLAMA_MODEL
MAX_SEARCH_RESULTS = 10
MAX_FILE_PREVIEW_LINES = 40

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a source file from the repository. "
                "Returns its language, role, imports, and a content preview. "
                "Use this to understand what a file does and what it depends on."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to the file inside the repository.",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "follow_import",
            "description": (
                "Resolve and read a file that is imported by an already-explored file. "
                "Use this to trace the dependency chain."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "from_file": {
                        "type": "string",
                        "description": "The file that contains the import statement.",
                    },
                    "import_path": {
                        "type": "string",
                        "description": (
                            "The import specifier exactly as it appears in source "
                            "(e.g. './util' or 'app.services.foo')."
                        ),
                    },
                },
                "required": ["from_file", "import_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_for_pattern",
            "description": (
                "Search repository files for a regex pattern. "
                "Useful for finding where a function is defined or called, "
                "or locating configuration values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression to search for.",
                    },
                    "file_extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "File extensions to limit the search to, e.g. [\".js\", \".ts\"]. "
                            "Omit to search all supported files."
                        ),
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_architecture_insight",
            "description": (
                "Record a high-level architectural insight you have discovered. "
                "Use this to note entry points, components, patterns, or concerns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "insight_type": {
                        "type": "string",
                        "description": (
                            "Category: 'entry_point', 'component', 'pattern', "
                            "'dependency', or 'concern'."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": "Clear description of the architectural insight.",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File paths involved in this insight.",
                    },
                },
                "required": ["insight_type", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_analysis",
            "description": (
                "Stop the analysis when you have sufficient understanding of the architecture. "
                "Call this when further exploration would not meaningfully change your understanding."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Summary of what you now understand and why you are stopping.",
                    }
                },
                "required": ["reason"],
            },
        },
    },
]


def run_agentic_analysis_loop(
    initial_state: Dict,
    max_steps: int = 15,
    on_progress: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """
    Agentic replacement for run_analysis_loop.

    An Ollama model drives exploration via tool calls. The model sees a running
    message history with tool results fed back each step, so it can reason about
    what it has learned before deciding what to explore next.

    Returns the same dict shape as run_analysis_loop for drop-in compatibility.
    """
    steps_limit = max(1, min(max_steps, 25))
    state = _copy_state(initial_state)
    state.setdefault("dependency_graph_summary", {})

    # Cache the full repo file list once so tool functions don't re-scan on every call.
    repo_path = Path(state["current_summary"]["local_path"]).resolve()
    try:
        _cached_scan = scan_repository(repo_path)
        state["_cached_files"] = _cached_scan["files"]
    except Exception:
        state["_cached_files"] = []

    # Kept separate — not part of AnalysisState model shape.
    architecture_insights: List[Dict] = []
    initial_explored_len = len(state.get("explored_files", []))

    messages: List = [_build_system_message(state)]
    step_trace: List[Dict] = []
    consecutive_no_file_steps = 0
    # Total non-file-reading steps (search, mark_insight, nudge turns).
    # Capped at max_steps // 3 so non-exploring actions can't silently
    # consume the whole step budget. When the cap is hit, the loop
    # force-reads the next unexplored file just like the consecutive guard.
    total_non_file_steps = 0
    max_non_file_steps = max_steps // 3
    # Max recent messages to keep (excluding the system prompt at index 0).
    # Prevents context from growing unboundedly and slowing down Ollama calls.
    _MAX_HISTORY = 12

    for step in range(1, steps_limit + 1):
        if state.get("stop_reason"):
            break

        previous_explored = list(state["explored_files"])

        # After 2 consecutive steps with no file explored, force-read the next
        # unexplored candidate so the model can reason about it.
        if consecutive_no_file_steps >= 2:
            forced = _next_unexplored(state)
            if forced:
                LOGGER.info("Step %d: force-reading '%s' after %d silent steps.", step, forced, consecutive_no_file_steps)
                result, _ = _tool_read_file(state, forced)
                messages.append({
                    "role": "user",
                    "content": f"[Auto-read] {result}\n\nContinue exploring the remaining files.",
                })
                new_file = _newly_explored_file(previous_explored, state["explored_files"])
                step_trace.append(_trace_entry(step, new_file, state))
                if on_progress and new_file:
                    on_progress({"type": "progress", "file": new_file, "step": step,
                                 "explored": len(state["explored_files"]), "confidence": round(state["confidence"], 2)})
                consecutive_no_file_steps = 0
                continue
            else:
                # No unexplored files left — allow stop.
                LOGGER.info("Step %d: all files explored, stopping.", step)
                state["stop_reason"] = "All candidate files have been explored."
                break

        # Non-file budget guard: even if non-file steps aren't consecutive,
        # too many cumulatively still wastes the run. Force-read once the
        # cap is exceeded.
        if total_non_file_steps >= max_non_file_steps:
            forced = _next_unexplored(state)
            if forced:
                LOGGER.info(
                    "Step %d: force-reading '%s' — non-file budget (%d/%d) exhausted.",
                    step, forced, total_non_file_steps, max_non_file_steps,
                )
                result, _ = _tool_read_file(state, forced)
                messages.append({"role": "tool", "content": result, "tool_use_id": "force_read"})
                step_trace.append(_trace_entry(step, "force_read_budget", state))
                total_non_file_steps = 0  # reset after force-read
                consecutive_no_file_steps = 0
                continue

        # Trim history: always keep system prompt (index 0) + last _MAX_HISTORY messages.
        if len(messages) > _MAX_HISTORY + 1:
            messages = [messages[0]] + messages[-_MAX_HISTORY:]

        response = _call_model_with_retry(messages, retries=2)
        if response is None:
            LOGGER.warning("Step %d: Ollama unavailable after retries, stopping.", step)
            state["stop_reason"] = "Ollama unavailable after retries."
            step_trace.append(_trace_entry(step, None, state))
            break

        # Append assistant turn to history so the model sees its own reasoning.
        messages.append(response.message)

        tool_calls = _extract_tool_calls(response)
        if not tool_calls:
            LOGGER.info("Step %d: model returned no tool call — injecting nudge.", step)
            messages.append(_nudge_message(state))
            step_trace.append(_trace_entry(step, None, state))
            consecutive_no_file_steps += 1
            total_non_file_steps += 1
            continue

        explored_this_step: Optional[str] = None
        stop_this_step = False
        file_explored_this_step = False

        for tc in tool_calls:
            result, side_effect = _dispatch_tool(
                state=state,
                insights=architecture_insights,
                tool_name=tc.function.name,
                args=tc.function.arguments or {},
            )
            # Feed result back so the model can reason about what it learned.
            messages.append({"role": "tool", "content": result})

            if side_effect == "explored":
                new_file = _newly_explored_file(previous_explored, state["explored_files"])
                if new_file:
                    explored_this_step = new_file
                    file_explored_this_step = True
                    previous_explored = list(state["explored_files"])
                    if on_progress:
                        on_progress({"type": "progress", "file": new_file, "step": step,
                                     "explored": len(state["explored_files"]), "confidence": round(state["confidence"], 2)})
            elif side_effect == "stop":
                stop_this_step = True

        if file_explored_this_step:
            consecutive_no_file_steps = 0
        elif not stop_this_step:
            # Tool calls made but no new file explored — nudge and count.
            messages.append(_nudge_message(state))
            consecutive_no_file_steps += 1
            total_non_file_steps += 1

        step_trace.append(_trace_entry(step, explored_this_step, state))

        if stop_this_step or state.get("stop_reason"):
            break

    state["dependency_graph_summary"] = _compute_dependency_graph_summary(state)
    # Persist for diagnostic visibility in the cache. Pydantic's
    # AnalysisState ignores extras, so this survives JSON serialization to
    # disk but is stripped from the API response.
    state["total_non_file_steps"] = total_non_file_steps
    # Keep candidate_files fresh so AnalysisState validation passes.
    _refresh_candidates_for_signal(state, limit=8)

    explored_files_in_order = state["explored_files"][initial_explored_len:]

    return {
        "steps_executed": len(step_trace),
        "explored_files_in_order": explored_files_in_order,
        "step_trace": step_trace,
        "final_summary": state["current_summary"],
        "final_confidence": state["confidence"],
        "remaining_unknowns": state["unknowns"],
        "stop_reason": state.get("stop_reason"),
        "dependency_graph_summary": state["dependency_graph_summary"],
        "final_state": state,
    }


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _build_dir_tree(files: List[str], max_depth: int = 2) -> str:
    """
    Build a compact directory tree string from a flat file list.
    Shows unique directory paths up to max_depth levels deep.
    Example output:
      src/
        components/
        services/
      tests/
    """
    dirs: set = set()
    for f in files:
        parts = Path(f).parts
        for depth in range(1, min(len(parts), max_depth + 1)):
            if depth < len(parts):  # only directories, not files
                dirs.add(parts[:depth])

    if not dirs:
        return ""

    sorted_dirs = sorted(dirs)
    lines = []
    for parts in sorted_dirs:
        indent = "  " * (len(parts) - 1)
        lines.append(f"{indent}{parts[-1]}/")
    return "\n".join(lines)


def _build_system_message(state: Dict) -> Dict:
    summary = state["current_summary"]
    explored = state["explored_files"]
    candidates = state.get("candidate_files", [])
    unknowns = state.get("unknowns", [])
    cached_files = state.get("_cached_files", [])

    candidate_lines = "\n".join(
        f"  - {c['file_path']}  ({c.get('reason', '')})"
        for c in candidates[:15]
        if not _is_noise_file(c["file_path"])
    )
    explored_str = ", ".join(explored) if explored else "none yet"
    unknowns_str = "; ".join(unknowns) if unknowns else "none"

    # Compute visited vs unvisited directory sets so the agent doesn't have
    # to mentally parse "Already explored" to figure out where it hasn't been.
    explored_files = state.get("explored_files", [])
    all_files = state.get("_cached_files", [])
    explored_dirs = set(
        os.path.dirname(f) for f in explored_files if os.path.dirname(f)
    )
    all_dirs = set(
        os.path.dirname(f) for f in all_files if os.path.dirname(f)
    )
    unvisited_dirs = sorted(all_dirs - explored_dirs)
    visited_dirs = sorted(explored_dirs)

    min_files = min(15, max(6, int(summary["file_count"] * 0.65)))

    dir_tree = _build_dir_tree(cached_files)
    dir_tree_section = f"Repository structure:\n{dir_tree}\n\n" if dir_tree else ""

    content = (
        f"You are analyzing the architecture of the repository '{summary['repo']}'.\n"
        f"Total source files: {summary['file_count']} | "
        f"Languages: {', '.join(summary['languages'])}\n\n"
        f"{dir_tree_section}"
        f"Already explored: {explored_str}\n"
        f"Open questions: {unknowns_str}\n\n"
        f"Suggested starting candidates:\n{candidate_lines}\n\n"
        f"Visited directories: {', '.join(visited_dirs) if visited_dirs else 'none yet'}\n"
        f"NOT YET VISITED (prioritize these): {', '.join(unvisited_dirs) if unvisited_dirs else 'all covered'}\n\n"
        f"RULES (follow strictly):\n"
        f"1. You MUST call read_file or follow_import at least {min_files} times before "
        f"calling stop_analysis. Do not stop early.\n"
        f"2. After reading a file, always follow at least one of its imports with follow_import "
        f"to trace the dependency chain.\n"
        f"3. Actively spread across directories — after every 2-3 files you "
        f"read in the same directory, deliberately pick a file from a directory "
        f"you have NOT yet visited. Check the directory tree and identify which "
        f"top-level folders have zero explored files, then prioritize those. "
        f"A good analysis covers components/, utils/, hooks/, services/, api/, "
        f"types/ — not just the entry point chain.\n"
        f"4. Use mark_architecture_insight to record what you learn about entry points, "
        f"components, and patterns.\n"
        f"5. Only call stop_analysis after you have read at least {min_files} files AND "
        f"have a clear picture of the overall architecture.\n"
    )
    repo_path = Path(summary["local_path"])
    content += _build_language_guidance(summary["languages"], cached_files, repo_path)
    return {"role": "system", "content": content}


def _build_language_guidance(languages: List[str], files: List[str], repo_path: Path) -> str:
    """
    Return a language-specific guidance block to append to the system prompt.
    Each language gets 3-5 lines; framework signals (Next.js, Vite, package.json
    entry) further refine the JS/TS block. Returns "" when no recognized
    language is present.
    """
    blocks: List[str] = []
    file_set = set(files)

    if "python" in languages:
        blocks.append(
            "Python entry points to prioritize: app.py, main.py, manage.py, "
            "wsgi.py, asgi.py, __init__.py in the top-level package, "
            "settings.py, urls.py (Django), config.py. Follow import chains "
            "from these outward."
        )

    if "typescript" in languages or "javascript" in languages:
        js_lines: List[str] = []
        if "next.config.js" in file_set or "next.config.ts" in file_set:
            js_lines.append(
                "Next.js detected — prioritize app/, pages/, middleware.ts, "
                "layout.tsx, page.tsx files."
            )
        if "vite.config.ts" in file_set or "vite.config.js" in file_set:
            js_lines.append(
                "Vite detected — start at src/main.tsx or src/main.ts to "
                "understand the entry point, then SPREAD OUT: explore one "
                "file from each major directory (components/, utils/, hooks/, "
                "api/, types/) before going deeper into any single chain."
            )
        pkg_entry = _read_package_json_entry(repo_path)
        if pkg_entry:
            js_lines.append(f"package.json entry: {pkg_entry}")
        js_lines.append(
            "General JS/TS: prioritize index.ts, App.tsx, main.ts, router "
            "files, and any file named index.ts inside a directory (barrel "
            "files reveal the public API of that module). "
            "Do not follow a single import chain all the way down — read one "
            "file per directory before revisiting any directory."
        )
        blocks.append("\n".join(js_lines))

    if "java" in languages:
        blocks.append(
            "Java entry points: Main.java, Application.java (Spring Boot), "
            "files with public static void main, *Controller.java (routing "
            "layer), *Service.java (business logic layer)."
        )

    if "go" in languages:
        blocks.append(
            "Go entry points: main.go, cmd/*/main.go. Follow from main() "
            "outward through package imports."
        )

    if "rust" in languages:
        blocks.append(
            "Rust entry points: src/main.rs, src/lib.rs. mod declarations in "
            "these files define the module tree — follow mod statements first."
        )

    if not blocks:
        return ""
    return "\nLanguage-specific guidance:\n" + "\n\n".join(blocks) + "\n"


def _read_package_json_entry(repo_path: Path) -> Optional[str]:
    """
    Return the 'main' (preferred) or 'module' field from package.json at the
    repo root, or None if absent / unparseable. Defensive — any error returns
    None silently rather than poisoning the prompt build.
    """
    pkg = repo_path / "package.json"
    if not pkg.is_file():
        return None
    try:
        data = json.loads(pkg.read_text(encoding="utf-8", errors="ignore"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    for key in ("main", "module"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _dispatch_tool(
    state: Dict,
    insights: List[Dict],
    tool_name: str,
    args: Dict,
) -> Tuple[str, Optional[str]]:
    try:
        if tool_name == "read_file":
            return _tool_read_file(state, args.get("file_path", ""))
        if tool_name == "follow_import":
            return _tool_follow_import(
                state, args.get("from_file", ""), args.get("import_path", "")
            )
        if tool_name == "search_for_pattern":
            return _tool_search_for_pattern(
                state, args.get("pattern", ""), args.get("file_extensions")
            )
        if tool_name == "mark_architecture_insight":
            return _tool_mark_insight(
                insights,
                args.get("insight_type", ""),
                args.get("description", ""),
                args.get("files", []),
            )
        if tool_name == "stop_analysis":
            return _tool_stop(state, args.get("reason", ""))
        return f"Unknown tool: {tool_name}", None
    except Exception as exc:
        LOGGER.warning("Tool '%s' raised an exception: %s", tool_name, exc)
        return f"Tool error: {exc}", None


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_read_file(state: Dict, file_path: str) -> Tuple[str, Optional[str]]:
    if not file_path:
        return "Error: file_path is required.", None

    if file_path in state["explored_files"]:
        fact = next(
            (f for f in state["inspected_facts"] if f["file_path"] == file_path), None
        )
        if fact:
            return (
                f"Already explored '{file_path}': language={fact['language']}, "
                f"role={fact['role_hint']}, imports={fact['imported_modules'][:10]}",
                None,
            )

    repo_path = Path(state["current_summary"]["local_path"]).resolve()
    candidate_is_import_target = file_path in _resolved_import_targets(
        state,
        repo_path=repo_path,
        package_roots=state.get("package_roots", []),
        scanned_files=set(state.get("_cached_files", [])),
    )
    inspected = _inspect_file(state, file_path)
    if inspected is None:
        return (
            f"Error: '{file_path}' not found or not readable in the repository.",
            None,
        )

    state["explored_files"].append(file_path)

    fact_evidence = _record_inspected_fact(state, inspected)
    fact_evidence["explored_import_target"] = candidate_is_import_target
    _record_dependency_edge(state, inspected)
    summary_evidence = _refine_summary(state, inspected)
    unknowns_cleared = _reduce_unknowns(state, inspected)
    _update_confidence(
        state,
        summary_evidence=summary_evidence,
        unknowns_cleared=unknowns_cleared,
        fact_evidence=fact_evidence,
    )

    repo_path = Path(state["current_summary"]["local_path"])
    preview = _file_preview(repo_path / file_path)

    result = (
        f"File: {file_path}\n"
        f"Language: {inspected['language']} | Role: {inspected['role_hint']} | "
        f"Size: {inspected['line_count_bucket']} ({inspected['line_count']} lines)\n"
        f"Imports: {inspected['imported_modules'] or 'none'}\n"
        f"--- preview ---\n{preview}"
    )
    return result, "explored"


def _tool_follow_import(
    state: Dict, from_file: str, import_path: str
) -> Tuple[str, Optional[str]]:
    if not from_file or not import_path:
        return "Error: from_file and import_path are both required.", None

    repo_path = Path(state["current_summary"]["local_path"]).resolve()
    scanned_files = set(state.get("_cached_files", []))
    package_roots = [Path(r) for r in state.get("package_roots", [])]

    resolved = _resolve_internal_import(
        repo_path=repo_path,
        source_file=from_file,
        import_specifier=import_path,
        package_roots=package_roots,
        scanned_files=scanned_files,
    )
    if resolved is None:
        return (
            f"Could not resolve '{import_path}' from '{from_file}' to an internal file. "
            "It may be an external package.",
            None,
        )

    return _tool_read_file(state, resolved)


def _tool_search_for_pattern(
    state: Dict,
    pattern: str,
    file_extensions: Optional[List[str]],
) -> Tuple[str, None]:
    if not pattern:
        return "Error: pattern is required.", None

    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return f"Invalid regex: {exc}", None

    repo_path = Path(state["current_summary"]["local_path"]).resolve()
    files = state.get("_cached_files", [])
    matches: List[str] = []

    for file_path in files:
        if file_extensions and not any(file_path.endswith(e) for e in file_extensions):
            continue
        try:
            content = (repo_path / file_path).read_text(encoding="utf-8", errors="ignore")
            for lineno, line in enumerate(content.splitlines(), 1):
                if compiled.search(line):
                    matches.append(f"{file_path}:{lineno}: {line.strip()[:120]}")
                    if len(matches) >= MAX_SEARCH_RESULTS:
                        break
        except OSError:
            continue
        if len(matches) >= MAX_SEARCH_RESULTS:
            break

    if not matches:
        return f"No matches found for '{pattern}'.", None
    return f"Found {len(matches)} match(es):\n" + "\n".join(matches), None


def _tool_mark_insight(
    insights: List[Dict],
    insight_type: str,
    description: str,
    files: List[str],
) -> Tuple[str, None]:
    insights.append({
        "insight_type": insight_type,
        "description": description,
        "files": files or [],
    })
    return f"Insight recorded: [{insight_type}] {description}", None


def _tool_stop(state: Dict, reason: str) -> Tuple[str, Optional[str]]:
    file_count = state.get("current_summary", {}).get("file_count", 0)
    # For small repos explore most of them; scale down for large repos.
    # e.g. 16 files → 10, 30 files → 12, 80 files → 14, 200 files → 15
    min_files = min(15, max(6, int(file_count * 0.65)))
    explored = len(state.get("explored_files", []))

    if explored < min_files:
        return (
            f"STOP REJECTED: You have only explored {explored} file(s). "
            f"You must explore at least {min_files} files before stopping. "
            f"Continue with read_file or follow_import.",
            None,  # no "stop" side effect — loop continues
        )

    state["stop_reason"] = reason or "Agent decided analysis is complete."
    return f"Analysis stopped: {state['stop_reason']}", "stop"


# ---------------------------------------------------------------------------
# Ollama call with retry + content-fallback tool-call parsing
# ---------------------------------------------------------------------------

@dataclass
class _ToolFunction:
    name: str
    arguments: Dict[str, Any]


@dataclass
class _ToolCall:
    function: _ToolFunction


def _extract_tool_calls(response) -> List[_ToolCall]:
    """
    qwen2.5-coder returns tool calls as JSON in message.content instead of
    populating message.tool_calls. Try tool_calls first; fall back to parsing content.

    The model may wrap the JSON in a markdown code block with surrounding prose,
    so we search for the JSON object/array anywhere in the content.
    """
    if response.message.tool_calls:
        return response.message.tool_calls

    content = (response.message.content or "").strip()
    if not content:
        return []

    # Try to find JSON inside a markdown code fence first.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", content, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1)
    else:
        # Fall back: find the first { or [ and try to parse from there.
        match = re.search(r"(\{|\[)", content)
        if not match:
            return []
        raw = content[match.start():]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []

    # Handle both a single call {"name":..., "arguments":...}
    # and an array of calls.
    if isinstance(parsed, dict) and "name" in parsed:
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    calls = []
    for item in parsed:
        name = item.get("name") or item.get("function", {}).get("name")
        args = item.get("arguments") or item.get("function", {}).get("arguments") or {}
        if name:
            calls.append(_ToolCall(function=_ToolFunction(name=name, arguments=args)))
    return calls


def _call_model_with_retry(messages: List, retries: int = 2):
    for attempt in range(retries + 1):
        try:
            return ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                tools=_TOOLS,
                options={"temperature": 0.2},
            )
        except Exception as exc:
            LOGGER.warning(
                "Ollama call failed (attempt %d/%d): %s", attempt + 1, retries + 1, exc
            )
            if attempt < retries:
                time.sleep(1 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOISE_SUFFIXES = {".min.js", ".min.css", ".min.mjs", ".bundle.js", ".chunk.js"}
_NOISE_PATH_SEGMENTS = {"node_modules", "vendor", "vendors", "dist", "build", ".git"}


def _is_noise_file(file_path: str) -> bool:
    """Return True for minified, vendored, or build-artifact files that carry no architecture signal."""
    p = Path(file_path)
    # Check suffix combinations (e.g. foo.min.js has suffix .js but name ends with .min)
    name = p.name.lower()
    if any(name.endswith(suf) for suf in _NOISE_SUFFIXES):
        return True
    # Check if any path segment is a known vendor/build directory
    parts = {part.lower() for part in p.parts}
    return bool(parts & _NOISE_PATH_SEGMENTS)


def _file_preview(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        preview = lines[:MAX_FILE_PREVIEW_LINES]
        tail = (
            f"\n... ({len(lines) - MAX_FILE_PREVIEW_LINES} more lines)"
            if len(lines) > MAX_FILE_PREVIEW_LINES
            else ""
        )
        return "\n".join(preview) + tail
    except OSError:
        return "(could not read file)"


def _next_unexplored(state: Dict) -> Optional[str]:
    """
    Return the next unexplored file, or None if all files have been explored.

    Selection order:
      1. Prioritised candidates from a directory not yet visited (breadth-first
         pressure — counteracts the depth-first chain-following tendency).
      2. Any prioritised candidate.
      3. Any file in the repo.
    """
    explored = set(state.get("explored_files", []))
    explored_dirs = set(os.path.dirname(f) for f in explored)

    # First pass: candidates in unvisited directories.
    for c in state.get("candidate_files", []):
        fp = c["file_path"]
        if fp not in explored and not _is_noise_file(fp):
            if os.path.dirname(fp) not in explored_dirs:
                return fp

    # Second pass: any unvisited candidate (visited directory is acceptable here).
    for c in state.get("candidate_files", []):
        fp = c["file_path"]
        if fp not in explored and not _is_noise_file(fp):
            return fp

    # Final fallback: every file in the repo.
    for f in state.get("_cached_files", []):
        if f not in explored and not _is_noise_file(f):
            return f
    return None


def _nudge_message(state: Dict) -> Dict:
    """
    Injected as a user turn when the model goes silent or makes no file-exploring call.
    Lists unexplored files explicitly so the model has a clear next action.

    Prefers files from directories the agent has not yet visited. Same-chain
    import targets are only surfaced when no unvisited-directory candidates
    are available — this counteracts the agent's tendency to follow a single
    import chain depth-first.
    """
    explored = set(state.get("explored_files", []))
    candidates = [
        c["file_path"] for c in state.get("candidate_files", [])
        if c["file_path"] not in explored and not _is_noise_file(c["file_path"])
    ]
    # Also surface any files reachable via imports that haven't been read yet.
    # Pass repo context when available so resolved paths carry file extensions
    # — degrades to naive arithmetic if state lacks current_summary (e.g. tests).
    local_path = state.get("current_summary", {}).get("local_path")
    repo_path = Path(local_path).resolve() if local_path else None
    import_targets = [
        t for t in _resolved_import_targets(
            state,
            repo_path=repo_path,
            package_roots=state.get("package_roots", []),
            scanned_files=set(state.get("_cached_files", [])) if repo_path else None,
        )
        if t not in explored and not _is_noise_file(t)
    ]

    explored_dirs = set(
        os.path.dirname(f) for f in state.get("explored_files", [])
    )
    unvisited_candidates = [
        c for c in candidates
        if os.path.dirname(c) not in explored_dirs
    ]

    if unvisited_candidates:
        file_list = "\n".join(f"  - {f}" for f in unvisited_candidates[:8])
        return {
            "role": "user",
            "content": (
                f"You have stayed in the same directory too long. These files are "
                f"from directories you have not yet explored — pick one and call "
                f"read_file:\n{file_list}"
            ),
        }

    unexplored = candidates + [t for t in import_targets if t not in candidates]
    if unexplored:
        file_list = "\n".join(f"  - {f}" for f in unexplored[:8])
        return {
            "role": "user",
            "content": (
                f"You have not yet read these files. Pick the most architecturally "
                f"significant one and call read_file on it:\n{file_list}"
            ),
        }
    # All known candidates exhausted — tell the model it can stop.
    return {
        "role": "user",
        "content": (
            "You have explored all known candidate files. "
            "Call stop_analysis with a summary of what you found."
        ),
    }


def _trace_entry(
    step: int, explored_file: Optional[str], state: Dict
) -> Dict:
    return {
        "step": step,
        "explored_file": explored_file,
        "confidence": state["confidence"],
        "remaining_candidates": len(state.get("candidate_files", [])),
        "stop_reason": state.get("stop_reason"),
    }
