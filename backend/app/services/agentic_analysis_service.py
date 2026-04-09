"""
Agentic analysis loop: an Ollama model drives file exploration via tool calls
instead of hardcoded heuristic scoring. Drop-in replacement for run_analysis_loop —
returns the same dict shape so the route needs no changes to its response handling.
"""
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ollama

from app.services.analysis_snapshot_service import (
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
from app.services.repo_scanner import scan_repository

LOGGER = logging.getLogger(__name__)

OLLAMA_MODEL = "qwen2.5-coder:7b"
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


def run_agentic_analysis_loop(initial_state: Dict, max_steps: int = 15) -> Dict:
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

    # Kept separate — not part of AnalysisState model shape.
    architecture_insights: List[Dict] = []
    initial_explored_len = len(state.get("explored_files", []))

    messages: List = [_build_system_message(state)]
    step_trace: List[Dict] = []

    for step in range(1, steps_limit + 1):
        if state.get("stop_reason"):
            break

        previous_explored = list(state["explored_files"])

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
            LOGGER.info("Step %d: model returned no tool call, skipping.", step)
            step_trace.append(_trace_entry(step, None, state))
            continue

        explored_this_step: Optional[str] = None
        stop_this_step = False

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
                    previous_explored = list(state["explored_files"])
            elif side_effect == "stop":
                stop_this_step = True

        step_trace.append(_trace_entry(step, explored_this_step, state))

        if stop_this_step or state.get("stop_reason"):
            break

    state["dependency_graph_summary"] = _compute_dependency_graph_summary(state)
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

def _build_system_message(state: Dict) -> Dict:
    summary = state["current_summary"]
    explored = state["explored_files"]
    candidates = state.get("candidate_files", [])
    unknowns = state.get("unknowns", [])

    candidate_lines = "\n".join(
        f"  - {c['file_path']}  ({c.get('reason', '')})"
        for c in candidates[:15]
    )
    explored_str = ", ".join(explored) if explored else "none yet"
    unknowns_str = "; ".join(unknowns) if unknowns else "none"

    content = (
        f"You are analyzing the architecture of the repository '{summary['repo']}'.\n"
        f"Total source files: {summary['file_count']} | "
        f"Languages: {', '.join(summary['languages'])}\n"
        f"Already explored: {explored_str}\n"
        f"Open questions: {unknowns_str}\n\n"
        f"Suggested starting candidates:\n{candidate_lines}\n\n"
        "Strategy: read entry-point files first, then follow their imports to trace "
        "the dependency chain. Use search_for_pattern when you need to find where "
        "something is defined. Record insights with mark_architecture_insight. "
        "Call stop_analysis once you have a clear picture — do not explore every file."
    )
    return {"role": "system", "content": content}


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

    candidate_is_import_target = file_path in _resolved_import_targets(state)
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
    scan_result = scan_repository(repo_path)
    scanned_files = set(scan_result["files"])
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
    files = scan_repository(repo_path)["files"]
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


def _tool_stop(state: Dict, reason: str) -> Tuple[str, str]:
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
