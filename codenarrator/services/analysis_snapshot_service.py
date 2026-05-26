import ast
import posixpath
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from codenarrator.services.repo_scanner import EXTENSION_LANGUAGE_MAP
from codenarrator.services.repo_metadata import ENTRY_POINT_FILES, KNOWN_TOP_LEVEL_DIRS
from codenarrator.services.repo_metadata import extract_repo_metadata
from codenarrator.services.repo_scanner import scan_repository


def build_analysis_snapshot(repo_path: Path) -> Dict:
    """
    Build a deterministic understanding snapshot for a local repository.
    """
    scan_result = scan_repository(repo_path)
    metadata = extract_repo_metadata(repo_path, scan_result)
    package_roots = _detect_python_package_roots(repo_path, scan_result["files"])

    repo_summary = {
        "repo": scan_result["repo"],
        "local_path": str(repo_path),
        "repo_type": metadata["repo_type"],
        "file_count": scan_result["file_count"],
        "languages": scan_result["languages"],
        "language_breakdown": metadata["language_breakdown"],
        "top_level_dirs": metadata["top_level_dirs"],
        "entry_points": metadata["entry_points"],
        "inspected_languages": [],
        "inspected_role_hints": [],
    }

    next_candidates = _build_next_candidates(scan_result, metadata, limit=5)
    unknowns = _build_unknowns(scan_result, metadata)
    confidence = _compute_confidence(scan_result, metadata)
    stop_reason = _derive_stop_reason(scan_result, next_candidates, confidence)

    analysis_state = {
        "repo_id": scan_result["repo"],
        "explored_files": [],
        "candidate_files": next_candidates,
        "inspected_facts": [],
        "dependency_edges": [],
        "unknowns": unknowns,
        "package_roots": package_roots,
        "current_summary": repo_summary,
        "confidence": confidence,
        "no_progress_steps": 0,
        "stop_reason": stop_reason,
    }
    _refresh_candidates_for_signal(analysis_state, limit=5)
    next_candidates = analysis_state["candidate_files"]

    return {
        "repo_summary": repo_summary,
        "next_candidates": next_candidates,
        "unknowns": unknowns,
        "confidence": confidence,
        "analysis_state": analysis_state,
    }


def advance_analysis_state(current_state: Dict) -> Dict:
    """
    Advance analysis by one deterministic step.

    The function is local-first and stateless: it consumes current state and
    returns the next state without persistence.
    """
    next_state = {
        "repo_id": current_state["repo_id"],
        "explored_files": list(current_state.get("explored_files", [])),
        "candidate_files": [dict(c) for c in current_state.get("candidate_files", [])],
        "inspected_facts": [dict(f) for f in current_state.get("inspected_facts", [])],
        "dependency_edges": [dict(e) for e in current_state.get("dependency_edges", [])],
        "unknowns": list(current_state.get("unknowns", [])),
        "package_roots": list(current_state.get("package_roots", [])),
        "current_summary": dict(current_state["current_summary"]),
        "confidence": float(current_state.get("confidence", 0.0)),
        "no_progress_steps": int(current_state.get("no_progress_steps", 0)),
        "stop_reason": current_state.get("stop_reason"),
    }

    candidate_file = _select_next_candidate(next_state)
    if candidate_file is None:
        next_state["stop_reason"] = "No remaining candidate files to inspect."
        return next_state

    # Check before exploring so the pre-exploration dep_edges are used.
    candidate_is_import_target = candidate_file in _resolved_import_targets(next_state)

    inspected = _inspect_file(next_state, candidate_file)
    if inspected is None:
        next_state["stop_reason"] = f"Candidate file is unavailable: {candidate_file}"
        next_state["candidate_files"] = [
            c for c in next_state["candidate_files"] if c["file_path"] != candidate_file
        ]
        return next_state

    next_state["explored_files"].append(candidate_file)
    next_state["candidate_files"] = [
        c for c in next_state["candidate_files"] if c["file_path"] != candidate_file
    ]

    summary_before = _summary_progress_signature(next_state["current_summary"])
    unknowns_before = list(next_state["unknowns"])

    fact_evidence = _record_inspected_fact(next_state, inspected)
    fact_evidence["explored_import_target"] = candidate_is_import_target
    _record_dependency_edge(next_state, inspected)
    summary_evidence = _refine_summary(next_state, inspected)
    unknowns_cleared = _reduce_unknowns(next_state, inspected)
    confidence_evidence = _update_confidence(
        next_state,
        summary_evidence=summary_evidence,
        unknowns_cleared=unknowns_cleared,
        fact_evidence=fact_evidence,
    )
    _refresh_candidates_for_signal(next_state, limit=8)

    summary_changed = _summary_progress_signature(next_state["current_summary"]) != summary_before
    unknowns_changed = next_state["unknowns"] != unknowns_before
    meaningful_progress = summary_changed or unknowns_changed or confidence_evidence or fact_evidence.get("explored_import_target", False)

    if meaningful_progress:
        next_state["no_progress_steps"] = 0
    else:
        next_state["no_progress_steps"] += 1

    if next_state["no_progress_steps"] >= 2:
        next_state["stop_reason"] = "No meaningful progress in 2 consecutive steps."
    elif not next_state["candidate_files"]:
        next_state["stop_reason"] = "No more meaningful candidates available."
    else:
        next_state["stop_reason"] = None

    return next_state


def run_analysis_loop(initial_state: Dict, max_steps: int = 15) -> Dict:
    """
    Run deterministic multi-step analysis until stop condition or max_steps.
    """
    steps_limit = max(1, min(max_steps, 25))
    current_state = _copy_state(initial_state)
    initial_explored_len = len(current_state.get("explored_files", []))

    step_trace: List[Dict] = []
    for step in range(1, steps_limit + 1):
        if current_state.get("stop_reason"):
            break
        if not current_state.get("candidate_files"):
            current_state["stop_reason"] = "No more meaningful candidates available."
            break

        previous_explored = list(current_state.get("explored_files", []))
        next_state = advance_analysis_state(current_state)
        explored_file = _newly_explored_file(previous_explored, next_state["explored_files"])

        step_trace.append(
            {
                "step": step,
                "explored_file": explored_file,
                "confidence": next_state["confidence"],
                "remaining_candidates": len(next_state["candidate_files"]),
                "stop_reason": next_state.get("stop_reason"),
            }
        )

        current_state = next_state

        if current_state.get("stop_reason"):
            break

    explored_files_in_order = current_state["explored_files"][initial_explored_len:]
    current_state["dependency_graph_summary"] = _compute_dependency_graph_summary(current_state)

    return {
        "steps_executed": len(step_trace),
        "explored_files_in_order": explored_files_in_order,
        "step_trace": step_trace,
        "final_summary": current_state["current_summary"],
        "final_confidence": current_state["confidence"],
        "remaining_unknowns": current_state["unknowns"],
        "stop_reason": current_state.get("stop_reason"),
        "dependency_graph_summary": _compute_dependency_graph_summary(current_state),
        "final_state": current_state,
    }


def _build_next_candidates(scan_result: Dict, metadata: Dict, limit: int) -> List[Dict]:
    entry_points: List[str] = metadata["entry_points"]
    files: List[str] = scan_result["files"]
    file_languages: Dict[str, str] = scan_result["file_languages"]
    language_breakdown: Dict[str, int] = metadata["language_breakdown"]

    if not files:
        return []

    candidates: List[Dict] = []
    seen: Set[str] = set()

    dominant_language = sorted(
        language_breakdown.items(),
        key=lambda item: (-item[1], item[0]),
    )[0][0]

    dominant_files = sorted(
        file_path
        for file_path in files
        if file_languages.get(file_path) == dominant_language
    )

    # Rule 1: likely entry points first.
    for file_path in entry_points:
        _add_candidate(
            candidates,
            seen,
            file_path,
            "Likely entry point (filename heuristic match).",
            limit,
        )

    # Rule 2: central files next (common architectural hubs).
    for file_path, reason in _central_file_candidates(files):
        _add_candidate(candidates, seen, file_path, reason, limit)

    # Rule 3: representative files from the dominant language.
    for file_path in dominant_files:
        _add_candidate(
            candidates,
            seen,
            file_path,
            (
                "Representative file from dominant language "
                f"'{dominant_language}'."
            ),
            limit,
        )

    # Rule 4: files that reduce ambiguity.
    for file_path, reason in _ambiguity_reducing_candidates(
        files,
        file_languages,
        dominant_language,
    ):
        _add_candidate(candidates, seen, file_path, reason, limit)

    return candidates


def _build_unknowns(scan_result: Dict, metadata: Dict) -> List[str]:
    unknowns: List[str] = []

    if scan_result["file_count"] == 0:
        unknowns.append("No supported source files were detected.")

    if not metadata["entry_points"]:
        unknowns.append("No obvious entry points found by filename heuristics.")

    if metadata["repo_type"] == "mixed":
        unknowns.append("Mixed-language boundaries are not analyzed yet.")

    if not metadata["top_level_dirs"]:
        unknowns.append("Top-level structure signals are limited.")

    return unknowns


def _compute_confidence(scan_result: Dict, metadata: Dict) -> float:
    score = 0.20

    file_count = scan_result["file_count"]
    languages = scan_result["languages"]

    if file_count > 0:
        score += 0.12
    if file_count >= 20:
        score += 0.08
    elif file_count < 5:
        score -= 0.08

    if len(languages) == 1:
        score += 0.10
    elif len(languages) > 1:
        score += 0.04

    if metadata["entry_points"]:
        score += 0.10
    if metadata["top_level_dirs"]:
        score += 0.08
    if metadata["repo_type"] == "mixed":
        score -= 0.10

    score = max(0.0, min(score, 0.70))
    return round(score, 2)


def _derive_stop_reason(
    scan_result: Dict,
    next_candidates: List[Dict],
    confidence: float,
) -> str | None:
    if scan_result["file_count"] == 0:
        return "No supported source files found to explore."
    if not next_candidates and confidence >= 0.8:
        return "Snapshot already coherent with no further candidates."
    return None


def _add_candidate(
    candidates: List[Dict],
    seen: Set[str],
    file_path: str,
    reason: str,
    limit: int,
) -> None:
    if len(candidates) >= limit or file_path in seen:
        return
    candidates.append({"file_path": file_path, "reason": reason})
    seen.add(file_path)


def _central_file_candidates(files: List[str]) -> List[Tuple[str, str]]:
    central_name_priority = [
        "main.py",
        "app.py",
        "index.ts",
        "index.js",
        "server.ts",
        "server.js",
        "settings.py",
        "config.py",
        "urls.py",
        "models.py",
        "routes.py",
        "api.py",
        "__init__.py",
    ]
    top_level_priority = [
        "src/",
        "app/",
        "backend/",
        "frontend/",
    ]

    ranked: List[Tuple[Tuple[int, int, str], str, str]] = []
    for file_path in files:
        if "/tests/" in f"/{file_path}" or file_path.startswith("tests/"):
            continue

        name = Path(file_path).name
        depth = len(Path(file_path).parts)

        if name in central_name_priority:
            name_rank = central_name_priority.index(name)
            ranked.append(
                (
                    (name_rank, depth, file_path),
                    file_path,
                    f"Central file pattern ('{name}') often anchors module flow.",
                )
            )
            continue

        for top_level in top_level_priority:
            if file_path.startswith(top_level):
                top_level_rank = top_level_priority.index(top_level)
                ranked.append(
                    (
                        (100 + top_level_rank, depth, file_path),
                        file_path,
                        (
                            "Top-level module file in "
                            f"'{top_level.rstrip('/')}' helps map structure."
                        ),
                    )
                )
                break

    ranked.sort(key=lambda item: item[0])
    return [(file_path, reason) for _, file_path, reason in ranked]


def _ambiguity_reducing_candidates(
    files: List[str],
    file_languages: Dict[str, str],
    dominant_language: str,
) -> List[Tuple[str, str]]:
    by_language: Dict[str, List[str]] = {}
    for file_path in files:
        language = file_languages.get(file_path, "unknown")
        by_language.setdefault(language, []).append(file_path)

    for language_files in by_language.values():
        language_files.sort()

    candidates: List[Tuple[str, str]] = []

    # Mixed-language ambiguity: include one representative file per non-dominant language.
    for language in sorted(by_language.keys()):
        if language == dominant_language:
            continue
        file_path = by_language[language][0]
        candidates.append(
            (
                file_path,
                (
                    "Reduces mixed-language ambiguity with one representative "
                    f"'{language}' file."
                ),
            )
        )

    # If still ambiguous (no entry points in earlier rules), include test file hints.
    test_files = sorted(
        file_path for file_path in files if "/test" in file_path or "tests/" in file_path
    )
    if test_files:
        candidates.append(
            (
                test_files[0],
                "Test file can clarify expected behavior when entry points are unclear.",
            )
        )

    return candidates


def _select_next_candidate(state: Dict) -> str | None:
    explored = set(state["explored_files"])
    for candidate in state["candidate_files"]:
        file_path = candidate["file_path"]
        if file_path not in explored:
            return file_path
    return None


def _copy_state(state: Dict) -> Dict:
    return {
        "repo_id": state["repo_id"],
        "explored_files": list(state.get("explored_files", [])),
        "candidate_files": [dict(c) for c in state.get("candidate_files", [])],
        "inspected_facts": [dict(f) for f in state.get("inspected_facts", [])],
        "dependency_edges": [dict(e) for e in state.get("dependency_edges", [])],
        "unknowns": list(state.get("unknowns", [])),
        "package_roots": list(state.get("package_roots", [])),
        "current_summary": dict(state["current_summary"]),
        "confidence": float(state.get("confidence", 0.0)),
        "no_progress_steps": int(state.get("no_progress_steps", 0)),
        "stop_reason": state.get("stop_reason"),
    }


def _newly_explored_file(previous: List[str], current: List[str]) -> str | None:
    if len(current) <= len(previous):
        return None
    return current[-1]


def _inspect_file(state: Dict, file_path: str) -> Dict | None:
    repo_path = Path(state["current_summary"]["local_path"])
    target = (repo_path / file_path).resolve()

    if not target.exists() or not target.is_file():
        return None
    if not target.is_relative_to(repo_path.resolve()):
        return None

    suffix = target.suffix.lower()
    language = EXTENSION_LANGUAGE_MAP.get(suffix, "unknown")
    top_level_dir = Path(file_path).parts[0] if Path(file_path).parts else ""
    content = target.read_text(encoding="utf-8", errors="ignore")
    line_count = len(content.splitlines())
    role_hint = _infer_role_hint(file_path)
    line_count_bucket = _line_count_bucket(line_count)
    imported_modules = _extract_imports_for_file(
        content=content,
        language=language,
    )

    return {
        "file_path": file_path,
        "name": target.name,
        "language": language,
        "top_level_dir": top_level_dir,
        "line_count": line_count,
        "line_count_bucket": line_count_bucket,
        "directory": str(Path(file_path).parent),
        "role_hint": role_hint,
        "imported_modules": imported_modules,
    }


def _refine_summary(state: Dict, inspected: Dict) -> Dict[str, bool]:
    summary = state["current_summary"]
    file_path = inspected["file_path"]
    file_name = inspected["name"]
    top_level_dir = inspected["top_level_dir"]
    language = inspected["language"]
    role_hint = inspected["role_hint"]

    evidence = {
        "new_entry_point": False,
        "new_top_level_signal": False,
        "new_summary_fact": False,
    }

    if file_name in ENTRY_POINT_FILES and file_path not in summary["entry_points"]:
        summary["entry_points"] = sorted(summary["entry_points"] + [file_path])
        evidence["new_entry_point"] = True

    if top_level_dir and top_level_dir not in summary["top_level_dirs"]:
        summary["top_level_dirs"] = sorted(summary["top_level_dirs"] + [top_level_dir])
        evidence["new_top_level_signal"] = True

    if language not in summary["inspected_languages"]:
        summary["inspected_languages"] = sorted(summary["inspected_languages"] + [language])
        evidence["new_summary_fact"] = True

    if role_hint not in summary["inspected_role_hints"]:
        summary["inspected_role_hints"] = sorted(summary["inspected_role_hints"] + [role_hint])
        evidence["new_summary_fact"] = True

    return evidence


def _reduce_unknowns(state: Dict, inspected: Dict) -> int:
    file_name = inspected["name"]
    unknowns_before = list(state["unknowns"])

    if file_name in ENTRY_POINT_FILES:
        state["unknowns"] = [
            u
            for u in state["unknowns"]
            if u != "No obvious entry points found by filename heuristics."
        ]

    if state["current_summary"]["top_level_dirs"]:
        state["unknowns"] = [
            u for u in state["unknowns"] if u != "Top-level structure signals are limited."
        ]

    inspected_languages = {
        fact["language"]
        for fact in state.get("inspected_facts", [])
        if fact.get("language") and fact["language"] != "unknown"
    }
    if len(inspected_languages) >= 2:
        state["unknowns"] = [
            u for u in state["unknowns"] if u != "Mixed-language boundaries are not analyzed yet."
        ]

    return len(unknowns_before) - len(state["unknowns"])


def _update_confidence(
    state: Dict,
    *,
    summary_evidence: Dict[str, bool],
    unknowns_cleared: int,
    fact_evidence: Dict[str, bool],
) -> bool:
    confidence = float(state["confidence"])
    delta = 0.0

    if summary_evidence["new_entry_point"]:
        delta += 0.10
    if summary_evidence["new_top_level_signal"]:
        delta += 0.05
    if unknowns_cleared > 0:
        delta += min(0.09, 0.03 * unknowns_cleared)
    if summary_evidence["new_summary_fact"] or fact_evidence["materially_new_fact"]:
        delta += 0.04

    state["confidence"] = round(max(0.0, min(confidence + delta, 0.95)), 2)
    return delta > 0


def _record_inspected_fact(state: Dict, inspected: Dict) -> Dict[str, bool]:
    facts = state["inspected_facts"]

    existing_languages = {f["language"] for f in facts}
    existing_roles = {f["role_hint"] for f in facts}
    existing_dirs = {f["directory"] for f in facts}
    existing_buckets = {f["line_count_bucket"] for f in facts}
    existing_imports = {
        module for fact in facts for module in fact.get("imported_modules", [])
    }

    new_fact = {
        "file_path": inspected["file_path"],
        "language": inspected["language"],
        "line_count_bucket": inspected["line_count_bucket"],
        "directory": inspected["directory"],
        "role_hint": inspected["role_hint"],
        "imports_found": len(inspected.get("imported_modules", [])),
        "imported_modules": inspected.get("imported_modules", []),
    }
    facts.append(new_fact)

    imported_modules = inspected.get("imported_modules", [])
    materially_new_fact = (
        inspected["language"] not in existing_languages
        or inspected["role_hint"] not in existing_roles
        or inspected["directory"] not in existing_dirs
        or inspected["line_count_bucket"] not in existing_buckets
        or any(module not in existing_imports for module in imported_modules)
    )
    return {"materially_new_fact": materially_new_fact}


def _record_dependency_edge(state: Dict, inspected: Dict) -> None:
    source = inspected["file_path"]
    imports = inspected.get("imported_modules", [])
    dedup_imports = sorted(set(imports))

    edges = state["dependency_edges"]
    for edge in edges:
        if edge["source"] == source:
            edge["imports"] = dedup_imports
            return

    edges.append(
        {
            "source": source,
            "imports": dedup_imports,
        }
    )


def _refresh_candidates_for_signal(state: Dict, limit: int) -> None:
    repo_path = Path(state["current_summary"]["local_path"]).resolve()
    scan_result = scan_repository(repo_path)
    files: List[str] = scan_result["files"]
    file_languages: Dict[str, str] = scan_result["file_languages"]
    explored = set(state["explored_files"])

    known_targets = _resolved_import_targets(
        state,
        repo_path=repo_path,
        package_roots=state.get("package_roots", []),
        scanned_files=set(files),
    )
    scored: List[Tuple[Tuple[int, str], Dict]] = []
    for file_path in files:
        if file_path in explored:
            continue

        score, reasons = _candidate_signal_score(state, file_path, file_languages, known_targets)
        if score <= 0:
            continue
        scored.append(
            (
                (-score, file_path),
                {
                    "file_path": file_path,
                    "reason": "; ".join(reasons),
                },
            )
        )

    scored.sort(key=lambda item: item[0])

    # Per-language coverage gate: when the dominant language is <50% explored,
    # drop non-dominant-language candidates from the queue. Prevents the agent
    # from chasing the strong "new language signal" into a second language
    # while the first one is still under-covered.
    explored_by_lang = Counter(
        file_languages.get(f, "unknown") for f in state.get("explored_files", [])
    )
    total_by_lang = Counter(file_languages.values())
    dominant_lang = max(total_by_lang, key=total_by_lang.get) if total_by_lang else None
    dominant_coverage = (
        explored_by_lang.get(dominant_lang, 0) / total_by_lang.get(dominant_lang, 1)
        if dominant_lang else 1.0
    )
    if dominant_lang and dominant_coverage < 0.5:
        scored = [
            item for item in scored
            if file_languages.get(item[1]["file_path"], dominant_lang) == dominant_lang
        ]

    candidates = [candidate for _, candidate in scored[:limit]]
    if not candidates:
        # Score threshold unmet — pick a small deterministic sample, not all unexplored files.
        fallback = sorted(fp for fp in files if fp not in explored)[:limit]
        candidates = [
            {
                "file_path": file_path,
                "reason": "Fallback candidate (no signal score; deterministic sample).",
            }
            for file_path in fallback
        ]

    state["candidate_files"] = candidates


def _resolved_import_targets(
    state: Dict,
    repo_path: Path | None = None,
    package_roots: List[str] | None = None,
    scanned_files: Set[str] | None = None,
) -> Counter:
    """
    Return a Counter mapping internal file paths → number of explored files
    that import that path.

    When ``repo_path`` and ``scanned_files`` are provided, uses the full
    ``_resolve_internal_import`` — which resolves relative imports to actual
    files with extensions and handles Python absolute imports via package
    roots. This matches what ``_compute_dependency_graph_summary`` produces.

    Without repo context, falls back to naive POSIX-path arithmetic on
    relative-only imports. The fallback strips file extensions (e.g.
    ``./formFieldsSource`` → ``datasources/formFieldsSource``), so callers
    that compare against real ``scanned_files`` paths should always pass
    repo context — otherwise the comparison never matches.

    The Counter shape lets callers reward files imported by multiple
    explored files higher than files imported by one.
    """
    targets: Counter = Counter()
    package_root_paths = [Path(r) for r in (package_roots or [])]
    for edge in state.get("dependency_edges", []):
        source = edge["source"]
        for imp in edge.get("imports", []):
            if repo_path is not None and scanned_files is not None:
                resolved = _resolve_internal_import(
                    repo_path=repo_path,
                    source_file=source,
                    import_specifier=imp,
                    package_roots=package_root_paths,
                    scanned_files=scanned_files,
                )
                if resolved:
                    targets[resolved] += 1
            elif imp.startswith(("./", "../")):
                source_dir = posixpath.dirname(source)
                naive = posixpath.normpath(posixpath.join(source_dir, imp))
                targets[naive] += 1
    return targets


def _candidate_signal_score(
    state: Dict,
    file_path: str,
    file_languages: Dict[str, str],
    known_targets: Counter | None = None,
) -> Tuple[int, List[str]]:
    name = Path(file_path).name
    top_level_dir = Path(file_path).parts[0] if Path(file_path).parts else ""
    language = file_languages.get(file_path, "unknown")
    role_hint = _infer_role_hint(file_path)

    inspected_facts = state.get("inspected_facts", [])
    inspected_languages = {f["language"] for f in inspected_facts}
    inspected_roles = {f["role_hint"] for f in inspected_facts}
    inspected_dirs = {f["directory"] for f in inspected_facts}
    unresolved_unknowns = set(state.get("unknowns", []))
    summary = state["current_summary"]

    score = 0
    reasons: List[str] = []

    if name in ENTRY_POINT_FILES and file_path not in summary["entry_points"]:
        score += 8
        reasons.append("entry-point heuristic")

    if (
        "No obvious entry points found by filename heuristics." in unresolved_unknowns
        and role_hint == "entry_point"
    ):
        score += 5
        reasons.append("can reduce entry-point ambiguity")

    if language not in inspected_languages:
        score += 4
        reasons.append(f"new language signal ({language})")

    directory_signal = str(Path(file_path).parent)
    if directory_signal not in inspected_dirs:
        score += 3
        reasons.append("new directory context")

    # Unvisited-directory bonus on top of "new directory context".
    # Combined: +7 for a candidate in a directory the agent hasn't entered yet,
    # which outweighs the +3 known-import-target chain-following bias below.
    explored_dirs = set(
        str(Path(f).parent) for f in state.get("explored_files", [])
    )
    file_dir = str(Path(file_path).parent)
    if file_dir not in explored_dirs:
        score += 4
        reasons.append("unvisited directory")

    if role_hint not in inspected_roles:
        score += 2
        reasons.append(f"new file role ({role_hint})")

    if role_hint == "central":
        score += 4
        reasons.append("central architecture hint")
    elif role_hint == "module":
        score += 1

    if top_level_dir in KNOWN_TOP_LEVEL_DIRS and top_level_dir not in summary["top_level_dirs"]:
        score += 2
        reasons.append(f"new top-level domain signal ({top_level_dir})")

    if top_level_dir in {"docs", "tests"}:
        score -= 2

    # Boost files that are known import targets from already-explored files.
    # These are high-value: exploring them reveals their own imports and
    # completes the dependency graph rather than hitting dead-end files.
    # Bonus scales with in-degree (how many explored files import this one),
    # capped at +9 so even a heavily-imported leaf doesn't dominate breadth.
    if known_targets is None:
        known_targets = _resolved_import_targets(state)
    if file_path in known_targets:
        import_count = known_targets[file_path]
        bonus = min(import_count * 3, 9)
        score += bonus
        reasons.append(f"known import target from {import_count} explored file(s)")

    if role_hint == "test":
        score -= 2
        if "No obvious entry points found by filename heuristics." in unresolved_unknowns:
            score += 1
            reasons.append("test can clarify behavior when entry point is unclear")

    # Most __init__.py files are empty package markers — push them down the
    # queue so the agent doesn't spend early budget on them. They remain
    # explorable (some do contain real code), just lower priority.
    if Path(file_path).name == "__init__.py":
        score -= 3
        reasons.append("__init__.py (package marker, low priority)")

    return score, reasons


def _summary_progress_signature(summary: Dict) -> Tuple[Tuple[str, ...], ...]:
    return (
        tuple(summary.get("entry_points", [])),
        tuple(summary.get("top_level_dirs", [])),
        tuple(summary.get("inspected_languages", [])),
        tuple(summary.get("inspected_role_hints", [])),
    )


def _extract_imports_for_file(*, content: str, language: str) -> List[str]:
    if language == "python":
        return _extract_python_imports(content)
    if language in {"javascript", "typescript"}:
        return _extract_javascript_imports(content)
    if language == "java":
        return _extract_java_imports(content)
    if language == "go":
        return _extract_go_imports(content)
    if language == "rust":
        return _extract_rust_imports(content)
    if language in {"cpp", "c"}:
        return _extract_cpp_imports(content)
    return []


def _extract_python_imports(content: str) -> List[str]:
    imports: List[str] = []
    seen: Set[str] = set()

    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.strip()
                    if name and name not in seen:
                        imports.append(name)
                        seen.add(name)
            elif isinstance(node, ast.ImportFrom):
                level = node.level or 0
                module = (node.module or "").strip()
                if level > 0:
                    imported = "." * level + module if module else "." * level
                else:
                    imported = module
                if imported and imported not in seen:
                    imports.append(imported)
                    seen.add(imported)
        return imports
    except SyntaxError:
        return _extract_python_imports_regex_fallback(content)


def _extract_python_imports_regex_fallback(content: str) -> List[str]:
    imports: List[str] = []
    seen: Set[str] = set()

    for match in re.finditer(r"^\s*import\s+([A-Za-z0-9_.,\s]+)", content, re.MULTILINE):
        modules = [m.strip() for m in match.group(1).split(",")]
        for module in modules:
            if module and module not in seen:
                imports.append(module)
                seen.add(module)

    for match in re.finditer(r"^\s*from\s+([.\w]+)\s+import\s+", content, re.MULTILINE):
        module = match.group(1).strip()
        if module and module not in seen:
            imports.append(module)
            seen.add(module)

    return imports


def _extract_javascript_imports(content: str) -> List[str]:
    imports: List[str] = []
    seen: Set[str] = set()

    patterns = [
        r'import\s+[^;\n]*?\sfrom\s+["\']([^"\']+)["\']',
        r'import\s+["\']([^"\']+)["\']',
        r'require\(\s*["\']([^"\']+)["\']\s*\)',
        # Re-exports: export { X } from './foo', export * from './foo', export { X as Y } from './foo'
        r'export\s+(?:\*|{[^}]*})\s+from\s+["\']([^"\']+)["\']',
        # Dynamic imports: import('./foo'), import('./foo').then(...)
        r'import\s*\(\s*["\']([^"\']+)["\']\s*\)',
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, content):
            module = match.group(1).strip()
            if module and module not in seen:
                imports.append(module)
                seen.add(module)

    return imports


def _extract_rust_imports(content: str) -> List[str]:
    imports: List[str] = []
    seen: Set[str] = set()

    # use crate::module, use super::module, use self::module, use external_crate::...
    for match in re.finditer(r"^\s*use\s+([\w:]+)", content, re.MULTILINE):
        module = match.group(1).strip().rstrip(":")
        if module and module not in seen:
            imports.append(module)
            seen.add(module)

    # mod submodule; (declares a local module file)
    for match in re.finditer(r"^\s*mod\s+(\w+)\s*;", content, re.MULTILINE):
        module = match.group(1).strip()
        if module and module not in seen:
            imports.append(module)
            seen.add(module)

    # extern crate name;
    for match in re.finditer(r"^\s*extern\s+crate\s+(\w+)\s*;", content, re.MULTILINE):
        module = match.group(1).strip()
        if module and module not in seen:
            imports.append(module)
            seen.add(module)

    return imports


def _extract_cpp_imports(content: str) -> List[str]:
    imports: List[str] = []
    seen: Set[str] = set()

    for match in re.finditer(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', content, re.MULTILINE):
        module = match.group(1).strip()
        if module and module not in seen:
            imports.append(module)
            seen.add(module)

    return imports


def _extract_java_imports(content: str) -> List[str]:
    imports: List[str] = []
    seen: Set[str] = set()
    for match in re.finditer(r"^\s*import\s+(?:static\s+)?([\w.]+)\s*;", content, re.MULTILINE):
        module = match.group(1).strip()
        if module and module not in seen:
            imports.append(module)
            seen.add(module)
    return imports


def _extract_go_imports(content: str) -> List[str]:
    imports: List[str] = []
    seen: Set[str] = set()

    # Single import: import "path/to/pkg"
    for match in re.finditer(r'^\s*import\s+"([^"]+)"', content, re.MULTILINE):
        module = match.group(1).strip()
        if module and module not in seen:
            imports.append(module)
            seen.add(module)

    # Grouped import block: import ( "pkg1" alias "pkg2" )
    block_match = re.search(r"import\s*\(([^)]+)\)", content, re.DOTALL)
    if block_match:
        for line in block_match.group(1).splitlines():
            m = re.search(r'"([^"]+)"', line)
            if m:
                module = m.group(1).strip()
                if module and module not in seen:
                    imports.append(module)
                    seen.add(module)

    return imports


def _compute_dependency_graph_summary(state: Dict) -> Dict:
    edges = state.get("dependency_edges", [])
    repo_path = Path(state["current_summary"]["local_path"]).resolve()
    scanned_files = set(scan_repository(repo_path)["files"])
    package_roots = [Path(root) for root in state.get("package_roots", [])]

    imported_counter: Counter[str] = Counter()
    file_import_counts: List[Dict] = []
    cluster_map: Dict[str, Set[str]] = defaultdict(set)
    internal_edge_set: Set[Tuple[str, str]] = set()

    for edge in edges:
        source = edge["source"]
        imports = sorted(set(edge.get("imports", [])))

        for module in imports:
            imported_counter[module] += 1
            cluster_key = _cluster_key(module)
            if cluster_key:
                cluster_map[cluster_key].add(source)

            resolved_internal = _resolve_internal_import(
                repo_path=repo_path,
                source_file=source,
                import_specifier=module,
                package_roots=package_roots,
                scanned_files=scanned_files,
            )
            if resolved_internal is not None:
                internal_edge_set.add((source, resolved_internal))

        file_import_counts.append(
            {
                "source": source,
                "imports_count": len(imports),
            }
        )

    most_imported_modules = [
        {"module": module, "count": count}
        for module, count in sorted(imported_counter.items(), key=lambda item: (-item[1], item[0]))[:10]
    ]

    highest_dependency_files = sorted(
        file_import_counts,
        key=lambda item: (-item["imports_count"], item["source"]),
    )[:10]

    clusters = []
    for cluster_key, files in sorted(cluster_map.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(files) < 2:
            continue
        clusters.append(
            {
                "cluster": cluster_key,
                "files": sorted(files),
            }
        )

    internal_edges = [
        {"from": source, "to": target}
        for source, target in sorted(internal_edge_set)
    ][:500]  # cap to avoid large payloads on complex repos

    return {
        "most_imported_modules": most_imported_modules,
        "highest_dependency_files": highest_dependency_files,
        "clusters": clusters,
        "internal_edges": internal_edges,
    }


def _cluster_key(module: str) -> str:
    if not module:
        return ""
    if module.startswith("."):
        return "relative"
    if "/" in module:
        return module.split("/", 1)[0]
    if "." in module:
        return module.split(".", 1)[0]
    return module


_JS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts"}
_C_EXTENSIONS = {".c", ".cpp", ".h", ".hpp"}


def _resolve_internal_import(
    *,
    repo_path: Path,
    source_file: str,
    import_specifier: str,
    package_roots: List[Path],
    scanned_files: Set[str],
) -> str | None:
    source_abs = (repo_path / source_file).resolve()
    source_dir = source_abs.parent
    source_suffix = Path(source_file).suffix.lower()

    # Dispatch by the source file's language. This avoids cross-language false
    # positives (e.g. a Go "./pkg" import going through the JS resolver, or a
    # Java "com.example" import matching a Python "com" package).
    if source_suffix in _JS_EXTENSIONS:
        if _is_js_relative_import(import_specifier):
            resolved = _resolve_js_relative_import(repo_path, source_dir, import_specifier)
            return str(resolved.relative_to(repo_path)) if resolved else None
        return None  # bare JS imports like 'react' are external

    if source_suffix == ".py":
        if _is_python_relative_import(import_specifier):
            resolved = _resolve_python_relative_import(repo_path, source_dir, import_specifier)
            return str(resolved.relative_to(repo_path)) if resolved else None
        if _should_attempt_absolute_python_resolution(import_specifier, package_roots, scanned_files):
            resolved = _resolve_absolute_import(import_specifier, package_roots, scanned_files)
            if resolved is not None:
                return resolved
        return None

    if source_suffix == ".java":
        return _resolve_java_import(import_specifier, scanned_files)
    if source_suffix == ".go":
        return _resolve_go_import(repo_path, source_dir, import_specifier, scanned_files)
    if source_suffix == ".rs":
        return _resolve_rust_import(repo_path, source_dir, import_specifier, scanned_files)
    if source_suffix in _C_EXTENSIONS:
        return _resolve_c_import(repo_path, source_dir, import_specifier, scanned_files)

    # Unknown source extension — fall back to shape-based heuristics for safety.
    if _is_js_relative_import(import_specifier):
        resolved = _resolve_js_relative_import(repo_path, source_dir, import_specifier)
        return str(resolved.relative_to(repo_path)) if resolved else None
    if _is_python_relative_import(import_specifier):
        resolved = _resolve_python_relative_import(repo_path, source_dir, import_specifier)
        return str(resolved.relative_to(repo_path)) if resolved else None

    return None


def _is_js_relative_import(import_specifier: str) -> bool:
    return import_specifier.startswith("./") or import_specifier.startswith("../")


def _is_python_relative_import(import_specifier: str) -> bool:
    return import_specifier.startswith(".")


def _resolve_js_relative_import(repo_path: Path, source_dir: Path, import_specifier: str) -> Path | None:
    candidate_base = (source_dir / import_specifier).resolve()
    return _resolve_candidate_path(repo_path, candidate_base, [".js", ".jsx", ".ts", ".tsx"])


def _resolve_python_relative_import(repo_path: Path, source_dir: Path, import_specifier: str) -> Path | None:
    level = 0
    for ch in import_specifier:
        if ch == ".":
            level += 1
        else:
            break

    module = import_specifier[level:]
    target_dir = source_dir
    for _ in range(max(0, level - 1)):
        target_dir = target_dir.parent

    candidate_base = target_dir / module.replace(".", "/") if module else target_dir
    return _resolve_candidate_path(repo_path, candidate_base, [".py"])


# Common roots Java sources live under in Maven/Gradle layouts.
_JAVA_SOURCE_ROOTS = ("", "src/main/java/", "src/", "java/")


def _resolve_java_import(import_specifier: str, scanned_files: Set[str]) -> str | None:
    """
    Map "com.example.Foo" → "com/example/Foo.java" and try common Java source
    roots. For static imports like "com.example.Foo.method", drop the trailing
    member segment and retry as a class import.
    """
    def try_resolve(class_path: str) -> str | None:
        rel = class_path.replace(".", "/") + ".java"
        for root in _JAVA_SOURCE_ROOTS:
            candidate = f"{root}{rel}"
            if candidate in scanned_files:
                return candidate
        return None

    resolved = try_resolve(import_specifier)
    if resolved is not None:
        return resolved
    if "." in import_specifier:
        return try_resolve(import_specifier.rsplit(".", 1)[0])
    return None


def _resolve_go_import(
    repo_path: Path,
    source_dir: Path,
    import_specifier: str,
    scanned_files: Set[str],
) -> str | None:
    """
    Resolve a Go import to a representative .go file in the target package directory.

    - "./local" / "../local" → directory relative to source_dir
    - "github.com/foo/bar/pkg" → strip module prefix heuristically by trying
      progressively-longer trailing suffixes of the import path against scanned
      package directories. Handles both repo-rooted ("pkg/foo") and module-pathed
      ("github.com/user/proj/pkg/foo") layouts.
    """
    if import_specifier.startswith("./") or import_specifier.startswith("../"):
        try:
            target_dir = (source_dir / import_specifier).resolve().relative_to(repo_path).as_posix()
        except ValueError:
            return None
        return _first_file_in_dir(target_dir, scanned_files, ".go")

    parts = import_specifier.split("/")
    for i in range(len(parts)):
        candidate_dir = "/".join(parts[i:])
        resolved = _first_file_in_dir(candidate_dir, scanned_files, ".go")
        if resolved is not None:
            return resolved
    return None


def _first_file_in_dir(directory: str, scanned_files: Set[str], extension: str) -> str | None:
    """Return the alphabetically-first file directly inside `directory` with the given extension."""
    prefix = (directory.rstrip("/") + "/") if directory else ""
    candidates = sorted(
        f for f in scanned_files
        if f.endswith(extension)
        and f.startswith(prefix)
        and "/" not in f[len(prefix):]
    )
    return candidates[0] if candidates else None


def _resolve_rust_import(
    repo_path: Path,
    source_dir: Path,
    import_specifier: str,
    scanned_files: Set[str],
) -> str | None:
    """
    Resolve a Rust import:
    - "crate::foo::bar" → {crate_dir}/foo/bar.rs or {crate_dir}/foo/bar/mod.rs;
       falls back to {crate_dir}/foo.rs if `bar` is an item inside `foo`.
    - "foo" (bare ident from `mod foo;` or `extern crate foo;`) →
       {source_dir}/foo.rs or {source_dir}/foo/mod.rs. External crates won't
       match locally and return None.
    """
    if import_specifier.startswith("crate::"):
        crate_root = _detect_rust_crate_root(scanned_files)
        if crate_root is None:
            return None
        crate_dir = posixpath.dirname(crate_root)
        path_parts = import_specifier[len("crate::"):].split("::")
        if not path_parts or not all(path_parts):
            return None

        prefix = (crate_dir + "/") if crate_dir else ""
        rel_module = "/".join(path_parts)
        for candidate in (f"{prefix}{rel_module}.rs", f"{prefix}{rel_module}/mod.rs"):
            if candidate in scanned_files:
                return candidate

        # The final segment may be an item (function/struct) inside the parent module.
        if len(path_parts) > 1:
            parent = "/".join(path_parts[:-1])
            for candidate in (f"{prefix}{parent}.rs", f"{prefix}{parent}/mod.rs"):
                if candidate in scanned_files:
                    return candidate
        return None

    if "::" not in import_specifier and import_specifier.isidentifier():
        try:
            source_rel = source_dir.relative_to(repo_path).as_posix()
        except ValueError:
            return None
        prefix = (source_rel + "/") if source_rel else ""
        for candidate in (f"{prefix}{import_specifier}.rs", f"{prefix}{import_specifier}/mod.rs"):
            if candidate in scanned_files:
                return candidate
    return None


def _detect_rust_crate_root(scanned_files: Set[str]) -> str | None:
    """Return src/lib.rs or src/main.rs if either is in scanned_files."""
    for candidate in ("src/lib.rs", "src/main.rs"):
        if candidate in scanned_files:
            return candidate
    return None


# TODO: the C extractor strips delimiters so we cannot distinguish
# #include <stdio.h> (system) from #include "stdio.h" (local). In
# practice system headers won't be in scanned_files so they return
# None correctly. Edge case: a local file shadowing a stdlib name
# (e.g. string.h) would incorrectly resolve. Fix requires preserving
# delimiters in _extract_cpp_imports.
def _resolve_c_import(
    repo_path: Path,
    source_dir: Path,
    import_specifier: str,
    scanned_files: Set[str],
) -> str | None:
    """
    Resolve a C/C++ #include header.

    The extractor strips the <>/"" delimiter, so we attempt resolution against
    both the source file's directory and the repo root. System headers like
    <stdio.h> won't appear in scanned_files and return None — which gives the
    same effective behavior as recognising them as external.
    """
    try:
        source_rel = source_dir.relative_to(repo_path).as_posix()
    except ValueError:
        return None

    prefix = (source_rel + "/") if source_rel else ""
    candidate = f"{prefix}{import_specifier}"
    if candidate in scanned_files:
        return candidate

    if import_specifier in scanned_files:
        return import_specifier

    return None


def _detect_python_package_roots(repo_path: Path, scanned_files: List[str]) -> List[str]:
    package_roots: List[str] = []
    scanned_set = set(scanned_files)
    repo_has_python = any(path.endswith(".py") for path in scanned_files)
    if not repo_has_python:
        return package_roots

    has_src_layout = any(path.startswith("src/") for path in scanned_files) and any(
        path.startswith("src/") and path.endswith("/__init__.py")
        for path in scanned_files
    )
    if has_src_layout:
        package_roots.append("src")

    # Detect which top-level directories *contain* packages (i.e. directories
    # whose immediate children have __init__.py). We pick the package root
    # based on whether those packages live directly at the repo root, under a
    # single wrapper directory (e.g. backend/), or are spread across multiple
    # wrappers.
    top_level_dirs = {
        Path(path).parts[0]
        for path in scanned_files
        if len(Path(path).parts) >= 2 and path.endswith("/__init__.py")
    }

    if len(top_level_dirs) == 1:
        # All __init__.py-containing dirs live under one top-level directory.
        wrapper = next(iter(top_level_dirs))
        # If the wrapper itself has an __init__.py it IS a package, accessible
        # from the repo root as `wrapper.subpkg` — so the resolver root is ".".
        # Otherwise the wrapper is a non-package container (e.g. "backend/")
        # whose children are the actual top-level packages — the resolver root
        # is the wrapper itself.
        if f"{wrapper}/__init__.py" in scanned_files:
            package_roots.append(".")
        else:
            package_roots.append(wrapper)
    elif len(top_level_dirs) > 1:
        # Packages span multiple top-level directories — root is the repo root.
        package_roots.append(".")

    unique_roots: List[str] = []
    seen: Set[str] = set()
    for root in package_roots:
        if root not in seen:
            unique_roots.append(root)
            seen.add(root)

    return unique_roots


def _should_attempt_absolute_python_resolution(
    import_specifier: str,
    package_roots: List[Path],
    scanned_files: Set[str],
) -> bool:
    if not import_specifier or not package_roots:
        return False

    first_segment = import_specifier.split(".", 1)[0]
    if not first_segment:
        return False

    known_top_level_names = _known_top_level_package_names(package_roots, scanned_files)
    return first_segment in known_top_level_names


def _resolve_absolute_import(
    module_string: str,
    package_roots: List[Path],
    scanned_files: Set[str],
) -> str | None:
    module_path = module_string.replace(".", "/")

    for package_root in package_roots:
        root_prefix = package_root.as_posix().strip(".")
        if root_prefix:
            file_candidate = f"{root_prefix}/{module_path}.py"
            init_candidate = f"{root_prefix}/{module_path}/__init__.py"
        else:
            file_candidate = f"{module_path}.py"
            init_candidate = f"{module_path}/__init__.py"

        if file_candidate in scanned_files:
            return file_candidate
        if init_candidate in scanned_files:
            return init_candidate

    return None


def _known_top_level_package_names(package_roots: List[Path], scanned_files: Set[str]) -> Set[str]:
    """
    Return the set of importable top-level names directly under each package
    root. Strips the ``.py`` extension so single-file modules (e.g.
    ``backend/config.py``) are recorded as ``"config"`` — matching the form
    of import specifiers like ``import config``. Skips ``__init__`` since
    ``import __init__`` is never valid.
    """
    names: Set[str] = set()
    for package_root in package_roots:
        root_prefix = package_root.as_posix().strip(".")
        for file_path in scanned_files:
            parts = Path(file_path).parts
            if not parts:
                continue
            if root_prefix:
                root_parts = tuple(Path(root_prefix).parts)
                if parts[: len(root_parts)] != root_parts:
                    continue
                if len(parts) > len(root_parts):
                    segment = parts[len(root_parts)]
                else:
                    continue
            else:
                segment = parts[0]
            if segment.endswith(".py"):
                segment = segment[:-3]
            if segment == "__init__":
                continue
            names.add(segment)
    return names


def _resolve_candidate_path(repo_path: Path, candidate_base: Path, extensions: List[str]) -> Path | None:
    candidates: List[Path] = []

    candidates.append(candidate_base)
    for ext in extensions:
        candidates.append(candidate_base.with_suffix(ext))
    for ext in extensions:
        candidates.append(candidate_base / f"index{ext}")
    candidates.append(candidate_base / "__init__.py")

    seen: Set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)

        if not candidate.exists() or not candidate.is_file():
            continue
        if not candidate.resolve().is_relative_to(repo_path):
            continue
        return candidate.resolve()

    return None


def _line_count_bucket(line_count: int) -> str:
    if line_count < 80:
        return "small"
    if line_count < 300:
        return "medium"
    return "large"


def _infer_role_hint(file_path: str) -> str:
    file_name = Path(file_path).name
    if file_name in ENTRY_POINT_FILES:
        return "entry_point"
    if "test" in file_name.lower() or "/tests/" in f"/{file_path}" or file_path.startswith("tests/"):
        return "test"
    if file_name in {"config.py", "settings.py", "routes.py", "api.py", "models.py"}:
        return "central"
    return "module"
