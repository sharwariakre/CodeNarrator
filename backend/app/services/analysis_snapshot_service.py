from pathlib import Path
from typing import Dict, List, Set, Tuple

from app.services.repo_scanner import EXTENSION_LANGUAGE_MAP
from app.services.repo_metadata import ENTRY_POINT_FILES
from app.services.repo_metadata import extract_repo_metadata
from app.services.repo_scanner import scan_repository


def build_analysis_snapshot(repo_path: Path) -> Dict:
    """
    Build a deterministic understanding snapshot for a local repository.
    """
    scan_result = scan_repository(repo_path)
    metadata = extract_repo_metadata(repo_path, scan_result)

    repo_summary = {
        "repo": scan_result["repo"],
        "local_path": str(repo_path),
        "repo_type": metadata["repo_type"],
        "file_count": scan_result["file_count"],
        "languages": scan_result["languages"],
        "language_breakdown": metadata["language_breakdown"],
        "top_level_dirs": metadata["top_level_dirs"],
        "entry_points": metadata["entry_points"],
    }

    next_candidates = _build_next_candidates(scan_result, metadata, limit=5)
    unknowns = _build_unknowns(scan_result, metadata)
    confidence = _compute_confidence(scan_result, metadata)
    stop_reason = _derive_stop_reason(scan_result, next_candidates, confidence)

    analysis_state = {
        "repo_id": scan_result["repo"],
        "explored_files": [],
        "candidate_files": next_candidates,
        "unknowns": unknowns,
        "current_summary": repo_summary,
        "confidence": confidence,
        "stop_reason": stop_reason,
    }

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
        "unknowns": list(current_state.get("unknowns", [])),
        "current_summary": dict(current_state["current_summary"]),
        "confidence": float(current_state.get("confidence", 0.0)),
        "stop_reason": current_state.get("stop_reason"),
    }

    candidate_file = _select_next_candidate(next_state)
    if candidate_file is None:
        next_state["stop_reason"] = "No remaining candidate files to inspect."
        return next_state

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

    unknowns_before = list(next_state["unknowns"])
    _refine_summary(next_state, inspected)
    _reduce_unknowns(next_state, inspected)
    _add_follow_up_candidates(next_state, inspected)
    _update_confidence(next_state, unknowns_before)

    if not next_state["candidate_files"]:
        next_state["stop_reason"] = "No more meaningful candidates available."
    else:
        next_state["stop_reason"] = None

    return next_state


def run_analysis_loop(initial_state: Dict, max_steps: int = 5) -> Dict:
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

    return {
        "steps_executed": len(step_trace),
        "explored_files_in_order": explored_files_in_order,
        "step_trace": step_trace,
        "final_summary": current_state["current_summary"],
        "final_confidence": current_state["confidence"],
        "remaining_unknowns": current_state["unknowns"],
        "stop_reason": current_state.get("stop_reason"),
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
    score = 0.35

    file_count = scan_result["file_count"]
    languages = scan_result["languages"]

    if file_count > 0:
        score += 0.20
    if file_count >= 20:
        score += 0.10
    elif file_count < 5:
        score -= 0.10

    if len(languages) == 1:
        score += 0.15
    elif len(languages) > 1:
        score += 0.05

    if metadata["entry_points"]:
        score += 0.15
    if metadata["top_level_dirs"]:
        score += 0.10
    if metadata["repo_type"] == "mixed":
        score -= 0.10

    score = max(0.0, min(score, 0.95))
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
        "unknowns": list(state.get("unknowns", [])),
        "current_summary": dict(state["current_summary"]),
        "confidence": float(state.get("confidence", 0.0)),
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
    line_count = len(target.read_text(encoding="utf-8", errors="ignore").splitlines())

    return {
        "file_path": file_path,
        "name": target.name,
        "language": language,
        "top_level_dir": top_level_dir,
        "line_count": line_count,
    }


def _refine_summary(state: Dict, inspected: Dict) -> None:
    summary = state["current_summary"]
    file_path = inspected["file_path"]
    file_name = inspected["name"]
    top_level_dir = inspected["top_level_dir"]

    if file_name in ENTRY_POINT_FILES and file_path not in summary["entry_points"]:
        summary["entry_points"] = sorted(summary["entry_points"] + [file_path])

    if top_level_dir and top_level_dir not in summary["top_level_dirs"]:
        summary["top_level_dirs"] = sorted(summary["top_level_dirs"] + [top_level_dir])


def _reduce_unknowns(state: Dict, inspected: Dict) -> None:
    file_name = inspected["name"]

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


def _add_follow_up_candidates(state: Dict, inspected: Dict) -> None:
    repo_path = Path(state["current_summary"]["local_path"]).resolve()
    inspected_path = Path(inspected["file_path"])
    parent_dir = inspected_path.parent

    if str(parent_dir) == ".":
        return

    explored = set(state["explored_files"])
    existing = {c["file_path"] for c in state["candidate_files"]}
    added = 0

    for sibling in sorted((repo_path / parent_dir).glob("*")):
        if added >= 2:
            break
        if not sibling.is_file():
            continue

        rel_path = str(sibling.relative_to(repo_path))
        if rel_path == inspected["file_path"]:
            continue
        if rel_path in explored or rel_path in existing:
            continue
        if sibling.suffix.lower() not in EXTENSION_LANGUAGE_MAP:
            continue

        state["candidate_files"].append(
            {
                "file_path": rel_path,
                "reason": "Sibling module of inspected file for local context expansion.",
            }
        )
        existing.add(rel_path)
        added += 1


def _update_confidence(state: Dict, unknowns_before: List[str]) -> None:
    confidence = float(state["confidence"])
    confidence += 0.04

    if len(state["unknowns"]) < len(unknowns_before):
        confidence += 0.03

    state["confidence"] = round(max(0.0, min(confidence, 0.95)), 2)
