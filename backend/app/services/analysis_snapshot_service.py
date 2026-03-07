from pathlib import Path
from typing import Dict, List, Set, Tuple

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
