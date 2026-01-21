from pathlib import Path
from typing import Dict, List, Set
from collections import Counter

# Heuristic entry-point filenames (language-agnostic)
ENTRY_POINT_FILES = {
    "main.py",
    "app.py",
    "__main__.py",
    "index.js",
    "server.js",
    "index.ts",
    "server.ts",
    "Main.java",
}

# Known structural directories (signals only)
KNOWN_TOP_LEVEL_DIRS = {
    "src",
    "backend",
    "frontend",
    "app",
    "tests",
    "docs",
}


def extract_repo_metadata(repo_path: Path, scan_result: Dict) -> Dict:
    """
    Derive deterministic repository-level metadata from scan results.

    Args:
        repo_path: Path to cloned repository
        scan_result: Output from scan_repository()

    Returns:
        {
            "top_level_dirs": [...],
            "language_breakdown": {...},
            "entry_points": [...],
            "repo_type": str
        }
    """

    files: List[str] = scan_result["files"]
    file_languages: Dict[str, str] = scan_result["file_languages"]
    languages: List[str] = scan_result["languages"]

    # ---- Top-level directories ----
    top_level_dirs: Set[str] = set()

    for item in repo_path.iterdir():
        if item.is_dir() and item.name in KNOWN_TOP_LEVEL_DIRS:
            top_level_dirs.add(item.name)

    # ---- Language breakdown (generic, data-driven) ----
    language_breakdown = Counter(file_languages.values())

    # ---- Entry point detection ----
    entry_points: List[str] = []

    for file_path in files:
        if Path(file_path).name in ENTRY_POINT_FILES:
            entry_points.append(file_path)

    # ---- Repo type heuristic ----
    repo_type = classify_repo_type(
        languages=languages,
        top_level_dirs=top_level_dirs,
        entry_points=entry_points,
    )

    return {
        "top_level_dirs": sorted(top_level_dirs),
        "language_breakdown": dict(language_breakdown),
        "entry_points": sorted(entry_points),
        "repo_type": repo_type,
    }


def classify_repo_type(
    *,
    languages: List[str],
    top_level_dirs: Set[str],
    entry_points: List[str],
) -> str:
    """
    Deterministic, explainable repository classification.
    No ML. No parsing. No magic.
    """

    if len(languages) > 1:
        return "mixed"

    if {"frontend", "src"} & top_level_dirs:
        if any(lang in {"javascript", "typescript"} for lang in languages):
            return "frontend"

    if entry_points:
        return "service"

    if len(languages) == 1:
        return "library"

    return "unknown"
