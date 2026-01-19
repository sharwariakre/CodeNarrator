from pathlib import Path
from typing import Dict, List, Set

# Directories we never want to scan
IGNORE_DIRS = {
    ".git",
    "node_modules",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".idea",
    ".vscode",
}

# File extensions we care about (can expand later)
EXTENSION_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
}


def scan_repository(repo_path: Path) -> Dict:
    """
    Walk a cloned repository and extract a structured view of its contents.

    Returns:
        {
            "repo": "<repo_name>",
            "languages": [...],
            "file_count": int,
            "files": [...]
        }
    """

    if not repo_path.exists() or not repo_path.is_dir():
        raise ValueError(f"Repository path does not exist: {repo_path}")

    files: List[str] = []
    languages: Set[str] = set()

    for path in repo_path.rglob("*"):
    if path.is_dir():
        continue

    if any(part in IGNORE_DIRS for part in path.parts):
        continue

    suffix = path.suffix.lower()

    if suffix in EXTENSION_LANGUAGE_MAP:
        relative_path = path.relative_to(repo_path)
        files.append(str(relative_path))
        languages.add(EXTENSION_LANGUAGE_MAP[suffix])

    return {
        "repo": repo_path.name,
        "languages": sorted(languages),
        "file_count": len(files),
        "files": sorted(files),
    }
