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

# File extensions → language mapping
# This is the SINGLE source of truth for language detection
EXTENSION_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".rs": "rust",
    ".go": "go",
}


def scan_repository(repo_path: Path) -> Dict:
    """
    Walk a cloned repository and extract a structured view of its contents.

    Returns:
        {
            "repo": "<repo_name>",
            "languages": [...],
            "file_count": int,
            "files": [...],
            "file_languages": { "<relative_path>": "<language>" }
        }
    """

    if not repo_path.exists() or not repo_path.is_dir():
        raise ValueError(f"Repository path does not exist: {repo_path}")

    files: List[str] = []
    languages: Set[str] = set()
    file_languages: Dict[str, str] = {}

    for path in repo_path.rglob("*"):
        if path.is_dir():
            continue

        # Skip files inside ignored directories
        if any(part in IGNORE_DIRS for part in path.parts):
            continue

        suffix = path.suffix.lower()

        if suffix in EXTENSION_LANGUAGE_MAP:
            relative_path = str(path.relative_to(repo_path))
            language = EXTENSION_LANGUAGE_MAP[suffix]

            files.append(relative_path)
            file_languages[relative_path] = language
            languages.add(language)

    return {
        "repo": repo_path.name,
        "languages": sorted(languages),
        "file_count": len(files),
        "files": sorted(files),
        "file_languages": file_languages,
    }
