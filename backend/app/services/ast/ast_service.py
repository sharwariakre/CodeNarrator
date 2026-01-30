from pathlib import Path
from typing import Optional

from app.core.language_registry import LANGUAGE_REGISTRY
from app.services.ast.ast_types import ASTResult
from app.services.repo_scanner import scan_repository
from app.services.ast.python_ast_service import extract_python_ast_for_repo


def extract_ast(*, repo_path: Path, language: str) -> Optional[ASTResult]:
    """
    Extract ASTs for all files of a given language in a repository.

    - Consults the language registry
    - Returns None if unsupported
    - Deterministic
    """

    capabilities = LANGUAGE_REGISTRY.get(language, LANGUAGE_REGISTRY["unknown"])
    if not capabilities["has_ast"]:
        return None

    scan = scan_repository(repo_path)

    # Use scanner’s language mapping so we don’t duplicate logic
    file_languages = scan.get("file_languages", {})
    files = scan.get("files", [])
    repo_slug = scan.get("repo", repo_path.name)

    if language == "python":
        python_files = [f for f in files if file_languages.get(f) == "python"]
        return extract_python_ast_for_repo(
            repo_path=repo_path,
            repo_slug=repo_slug,
            python_files=python_files,
        )

    # Future languages will plug in here in their own services
    raise NotImplementedError(f"AST extraction not yet implemented for language: {language}")
