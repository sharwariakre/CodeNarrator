from pathlib import Path
from typing import Optional

from app.core.language_registry import LANGUAGE_REGISTRY
from app.services.ast.ast_types import ASTResult


def extract_ast(
    *,
    repo_path: Path,
    language: str,
) -> Optional[ASTResult]:
    """
    Extract ASTs for all files of a given language in a repository.

    This function:
    - MUST consult the language registry
    - MUST return None if AST extraction is unsupported
    - MUST be deterministic

    Parsing logic is intentionally NOT implemented here yet.
    """

    capabilities = LANGUAGE_REGISTRY.get(
        language, LANGUAGE_REGISTRY["unknown"]
    )

    if not capabilities["has_ast"]:
        return None

    # Parsing logic will be implemented in a later phase
    raise NotImplementedError(
        f"AST extraction not yet implemented for language: {language}"
    )
