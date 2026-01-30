"""
Language capability registry for CodeNarrator.

This file declares WHAT the system knows about languages.
It must NOT contain parsing logic, heuristics, or side effects.

Services may consult this registry to decide what actions are possible.
"""

from enum import Enum
from typing import Dict, TypedDict


class ParserBackend(str, Enum):
    TREE_SITTER = "tree_sitter"
    NONE = "none"


class LanguageCapabilities(TypedDict):
    has_ast: bool
    parser_backend: ParserBackend
    supports_import_graph: bool
    supports_entry_points: bool
    status: str  # stable | experimental | unknown


# -------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH FOR LANGUAGE CAPABILITIES
# -------------------------------------------------------------------

LANGUAGE_REGISTRY: Dict[str, LanguageCapabilities] = {
    "python": {
        "has_ast": True,
        "parser_backend": ParserBackend.TREE_SITTER,
        "supports_import_graph": True,
        "supports_entry_points": True,
        "status": "stable",
    },
    "javascript": {
        "has_ast": True,
        "parser_backend": ParserBackend.TREE_SITTER,
        "supports_import_graph": True,
        "supports_entry_points": True,
        "status": "stable",
    },
    "typescript": {
        "has_ast": True,
        "parser_backend": ParserBackend.TREE_SITTER,
        "supports_import_graph": True,
        "supports_entry_points": True,
        "status": "stable",
    },
    "java": {
        "has_ast": True,
        "parser_backend": ParserBackend.TREE_SITTER,
        "supports_import_graph": True,
        "supports_entry_points": True,
        "status": "experimental",
    },
    "cpp": {
        "has_ast": False,
        "parser_backend": ParserBackend.NONE,
        "supports_import_graph": False,
        "supports_entry_points": False,
        "status": "unknown",
    },
    "c": {
        "has_ast": False,
        "parser_backend": ParserBackend.NONE,
        "supports_import_graph": False,
        "supports_entry_points": False,
        "status": "unknown",
    },
    "rust": {
        "has_ast": False,
        "parser_backend": ParserBackend.NONE,
        "supports_import_graph": False,
        "supports_entry_points": False,
        "status": "unknown",
    },
    "go": {
        "has_ast": False,
        "parser_backend": ParserBackend.NONE,
        "supports_import_graph": False,
        "supports_entry_points": False,
        "status": "unknown",
    },
    "unknown": {
        "has_ast": False,
        "parser_backend": ParserBackend.NONE,
        "supports_import_graph": False,
        "supports_entry_points": False,
        "status": "unknown",
    },
}
