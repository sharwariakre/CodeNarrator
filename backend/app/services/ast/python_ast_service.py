from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Optional

from app.services.ast.ast_types import ASTFile, ASTNode, ASTResult


def extract_python_ast_for_repo(*, repo_path: Path, repo_slug: str, python_files: List[str]) -> ASTResult:
    """
    Extract a structural AST for a repository's Python files.

    Structural scope (by design):
      - module root
      - class definitions
      - function definitions (incl. methods)
      - nested class/function defs

    Skips:
      - statements/expressions/control flow (for now)
    """

    ast_files: List[ASTFile] = []

    for rel_path in python_files:
        abs_path = repo_path / rel_path
        root = _parse_python_file(abs_path)

        # If file can't be parsed, we still return a node that marks the file boundary.
        # We keep it deterministic and non-fatal.
        if root is None:
            file_root: ASTNode = {
                "node_type": "module",
                "name": None,
                "start_line": 1,
                "end_line": 1,
                "children": [],
            }
        else:
            file_root = root

        ast_files.append(
            {
                "file_path": rel_path,
                "language": "python",
                "root": file_root,
            }
        )

    return {
        "repo": repo_slug,
        "language": "python",
        "files": ast_files,
    }


def _parse_python_file(file_path: Path) -> Optional[ASTNode]:
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Non-fatal: return None; caller will handle with empty module node
        return None

    # Build a minimal structural AST rooted at module
    children = _extract_def_children(tree.body)

    start_line = 1
    end_line = _safe_end_line_from_children(children)

    return {
        "node_type": "module",
        "name": None,
        "start_line": start_line,
        "end_line": end_line,
        "children": children,
    }


def _extract_def_children(stmts: List[ast.stmt]) -> List[ASTNode]:
    """
    Extract only definition-like nodes in source order.
    """
    out: List[ASTNode] = []
    for stmt in stmts:
        node = _convert_stmt_if_def(stmt)
        if node is not None:
            out.append(node)
    return out


def _convert_stmt_if_def(stmt: ast.stmt) -> Optional[ASTNode]:
    if isinstance(stmt, ast.ClassDef):
        return _convert_class(stmt)
    if isinstance(stmt, ast.FunctionDef):
        return _convert_function(stmt, async_fn=False)
    if isinstance(stmt, ast.AsyncFunctionDef):
        return _convert_function(stmt, async_fn=True)

    # Skip everything else for Phase 3B (intentionally)
    return None


def _convert_class(node: ast.ClassDef) -> ASTNode:
    start_line, end_line = _node_span(node)
    children = _extract_def_children(node.body)

    # Make sure end_line covers nested defs even if end_lineno is missing
    end_line = max(end_line, _safe_end_line_from_children(children))

    return {
        "node_type": "class",
        "name": node.name,
        "start_line": start_line,
        "end_line": end_line,
        "children": children,
    }


def _convert_function(node: ast.AST, async_fn: bool) -> ASTNode:
    # node is FunctionDef or AsyncFunctionDef
    start_line, end_line = _node_span(node)
    body = getattr(node, "body", [])
    children = _extract_def_children(body)

    end_line = max(end_line, _safe_end_line_from_children(children))

    return {
        "node_type": "async_function" if async_fn else "function",
        "name": getattr(node, "name", None),
        "start_line": start_line,
        "end_line": end_line,
        "children": children,
    }


def _node_span(node: ast.AST) -> (int, int):
    """
    Best-effort span for a node.
    Python 3.8+ usually provides end_lineno. If missing, fall back to lineno.
    """
    start = int(getattr(node, "lineno", 1) or 1)
    end = getattr(node, "end_lineno", None)
    if end is None:
        end = start
    return start, int(end)


def _safe_end_line_from_children(children: List[ASTNode]) -> int:
    """
    Determine an end_line for a parent node from its children.
    If no children exist, default to 1.
    """
    if not children:
        return 1
    return max(int(ch.get("end_line", 1)) for ch in children)
