from typing import List, Dict, Optional, TypedDict


class ASTNode(TypedDict):
    """
    Language-agnostic AST node.
    This is intentionally minimal.
    """
    node_type: str
    name: Optional[str]
    start_line: int
    end_line: int
    children: List["ASTNode"]


class ASTFile(TypedDict):
    """
    AST representation for a single file.
    """
    file_path: str
    language: str
    root: ASTNode


class ASTResult(TypedDict):
    """
    Result of AST extraction for a repository or subset.
    """
    repo: str
    language: str
    files: List[ASTFile]
