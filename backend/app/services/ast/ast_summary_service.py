from typing import Dict, List

from app.services.ast.ast_types import ASTResult, ASTNode


def summarize_ast(ast_result: ASTResult) -> Dict:
    """
    Compute a deterministic, repo-level summary from an ASTResult.

    This function performs NO parsing and NO file IO.
    It only derives facts from the AST structure.

    Returns a JSON-serializable dictionary.
    """

    total_files = len(ast_result["files"])
    total_classes = 0
    total_functions = 0
    max_nesting_depth = 0

    file_summaries: List[Dict] = []

    for ast_file in ast_result["files"]:
        root = ast_file["root"]

        file_classes = 0
        file_functions = 0
        file_max_depth = 0

        def walk(node: ASTNode, depth: int):
            nonlocal file_classes, file_functions, file_max_depth

            file_max_depth = max(file_max_depth, depth)

            node_type = node["node_type"]
            if node_type == "class":
                file_classes += 1
            elif node_type in {"function", "async_function"}:
                file_functions += 1

            for child in node.get("children", []):
                walk(child, depth + 1)

        walk(root, depth=1)

        total_classes += file_classes
        total_functions += file_functions
        max_nesting_depth = max(max_nesting_depth, file_max_depth)

        file_summaries.append(
            {
                "file_path": ast_file["file_path"],
                "classes": file_classes,
                "functions": file_functions,
                "max_depth": file_max_depth,
            }
        )

    # Sort files by structural complexity (descending)
    file_summaries.sort(
        key=lambda f: (f["classes"] + f["functions"], f["max_depth"]),
        reverse=True,
    )

    return {
        "repo": ast_result["repo"],
        "language": ast_result["language"],
        "files_analyzed": total_files,
        "total_classes": total_classes,
        "total_functions": total_functions,
        "max_nesting_depth": max_nesting_depth,
        "most_complex_files": file_summaries[:5],
    }
