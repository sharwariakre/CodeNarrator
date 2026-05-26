"""
codenarrator — autonomous codebase understanding pipeline.

Public API:

    from codenarrator import analyze

    result = analyze("https://github.com/owner/repo", depth="standard")
    result.show()                  # open the HTML report in a browser
    result.to_json("graph.json")   # dump the dependency graph
    print(result.explored_files)

For now the package wraps the existing FastAPI backend's service layer
in-process; it does not require a running server. The backend's
``app.*`` modules are added to ``sys.path`` at import time.
"""
from codenarrator.pipeline import analyze
from codenarrator.result import AnalysisResult

__all__ = ["analyze", "AnalysisResult"]
__version__ = "0.1.0"
