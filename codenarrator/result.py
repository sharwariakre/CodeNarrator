"""
AnalysisResult — the value object returned by ``analyze()``.

Bundles the final state, AI interpretation, and rendered HTML so the
caller can render it inline, dump JSON, or pull individual fields.
"""
from __future__ import annotations

import json
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional


class AnalysisResult:
    """All artifacts produced by a single ``analyze()`` invocation."""

    def __init__(
        self,
        state: Dict[str, Any],
        interpretation: Optional[Dict[str, Any]],
        report_html: str,
        report_path: str,
        llm: str = "ollama",
    ):
        self.state = state
        self.interpretation = interpretation
        self.report_html = report_html
        self.report_path = report_path
        self.llm = llm

    # ---- side-effect actions -------------------------------------------------

    def show(self) -> None:
        """Open the rendered HTML report in the system's default browser."""
        webbrowser.open(f"file://{self.report_path}")

    def to_html(self, path: str) -> None:
        """Write the rendered HTML to ``path``."""
        Path(path).write_text(self.report_html, encoding="utf-8")

    def to_json(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Return a JSON-serializable snapshot of the core analysis artifacts.
        If ``path`` is given, also write the JSON to that path.
        """
        graph = self.state.get("dependency_graph_summary", {})
        data = {
            "internal_edges": graph.get("internal_edges", []),
            "explored_files": self.state.get("explored_files", []),
            "file_count": self.state.get("current_summary", {}).get("file_count", 0),
        }
        if path:
            Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

    # ---- convenience accessors -----------------------------------------------

    @property
    def dependency_graph(self) -> List[Dict[str, str]]:
        """Resolved internal file-to-file edges (``[{"from": ..., "to": ...}, ...]``)."""
        return self.state.get("dependency_graph_summary", {}).get("internal_edges", [])

    @property
    def explored_files(self) -> List[str]:
        """Paths of every file the agent opened during exploration."""
        return self.state.get("explored_files", [])

    @property
    def architecture_summary(self) -> Optional[Dict[str, Any]]:
        """The AI interpretation dict (``architecture_pattern``, ``main_components``, ...)."""
        return self.interpretation
