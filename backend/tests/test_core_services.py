"""
Unit tests for core analysis services.
Covers: import extraction, internal import resolution, noise file filtering,
and dependency graph computation.
"""
import tempfile
from pathlib import Path

import pytest

from app.services.analysis_snapshot_service import (
    _compute_dependency_graph_summary,
    _extract_imports_for_file,
    _extract_java_imports,
    _extract_go_imports,
    _extract_javascript_imports,
    _extract_python_imports,
    _resolve_internal_import,
)
from app.services.agentic_analysis_service import _is_noise_file


# ---------------------------------------------------------------------------
# _is_noise_file
# ---------------------------------------------------------------------------

class TestIsNoiseFile:
    def test_min_js(self):
        assert _is_noise_file("jquery.min.js") is True

    def test_min_css(self):
        assert _is_noise_file("bootstrap.min.css") is True

    def test_bundle_js(self):
        assert _is_noise_file("app.bundle.js") is True

    def test_chunk_js(self):
        assert _is_noise_file("main.chunk.js") is True

    def test_node_modules(self):
        assert _is_noise_file("node_modules/lodash/index.js") is True

    def test_vendor_dir(self):
        assert _is_noise_file("vendor/jquery.js") is True

    def test_vendors_dir(self):
        assert _is_noise_file("vendors/react.js") is True

    def test_dist_dir(self):
        assert _is_noise_file("dist/main.js") is True

    def test_build_dir(self):
        assert _is_noise_file("build/output.js") is True

    def test_normal_js(self):
        assert _is_noise_file("src/leetcode.js") is False

    def test_normal_py(self):
        assert _is_noise_file("app/services/auth.py") is False

    def test_normal_root_file(self):
        assert _is_noise_file("background.js") is False

    def test_min_in_name_not_suffix(self):
        # "admin.js" ends with .js not .min.js — should not be filtered
        assert _is_noise_file("admin.js") is False

    def test_nested_under_src(self):
        assert _is_noise_file("src/components/Button.tsx") is False


# ---------------------------------------------------------------------------
# Python import extraction
# ---------------------------------------------------------------------------

class TestExtractPythonImports:
    def test_simple_import(self):
        result = _extract_python_imports("import os\nimport sys\n")
        assert "os" in result
        assert "sys" in result

    def test_from_import(self):
        result = _extract_python_imports("from pathlib import Path\n")
        assert "pathlib" in result

    def test_relative_import(self):
        result = _extract_python_imports("from .utils import helper\n")
        assert ".utils" in result

    def test_relative_import_double_dot(self):
        result = _extract_python_imports("from ..models import User\n")
        assert "..models" in result

    def test_deduplication(self):
        result = _extract_python_imports("import os\nimport os\n")
        assert result.count("os") == 1

    def test_syntax_error_falls_back_to_regex(self):
        # Invalid Python — AST fails, regex fallback should still find the import
        result = _extract_python_imports("import os\n$$$invalid$$$\n")
        assert "os" in result

    def test_empty_content(self):
        assert _extract_python_imports("") == []

    def test_no_imports(self):
        assert _extract_python_imports("x = 1\nprint(x)\n") == []


# ---------------------------------------------------------------------------
# JavaScript/TypeScript import extraction
# ---------------------------------------------------------------------------

class TestExtractJavaScriptImports:
    def test_es_module_from(self):
        result = _extract_javascript_imports("import React from 'react';")
        assert "react" in result

    def test_es_module_named(self):
        result = _extract_javascript_imports("import { useState } from 'react';")
        assert "react" in result

    def test_require(self):
        result = _extract_javascript_imports("const fs = require('fs');")
        assert "fs" in result

    def test_relative_import(self):
        result = _extract_javascript_imports("import util from './util';")
        assert "./util" in result

    def test_bare_import(self):
        result = _extract_javascript_imports("import './styles.css';")
        assert "./styles.css" in result

    def test_deduplication(self):
        result = _extract_javascript_imports("import a from 'react';\nimport b from 'react';")
        assert result.count("react") == 1

    def test_empty_content(self):
        assert _extract_javascript_imports("") == []


# ---------------------------------------------------------------------------
# Java import extraction
# ---------------------------------------------------------------------------

class TestExtractJavaImports:
    def test_basic_import(self):
        result = _extract_java_imports("import java.util.List;\nimport java.util.Map;\n")
        assert "java.util.List" in result
        assert "java.util.Map" in result

    def test_static_import(self):
        result = _extract_java_imports("import static org.junit.Assert.assertEquals;\n")
        assert "org.junit.Assert.assertEquals" in result

    def test_custom_package(self):
        result = _extract_java_imports("import com.example.service.UserService;\n")
        assert "com.example.service.UserService" in result

    def test_deduplication(self):
        code = "import java.util.List;\nimport java.util.List;\n"
        result = _extract_java_imports(code)
        assert result.count("java.util.List") == 1

    def test_empty_content(self):
        assert _extract_java_imports("") == []

    def test_no_imports(self):
        assert _extract_java_imports("public class Foo {}") == []


# ---------------------------------------------------------------------------
# Go import extraction
# ---------------------------------------------------------------------------

class TestExtractGoImports:
    def test_single_import(self):
        result = _extract_go_imports('import "fmt"\n')
        assert "fmt" in result

    def test_grouped_imports(self):
        code = 'import (\n    "net/http"\n    "encoding/json"\n)\n'
        result = _extract_go_imports(code)
        assert "net/http" in result
        assert "encoding/json" in result

    def test_aliased_import_in_group(self):
        code = 'import (\n    alias "github.com/user/repo/pkg"\n)\n'
        result = _extract_go_imports(code)
        assert "github.com/user/repo/pkg" in result

    def test_deduplication(self):
        code = 'import "fmt"\nimport "fmt"\n'
        result = _extract_go_imports(code)
        assert result.count("fmt") == 1

    def test_empty_content(self):
        assert _extract_go_imports("") == []


# ---------------------------------------------------------------------------
# _extract_imports_for_file dispatch
# ---------------------------------------------------------------------------

class TestExtractImportsForFile:
    def test_dispatches_python(self):
        result = _extract_imports_for_file(content="import os\n", language="python")
        assert "os" in result

    def test_dispatches_javascript(self):
        result = _extract_imports_for_file(content="import x from 'y';", language="javascript")
        assert "y" in result

    def test_dispatches_typescript(self):
        result = _extract_imports_for_file(content="import x from 'y';", language="typescript")
        assert "y" in result

    def test_dispatches_java(self):
        result = _extract_imports_for_file(content="import java.util.List;\n", language="java")
        assert "java.util.List" in result

    def test_dispatches_go(self):
        result = _extract_imports_for_file(content='import "fmt"\n', language="go")
        assert "fmt" in result

    def test_unsupported_language_returns_empty(self):
        result = _extract_imports_for_file(content="#include <stdio.h>", language="c")
        assert result == []


# ---------------------------------------------------------------------------
# _resolve_internal_import
# ---------------------------------------------------------------------------

class TestResolveInternalImport:
    def _make_repo(self, files: dict) -> Path:
        """Create a temp repo with the given {relative_path: content} files."""
        # .resolve() ensures no symlink components (important on macOS where /tmp -> /private/tmp)
        tmp = Path(tempfile.mkdtemp()).resolve()
        for rel_path, content in files.items():
            full = tmp / rel_path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content)
        return tmp

    def test_js_relative_same_dir(self):
        repo = self._make_repo({
            "src/index.js": "",
            "src/util.js": "",
        })
        result = _resolve_internal_import(
            repo_path=repo,
            source_file="src/index.js",
            import_specifier="./util",
            package_roots=[],
            scanned_files={"src/index.js", "src/util.js"},
        )
        assert result == "src/util.js"

    def test_js_relative_parent_dir(self):
        repo = self._make_repo({
            "src/components/Button.js": "",
            "src/util.js": "",
        })
        result = _resolve_internal_import(
            repo_path=repo,
            source_file="src/components/Button.js",
            import_specifier="../util",
            package_roots=[],
            scanned_files={"src/components/Button.js", "src/util.js"},
        )
        assert result == "src/util.js"

    def test_python_relative_import(self):
        repo = self._make_repo({
            "app/__init__.py": "",
            "app/main.py": "",
            "app/utils.py": "",
        })
        result = _resolve_internal_import(
            repo_path=repo,
            source_file="app/main.py",
            import_specifier=".utils",
            package_roots=[repo / "app"],
            scanned_files={"app/__init__.py", "app/main.py", "app/utils.py"},
        )
        assert result == "app/utils.py"

    def test_external_package_returns_none(self):
        repo = self._make_repo({"src/index.js": ""})
        result = _resolve_internal_import(
            repo_path=repo,
            source_file="src/index.js",
            import_specifier="react",
            package_roots=[],
            scanned_files={"src/index.js"},
        )
        assert result is None

    def test_nonexistent_relative_returns_none(self):
        repo = self._make_repo({"src/index.js": ""})
        result = _resolve_internal_import(
            repo_path=repo,
            source_file="src/index.js",
            import_specifier="./doesnotexist",
            package_roots=[],
            scanned_files={"src/index.js"},
        )
        assert result is None


# ---------------------------------------------------------------------------
# _compute_dependency_graph_summary
# ---------------------------------------------------------------------------

class TestComputeDependencyGraphSummary:
    def _make_state(self, edges, repo: Path):
        return {
            "dependency_edges": edges,
            "current_summary": {"local_path": str(repo)},
            "package_roots": [],
            "inspected_facts": [],
            "explored_files": [],
        }

    def test_empty_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            state = self._make_state([], repo)
            result = _compute_dependency_graph_summary(state)
            assert result["internal_edges"] == []
            assert result["most_imported_modules"] == []
            assert result["highest_dependency_files"] == []
            assert result["clusters"] == []

    def test_counts_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            state = self._make_state([
                {"source": "a.py", "imports": ["os", "sys", "os"]},
                {"source": "b.py", "imports": ["os"]},
            ], repo)
            result = _compute_dependency_graph_summary(state)
            module_counts = {m["module"]: m["count"] for m in result["most_imported_modules"]}
            # os deduped per-file: a.py counts once, b.py counts once → total 2
            assert module_counts["os"] == 2

    def test_highest_dependency_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            state = self._make_state([
                {"source": "hub.py", "imports": ["a", "b", "c", "d"]},
                {"source": "leaf.py", "imports": ["a"]},
            ], repo)
            result = _compute_dependency_graph_summary(state)
            top = result["highest_dependency_files"][0]
            assert top["source"] == "hub.py"
            assert top["imports_count"] == 4

    def test_internal_edges_capped_at_500(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            # Create 601 real JS files so scan_repository finds them
            src = repo / "src"
            src.mkdir()
            for i in range(601):
                (src / f"f{i}.js").write_text("")
            edges = [
                {"source": f"src/f{i}.js", "imports": [f"./f{i+1}"]}
                for i in range(600)
            ]
            state = self._make_state(edges, repo)
            result = _compute_dependency_graph_summary(state)
            assert len(result["internal_edges"]) <= 500
