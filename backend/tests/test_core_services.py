"""
Unit tests for core analysis services.
Covers: import extraction, internal import resolution, noise file filtering,
and dependency graph computation.
"""
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from app.services.analysis_snapshot_service import (
    _candidate_signal_score,
    _compute_dependency_graph_summary,
    _detect_python_package_roots,
    _extract_imports_for_file,
    _extract_java_imports,
    _extract_go_imports,
    _extract_javascript_imports,
    _extract_python_imports,
    _known_top_level_package_names,
    _refresh_candidates_for_signal,
    _resolve_internal_import,
    _resolved_import_targets,
)
from app.services.agentic_analysis_service import (
    _build_language_guidance,
    _is_noise_file,
    _next_unexplored,
    _nudge_message,
)
from app.services import report_generator
from app.services.report_generator import generate_html_report
from app.services import ai_interpreter
from app.services.ai_interpreter import (
    _build_fallback_key_dependencies,
    _build_prompt,
    _validate_interpretation,
    interpret_architecture,
)
from app.services.analysis_state_store import (
    delete_state,
    load_state,
    save_state,
)


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

    def test_reexport_named(self):
        result = _extract_javascript_imports("export { Button } from './Button';")
        assert "./Button" in result

    def test_reexport_star(self):
        result = _extract_javascript_imports("export * from './utils';")
        assert "./utils" in result

    def test_reexport_aliased(self):
        result = _extract_javascript_imports("export { Foo as Bar } from './foo';")
        assert "./foo" in result

    def test_dynamic_import(self):
        result = _extract_javascript_imports("const m = import('./lazy');")
        assert "./lazy" in result

    def test_dynamic_import_chained(self):
        result = _extract_javascript_imports("import('./lazy').then(m => m.default);")
        assert "./lazy" in result


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

    def test_dispatches_c(self):
        result = _extract_imports_for_file(content="#include <stdio.h>", language="c")
        assert "stdio.h" in result

    def test_unknown_language_returns_empty(self):
        result = _extract_imports_for_file(content="some content", language="unknown")
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

    # ---- Java ----

    def test_java_maven_layout(self):
        repo = self._make_repo({
            "src/main/java/com/example/Foo.java": "",
            "src/main/java/com/example/Bar.java": "",
        })
        scanned = {"src/main/java/com/example/Foo.java", "src/main/java/com/example/Bar.java"}
        result = _resolve_internal_import(
            repo_path=repo,
            source_file="src/main/java/com/example/Foo.java",
            import_specifier="com.example.Bar",
            package_roots=[],
            scanned_files=scanned,
        )
        assert result == "src/main/java/com/example/Bar.java"

    def test_java_static_import_drops_member(self):
        repo = self._make_repo({
            "com/example/Foo.java": "",
            "com/example/Bar.java": "",
        })
        scanned = {"com/example/Foo.java", "com/example/Bar.java"}
        result = _resolve_internal_import(
            repo_path=repo,
            source_file="com/example/Bar.java",
            import_specifier="com.example.Foo.someMethod",
            package_roots=[],
            scanned_files=scanned,
        )
        assert result == "com/example/Foo.java"

    def test_java_external_returns_none(self):
        repo = self._make_repo({"src/Foo.java": ""})
        result = _resolve_internal_import(
            repo_path=repo,
            source_file="src/Foo.java",
            import_specifier="java.util.List",
            package_roots=[],
            scanned_files={"src/Foo.java"},
        )
        assert result is None

    # ---- Go ----

    def test_go_relative_import(self):
        repo = self._make_repo({
            "main.go": "",
            "pkg/util.go": "",
        })
        result = _resolve_internal_import(
            repo_path=repo,
            source_file="main.go",
            import_specifier="./pkg",
            package_roots=[],
            scanned_files={"main.go", "pkg/util.go"},
        )
        assert result == "pkg/util.go"

    def test_go_module_path_suffix_match(self):
        repo = self._make_repo({
            "cmd/server/main.go": "",
            "internal/auth/auth.go": "",
        })
        result = _resolve_internal_import(
            repo_path=repo,
            source_file="cmd/server/main.go",
            import_specifier="github.com/user/project/internal/auth",
            package_roots=[],
            scanned_files={"cmd/server/main.go", "internal/auth/auth.go"},
        )
        assert result == "internal/auth/auth.go"

    def test_go_external_returns_none(self):
        repo = self._make_repo({"main.go": ""})
        result = _resolve_internal_import(
            repo_path=repo,
            source_file="main.go",
            import_specifier="fmt",
            package_roots=[],
            scanned_files={"main.go"},
        )
        assert result is None

    # ---- Rust ----

    def test_rust_use_crate_path(self):
        repo = self._make_repo({
            "src/lib.rs": "",
            "src/util.rs": "",
        })
        result = _resolve_internal_import(
            repo_path=repo,
            source_file="src/lib.rs",
            import_specifier="crate::util",
            package_roots=[],
            scanned_files={"src/lib.rs", "src/util.rs"},
        )
        assert result == "src/util.rs"

    def test_rust_use_crate_item_falls_back_to_module(self):
        # crate::util::helper — helper is an item inside util.rs, not its own module.
        repo = self._make_repo({
            "src/lib.rs": "",
            "src/util.rs": "",
        })
        result = _resolve_internal_import(
            repo_path=repo,
            source_file="src/lib.rs",
            import_specifier="crate::util::helper",
            package_roots=[],
            scanned_files={"src/lib.rs", "src/util.rs"},
        )
        assert result == "src/util.rs"

    def test_rust_mod_declaration(self):
        repo = self._make_repo({
            "src/main.rs": "",
            "src/handlers.rs": "",
        })
        result = _resolve_internal_import(
            repo_path=repo,
            source_file="src/main.rs",
            import_specifier="handlers",
            package_roots=[],
            scanned_files={"src/main.rs", "src/handlers.rs"},
        )
        assert result == "src/handlers.rs"

    def test_rust_extern_crate_returns_none(self):
        repo = self._make_repo({"src/lib.rs": ""})
        result = _resolve_internal_import(
            repo_path=repo,
            source_file="src/lib.rs",
            import_specifier="serde",
            package_roots=[],
            scanned_files={"src/lib.rs"},
        )
        assert result is None

    # ---- C / C++ ----

    def test_c_quoted_include_same_dir(self):
        repo = self._make_repo({
            "src/main.c": "",
            "src/utils.h": "",
        })
        result = _resolve_internal_import(
            repo_path=repo,
            source_file="src/main.c",
            import_specifier="utils.h",
            package_roots=[],
            scanned_files={"src/main.c", "src/utils.h"},
        )
        assert result == "src/utils.h"

    def test_c_quoted_include_project_root(self):
        repo = self._make_repo({
            "src/main.c": "",
            "config.h": "",
        })
        result = _resolve_internal_import(
            repo_path=repo,
            source_file="src/main.c",
            import_specifier="config.h",
            package_roots=[],
            scanned_files={"src/main.c", "config.h"},
        )
        assert result == "config.h"

    def test_c_system_header_returns_none(self):
        repo = self._make_repo({"src/main.c": ""})
        result = _resolve_internal_import(
            repo_path=repo,
            source_file="src/main.c",
            import_specifier="stdio.h",
            package_roots=[],
            scanned_files={"src/main.c"},
        )
        assert result is None

    # ---- _resolved_import_targets ----

    def test_resolved_import_targets_uses_real_resolver_with_repo_context(self):
        # The bug we fixed: ./formFieldsSource was being stored as
        # "datasources/formFieldsSource" (no extension), which never matched
        # the actual file "datasources/formFieldsSource.ts" in scanned_files.
        # With repo context, the resolver now returns the correct path.
        repo = self._make_repo({
            "datasources/index.ts": "",
            "datasources/formFieldsSource.ts": "",
        })
        state = {
            "dependency_edges": [
                {
                    "source": "datasources/index.ts",
                    "imports": ["./formFieldsSource"],
                }
            ],
        }
        scanned = {"datasources/index.ts", "datasources/formFieldsSource.ts"}

        result = _resolved_import_targets(
            state,
            repo_path=repo,
            package_roots=[],
            scanned_files=scanned,
        )

        assert "datasources/formFieldsSource.ts" in result
        # The naive extension-less path must NOT be in the result.
        assert "datasources/formFieldsSource" not in result

    def test_resolved_import_targets_falls_back_without_repo_context(self):
        # Backward compat: callers that don't pass repo context still get the
        # naive arithmetic. The dead-code call sites still rely on this.
        state = {
            "dependency_edges": [
                {
                    "source": "datasources/index.ts",
                    "imports": ["./formFieldsSource"],
                }
            ],
        }

        result = _resolved_import_targets(state)

        # Naive arithmetic preserves the legacy (broken) behavior — confirms
        # the fallback path is exercised when no repo context is passed.
        assert "datasources/formFieldsSource" in result


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


# ---------------------------------------------------------------------------
# _build_language_guidance
# ---------------------------------------------------------------------------

class TestBuildLanguageGuidance:
    def test_empty_when_no_recognized_languages(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert _build_language_guidance([], [], Path(tmp)) == ""
            assert _build_language_guidance(["cobol"], [], Path(tmp)) == ""

    def test_python_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_language_guidance(["python"], [], Path(tmp))
            assert "Python entry points" in out
            assert "manage.py" in out
            assert "urls.py" in out

    def test_javascript_general_when_no_framework(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_language_guidance(["javascript"], ["src/index.js"], Path(tmp))
            assert "General JS/TS" in out
            assert "Next.js" not in out
            assert "Vite" not in out
            # Breadth-first instruction appended to the general JS/TS block.
            assert "read one file per directory" in out

    def test_typescript_picks_up_next_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_language_guidance(
                ["typescript"], ["next.config.ts", "app/page.tsx"], Path(tmp)
            )
            assert "Next.js detected" in out
            assert "app/" in out

    def test_typescript_picks_up_vite_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_language_guidance(
                ["typescript"], ["vite.config.ts", "src/main.tsx"], Path(tmp)
            )
            assert "Vite detected" in out
            assert "src/main.tsx" in out
            # Breadth-first instruction in the Vite block.
            assert "SPREAD OUT" in out

    def test_reads_package_json_main_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "package.json").write_text('{"main": "./dist/index.js"}')
            out = _build_language_guidance(["javascript"], [], repo)
            assert "./dist/index.js" in out

    def test_falls_back_to_module_when_main_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "package.json").write_text('{"module": "./esm/index.mjs"}')
            out = _build_language_guidance(["javascript"], [], repo)
            assert "./esm/index.mjs" in out

    def test_malformed_package_json_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "package.json").write_text("{ not valid json")
            out = _build_language_guidance(["javascript"], [], repo)
            # Should still produce general JS/TS guidance, just no entry hint.
            assert "General JS/TS" in out
            assert "package.json entry" not in out

    def test_java_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_language_guidance(["java"], [], Path(tmp))
            assert "Java entry points" in out
            assert "Application.java" in out

    def test_go_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_language_guidance(["go"], [], Path(tmp))
            assert "Go entry points" in out
            assert "main.go" in out

    def test_rust_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_language_guidance(["rust"], [], Path(tmp))
            assert "Rust entry points" in out
            assert "src/main.rs" in out
            assert "mod declarations" in out

    def test_multiple_languages_concatenated(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_language_guidance(
                ["python", "typescript", "go"], ["src/index.ts"], Path(tmp)
            )
            assert "Python entry points" in out
            assert "General JS/TS" in out
            assert "Go entry points" in out


# ---------------------------------------------------------------------------
# generate_html_report — smoke test for the legend and phantom tooltip
# ---------------------------------------------------------------------------

class TestGenerateHtmlReport:
    def _make_final_state(self, repo: Path) -> dict:
        return {
            "repo_id": "test",
            "explored_files": ["a.py"],
            "candidate_files": [],
            "inspected_facts": [
                {
                    "file_path": "a.py",
                    "language": "python",
                    "line_count_bucket": "small",
                    "directory": ".",
                    "role_hint": "module",
                    "imports_found": 1,
                    "imported_modules": ["b"],
                }
            ],
            "dependency_edges": [{"source": "a.py", "imports": ["./b"]}],
            "dependency_graph_summary": {},
            "package_roots": ["."],
            "unknowns": [],
            "current_summary": {
                "repo": "test",
                "local_path": str(repo),
                "repo_type": "library",
                "file_count": 2,
                "languages": ["python"],
                "language_breakdown": {"python": 2},
                "top_level_dirs": [],
                "entry_points": [],
                "inspected_languages": ["python"],
                "inspected_role_hints": ["module"],
            },
            "confidence": 0.5,
            "no_progress_steps": 0,
            "stop_reason": None,
        }

    def test_legend_and_phantom_tooltip_present(self, monkeypatch):
        # Skip the network fetch for D3; the report still renders with a CDN tag fallback.
        monkeypatch.setattr(report_generator, "_D3_CACHE", "// stubbed", raising=False)

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            (repo / "a.py").write_text("from . import b\n")
            (repo / "b.py").write_text("")

            output = repo / "report.html"
            generate_html_report(self._make_final_state(repo), None, output)
            content = output.read_text(encoding="utf-8")

            # Structured legend
            assert "Solid node" in content
            assert "explored by agent" in content
            assert "Dashed node" in content
            assert "referenced in imports but not explored" in content
            assert "Node size" in content
            assert "in-degree" in content
            assert "Node color" in content

            # Phantom node tooltip branch
            assert 'd.cluster === "unvisited"' in content
            assert "Referenced in imports but not explored by the agent." in content


# ---------------------------------------------------------------------------
# ai_interpreter — prompt constraint + deterministic key_dependencies fallback
# ---------------------------------------------------------------------------

class TestInterpreterPromptAndFallback:
    def test_prompt_contains_file_to_file_constraint(self):
        prompt = _build_prompt({
            "internal_edges": [],
            "clusters": [],
            "highest_dependency_files": [],
            "inspected_facts": [],
        })
        # Hard constraint that distinguishes file-to-file edges from external libs.
        assert "Both 'from' and 'to' must be actual file paths" in prompt
        # Single-sentence example replaces the inline JSON pair (kept short to
        # avoid encouraging long model output that risks the Ollama timeout).
        assert "is valid" in prompt
        assert "is not" in prompt
        assert "fastapi" in prompt

    def test_fallback_ranks_by_target_in_degree(self):
        graph_summary = {
            "internal_edges": [
                {"from": "app/main.py",          "to": "app/services/auth.py"},
                {"from": "app/services/auth.py", "to": "app/models/user.py"},
                {"from": "app/main.py",          "to": "app/models/user.py"},
                # user.py has in-degree 2; auth.py has in-degree 1.
            ],
        }
        result = _build_fallback_key_dependencies(graph_summary)
        assert len(result) > 0
        assert result[0]["to"] == "app/models/user.py"
        for dep in result:
            assert dep["reason"] == "high-import-count dependency (deterministic fallback)"

    def test_fallback_caps_at_five_edges(self):
        # Many edges all pointing at distinct targets — fallback caps result at 5.
        edges = [{"from": f"src/a{i}.py", "to": f"src/b{i}.py"} for i in range(10)]
        result = _build_fallback_key_dependencies({"internal_edges": edges})
        assert len(result) == 5

    def test_fallback_empty_when_no_internal_edges(self):
        assert _build_fallback_key_dependencies({"internal_edges": []}) == []
        assert _build_fallback_key_dependencies({}) == []

    def test_validate_strips_external_and_phantom_edges(self):
        # Without graph_summary the validator just filters — the fallback
        # responsibility now lives in interpret_architecture.
        interpretation = {
            "architecture_pattern": "MVC",
            "main_components": [],
            "key_dependencies": [
                {"from": "app/main.py", "to": "fastapi", "reason": "uses FastAPI"},
                {"from": "app/main.py", "to": "app/services/auth.py", "reason": "real"},
            ],
            "summary_for_new_developer": "...",
        }
        explored_paths = {"app/main.py", "app/services/auth.py"}
        result = _validate_interpretation(interpretation, explored_paths)
        assert len(result["key_dependencies"]) == 1
        assert result["key_dependencies"][0]["to"] == "app/services/auth.py"

    def test_interpret_architecture_returns_minimal_dict_on_ollama_failure(self, monkeypatch):
        # Force a timeout-like failure from _call_ollama; interpret_architecture
        # should swallow it and return a deterministic minimal dict rather than None.
        def boom(prompt):
            raise RuntimeError("timed out")
        monkeypatch.setattr(ai_interpreter, "_call_ollama", boom)

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            (repo / "a.py").write_text("from . import b\n")
            (repo / "b.py").write_text("")

            analysis_state = {
                "repo_id": "test",
                "current_summary": {"local_path": str(repo), "repo": "test"},
                "inspected_facts": [
                    {"file_path": "a.py", "language": "python", "role_hint": "module",
                     "imported_modules": [".b"]},
                ],
                "dependency_edges": [{"source": "a.py", "imports": [".b"]}],
                "package_roots": ["."],
            }

            result = interpret_architecture(analysis_state)

            # Never None now.
            assert isinstance(result, dict)
            assert result["architecture_pattern"] == "Not available"
            assert result["main_components"] == []
            assert result["confidence"] == 0.0
            # The deterministic fallback fires even when AI fails entirely.
            assert len(result["key_dependencies"]) >= 1
            for dep in result["key_dependencies"]:
                assert dep["reason"] == "high-import-count dependency (deterministic fallback)"


# ---------------------------------------------------------------------------
# Directory-aware exploration: _next_unexplored and _nudge_message
# ---------------------------------------------------------------------------

class TestNudgeAndNextUnexplored:
    def test_next_unexplored_prefers_unvisited_directory(self):
        # Same-dir candidate is listed first, but the unvisited-dir candidate
        # should still win because of the first-pass dir-aware preference.
        state = {
            "explored_files": ["src/components/Button.tsx"],
            "candidate_files": [
                {"file_path": "src/components/Card.tsx", "reason": "same dir"},
                {"file_path": "src/utils/format.ts",     "reason": "unvisited"},
            ],
            "_cached_files": [
                "src/components/Button.tsx",
                "src/components/Card.tsx",
                "src/utils/format.ts",
            ],
        }
        assert _next_unexplored(state) == "src/utils/format.ts"

    def test_next_unexplored_falls_back_when_all_dirs_visited(self):
        # Only candidate lives in the already-visited directory — first pass
        # finds nothing, second pass returns it rather than going to fallback.
        state = {
            "explored_files": ["src/components/Button.tsx"],
            "candidate_files": [
                {"file_path": "src/components/Card.tsx", "reason": "same dir"},
            ],
            "_cached_files": [
                "src/components/Button.tsx",
                "src/components/Card.tsx",
            ],
        }
        assert _next_unexplored(state) == "src/components/Card.tsx"

    def test_nudge_prefers_unvisited_directory_candidates(self):
        state = {
            "explored_files": ["src/components/Button.tsx"],
            "candidate_files": [
                {"file_path": "src/components/Card.tsx", "reason": "same dir"},
                {"file_path": "src/utils/format.ts",     "reason": "unvisited"},
            ],
            "dependency_edges": [],
        }
        result = _nudge_message(state)
        content = result["content"]
        # New "stayed in same directory" wording is used.
        assert "stayed in the same directory" in content
        # Unvisited-dir candidate is surfaced.
        assert "src/utils/format.ts" in content
        # Same-dir candidate is NOT listed when an unvisited one is available.
        assert "src/components/Card.tsx" not in content

    def test_nudge_falls_back_to_combined_list_when_no_unvisited_dirs(self):
        # All candidates live in the already-visited directory and there are no
        # import targets to surface — we expect the original generic wording.
        state = {
            "explored_files": ["src/components/Button.tsx"],
            "candidate_files": [
                {"file_path": "src/components/Card.tsx", "reason": "same dir"},
            ],
            "dependency_edges": [],
        }
        result = _nudge_message(state)
        content = result["content"]
        assert "stayed in the same directory" not in content
        # Falls back to the original generic prompt …
        assert "have not yet read these files" in content
        # … which surfaces the same-dir candidate as a last resort.
        assert "src/components/Card.tsx" in content


# ---------------------------------------------------------------------------
# analysis_state_store — delete_state and force_refresh semantics
# ---------------------------------------------------------------------------

class TestDeleteStateAndForceRefresh:
    def test_delete_state_removes_existing_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_dir = tmp_path / "cache"
            save_state(
                repo_id="myrepo",
                local_path=str(tmp_path),
                final_state={"k": "v"},
                cache_dir=cache_dir,
            )
            cache_file = cache_dir / "myrepo.json"
            assert cache_file.exists()

            delete_state(repo_id="myrepo", cache_dir=cache_dir)
            assert not cache_file.exists()

    def test_delete_state_missing_file_is_noop(self):
        # Should not raise, and should not create the file.
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            cache_dir.mkdir()
            delete_state(repo_id="never-existed", cache_dir=cache_dir)
            assert not (cache_dir / "never-existed.json").exists()

    def test_load_returns_none_after_delete_even_with_matching_hash(self, monkeypatch):
        # Pin the git hash so load_state's staleness check would normally
        # succeed — this isolates the delete behavior from staleness logic.
        monkeypatch.setattr(
            "app.services.analysis_state_store._get_git_commit_hash",
            lambda local_path: "abc123",
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_dir = tmp_path / "cache"
            save_state(
                repo_id="myrepo",
                local_path=str(tmp_path),
                final_state={"k": "v"},
                cache_dir=cache_dir,
            )

            # Sanity: with the git hash unchanged, load returns the cached state.
            assert load_state("myrepo", str(tmp_path), cache_dir) == {"k": "v"}

            delete_state(repo_id="myrepo", cache_dir=cache_dir)

            # After delete, load returns None — even though git HEAD didn't move.
            assert load_state("myrepo", str(tmp_path), cache_dir) is None


# ---------------------------------------------------------------------------
# _candidate_signal_score: unvisited-directory bonus and reduced chain pull
# ---------------------------------------------------------------------------

class TestCandidateSignalScore:
    @staticmethod
    def _state_with_explored(explored_files):
        """Minimal state with explored_files mirrored as inspected_facts."""
        return {
            "explored_files": list(explored_files),
            "inspected_facts": [
                {
                    "file_path": fp,
                    "language": "python",
                    "role_hint": "module",
                    "directory": str(Path(fp).parent),
                    "line_count_bucket": "small",
                    "imported_modules": [],
                }
                for fp in explored_files
            ],
            "unknowns": [],
            "current_summary": {
                "top_level_dirs": ["src"],
                "entry_points": [],
                "languages": ["python"],
            },
            "dependency_edges": [],
        }

    def test_unvisited_directory_reason_fires(self):
        state = self._state_with_explored(["src/foo.py"])
        file_languages = {"src/foo.py": "python", "src/utils/bar.py": "python"}

        score, reasons = _candidate_signal_score(
            state, "src/utils/bar.py", file_languages, known_targets=set()
        )

        assert "unvisited directory" in reasons
        # The pre-existing per-fact "new directory context" still fires alongside.
        assert "new directory context" in reasons

    def test_unvisited_directory_reason_does_not_fire_in_visited_dir(self):
        # The candidate lives in the same directory the agent has already been in.
        state = self._state_with_explored(["src/utils/foo.py"])
        file_languages = {"src/utils/foo.py": "python", "src/utils/bar.py": "python"}

        score, reasons = _candidate_signal_score(
            state, "src/utils/bar.py", file_languages, known_targets=set()
        )

        assert "unvisited directory" not in reasons

    def test_unvisited_directory_score_delta(self):
        # Score the same candidate against two states that differ only in
        # whether its parent directory has been explored. The score delta
        # equals the +3 (new directory context) + +4 (unvisited directory)
        # contributions for a total of +7.
        file_languages = {
            "src/foo.py": "python",
            "src/utils/bar.py": "python",
            "src/utils/baz.py": "python",
        }
        state_unvisited = self._state_with_explored(["src/foo.py"])
        state_visited = self._state_with_explored(["src/foo.py", "src/utils/baz.py"])

        score_unvisited, _ = _candidate_signal_score(
            state_unvisited, "src/utils/bar.py", file_languages, known_targets=set()
        )
        score_visited, _ = _candidate_signal_score(
            state_visited, "src/utils/bar.py", file_languages, known_targets=set()
        )

        assert score_unvisited - score_visited == 7

    def test_known_import_target_bonus_three_for_single_importer(self):
        # In-degree of 1 still earns +3 — same as the pre-Counter flat bonus.
        state = self._state_with_explored(["src/foo.py"])
        file_languages = {"src/foo.py": "python", "src/bar.py": "python"}

        score_off, _ = _candidate_signal_score(
            state, "src/bar.py", file_languages, known_targets=Counter()
        )
        score_on, reasons_on = _candidate_signal_score(
            state, "src/bar.py", file_languages,
            known_targets=Counter({"src/bar.py": 1}),
        )

        assert "known import target from 1 explored file(s)" in reasons_on
        assert score_on - score_off == 3

    def test_known_import_target_scales_with_in_degree(self):
        state = self._state_with_explored(["src/foo.py"])
        file_languages = {"src/foo.py": "python", "src/bar.py": "python"}

        score_1, _ = _candidate_signal_score(
            state, "src/bar.py", file_languages,
            known_targets=Counter({"src/bar.py": 1}),
        )
        score_2, _ = _candidate_signal_score(
            state, "src/bar.py", file_languages,
            known_targets=Counter({"src/bar.py": 2}),
        )
        score_3, _ = _candidate_signal_score(
            state, "src/bar.py", file_languages,
            known_targets=Counter({"src/bar.py": 3}),
        )

        # 1*3=3, 2*3=6, 3*3=9 → each step +3 below the cap.
        assert score_2 - score_1 == 3
        assert score_3 - score_2 == 3

    def test_known_import_target_capped_at_nine(self):
        state = self._state_with_explored(["src/foo.py"])
        file_languages = {"src/foo.py": "python", "src/bar.py": "python"}

        score_3, _ = _candidate_signal_score(
            state, "src/bar.py", file_languages,
            known_targets=Counter({"src/bar.py": 3}),
        )
        score_10, reasons_10 = _candidate_signal_score(
            state, "src/bar.py", file_languages,
            known_targets=Counter({"src/bar.py": 10}),
        )

        # 3 importers → +9; 10 importers → capped at +9 (no further reward).
        assert score_3 == score_10
        # Reason string still reflects the true in-degree even though
        # the score is capped — useful for debugging.
        assert "known import target from 10 explored file(s)" in reasons_10

    def test_unvisited_dir_outweighs_chain_candidate(self):
        # Agent explored one file in src/services. Compare:
        #   - chain candidate: another file in src/services, in known_targets
        #   - breadth candidate: a file in src/utils, NOT in known_targets
        # Breadth must outscore chain so the agent actually moves.
        file_languages = {
            "src/services/auth.py":   "python",
            "src/services/helper.py": "python",  # chain candidate
            "src/utils/format.py":    "python",  # breadth candidate
        }
        state = self._state_with_explored(["src/services/auth.py"])

        chain_score, chain_reasons = _candidate_signal_score(
            state, "src/services/helper.py", file_languages,
            known_targets=Counter({"src/services/helper.py": 1}),
        )
        breadth_score, breadth_reasons = _candidate_signal_score(
            state, "src/utils/format.py", file_languages,
            known_targets=Counter(),
        )

        assert breadth_score > chain_score, (
            f"breadth ({breadth_score}: {breadth_reasons}) "
            f"should outscore chain ({chain_score}: {chain_reasons})"
        )


# ---------------------------------------------------------------------------
# _detect_python_package_roots and __init__.py candidate penalty
# ---------------------------------------------------------------------------

class TestPythonPackageRootDetection:
    def test_returns_wrapper_for_nested_layout(self):
        # Real-world case (aria-companion): all packages live under backend/,
        # and backend/ itself is NOT a package. Root must be "backend" so the
        # absolute resolver knows where to look for "db.database".
        scanned = [
            "backend/main.py",
            "backend/config.py",
            "backend/db/__init__.py",
            "backend/db/database.py",
            "backend/models/__init__.py",
            "backend/models/user.py",
            "backend/services/__init__.py",
            "backend/routers/__init__.py",
        ]
        result = _detect_python_package_roots(Path("/fake"), scanned)
        assert result == ["backend"]

    def test_returns_dot_for_flat_layout(self):
        # Packages live directly at the repo root.
        scanned = [
            "db/__init__.py",
            "db/database.py",
            "models/__init__.py",
            "models/user.py",
        ]
        result = _detect_python_package_roots(Path("/fake"), scanned)
        assert result == ["."]

    def test_returns_dot_for_multiple_top_level_dirs(self):
        # Two different wrapper directories each contain packages — no single
        # wrapper to anchor; fall back to the repo root.
        scanned = [
            "backend/db/__init__.py",
            "backend/db/database.py",
            "frontend/utils/__init__.py",
            "frontend/utils/helpers.py",
        ]
        result = _detect_python_package_roots(Path("/fake"), scanned)
        assert result == ["."]

    def test_wrapper_that_is_itself_a_package_returns_dot(self):
        # If backend/__init__.py exists, "backend" is itself an importable
        # package — absolute imports like "backend.db.database" must resolve
        # from the repo root, so the root is "." (not "backend").
        scanned = [
            "backend/__init__.py",
            "backend/main.py",
            "backend/db/__init__.py",
            "backend/db/database.py",
        ]
        result = _detect_python_package_roots(Path("/fake"), scanned)
        assert result == ["."]

    def test_known_top_level_names_strips_py_extension(self):
        # `import config` from inside backend/ must be recognized — and the
        # name set has to read "config", not "config.py", for the check to
        # match the import specifier.
        scanned = {
            "backend/config.py",
            "backend/main.py",
            "backend/db/__init__.py",
            "backend/db/database.py",
        }
        names = _known_top_level_package_names([Path("backend")], scanned)
        assert "config" in names
        assert "config.py" not in names
        # Package directories still appear without extension (no change there).
        assert "db" in names

    def test_known_top_level_names_excludes_init(self):
        # "__init__" is never a valid import specifier; including it in the
        # names set would let `import __init__` (or similar invalid forms)
        # falsely pass the "should attempt absolute resolution" guard.
        scanned = {
            "backend/db/__init__.py",
            "backend/db/database.py",
        }
        names = _known_top_level_package_names([Path("backend")], scanned)
        assert "__init__" not in names
        assert "db" in names


# ---------------------------------------------------------------------------
# __init__.py candidate-score penalty
# ---------------------------------------------------------------------------

class TestInitPyPenalty:
    def test_init_py_scores_three_lower_than_equivalent_module(self):
        # Two candidates in the same (unvisited) directory, both Python,
        # both role "module". The only difference: one is __init__.py.
        # The penalty should be exactly -3.
        state = {
            "explored_files": ["src/foo.py"],
            "inspected_facts": [
                {
                    "file_path": "src/foo.py",
                    "language": "python",
                    "role_hint": "module",
                    "directory": "src",
                    "line_count_bucket": "small",
                    "imported_modules": [],
                }
            ],
            "unknowns": [],
            "current_summary": {
                "top_level_dirs": ["src"],
                "entry_points": [],
                "languages": ["python"],
            },
            "dependency_edges": [],
        }
        file_languages = {
            "src/foo.py": "python",
            "src/bar/__init__.py": "python",
            "src/bar/module.py": "python",
        }

        init_score, init_reasons = _candidate_signal_score(
            state, "src/bar/__init__.py", file_languages, known_targets=Counter()
        )
        module_score, module_reasons = _candidate_signal_score(
            state, "src/bar/module.py", file_languages, known_targets=Counter()
        )

        assert init_score == module_score - 3
        # Reason string surfaces the penalty for debuggability.
        assert any("__init__.py" in r for r in init_reasons)
        assert not any("__init__.py" in r for r in module_reasons)


# ---------------------------------------------------------------------------
# Per-language coverage gate in _refresh_candidates_for_signal
# ---------------------------------------------------------------------------

class TestLanguageCoverageGate:
    @staticmethod
    def _make_repo_with_mixed_files(tmp: str, py_count: int, js_count: int):
        repo = Path(tmp).resolve()
        for i in range(py_count):
            (repo / f"backend_{i}.py").write_text("")
        for i in range(js_count):
            (repo / f"frontend_{i}.js").write_text("")
        return repo

    @staticmethod
    def _make_state(repo: Path, explored_files):
        return {
            "current_summary": {
                "local_path": str(repo),
                "top_level_dirs": [],
                "entry_points": [],
                "languages": ["python", "javascript"],
            },
            "explored_files": list(explored_files),
            "candidate_files": [],
            "inspected_facts": [
                {
                    "file_path": fp,
                    "language": "python",
                    "role_hint": "module",
                    "directory": str(Path(fp).parent),
                    "line_count_bucket": "small",
                    "imported_modules": [],
                }
                for fp in explored_files
            ],
            "dependency_edges": [],
            "package_roots": [],
            "unknowns": [],
        }

    def test_low_dominant_coverage_filters_non_dominant_language(self):
        # 10 Python files, 5 JS. Dominant = Python. Zero explored → 0% coverage.
        # Gate should remove every JS candidate from the queue.
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_repo_with_mixed_files(tmp, py_count=10, js_count=5)
            state = self._make_state(repo, explored_files=[])

            _refresh_candidates_for_signal(state, limit=8)

            assert len(state["candidate_files"]) > 0, "should have at least some candidates"
            for c in state["candidate_files"]:
                assert c["file_path"].endswith(".py"), (
                    f"JS candidate slipped past coverage gate: {c['file_path']}"
                )

    def test_high_dominant_coverage_allows_non_dominant_language(self):
        # 10 Python files, 5 JS. Explore 6/10 Python (60% coverage ≥ 50%).
        # The gate disengages and JS candidates can appear.
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_repo_with_mixed_files(tmp, py_count=10, js_count=5)
            explored = [f"backend_{i}.py" for i in range(6)]
            state = self._make_state(repo, explored_files=explored)

            _refresh_candidates_for_signal(state, limit=8)

            extensions = {Path(c["file_path"]).suffix for c in state["candidate_files"]}
            assert ".js" in extensions, (
                f"JS candidates should appear once dominant coverage ≥ 0.5; "
                f"got extensions: {extensions}"
            )
