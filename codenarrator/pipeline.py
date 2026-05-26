"""
End-to-end analysis pipeline.

Wires the existing service layer (clone → snapshot → agentic loop → AI
interpretation → HTML report) into a single in-process function call,
bypassing FastAPI entirely.

The signatures below mirror the underlying services *as they actually
are*, not as the public ``analyze()`` API pretends. ``analyze()`` is
the friendly facade.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

# Make the backend's ``app.*`` modules importable from this package without
# requiring an installed-mode layout. This must happen before the first
# ``from app...`` import below.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# Service-layer imports (after the sys.path insertion).
from app.core.config import settings  # noqa: E402
from app.services import agentic_analysis_service as _agentic_svc  # noqa: E402
from app.services import ai_interpreter as _interpreter_svc  # noqa: E402
from app.services.agentic_analysis_service import run_agentic_analysis_loop  # noqa: E402
from app.services.ai_interpreter import interpret_architecture  # noqa: E402
from app.services.analysis_snapshot_service import build_analysis_snapshot  # noqa: E402
from app.services.git_service import clone_or_update_repo  # noqa: E402
from app.services.report_generator import generate_html_report  # noqa: E402

from codenarrator.result import AnalysisResult


# Anchor data paths to the canonical backend/data/ tree so the CLI pipeline
# writes clones, cache, and reports into the same places the FastAPI flow
# does — both share the same backing store.
# (There is no REPORTS_DIR setting; the pipeline writes reports by defaulting
# analyze()'s output_dir to _BACKEND_ROOT / "data" / "reports".)
settings.REPO_BASE_DIR = _BACKEND_ROOT / "data" / "repos"
settings.ANALYSIS_CACHE_DIR = _BACKEND_ROOT / "data" / "analysis_cache"


# Map the friendly depth name to a step budget. Matches the UI's DEPTH_OPTIONS.
DEPTH_STEPS = {
    "standard": 20,
    "deep": 30,
}


def analyze(
    repo_url: str,
    depth: str = "standard",
    llm: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> AnalysisResult:
    """
    Clone a repository, run the full analysis pipeline in-process, and
    return an ``AnalysisResult`` containing the dependency graph, AI
    interpretation, and rendered HTML report.

    Parameters
    ----------
    repo_url:
        Any URL accepted by ``clone_or_update_repo`` (GitHub, GitLab,
        Bitbucket HTTPS or SSH).
    depth:
        ``"standard"`` (20 steps) or ``"deep"`` (30 steps).
    llm:
        Provider tag, stored on the result for future pluggable
        backends. Currently only ``"ollama"`` is wired.
    model:
        Override for ``settings.OLLAMA_MODEL`` (e.g. ``"llama3.1:8b"``).
        The override is applied for the duration of this call and
        restored on exit, even if the pipeline raises.
    api_key:
        Personal access token forwarded to ``clone_or_update_repo`` as
        ``git_token`` for private repo cloning.
    output_dir:
        Where to write the HTML report. Defaults to a fresh temp dir.

    Returns
    -------
    AnalysisResult
    """
    if depth not in DEPTH_STEPS:
        raise ValueError(
            f"depth must be one of {sorted(DEPTH_STEPS)}, got {depth!r}"
        )
    max_steps = DEPTH_STEPS[depth]

    # Step 1 — clone (or update if already present).
    local_path = clone_or_update_repo(
        repo_url=repo_url,
        force_clean=False,
        git_token=api_key,
    )

    # The service modules captured ``settings.OLLAMA_MODEL`` at import time
    # into their own module-level ``OLLAMA_MODEL`` constants — so to honour
    # ``model=...`` we have to patch the modules directly, not just settings.
    _orig_settings_model = settings.OLLAMA_MODEL
    _orig_agentic_model = _agentic_svc.OLLAMA_MODEL
    _orig_interpreter_model = _interpreter_svc.OLLAMA_MODEL

    if model:
        settings.OLLAMA_MODEL = model
        _agentic_svc.OLLAMA_MODEL = model
        _interpreter_svc.OLLAMA_MODEL = model

    try:
        # Step 2 — deterministic snapshot.
        snapshot = build_analysis_snapshot(local_path)
        initial_state = snapshot["analysis_state"]

        # Step 3 — agentic exploration loop.
        loop_out = run_agentic_analysis_loop(initial_state, max_steps=max_steps)
        final_state = loop_out["final_state"]

        # Step 4 — AI interpretation (always returns a dict; deterministic
        # fallback for key_dependencies fires inside on Ollama failure).
        interpretation = interpret_architecture(final_state)

        # Step 5 — render report. Default to the canonical backend/data/reports
        # directory so output lands alongside the FastAPI flow's reports.
        if output_dir:
            out_dir = Path(output_dir).expanduser().resolve()
        else:
            out_dir = _BACKEND_ROOT / "data" / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)

        report_path = out_dir / f"{_slug(Path(local_path).name)}-report.html"
        report_path = generate_html_report(final_state, interpretation, report_path)
        report_html = report_path.read_text(encoding="utf-8")
    finally:
        settings.OLLAMA_MODEL = _orig_settings_model
        _agentic_svc.OLLAMA_MODEL = _orig_agentic_model
        _interpreter_svc.OLLAMA_MODEL = _orig_interpreter_model

    return AnalysisResult(
        state=final_state,
        interpretation=interpretation,
        report_html=report_html,
        report_path=str(report_path),
        llm=llm,
    )


def _slug(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return cleaned or "report"
