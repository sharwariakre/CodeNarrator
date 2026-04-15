import asyncio
import json
from pathlib import Path
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.core.config import settings
from app.models import (
    AnalysisLoopRequest,
    AnalysisLoopResponse,
    CachedStateRequest,
    CachedStateResponse,
    GenerateReportRequest,
    GenerateReportResponse,
    IngestRepoRequest,
    IngestRepoResponse,
    InterpretArchitectureRequest,
    InterpretArchitectureResponse,
    RepoAnalysisSnapshotRequest,
    RepoAnalysisSnapshotResponse,
)
from app.services.git_service import clone_or_update_repo, GitCloneError
from app.services.analysis_snapshot_service import (
    build_analysis_snapshot,
    run_analysis_loop,  # kept but not called — heuristic fallback
)
from app.services.agentic_analysis_service import run_agentic_analysis_loop
from app.services.ai_interpreter import interpret_architecture
from app.services.report_generator import generate_html_report
from app.services.analysis_state_store import save_state, load_state

router = APIRouter(prefix="/repos", tags=["repos"])


@router.post("/ingest", response_model=IngestRepoResponse)
async def ingest_repo(payload: IngestRepoRequest):
    """
    Clone or update a repository and return its local path.
    This will later trigger parsing, embeddings, etc.
    """
    try:
        local_path = await asyncio.to_thread(
            clone_or_update_repo,
            repo_url=payload.repo_url,
            force_clean=payload.force_clean,
            git_token=payload.git_token,
        )
    except GitCloneError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return IngestRepoResponse(
        repo_url=str(payload.repo_url),
        local_path=str(local_path),
        status="ready"
    )


@router.post("/snapshot", response_model=RepoAnalysisSnapshotResponse)
async def get_repo_snapshot(payload: RepoAnalysisSnapshotRequest):
    requested_path = _resolve_local_path(payload.local_path)
    repo_base_dir = _resolve_repo_base_dir()

    try:
        requested_path.relative_to(repo_base_dir)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"local_path must be inside {repo_base_dir}",
        )

    if not requested_path.exists() or not requested_path.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Repository path not found: {requested_path}",
        )

    try:
        snapshot = await asyncio.to_thread(build_analysis_snapshot, requested_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return RepoAnalysisSnapshotResponse(**snapshot)


@router.post("/snapshot/run", response_model=AnalysisLoopResponse)
async def run_repo_snapshot_loop(payload: AnalysisLoopRequest):
    repo_base_dir = _resolve_repo_base_dir()
    local_path = payload.analysis_state.current_summary.local_path
    requested_path = _resolve_local_path(local_path)

    try:
        requested_path.relative_to(repo_base_dir)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"analysis_state.current_summary.local_path must be inside {repo_base_dir}",
        )

    if not requested_path.exists() or not requested_path.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Repository path not found: {requested_path}",
        )

    initial_state = payload.analysis_state.model_dump()
    loop_result = await asyncio.to_thread(
        run_agentic_analysis_loop, initial_state, payload.max_steps
    )
    final_state = loop_result["final_state"]
    save_state(
        repo_id=final_state["repo_id"],
        local_path=final_state["current_summary"]["local_path"],
        final_state=final_state,
        cache_dir=_resolve_cache_dir(),
    )
    return AnalysisLoopResponse(**loop_result)


@router.post("/snapshot/run/stream")
async def stream_repo_snapshot_loop(payload: AnalysisLoopRequest):
    """
    Same as /snapshot/run but streams Server-Sent Events during the loop.
    Each event is a JSON line prefixed with 'data: '.
    Intermediate events: {"type": "progress", "file": "...", "step": n, "explored": n, "confidence": x}
    Final event:        {"type": "done", ...AnalysisLoopResponse fields...}
    """
    repo_base_dir = _resolve_repo_base_dir()
    local_path = payload.analysis_state.current_summary.local_path
    requested_path = _resolve_local_path(local_path)

    try:
        requested_path.relative_to(repo_base_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"local_path must be inside {repo_base_dir}")

    if not requested_path.exists() or not requested_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Repository path not found: {requested_path}")

    initial_state = payload.analysis_state.model_dump()
    event_loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_progress(event: dict):
        event_loop.call_soon_threadsafe(queue.put_nowait, event)

    async def generate():
        task = asyncio.create_task(
            asyncio.to_thread(run_agentic_analysis_loop, initial_state, payload.max_steps, on_progress)
        )
        # Drain progress events while the loop runs.
        while not task.done():
            try:
                event = await asyncio.wait_for(asyncio.shield(queue.get()), timeout=0.2)
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                continue

        # Drain any remaining events after the task finishes.
        while not queue.empty():
            event = queue.get_nowait()
            yield f"data: {json.dumps(event)}\n\n"

        loop_result = task.result()
        final_state = loop_result["final_state"]
        save_state(
            repo_id=final_state["repo_id"],
            local_path=final_state["current_summary"]["local_path"],
            final_state=final_state,
            cache_dir=_resolve_cache_dir(),
        )
        done_event = {**AnalysisLoopResponse(**loop_result).model_dump(), "type": "done"}
        yield f"data: {json.dumps(done_event)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/state", response_model=CachedStateResponse)
async def get_cached_state(payload: CachedStateRequest):
    """Return persisted analysis state if it exists and matches current git HEAD."""
    cached = load_state(
        repo_id=payload.repo_id,
        local_path=payload.local_path,
        cache_dir=_resolve_cache_dir(),
    )
    if cached is None:
        return CachedStateResponse(repo_id=payload.repo_id, found=False)
    return CachedStateResponse(repo_id=payload.repo_id, found=True, final_state=cached)


@router.post("/interpret", response_model=InterpretArchitectureResponse)
async def interpret_repo_architecture(payload: InterpretArchitectureRequest):
    interpretation = await asyncio.to_thread(
        interpret_architecture, payload.final_state.model_dump()
    )
    return InterpretArchitectureResponse(interpretation=interpretation)


@router.post("/report", response_model=GenerateReportResponse)
async def generate_repo_report(payload: GenerateReportRequest):
    reports_dir = _resolve_reports_dir()
    safe_name = _sanitize_output_filename(payload.output_filename)
    output_path = reports_dir / safe_name

    saved_path = await asyncio.to_thread(
        generate_html_report,
        final_state=payload.final_state.model_dump(),
        interpretation=payload.interpretation,
        output_path=output_path,
    )
    return GenerateReportResponse(report_path=str(saved_path))


def _resolve_local_path(local_path: str) -> Path:
    requested_path = Path(local_path).expanduser()
    if not requested_path.is_absolute():
        return (Path.cwd() / requested_path).resolve()
    return requested_path.resolve()


def _resolve_repo_base_dir() -> Path:
    repo_base_dir = settings.REPO_BASE_DIR
    if not repo_base_dir.is_absolute():
        return (Path.cwd() / repo_base_dir).resolve()
    return repo_base_dir.resolve()


def _resolve_cache_dir() -> Path:
    cache_dir = settings.ANALYSIS_CACHE_DIR
    if not cache_dir.is_absolute():
        return (Path.cwd() / cache_dir).resolve()
    return cache_dir.resolve()


def _resolve_reports_dir() -> Path:
    repo_base = _resolve_repo_base_dir()
    return (repo_base.parent / "reports").resolve()


def _sanitize_output_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-.")
    if not cleaned:
        cleaned = "architecture-report"
    if not cleaned.endswith(".html"):
        cleaned = f"{cleaned}.html"
    return cleaned
