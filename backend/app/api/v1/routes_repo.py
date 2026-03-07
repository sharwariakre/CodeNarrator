from pathlib import Path

from fastapi import APIRouter, HTTPException
from app.core.config import settings
from app.models import (
    AnalysisLoopRequest,
    AnalysisLoopResponse,
    IngestRepoRequest,
    IngestRepoResponse,
    RepoAnalysisSnapshotRequest,
    RepoAnalysisSnapshotResponse,
)
from app.services.git_service import clone_or_update_repo, GitCloneError
from app.services.analysis_snapshot_service import (
    build_analysis_snapshot,
    run_analysis_loop,
)

router = APIRouter(prefix="/repos", tags=["repos"])


@router.post("/ingest", response_model=IngestRepoResponse)
async def ingest_repo(payload: IngestRepoRequest):
    """
    Clone or update a repository and return its local path.
    This will later trigger parsing, embeddings, etc.
    """
    try:
        local_path = clone_or_update_repo(
            repo_url=payload.repo_url,
            force_clean=payload.force_clean
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

    if not requested_path.is_relative_to(repo_base_dir):
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
        snapshot = build_analysis_snapshot(requested_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return RepoAnalysisSnapshotResponse(**snapshot)


@router.post("/snapshot/run", response_model=AnalysisLoopResponse)
async def run_repo_snapshot_loop(payload: AnalysisLoopRequest):
    repo_base_dir = _resolve_repo_base_dir()
    local_path = payload.analysis_state.current_summary.local_path
    requested_path = _resolve_local_path(local_path)

    if not requested_path.is_relative_to(repo_base_dir):
        raise HTTPException(
            status_code=400,
            detail=f"analysis_state.current_summary.local_path must be inside {repo_base_dir}",
        )

    if not requested_path.exists() or not requested_path.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Repository path not found: {requested_path}",
        )

    loop_result = run_analysis_loop(
        initial_state=payload.analysis_state.model_dump(),
        max_steps=payload.max_steps,
    )
    return AnalysisLoopResponse(**loop_result)


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
