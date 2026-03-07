from pathlib import Path

from fastapi import APIRouter, HTTPException
from app.core.config import settings
from app.models import (
    IngestRepoRequest,
    IngestRepoResponse,
    RepoAnalysisSnapshotRequest,
    RepoAnalysisSnapshotResponse,
)
from app.services.git_service import clone_or_update_repo, GitCloneError
from app.services.analysis_snapshot_service import build_analysis_snapshot

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
    requested_path = Path(payload.local_path).expanduser()
    if not requested_path.is_absolute():
        requested_path = (Path.cwd() / requested_path).resolve()
    else:
        requested_path = requested_path.resolve()

    repo_base_dir = settings.REPO_BASE_DIR
    if not repo_base_dir.is_absolute():
        repo_base_dir = (Path.cwd() / repo_base_dir).resolve()
    else:
        repo_base_dir = repo_base_dir.resolve()

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
