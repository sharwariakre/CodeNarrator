from fastapi import APIRouter, HTTPException
from app.models import IngestRepoRequest, IngestRepoResponse
from app.services.git_service import clone_or_update_repo, GitCloneError

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
