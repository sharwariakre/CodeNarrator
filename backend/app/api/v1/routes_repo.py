from fastapi import APIRouter, HTTPException
from pathlib import Path

from app.models import IngestRepoRequest, IngestRepoResponse
from app.services.git_service import clone_or_update_repo, GitCloneError
from app.core.config import settings
from app.services.ast.ast_service import extract_ast


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

@router.get("/{repo_slug}/ast/python")
def get_python_ast(repo_slug: str):
    repo_path = Path(settings.REPO_BASE_DIR) / repo_slug

    if not repo_path.exists():
        raise HTTPException(status_code=404, detail="Repository not found")

    try:
        result = extract_ast(repo_path=repo_path, language="python")
        if result is None:
            raise HTTPException(
                status_code=400,
                detail="AST extraction not supported for Python",
            )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
