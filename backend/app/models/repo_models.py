from pydantic import BaseModel, HttpUrl
from pathlib import Path


class IngestRepoRequest(BaseModel):
    repo_url: HttpUrl
    force_clean: bool = False


class IngestRepoResponse(BaseModel):
    repo_url: str
    local_path: str
    status: str
