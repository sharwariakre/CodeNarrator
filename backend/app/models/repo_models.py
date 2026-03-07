from pydantic import BaseModel, HttpUrl


class IngestRepoRequest(BaseModel):
    repo_url: HttpUrl
    force_clean: bool = False


class IngestRepoResponse(BaseModel):
    repo_url: str
    local_path: str
    status: str


class RepoAnalysisSnapshotRequest(BaseModel):
    local_path: str


class SnapshotCandidate(BaseModel):
    file_path: str
    reason: str


class RepoSummary(BaseModel):
    repo: str
    local_path: str
    repo_type: str
    file_count: int
    languages: list[str]
    language_breakdown: dict[str, int]
    top_level_dirs: list[str]
    entry_points: list[str]


class AnalysisState(BaseModel):
    repo_id: str
    explored_files: list[str]
    candidate_files: list[SnapshotCandidate]
    unknowns: list[str]
    current_summary: RepoSummary
    confidence: float
    stop_reason: str | None = None


class RepoAnalysisSnapshotResponse(BaseModel):
    repo_summary: RepoSummary
    next_candidates: list[SnapshotCandidate]
    unknowns: list[str]
    confidence: float
    analysis_state: AnalysisState
