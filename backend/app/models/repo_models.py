from pydantic import BaseModel, Field, HttpUrl


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
    inspected_languages: list[str] = Field(default_factory=list)
    inspected_role_hints: list[str] = Field(default_factory=list)


class InspectedFileFact(BaseModel):
    file_path: str
    language: str
    line_count_bucket: str
    directory: str
    role_hint: str
    imports_found: int = 0
    imported_modules: list[str] = Field(default_factory=list)


class DependencyEdge(BaseModel):
    source: str
    imports: list[str] = Field(default_factory=list)


class AnalysisState(BaseModel):
    repo_id: str
    explored_files: list[str]
    candidate_files: list[SnapshotCandidate]
    inspected_facts: list[InspectedFileFact] = Field(default_factory=list)
    dependency_edges: list[DependencyEdge] = Field(default_factory=list)
    unknowns: list[str]
    current_summary: RepoSummary
    confidence: float
    no_progress_steps: int = 0
    stop_reason: str | None = None


class RepoAnalysisSnapshotResponse(BaseModel):
    repo_summary: RepoSummary
    next_candidates: list[SnapshotCandidate]
    unknowns: list[str]
    confidence: float
    analysis_state: AnalysisState


class AnalysisLoopRequest(BaseModel):
    analysis_state: AnalysisState
    max_steps: int = 5


class AnalysisStepTrace(BaseModel):
    step: int
    explored_file: str | None
    confidence: float
    remaining_candidates: int
    stop_reason: str | None


class AnalysisLoopResponse(BaseModel):
    steps_executed: int
    explored_files_in_order: list[str]
    step_trace: list[AnalysisStepTrace]
    final_summary: RepoSummary
    final_confidence: float
    remaining_unknowns: list[str]
    stop_reason: str | None
    dependency_graph_summary: dict
    final_state: AnalysisState


class InterpretArchitectureRequest(BaseModel):
    final_state: AnalysisState


class InterpretArchitectureResponse(BaseModel):
    interpretation: dict | None
