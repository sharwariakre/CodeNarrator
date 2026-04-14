const BASE = "/api/v1/repos";

export interface IngestResponse {
  repo_url: string;
  local_path: string;
  status: string;
}

export interface RepoSummary {
  repo: string;
  local_path: string;
  repo_type: string;
  file_count: number;
  languages: string[];
  entry_points: string[];
}

export interface SnapshotCandidate {
  file_path: string;
  reason: string;
}

export interface AnalysisState {
  repo_id: string;
  explored_files: string[];
  candidate_files: SnapshotCandidate[];
  inspected_facts: unknown[];
  dependency_edges: unknown[];
  dependency_graph_summary: Record<string, unknown>;
  package_roots: string[];
  unknowns: string[];
  current_summary: RepoSummary;
  confidence: number;
  no_progress_steps: number;
  stop_reason: string | null;
}

export interface SnapshotResponse {
  repo_summary: RepoSummary;
  next_candidates: SnapshotCandidate[];
  unknowns: string[];
  confidence: number;
  analysis_state: AnalysisState;
}

export interface LoopResponse {
  steps_executed: number;
  explored_files_in_order: string[];
  final_summary: RepoSummary;
  final_confidence: number;
  remaining_unknowns: string[];
  stop_reason: string | null;
  dependency_graph_summary: Record<string, unknown>;
  final_state: AnalysisState;
}

export interface InterpretResponse {
  interpretation: Record<string, unknown> | null;
}

export interface ReportResponse {
  report_path: string;
}

export async function ingestRepo(repoUrl: string, forceClean = false): Promise<IngestResponse> {
  const res = await fetch(`${BASE}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo_url: repoUrl, force_clean: forceClean }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  return res.json();
}

export async function getSnapshot(localPath: string): Promise<SnapshotResponse> {
  const res = await fetch(`${BASE}/snapshot`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ local_path: localPath }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  return res.json();
}

export async function runLoop(state: AnalysisState, maxSteps = 15): Promise<LoopResponse> {
  const res = await fetch(`${BASE}/snapshot/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ analysis_state: state, max_steps: maxSteps }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  return res.json();
}

export interface ProgressEvent {
  type: "progress";
  file: string;
  step: number;
  explored: number;
  confidence: number;
}

export async function runLoopStream(
  state: AnalysisState,
  maxSteps: number,
  onProgress: (event: ProgressEvent) => void,
): Promise<LoopResponse> {
  const res = await fetch(`${BASE}/snapshot/run/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ analysis_state: state, max_steps: maxSteps }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = JSON.parse(line.slice(6));
      if (payload.type === "done") return payload as LoopResponse;
      if (payload.type === "progress") onProgress(payload as ProgressEvent);
    }
  }
  throw new Error("Stream ended without a done event");
}

export async function interpretArchitecture(finalState: AnalysisState): Promise<InterpretResponse> {
  const res = await fetch(`${BASE}/interpret`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ final_state: finalState }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  return res.json();
}

export async function generateReport(
  finalState: AnalysisState,
  interpretation: Record<string, unknown> | null,
  outputFilename: string
): Promise<ReportResponse> {
  const res = await fetch(`${BASE}/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ final_state: finalState, interpretation, output_filename: outputFilename }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  return res.json();
}

export interface CachedStateResponse {
  repo_id: string;
  found: boolean;
  final_state: AnalysisState | null;
}

export async function getCachedState(repoId: string, localPath: string): Promise<CachedStateResponse> {
  const res = await fetch(`${BASE}/state`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo_id: repoId, local_path: localPath }),
  });
  if (!res.ok) return { repo_id: repoId, found: false, final_state: null };
  return res.json();
}

export async function fetchReportHtml(reportPath: string): Promise<string> {
  // The backend serves files from data/reports via /reports static mount (to be added)
  // For now, use the /report-file endpoint
  const res = await fetch(`/report-file?path=${encodeURIComponent(reportPath)}`);
  if (!res.ok) throw new Error("Could not fetch report HTML");
  return res.text();
}
