import { useState, useEffect, useRef } from "react";
import {
  ingestRepo,
  getSnapshot,
  getCachedState,
  runLoop,
  interpretArchitecture,
  generateReport,
  fetchReportHtml,
} from "./api";
import type { AnalysisState } from "./api";
import "./App.css";

type StepStatus = "idle" | "running" | "done" | "skipped" | "error";

interface Step {
  id: string;
  label: string;
  status: StepStatus;
  detail?: string;
}

function makeSteps(): Step[] {
  return [
    { id: "ingest",    label: "Clone / update repository",    status: "idle" },
    { id: "snapshot",  label: "Build initial snapshot",        status: "idle" },
    { id: "cache",     label: "Check analysis cache",          status: "idle" },
    { id: "loop",      label: "Run analysis loop",             status: "idle" },
    { id: "interpret", label: "AI architecture interpretation", status: "idle" },
    { id: "report",    label: "Generate HTML report",          status: "idle" },
  ];
}

const ICONS: Record<StepStatus, string> = {
  idle:    "○",
  running: "◌",
  done:    "✓",
  skipped: "⤼",
  error:   "✗",
};

function useElapsedSeconds(active: boolean) {
  const [seconds, setSeconds] = useState(0);
  const ref = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    if (active) {
      setSeconds(0);
      ref.current = setInterval(() => setSeconds((s) => s + 1), 1000);
    } else {
      if (ref.current) clearInterval(ref.current);
    }
    return () => { if (ref.current) clearInterval(ref.current); };
  }, [active]);
  return seconds;
}

export default function App() {
  const [url, setUrl]               = useState("");
  const [steps, setSteps]           = useState<Step[]>(makeSteps());
  const [running, setRunning]       = useState(false);
  const [loopRunning, setLoopRunning] = useState(false);
  const [reportHtml, setReportHtml] = useState<string | null>(null);
  const [globalError, setGlobalError] = useState<string | null>(null);

  const loopElapsed = useElapsedSeconds(loopRunning);

  function updateStep(id: string, patch: Partial<Step>) {
    setSteps((prev) => prev.map((s) => (s.id === id ? { ...s, ...patch } : s)));
  }

  async function handleAnalyze() {
    if (!url.trim()) return;
    setRunning(true);
    setLoopRunning(false);
    setReportHtml(null);
    setGlobalError(null);
    setSteps(makeSteps());

    try {
      // 1. Ingest
      updateStep("ingest", { status: "running" });
      const ingested = await ingestRepo(url.trim());
      updateStep("ingest", { status: "done", detail: ingested.local_path });
      const localPath = ingested.local_path;

      // 2. Snapshot (needed to get repo_id and initial state)
      updateStep("snapshot", { status: "running" });
      const snapshot = await getSnapshot(localPath);
      updateStep("snapshot", {
        status: "done",
        detail: `${snapshot.repo_summary.file_count} files · ${snapshot.repo_summary.languages.join(", ")}`,
      });

      // 3. Cache check
      updateStep("cache", { status: "running" });
      const cached = await getCachedState(snapshot.analysis_state.repo_id, localPath);

      let finalState: AnalysisState;

      if (cached.found && cached.final_state) {
        const s = cached.final_state;
        updateStep("cache", {
          status: "done",
          detail: `Cache hit · ${s.explored_files.length} files · confidence ${(s.confidence * 100).toFixed(0)}%`,
        });
        updateStep("loop", { status: "skipped", detail: "Using cached analysis" });
        finalState = s;
      } else {
        updateStep("cache", { status: "done", detail: "No cache — running fresh analysis" });

        // 4. Analysis loop
        updateStep("loop", { status: "running" });
        setLoopRunning(true);
        const loopResult = await runLoop(snapshot.analysis_state, 15);
        setLoopRunning(false);
        updateStep("loop", {
          status: "done",
          detail: `${loopResult.steps_executed} steps · ${loopResult.explored_files_in_order.length} files · confidence ${(loopResult.final_confidence * 100).toFixed(0)}%`,
        });
        finalState = loopResult.final_state;
      }

      // 5. Interpret
      updateStep("interpret", { status: "running" });
      const interpreted = await interpretArchitecture(finalState);
      const components = (interpreted.interpretation as { main_components?: unknown[] } | null)
        ?.main_components?.length ?? 0;
      updateStep("interpret", {
        status: "done",
        detail: interpreted.interpretation ? `${components} components` : "No interpretation returned",
      });

      // 6. Report
      updateStep("report", { status: "running" });
      const repoSlug = url
        .replace(/https?:\/\/github\.com\//i, "")
        .replace(/[^a-zA-Z0-9-]/g, "-")
        .replace(/-+/g, "-")
        .slice(0, 60);
      const reportResp = await generateReport(
        finalState,
        interpreted.interpretation as Record<string, unknown> | null,
        `${repoSlug}-report.html`,
      );
      updateStep("report", { status: "done", detail: reportResp.report_path });

      const html = await fetchReportHtml(reportResp.report_path);
      setReportHtml(html);
    } catch (err: unknown) {
      setLoopRunning(false);
      const msg = err instanceof Error ? err.message : String(err);
      setGlobalError(msg);
      setSteps((prev) =>
        prev.map((s) => (s.status === "running" ? { ...s, status: "error" } : s))
      );
    } finally {
      setRunning(false);
    }
  }

  const anyStepStarted = steps.some((s) => s.status !== "idle");

  return (
    <div className="app">
      <header>
        <h1>CodeNarrator</h1>
        <p className="subtitle">Paste a GitHub URL — get an architecture report</p>
      </header>

      <div className="input-row">
        <input
          className="url-input"
          type="text"
          placeholder="https://github.com/owner/repo"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !running) handleAnalyze(); }}
          disabled={running}
        />
        <button
          className="analyze-btn"
          onClick={handleAnalyze}
          disabled={running || !url.trim()}
        >
          {running ? "Analyzing…" : "Analyze"}
        </button>
      </div>

      {anyStepStarted && (
        <div className="steps">
          {steps.map((step) => {
            const isLoop = step.id === "loop";
            const detail =
              isLoop && step.status === "running"
                ? `Running heuristic analysis… ${loopElapsed}s elapsed`
                : step.detail;
            return (
              <div key={step.id} className={`step step-${step.status}`}>
                <span className="step-icon">{ICONS[step.status]}</span>
                <span className="step-label">{step.label}</span>
                {detail && <span className="step-detail">{detail}</span>}
              </div>
            );
          })}
        </div>
      )}

      {globalError && (
        <div className="error-box">
          <strong>Error:</strong> {globalError}
        </div>
      )}

      {reportHtml && (
        <div className="report-wrapper">
          <div className="report-header">Architecture Report</div>
          <iframe
            className="report-frame"
            srcDoc={reportHtml}
            title="Architecture Report"
            sandbox="allow-scripts allow-same-origin"
          />
        </div>
      )}
    </div>
  );
}
