import { useState, useEffect, useRef } from "react";
import {
  ingestRepo,
  getSnapshot,
  runLoop,
  interpretArchitecture,
  generateReport,
  fetchReportHtml,
} from "./api";
import type { AnalysisState } from "./api";
import "./App.css";

type StepStatus = "idle" | "running" | "done" | "error";

interface Step {
  id: string;
  label: string;
  status: StepStatus;
  detail?: string;
}

function makeSteps(): Step[] {
  return [
    { id: "ingest",    label: "Clone / update repository",     status: "idle" },
    { id: "snapshot",  label: "Build initial snapshot",         status: "idle" },
    { id: "loop",      label: "Run analysis loop",              status: "idle" },
    { id: "interpret", label: "AI architecture interpretation", status: "idle" },
    { id: "report",    label: "Generate HTML report",           status: "idle" },
  ];
}

const ICONS: Record<StepStatus, string> = {
  idle:    "○",
  running: "◌",
  done:    "✓",
  error:   "✗",
};

function ElapsedTimer({ prefix }: { prefix: string }) {
  const [seconds, setSeconds] = useState(0);
  const ref = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    setSeconds(0);
    ref.current = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => { if (ref.current) clearInterval(ref.current); };
  }, []);
  return <span className="step-detail">{prefix}{seconds}s elapsed</span>;
}

const DEPTH_OPTIONS = [
  { label: "Quick",    steps: 10, hint: "~3-5 min" },
  { label: "Standard", steps: 20, hint: "~8-10 min" },
  { label: "Deep",     steps: 30, hint: "~15 min" },
];

export default function App() {
  const [url, setUrl]               = useState("");
  const [steps, setSteps]           = useState<Step[]>(makeSteps());
  const [running, setRunning]       = useState(false);
  const [reportHtml, setReportHtml] = useState<string | null>(null);
  const [globalError, setGlobalError] = useState<string | null>(null);
  const [depthIdx, setDepthIdx]     = useState(1); // default: Standard

  function updateStep(id: string, patch: Partial<Step>) {
    setSteps((prev) => prev.map((s) => (s.id === id ? { ...s, ...patch } : s)));
  }

  async function handleAnalyze() {
    if (!url.trim()) return;
    setRunning(true);
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

      // 3. Analysis loop
      updateStep("loop", { status: "running" });
      const loopResult = await runLoop(snapshot.analysis_state, DEPTH_OPTIONS[depthIdx].steps);
      const finalState: AnalysisState = loopResult.final_state;
      updateStep("loop", {
        status: "done",
        detail: `${loopResult.steps_executed} steps · ${loopResult.explored_files_in_order.length} files · confidence ${(loopResult.final_confidence * 100).toFixed(0)}%`,
      });

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

      <div className="depth-row">
        {DEPTH_OPTIONS.map((opt, i) => (
          <button
            key={opt.label}
            className={`depth-btn${depthIdx === i ? " depth-btn-active" : ""}`}
            onClick={() => setDepthIdx(i)}
            disabled={running}
            title={opt.hint}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {anyStepStarted && (
        <div className="steps">
          {steps.map((step) => {
            const isLoopRunning = step.id === "loop" && step.status === "running";
            return (
              <div key={step.id} className={`step step-${step.status}`}>
                <span className="step-icon">{ICONS[step.status]}</span>
                <span className="step-label">{step.label}</span>
                {isLoopRunning
                  ? <ElapsedTimer prefix="Running… " />
                  : step.detail && <span className="step-detail">{step.detail}</span>
                }
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
