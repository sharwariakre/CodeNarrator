# CodeNarrator

**An autonomous codebase understanding agent.**

CodeNarrator ingests any Git repository and autonomously explores it — inspecting files, building a dependency graph, tracking its own understanding over time, and producing an AI-generated architectural summary. Paste a GitHub URL and get a full architecture report. No human guidance required between input and output.

Built with a React + TypeScript frontend, a FastAPI backend, and a Qwen2.5-Coder 7B model running via Ollama.

---

## What It Does

```
GitHub URL
  → Clone repo locally
  → Build initial understanding snapshot
  → Run agentic exploration loop (Ollama tool-calling)
  → Extract imports and build dependency graph
  → Resolve internal file-to-file edges
  → Call local AI model for architectural interpretation
  → Generate self-contained HTML report with D3 graph
  → Render report inline in the browser
```

Given a repo like `https://github.com/user/project`, CodeNarrator produces:

- Which files are architectural entry points
- How files depend on each other (resolved internal edges)
- Which files form clusters of related functionality
- An AI-generated breakdown of main components and their relationships
- A plain English summary a new developer can read to understand the codebase

---

## Example Report

![CodeNarrator dependency graph for Dead-Serious](docs/assets/dependency-mapping.png)

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     React Frontend (Vite)                        │
│                                                                  │
│  URL input → sequential pipeline steps with live status         │
│  → renders final HTML report inline in iframe                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP (Vite proxy → localhost:8000)
┌───────────────────────────▼─────────────────────────────────────┐
│                      FastAPI Backend                             │
│                                                                  │
│  POST /repos/ingest                                              │
│       ↓                                                          │
│  GitPython clone → local disk                                    │
│                                                                  │
│  POST /repos/snapshot                                            │
│       ↓                                                          │
│  repo_scanner + repo_metadata → initial analysis_state          │
│                                                                  │
│  POST /repos/snapshot/run                                        │
│       ↓                                                          │
│  Agentic Loop (Ollama tool-calling):                             │
│    model picks tool → read_file / follow_import /               │
│    search_for_pattern / mark_architecture_insight / stop        │
│    → tool result fed back → model reasons → repeat              │
│       ↓                                                          │
│  dependency_graph_summary (edges, clusters, rankings)           │
│                                                                  │
│  POST /repos/interpret                                           │
│       ↓                                                          │
│  Ollama (Qwen2.5-Coder 7B) → structured interpretation          │
│                                                                  │
│  POST /repos/report                                              │
│       ↓                                                          │
│  Self-contained HTML report with D3.js graph (D3 embedded)      │
│                                                                  │
│  GET  /report-file                                               │
│       ↓                                                          │
│  Serves generated HTML to the frontend iframe                   │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

#### Analysis State
The central memory object that flows through the entire system. Tracks everything the agent has observed and decided:

```python
{
  "repo_id": str,
  "explored_files": list[str],        # files already inspected
  "candidate_files": list[dict],      # files ranked for next inspection
  "inspected_facts": list[dict],      # what was learned from each file
  "dependency_edges": list[dict],     # raw import data per file
  "dependency_graph_summary": dict,   # resolved edges, clusters, rankings
  "package_roots": list[str],         # detected Python package roots
  "unknowns": list[str],              # explicitly unresolved questions
  "current_summary": dict,            # evolving repo understanding
  "confidence": float,                # evidence-based confidence 0.0–0.95
  "no_progress_steps": int,           # consecutive steps with no new signal
  "stop_reason": str | None           # why the loop stopped
}
```

#### The Agentic Loop

The loop is model-driven. An Ollama model receives the current analysis state and a running message history, then calls tools to decide what to explore next. Each step feeds tool results back into the history so the model reasons about what it has learned before deciding the next action.

**Tools available to the model:**

| Tool | Purpose |
|------|---------|
| `read_file` | Read a source file — returns language, role, imports, content preview |
| `follow_import` | Resolve and read a file imported by an already-explored file |
| `search_for_pattern` | Regex search across repo files |
| `mark_architecture_insight` | Record a discovered architectural insight |
| `stop_analysis` | Signal that exploration is complete |

**Loop controls (code-enforced, model cannot bypass):**
- `stop_analysis` is rejected until a minimum number of files have been explored (scales with repo size: `min(15, max(6, file_count × 0.65))`)
- After 2 consecutive steps with no new file explored, a nudge message lists unexplored files and forces the model to pick one
- After 2 consecutive nudges with no response, the next unexplored file is force-read automatically
- Ollama calls retry up to 2× on failure with backoff

#### Evidence-Based Confidence

Confidence increases only when the loop actually discovers something new:

| Evidence | Delta |
|----------|-------|
| New entry point discovered | +0.10 |
| New top-level directory signal | +0.05 |
| Unknown cleared | +0.03 per unknown (max +0.09) |
| Materially new inspected fact | +0.04 |

Capped at 0.95 — never 1.0 — because complete understanding from static analysis alone is not achievable.

#### Dependency Mapping

During each file inspection, imports are extracted and stored:

**Python** — AST parser first, regex fallback:
- `import X` and `from X import Y` forms
- Relative imports: `from . import X`, `from ..utils import Y`
- Absolute imports resolved to internal files using detected package roots

**JavaScript/JSX** — regex based:
- ES modules: `import X from 'Y'`
- CommonJS: `require('Y')`

**Internal edge resolution** — relative and absolute imports are resolved to actual repo files:
- `./pages/CreateVault` → tries `.js`, `.jsx`, `.ts`, `.tsx`, `/index.js` variants
- `from app.services.foo import X` → resolved against detected package roots
- Edge only created if target file exists in the scanned repo
- External libraries remain unresolved as external dependencies

#### AI Interpretation

After the loop completes, the final state is passed to a local Ollama model:

**Input to model:**
- `internal_edges` — resolved file-to-file connections
- `clusters` — groups of files sharing import patterns
- `highest_dependency_files` — most connected files
- `inspected_facts` — file paths, languages, role hints, imported modules

**Output from model:**
```json
{
  "architecture_pattern": "...",
  "main_components": [{"name": "...", "files": [...], "description": "..."}],
  "key_dependencies": [{"from": "...", "to": "...", "reason": "..."}],
  "summary_for_new_developer": "..."
}
```

File references in the AI output are validated against actually-explored files — phantom file mentions are stripped before the report is generated.

The AI layer is **optional and non-blocking**. If Ollama is unreachable or returns malformed JSON, interpretation returns `null` and the report is still generated with the deterministic analysis.

#### HTML Report

The report is a fully self-contained HTML file:
- D3.js is fetched once and **embedded inline** at generation time (no CDN dependency, works offline)
- Dependency graph with force-directed layout, color-coded clusters, hover tooltips
- Node size = incoming internal dependencies
- Grey dashed nodes = files imported but not explored
- AI component list, key dependencies, explored files table

---

## Key Design Decisions

**Why agentic exploration**

A model-driven loop can adapt to what it finds — if one file reveals something unexpected, the model can follow that thread. A hardcoded scoring heuristic always applies the same logic regardless of what it has discovered. The agentic approach lets the model reason about its own understanding and prioritize accordingly.

**Why code-enforced stop controls**

Leaving the stop decision entirely to the model results in premature termination — the model calls `stop_analysis` after 2-3 files because it believes it has enough context. Code-level guards that reject early stops and force-read unexplored files ensure meaningful coverage regardless of model behavior. The model still drives exploration; it just cannot quit early.

**Why evidence-based confidence instead of step-based**

Step-based confidence is meaningless — you could reach high confidence by inspecting ten identical files. Evidence-based confidence means the score reflects actual understanding gained.

**Why separate snapshot from loop execution**

Snapshot is one-time static observation. The loop is iterative reasoning. Keeping them separate means you can restart the loop from the same starting point, compare different loop strategies, and evaluate loop quality independently.

**Why local-first**

No cloud costs, no latency, no data privacy concerns. A developer can run CodeNarrator against a private proprietary codebase without any data leaving their machine.

**Why internal edge resolution matters**

Raw relative imports (`./pages/CreateVault`) stored as strings tell you nothing about component relationships. Resolved internal edges tell you exactly which files depend on which other files — the actual architectural wiring of the codebase.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Frontend | React + TypeScript + Vite | Fast dev server, type-safe pipeline orchestration, iframe report rendering |
| API framework | FastAPI | Native Pydantic integration, async support, automatic OpenAPI docs |
| Data validation | Pydantic v2 | Request/response validation at API boundary, clean model definitions |
| Repo cloning | GitPython | Programmatic Git operations, local-first cloning |
| Python AST parsing | `ast` (stdlib) | Zero-dependency, reliable import extraction for Python files |
| JS import extraction | Regex | No Node.js dependency needed for pattern-based extraction |
| AI model | Qwen2.5-Coder 7B | Code-aware, runs locally on 16GB RAM MacBook, strong structured output |
| Model serving | Ollama | Local model inference, simple API, no billing |
| Visualization | D3.js (embedded) | Force-directed graph, embedded inline so report works offline |
| Report format | Self-contained HTML | Opens in any browser, no server needed, shareable as a single file |

---

## API Reference

### `POST /api/v1/repos/ingest`
Clone a remote repository locally.

```json
// Request
{"repo_url": "https://github.com/user/repo", "force_clean": false}

// Response
{"repo_url": "...", "local_path": "data/repos/...", "status": "ready"}
```

### `POST /api/v1/repos/snapshot`
Build initial analysis state from an ingested repo.

```json
// Request
{"local_path": "data/repos/github.com__user__repo"}

// Response
{
  "repo_summary": {...},
  "next_candidates": [...],
  "unknowns": [...],
  "confidence": 0.62,
  "analysis_state": {...}
}
```

### `POST /api/v1/repos/snapshot/run`
Run the agentic analysis loop from an initial state.

```json
// Request
{"analysis_state": {...}, "max_steps": 20}

// Response
{
  "steps_executed": 12,
  "explored_files_in_order": [...],
  "step_trace": [...],
  "final_summary": {...},
  "final_confidence": 0.80,
  "remaining_unknowns": [],
  "stop_reason": "Agent decided analysis is complete.",
  "dependency_graph_summary": {...},
  "final_state": {...}
}
```

### `POST /api/v1/repos/interpret`
Call AI model to interpret the final analysis state.

```json
// Request
{"final_state": {...}}

// Response
{
  "interpretation": {
    "architecture_pattern": "Client-Server",
    "main_components": [...],
    "key_dependencies": [...],
    "summary_for_new_developer": "..."
  }
}
```

### `POST /api/v1/repos/report`
Generate a self-contained HTML report.

```json
// Request
{
  "final_state": {...},
  "interpretation": {...},
  "output_filename": "my-repo-report"
}

// Response
{"report_path": "/absolute/path/to/data/reports/my-repo-report.html"}
```

### `POST /api/v1/repos/state`
Look up a cached analysis result for a repo (matched by repo ID and git HEAD commit).

```json
// Request
{"repo_id": "github.com__user__repo", "local_path": "data/repos/..."}

// Response
{"repo_id": "...", "found": true, "final_state": {...}}
```

### `GET /report-file?path=...`
Serve a generated HTML report file by absolute path. Only serves files inside the `data/reports/` directory.

---

## Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git
- [Ollama](https://ollama.com) installed

### Installation

```bash
# Clone the repo
git clone https://github.com/sharwariakre/CodeNarrator
cd CodeNarrator

# Backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Pull the AI model
ollama pull qwen2.5-coder:7b

# Frontend
cd ../frontend
npm install
```

### Running

```bash
# Terminal 1 — start Ollama (if not already running)
ollama serve

# Terminal 2 — start the backend
cd backend
source venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 3 — start the frontend
cd frontend
npm run dev
```

Open `http://localhost:5173`, paste a GitHub URL, and click Analyze.

Backend API docs available at `http://127.0.0.1:8000/docs`

### Running via API directly

```bash
# 1. Ingest a repo
curl -X POST http://127.0.0.1:8000/api/v1/repos/ingest \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/user/repo", "force_clean": false}'

# 2. Get initial snapshot
curl -s -X POST http://127.0.0.1:8000/api/v1/repos/snapshot \
  -H "Content-Type: application/json" \
  -d '{"local_path": "data/repos/github.com__user__repo"}' > snapshot.json

# 3. Run agentic analysis loop
jq '{analysis_state: .analysis_state, max_steps: 20}' snapshot.json | \
  curl -s -X POST http://127.0.0.1:8000/api/v1/repos/snapshot/run \
  -H "Content-Type: application/json" -d @- > loop.json

# 4. Get AI interpretation
jq '{final_state: .final_state}' loop.json | \
  curl -s -X POST http://127.0.0.1:8000/api/v1/repos/interpret \
  -H "Content-Type: application/json" -d @- > interpret.json

# 5. Generate HTML report
jq -s '{final_state: .[0].final_state, interpretation: .[1].interpretation, output_filename: "my-report"}' \
  loop.json interpret.json | \
  curl -s -X POST http://127.0.0.1:8000/api/v1/repos/report \
  -H "Content-Type: application/json" -d @-

# 6. Open the report
open backend/data/reports/my-report.html
```

---

## Project Structure

```
CodeNarrator/
├── frontend/                         # React + TypeScript UI
│   ├── src/
│   │   ├── App.tsx                   # Pipeline orchestration + step UI
│   │   ├── api.ts                    # Typed fetch wrappers for all endpoints
│   │   └── App.css                   # Styles
│   └── vite.config.ts                # Dev proxy → backend
└── backend/
    ├── app/
    │   ├── api/
    │   │   └── v1/
    │   │       ├── routes_repo.py    # All repo analysis endpoints
    │   │       └── routes_health.py  # Health check
    │   ├── core/
    │   │   └── config.py             # Settings (repo base dir, cache dir, etc.)
    │   ├── models/
    │   │   └── repo_models.py        # Pydantic request/response models
    │   ├── services/
    │   │   ├── git_service.py                # Repo cloning via GitPython
    │   │   ├── repo_scanner.py               # File tree walking, language detection
    │   │   ├── repo_metadata.py              # Entry points, repo type, top-level dirs
    │   │   ├── analysis_snapshot_service.py  # Snapshot builder, heuristic scoring
    │   │   ├── agentic_analysis_service.py   # Agentic loop with Ollama tool-calling
    │   │   ├── analysis_state_store.py       # State persistence with git-HEAD staleness check
    │   │   ├── ai_interpreter.py             # Ollama interpretation + output validation
    │   │   └── report_generator.py           # HTML report generation (D3 embedded)
    │   └── main.py                   # FastAPI app, CORS, /report-file endpoint
    └── data/
        ├── repos/                    # Cloned repositories
        ├── analysis_cache/           # Persisted analysis state (JSON, keyed by repo + git HEAD)
        └── reports/                  # Generated HTML reports
```

---

## Supported Languages

| Language | Extensions |
|----------|-----------|
| Python | `.py` |
| JavaScript | `.js`, `.jsx` |
| TypeScript | `.ts`, `.tsx` |
| Java | `.java` |
| Go | `.go` |
| Rust | `.rs` |
| C/C++ | `.c`, `.cpp`, `.h` |
| Ruby | `.rb` |
| HTML/CSS | `.html`, `.css` |

---

## Known Limitations

- **Model quality cap** — Qwen2.5-Coder 7B running locally on CPU is slow (20-40s per tool call). The agentic loop takes 5-15 minutes on a fresh repo. Faster hardware or a larger/cloud model would significantly improve throughput.
- **Cluster detection** groups by import prefix heuristics. Repos using path aliases (e.g. `@components/Button`) will produce less meaningful clusters.
- **JavaScript AST** is not used — imports are extracted via regex. Dynamic imports and complex re-export patterns are not captured.
- **Report size** scales with repo size. For repos with 500+ files the embedded JSON in the HTML report may become large and the D3 graph may become slow.
- **Single run, no streaming** — the frontend blocks on the loop call for the full duration. No incremental progress is shown during the loop (only elapsed time).

---

## What's Next

- **Streaming progress** — stream step-by-step loop progress to the frontend via SSE so each file explored appears in real time
- **Minified file filtering** — skip `*.min.js` / `*.min.css` from the candidate list; they never add architectural signal
- **Richer structural facts** — function/class counts, detected patterns (DB access, API routes, config) to give AI more meaningful context
- **VS Code extension** — show the dependency graph and architectural summary inline while navigating a repo
- **Multi-language AST** — proper AST parsing for JavaScript/TypeScript to handle barrel files and complex re-exports

---

## Background

Built as part of exploring agentic AI systems for software engineering. The core idea: most developers spend significant time just orienting themselves in an unfamiliar codebase. CodeNarrator automates that orientation — not by having a human ask questions, but by having the system autonomously explore and reason about what it finds.

The architecture deliberately separates deterministic reasoning (what files exist, how they connect) from AI interpretation (what those connections mean). This keeps the system reliable, debuggable, and useful even when the AI layer is unavailable.

---

*Built with React, TypeScript, Vite, FastAPI, Pydantic, GitPython, D3.js, and Qwen2.5-Coder via Ollama.*
