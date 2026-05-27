# CodeNarrator

> Autonomous codebase understanding agent.

Give it a Git URL and it produces an interactive HTML dependency graph plus a natural-language architecture summary of the codebase. Everything runs locally via Ollama — no API keys, no cloud calls.

## Prerequisites

- Python **3.10+**
- [Ollama](https://ollama.ai) installed and running
- A model pulled: `ollama pull qwen2.5-coder:7b`

## Installation

```bash
pip install codenarrator-ai
```

## Usage

```python
from codenarrator import analyze

result = analyze("https://github.com/user/repo")

result.show()                  # opens the HTML report in your browser
result.to_html("report.html")  # save the HTML to a file
result.to_json("graph.json")   # export the dependency graph as JSON
```

Deeper exploration (30 steps instead of the default 20):

```python
result = analyze("https://github.com/user/repo", depth="deep")
```

Private repositories:

```python
result = analyze(
    "https://github.com/user/private-repo",
    api_key="your-github-token",
)
```

## Result object

| Property / Method | Description |
| --- | --- |
| `result.show()` | Open the HTML report in your browser |
| `result.to_html(path)` | Save the HTML report to a file |
| `result.to_json(path)` | Export dependency graph as JSON |
| `result.dependency_graph` | List of internal edges |
| `result.explored_files` | Files the agent explored |
| `result.architecture_summary` | AI-generated architecture summary |

## Configuration

Environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `CODENARRATOR_DATA_DIR` | Where clones, cache, and reports go | `~/codenarrator/` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Model to use | `qwen2.5-coder:7b` |

## Output

Reports and clones live under `~/codenarrator/` by default:

```
~/codenarrator/
├── repos/      ← cloned repositories
├── cache/      ← analysis cache (keyed by repo + git HEAD)
└── reports/    ← generated HTML reports
```

## Supported languages

| Language | Full dependency graph | Import extraction |
| --- | :---: | :---: |
| Python | ✓ | ✓ |
| TypeScript | ✓ | ✓ |
| JavaScript | ✓ | ✓ |
| Java | — | ✓ |
| Go | — | ✓ |
| Rust | — | ✓ |
| C / C++ | — | ✓ |

## How it works

Two layers:

**Deterministic layer** — file scanning, import extraction, internal-edge resolution, dependency graph computation. Pure Python, no LLM involved. Always produces a complete graph from static analysis alone.

**Agentic layer** — a local LLM (Qwen2.5-Coder 7B by default) explores the codebase via five tool calls:

- `read_file` — inspect a source file
- `follow_import` — jump to a file imported by something already read
- `search_for_pattern` — regex search across the repo
- `mark_architecture_insight` — record a finding
- `stop_analysis` — finish the run

The model produces a natural-language architecture summary alongside the deterministic graph. If the LLM times out or returns junk, the report still renders from the deterministic layer.

## Roadmap

- Pluggable LLM backends (OpenAI, Anthropic, any OpenAI-compatible API)
- CLI: `codenarrator analyze <url>`
- Full dependency graph support for Java, Go, Rust, C/C++
- Incremental analysis — only re-explore files changed since last run

## Local development

> *For contributors only.* End users should use `pip install codenarrator-ai` above.

```bash
git clone https://github.com/sharwariakre/CodeNarrator
cd CodeNarrator
pip install -e .  # for local development

# Start Ollama (separate terminal)
ollama serve
ollama pull qwen2.5-coder:7b
```

Optional web UI (FastAPI backend + React frontend):

```bash
# Terminal 1
cd backend && uvicorn app.main:app --reload

# Terminal 2
cd frontend && npm install && npm run dev
```

Run the test suite:

```bash
cd backend && python -m pytest tests/
```
