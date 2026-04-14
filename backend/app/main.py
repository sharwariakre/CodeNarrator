import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.api.v1.routes_health import router as health_router
from app.api.v1.routes_repo import router as repo_router
from app.services.report_generator import _get_d3_js

app = FastAPI(
    title="CodeNarrator API",
    version="0.1.0",
    description="Backend API for CodeNarrator – codebase analysis and explanation."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

app.include_router(health_router, prefix="/api/v1")
app.include_router(repo_router, prefix="/api/v1")


@app.on_event("startup")
async def prefetch_d3():
    """Pre-warm the D3 cache in the background so report generation never blocks on a CDN fetch."""
    asyncio.get_event_loop().run_in_executor(None, _get_d3_js)


@app.get("/report-file", response_class=HTMLResponse)
async def serve_report_file(path: str = Query(..., description="Absolute path to the HTML report")):
    """Serve a generated HTML report file by absolute path."""
    report_path = Path(path).resolve()
    reports_root = (Path(__file__).parent.parent / "data" / "reports").resolve()
    try:
        report_path.relative_to(reports_root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: path outside reports directory")
    if not report_path.exists() or not report_path.is_file():
        raise HTTPException(status_code=404, detail="Report file not found")
    return HTMLResponse(content=report_path.read_text(encoding="utf-8"))
