from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.api.v1.routes_health import router as health_router
from app.api.v1.routes_repo import router as repo_router

app = FastAPI(
    title="CodeNarrator API",
    version="0.1.0",
    description="Backend API for CodeNarrator – codebase analysis and explanation."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api/v1")
app.include_router(repo_router, prefix="/api/v1")


@app.get("/report-file", response_class=HTMLResponse)
async def serve_report_file(path: str = Query(..., description="Absolute path to the HTML report")):
    """Serve a generated HTML report file by absolute path."""
    report_path = Path(path).resolve()
    # Safety: only serve files inside the data/reports directory
    reports_root = (Path(__file__).parent.parent / "data" / "reports").resolve()
    if not report_path.is_relative_to(reports_root):
        raise HTTPException(status_code=403, detail="Access denied: path outside reports directory")
    if not report_path.exists() or not report_path.is_file():
        raise HTTPException(status_code=404, detail="Report file not found")
    return HTMLResponse(content=report_path.read_text(encoding="utf-8"))
