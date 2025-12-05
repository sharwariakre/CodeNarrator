from fastapi import FastAPI
from app.api.v1.routes_health import router as health_router
from app.api.v1.routes_repo import router as repo_router

app = FastAPI(
    title="CodeNarrator API",
    version="0.1.0",
    description="Backend API for CodeNarrator – codebase analysis and explanation."
)

app.include_router(health_router, prefix="/api/v1")
app.include_router(repo_router, prefix="/api/v1")
