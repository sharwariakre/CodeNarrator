from pydantic_settings import BaseSettings
from pathlib import Path

# Anchor data paths to the repo root so the defaults resolve regardless
# of the cwd the codenarrator package is loaded from.
# Path(__file__) = codenarrator/core/config.py
#   .parent               = codenarrator/core
#   .parent.parent        = codenarrator
#   .parent.parent.parent = repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_DATA = _REPO_ROOT / "backend" / "data"


class Settings(BaseSettings):
    # Base directory for cloned repositories
    REPO_BASE_DIR: Path = _BACKEND_DATA / "repos"
    # Directory for persisted analysis state cache
    ANALYSIS_CACHE_DIR: Path = _BACKEND_DATA / "analysis_cache"
    # Ollama model used for both agentic loop and architecture interpretation
    OLLAMA_MODEL: str = "qwen2.5-coder:7b"
    # Maximum allowed repo size in MB before clone is rejected (0 = no limit)
    REPO_MAX_SIZE_MB: int = 500

    class Config:
        env_file = ".env"

settings = Settings()
