from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Base directory for cloned repositories
    REPO_BASE_DIR: Path = Path("./data/repos")
    # Directory for persisted analysis state cache
    ANALYSIS_CACHE_DIR: Path = Path("./data/analysis_cache")
    # Ollama model used for both agentic loop and architecture interpretation
    OLLAMA_MODEL: str = "qwen2.5-coder:7b"
    # Maximum allowed repo size in MB before clone is rejected (0 = no limit)
    REPO_MAX_SIZE_MB: int = 500

    class Config:
        env_file = ".env"

settings = Settings()
