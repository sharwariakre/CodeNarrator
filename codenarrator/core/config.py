from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Default user data directory: ~/.codenarrator/
# Can be overridden via the CODENARRATOR_DATA_DIR environment variable.
_DEFAULT_DATA_DIR = Path.home() / "codenarrator"


class Settings(BaseSettings):
    # Root directory for all package-managed state (clones, cache, reports).
    DATA_DIR: Path = Path(
        os.environ.get("CODENARRATOR_DATA_DIR", str(_DEFAULT_DATA_DIR))
    )

    # Ollama model and host — both overridable via env / .env.
    OLLAMA_MODEL: str = "qwen2.5-coder:7b"
    OLLAMA_HOST: str = "http://localhost:11434"

    # Maximum allowed repo size in MB before clone is rejected (0 = no limit)
    REPO_MAX_SIZE_MB: int = 500

    model_config = SettingsConfigDict(env_file=".env")

    @property
    def REPO_BASE_DIR(self) -> Path:
        """Directory where cloned repositories are stored."""
        return self.DATA_DIR / "repos"

    @property
    def ANALYSIS_CACHE_DIR(self) -> Path:
        """Directory where persisted analysis states are stored."""
        return self.DATA_DIR / "cache"


settings = Settings()
