from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Base directory for cloned repositories
    REPO_BASE_DIR: Path = Path("./data/repos")

    class Config:
        env_file = ".env"

settings = Settings()
