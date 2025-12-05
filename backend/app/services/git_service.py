from pathlib import Path
from typing import Optional
import shutil

from git import Repo, GitCommandError

from app.core.config import settings


class GitCloneError(Exception):
    pass


def get_repo_local_path(repo_url: str) -> Path:
    """
    Derive a local path for cloning based on repo name.
    Example:
      https://github.com/user/project -> data/repos/user__project
    """

    settings.REPO_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Convert HttpUrl (Pydantic type) to string
    url_str = str(repo_url)

    # Create a filesystem-safe slug
    slug = (
        url_str.replace("https://", "")
        .replace("http://", "")
        .replace("git@", "")
        .replace("github.com:", "")
        .replace("/", "__")
        .replace(".git", "")
    )

    return settings.REPO_BASE_DIR / slug


def clone_or_update_repo(repo_url: str, force_clean: bool = False) -> Path:
    """
    Clone the repo if not present; otherwise pull latest.
    If force_clean is True, delete and re-clone.
    """
    local_path = get_repo_local_path(repo_url)

    if force_clean and local_path.exists():
        shutil.rmtree(local_path)

    if not local_path.exists():
        try:
            Repo.clone_from(repo_url, local_path)
        except GitCommandError as e:
            raise GitCloneError(f"Failed to clone repo: {e}") from e
    else:
        # Try to pull latest
        try:
            repo = Repo(local_path)
            origin = repo.remotes.origin
            origin.pull()
        except GitCommandError as e:
            # Not fatal for MVP – we can ignore or re-clone
            raise GitCloneError(f"Failed to update repo: {e}") from e

    return local_path
