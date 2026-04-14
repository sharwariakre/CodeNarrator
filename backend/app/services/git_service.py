import re
import urllib.request
import urllib.error
import json
from pathlib import Path
from typing import Optional
import shutil

from git import Repo, GitCommandError

from app.core.config import settings


class GitCloneError(Exception):
    pass


def _check_github_repo_size(repo_url: str) -> None:
    """
    Query the GitHub API for repo size and raise GitCloneError if it exceeds
    settings.REPO_MAX_SIZE_MB. Only applies to github.com URLs.
    Does nothing if REPO_MAX_SIZE_MB is 0 (disabled) or the URL is not GitHub.
    """
    if settings.REPO_MAX_SIZE_MB <= 0:
        return

    url_str = str(repo_url)
    match = re.search(r"github\.com[/:]([^/]+)/([^/\s.]+?)(?:\.git)?$", url_str)
    if not match:
        return  # not a GitHub URL — skip check

    owner, repo = match.group(1), match.group(2)
    api_url = f"https://api.github.com/repos/{owner}/{repo}"

    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "CodeNarrator"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        size_kb = data.get("size", 0)
        size_mb = size_kb / 1024
        if size_mb > settings.REPO_MAX_SIZE_MB:
            raise GitCloneError(
                f"Repository is too large ({size_mb:.0f} MB). "
                f"Maximum supported size is {settings.REPO_MAX_SIZE_MB} MB."
            )
    except GitCloneError:
        raise
    except Exception:
        pass  # API unreachable or rate-limited — allow clone to proceed


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
        _check_github_repo_size(repo_url)
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
