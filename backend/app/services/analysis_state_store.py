import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

LOGGER = logging.getLogger(__name__)


def save_state(repo_id: str, local_path: str, final_state: Dict, cache_dir: Path) -> None:
    """Persist final_state to disk alongside the current git commit hash."""
    commit_hash = _get_git_commit_hash(local_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(cache_dir, repo_id)
    payload = {
        "repo_id": repo_id,
        "commit_hash": commit_hash,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "final_state": final_state,
    }
    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    LOGGER.info("Saved analysis state for %s (commit %s)", repo_id, commit_hash or "unknown")


def load_state(repo_id: str, local_path: str, cache_dir: Path) -> Optional[Dict]:
    """
    Load cached final_state if it exists and matches the current git HEAD.
    Returns None if not found or stale.
    """
    cache_file = _cache_path(cache_dir, repo_id)
    if not cache_file.exists():
        return None

    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Failed to read cache for %s: %s", repo_id, exc)
        return None

    current_hash = _get_git_commit_hash(local_path)
    saved_hash = payload.get("commit_hash")

    if current_hash and saved_hash and current_hash != saved_hash:
        LOGGER.info(
            "Cache stale for %s: saved at %s, current %s",
            repo_id, saved_hash, current_hash,
        )
        return None

    return payload.get("final_state")


def _cache_path(cache_dir: Path, repo_id: str) -> Path:
    safe_name = repo_id.replace("/", "__").replace("\\", "__")
    return cache_dir / f"{safe_name}.json"


def _get_git_commit_hash(local_path: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", local_path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None
