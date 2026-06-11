"""
GitHub Auto-Sync
================
Pushes the latest brain state (checkpoints, CKG, weights) to GitHub
after every save.  Runs entirely in a background daemon thread so it
never blocks the mesh.

Reads credentials from env vars (loaded from .env at startup):
  GITHUB_TOKEN  — Personal Access Token with repo scope
  GITHUB_USER   — GitHub username
  GITHUB_REMOTE — full HTTPS remote URL
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_push_count   = 0
_last_push_ts: str | None = None
_last_push_ok: bool       = False
_last_push_msg: str       = "never pushed"


def _run(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=60, cwd=cwd or os.getcwd(),
        )
        return r.returncode, (r.stdout + r.stderr).strip()
    except Exception as exc:
        return 1, str(exc)


def _token_remote() -> str | None:
    token  = os.environ.get("GITHUB_TOKEN", "").strip()
    user   = os.environ.get("GITHUB_USER", "").strip()
    remote = os.environ.get("GITHUB_REMOTE", "").strip()
    if not token or not remote:
        return None
    if user and "://" in remote:
        proto, rest = remote.split("://", 1)
        remote = f"{proto}://{user}:{token}@{rest.split('@')[-1]}"
    return remote


def push_now(tag: str = "") -> dict:
    """
    Commit every tracked/untracked change and push to GitHub.
    Called from a background thread — never raises.
    """
    global _push_count, _last_push_ts, _last_push_ok, _last_push_msg

    with _lock:
        remote = _token_remote()
        if not remote:
            msg = "GITHUB_TOKEN or GITHUB_REMOTE not set — skipping push"
            logger.warning(f"[GitHubSync] {msg}")
            return {"ok": False, "msg": msg}

        ts  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        msg = f"auto-sync: brain checkpoint {tag or ts}"

        code, out = _run(["git", "add", "-A"])
        if code != 0:
            logger.warning(f"[GitHubSync] git add failed: {out}")

        code, out = _run(["git", "commit", "--no-optional-locks",
                          "--allow-empty", "-m", msg])
        if code != 0 and "nothing to commit" not in out:
            logger.warning(f"[GitHubSync] git commit: {out}")

        code, out = _run(["git", "push", remote, "main"])
        ok = code == 0

        _push_count   += 1
        _last_push_ts  = ts
        _last_push_ok  = ok
        _last_push_msg = out[:200] if out else ("ok" if ok else "unknown error")

        if ok:
            logger.info(f"[GitHubSync] ✓ pushed  tag={tag or ts}  ({_push_count} total)")
        else:
            logger.warning(f"[GitHubSync] push failed: {_last_push_msg}")

        return {"ok": ok, "msg": _last_push_msg, "tag": tag or ts}


def push_background(tag: str = "") -> None:
    """Fire-and-forget background push."""
    t = threading.Thread(target=push_now, args=(tag,), daemon=True, name="github-sync")
    t.start()


def status() -> dict:
    return {
        "push_count":    _push_count,
        "last_push_ts":  _last_push_ts,
        "last_push_ok":  _last_push_ok,
        "last_push_msg": _last_push_msg,
        "token_set":     bool(os.environ.get("GITHUB_TOKEN")),
        "remote":        os.environ.get("GITHUB_REMOTE", "not set"),
    }
