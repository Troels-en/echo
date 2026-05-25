"""Echo → Claude Code trigger: turn a phone text/voice request into a real dev task,
executed by a headless `claude -p` agent in a target repo.

Safety model (matches the user's 'Bestätigungsschicht' notes):
- Confirmation required before anything runs (handled in bot via inline buttons).
- Work happens on a NEW branch, never on the current/main branch.
- Agent commits but NEVER pushes — everything stays local + reviewable.
- Only repos found under DEV_ROOT are eligible.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

from . import interactive

log = logging.getLogger(__name__)

DEV_ROOT = Path(os.path.expanduser(os.getenv("DEV_ROOT", "~"))).resolve()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def resolve_repo(hint: str) -> Path | None:
    """Find a git repo directly under DEV_ROOT whose name matches the hint (fuzzy)."""
    if not hint:
        return None
    want = _slug(hint)
    candidates: list[Path] = []
    for d in DEV_ROOT.iterdir():
        if not d.is_dir() or not (d / ".git").exists():
            continue
        name = _slug(d.name)
        if want == name or want in name or name in want:
            candidates.append(d)
    if not candidates:
        return None
    # prefer exact match, else shortest name (most specific)
    candidates.sort(key=lambda p: (want != _slug(p.name), len(p.name)))
    return candidates[0]


def _git(repo: Path, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, timeout=timeout)


def run_devtask(repo: Path, task: str, session_id: str = "", answers: str = "",
                timeout: int = 1800) -> dict:
    """Create a branch, run the agent (resuming the plan session if given), commit, then always
    restore the original branch. Returns {branch, base, report, changed, committed, error}."""
    orig = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() or "main"
    branch = f"echo/dev-{datetime.now():%Y%m%d-%H%M%S}"
    co = _git(repo, "checkout", "-b", branch)
    if co.returncode != 0:
        return {"error": f"git branch failed: {co.stderr[-300:]}"}

    log.info("devtask: agent in %s on %s (resume=%s)", repo, branch, bool(session_id))
    try:
        res = interactive.execute(interactive.DEVTASK, session_id, answers, repo, timeout)
        report = res.get("report", "")
        if res.get("error"):
            report = f"{report}\n[{res['error']}]".strip()

        _git(repo, "add", "-A")
        commit = _git(repo, "commit", "-m", f"echo devtask: {task[:60]}")
        committed = commit.returncode == 0
        changed = (_git(repo, "diff", "--stat", "HEAD~1", "HEAD").stdout.strip()
                   if committed else "(keine Änderungen committet)")
        return {"branch": branch, "base": orig, "report": report[-2500:],
                "changed": changed, "committed": committed, "error": res.get("error", "")}
    finally:
        # Always return the repo to where it was — the work stays on `branch`, retrievable for
        # review. Critical when the target is a live repo (e.g. echo_vault itself).
        _git(repo, "checkout", orig)
