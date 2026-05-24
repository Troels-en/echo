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

log = logging.getLogger(__name__)

DEV_ROOT = Path(os.path.expanduser(os.getenv("DEV_ROOT", "~"))).resolve()
_CLAUDE_BIN = "claude"


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


_TASK_PROMPT = """You are an autonomous coding agent working inside this git repository.
Implement the following request on the CURRENT branch (already created for you). Rules:
- Make the smallest correct change. Match existing style. Do not push.
- If you add/run code, verify it compiles/tests where feasible.
- When done, end with a 3-5 line summary: what you changed, files touched, how to verify.

REQUEST:
{task}"""


def run_devtask(repo: Path, task: str, timeout: int = 1800) -> dict:
    """Create a branch, run the claude agent, commit. Returns {branch, report, changed, error}."""
    branch = f"echo/dev-{datetime.now():%Y%m%d-%H%M%S}"
    co = _git(repo, "checkout", "-b", branch)
    if co.returncode != 0:
        return {"error": f"git branch failed: {co.stderr[-300:]}"}

    cmd = [
        _CLAUDE_BIN, "-p", _TASK_PROMPT.format(task=task),
        "--permission-mode", "acceptEdits",
        "--allowedTools", "Read", "Write", "Edit", "Bash", "Glob", "Grep",
        "--output-format", "text",
    ]
    log.info("devtask: claude agent in %s on %s", repo, branch)
    try:
        res = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"branch": branch, "error": f"Agent-Timeout nach {timeout}s (Branch {branch} bleibt zum Prüfen)."}
    report = (res.stdout or "").strip()
    if res.returncode != 0:
        report += f"\n[agent exit {res.returncode}: {res.stderr[-200:]}]"

    _git(repo, "add", "-A")
    commit = _git(repo, "commit", "-m", f"echo devtask: {task[:60]}")
    committed = commit.returncode == 0
    changed = _git(repo, "diff", "--stat", "HEAD~1", "HEAD").stdout.strip() if committed else "(keine Änderungen committet)"

    return {"branch": branch, "report": report[-2500:], "changed": changed, "committed": committed, "error": ""}
