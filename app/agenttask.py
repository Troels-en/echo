"""General agent task: Echo EXECUTES a multi-step request on the user's knowledge/data
(Notion + Obsidian vaults + files) by delegating to a headless `claude -p` agent — instead
of just logging a Todoist task. The generalization of devtask beyond code.

Examples: "pull my Notion habits into the vault", "clean up SecondBrain", "summarize my
finance notes into one page", "sync X from Notion to Y".

Safety: confirmation required before running (handled in bot). The agent may read/write the
user's vaults and Notion — it is told to be additive/non-destructive. Runs from the home dir.
The execution itself (incl. resuming the read-only plan session) lives in app/interactive.py.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from . import interactive

log = logging.getLogger(__name__)

HOME = Path(os.path.expanduser("~"))


def run_agenttask(task: str, session_id: str = "", answers: str = "", timeout: int = 1800) -> dict:
    """Run the executor agent for a general task, resuming the plan session if given.
    Returns {report, error}."""
    log.info("agenttask: executor for %r (resume=%s)", task[:80], bool(session_id))
    return interactive.execute(interactive.AGENTTASK, session_id, answers, HOME, timeout)
