"""General agent task: Echo EXECUTES a multi-step request on the user's knowledge/data
(Notion + Obsidian vaults + files) by delegating to a headless `claude -p` agent — instead
of just logging a Todoist task. The generalization of devtask beyond code.

Examples: "pull my Notion habits into the vault", "clean up SecondBrain", "summarize my
finance notes into one page", "sync X from Notion to Y".

Safety: confirmation required before running (handled in bot). The agent may read/write the
user's vaults and Notion — it is told to be additive/non-destructive. Runs from the home dir.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

HOME = Path(os.path.expanduser("~"))
_CLAUDE_BIN = "claude"
_NOTION_TOOLS = [
    "mcp__claude_ai_Notion__notion-search",
    "mcp__claude_ai_Notion__notion-fetch",
    "mcp__claude_ai_Notion__notion-create-pages",
    "mcp__claude_ai_Notion__notion-update-page",
]

_AGENT_PROMPT = """You are Echo's executor agent on the user's machine. You can read/write their
Obsidian vaults (~/*_Vault), their SecondBrain (~/SecondBrain), other files under home, and their
Notion (via the Notion MCP). Carry out the task below END-TO-END — actually do it, don't just plan.

Rules:
- Be ADDITIVE and non-destructive: create or append; never delete files/pages or overwrite large
  content unless the task explicitly says so. When unsure, create a new note rather than edit in place.
- Keep changes minimal and well-organized; match the existing structure of the vault/wiki.
- Do NOT run destructive shell commands (no rm, git push, etc.).
- When done, give a 3-6 line summary: what you read, what you created/updated (with paths/page titles).

TASK:
{task}"""


def run_agenttask(task: str, timeout: int = 1800) -> dict:
    """Run the executor agent for a general task. Returns {report, error}."""
    cmd = [
        _CLAUDE_BIN, "-p", _AGENT_PROMPT.format(task=task),
        "--permission-mode", "acceptEdits",
        "--allowedTools", "Read", "Write", "Edit", "Glob", "Grep", "Bash", *_NOTION_TOOLS,
        "--output-format", "text",
    ]
    log.info("agenttask: claude executor for %r", task[:80])
    try:
        res = subprocess.run(cmd, cwd=str(HOME), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"report": "", "error": f"Agent-Timeout nach {timeout}s."}
    report = (res.stdout or "").strip()
    if res.returncode != 0:
        return {"report": report[-2000:], "error": f"agent exit {res.returncode}: {res.stderr[-200:]}"}
    return {"report": report[-2500:], "error": ""}
