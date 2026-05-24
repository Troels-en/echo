"""Mirror habit check-ins into Notion — via a headless `claude -p` agent that has the
Notion MCP (account-connected), so NO token/page setup is needed. The agent finds or
creates the target page itself. Habits_Vault stays the source of truth; this is a
read-only convenience mirror. Best-effort: failures never block the check-in.
"""
from __future__ import annotations

import logging
import subprocess

log = logging.getLogger(__name__)

_CLAUDE_BIN = "claude"
_NOTION_TOOLS = [
    "mcp__claude_ai_Notion__notion-search",
    "mcp__claude_ai_Notion__notion-fetch",
    "mcp__claude_ai_Notion__notion-create-pages",
    "mcp__claude_ai_Notion__notion-update-page",
]


def mirror_habit_log(date: str, entry: str, timeout: int = 180) -> bool:
    """Append '<date>: <entry>' to a Notion page 'Echo Habit Log' (create if missing),
    via a claude agent using the Notion MCP. Returns True on success."""
    prompt = (
        "Use the Notion MCP. Append my daily habit check-in to a Notion page titled "
        "'Echo Habit Log' — create that page in my workspace if it doesn't exist yet. "
        f"Add exactly one bulleted line: \"{date}: {entry}\". Do not duplicate existing "
        "lines. Reply only with 'DONE' when added."
    )
    cmd = [_CLAUDE_BIN, "-p", prompt,
           "--allowedTools", *_NOTION_TOOLS,
           "--output-format", "text"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        ok = res.returncode == 0 and "DONE" in (res.stdout or "").upper()
        if not ok:
            log.warning("notion mirror agent: rc=%s out=%s", res.returncode, (res.stdout or "")[-200:])
        return ok
    except Exception as e:
        log.warning("notion mirror failed: %s", e)
        return False
