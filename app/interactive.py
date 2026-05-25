"""Interactive engine shared by devtask + agenttask.

A read-only PLAN pass runs the real `claude -p` agent (so it can SEE what already exists in the
repo / vaults / Notion) and decides whether clarifying questions are needed before any write.
The plan pass returns a session_id; execution RESUMES that same session, so the discovery done
while planning carries into execution and the agent does not redo existing work.

One engine; per-feature differences live in Profile (tools + execution rules). cwd and the
git branch lifecycle for devtask stay in the feature wrappers (app/devtask.py).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .llm import LLMError, _extract_json

log = logging.getLogger(__name__)

HOME = Path(os.path.expanduser("~"))
_CLAUDE_BIN = "claude"

_NOTION_TOOLS = [
    "mcp__claude_ai_Notion__notion-search",
    "mcp__claude_ai_Notion__notion-fetch",
    "mcp__claude_ai_Notion__notion-create-pages",
    "mcp__claude_ai_Notion__notion-update-page",
]


@dataclass
class Profile:
    kind: str               # "devtask" | "agenttask"
    where: str              # human description of where existing work lives (for the plan prompt)
    plan_tools: list[str]   # READ-ONLY discovery tools (no Write/Edit/Bash -> mutation impossible)
    exec_tools: list[str]   # tools allowed at execution time
    exec_rules: str         # feature-specific execution rules


DEVTASK = Profile(
    kind="devtask",
    where="this git repository",
    plan_tools=["Read", "Glob", "Grep"],
    exec_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
    exec_rules=("Make the smallest correct change. Match existing style. Do NOT push. "
                "If you add or run code, verify it compiles or tests where feasible."),
)

AGENTTASK = Profile(
    kind="agenttask",
    where="their Obsidian vaults (~/*_Vault), their SecondBrain (~/SecondBrain), files under home, and their Notion (via the Notion MCP)",
    plan_tools=["Read", "Glob", "Grep", _NOTION_TOOLS[0], _NOTION_TOOLS[1]],
    exec_tools=["Read", "Write", "Edit", "Glob", "Grep", "Bash", *_NOTION_TOOLS],
    exec_rules=("Be ADDITIVE and non-destructive: create or append, never delete or overwrite "
                "large content. Do NOT run destructive shell commands (no rm, git push, etc.). "
                "When unsure, create a new note rather than edit in place."),
)


_PLAN_PROMPT = """You are Echo's {kind} agent, in PLAN MODE.
DO NOT write, edit, create, commit, or run any mutating command. Use ONLY the read-only tools
available to you to discover what ALREADY EXISTS relevant to the task in {where}. Then decide
whether you can do the task correctly without asking the user anything.

Output ONLY a JSON object, no prose and no code fence:
{{
  "existing_work": "<1 to 3 lines: relevant files or pages that already exist, or 'none found'>",
  "plan": "<what you will do, 2 to 4 short lines>",
  "questions": [{{"id": "<slug>", "question": "<one clear question>", "choices": ["<opt>", "..."]}}],
  "ready": <true if you need no questions and can proceed>,
  "recommended_default": "<if there are questions: the safest default course, else empty>"
}}
Ask a question ONLY if the answer would change what you do. Prefer 0 to 3 sharp questions.

TASK:
{task}"""


_EXEC_PROMPT = """You are no longer in plan mode. EXECUTE the task now, end-to-end. Actually do it.
{answers_block}{rules}
Build on what you already found during planning; do not redo work that already exists.
When done, give a 3 to 6 line summary: what you read, what you created or updated (with paths or titles)."""


def _run(args: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run([_CLAUDE_BIN, *args], cwd=str(cwd),
                          capture_output=True, text=True, timeout=timeout)


def plan(profile: Profile, task: str, cwd: Path, timeout: int = 600) -> dict:
    """Read-only discovery + clarify pass. Returns:
    {session_id, existing_work, plan, questions, ready, recommended_default, error}."""
    prompt = _PLAN_PROMPT.format(kind=profile.kind, where=profile.where, task=task)
    args = ["-p", prompt, "--permission-mode", "acceptEdits",
            "--allowedTools", *profile.plan_tools, "--output-format", "json"]
    log.info("plan: %s for %r", profile.kind, task[:80])
    try:
        res = _run(args, cwd, timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"Plan-Timeout nach {timeout}s.", "ready": False, "questions": []}
    if res.returncode != 0:
        return {"error": f"plan exit {res.returncode}: {res.stderr[-200:]}", "ready": False, "questions": []}
    try:
        envelope = json.loads(res.stdout)
        session_id = envelope.get("session_id", "")
        text = envelope.get("result", "")
    except json.JSONDecodeError:
        return {"error": "plan output not JSON", "ready": False, "questions": []}
    try:
        parsed = _extract_json(text)
    except LLMError:
        # no parseable plan JSON: proceed without questions rather than block the user
        return {"session_id": session_id, "existing_work": "", "plan": text[:400],
                "questions": [], "ready": True, "recommended_default": "", "error": ""}
    return {
        "session_id": session_id,
        "existing_work": parsed.get("existing_work", ""),
        "plan": parsed.get("plan", ""),
        "questions": parsed.get("questions") or [],
        "ready": bool(parsed.get("ready", False)),
        "recommended_default": parsed.get("recommended_default", ""),
        "error": "",
    }


def execute(profile: Profile, session_id: str, answers: str, cwd: Path, timeout: int = 1800) -> dict:
    """Resume the plan session (if session_id) and run the task. Returns {report, error}."""
    answers_block = f"The user answered your questions:\n{answers}\n\n" if answers.strip() else ""
    prompt = _EXEC_PROMPT.format(answers_block=answers_block, rules=profile.exec_rules)
    args = ["-p", prompt]
    if session_id:
        args += ["--resume", session_id]
    args += ["--permission-mode", "acceptEdits",
             "--allowedTools", *profile.exec_tools, "--output-format", "text"]
    log.info("execute: %s resume=%s", profile.kind, bool(session_id))
    try:
        res = _run(args, cwd, timeout)
    except subprocess.TimeoutExpired:
        return {"report": "", "error": f"Timeout nach {timeout}s."}
    report = (res.stdout or "").strip()
    if res.returncode != 0:
        return {"report": report[-2000:], "error": f"agent exit {res.returncode}: {res.stderr[-200:]}"}
    return {"report": report[-2500:], "error": ""}
