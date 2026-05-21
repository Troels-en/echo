"""LLM router via CLI (codex / claude) subprocess. Returns parsed JSON."""
from __future__ import annotations

import json
import logging
import re
import subprocess

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


CODEX_BIN = "codex"
CLAUDE_BIN = "claude"

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_BARE_JSON = re.compile(r"(\{(?:[^{}]|(?:\{[^{}]*\}))*\})", re.DOTALL)


def _extract_json(text: str) -> dict:
    m = _JSON_BLOCK.search(text)
    candidate = m.group(1) if m else None
    if not candidate:
        m = _BARE_JSON.search(text)
        candidate = m.group(1) if m else None
    if not candidate:
        raise LLMError(f"no JSON in LLM output: {text[:300]!r}")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise LLMError(f"invalid JSON: {e}: {candidate[:300]!r}") from e


def _run_codex(prompt: str, timeout: int = 90) -> str:
    cmd = [CODEX_BIN, "exec", "--skip-git-repo-check", prompt]
    log.info("codex exec (len=%d)", len(prompt))
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise LLMError(f"codex exit {res.returncode}: {res.stderr[-400:]}")
    return res.stdout


def _run_claude(prompt: str, timeout: int = 90) -> str:
    cmd = [CLAUDE_BIN, "-p", prompt]
    log.info("claude -p (len=%d)", len(prompt))
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise LLMError(f"claude exit {res.returncode}: {res.stderr[-400:]}")
    return res.stdout


def call_json(prompt: str, primary: str = "codex", fallback: str = "claude") -> dict:
    runners = {"codex": _run_codex, "claude": _run_claude}
    last_err: Exception | None = None
    for backend in (primary, fallback):
        runner = runners.get(backend)
        if not runner:
            continue
        try:
            raw = runner(prompt)
            return _extract_json(raw)
        except Exception as e:
            log.warning("%s failed: %s", backend, e)
            last_err = e
            continue
    raise LLMError(f"all LLM backends failed: {last_err}")
