"""Short-term conversation memory: the last N message turns, persisted to disk and
injected into the classifier + ask/query prompts so follow-ups like "weißt du was ich
meine?" or "und dazu noch X" resolve against what was just said.

Distinct from app/memory.py (durable long-term facts about the user). This is the
rolling chat window and is intentionally lossy.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .config import REPO_ROOT

log = logging.getLogger(__name__)

_PATH = REPO_ROOT / "data" / "shortterm.json"
_MAX_TURNS = 10          # keep at most this many turns on disk
_MAX_AGE_S = 2 * 3600    # turns older than this are ignored (stale conversation)
_MAX_CHARS = 600         # truncate each stored turn


def _load() -> list[dict]:
    if not _PATH.exists():
        return []
    try:
        return json.loads(_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("shortterm load failed: %s", e)
        return []


def add(role: str, text: str) -> None:
    """Append a turn (role = 'user' | 'echo'). Best-effort; never raises into handlers."""
    text = (text or "").strip()
    if not text:
        return
    try:
        turns = _load()
        turns.append({"ts": time.time(), "role": role, "text": text[:_MAX_CHARS]})
        turns = turns[-_MAX_TURNS:]
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        _PATH.write_text(json.dumps(turns, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        log.warning("shortterm add failed: %s", e)


def recent_text(n: int = 6) -> str:
    """Last n non-stale turns as 'role: text' lines, oldest first. Empty if none."""
    now = time.time()
    turns = [t for t in _load() if now - t.get("ts", 0) <= _MAX_AGE_S]
    turns = turns[-n:]
    return "\n".join(f"{t['role']}: {t['text']}" for t in turns)


def clear() -> None:
    _PATH.unlink(missing_ok=True)
