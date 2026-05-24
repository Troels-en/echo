"""Structured interaction transcript — for reviewing + improving Echo.

For every incoming message it records what Echo UNDERSTOOD (intent + extracted fields)
and optionally what it did. Stored as JSONL at data/transcript.jsonl so it can be reviewed
to spot misclassifications ("you said X, Echo thought Y"). Best-effort; never raises.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import REPO_ROOT

log = logging.getLogger(__name__)

_PATH = REPO_ROOT / "data" / "transcript.jsonl"


def record(source: str, text: str, classification: dict | None, outcome: str = "") -> None:
    """Append one interaction line. source = 'text' | 'voice'. outcome = optional what-Echo-did."""
    try:
        c = classification or {}
        entry = {
            "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "source": source,
            "text": (text or "")[:600],
            "intent": c.get("intent"),
            "conf": c.get("intent_confidence"),
            "vault": c.get("vault") or "",
            "title": c.get("title") or "",
            "also_question": c.get("also_question") or "",
            "dev_repo": c.get("dev_repo") or "",
            "agent_task": c.get("agent_task") or "",
            "n_tasks": len(c.get("tasks") or []),
            "outcome": outcome,
        }
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        with _PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("transcript record failed: %s", e)


def recent(n: int = 30) -> list[dict]:
    """Last n interactions (for review)."""
    if not _PATH.exists():
        return []
    try:
        lines = _PATH.read_text(encoding="utf-8").splitlines()[-n:]
        return [json.loads(l) for l in lines if l.strip()]
    except Exception as e:
        log.warning("transcript read failed: %s", e)
        return []
