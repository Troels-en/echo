"""Tiny JSON state store for runtime prefs (chat_id, briefing settings)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = REPO_ROOT / "data" / "state.json"

DEFAULTS = {
    "chat_id": None,
    "briefing_enabled": True,
    "briefing_time": "07:30",  # local Europe/Berlin
}


def load() -> dict:
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            return {**DEFAULTS, **data}
        except Exception as e:
            log.warning("state load failed: %s", e)
    return dict(DEFAULTS)


def save(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def set_key(key: str, value) -> dict:
    s = load()
    s[key] = value
    save(s)
    return s
