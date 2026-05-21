"""Persistent personalization memory: durable facts about the user.

Facts are learned passively during note ingest (people, preferences, projects,
patterns) and injected into prompts so the assistant gets more personal over time.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
MEM_FILE = REPO_ROOT / "data" / "memory.json"

VALID_TYPES = {"person", "preference", "project", "pattern", "fact"}
MAX_FACTS = 120
CONTEXT_CAP = 40  # facts injected into prompts


def _load() -> list[dict]:
    if MEM_FILE.exists():
        try:
            return json.loads(MEM_FILE.read_text())
        except Exception as e:
            log.warning("memory load failed: %s", e)
    return []


def _save(facts: list[dict]) -> None:
    MEM_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEM_FILE.write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def add_facts(new: list[dict]) -> int:
    """new: [{text, type}]. Dedupes by normalized text. Returns count added."""
    facts = _load()
    existing = {_norm(f["text"]) for f in facts}
    added = 0
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    for item in new:
        text = (item.get("text") or "").strip()
        ftype = item.get("type", "fact")
        if not text or _norm(text) in existing:
            continue
        if ftype not in VALID_TYPES:
            ftype = "fact"
        facts.append({"text": text, "type": ftype, "created": now})
        existing.add(_norm(text))
        added += 1
    if len(facts) > MAX_FACTS:
        facts = facts[-MAX_FACTS:]
    if added:
        _save(facts)
    return added


def context(cap: int = CONTEXT_CAP) -> str:
    """Compact memory string for prompt injection. Empty if no facts."""
    facts = _load()
    if not facts:
        return ""
    recent = facts[-cap:]
    by_type: dict[str, list[str]] = {}
    for f in recent:
        by_type.setdefault(f["type"], []).append(f["text"])
    lines = []
    for t in ("person", "preference", "project", "pattern", "fact"):
        if t in by_type:
            lines.append(f"{t}: " + "; ".join(by_type[t]))
    return "\n".join(lines)


def all_facts() -> list[dict]:
    return _load()


def forget(substring: str) -> int:
    facts = _load()
    sub = _norm(substring)
    kept = [f for f in facts if sub not in _norm(f["text"])]
    removed = len(facts) - len(kept)
    if removed:
        _save(kept)
    return removed
