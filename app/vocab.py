"""Self-service Whisper vocabulary: names/jargon the user teaches the bot so
transcription spells them correctly. Fed into the Whisper initial prompt.

Managed entirely from Telegram via /vocab, no file editing needed.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
VOCAB_FILE = REPO_ROOT / "data" / "whisper_vocab.json"

# Base prompt (overridable via WHISPER_PROMPT); learned terms are appended.
_BASE = os.getenv(
    "WHISPER_PROMPT",
    "Begriffe: Echo, Remi, Gründerszene, Todoist, Obsidian.",
).strip()
MAX_TERMS = 200


def _load() -> list[str]:
    if VOCAB_FILE.exists():
        try:
            return json.loads(VOCAB_FILE.read_text())
        except Exception as e:
            log.warning("vocab load failed: %s", e)
    return []


def _save(terms: list[str]) -> None:
    VOCAB_FILE.parent.mkdir(parents=True, exist_ok=True)
    VOCAB_FILE.write_text(json.dumps(terms, indent=2, ensure_ascii=False), encoding="utf-8")


def all_terms() -> list[str]:
    return _load()


def add(arg: str) -> list[str]:
    """Add one term, or several comma-separated. Multi-word terms allowed.
    Returns the list of newly added terms (empty if all were duplicates)."""
    parts = [t.strip() for t in arg.split(",")]
    terms = _load()
    lower = {t.lower() for t in terms}
    added: list[str] = []
    for t in parts:
        if t and t.lower() not in lower:
            terms.append(t)
            lower.add(t.lower())
            added.append(t)
    if added:
        _save(terms[-MAX_TERMS:])
    return added


def remove(term: str) -> bool:
    terms = _load()
    kept = [t for t in terms if t.lower() != term.strip().lower()]
    if len(kept) == len(terms):
        return False
    _save(kept)
    return True


def prompt() -> str:
    """Whisper initial prompt = base + learned terms. Read fresh each call so
    newly taught terms apply immediately, no restart needed."""
    terms = _load()
    if not terms:
        return _BASE
    learned = "Eigennamen: " + ", ".join(terms) + "."
    return f"{_BASE} {learned}".strip() if _BASE else learned
