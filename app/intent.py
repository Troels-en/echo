"""Intent classifier: route any incoming text to query | note | complete.
Completion NEVER auto-closes — it returns ranked candidate task ids for the user to confirm.
"""
from __future__ import annotations

import logging

import httpx

from .config import Config
from .llm import call_json
from .todoist import _hdr, API

log = logging.getLogger(__name__)


INTENT_PROMPT = """You are a router for a personal voice-vault assistant. Classify the user input into ONE intent.

INTENTS:
- "query"    — user asks a question wanting an answer from their notes (e.g. "Was waren meine besten Ideen?", "Wie war mein letzter Lauf?")
- "complete" — user reports finishing one or more existing tasks (e.g. "Habe X gekauft", "Bewerbung verschickt", "die letzten drei Punkte erledigt")
- "note"     — user captures a new thought, idea, observation, or future task (default)

INPUT:
\"\"\"{text}\"\"\"

Return ONLY a JSON object:
{{
  "intent": "query" | "complete" | "note",
  "confidence": <float 0..1>
}}

Default to "note" if unsure. Pick "query" only if clearly asking for info FROM existing notes. Pick "complete" only if user reports DONE work."""


MATCH_PROMPT = """The user reported finishing some task(s). Find which OPEN TASKS they mean.

USER SAID:
\"\"\"{text}\"\"\"

OPEN TASKS (newest first):
{tasks}

Return ONLY a JSON object:
{{
  "matches": [<task ids the user most likely means, ranked best-first, max 5>],
  "reason": "<one short sentence why>"
}}

Rules:
- If the user references a COUNT or position ("die letzten drei", "last two") return that many of the NEWEST tasks in order.
- If they describe content ("GitHub-Profil aufgebaut"), match by meaning.
- Never invent ids. Only ids from the list. If nothing matches, return empty matches."""


def detect_intent(text: str, cfg: Config) -> dict:
    text = text.strip()
    if not text:
        return {"intent": "note", "confidence": 0.0}
    try:
        r = call_json(INTENT_PROMPT.format(text=text), primary=cfg.llm_primary, fallback=cfg.llm_fallback)
    except Exception as e:
        log.warning("intent classify failed → note: %s", e)
        return {"intent": "note", "confidence": 0.0}
    intent = r.get("intent", "note")
    if intent not in ("query", "complete", "note"):
        intent = "note"
    return {"intent": intent, "confidence": float(r.get("confidence", 0.0))}


def fetch_open_tasks(limit: int = 60) -> list[dict]:
    """Return open tasks newest-first. Filters Todoist onboarding/template junk."""
    with httpx.Client(timeout=10.0) as c:
        r = c.get(f"{API}/tasks", headers=_hdr())
        r.raise_for_status()
        tasks = r.json().get("results", [])
    # newest first by added/created date
    def _added(t: dict) -> str:
        return t.get("added_at") or t.get("created_at") or ""
    tasks.sort(key=_added, reverse=True)
    # drop obvious onboarding junk (contain markdown links / tutorial phrases)
    junk = ("This is a task", "Drag it", "Select this task", "Add sub-tasks",
            "Schedule this task", "Switch between", "getting started",
            "Kickstart your projects", "help center", "Quick Add",
            "new section", "Add a \"Done\"", "Start your own project",
            "number one thing", "Organize these tasks", "Get organized anywhere")
    clean = [t for t in tasks if not any(j in t.get("content", "") for j in junk)]
    return clean[:limit]


def match_tasks_for_completion(text: str, cfg: Config) -> tuple[list[dict], str]:
    """Return (candidate_tasks, reason). candidate_tasks: full task dicts, ranked."""
    open_tasks = fetch_open_tasks()
    if not open_tasks:
        return [], "Keine offenen Tasks."

    task_lines = "\n".join(f"- id={t['id']} content={t['content']!r}" for t in open_tasks)
    try:
        r = call_json(
            MATCH_PROMPT.format(text=text, tasks=task_lines),
            primary=cfg.llm_primary, fallback=cfg.llm_fallback,
        )
    except Exception as e:
        log.warning("match failed: %s", e)
        return [], "Matching fehlgeschlagen."

    by_id = {t["id"]: t for t in open_tasks}
    matches = [by_id[mid] for mid in r.get("matches", []) if mid in by_id]
    return matches, r.get("reason", "")
