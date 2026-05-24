"""Intent classifier: route any incoming text to query | note | complete.
Completion NEVER auto-closes — it returns ranked candidate task ids for the user to confirm.
"""
from __future__ import annotations

import logging

import httpx

from datetime import datetime, timezone

from .config import Config
from .llm import call_json
from .todoist import _hdr, API
from . import memory

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


RANK_PROMPT = """You prioritize the user's open tasks. Decide what they should do FIRST.

NOW (Europe/Berlin): {now}

WHAT YOU KNOW ABOUT THE USER (their goals/projects — weight tasks that serve these higher):
{memory}

USER'S STATED FOCUS RIGHT NOW (if any): {focus}

OPEN TASKS (content · due · todoist priority 1-4):
{tasks}

Rank by what truly matters now: hard deadlines first, then alignment with the user's goals,
then todoist priority. A near deadline (today/this week) on something important beats everything.

Return ONLY JSON:
{{
  "ranked": [
    {{"content": "<task content, verbatim>", "why": "<one short reason: deadline / goal-fit / urgency>"}}
  ],
  "note": "<one short overall hint, e.g. what to drop or batch; optional>"
}}
Rank at most the top 6. Be decisive."""


def rank_tasks(cfg: Config, focus: str = "") -> str:
    """Fetch open tasks and LLM-rank them by deadline + the user's goals (from memory)."""
    tasks = fetch_open_tasks(limit=60)
    if not tasks:
        return "✅ Keine offenen Tasks."
    lines = []
    for t in tasks:
        due = t.get("due") or {}
        due_s = (due.get("string") or due.get("date") or "kein") if isinstance(due, dict) else "kein"
        lines.append(f"- {t.get('content','')!r} · due={due_s} · prio={t.get('priority', 1)}")
    now = datetime.now(timezone.utc).astimezone().strftime("%A %Y-%m-%d %H:%M")
    try:
        r = call_json(
            RANK_PROMPT.format(now=now, memory=memory.context() or "(nichts bekannt)",
                               focus=focus or "(nicht genannt)", tasks="\n".join(lines)),
            primary=cfg.llm_primary, fallback=cfg.llm_fallback,
        )
    except Exception as e:
        log.warning("rank failed: %s", e)
        return "❌ Priorisierung fehlgeschlagen."
    ranked = r.get("ranked") or []
    if not ranked:
        return "Konnte nicht priorisieren."
    out = ["🎯 *Was zuerst:*"]
    for i, item in enumerate(ranked[:6], 1):
        out.append(f"{i}. *{item.get('content','')[:70]}*\n   _{item.get('why','')[:90]}_")
    if r.get("note"):
        out.append(f"\n💡 {r['note']}")
    return "\n".join(out)


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
