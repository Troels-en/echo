"""Read recent Gmail, summarize, extract actionable tasks + calendar events."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from .config import Config
from .llm import call_json
from . import gcal

log = logging.getLogger(__name__)


TRIAGE_PROMPT = """You triage the user's recent emails. Summarize and extract actionable items.

NOW (Europe/Berlin): {now}

EMAILS:
{emails}

Return ONLY a JSON object:
{{
  "digest": [
    {{"from": "<sender short>", "subject": "<subject>", "summary": "<one line>", "needs_action": <true|false>}}
  ],
  "tasks": [
    {{"content": "<imperative action derived from a mail>", "due_string": "<'today'|'tomorrow'|'next monday'|'<DD MMM>' or empty>", "priority": <1-4>}}
  ],
  "events": [
    {{"summary": "<event title>", "start": "<ISO 8601 resolved against NOW>", "end": "<ISO or empty>", "location": "<or empty>"}}
  ]
}}

Rules:
- Ignore pure newsletters/promotions for tasks/events (still list in digest with needs_action=false).
- Only extract a task if the mail clearly asks the user to DO something.
- Only extract an event if the mail names a specific date/time meeting/appointment.
- Keep digest to the emails given. Be concise."""


def triage(cfg: Config, max_results: int = 8, query: str = "in:inbox") -> dict:
    mails = gcal.list_recent_mail(max_results=max_results, query=query)
    if not mails:
        return {"digest": [], "tasks": [], "events": [], "count": 0}

    email_block = "\n\n".join(
        f"[{i+1}] FROM: {m['from']}\nSUBJECT: {m['subject']}\nBODY: {m['body'][:800]}"
        for i, m in enumerate(mails)
    )
    now = datetime.now(timezone.utc).astimezone().strftime("%A %Y-%m-%d %H:%M")
    prompt = TRIAGE_PROMPT.format(now=now, emails=email_block)
    result = call_json(prompt, primary=cfg.llm_primary, fallback=cfg.llm_fallback)
    result["count"] = len(mails)
    return result


SEARCH_PROMPT = """Answer the user's question using only these emails.

QUESTION: {question}

EMAILS:
{emails}

Return ONLY JSON: {{"answer": "<concise answer, cite sender>", "found": <true|false>}}"""


def search(cfg: Config, search_terms: str, question: str, max_results: int = 6) -> dict:
    """Gmail search by terms, then answer the user's question from results."""
    query = search_terms or question
    mails = gcal.list_recent_mail(max_results=max_results, query=query)
    if not mails:
        # retry broader
        mails = gcal.list_recent_mail(max_results=max_results, query=f"{query} in:anywhere")
    if not mails:
        return {"answer": f"Keine Mails gefunden für '{query}'.", "found": False, "count": 0}
    email_block = "\n\n".join(
        f"FROM: {m['from']}\nSUBJECT: {m['subject']}\nBODY: {m['body'][:600]}"
        for m in mails
    )
    result = call_json(
        SEARCH_PROMPT.format(question=question, emails=email_block),
        primary=cfg.llm_primary, fallback=cfg.llm_fallback,
    )
    result["count"] = len(mails)
    return result


CLEAN_PROMPT = """Identify OBVIOUS junk in this inbox: dead newsletters, promotional spam, expired offers, automated noise the user clearly doesn't need.

CRITICAL: When in ANY doubt, KEEP it. Only flag clear junk. Never flag personal mail, work mail, receipts, security alerts, or anything possibly important.

EMAILS:
{emails}

Return ONLY JSON:
{{"trash": [{{"id": "<message id>", "from": "<sender>", "subject": "<subject>", "reason": "<why junk>"}}]}}"""


def find_cleanable(cfg: Config, max_results: int = 25) -> list[dict]:
    mails = gcal.list_recent_mail(max_results=max_results, query="in:inbox")
    if not mails:
        return []
    email_block = "\n\n".join(
        f"id={m['id']} FROM: {m['from']}\nSUBJECT: {m['subject']}\nSNIPPET: {m['snippet'][:150]}"
        for m in mails
    )
    result = call_json(
        CLEAN_PROMPT.format(emails=email_block),
        primary=cfg.llm_primary, fallback=cfg.llm_fallback,
    )
    valid_ids = {m["id"] for m in mails}
    return [t for t in result.get("trash", []) if t.get("id") in valid_ids]
