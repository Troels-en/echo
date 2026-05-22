"""General-question answering: quick LLM answer, escalate to web research when warranted.
Returns a structured result the bot can reply with and save to the vault.
"""
from __future__ import annotations

import logging

from .config import Config
from .llm import call_json, research_web, LLMError

log = logging.getLogger(__name__)


TRIAGE_PROMPT = """You answer a general question for the user (NOT from their personal notes — general knowledge or research).

NOW (Europe/Berlin): {now}

QUESTION:
\"\"\"{question}\"\"\"

VAULTS the answer could later be filed under (pick the best fit, or "Misc_Vault"):
{vault_list}

First decide: can you answer well from your own knowledge, or does this need LIVE web research
(current events, prices, latest versions, "best X right now", anything time-sensitive or needing sources)?

Return ONLY a JSON object:
{{
  "needs_web": <true if live web research would materially improve the answer, else false>,
  "answer": "<your best answer in the user's language (German if they wrote German). Empty string if needs_web is true — a researcher will answer instead.>",
  "title": "<short title for the saved note, max 8 words>",
  "tags": ["<2 to 4 short obsidian tags>"],
  "vault": "<best-fit vault name from the list, or Misc_Vault>",
  "importance": <integer 1..5: 1=trivial/throwaway, 3=useful reference, 5=high-value keep>
}}"""


RESEARCH_PROMPT = """Research this question using web search and give a thorough, well-sourced answer.
Answer in the user's language (German if the question is German). Be concise but complete.
Include key sources/links inline. Do NOT ask follow-up questions — just answer.

QUESTION: {question}"""


def smart_answer(question: str, cfg: Config) -> dict:
    """Return {answer, title, tags, vault, importance, used_web}."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).astimezone().strftime("%A %Y-%m-%d %H:%M")
    vault_list = "\n".join(f"- {v.name}" for v in cfg.vaults.values())

    triage = call_json(
        TRIAGE_PROMPT.format(now=now, question=question, vault_list=vault_list),
        primary=cfg.llm_primary, fallback=cfg.llm_fallback,
    )

    needs_web = bool(triage.get("needs_web"))
    answer = (triage.get("answer") or "").strip()
    used_web = False

    if needs_web or not answer:
        try:
            answer = research_web(
                RESEARCH_PROMPT.format(question=question),
                model=cfg.ask_model, timeout=cfg.ask_web_timeout,
            )
            used_web = True
        except LLMError as e:
            log.warning("web research failed, falling back to triage answer: %s", e)
            if not answer:
                answer = "Konnte keine Antwort erzeugen (Recherche fehlgeschlagen)."

    vault = triage.get("vault") or cfg.default_vault
    if vault not in cfg.vaults:
        vault = cfg.default_vault

    importance = triage.get("importance", 3)
    try:
        importance = max(1, min(5, int(importance)))
    except (TypeError, ValueError):
        importance = 3

    return {
        "answer": answer,
        "title": (triage.get("title") or question[:50]).strip(),
        "tags": triage.get("tags", []) or [],
        "vault": vault,
        "importance": importance,
        "used_web": used_web,
    }
