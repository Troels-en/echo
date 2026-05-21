"""RAG: embed question → retrieve top-k notes → ask LLM with citations."""
from __future__ import annotations

import logging
from pathlib import Path

from .config import Config
from .llm import call_json, _run_codex, _run_claude, LLMError
from . import store

log = logging.getLogger(__name__)


RAG_PROMPT = """You answer the user's question using the user's own notes AND their open tasks below.

Cite note-based claims with [n] from SOURCES. For task/priority questions, use OPEN TASKS.
Task priority is Eisenhower: 4 = urgent+important (Quadrant 1, "do now"), 3 = important not urgent (Q2, "schedule"), 2 = urgent not important (Q3, "delegate"), 1 = neither (Q4, "drop"). due dates indicate urgency.

If you lack enough info, say so. Do NOT fabricate.

OPEN TASKS (content | priority | due):
{tasks}

NOTE SOURCES:
{sources}

QUESTION: {question}

Return ONLY a JSON object:
{{
  "answer": "<answer in the question's language; cite notes with [n]; for tasks reference them by content>",
  "used_sources": [<note source numbers actually cited>],
  "confidence": <float 0..1>
}}"""


def _format_sources(hits: list[store.Hit], max_chars: int = 600) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        snippet = (h.summary or h.content).strip()[:max_chars]
        lines.append(f"[{i}] vault={h.vault} title={h.title}\n    {snippet}\n    path: {h.path}")
    return "\n\n".join(lines)


def _open_tasks_block(limit: int = 50) -> str:
    """Compact open-task list (content | priority | due) for task/priority questions."""
    try:
        from . import intent
        tasks = intent.fetch_open_tasks(limit=limit)
    except Exception as e:
        log.warning("rag task fetch failed: %s", e)
        return "(Tasks nicht verfügbar)"
    if not tasks:
        return "(keine offenen Tasks)"
    lines = []
    for t in tasks:
        due = ""
        if t.get("due") and t["due"].get("date"):
            due = t["due"]["date"][:10]
        lines.append(f"- {t.get('content','')[:80]} | P{t.get('priority',1)} | {due}")
    return "\n".join(lines)


def answer_question(question: str, cfg: Config, k: int = 8, vault: str | None = None) -> dict:
    hits = store.search(cfg.data_dir / "store.db", question, k=k, vault=vault)
    tasks_block = _open_tasks_block()

    sources = _format_sources(hits) if hits else "(keine Notizen gefunden)"
    prompt = RAG_PROMPT.format(tasks=tasks_block, sources=sources, question=question)
    result = call_json(prompt, primary=cfg.llm_primary, fallback=cfg.llm_fallback)
    result["hits"] = hits
    return result
