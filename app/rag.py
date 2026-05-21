"""RAG: embed question → retrieve top-k notes → ask LLM with citations."""
from __future__ import annotations

import logging
from pathlib import Path

from .config import Config
from .llm import call_json, _run_codex, _run_claude, LLMError
from . import store

log = logging.getLogger(__name__)


RAG_PROMPT = """You answer the user's question using ONLY the user's own notes below as context. You must cite each claim with [n] where n is the source number from the SOURCES list.

If the notes do not contain enough information to answer, say so explicitly and suggest what to capture next. Do NOT fabricate.

SOURCES:
{sources}

QUESTION: {question}

Return ONLY a JSON object:
{{
  "answer": "<your answer with inline [n] citations, in the same language as the question>",
  "used_sources": [<list of source numbers actually cited>],
  "confidence": <float 0..1>
}}"""


def _format_sources(hits: list[store.Hit], max_chars: int = 600) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        snippet = (h.summary or h.content).strip()[:max_chars]
        lines.append(f"[{i}] vault={h.vault} title={h.title}\n    {snippet}\n    path: {h.path}")
    return "\n\n".join(lines)


def answer_question(question: str, cfg: Config, k: int = 8, vault: str | None = None) -> dict:
    hits = store.search(cfg.data_dir / "store.db", question, k=k, vault=vault)
    if not hits:
        return {
            "answer": "Keine relevanten Notes gefunden.",
            "used_sources": [],
            "confidence": 0.0,
            "hits": [],
        }

    sources = _format_sources(hits)
    prompt = RAG_PROMPT.format(sources=sources, question=question)
    result = call_json(prompt, primary=cfg.llm_primary, fallback=cfg.llm_fallback)
    result["hits"] = hits
    return result
