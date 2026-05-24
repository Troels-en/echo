"""Bridge between Echo (raw capture + RAG) and the SecondBrain LLM-Wiki (curated long-term).

Two directions:
1. index_wiki()      — pull the curated wiki/*.md into Echo's vector store (vault="SecondBrain")
                       so /query and /ask retrieve high-signal synthesized pages alongside raw notes.
2. synthesize_week() — push: stage Echo's recent notes into SecondBrain/raw/, run the
                       `second-brain-ingest` SKILL via a headless `claude -p` agent (skills can
                       only run inside an agent, not as a library), then re-index the new wiki.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import Config
from . import store

log = logging.getLogger(__name__)

SECONDBRAIN_ROOT = Path(os.path.expanduser(os.getenv("SECONDBRAIN_ROOT", "~/SecondBrain"))).resolve()
WIKI = SECONDBRAIN_ROOT / "wiki"
RAW = SECONDBRAIN_ROOT / "raw"
_ECHO_RAW = RAW / "echo"  # where we stage Echo's recent notes for ingestion

_FM = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_TITLE = re.compile(r"^# (.+)$", re.MULTILINE)
_SUMMARY = re.compile(r"^> (.+)$", re.MULTILINE)
_CLAUDE_BIN = "claude"


def _parse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = _FM.match(text)
    body = text[m.end():] if m else text
    title = (_TITLE.search(body).group(1).strip() if _TITLE.search(body) else path.stem)
    summary = (_SUMMARY.search(body).group(1).strip() if _SUMMARY.search(body) else "")
    return {"title": title, "summary": summary, "content": body.strip()}


def index_wiki(cfg: Config, reindex: bool = True) -> int:
    """Index SecondBrain/wiki/*.md into Echo's store.db under vault 'SecondBrain'. Returns count."""
    if not WIKI.exists():
        log.warning("SecondBrain wiki not found at %s", WIKI)
        return 0
    db = cfg.data_dir / "store.db"
    store.init_schema(db)
    n = 0
    for md in WIKI.rglob("*.md"):
        if ".obsidian" in md.parts:
            continue
        abs_path = str(md.resolve())
        if not reindex and store.path_exists(db, abs_path):
            continue
        try:
            p = _parse(md)
            store.upsert_note(db, path=abs_path, vault="SecondBrain",
                              title=p["title"], summary=p["summary"], content=p["content"])
            n += 1
        except Exception as e:
            log.warning("wiki index failed for %s: %s", md, e)
    log.info("indexed %d SecondBrain wiki pages", n)
    return n


def stage_recent_notes(cfg: Config, days: int = 7) -> int:
    """Copy Echo notes created in the last `days` into SecondBrain/raw/echo/ for ingestion.
    Idempotent (skips files already staged). Returns number staged."""
    _ECHO_RAW.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    staged = 0
    for spec in cfg.vaults.values():
        inbox = spec.path / "inbox"
        if not inbox.exists():
            continue
        for md in inbox.glob("*.md"):
            try:
                mtime = datetime.fromtimestamp(md.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    continue
                dest = _ECHO_RAW / f"{spec.name}__{md.name}"
                if dest.exists():
                    continue
                shutil.copy2(md, dest)
                staged += 1
            except Exception as e:
                log.warning("stage failed for %s: %s", md, e)
    log.info("staged %d recent notes into %s", staged, _ECHO_RAW)
    return staged


def run_ingest(timeout: int = 1200) -> str:
    """Run the `second-brain-ingest` skill via a headless claude agent (skills only run
    inside an agent). cwd = SecondBrain so its CLAUDE.md + skill apply. Returns agent stdout."""
    prompt = (
        "Use the second-brain-ingest skill to process every new file under raw/ "
        "(especially raw/echo/) into the wiki: synthesize, deduplicate against existing "
        "pages, and add cross-references. Then briefly report what you added or updated."
    )
    cmd = [
        _CLAUDE_BIN, "-p", prompt,
        "--permission-mode", "acceptEdits",
        "--allowedTools", "Read", "Write", "Edit", "Glob", "Grep", "Bash",
        "--output-format", "text",
    ]
    log.info("running second-brain-ingest via claude -p (cwd=%s)", SECONDBRAIN_ROOT)
    res = subprocess.run(cmd, cwd=str(SECONDBRAIN_ROOT), capture_output=True,
                         text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"ingest agent exit {res.returncode}: {res.stderr[-400:]}")
    return res.stdout.strip()


def synthesize_week(cfg: Config, days: int = 7) -> dict:
    """Full weekly pipeline: stage recent notes → ingest into wiki (agent+skill) → re-index
    wiki into Echo RAG. Returns {staged, ingest_report, wiki_indexed}."""
    staged = stage_recent_notes(cfg, days=days)
    report = ""
    if staged:
        try:
            report = run_ingest()
        except Exception as e:
            log.exception("ingest failed")
            report = f"(Ingest fehlgeschlagen: {e})"
    indexed = index_wiki(cfg, reindex=True)
    return {"staged": staged, "ingest_report": report, "wiki_indexed": indexed}
