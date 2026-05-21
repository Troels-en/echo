"""Inbox review: surface low-confidence / Misc notes, suggest better vaults, move them."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .config import Config
from .llm import call_json
from . import store

log = logging.getLogger(__name__)


def review_candidates(cfg: Config, limit: int = 10) -> list[dict]:
    """Notes sitting in the default/fallback vault's inbox — these need a real home."""
    default_spec = cfg.vaults.get(cfg.default_vault)
    if not default_spec:
        return []
    inbox = default_spec.path / "inbox"
    if not inbox.exists():
        return []
    out = []
    for md in sorted(inbox.glob("*.md"), reverse=True)[:limit]:
        text = md.read_text(encoding="utf-8", errors="ignore")
        title = md.stem
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        out.append({"path": str(md.resolve()), "title": title, "text": text[:1200]})
    return out


SUGGEST_PROMPT = """Suggest the best vault for this note. The user has these vaults:
{vaults}

NOTE:
\"\"\"{note}\"\"\"

Return ONLY JSON: {{"vault": "<best vault name>", "confidence": <0..1>, "reason": "<short>"}}"""


def suggest_vault(note_text: str, cfg: Config) -> dict:
    vaults = "\n".join(
        f"- {v.name}: {', '.join(v.keywords) if v.keywords else '(fallback)'}"
        for v in cfg.vaults.values()
    )
    try:
        return call_json(
            SUGGEST_PROMPT.format(vaults=vaults, note=note_text[:800]),
            primary=cfg.llm_primary, fallback=cfg.llm_fallback,
        )
    except Exception as e:
        log.warning("suggest failed: %s", e)
        return {"vault": cfg.default_vault, "confidence": 0.0, "reason": ""}


def move_note(src_path: str, target_vault: str, cfg: Config) -> str:
    """Move a note file into target vault's inbox, update the vector store path."""
    spec = cfg.vaults.get(target_vault)
    if not spec:
        raise ValueError(f"unknown vault {target_vault}")
    src = Path(src_path)
    dst_dir = spec.path / "inbox"
    dst_dir.mkdir(exist_ok=True)
    dst = dst_dir / src.name
    shutil.move(str(src), str(dst))

    # update store: re-key path + vault
    db = cfg.data_dir / "store.db"
    try:
        text = dst.read_text(encoding="utf-8", errors="ignore")
        title = dst.stem
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        # remove old row, insert under new path/vault
        import sqlite3, sqlite_vec
        conn = sqlite3.connect(str(db))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        row = conn.execute("SELECT id FROM notes WHERE path=?", (str(src.resolve()),)).fetchone()
        if row:
            conn.execute("DELETE FROM notes_vec WHERE note_id=?", (row[0],))
            conn.execute("DELETE FROM notes WHERE id=?", (row[0],))
            conn.commit()
        conn.close()
        store.upsert_note(db, str(dst.resolve()), target_vault, title, "", text)
    except Exception as e:
        log.warning("store update on move failed: %s", e)
    return str(dst)
