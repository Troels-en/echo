"""SQLite + sqlite-vec store for semantic note search."""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sqlite_vec

from . import embed as embed_mod

log = logging.getLogger(__name__)


@dataclass
class Hit:
    note_id: int
    path: str
    vault: str
    title: str
    summary: str
    content: str
    distance: float


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        d = embed_mod.dim()
        conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                vault TEXT NOT NULL,
                title TEXT,
                summary TEXT,
                content TEXT NOT NULL,
                created TEXT,
                updated TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_notes_vault ON notes(vault);
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_vec USING vec0(
                note_id INTEGER PRIMARY KEY,
                embedding float[{d}]
            );
        """)
        conn.commit()
        log.info("store ready: %s (dim=%d)", db_path, d)
    finally:
        conn.close()


def upsert_note(
    db_path: Path,
    path: str,
    vault: str,
    title: str,
    summary: str,
    content: str,
    created: str | None = None,
) -> int:
    """Insert or replace note. Returns note_id."""
    embed_text = "\n".join(filter(None, [title, summary, content]))
    vec = embed_mod.embed(embed_text).astype(np.float32)

    conn = _connect(db_path)
    try:
        cur = conn.execute("SELECT id FROM notes WHERE path = ?", (path,))
        row = cur.fetchone()
        if row:
            note_id = row[0]
            conn.execute(
                "UPDATE notes SET vault=?, title=?, summary=?, content=?, updated=CURRENT_TIMESTAMP WHERE id=?",
                (vault, title, summary, content, note_id),
            )
            conn.execute("DELETE FROM notes_vec WHERE note_id=?", (note_id,))
        else:
            cur = conn.execute(
                "INSERT INTO notes (path, vault, title, summary, content, created) VALUES (?,?,?,?,?,?)",
                (path, vault, title, summary, content, created),
            )
            note_id = cur.lastrowid

        conn.execute(
            "INSERT INTO notes_vec (note_id, embedding) VALUES (?, ?)",
            (note_id, vec.tobytes()),
        )
        conn.commit()
        return note_id
    finally:
        conn.close()


def search(db_path: Path, query: str, k: int = 10, vault: str | None = None) -> list[Hit]:
    vec = embed_mod.embed(query).astype(np.float32)
    conn = _connect(db_path)
    try:
        if vault:
            sql = """
                SELECT n.id, n.path, n.vault, n.title, n.summary, n.content, v.distance
                FROM notes_vec v
                JOIN notes n ON n.id = v.note_id
                WHERE v.embedding MATCH ? AND k = ? AND n.vault = ?
                ORDER BY v.distance
            """
            rows = conn.execute(sql, (vec.tobytes(), k, vault)).fetchall()
        else:
            sql = """
                SELECT n.id, n.path, n.vault, n.title, n.summary, n.content, v.distance
                FROM notes_vec v
                JOIN notes n ON n.id = v.note_id
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
            """
            rows = conn.execute(sql, (vec.tobytes(), k)).fetchall()
        return [
            Hit(note_id=r[0], path=r[1], vault=r[2], title=r[3] or "",
                summary=r[4] or "", content=r[5] or "", distance=r[6])
            for r in rows
        ]
    finally:
        conn.close()


def count(db_path: Path) -> int:
    conn = _connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    finally:
        conn.close()


def path_exists(db_path: Path, path: str) -> bool:
    conn = _connect(db_path)
    try:
        return conn.execute("SELECT 1 FROM notes WHERE path = ?", (path,)).fetchone() is not None
    finally:
        conn.close()
