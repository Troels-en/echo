"""Append-only interaction log: every bot interaction → data/events.db (sqlite).

Best-effort by design: a logging failure must NEVER break the main flow, so every
public write swallows its own exceptions. Separate DB from the vector store
(store.db) so a corrupt/locked events.db can't touch RAG.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVENTS_DB = REPO_ROOT / "data" / "events.db"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    db_path = db_path or EVENTS_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    return conn


def init_schema(db_path: Path | None = None) -> None:
    conn = _connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                intent TEXT NOT NULL,
                vault TEXT,
                input_len INTEGER DEFAULT 0,
                source TEXT NOT NULL,
                ref TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
            CREATE INDEX IF NOT EXISTS idx_events_intent ON events(intent);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_events_backfill
                ON events(source, ref) WHERE ref IS NOT NULL;
            """
        )
        conn.commit()
    finally:
        conn.close()


def log_event(
    intent: str,
    vault: str | None = None,
    input_len: int = 0,
    source: str = "text",
    ts: str | None = None,
    ref: str | None = None,
    db_path: Path | None = None,
) -> None:
    """Record one interaction. Best-effort: never raises into the caller."""
    try:
        conn = _connect(db_path)
        try:
            conn.execute(
                "INSERT OR IGNORE INTO events (ts, intent, vault, input_len, source, ref) "
                "VALUES (?,?,?,?,?,?)",
                (ts or _now(), intent, vault, int(input_len or 0), source, ref),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001 — logging must not break the bot
        log.warning("event log failed (ignored): %s", e)


def all_events(db_path: Path | None = None) -> list[dict]:
    try:
        conn = _connect(db_path)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ts, intent, vault, input_len, source, ref FROM events ORDER BY ts"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        log.warning("event read failed (ignored): %s", e)
        return []


def replace_backfill(rows: list[dict], db_path: Path | None = None) -> int:
    """Idempotent backfill: drop prior source='backfill' rows, reinsert.

    rows: [{ts, intent, vault, input_len, ref}]. ref dedups within the backfill.
    Returns number inserted.
    """
    try:
        init_schema(db_path)
        conn = _connect(db_path)
        try:
            conn.execute("DELETE FROM events WHERE source = 'backfill'")
            n = 0
            for r in rows:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO events (ts, intent, vault, input_len, source, ref) "
                    "VALUES (?,?,?,?,'backfill',?)",
                    (r["ts"], r["intent"], r.get("vault"), int(r.get("input_len") or 0), r.get("ref")),
                )
                n += cur.rowcount
            conn.commit()
            return n
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        log.warning("backfill failed (ignored): %s", e)
        return 0
