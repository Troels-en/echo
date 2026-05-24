"""Tiny in-process registry of running background jobs so Echo can answer
"wie lange noch / status / bist du fertig" with the truth instead of guessing.

Single-process bot → a module-level dict is enough. Best-effort, never raises.
"""
from __future__ import annotations

import threading
import time

_jobs: dict[int, dict] = {}
_lock = threading.Lock()
_seq = 0

# Rough typical durations per job kind, for the ETA hint.
_ETA = {
    "research": "~2-3 Min",
    "mail-research": "~2-3 Min",
    "synthesize": "~8-12 Min",
    "podcast": "~1-2 Min",
}


def start(kind: str, label: str = "") -> int:
    """Register a running job. Returns an id to pass to finish()."""
    global _seq
    with _lock:
        _seq += 1
        jid = _seq
        _jobs[jid] = {"kind": kind, "label": label, "started": time.time()}
    return jid


def finish(jid: int) -> None:
    with _lock:
        _jobs.pop(jid, None)


def active() -> list[dict]:
    with _lock:
        return list(_jobs.values())


def status_text() -> str:
    """German summary of what's running, with elapsed time + typical ETA."""
    js = active()
    if not js:
        return "✅ Gerade läuft nichts im Hintergrund."
    lines = ["⏳ Läuft gerade:"]
    now = time.time()
    for j in js:
        mins = int((now - j["started"]) // 60)
        eta = _ETA.get(j["kind"], "")
        lbl = f" – {j['label'][:50]}" if j.get("label") else ""
        since = f"seit {mins} Min" if mins else "gerade gestartet"
        tail = f", typisch {eta}" if eta else ""
        lines.append(f"• {j['kind']}{lbl} ({since}{tail})")
    return "\n".join(lines)
