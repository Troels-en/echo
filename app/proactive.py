"""Proactive nudges — Echo reaches out unprompted, instead of only reacting.

Morning focus + evening habit check-in, grounded in the Habits_Vault Master Routine
("Bad-day minimum"). Evening-checkin replies get logged into Habits_Vault/06_Logs/.
The Habits_Vault is the source of truth; a Notion mirror is a separate follow-up.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

HABITS_VAULT = Path(os.path.expanduser(os.getenv("HABITS_VAULT_ROOT", "~/Habits_Vault"))).resolve()
_ROUTINE = HABITS_VAULT / "01_Routines" / "Master Routine.md"
_LOGS = HABITS_VAULT / "06_Logs"

_FALLBACK_MORNING = "Rolladen auf → Wasser + Supplements → 90 Sek Cat-Cow + Bird-Dog → Frühstück."
_FALLBACK_EVENING = "Handy quer durchs Zimmer face-down DND → Zähne + Brackets → Licht aus."


def _bad_day_minimum() -> tuple[str, str]:
    """Pull the Morning/Evening 'bad-day minimum' lines from the Master Routine."""
    morning, evening = _FALLBACK_MORNING, _FALLBACK_EVENING
    try:
        text = _ROUTINE.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"\*\*Morning[^:]*:\*\*\s*(.+)", text)
        e = re.search(r"\*\*Evening[^:]*:\*\*\s*(.+)", text)
        if m:
            morning = m.group(1).strip()
        if e:
            evening = e.group(1).strip()
    except Exception as ex:
        log.warning("could not read Master Routine: %s", ex)
    return morning, evening


def morning_text() -> str:
    morning, _ = _bad_day_minimum()
    return (
        "☀️ *Guten Morgen.*\n"
        f"Bad-day-Minimum: {morning}\n\n"
        "Worauf willst du dich heute fokussieren? (1 Sache reicht.)"
    )


def evening_text() -> str:
    _, evening = _bad_day_minimum()
    return (
        "🌙 *Tagesabschluss.*\n"
        f"Minimum: {evening}\n\n"
        "Wie lief dein Tag? Kurz durchgehen:\n"
        "1) Schlaf-Onset (1=sofort…5=>45min)\n"
        "2) Morgen-Energie (1–10)\n"
        "3) Fokus am Tagesende (1–10)\n"
        "4) Handy in der Küche? (J/N)\n"
        "5) Koffein-Cutoff 14:00 gehalten? (J/N)\n"
        "6) Mittags gelaufen? (J/N)\n\n"
        "Schreib einfach frei zurück (z.B. „2, 7, 6, J, J, N") — ich logge es in deinen Habits-Vault."
    )


def log_checkin(answer: str) -> Path:
    """Append an evening check-in answer to today's habit log in Habits_Vault/06_Logs."""
    _LOGS.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    path = _LOGS / f"{today}-checkin.md"
    ts = datetime.now().strftime("%H:%M")
    if not path.exists():
        path.write_text(
            f"---\ntags: [habit-log, checkin]\ncreated: {datetime.now().isoformat(timespec='seconds')}\n---\n\n"
            f"# Habit Check-in {today}\n\n",
            encoding="utf-8",
        )
    with path.open("a", encoding="utf-8") as f:
        f.write(f"- **{ts}** {answer.strip()}\n")
    log.info("logged habit check-in to %s", path)
    return path
