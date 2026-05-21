"""Build a personalized morning briefing from calendar + tasks + recent notes."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from .config import Config
from . import gcal
from .todoist import _hdr, API

log = logging.getLogger(__name__)
TZ = ZoneInfo("Europe/Berlin")

PRIO_EMOJI = {4: "🔴", 3: "🟠", 2: "🟡", 1: "⚪"}


def _today_events() -> list[dict]:
    if not gcal.is_configured():
        return []
    try:
        now = datetime.now(TZ)
        end = now.replace(hour=23, minute=59, second=59)
        cal = gcal._calendar()
        res = cal.events().list(
            calendarId="primary",
            timeMin=now.isoformat(),
            timeMax=end.isoformat(),
            singleEvents=True, orderBy="startTime",
        ).execute()
        out = []
        for e in res.get("items", []):
            start = e["start"].get("dateTime", e["start"].get("date"))
            out.append({"summary": e.get("summary", "(kein Titel)"), "start": start})
        return out
    except Exception as e:
        log.warning("briefing events failed: %s", e)
        return []


def _open_tasks_today() -> tuple[list[dict], list[dict], int, list[dict]]:
    """Return (due_today, overdue_top, overdue_total, high_priority_no_date)."""
    try:
        with httpx.Client(timeout=10.0) as c:
            r = c.get(f"{API}/tasks", headers=_hdr())
            r.raise_for_status()
            tasks = r.json().get("results", [])
    except Exception as e:
        log.warning("briefing tasks failed: %s", e)
        return [], [], 0, []

    today = datetime.now(TZ).date()
    due_today, overdue, high = [], [], []
    junk = ("This is a task", "Drag it", "getting started", "Kickstart", "help center")
    for t in tasks:
        if any(j in t.get("content", "") for j in junk):
            continue
        due_obj = t.get("due")
        dated = False
        if due_obj and due_obj.get("date"):
            try:
                d = datetime.fromisoformat(due_obj["date"][:10]).date()
                dated = True
                if d == today:
                    due_today.append(t)
                elif d < today:
                    overdue.append(t)
                continue
            except ValueError:
                pass
        if not dated and t.get("priority", 1) >= 3:
            high.append(t)

    by_prio = lambda x: x.get("priority", 1)
    due_today.sort(key=by_prio, reverse=True)
    overdue.sort(key=by_prio, reverse=True)
    high.sort(key=by_prio, reverse=True)
    return due_today, overdue[:3], len(overdue), high[:5]


def _fmt_time(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%H:%M")
    except Exception:
        return "ganztägig"


def _overnight_important_mail(cfg: Config) -> list[str]:
    """One-liners for action-needed mails since yesterday. Empty if none/unconfigured."""
    if not gcal.is_configured():
        return []
    try:
        from . import mailtriage
        result = mailtriage.triage(cfg, max_results=8, query="newer_than:1d in:inbox")
        out = []
        for d in result.get("digest", []):
            if d.get("needs_action"):
                out.append(f"⚡ {d.get('from','?')[:25]} — {d.get('summary','')[:60]}")
        return out[:5]
    except Exception as e:
        log.warning("overnight mail failed: %s", e)
        return []


def build_briefing(cfg: Config, include_mail: bool = True) -> str:
    """Daily-short briefing: calendar + tasks + overdue + overnight important mail."""
    now = datetime.now(TZ)
    greeting = "Guten Morgen"
    if now.hour >= 18:
        greeting = "Guten Abend"
    elif now.hour >= 12:
        greeting = "Hallo"

    lines = [f"*{greeting}!* 📋  {now.strftime('%A, %d.%m.')}", ""]

    events = _today_events()
    if events:
        lines.append("*📅 Heute im Kalender:*")
        for e in events:
            lines.append(f"  • {_fmt_time(e['start'])}  {e['summary']}")
        lines.append("")
    else:
        lines.append("📅 Keine Termine heute.")
        lines.append("")

    due_today, overdue_top, overdue_total, high = _open_tasks_today()

    def _clip(s: str, n: int = 60) -> str:
        return s if len(s) <= n else s[:n] + "…"

    if due_today:
        lines.append(f"*⏰ Heute fällig ({len(due_today)}):*")
        for t in due_today:
            lines.append(f"  {PRIO_EMOJI.get(t.get('priority', 1), '')} {_clip(t['content'])}")
        lines.append("")
    if overdue_total:
        lines.append(f"*🔁 Überfällig ({overdue_total}) — Top 3:*")
        for t in overdue_top:
            lines.append(f"  {PRIO_EMOJI.get(t.get('priority', 1), '')} {_clip(t['content'])}")
        lines.append("")
    if high:
        lines.append("*🎯 Wichtig (ohne Datum):*")
        for t in high:
            lines.append(f"  {PRIO_EMOJI.get(t.get('priority', 1), '')} {_clip(t['content'])}")
        lines.append("")

    if include_mail:
        mails = _overnight_important_mail(cfg)
        if mails:
            lines.append("*📧 Wichtige Mails über Nacht:*")
            lines += [f"  {m}" for m in mails]
            lines.append("")

    if not events and not due_today and not overdue_total and not high:
        lines.append("Nichts Dringendes. Freier Tag. 🌤️")

    lines.append("_Sag 'news' für Entwicklungen in KI/Gründerszene._")
    return "\n".join(lines)
