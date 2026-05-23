"""Visual overview of everything fed into Echo.

Aggregates the vaults + Todoist + (optionally) Google into a single stats dict, then
renders it three ways:
- `build_markdown` → an Obsidian dashboard note (Misc_Vault/Echo_Overview.md)
- `build_telegram` → a concise German summary for the /overview command
- `recent_inputs`  → a flat row list for mirroring into Notion (via the Claude Notion MCP)

Read-only over the vaults: never mutates source notes.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from .config import Config
from . import store

log = logging.getLogger(__name__)

PRIO_EMOJI = {4: "🔴", 3: "🟠", 2: "🟡", 1: "⚪"}
# How an input entered Echo. Source frontmatter doesn't reliably separate voice from text
# (text ingestion also writes source: voice), so everything that isn't an answer or a
# profile fact is just a captured note ("Notiz").
SOURCE_LABEL = {"ask": "Antwort", "self-vault": "Profil"}
DEFAULT_TYPE = "Notiz"
DASHBOARD_NAME = "Echo_Overview.md"


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Parse the leading `---` YAML-ish frontmatter block into a flat str->str dict.

    Hand-rolled (not yaml.safe_load) because Echo writes tags as `[#a, #b]` where the
    space-prefixed `#` would be read as a YAML comment and silently truncate the list.
    """
    if not text.startswith("---"):
        return {}
    lines = text.splitlines()
    out: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        out[key.strip()] = val.strip()
    return out


def _parse_created(fm: dict[str, str], path: Path) -> tuple[datetime, bool]:
    """Resolve a note's creation date and whether it is reliable.

    Reliable = frontmatter `created` or a `YYYY-MM-DD` filename prefix. Falls back to file
    mtime (reliable=False) only so the note still has a sortable date; recency buckets ignore
    unreliable dates so vault-setup mtimes never masquerade as recent activity.
    """
    raw = fm.get("created", "")
    if raw:
        try:
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(), True
        except ValueError:
            pass
    # filename like 2026-05-22-1045-slug.md
    try:
        return datetime.strptime(path.stem[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc).astimezone(), True
    except (ValueError, IndexError):
        pass
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).astimezone(), False


def _title(text: str, path: Path) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def _tags(fm: dict[str, str]) -> list[str]:
    raw = fm.get("tags", "").strip().strip("[]")
    if not raw:
        return []
    return [t.strip().lstrip("#") for t in raw.split(",") if t.strip()]


def _scan_notes(cfg: Config) -> list[dict]:
    """Read every markdown note in the configured vaults into lightweight records."""
    records: list[dict] = []
    for spec in cfg.vaults.values():
        for md in spec.path.rglob("*.md"):
            if md.name == DASHBOARD_NAME:
                continue  # don't count the generated dashboard as an input
            try:
                text = md.read_text(encoding="utf-8")
            except Exception as e:
                log.warning("read failed %s: %s", md, e)
                continue
            fm = _parse_frontmatter(text)
            source = fm.get("source", "unknown")
            created, dated = _parse_created(fm, md)
            try:
                rel = str(md.relative_to(cfg.vault_root))
            except ValueError:
                rel = str(md)
            records.append({
                "path": rel,
                "abs_path": str(md.resolve()),
                "vault": spec.name,
                "title": _title(text, md),
                "created": created,
                "dated": dated,
                "source": source,
                "type": SOURCE_LABEL.get(source, DEFAULT_TYPE),
                "tags": _tags(fm),
                "importance": fm.get("importance"),
                "is_answer": source == "ask",
            })
    return records


def _todoist_stats() -> dict:
    """Open + completed Todoist task stats. Best-effort; degrades to zeros on failure."""
    from .todoist import _hdr, API
    stats = {"open": 0, "due_today": 0, "overdue": 0, "by_priority": {1: 0, 2: 0, 3: 0, 4: 0},
             "completed_7d": None, "available": False}
    junk = ("This is a task", "Drag it", "getting started", "Kickstart", "help center")
    today = datetime.now().astimezone().date()
    try:
        with httpx.Client(timeout=15.0) as c:
            r = c.get(f"{API}/tasks", headers=_hdr())
            r.raise_for_status()
            tasks = r.json().get("results", [])
    except Exception as e:
        log.warning("todoist open tasks failed: %s", e)
        return stats

    stats["available"] = True
    for t in tasks:
        if any(j in t.get("content", "") for j in junk):
            continue
        stats["open"] += 1
        pri = t.get("priority", 1)
        stats["by_priority"][pri] = stats["by_priority"].get(pri, 0) + 1
        due = t.get("due") or {}
        if due.get("date"):
            try:
                d = datetime.fromisoformat(due["date"][:10]).date()
                if d == today:
                    stats["due_today"] += 1
                elif d < today:
                    stats["overdue"] += 1
            except ValueError:
                pass

    # completed in the last 7 days (best-effort; endpoint shape varies)
    try:
        since = (datetime.now().astimezone() - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
        with httpx.Client(timeout=15.0) as c:
            r = c.get(f"{API}/tasks/completed/by_completion_date",
                      headers=_hdr(), params={"since": since,
                                              "until": datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S")})
            if r.status_code == 200:
                stats["completed_7d"] = len(r.json().get("items", r.json().get("results", [])))
    except Exception as e:
        log.warning("todoist completed failed: %s", e)
    return stats


def _event_stats(cfg: Config) -> dict:
    """Count calendar events today + next 7 days. Best-effort; empty if Google unconfigured."""
    from . import gcal
    out = {"available": False, "today": 0, "week": 0}
    if not gcal.is_configured():
        return out
    try:
        now = datetime.now().astimezone()
        week_end = now + timedelta(days=7)
        cal = gcal._calendar()
        res = cal.events().list(
            calendarId="primary", timeMin=now.isoformat(), timeMax=week_end.isoformat(),
            singleEvents=True, orderBy="startTime",
        ).execute()
        items = res.get("items", [])
        out["available"] = True
        out["week"] = len(items)
        today = now.date()
        for e in items:
            start = e["start"].get("dateTime", e["start"].get("date", ""))
            try:
                if datetime.fromisoformat(start).date() == today:
                    out["today"] += 1
            except ValueError:
                pass
    except Exception as e:
        log.warning("event stats failed: %s", e)
    return out


def aggregate(cfg: Config, now: datetime | None = None) -> dict:
    """Compute the full overview stats dict from vaults + Todoist + Google."""
    now = now or datetime.now().astimezone()
    cutoff_7 = now - timedelta(days=7)
    cutoff_30 = now - timedelta(days=30)

    notes = _scan_notes(cfg)
    per_vault: dict[str, dict] = {}
    for name in cfg.vaults:
        per_vault[name] = {"total": 0, "last7": 0, "last30": 0}
    by_source: dict[str, int] = {}
    answers = {"total": 0, "by_importance": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}}

    for n in notes:
        v = per_vault.setdefault(n["vault"], {"total": 0, "last7": 0, "last30": 0})
        v["total"] += 1
        if n["dated"] and n["created"] >= cutoff_30:
            v["last30"] += 1
        if n["dated"] and n["created"] >= cutoff_7:
            v["last7"] += 1
        by_source[n["source"]] = by_source.get(n["source"], 0) + 1
        if n["is_answer"]:
            answers["total"] += 1
            try:
                imp = int(n["importance"]) if n["importance"] else 3
            except (ValueError, TypeError):
                imp = 3
            answers["by_importance"][imp] = answers["by_importance"].get(imp, 0) + 1

    recent_7 = sorted([n for n in notes if n["dated"] and n["created"] >= cutoff_7],
                      key=lambda n: n["created"], reverse=True)
    recent_30 = sorted([n for n in notes if n["dated"] and n["created"] >= cutoff_30],
                       key=lambda n: n["created"], reverse=True)

    try:
        indexed = store.count(cfg.data_dir / "store.db")
    except Exception:
        indexed = None

    return {
        "generated_at": now,
        "total_notes": len(notes),
        "indexed_notes": indexed,
        "per_vault": per_vault,
        "by_source": by_source,
        "answers": answers,
        "recent_7": recent_7,
        "recent_7_count": len(recent_7),
        "recent_30_count": len(recent_30),
        "tasks": _todoist_stats(),
        "events": _event_stats(cfg),
    }


def recent_inputs(cfg: Config, days: int = 30) -> list[dict]:
    """Flat row list for the Notion mirror. Stable `key` = relative note path (idempotency)."""
    cutoff = datetime.now().astimezone() - timedelta(days=days)
    rows = []
    for n in _scan_notes(cfg):
        if not n["dated"] or n["created"] < cutoff:
            continue
        rows.append({
            "key": n["path"],
            "title": n["title"],
            "vault": n["vault"],
            "date": n["created"].strftime("%Y-%m-%d"),
            "type": n["type"],
            "importance": int(n["importance"]) if (n["importance"] or "").isdigit() else None,
            "tags": n["tags"],
        })
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows


def build_markdown(stats: dict) -> str:
    """Render the Obsidian dashboard note (plain markdown tables, German)."""
    gen = stats["generated_at"].strftime("%Y-%m-%d %H:%M")
    out = [
        "---",
        f"created: {stats['generated_at'].isoformat(timespec='seconds')}",
        "source: echo-overview",
        "tags: [echo/overview, dashboard]",
        "---",
        "",
        "# Echo — Übersicht",
        "",
        f"> Automatisch erzeugt von Echo · Stand {gen}. Nicht manuell bearbeiten.",
        "",
        "## Auf einen Blick",
        "",
        "| Kennzahl | Wert |",
        "| --- | --- |",
        f"| Notizen gesamt | {stats['total_notes']} |",
        f"| Neu (7 Tage) | {stats['recent_7_count']} |",
        f"| Neu (30 Tage) | {stats['recent_30_count']} |",
        f"| Antworten (#echo/answer) | {stats['answers']['total']} |",
    ]
    if stats.get("indexed_notes") is not None:
        out.append(f"| Im Vektor-Index | {stats['indexed_notes']} |")
    t = stats["tasks"]
    if t["available"]:
        out.append(f"| Tasks offen | {t['open']} |")
        out.append(f"| davon heute fällig | {t['due_today']} |")
        out.append(f"| davon überfällig | {t['overdue']} |")
        if t.get("completed_7d") is not None:
            out.append(f"| Erledigt (7 Tage) | {t['completed_7d']} |")
    e = stats["events"]
    if e["available"]:
        out.append(f"| Termine heute | {e['today']} |")
        out.append(f"| Termine (7 Tage) | {e['week']} |")

    out += ["", "## Notizen pro Vault", "", "| Vault | Gesamt | 7 Tage | 30 Tage |", "| --- | --- | --- | --- |"]
    for name, v in sorted(stats["per_vault"].items(), key=lambda kv: kv[1]["total"], reverse=True):
        if v["total"] == 0 and v["last30"] == 0:
            continue
        out.append(f"| {name} | {v['total']} | {v['last7']} | {v['last30']} |")

    if stats["answers"]["total"]:
        out += ["", "## Antworten nach Wichtigkeit", "", "| Wichtigkeit | Anzahl |", "| --- | --- |"]
        for imp in (5, 4, 3, 2, 1):
            c = stats["answers"]["by_importance"].get(imp, 0)
            if c:
                out.append(f"| {'⭐' * imp} ({imp}) | {c} |")

    out += ["", "## Zuletzt erfasst (7 Tage)", ""]
    if stats["recent_7"]:
        out += ["| Datum | Titel | Vault | Typ |", "| --- | --- | --- | --- |"]
        for n in stats["recent_7"][:30]:
            stem = Path(n["path"]).stem
            title = n["title"].replace("|", "\\|")
            out.append(f"| {n['created'].strftime('%Y-%m-%d')} | [[{stem}\\|{title}]] | {n['vault']} | {n['type']} |")
    else:
        out.append("_Nichts in den letzten 7 Tagen._")
    out.append("")
    return "\n".join(out)


def build_telegram(stats: dict) -> str:
    """Concise German summary for the /overview command (Markdown, < 4000 chars)."""
    gen = stats["generated_at"].strftime("%d.%m. %H:%M")
    lines = [f"📊 *Echo Übersicht* · _{gen}_", ""]
    lines.append(f"📝 *{stats['total_notes']}* Notizen · *{stats['recent_7_count']}* neu (7T) · *{stats['recent_30_count']}* (30T)")
    if stats["answers"]["total"]:
        lines.append(f"💡 *{stats['answers']['total']}* Antworten gespeichert")

    t = stats["tasks"]
    if t["available"]:
        extra = []
        if t["due_today"]:
            extra.append(f"{t['due_today']} heute")
        if t["overdue"]:
            extra.append(f"{t['overdue']} überfällig")
        suffix = f" ({', '.join(extra)})" if extra else ""
        line = f"✅ *{t['open']}* Tasks offen{suffix}"
        if t.get("completed_7d") is not None:
            line += f" · {t['completed_7d']} erledigt (7T)"
        lines.append(line)
    e = stats["events"]
    if e["available"]:
        lines.append(f"📅 *{e['today']}* Termine heute · {e['week']} diese Woche")

    active = [(name, v) for name, v in stats["per_vault"].items() if v["last30"] > 0]
    active.sort(key=lambda kv: kv[1]["last30"], reverse=True)
    if active:
        lines += ["", "*Aktivste Vaults (30T):*"]
        for name, v in active[:5]:
            lines.append(f"  • {name}: {v['last30']}")

    if stats["recent_7"]:
        lines += ["", "*Zuletzt:*"]
        for n in stats["recent_7"][:5]:
            lines.append(f"  • _{n['created'].strftime('%d.%m.')}_ {n['title'][:45]} ({n['vault']})")

    lines += ["", "_Volle Übersicht: `Misc_Vault/Echo_Overview.md`_"]
    return "\n".join(lines)


def write_dashboard(cfg: Config, stats: dict | None = None) -> Path:
    """Aggregate (if needed) and write the Obsidian dashboard note. Returns its path."""
    stats = stats or aggregate(cfg)
    target_vault = cfg.vaults.get("Misc_Vault") or next(iter(cfg.vaults.values()))
    path = target_vault.path / DASHBOARD_NAME
    path.write_text(build_markdown(stats), encoding="utf-8")
    log.info("wrote overview dashboard: %s", path)
    return path
