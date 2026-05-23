"""Progress statistics + XP for Echo.

Reads the interaction log (data/events.db, see app.events), the memory facts
(data/memory.json), and vault note frontmatter to answer "how do I use Echo and
how does it evolve". Backfills an initial history from existing note/memory
timestamps so stats are meaningful before live logging accumulates.

Rendering uses matplotlib's headless Agg backend (no display needed).
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from . import events as events_mod  # noqa: E402

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
MEM_FILE = REPO_ROOT / "data" / "memory.json"

# XP weights per intent. Tunable; documented in HANDOFF. Higher = more effortful/valuable.
XP_WEIGHTS = {
    "complete": 15,  # closing a task = real-world follow-through
    "note": 10,
    "ask": 8,
    "event": 8,
    "mail": 5,
    "query": 3,
    "news": 2,
}
XP_DEFAULT = 1
STREAK_BONUS_PER_DAY = 5  # XP added per day of the current streak

# matplotlib palette: one accent + grays, no emojis.
ACCENT = "#2563eb"
ACCENT2 = "#94a3b8"
GRID = "#e5e7eb"

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
_CREATED = re.compile(r"^created:\s*(.+?)\s*$", re.MULTILINE)


def _parse_ts(raw: str) -> datetime | None:
    raw = raw.strip().strip("'\"")
    for fmt in (None, "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            if fmt is None:
                return datetime.fromisoformat(raw)
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _note_created(md_path: Path) -> datetime | None:
    try:
        head = md_path.read_text(encoding="utf-8")[:1500]
    except Exception:
        return None
    m = _FRONTMATTER.search(head)
    block = m.group(1) if m else head
    cm = _CREATED.search(block)
    if cm:
        return _parse_ts(cm.group(1))
    return None


def _iter_vault_notes(cfg):
    for name, spec in cfg.vaults.items():
        for md in spec.path.rglob("*.md"):
            yield name, md


def backfill(cfg) -> int:
    """Seed events.db with an initial history from existing notes + memory facts.

    Idempotent (see events.replace_backfill). Note events carry source='backfill';
    live interactions are source in {voice,text}.
    """
    rows: list[dict] = []
    for vault, md in _iter_vault_notes(cfg):
        created = _note_created(md)
        if not created:
            continue
        rows.append(
            {
                "ts": created.isoformat(timespec="seconds"),
                "intent": "note",
                "vault": vault,
                "input_len": 0,
                "ref": str(md.resolve()),
            }
        )
    return events_mod.replace_backfill(rows)


def _memory_facts() -> list[dict]:
    if not MEM_FILE.exists():
        return []
    try:
        return json.loads(MEM_FILE.read_text())
    except Exception as e:
        log.warning("memory read failed: %s", e)
        return []


def _day(ts: str) -> date | None:
    dt = _parse_ts(ts)
    return dt.date() if dt else None


def _current_streak(days: set[date]) -> int:
    if not days:
        return 0
    today = date.today()
    # Streak counts back from today; if nothing today, allow it to anchor on the
    # most recent active day so a streak isn't "lost" mid-day before any event.
    anchor = today if today in days else max(days)
    streak = 0
    d = anchor
    while d in days:
        streak += 1
        d -= timedelta(days=1)
    return streak


def compute(cfg) -> dict:
    """All progress stats as a plain dict (JSON-serialisable)."""
    evs = events_mod.all_events()
    total = len(evs)

    by_intent = Counter(e["intent"] for e in evs)
    by_source = Counter(e["source"] for e in evs)

    per_day: dict[date, int] = defaultdict(int)
    active_days: set[date] = set()
    for e in evs:
        d = _day(e["ts"])
        if d:
            per_day[d] += 1
            active_days.add(d)

    # Notes per vault (from live + backfilled note events).
    notes_per_vault = Counter(e["vault"] for e in evs if e["intent"] == "note" and e["vault"])

    # Memory-fact growth over time (cumulative by creation day).
    facts = _memory_facts()
    fact_days: dict[date, int] = defaultdict(int)
    for f in facts:
        d = _day(f.get("created", ""))
        if d:
            fact_days[d] += 1

    streak = _current_streak(active_days)
    xp = sum(XP_WEIGHTS.get(e["intent"], XP_DEFAULT) for e in evs)
    xp += STREAK_BONUS_PER_DAY * streak

    span_start = min(active_days) if active_days else None
    span_end = max(active_days) if active_days else None

    return {
        "total": total,
        "by_intent": dict(by_intent.most_common()),
        "by_source": dict(by_source.most_common()),
        "per_day": {d.isoformat(): n for d, n in sorted(per_day.items())},
        "notes_per_vault": dict(notes_per_vault.most_common()),
        "memory_facts": len(facts),
        "memory_growth": {d.isoformat(): n for d, n in sorted(fact_days.items())},
        "active_days": len(active_days),
        "streak": streak,
        "xp": xp,
        "span": [span_start.isoformat() if span_start else None,
                 span_end.isoformat() if span_end else None],
    }


def level_for_xp(xp: int) -> tuple[int, int, int]:
    """Quadratic level curve: level L needs 100*L^2 XP. Returns (level, into, span)."""
    level = int((xp / 100) ** 0.5) if xp > 0 else 0
    cur_floor = 100 * level * level
    next_floor = 100 * (level + 1) * (level + 1)
    return level + 1, xp - cur_floor, next_floor - cur_floor


def format_summary(stats: dict) -> str:
    """Telegram Markdown summary (German, matches app voice). No emojis-as-icons abuse."""
    level, into, span = level_for_xp(stats["xp"])
    lines = [
        "📊 *Echo Fortschritt*",
        "",
        f"⭐ *XP {stats['xp']}* · Level {level} ({into}/{span})",
        f"🔥 Streak: {stats['streak']} Tag(e) · {stats['active_days']} aktive Tage",
        f"💬 {stats['total']} Interaktionen · 🧠 {stats['memory_facts']} Fakten gemerkt",
        "",
        "*Nach Intent:*",
    ]
    for intent, n in stats["by_intent"].items():
        lines.append(f"  • {intent}: {n}")
    if stats["notes_per_vault"]:
        lines.append("")
        lines.append("*Notizen pro Vault:*")
        for vault, n in list(stats["notes_per_vault"].items())[:8]:
            lines.append(f"  • {vault}: {n}")
    s0, s1 = stats["span"]
    if s0:
        lines.append("")
        lines.append(f"_Zeitraum {s0} → {s1}_")
    return "\n".join(lines)


def render_chart(stats: dict, out_path: Path) -> Path | None:
    """3-panel PNG: interactions/day, intent breakdown, cumulative growth.

    Returns the path on success, None if there's no data to plot.
    """
    if stats["total"] == 0:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    fig.suptitle("Echo — Fortschritt", fontsize=14, fontweight="bold")

    # 1) Interactions per day.
    ax = axes[0]
    per_day = stats["per_day"]
    if per_day:
        days = [datetime.fromisoformat(d) for d in per_day]
        ax.bar(days, list(per_day.values()), color=ACCENT, width=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        fig.autofmt_xdate(rotation=45)
    ax.set_title("Interaktionen pro Tag", fontsize=11)
    ax.set_ylabel("Anzahl")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    # 2) Intent breakdown (horizontal).
    ax = axes[1]
    bi = stats["by_intent"]
    if bi:
        labels = list(bi.keys())
        vals = list(bi.values())
        ax.barh(labels, vals, color=ACCENT)
        ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v}", va="center", fontsize=9, color="#374151")
    ax.set_title("Verteilung nach Intent", fontsize=11)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    # 3) Cumulative growth: interactions vs memory facts.
    ax = axes[2]
    if per_day:
        days_sorted = sorted(datetime.fromisoformat(d) for d in per_day)
        cum, run = [], 0
        for d in days_sorted:
            run += per_day[d.date().isoformat()]
            cum.append(run)
        ax.plot(days_sorted, cum, color=ACCENT, marker="o", markersize=3, label="Interaktionen (kum.)")
    mg = stats["memory_growth"]
    if mg:
        md_days = sorted(datetime.fromisoformat(d) for d in mg)
        cum, run = [], 0
        for d in md_days:
            run += mg[d.date().isoformat()]
            cum.append(run)
        ax.plot(md_days, cum, color=ACCENT2, marker="s", markersize=3, label="Memory-Fakten (kum.)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.set_title("Wachstum über Zeit", fontsize=11)
    ax.set_ylabel("Kumuliert")
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if ax.has_data():
        ax.legend(fontsize=8, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
