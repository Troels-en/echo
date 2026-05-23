"""Persistent personalization memory: durable facts about the user.

Facts are learned passively during note ingest (people, preferences, projects,
patterns) and injected into prompts so the assistant gets more personal over time.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
MEM_FILE = REPO_ROOT / "data" / "memory.json"

VALID_TYPES = {"person", "preference", "project", "pattern", "fact"}
MAX_FACTS = 120
CONTEXT_CAP = 40  # facts injected into prompts


def _load() -> list[dict]:
    if MEM_FILE.exists():
        try:
            return json.loads(MEM_FILE.read_text())
        except Exception as e:
            log.warning("memory load failed: %s", e)
    return []


def _save(facts: list[dict]) -> None:
    MEM_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEM_FILE.write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _alloc_id(facts: list[dict]) -> int:
    return max((int(f.get("id", 0)) for f in facts), default=0) + 1


def _ensure_ids(facts: list[dict]) -> bool:
    """Assign stable integer ids to any legacy facts missing one. Returns True if changed."""
    changed = False
    nxt = _alloc_id(facts)
    for f in facts:
        if not f.get("id"):
            f["id"] = nxt
            nxt += 1
            changed = True
    return changed


def add_facts(new: list[dict]) -> int:
    """new: [{text, type}]. Dedupes by normalized text. Returns count added."""
    facts = _load()
    changed = _ensure_ids(facts)
    existing = {_norm(f["text"]) for f in facts}
    added = 0
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    for item in new:
        text = (item.get("text") or "").strip()
        ftype = item.get("type", "fact")
        if not text or _norm(text) in existing:
            continue
        if ftype not in VALID_TYPES:
            ftype = "fact"
        facts.append({"id": _alloc_id(facts), "text": text, "type": ftype, "created": now})
        existing.add(_norm(text))
        added += 1
    if len(facts) > MAX_FACTS:
        facts = facts[-MAX_FACTS:]
    if added or changed:
        _save(facts)
    return added


def context(cap: int = CONTEXT_CAP) -> str:
    """Compact memory string for prompt injection. Empty if no facts."""
    facts = _load()
    if not facts:
        return ""
    recent = facts[-cap:]
    by_type: dict[str, list[str]] = {}
    for f in recent:
        by_type.setdefault(f["type"], []).append(f["text"])
    lines = []
    for t in ("person", "preference", "project", "pattern", "fact"):
        if t in by_type:
            lines.append(f"{t}: " + "; ".join(by_type[t]))
    return "\n".join(lines)


def all_facts() -> list[dict]:
    return _load()


def forget(substring: str) -> int:
    facts = _load()
    sub = _norm(substring)
    kept = [f for f in facts if sub not in _norm(f["text"])]
    removed = len(facts) - len(kept)
    if removed:
        _save(kept)
    return removed


TYPE_ORDER = ("person", "preference", "project", "pattern", "fact")
TYPE_LABELS = {
    "person": "Personen",
    "preference": "Vorlieben",
    "project": "Projekte",
    "pattern": "Muster",
    "fact": "Fakten",
}
TYPE_ICONS = {"person": "👤", "preference": "❤️", "project": "🚀", "pattern": "🔁", "fact": "·"}


def get_fact(fact_id: int) -> dict | None:
    for f in _load():
        if int(f.get("id", 0)) == int(fact_id):
            return f
    return None


def edit_fact(fact_id: int, new_text: str) -> bool:
    """Replace a fact's text by id. Returns True if found and changed."""
    new_text = new_text.strip()
    if not new_text:
        return False
    facts = _load()
    _ensure_ids(facts)
    hit = next((f for f in facts if int(f.get("id", 0)) == int(fact_id)), None)
    if hit is None:
        return False
    hit["text"] = new_text
    hit["edited"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    _save(facts)
    return True


def delete_fact(fact_id: int) -> bool:
    """Delete a single fact by id. Returns True if removed."""
    facts = _load()
    kept = [f for f in facts if int(f.get("id", 0)) != int(fact_id)]
    if len(kept) == len(facts):
        return False
    _save(kept)
    return True


def merge_facts(ids: list[int]) -> dict | None:
    """Merge facts into the first id (keep its text/type), delete the rest. Returns the survivor."""
    facts = _load()
    _ensure_ids(facts)
    ids = [int(i) for i in ids]
    if len(ids) < 2:
        return None
    survivor = next((f for f in facts if int(f["id"]) == ids[0]), None)
    if survivor is None:
        return None
    drop = set(ids[1:])
    kept = [f for f in facts if int(f["id"]) not in drop]
    if len(kept) == len(facts):
        return None
    _save(kept)
    return survivor


def list_structured() -> dict[str, list[dict]]:
    """Facts grouped by type, newest-first within each group, exact-duplicate texts collapsed."""
    facts = _load()
    if _ensure_ids(facts):
        _save(facts)
    seen: set[str] = set()
    grouped: dict[str, list[dict]] = {}
    for f in sorted(facts, key=lambda x: x.get("created", ""), reverse=True):
        key = _norm(f["text"])
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(f.get("type", "fact"), []).append(f)
    return {t: grouped[t] for t in TYPE_ORDER if t in grouped}


def find_duplicates() -> list[list[dict]]:
    """Groups of facts whose normalized text is identical (post-dedup these are rare,
    but near-identical wording can slip through). Each group has 2+ members."""
    facts = _load()
    _ensure_ids(facts)
    by_norm: dict[str, list[dict]] = {}
    for f in facts:
        by_norm.setdefault(_norm(f["text"]), []).append(f)
    return [g for g in by_norm.values() if len(g) > 1]


def export_markdown(target_dir: Path, filename: str = "Memory_Overview.md") -> Path:
    """Write/refresh a human-readable overview note into an Obsidian vault. Returns the path."""
    grouped = list_structured()
    total = sum(len(v) for v in grouped.values())
    now = datetime.now(timezone.utc).astimezone()
    lines = [
        "---",
        f'generated: {now.isoformat(timespec="seconds")}',
        "tags: [echo/memory]",
        "---",
        "",
        "# Was Echo über mich weiß",
        "",
        f"_Stand: {now.strftime('%Y-%m-%d %H:%M')} · {total} Fakten_",
        "",
        "> Bearbeiten in Telegram: `/editmemory <id> <neuer text>` · "
        "Löschen: `/forget <id>` · Neu erzeugen: `/memorymd`",
        "",
    ]
    for t, items in grouped.items():
        lines.append(f"## {TYPE_ICONS.get(t, '·')} {TYPE_LABELS.get(t, t)}")
        lines.append("")
        for f in items:
            since = (f.get("created") or "")[:10]
            suffix = f"  <sub>seit {since}</sub>" if since else ""
            lines.append(f"- **[{f.get('id')}]** {f['text']}{suffix}")
        lines.append("")
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote memory overview: %s (%d facts)", path, total)
    return path
