"""News briefing: fetch RSS, LLM-filter against user interests, list accelerator deadlines."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import feedparser
import yaml

from .config import Config, REPO_ROOT
from .llm import call_json
from . import memory

log = logging.getLogger(__name__)

INTERESTS_FILE = REPO_ROOT / "config" / "interests.yml"


def _load_interests() -> dict:
    path = INTERESTS_FILE
    if not path.exists():
        path = REPO_ROOT / "config" / "interests.example.yml"
    if not path.exists():
        return {"topics": [], "rss_feeds": [], "accelerators": []}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _fetch_rss(feeds: list[dict], per_feed: int = 6, max_age_days: int = 4) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    items = []
    for feed in feeds:
        try:
            parsed = feedparser.parse(feed["url"])
            for e in parsed.entries[:per_feed]:
                pub = None
                if getattr(e, "published_parsed", None):
                    pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
                if pub and pub < cutoff:
                    continue
                items.append({
                    "source": feed.get("name", ""),
                    "title": getattr(e, "title", ""),
                    "summary": getattr(e, "summary", "")[:300],
                    "link": getattr(e, "link", ""),
                    "published": pub.isoformat() if pub else "",
                })
        except Exception as ex:
            log.warning("rss fetch failed %s: %s", feed.get("url"), ex)
    return items


FILTER_PROMPT = """Filter news items for relevance to the user. Keep only genuinely relevant ones.

USER TOPICS OF INTEREST:
{topics}

WHAT YOU KNOW ABOUT THE USER:
{memory}

NEWS ITEMS (numbered):
{items}

Return ONLY a JSON object:
{{
  "relevant": [
    {{"n": <item number>, "why": "<3-6 word reason>"}}
  ]
}}

Keep max 8, ranked by relevance. Drop generic/clickbait/off-topic. Prefer concrete developments over opinion."""


def build_news_briefing(cfg: Config) -> str:
    interests = _load_interests()
    topics = interests.get("topics", [])
    feeds = interests.get("rss_feeds", [])
    accelerators = interests.get("accelerators", [])

    items = _fetch_rss(feeds)
    lines = ["*📰 News-Briefing*", ""]

    if items:
        numbered = "\n".join(
            f"[{i+1}] ({it['source']}) {it['title']} — {it['summary'][:120]}"
            for i, it in enumerate(items)
        )
        try:
            result = call_json(
                FILTER_PROMPT.format(
                    topics="\n".join(f"- {t}" for t in topics),
                    memory=memory.context() or "(nichts)",
                    items=numbered,
                ),
                primary=cfg.llm_primary, fallback=cfg.llm_fallback,
            )
            relevant = result.get("relevant", [])
        except Exception as e:
            log.warning("news filter failed: %s", e)
            relevant = [{"n": i + 1, "why": ""} for i in range(min(6, len(items)))]

        if relevant:
            lines.append("*🤖 Für dich relevant:*")
            for r in relevant[:8]:
                idx = r.get("n", 0) - 1
                if 0 <= idx < len(items):
                    it = items[idx]
                    why = f" _{r['why']}_" if r.get("why") else ""
                    lines.append(f"  • [{it['title'][:70]}]({it['link']}) ({it['source']}){why}")
            lines.append("")
    else:
        lines.append("_Keine aktuellen News gefunden (RSS leer/unerreichbar)._")
        lines.append("")

    if accelerators:
        lines.append("*🚀 Accelerator / Programme:*")
        for a in accelerators:
            dl = a.get("next_deadline", "rolling")
            lines.append(f"  • [{a['name']}]({a.get('url','')}) — {dl}")
        lines.append("")
        lines.append("_Deadlines kuratiert — vor Bewerbung auf Website prüfen._")

    return "\n".join(lines)
