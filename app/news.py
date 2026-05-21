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
                    "type": "article",
                    "source": feed.get("name", ""),
                    "title": getattr(e, "title", ""),
                    "summary": getattr(e, "summary", "")[:300],
                    "link": getattr(e, "link", ""),
                    "published": pub.isoformat() if pub else "",
                })
        except Exception as ex:
            log.warning("rss fetch failed %s: %s", feed.get("url"), ex)
    return items


def _fetch_youtube(channels: list[dict], per_channel: int = 3, max_age_days: int = 7) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    items = []
    for ch in channels:
        cid = ch.get("channel_id")
        if not cid:
            continue
        url = f"https://www.youtube.com/feeds/videos.xml?channel_id={cid}"
        try:
            parsed = feedparser.parse(url)
            for e in parsed.entries[:per_channel]:
                pub = None
                if getattr(e, "published_parsed", None):
                    pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
                if pub and pub < cutoff:
                    continue
                items.append({
                    "type": "youtube",
                    "source": ch.get("name", "YouTube"),
                    "title": getattr(e, "title", ""),
                    "summary": "",
                    "link": getattr(e, "link", ""),
                    "published": pub.isoformat() if pub else "",
                })
        except Exception as ex:
            log.warning("youtube fetch failed %s: %s", cid, ex)
    return items


def _youtube_transcript(video_url: str, max_chars: int = 6000) -> str:
    """Fetch auto-captions via yt-dlp. Returns truncated plain text, '' on failure."""
    import glob
    import os
    import re
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "%(id)s")
        try:
            subprocess.run(
                ["yt-dlp", "--skip-download", "--write-auto-subs",
                 "--sub-lang", "en.*,en,de", "--sub-format", "vtt", "-o", out, video_url],
                capture_output=True, text=True, timeout=60,
            )
        except Exception as e:
            log.warning("yt-dlp transcript failed: %s", e)
            return ""
        vtts = glob.glob(os.path.join(td, "*.vtt"))
        if not vtts:
            return ""
        raw = open(vtts[0], encoding="utf-8", errors="ignore").read()
        seen, lines = set(), []
        for ln in raw.splitlines():
            if "-->" in ln or not ln.strip() or ln.startswith(("WEBVTT", "Kind:", "Language:")):
                continue
            ln = re.sub(r"<[^>]+>", "", ln).strip()
            if ln and ln not in seen:
                seen.add(ln)
                lines.append(ln)
        return " ".join(lines)[:max_chars]


VIDEO_SUMMARY_PROMPT = """Summarize this YouTube transcript for an AI-focused founder. MAX 20 words. Concrete takeaway only, no filler, no "the video discusses".

TITLE: {title}
TRANSCRIPT: {transcript}

Return ONLY JSON: {{"summary": "<max 20 words>"}}"""


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


def build_news_briefing(cfg: Config, max_video_summaries: int = 2) -> str:
    interests = _load_interests()
    topics = interests.get("topics", [])
    feeds = interests.get("rss_feeds", [])
    channels = interests.get("youtube_channels", []) or []
    accelerators = interests.get("accelerators", [])

    items = _fetch_rss(feeds) + _fetch_youtube(channels)
    lines = ["*📰 News-Briefing*", ""]

    if items:
        numbered = "\n".join(
            f"[{i+1}] ({it['type']}/{it['source']}) {it['title']} — {it['summary'][:120]}"
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
            # Transcribe + summarize only the top relevant YouTube videos (token-bounded)
            video_summaries: dict[int, str] = {}
            vids_done = 0
            for r in relevant[:8]:
                idx = r.get("n", 0) - 1
                if 0 <= idx < len(items) and items[idx]["type"] == "youtube" and vids_done < max_video_summaries:
                    transcript = _youtube_transcript(items[idx]["link"])
                    if transcript:
                        try:
                            s = call_json(
                                VIDEO_SUMMARY_PROMPT.format(title=items[idx]["title"], transcript=transcript),
                                primary=cfg.llm_primary, fallback=cfg.llm_fallback,
                            )
                            video_summaries[idx] = s.get("summary", "")
                            vids_done += 1
                        except Exception as e:
                            log.warning("video summary failed: %s", e)

            lines.append("*🤖 Für dich relevant:*")
            for r in relevant[:8]:
                idx = r.get("n", 0) - 1
                if 0 <= idx < len(items):
                    it = items[idx]
                    icon = "🎥" if it["type"] == "youtube" else "📄"
                    extra = video_summaries.get(idx) or r.get("why", "")
                    extra = f" _{extra}_" if extra else ""
                    lines.append(f"  {icon} [{it['title'][:70]}]({it['link']}) ({it['source']}){extra}")
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
