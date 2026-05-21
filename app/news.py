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


SYNTHESIS_PROMPT = """You are writing a personal news briefing for an AI-focused founder. SYNTHESIZE — do not just list items.

WINDOW: news from the last {window_days} day(s). Emphasize TODAY and YESTERDAY most.

USER INTERESTS:
{topics}

WHAT YOU KNOW ABOUT THE USER (bias toward these):
{memory}

ITEMS (source | age | title | snippet) — articles + video summaries:
{items}

Write a tight, synthesized briefing in German Markdown:
- GROUP related items into themes/stories. Do NOT echo items 1:1.
- A story covered by MULTIPLE sources = more important → lead with it, note the corroboration.
- Prioritize the most recent (today/yesterday) over older.
- For each story: 1-2 sentences of substance + the single best link as a Markdown link.
- Skip filler, clickbait, off-topic. Quality over quantity — 3 to 6 strong stories is ideal.
- End with one line: any cross-cutting trend you noticed.

Return ONLY JSON: {{"briefing": "<markdown briefing>"}}"""


def _age_label(iso: str) -> str:
    if not iso:
        return "?"
    try:
        dt = datetime.fromisoformat(iso)
        days = (datetime.now(timezone.utc) - dt).days
        return {0: "heute", 1: "gestern"}.get(days, f"vor {days}d")
    except Exception:
        return "?"


def build_news_briefing(cfg: Config, max_video_summaries: int = 2) -> str:
    from . import state as state_mod
    interests = _load_interests()
    topics = interests.get("topics", [])
    feeds = interests.get("rss_feeds", [])
    channels = interests.get("youtube_channels", []) or []
    accelerators = interests.get("accelerators", [])

    # Recency window = time since last news request (min 1 day, default 3 on first run)
    st = state_mod.load()
    last_ts = st.get("last_news_ts")
    window_days = 3
    if last_ts:
        try:
            delta = datetime.now(timezone.utc) - datetime.fromisoformat(last_ts)
            window_days = max(1, min(14, delta.days + 1))
        except Exception:
            pass

    items = _fetch_rss(feeds, max_age_days=window_days) + _fetch_youtube(channels, max_age_days=window_days)
    if not items:
        return "*📰 News-Briefing*\n\n_Keine aktuellen News im Fenster gefunden._"

    # Transcribe the most recent YouTube videos (token-bounded) to enrich synthesis
    yt = sorted([it for it in items if it["type"] == "youtube"],
                key=lambda x: x.get("published", ""), reverse=True)
    for it in yt[:max_video_summaries]:
        transcript = _youtube_transcript(it["link"])
        if transcript:
            try:
                s = call_json(
                    VIDEO_SUMMARY_PROMPT.format(title=it["title"], transcript=transcript),
                    primary=cfg.llm_primary, fallback=cfg.llm_fallback,
                )
                it["summary"] = s.get("summary", "") or it["summary"]
            except Exception as e:
                log.warning("video summary failed: %s", e)

    item_block = "\n".join(
        f"- [{it['source']} | {_age_label(it.get('published',''))}] {it['title']} — {it['summary'][:160]} <{it['link']}>"
        for it in items
    )
    try:
        result = call_json(
            SYNTHESIS_PROMPT.format(
                window_days=window_days,
                topics="\n".join(f"- {t}" for t in topics),
                memory=memory.context() or "(nichts)",
                items=item_block,
            ),
            primary=cfg.llm_primary, fallback=cfg.llm_fallback,
        )
        briefing = result.get("briefing", "").strip()
    except Exception as e:
        log.warning("news synthesis failed: %s", e)
        briefing = "_Synthese fehlgeschlagen._"

    out = [f"*📰 News-Briefing* _(letzte {window_days}d)_", "", briefing]

    if accelerators:
        out += ["", "*🚀 Accelerator / Programme:*"]
        for a in accelerators:
            out.append(f"  • [{a['name']}]({a.get('url','')}) — {a.get('next_deadline','rolling')}")

    # Mark this request time so the next briefing only covers what's new since now
    state_mod.set_key("last_news_ts", datetime.now(timezone.utc).isoformat())
    return "\n".join(out)
