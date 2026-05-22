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

# Recency policy
DEFAULT_WINDOW_HOURS = 36  # default coverage window when there's no prior request
MIN_WINDOW_HOURS = 12
MAX_WINDOW_HOURS = 14 * 24
HARD_CAP_DAYS = 14  # never surface anything older than this, even flagged


def _load_interests() -> dict:
    path = INTERESTS_FILE
    if not path.exists():
        path = REPO_ROOT / "config" / "interests.example.yml"
    if not path.exists():
        return {"topics": [], "rss_feeds": [], "accelerators": []}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _classify_recency(pub: datetime | None, window_hours: int) -> str:
    """recent = within window · older = dated but past window · unknown = no date."""
    if pub is None:
        return "unknown"
    if pub < datetime.now(timezone.utc) - timedelta(hours=window_hours):
        return "older"
    return "recent"


def _fetch_rss(feeds: list[dict], window_hours: int, per_feed: int = 6) -> list[dict]:
    # Keep items within a generous hard cap; recency vs. the window is *flagged*,
    # not silently dropped, so the briefing can be honest about stale/undated items.
    hard_cut = datetime.now(timezone.utc) - timedelta(days=HARD_CAP_DAYS)
    items = []
    for feed in feeds:
        try:
            parsed = feedparser.parse(feed["url"])
            for e in parsed.entries[:per_feed]:
                pub = None
                if getattr(e, "published_parsed", None):
                    pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
                if pub and pub < hard_cut:
                    continue
                items.append({
                    "type": "article",
                    "source": feed.get("name", ""),
                    "title": getattr(e, "title", ""),
                    "summary": getattr(e, "summary", "")[:300],
                    "link": getattr(e, "link", ""),
                    "published": pub.isoformat() if pub else "",
                    "recency": _classify_recency(pub, window_hours),
                })
        except Exception as ex:
            log.warning("rss fetch failed %s: %s", feed.get("url"), ex)
    return items


def _fetch_youtube(channels: list[dict], window_hours: int, per_channel: int = 3) -> list[dict]:
    hard_cut = datetime.now(timezone.utc) - timedelta(days=HARD_CAP_DAYS)
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
                if pub and pub < hard_cut:
                    continue
                items.append({
                    "type": "youtube",
                    "source": ch.get("name", "YouTube"),
                    "title": getattr(e, "title", ""),
                    "summary": "",
                    "link": getattr(e, "link", ""),
                    "published": pub.isoformat() if pub else "",
                    "recency": _classify_recency(pub, window_hours),
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

WINDOW: this briefing covers the {window_label}. Emphasize the most recent items most.

USER INTERESTS:
{topics}

WHAT YOU KNOW ABOUT THE USER (use this to explain personal relevance):
{memory}

ITEMS — each line: [source | DATE | recency-flag] title — snippet <link>
- recency-flag "⚠älter" = older than the window; "⚠Datum unklar" = no reliable date.
{items}

Write a tight, synthesized briefing in German Markdown. For EVERY story:
- GROUP related items into themes/stories. Do NOT echo items 1:1.
- State the publication DATE of the source(s) inline (e.g. "(22.05.)"). This is mandatory — the reader must see how recent each story is.
- If a story rests on a "⚠älter" or "⚠Datum unklar" item, say so explicitly (e.g. "(Datum unklar)") instead of presenting it as fresh news.
- A story covered by MULTIPLE sources = more important → lead with it, note the corroboration.
- 1-2 sentences of substance + the single best link as a Markdown link.
- If the item is technical or uses jargon, add a short plain-German explanation a non-expert understands ("Einfach gesagt: …").
- Add one line "→ Für dich:" tying the story to the user's interests/context above (why it matters to THEM specifically). Skip only if there is genuinely no connection.
- Skip filler, clickbait, off-topic. Quality over quantity — 3 to 6 strong stories is ideal.
- End with one line: any cross-cutting trend you noticed.

Return ONLY JSON: {{"briefing": "<markdown briefing>"}}"""


ACCEL_PROMPT = """For each accelerator/founder program below, write ONE short German sentence explaining why it is relevant to THIS user, based on their interests and context. Be concrete, no filler.

USER INTERESTS:
{topics}

WHAT YOU KNOW ABOUT THE USER:
{memory}

PROGRAMS:
{programs}

Return ONLY JSON: {{"reasons": {{"<program name>": "<one German sentence>", ...}}}}"""


def _date_label(iso: str) -> str:
    """Absolute publication date, e.g. '22.05.'. 'Datum unklar' when missing."""
    if not iso:
        return "Datum unklar"
    try:
        return datetime.fromisoformat(iso).strftime("%d.%m.")
    except Exception:
        return "Datum unklar"


_RECENCY_FLAG = {"older": " ⚠älter", "unknown": " ⚠Datum unklar"}


def _window_label(window_hours: int) -> str:
    """Human window for the header: 'letzte 36h' under 48h, else 'letzte Nd'."""
    if window_hours < 48:
        return f"letzte {window_hours}h"
    return f"letzte {round(window_hours / 24)}d"


def _accelerator_reasons(accelerators: list[dict], topics: list[str], mem_ctx: str, cfg: Config) -> dict:
    """One German relevance sentence per program. Falls back to {} (caller uses notes)."""
    if not accelerators:
        return {}
    programs = "\n".join(
        f"- {a['name']}: {a.get('note', '')}".rstrip(": ") for a in accelerators
    )
    try:
        result = call_json(
            ACCEL_PROMPT.format(
                topics="\n".join(f"- {t}" for t in topics),
                memory=mem_ctx,
                programs=programs,
            ),
            primary=cfg.llm_primary, fallback=cfg.llm_fallback,
        )
        reasons = result.get("reasons", {})
        return reasons if isinstance(reasons, dict) else {}
    except Exception as e:
        log.warning("accelerator reasons failed: %s", e)
        return {}


def build_news_briefing(cfg: Config, max_video_summaries: int = 2) -> str:
    from . import state as state_mod
    interests = _load_interests()
    topics = interests.get("topics", [])
    feeds = interests.get("rss_feeds", [])
    channels = interests.get("youtube_channels", []) or []
    accelerators = interests.get("accelerators", [])

    # Recency window = time since last news request, in hours (default 36h on first run).
    st = state_mod.load()
    last_ts = st.get("last_news_ts")
    window_hours = DEFAULT_WINDOW_HOURS
    if last_ts:
        try:
            delta = datetime.now(timezone.utc) - datetime.fromisoformat(last_ts)
            window_hours = max(MIN_WINDOW_HOURS, min(MAX_WINDOW_HOURS, round(delta.total_seconds() / 3600)))
        except Exception:
            pass

    items = _fetch_rss(feeds, window_hours) + _fetch_youtube(channels, window_hours)
    if not items:
        return f"*📰 News-Briefing* _({_window_label(window_hours)})_\n\n_Keine aktuellen News im Fenster gefunden._"

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

    mem_ctx = memory.context() or "(nichts)"
    item_block = "\n".join(
        f"- [{it['source']} | {_date_label(it.get('published',''))}{_RECENCY_FLAG.get(it.get('recency',''), '')}] "
        f"{it['title']} — {it['summary'][:160]} <{it['link']}>"
        for it in items
    )
    try:
        result = call_json(
            SYNTHESIS_PROMPT.format(
                window_label=_window_label(window_hours),
                topics="\n".join(f"- {t}" for t in topics),
                memory=mem_ctx,
                items=item_block,
            ),
            primary=cfg.llm_primary, fallback=cfg.llm_fallback,
        )
        briefing = result.get("briefing", "").strip()
    except Exception as e:
        log.warning("news synthesis failed: %s", e)
        briefing = "_Synthese fehlgeschlagen._"

    out = [f"*📰 News-Briefing* _({_window_label(window_hours)})_", "", briefing]

    if accelerators:
        reasons = _accelerator_reasons(accelerators, topics, mem_ctx, cfg)
        out += ["", "*🚀 Accelerator / Programme:*"]
        for a in accelerators:
            out.append(f"  • [{a['name']}]({a.get('url','')}) — {a.get('next_deadline','rolling')}")
            reason = reasons.get(a["name"]) or a.get("note", "")
            if reason:
                out.append(f"     ↳ _{reason}_")

    # Mark this request time so the next briefing only covers what's new since now
    state_mod.set_key("last_news_ts", datetime.now(timezone.utc).isoformat())
    return "\n".join(out)
