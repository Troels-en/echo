"""Mirror habit check-ins into Notion (read-only convenience layer). Habits_Vault stays the
source of truth; this just appends a line to a Notion page so the user can read in Notion.

Needs a Notion internal integration (NOTION_TOKEN) + the target page shared with it
(NOTION_HABITS_PAGE). If unset, mirroring is silently skipped — local vault log still happens.
"""
from __future__ import annotations

import logging
import os

import httpx

log = logging.getLogger(__name__)

_TOKEN = os.getenv("NOTION_TOKEN", "").strip()
_PAGE = os.getenv("NOTION_HABITS_PAGE", "").strip()
_API = "https://api.notion.com/v1"
_VERSION = "2022-06-28"


def configured() -> bool:
    return bool(_TOKEN and _PAGE)


def mirror_habit_log(date: str, entry: str) -> bool:
    """Append '<date>: <entry>' as a bullet to the configured Notion page. Best-effort."""
    if not configured():
        return False
    headers = {
        "Authorization": f"Bearer {_TOKEN}",
        "Notion-Version": _VERSION,
        "Content-Type": "application/json",
    }
    body = {
        "children": [{
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": f"{date}: {entry}"[:1900]}}],
            },
        }]
    }
    try:
        r = httpx.patch(f"{_API}/blocks/{_PAGE}/children", json=body, headers=headers, timeout=20.0)
        if r.status_code != 200:
            log.warning("notion mirror %s: %s", r.status_code, r.text[:200])
            return False
        return True
    except Exception as e:
        log.warning("notion mirror failed: %s", e)
        return False
