"""Google Calendar + Gmail client. Uses OAuth token from secrets/google_token.json."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
TOKEN = ROOT / "secrets" / "google_token.json"
CREDS = ROOT / "secrets" / "google_credentials.json"
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.readonly",
]
TZ = ZoneInfo("Europe/Berlin")


class GoogleError(RuntimeError):
    pass


def is_configured() -> bool:
    return TOKEN.exists()


@lru_cache(maxsize=1)
def _creds() -> Credentials:
    if not TOKEN.exists():
        raise GoogleError(
            "Google not authorized. Run: .venv/bin/python scripts/google_auth.py"
        )
    creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)
    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN.write_text(creds.to_json(), encoding="utf-8")
    return creds


def _calendar():
    return build("calendar", "v3", credentials=_creds(), cache_discovery=False)


def _gmail():
    return build("gmail", "v1", credentials=_creds(), cache_discovery=False)


def _header(payload: dict, name: str) -> str:
    for h in payload.get("headers", []):
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _extract_body(payload: dict) -> str:
    import base64
    def _decode(data: str) -> str:
        try:
            return base64.urlsafe_b64decode(data.encode()).decode("utf-8", "ignore")
        except Exception:
            return ""
    mime = payload.get("mimeType", "")
    body = payload.get("body", {})
    if mime == "text/plain" and body.get("data"):
        return _decode(body["data"])
    for part in payload.get("parts", []) or []:
        text = _extract_body(part)
        if text:
            return text
    if body.get("data"):
        return _decode(body["data"])
    return ""


def trash_mail(message_id: str) -> None:
    """Move a message to trash (recoverable, not permanent delete)."""
    _gmail().users().messages().trash(userId="me", id=message_id).execute()
    log.info("gmail trashed: %s", message_id)


def list_recent_mail(max_results: int = 8, query: str = "is:unread category:primary") -> list[dict]:
    """Return recent messages: from, subject, snippet, body (truncated)."""
    svc = _gmail()
    listing = svc.users().messages().list(
        userId="me", q=query, maxResults=max_results,
    ).execute()
    out = []
    for ref in listing.get("messages", []):
        msg = svc.users().messages().get(userId="me", id=ref["id"], format="full").execute()
        payload = msg.get("payload", {})
        out.append({
            "id": ref["id"],
            "from": _header(payload, "From"),
            "subject": _header(payload, "Subject"),
            "snippet": msg.get("snippet", ""),
            "body": _extract_body(payload)[:1500],
        })
    return out


def create_event(
    summary: str,
    start: datetime,
    end: datetime | None = None,
    description: str = "",
    location: str = "",
) -> dict:
    """start/end naive or tz-aware datetimes. Defaults to 1h duration."""
    if start.tzinfo is None:
        start = start.replace(tzinfo=TZ)
    if end is None:
        end = start + timedelta(hours=1)
    if end.tzinfo is None:
        end = end.replace(tzinfo=TZ)

    body = {
        "summary": summary,
        "description": description,
        "location": location,
        "start": {"dateTime": start.isoformat(), "timeZone": "Europe/Berlin"},
        "end": {"dateTime": end.isoformat(), "timeZone": "Europe/Berlin"},
    }
    log.info("gcal create_event: %s @ %s", summary, start.isoformat())
    ev = _calendar().events().insert(calendarId="primary", body=body).execute()
    return {
        "id": ev["id"],
        "htmlLink": ev.get("htmlLink", ""),
        "summary": ev.get("summary", summary),
        "start": start.isoformat(),
    }


def list_upcoming(max_results: int = 10) -> list[dict]:
    now = datetime.now(TZ).isoformat()
    res = (
        _calendar().events().list(
            calendarId="primary", timeMin=now, maxResults=max_results,
            singleEvents=True, orderBy="startTime",
        ).execute()
    )
    out = []
    for e in res.get("items", []):
        start = e["start"].get("dateTime", e["start"].get("date"))
        out.append({"summary": e.get("summary", "(kein Titel)"), "start": start})
    return out
