"""One-time Google OAuth consent flow. Opens browser, saves token for the bot.

Prereq: place your OAuth client file at secrets/google_credentials.json
(Google Cloud Console > APIs & Services > Credentials > OAuth client ID > Desktop app > Download JSON).

Run:  .venv/bin/python scripts/google_auth.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google_auth_oauthlib.flow import InstalledAppFlow

ROOT = Path(__file__).resolve().parent.parent
CREDS = ROOT / "secrets" / "google_credentials.json"
TOKEN = ROOT / "secrets" / "google_token.json"

SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]


def main() -> None:
    if not CREDS.exists():
        print(f"ERROR: missing {CREDS}")
        print("Download OAuth client JSON from Google Cloud Console and save it there.")
        sys.exit(1)

    flow = InstalledAppFlow.from_client_secrets_file(str(CREDS), SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN.write_text(creds.to_json(), encoding="utf-8")
    print(f"OK. Token saved to {TOKEN}")
    print("The bot can now access Google Calendar + Gmail (read + send).")


if __name__ == "__main__":
    main()
