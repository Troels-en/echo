#!/usr/bin/env python
"""Refresh the Echo overview: write the Obsidian dashboard + emit the Notion sync payload.

Usage:
    .venv/bin/python scripts/export_overview.py [--days 30] [--json data/notion_sync.json]

The Notion mirror itself runs through the Claude Notion MCP (the bot process has no MCP
access). This script produces the rows; a Claude Code session then upserts them into the
"Echo Inputs" database, matching on the stable `key` (relative note path) so re-runs never
duplicate. See HANDOFF.md for the exact sync procedure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config
from app import overview as ov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30, help="recency window for the Notion payload")
    ap.add_argument("--json", default="data/notion_sync.json", help="where to write the payload")
    args = ap.parse_args()

    cfg = Config.load()
    stats = ov.aggregate(cfg)
    dash = ov.write_dashboard(cfg, stats)
    rows = ov.recent_inputs(cfg, days=args.days)

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"dashboard: {dash}")
    print(f"notion payload: {out} ({len(rows)} rows, last {args.days}d)")
    print(f"notes total: {stats['total_notes']} | 7d: {stats['recent_7_count']} | 30d: {stats['recent_30_count']}")


if __name__ == "__main__":
    main()
