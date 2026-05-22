# Session 07 — Visual overview · HANDOFF

Branch: `feat/07-visual-overview` · base `793920f`

## What this builds
A visual overview of everything fed into Echo, in three surfaces:
1. **Obsidian dashboard** — `Misc_Vault/Echo_Overview.md`, regenerated on each run (markdown tables, German).
2. **Telegram** — `/overview` command → concise summary + refreshes the dashboard.
3. **Notion mirror** — an "Echo Inputs" database (via the Claude Notion MCP) holding recent inputs, idempotent on note path.

## Files
### Added
- `app/overview.py` — aggregation + the three renderers. Public API:
  - `aggregate(cfg, now=None) -> dict` — scans `cfg.vaults` + Todoist + Google into stats.
  - `build_markdown(stats) -> str` / `build_telegram(stats) -> str` — renderers.
  - `write_dashboard(cfg, stats=None) -> Path` — writes `Misc_Vault/Echo_Overview.md`.
  - `recent_inputs(cfg, days=30) -> list[dict]` — flat rows for Notion (`key` = relative note path).
- `scripts/export_overview.py` — CLI: refresh dashboard + write `data/notion_sync.json` payload.

### Changed (SHARED FILE — merge carefully)
- `app/bot.py` — three additive edits only:
  1. import line (line ~26): appended `, overview as overview_mod` to the existing `from . import ...`.
  2. new `async def cmd_overview(...)` inserted directly above `def main()`.
  3. registered `app.add_handler(CommandHandler("overview", cmd_overview))` after the `inbox` handler.
  No existing lines modified. Low merge risk (additions at distinct points).

## New deps
None. Reuses `httpx` (Todoist), existing `gcal`, `store`, `config`.

## How to test (exact commands)
```bash
cd /Users/troelsenigk/echo-wt/07-visual-overview
.venv/bin/python -c "import app.bot, app.overview; print('compile OK')"
# full run: writes the dashboard + payload, prints counts
.venv/bin/python scripts/export_overview.py
# inspect the dashboard
cat /Users/troelsenigk/Misc_Vault/Echo_Overview.md
```
Verified on real data: 512 notes, 44 new (7d), 217 (30d); live Todoist (46 open, 20 overdue, 4 done/7d) and Google Calendar (1 today, 5 this week) both pulled successfully.

`/overview` in Telegram is wired but NOT exercised live (hard rule: never start the production bot). The handler calls the same `aggregate` + `write_dashboard` + `build_telegram` path that the CLI verifies.

## Notion mirror — DONE (evidence)
- DB: **Echo Inputs** → https://www.notion.so/e39ac470db6d4722b941fdbe81f3c5ed
  (under a new top-level **Echo** page; data source `1e389cb1-8ad8-4bb1-a02e-fac895d0ce0a`).
- Schema: `Title` (title), `Key` (text, idempotency key = relative note path), `Vault` (select),
  `Date` (date), `Type` (select: Notiz/Antwort/Profil), `Importance` (number), `Tags` (text, comma-joined).
- Inserted **10 real rows** (the 10 most recent inputs) — confirmed by the create response (10 page URLs).

### How the sync is triggered (and why it's split)
The live bot process has **no MCP access**, so it cannot push to Notion directly. The split:
1. `scripts/export_overview.py` (runnable by the bot host / cron) writes `data/notion_sync.json`.
2. A **Claude Code session with the Notion MCP** reads that payload and upserts:
   - search/fetch existing `Key` values in the DB → insert only rows whose `key` is absent
     (update in place if present). `key` = relative note path = stable, so re-runs never duplicate.
This is the documented, repeatable procedure; the initial 10-row load above was done this way.

## Assumptions made
- "The vaults" = `cfg.vaults` (the routed vaults in `config/vaults.yml`), authoritative over a disk glob.
  Note `vaults.yml` is a SHARED symlink; other sessions added `Self_Vault` / `Engineering_Playbook` during the run — picked up automatically.
- Recency (7/30d) counts only notes with a **reliable** date (frontmatter `created` or `YYYY-MM-DD` filename).
  Undated/legacy notes fall back to file mtime for sorting but are excluded from recency buckets, so
  one-time vault-setup mtimes never masquerade as recent activity.
- Input "type": `source: ask` → Antwort, `source: self-vault` → Profil, everything else → Notiz
  (frontmatter `source` does not reliably separate voice from typed text).
- Dashboard lives permanently at `Misc_Vault/Echo_Overview.md` (a product, not a test artifact); it
  excludes itself from the scan and is overwritten idempotently each run.
- Todoist "completed (7d)" is best-effort via `/tasks/completed/by_completion_date`; degrades to omitted on failure.
- German user-facing strings throughout.

## Known gaps / follow-ups
- No automated scheduler for the Notion push (manual/agent-triggered today). A `job_queue.run_daily`
  hook could call `export_overview` + an MCP-side sync, but the bot can't reach the MCP, so a small
  cron + Claude routine is the natural home.
- `/overview` not run against live Telegram (by design). Recommend a quick `/overview` smoke test after merge.
- `data/notion_sync.json` is written into the shared (symlinked) `data/` dir; gitignored, harmless.

## Merge notes for orchestrator
- Only shared-file edit is `app/bot.py` (3 additive hunks listed above). `app/overview.py` and
  `scripts/export_overview.py` are new and conflict-free.
- No `pyproject.toml` / `uv.lock` changes.
- No changes to `.env`, config, or other sessions' files.
