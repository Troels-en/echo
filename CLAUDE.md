# Echo — Claude Code guide

Echo is a voice-first personal assistant: Telegram voice/text → local Whisper → LLM intent router → notes (Obsidian Markdown), tasks (Todoist), calendar + email (Google), RAG Q&A, daily/news briefings, and a memory layer.

**If you are Claude Code and the user just cloned this repo, run the Onboarding Flow below before anything else.**

---

## Onboarding Flow (run on first open)

Detect first-run by checking: does `.env` exist? does `config/vaults.yml` exist? is `TELEGRAM_BOT_TOKEN` filled? If any are missing/placeholder, greet the user and walk them through these steps **one at a time**, confirming each before moving on. Use the AskUserQuestion tool for choices.

### Step 1 — Dependencies
Check these exist; if not, tell the user the install command:
- `uv` (Python). `whisper-cli` / `whisper-server` (`brew install whisper-cpp`). `ffmpeg` (`brew install ffmpeg`).
Then run `uv sync`.

### Step 2 — Config files
```bash
cp .env.example .env
cp config/vaults.example.yml config/vaults.yml
cp config/interests.example.yml config/interests.yml
```
(The app falls back to the `.example` files if these are missing, so this is optional but recommended for customization.)

### Step 3 — Telegram bot (required)
Tell the user: open [@BotFather](https://t.me/BotFather) in Telegram → `/newbot` → copy the token.
Then: open `.env`, replace `TELEGRAM_BOT_TOKEN=` with their token. Offer to open `.env` for them. **Never ask them to paste the token into chat — it goes in `.env`.**

### Step 4 — LLM choice
Ask via AskUserQuestion:
- **CLI mode** (free, no API cost): uses local `codex` and/or `claude` CLI subscriptions. Set `LLM_MODE=cli`. Verify the CLI(s) are logged in.
- **API mode** (pay per token): set `LLM_MODE=api` + `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `.env`.

### Step 5 — Whisper model
Ask: speed vs quality.
- `small` (fast, multilingual) — `bash scripts/download_model.sh small`
- `large-v3-turbo-q5_0` (best quality, slower) — `bash scripts/download_model.sh large-v3-turbo-q5_0`
Set `WHISPER_MODEL` in `.env` accordingly.

### Step 6 — Vaults
Ask: where do their Obsidian vaults live (`VAULT_ROOT` in `.env`, default home dir)? Then help them edit `config/vaults.yml` — vault names matching `*_Vault` folders, keywords, and which Todoist project each maps to.

### Step 7 — Optional integrations (ask which they want)
- **Todoist:** add `TODOIST_API_TOKEN` to `.env`, run `python scripts/setup_todoist.py`.
- **Google Calendar + Gmail:** create OAuth Desktop credentials in Google Cloud Console (enable Calendar API + Gmail API), save as `secrets/google_credentials.json`, run `python scripts/google_auth.py`. Add the user's email as a test user on the OAuth consent screen.

### Step 8 — News briefing (ask preferences)
Ask the user:
1. **Do you want a news briefing at all?** (If no, skip — they can still use everything else.)
2. **What topics?** (e.g. AI, startups, a specific industry) → write to `topics` in `config/interests.yml`.
3. **Which sources?** Suggest RSS feeds for their topics → add to `rss_feeds`.
Tell them they can later just say "more X, less Y" to Echo to tune relevance.

### Step 9 — Daily briefing
Ask: do they want a morning briefing, and at what time? Default 07:30. (Set via `/briefingtime` in the bot after first `/start`.)

### Step 10 — Launch
```bash
bash scripts/start.sh
```
Then the user messages their bot `/start` on Telegram. Confirm it responds.

---

## Repo conventions
- **Secrets** live in `.env` and `secrets/` — both gitignored. Never commit or echo them.
- **Personal config** (`config/vaults.yml`, `config/interests.yml`) is gitignored; only `*.example.yml` ships.
- **Runtime state** (`data/`: models, vector store, memory, chat state) is gitignored.
- Storage is plain Markdown (Obsidian-compatible). The vector index in `data/store.db` is derived and regenerable via `scripts/backfill.py`.

## Architecture (where things live)
- `app/bot.py` — Telegram handlers + intent routing
- `app/vault.py` — LLM classify (intent + vault + tasks + facts), note writing (`write_note`, `write_answer_note`), backlinks
- `app/ask.py` — general-question answering (`smart_answer`): quick LLM answer, escalates to `claude -p` + web research when warranted
- `app/rag.py` + `app/store.py` + `app/embed.py` — semantic search / RAG
- `app/todoist.py`, `app/gcal.py` — integrations (Todoist, Google Calendar + Gmail)
- `app/mailtriage.py`, `app/news.py`, `app/briefing.py` — email, news, daily briefing
- `app/memory.py` — durable personalization facts
- `config/*.yml` — vault routing + news interests

## Intents (auto-routed by `vault.classify`)
`note` (default, → vault) · `query` (RAG over user's OWN notes) · `complete` (close Todoist tasks) · `event` (calendar) · `mail` · `news` · `ask` (general/world question or live research → `ask.smart_answer`).
- `ask` answers are ALWAYS saved via `write_answer_note`: tagged `#echo/answer`, with `importance: 1..5` and `web_research:` frontmatter. `/ask <q>` forces the ask path; plain text/voice is auto-routed.
- Web research = `claude -p` with web tools allowed. This account's web path is the **exa MCP** (`mcp__exa__web_search_exa`); `app/llm.research_web` allowlists both builtin WebSearch/WebFetch and the exa MCP tools. Tune model/timeout via `ASK_MODEL` / `ASK_WEB_TIMEOUT`.

## Roadmap (Phase 2 — not built yet)
- **Weekly cleanup job:** `claude -p` reviews `#echo/answer` notes, re-sorts into best vault, dedups, prunes low-`importance`. (Importance frontmatter exists to support this.)
- **NotebookLM podcast:** turn the daily/news briefing into an audio podcast.
