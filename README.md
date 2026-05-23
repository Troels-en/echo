# Echo

**A voice-first personal assistant with memory.** Speak into Telegram — Echo transcribes, understands what you mean, and acts: files notes into your Obsidian vaults, creates tasks, schedules calendar events, triages email, answers questions from your own notes, and briefs you each morning. It learns about you over time.

Runs locally. Uses your own LLM (via `codex`/`claude` CLI subscriptions or API keys). Your notes stay on your machine as plain Markdown.

---

## What it does

Send a voice note or text to your Telegram bot. Echo detects intent and routes automatically — no commands needed:

| You say | Echo does |
|---------|-----------|
| "Idea: build a tool that…" | Files a note in the right vault, auto-links related notes, extracts any tasks |
| "Remind me to call the dentist tomorrow" | Creates a Todoist task (split if multiple), Eisenhower priority |
| "Meeting with Sara Thursday 3pm" | Creates a Google Calendar event (after you confirm) |
| "What were my best ideas about X?" | RAG answer grounded in your own notes, with citations |
| "Explain SAFE notes / research the best X" | General-knowledge answer; escalates to live web research when it helps |
| "Did Sara reply yet?" | Searches your inbox and answers |
| "Clean my mailbox" | Finds obvious junk, asks before trashing |
| "What's new in AI?" | News briefing from your RSS feeds, filtered to your interests |
| "I finished the report" | Shows matching open tasks → you tap which to close |

Every destructive or external action is **confirm-before-act**. Echo never silently closes a task, sends mail, or deletes anything.

### Core features
- **Voice capture** — Telegram voice notes, transcribed locally with whisper.cpp
- **Auto-categorization** — notes routed to the right Obsidian vault by an LLM
- **Auto-backlinks** — new notes link to semantically related existing notes (`[[wikilinks]]`)
- **RAG copilot** — ask questions, get answers grounded in your notes with citations
- **General Q&A + web research** — `ask` intent answers world questions; escalates to live web research (`claude -p`), saves the answer with an importance rank
- **Task management** — Todoist tasks, auto-split, Eisenhower priority, cross-cut labels
- **Calendar** — natural-language event creation (Google Calendar)
- **Email** — triage, intent-driven search, inbox cleanup (Gmail)
- **Daily + news briefing** — morning push; news with dates, recency check, plain-language explanations tied to your context
- **Memory** — learns durable facts about you; inspect and correct them (`/memory`, `/editmemory`, `/forget`)

### Added capabilities (commands)
- **Voice replies** — `/voice on` makes Echo answer with a spoken voice memo (ElevenLabs TTS)
- **Audio podcast** — `/podcast` turns the briefing into a German audio podcast
- **Document search** — `/indexdocs` then `/finddoc <q>`: search documents on disk (`~/Documents`) + Gmail attachments, summarize, link into a vault
- **Writing agent** — `/draft <brief>`: humanized German drafts (e.g. cover letters) in your own style, from a `Self_Vault` of facts + style about you
- **Overview** — `/overview`: an Obsidian dashboard + Telegram summary + Notion mirror of everything you've fed in
- **Progress stats** — `/stats`: usage, intent breakdown, streak, XP, with a chart

---

## Architecture

```
Telegram (voice/text)
   │
   ▼
whisper.cpp server  ──►  transcript
   │
   ▼
LLM router (one call)  ──►  intent + classification + facts
   │
   ├─ note   → vault Markdown + backlinks + Todoist tasks + vector index
   ├─ query  → RAG over sqlite-vec  → cited answer
   ├─ ask    → LLM answer / live web research → saved with importance rank
   ├─ event  → Google Calendar (confirm)
   ├─ mail   → Gmail triage / search / clean (confirm)
   ├─ news   → RSS fetch + LLM relevance filter
   └─ complete → match open tasks → confirm → close

Commands add: /voice (TTS replies) · /podcast (audio) · /finddoc (doc search)
              /draft (style agent) · /overview (dashboard) · /stats (usage+XP)
```

- **Storage:** plain Markdown (Obsidian-compatible) + SQLite (`sqlite-vec`) for embeddings
- **Embeddings:** local, multilingual (`sentence-transformers`)
- **LLM:** pluggable — `codex`/`claude` CLI (subscription, no API cost) or OpenAI/Anthropic API
- **Transcription:** local whisper.cpp (no audio leaves your machine)

---

## Setup

### Prerequisites
- Python 3.11+ and [`uv`](https://github.com/astral-sh/uv)
- [`whisper.cpp`](https://github.com/ggerganov/whisper.cpp) (`brew install whisper-cpp`)
- `ffmpeg` (`brew install ffmpeg`)
- A Telegram bot token ([@BotFather](https://t.me/BotFather))
- An LLM: either `codex`/`claude` CLI logged in, or an API key

### Install
```bash
git clone https://github.com/<you>/echo.git
cd echo
uv sync

cp .env.example .env                       # fill in TELEGRAM_BOT_TOKEN
cp config/vaults.example.yml config/vaults.yml      # customize your vaults
cp config/interests.example.yml config/interests.yml # customize news topics

# download a whisper model
bash scripts/download_model.sh small        # or large-v3-turbo for best quality
```

### Optional integrations
- **Todoist:** add `TODOIST_API_TOKEN` to `.env`, run `python scripts/setup_todoist.py`
- **Google Calendar + Gmail:** see comments in `.env` — create OAuth credentials, run `python scripts/google_auth.py`
- **Voice replies + podcast:** add `ELEVENLABS_API_KEY` to `.env` (text-to-speech + voices scopes). Falls back to macOS `say` if absent.
- **Document search:** set `DOC_SEARCH_ROOT` in `.env` (default `~/Documents`), then `/indexdocs` in the bot.
- **Notion mirror** (`/overview`): goes through the Claude Notion MCP; runs agent/cron-side, not in the bot process.

### Run
```bash
bash scripts/start.sh    # starts whisper-server + the bot
```
Then message your bot `/start` on Telegram.

---

## LLM modes
Set in `.env`:
- `LLM_MODE=cli` — uses local `codex`/`claude` CLI (free up to your subscription quota)
- `LLM_MODE=api` — uses `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` (pay per token)

---

## Privacy
- Notes are plain Markdown on your disk. Audio is transcribed locally and deleted.
- Secrets live in `.env` and `secrets/` — both gitignored. Never commit them.
- Embeddings/memory/state live in `data/` — gitignored, never leaves your machine.

## License
MIT — see [LICENSE](LICENSE).
