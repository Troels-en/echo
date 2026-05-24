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

### More capabilities (all reachable by just talking — slash commands optional)
- **No commands to memorize** — the intent router maps natural language ("mach mir einen Podcast", "finde mein Steuerdokument", "was zuerst?") to the right capability. `/help` lists everything.
- **Conversation memory** — remembers the last ~10 turns so follow-ups ("und wer nutzt das?") resolve in context.
- **Multi-intent** — one message can hold a capture *and* a question; Echo does both (files the task **and** answers).
- **Background web research** — deep questions run off-thread ("researching, back in ~2-3 min"); the bot stays responsive. Ask "wie lange noch?" for a real job status.
- **Voice replies** — Echo answers with a spoken voice memo (ElevenLabs / Gemini TTS).
- **Audio podcast** — turns the briefing into a two-host German podcast (Gemini multi-speaker TTS, with fallbacks).
- **Document search** — searches documents on disk + Gmail attachments, summarizes, links into a vault.
- **Writing agent** — humanized German drafts (e.g. cover letters) in your own style, from a `Self_Vault` about you.
- **Overview & stats** — Obsidian dashboard + Telegram summary + Notion mirror; usage/streak/XP with a chart.
- **Task prioritization** — ranks open tasks by deadline + your goals (from memory), tells you what to do first.
- **SecondBrain bridge** — a curated LLM-Wiki is indexed into Echo's RAG; a weekly job synthesizes your notes into it.
- **Proactive nudges** — morning focus + evening habit check-in (grounded in your Habits vault); replies logged automatically.
- **Email to self** — emails you a briefing or a researched topic (Gmail).
- **Dev-task trigger** — "baue X in repo Y" spawns a headless Claude Code agent in that repo (on a new branch, never pushed) after you confirm.

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
- **Notion mirror**: no token needed — Echo delegates to a `claude -p` agent that uses the account-connected Notion MCP.
- **Dev-task trigger:** repos under `DEV_ROOT` (default `~`) are eligible; the agent works on a new branch and never pushes.

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

## Disclaimer

Personal project, shared as-is — **use at your own risk**, no warranty.

- **Autonomous agents:** the dev-task trigger runs a headless coding agent that writes and commits code in your repos. It works on a new branch and never pushes, but **review the diff before merging.** Treat generated code as a draft.
- **It acts on your accounts:** reads Gmail/Calendar, creates Todoist tasks, can email **you** (never others — outreach is draft-only by design). Confirm-before-act on destructive/outbound steps.
- **Bring your own keys/accounts.** You need your own Telegram bot, LLM (CLI subscription or API key), and optional integration credentials. Costs (LLM/TTS) are yours.
- **Unofficial integrations** (e.g. NotebookLM via `notebooklm-py`) can break when providers change; treat them as best-effort.
- Not affiliated with Anthropic, OpenAI, Google, Notion, Todoist, or Telegram.

## License
MIT — see [LICENSE](LICENSE).
