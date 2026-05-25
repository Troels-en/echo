"""Telegram bot entrypoint. Voice messages → transcribe → classify → write note + Todoist task."""
from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from .config import Config
from .transcribe import transcribe, TranscribeError
from .vault import classify, write_note, write_answer_note, vault_todoist_config, find_related
from .llm import LLMError
from . import todoist as td
from . import store, rag, ask as ask_mod, intent as intent_mod, gcal, briefing as briefing_mod, state as state_mod, mailtriage, memory as memory_mod, news as news_mod, review as review_mod, agents as agents_mod, docsearch as docsearch_mod, podcast as podcast_mod, overview as overview_mod, events as events_mod, stats as stats_mod, tts as tts_mod, shortterm as shortterm_mod, secondbrain as secondbrain_mod, jobs as jobs_mod, proactive as proactive_mod, devtask as devtask_mod, notionsync as notionsync_mod, agenttask as agenttask_mod, transcript as transcript_mod, interactive as interactive_mod

log = logging.getLogger(__name__)

# whisper-server processes one transcription at a time → serialize voice notes.
_transcribe_lock = asyncio.Lock()


def _is_allowed(update: Update, cfg: Config) -> bool:
    if not cfg.allowed_user_ids:
        return True
    user = update.effective_user
    return bool(user and user.id in cfg.allowed_user_ids)


def _to_wav(src: Path) -> Path:
    dst = src.with_suffix(".wav")
    res = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
        capture_output=True, text=True, timeout=120,
    )
    if res.returncode != 0:
        raise TranscribeError(f"ffmpeg failed: {res.stderr[-300:]}")
    return dst


def _log_event(intent: str, classification: dict, text: str, source: str) -> None:
    """Best-effort interaction logging for /stats. Never raises into handlers."""
    vault = classification.get("vault") if intent == "note" else None
    events_mod.log_event(intent=intent, vault=vault, input_len=len(text or ""), source=source)


async def _safe_edit(message, text: str) -> None:
    """Edit a message as Markdown; if Telegram rejects the entities (unbalanced */_/`/[ in
    LLM output), resend as plain text so the user still gets the answer."""
    from telegram.error import BadRequest
    try:
        await message.edit_text(text, parse_mode="Markdown")
    except BadRequest as e:
        if "entit" in str(e).lower() or "parse" in str(e).lower():
            await message.edit_text(text)
        else:
            raise


async def _safe_reply(msg, text: str) -> None:
    """Send a new reply as Markdown; fall back to plain text on entity-parse rejection."""
    from telegram.error import BadRequest
    try:
        await msg.reply_text(text, parse_mode="Markdown")
    except BadRequest as e:
        if "entit" in str(e).lower() or "parse" in str(e).lower():
            await msg.reply_text(text)
        else:
            raise


async def _safe_reply_kb(msg, text: str, kb) -> None:
    """reply_text with an inline keyboard; fall back to plain text (keeping buttons) if
    Markdown entity parsing fails — otherwise a stray * / _ / ( in LLM text kills the message."""
    from telegram.error import BadRequest
    try:
        await msg.reply_text(text, parse_mode="Markdown", reply_markup=kb)
    except BadRequest as e:
        if "entit" in str(e).lower() or "parse" in str(e).lower():
            await msg.reply_text(text, reply_markup=kb)
        else:
            raise


async def _safe_edit_kb(message, text: str, kb) -> None:
    """edit_text with an inline keyboard; plain-text fallback (keeping buttons) on entity-parse reject."""
    from telegram.error import BadRequest
    try:
        await message.edit_text(text, parse_mode="Markdown", reply_markup=kb)
    except BadRequest as e:
        if "entit" in str(e).lower() or "parse" in str(e).lower():
            await message.edit_text(text, reply_markup=kb)
        else:
            raise


async def cmd_start(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id:
        state_mod.set_key("chat_id", chat_id)
    txt = f"Echo bereit. Schick mir eine Voice-Note.\nUser ID: `{user.id}`" if user else "Echo bereit."
    txt += "\n\nDaily-Briefing aktiv. `/briefing` für jetzt, `/briefingtime 07:30` zum Ändern."
    await update.message.reply_text(txt, parse_mode="Markdown")


async def cmd_briefing(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    if update.effective_chat:
        state_mod.set_key("chat_id", update.effective_chat.id)
    progress = await update.message.reply_text("📋 Baue Briefing...")
    try:
        text = await asyncio.to_thread(briefing_mod.build_briefing, cfg)
        await _safe_edit(progress, text)
        await _maybe_send_voice(text, cfg, update.message)
    except Exception as e:
        log.exception("briefing failed")
        await progress.edit_text(f"❌ Briefing-Fehler: {e}")


async def cmd_briefingtime(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    arg = " ".join(ctx.args).strip() if ctx.args else ""
    import re as _re
    if not _re.match(r"^\d{1,2}:\d{2}$", arg):
        await update.message.reply_text("Nutzung: `/briefingtime 07:30`", parse_mode="Markdown")
        return
    state_mod.set_key("briefing_time", arg)
    _reschedule_briefing(ctx.application)
    await update.message.reply_text(f"⏰ Daily-Briefing jetzt um {arg} Uhr.")


async def _briefing_job(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    from datetime import date
    cfg: Config = ctx.application.bot_data["cfg"]
    st = state_mod.load()
    if not st.get("briefing_enabled") or not st.get("chat_id"):
        return
    try:
        text = await asyncio.to_thread(briefing_mod.build_briefing, cfg)
        await ctx.bot.send_message(st["chat_id"], text, parse_mode="Markdown")
        state_mod.set_key("last_briefing_sent", date.today().isoformat())
    except Exception as e:
        log.exception("scheduled briefing failed: %s", e)


def _reschedule_briefing(app) -> None:
    from datetime import time as dtime
    from zoneinfo import ZoneInfo
    jq = app.job_queue
    for job in jq.get_jobs_by_name("daily_briefing"):
        job.schedule_removal()
    st = state_mod.load()
    hh, mm = (int(x) for x in st.get("briefing_time", "07:30").split(":"))
    jq.run_daily(
        _briefing_job,
        time=dtime(hour=hh, minute=mm, tzinfo=ZoneInfo("Europe/Berlin")),
        name="daily_briefing",
        job_kwargs={"misfire_grace_time": 3600},
    )
    log.info("daily briefing scheduled for %02d:%02d", hh, mm)


async def cmd_id(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if user:
        await update.message.reply_text(f"User ID: `{user.id}`", parse_mode="Markdown")


async def _create_tasks(classification: dict) -> list[td.Task]:
    """Create one Todoist task per extracted action. Returns created tasks."""
    task_list = classification.get("tasks") or []
    if not task_list:
        return []

    vault_name = classification["vault"]
    project_name, default_labels, create_tasks = vault_todoist_config(vault_name)
    if not create_tasks:
        return []

    created: list[td.Task] = []
    for task_data in task_list:
        content = (task_data.get("content") or "").strip()
        if not content:
            continue
        llm_labels = task_data.get("labels", []) or []
        labels = list(dict.fromkeys(default_labels + llm_labels))
        due = task_data.get("due_string") or None
        pri = task_data.get("priority") or None
        if pri == 0:
            pri = None
        try:
            t = await asyncio.to_thread(
                td.create_task,
                content=content,
                project=project_name,
                labels=labels,
                due_string=due,
                priority=pri,
            )
            created.append(t)
        except Exception as e:
            log.error("todoist create failed for %r: %s", content, e)
    return created


async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        await update.message.reply_text("Nicht autorisiert.")
        return

    msg = update.message
    voice = msg.voice or msg.audio
    if not voice:
        await msg.reply_text("Keine Voice-Note gefunden.")
        return

    work_id = uuid4().hex[:8]
    ogg_path = cfg.data_dir / "audio" / f"{work_id}.ogg"
    progress = await msg.reply_text("📥 Empfangen, lade runter...")

    transcript = ""
    classification = None
    outcome = ""
    try:
        tg_file = await voice.get_file()
        await tg_file.download_to_drive(custom_path=str(ogg_path))

        # Serialize transcription: the whisper-server handles one job at a time, so two
        # voice notes in quick succession would otherwise clobber each other.
        if _transcribe_lock.locked():
            await progress.edit_text("⏳ Eine andere Voice-Note wird gerade transkribiert — du bist als Nächstes...")
        async with _transcribe_lock:
            await progress.edit_text("🎙️ Transkribiere...")
            wav_path = await asyncio.to_thread(_to_wav, ogg_path)
            transcript = await asyncio.to_thread(transcribe, wav_path, cfg.whisper_model_path)
        if not transcript.strip():
            await progress.edit_text("⚠️ Leere Transkription.")
            return

        # If a clarify dialog is awaiting answers, this voice note IS the answer — not a new request.
        if await _consume_dialog_answer(transcript, msg, ctx):
            await progress.delete()
            return

        await progress.edit_text(f"🧠 Verstehe...\n\n_{transcript[:200]}_", parse_mode="Markdown")
        # Single LLM call: intent + (if note) classification, with recent-conversation context
        history = shortterm_mod.recent_text()
        classification = await asyncio.to_thread(classify, transcript, cfg, history)
        intent = classification.get("intent", "note")
        log.info("voice intent: %s", intent)
        _log_event(intent, classification, transcript, "voice")
        shortterm_mod.add("user", transcript)
        if intent in ("devtask", "agenttask"):
            outcome = f"{intent}: Bestätigung angefragt"
        if intent == "query":
            await progress.delete()
            await _answer_query(transcript, cfg, msg, history)
            return
        if intent == "complete":
            await progress.delete()
            await _present_completion_candidates(transcript, cfg, msg)
            await _maybe_answer_followup(classification, cfg, msg, history)
            return
        if intent == "event":
            await progress.delete()
            await _present_event(classification, cfg, msg, ctx)
            await _maybe_answer_followup(classification, cfg, msg, history)
            return
        if intent == "mail":
            await progress.delete()
            await _handle_mail(cfg, msg, ctx, transcript, classification)
            return
        if intent == "news":
            await progress.delete()
            await _send_news(cfg, msg)
            return
        if intent == "ask":
            await progress.delete()
            await _answer_ask(transcript, cfg, msg, history)
            return
        if intent == "status":
            await progress.delete()
            await msg.reply_text(jobs_mod.status_text())
            return
        if intent == "devtask":
            await progress.delete()
            await _present_devtask(classification, cfg, msg, ctx)
            return
        if intent == "agenttask":
            await progress.delete()
            await _present_agenttask(classification, cfg, msg, ctx)
            return
        if intent == "prioritize":
            await progress.delete()
            await _do_prioritize(transcript, cfg, msg)
            return
        if intent == "help":
            await progress.delete()
            await msg.reply_text(HELP_TEXT, parse_mode="Markdown")
            return
        if intent in _ACTION_INTENTS:
            await progress.delete()
            await _route_action(intent, transcript, update, ctx)
            return

        tasks = await _create_tasks(classification)
        related = await asyncio.to_thread(find_related, transcript, classification["vault"], cfg)

        note_path = await asyncio.to_thread(
            write_note, transcript, classification, cfg, tasks=tasks, related=related,
        )
        outcome = f"note -> {note_path.name}" + (f" (+{len(tasks)} Task)" if tasks else "")

        # Index into vector store for RAG
        try:
            await asyncio.to_thread(
                store.upsert_note,
                cfg.data_dir / "store.db",
                str(note_path.resolve()),
                classification["vault"],
                classification.get("title", ""),
                classification.get("summary", ""),
                transcript,
                None,
            )
        except Exception as e:
            log.error("indexing failed (note still written): %s", e)

        await progress.edit_text(
            _format_ingest_reply(classification, tasks, note_path, cfg),
            parse_mode="Markdown",
            reply_markup=_tasks_keyboard(tasks),
        )
        await _maybe_answer_followup(classification, cfg, msg, history)
    except (TranscribeError, LLMError) as e:
        log.exception("voice handler failed")
        outcome = f"error: {e}"
        await progress.edit_text(f"❌ Fehler: {e}")
    except Exception as e:
        log.exception("unexpected error")
        outcome = f"error: {e}"
        await progress.edit_text(f"❌ Unerwarteter Fehler: {e}")
    finally:
        if transcript.strip():
            intent = classification.get("intent") if classification else None
            transcript_mod.record("voice", transcript, classification, outcome or (f"-> {intent}" if intent else ""))
        for p in [ogg_path, ogg_path.with_suffix(".wav")]:
            p.unlink(missing_ok=True)


async def _answer_query(question: str, cfg: Config, msg, history: str = "") -> None:
    progress = await msg.reply_text("🔍 Suche in Vault...")
    try:
        result = await asyncio.to_thread(rag.answer_question, question, cfg)
        hits = result.get("hits", [])
        if not hits:
            await progress.edit_text("Keine relevanten Notes gefunden.")
            return

        await progress.edit_text(f"🧠 Synthetisiere aus {len(hits)} Notes...")
        answer = result.get("answer", "")
        used = result.get("used_sources", [])
        conf = result.get("confidence", 0)

        src_lines = []
        for i, h in enumerate(hits, 1):
            if used and i not in used:
                continue
            try:
                rel = Path(h.path).relative_to(cfg.vault_root)
                src_lines.append(f"[{i}] `{rel}`")
            except Exception:
                src_lines.append(f"[{i}] `{h.path}`")

        out = f"{answer}\n\n_conf {conf:.2f}_"
        if src_lines:
            out += "\n\n*Quellen:*\n" + "\n".join(src_lines)
        if len(out) > 4000:
            out = out[:3900] + "\n\n_(gekürzt)_"
        await _safe_edit(progress, out)
        shortterm_mod.add("echo", answer)
    except Exception as e:
        log.exception("ask failed")
        await progress.edit_text(f"❌ Fehler: {e}")


async def _answer_ask(question: str, cfg: Config, msg, history: str = "") -> None:
    """General question. Quick answers come back inline; deep web research runs in the
    background (Echo says so + an ETA) so the bot stays responsive to other messages."""
    progress = await msg.reply_text("🤔 Denke nach...")
    try:
        triage_data = await asyncio.to_thread(ask_mod.triage, question, cfg, history)
    except Exception as e:
        log.exception("ask triage failed")
        await progress.edit_text(f"❌ Fehler: {e}")
        return

    if ask_mod.needs_web(triage_data):
        ack = ("🔍 Tiefen-Recherche läuft im Hintergrund — ich melde mich in ~2-3 Min mit der Antwort. "
               "Du kannst in der Zwischenzeit weiter Fragen stellen.")
        await _safe_edit(progress, ack)
        shortterm_mod.add("echo", f"[Hintergrund-Recherche gestartet zu: {question[:80]}] {ack}")
        jid = jobs_mod.start("research", question[:60])
        asyncio.create_task(_finish_research_bg(question, cfg, history, triage_data, msg, jid))
        return

    answer = (triage_data.get("answer") or "").strip() or "Keine Antwort."
    result = ask_mod.finalize(triage_data, answer, False, cfg, question)
    await _deliver_ask(result, question, cfg, msg, progress=progress)


async def _finish_research_bg(question: str, cfg: Config, history: str,
                              triage_data: dict, msg, jid: int | None = None) -> None:
    """Run the slow web research off the handler, then push the answer as a new message."""
    try:
        answer = await asyncio.to_thread(ask_mod.run_research, question, cfg, history)
    except Exception as e:
        log.exception("background research failed")
        await msg.reply_text(f"❌ Recherche fehlgeschlagen: {e}")
        return
    finally:
        if jid is not None:
            jobs_mod.finish(jid)
    result = ask_mod.finalize(triage_data, answer, True, cfg, question)
    await _deliver_ask(result, question, cfg, msg, progress=None)


async def _deliver_ask(result: dict, question: str, cfg: Config, msg, progress=None) -> None:
    """Save the answer to the vault + send it (edit `progress` if given, else a new message)."""
    answer = result.get("answer", "").strip() or "Keine Antwort."
    note_path = None
    try:
        note_path = await asyncio.to_thread(write_answer_note, question, result, cfg)
        try:
            await asyncio.to_thread(
                store.upsert_note, cfg.data_dir / "store.db",
                str(note_path.resolve()), result["vault"],
                result.get("title", ""), question, answer,
            )
        except Exception as e:
            log.error("indexing answer failed (note still written): %s", e)
    except Exception as e:
        log.error("saving answer note failed: %s", e)

    web_tag = "🌐 Web-Recherche" if result.get("used_web") else "💬 LLM"
    footer = f"\n\n_{web_tag} · ⭐ {result.get('importance', 3)}/5"
    if note_path:
        try:
            footer += f" · 📄 `{note_path.relative_to(cfg.vault_root)}`"
        except Exception:
            pass
    footer += "_"

    out = answer + footer
    if len(out) > 4000:
        out = answer[: 3900 - len(footer)] + "\n\n_(gekürzt)_" + footer
    if progress is not None:
        await _safe_edit(progress, out)
    else:
        await _safe_reply(msg, out)
    shortterm_mod.add("echo", answer)
    await _maybe_send_voice(answer, cfg, msg)


async def _maybe_send_voice(text: str, cfg: Config, msg) -> None:
    """When /voice is ON and TTS is configured, also send `text` as a voice memo.

    Text reply is always kept; voice is additive. Never raises into the caller —
    a TTS/transcode failure just logs and skips the voice memo.
    """
    if not state_mod.load().get("voice_enabled"):
        return
    if not tts_mod.available(cfg):
        log.info("voice on but ELEVENLABS_API_KEY missing; skipping voice memo")
        return
    audio_path = None
    try:
        audio_path = await asyncio.to_thread(tts_mod.synthesize, text, cfg)
        with audio_path.open("rb") as f:
            await msg.reply_voice(voice=f)
    except Exception as e:
        log.error("voice synthesis/send failed (text reply already sent): %s", e)
    finally:
        if audio_path is not None:
            audio_path.unlink(missing_ok=True)


async def cmd_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Toggle voice-memo replies. `/voice on|off`; no arg shows current state."""
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        await update.message.reply_text("Nicht autorisiert.")
        return
    arg = (ctx.args[0].lower() if ctx.args else "").strip()
    if arg in ("on", "an", "ein"):
        state_mod.set_key("voice_enabled", True)
        extra = "" if tts_mod.available(cfg) else "\n⚠️ `ELEVENLABS_API_KEY` fehlt — kein Audio bis Key gesetzt."
        await update.message.reply_text("🔊 Voice-Antworten AN." + extra, parse_mode="Markdown")
    elif arg in ("off", "aus"):
        state_mod.set_key("voice_enabled", False)
        await update.message.reply_text("🔇 Voice-Antworten AUS.")
    else:
        on = state_mod.load().get("voice_enabled")
        await update.message.reply_text(
            f"Voice-Antworten: {'AN' if on else 'AUS'}. Nutzung: `/voice on` | `/voice off`.",
            parse_mode="Markdown",
        )


def _pending_events(ctx) -> dict:
    return ctx.application.bot_data.setdefault("pending_events", {})


async def _present_event(classification: dict, cfg: Config, msg, ctx) -> None:
    """Show parsed event and ask for confirmation before writing to calendar."""
    ev = classification.get("event") or {}
    summary = (ev.get("summary") or "").strip()
    start = (ev.get("start") or "").strip()
    if not summary or not start:
        await msg.reply_text("Konnte Termin nicht erkennen. Sag z.B. 'morgen 15 Uhr Zahnarzt'.")
        return

    if not gcal.is_configured():
        await msg.reply_text(
            "📅 Google Calendar noch nicht verbunden.\n"
            "Setup: `secrets/google_credentials.json` ablegen, dann "
            "`.venv/bin/python scripts/google_auth.py` ausführen.",
            parse_mode="Markdown",
        )
        return

    from datetime import datetime
    try:
        start_dt = datetime.fromisoformat(start)
    except ValueError:
        await msg.reply_text(f"Datum unklar: {start!r}")
        return

    token = uuid4().hex[:8]
    end = (ev.get("end") or "").strip()
    _pending_events(ctx)[token] = {
        "summary": summary, "start": start, "end": end,
        "location": ev.get("location", ""),
    }
    when = start_dt.strftime("%a %d.%m. %H:%M")
    loc = f"\n📍 {ev['location']}" if ev.get("location") else ""
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Eintragen", callback_data=f"ev:{token}"),
        InlineKeyboardButton("✗ Abbrechen", callback_data="cancel"),
    ]])
    await msg.reply_text(
        f"📅 *{summary}*\n🕐 {when}{loc}\n\nIn Kalender eintragen?",
        parse_mode="Markdown", reply_markup=keyboard,
    )


def _pending_mail(ctx) -> dict:
    return ctx.application.bot_data.setdefault("pending_mail", {})


def _pending_clean(ctx) -> dict:
    return ctx.application.bot_data.setdefault("pending_clean", {})


async def _handle_mail(cfg: Config, msg, ctx, text: str = "", classification: dict | None = None) -> None:
    """Dispatch mail by action: triage | search | clean."""
    if not gcal.is_configured():
        await msg.reply_text("📧 Gmail nicht verbunden. Erst Google-OAuth (siehe .env).")
        return
    action = "triage"
    search_terms = ""
    if classification:
        ma = classification.get("mail_action") or {}
        action = ma.get("action") or "triage"
        search_terms = ma.get("search_terms") or ""

    if action == "search":
        progress = await msg.reply_text("🔎 Durchsuche Mails...")
        try:
            r = await asyncio.to_thread(mailtriage.search, cfg, search_terms, text)
            await progress.edit_text(f"📧 {r.get('answer','(keine Antwort)')}", parse_mode="Markdown")
        except Exception as e:
            log.exception("mail search failed")
            await progress.edit_text(f"❌ Fehler: {e}")
        return

    if action == "clean":
        progress = await msg.reply_text("🧹 Suche Müll im Postfach...")
        try:
            cleanable = await asyncio.to_thread(mailtriage.find_cleanable, cfg)
        except Exception as e:
            log.exception("clean scan failed")
            await progress.edit_text(f"❌ Fehler: {e}")
            return
        if not cleanable:
            await progress.edit_text("✨ Nichts eindeutig Müll. Postfach bleibt unangetastet.")
            return
        token = uuid4().hex[:8]
        _pending_clean(ctx)[token] = [c["id"] for c in cleanable]
        lines = [f"🧹 *{len(cleanable)} Mails als Müll erkannt:*", ""]
        for c in cleanable[:15]:
            lines.append(f"  • {c.get('from','?')[:25]} — {c.get('subject','')[:40]}  _{c.get('reason','')[:25]}_")
        lines.append("\n_Bei Zweifel behalten. In Papierkorb (wiederherstellbar)._")
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton(f"🗑️ {len(cleanable)} in Papierkorb", callback_data=f"clean:{token}"),
            InlineKeyboardButton("✗ Nichts", callback_data="cancel"),
        ]])
        await progress.edit_text("\n".join(lines), parse_mode="Markdown", reply_markup=keyboard)
        return

    # default: triage
    await _triage_mail(cfg, msg, ctx)


async def _triage_mail(cfg: Config, msg, ctx) -> None:
    if not gcal.is_configured():
        await msg.reply_text("📧 Gmail nicht verbunden. Erst Google-OAuth (siehe .env).")
        return
    progress = await msg.reply_text("📧 Lese Postfach...")
    try:
        result = await asyncio.to_thread(mailtriage.triage, cfg)
    except Exception as e:
        log.exception("triage failed")
        await progress.edit_text(f"❌ Mail-Fehler: {e}")
        return

    if result.get("count", 0) == 0:
        await progress.edit_text("Postfach leer (in:inbox).")
        return

    digest = result.get("digest", [])
    tasks = result.get("tasks", [])
    events = result.get("events", [])

    lines = [f"📧 *{result['count']} Mails:*", ""]
    for d in digest:
        flag = "⚡" if d.get("needs_action") else "·"
        lines.append(f"{flag} *{d.get('from','?')[:25]}* — {d.get('summary','')[:70]}")
    if tasks:
        lines += ["", f"*📋 {len(tasks)} Task(s) vorgeschlagen:*"]
        for t in tasks:
            lines.append(f"  • {t['content'][:60]}")
    if events:
        lines += ["", f"*📅 {len(events)} Termin(e) vorgeschlagen:*"]
        for e in events:
            lines.append(f"  • {e.get('summary','')[:50]} ({e.get('start','')[:16]})")

    keyboard_rows = []
    if tasks or events:
        token = uuid4().hex[:8]
        _pending_mail(ctx)[token] = {"tasks": tasks, "events": events}
        row = []
        if tasks:
            row.append(InlineKeyboardButton(f"📋 {len(tasks)} Tasks anlegen", callback_data=f"mailt:{token}"))
        if events:
            row.append(InlineKeyboardButton(f"📅 {len(events)} Termine", callback_data=f"maile:{token}"))
        keyboard_rows.append(row)
        keyboard_rows.append([InlineKeyboardButton("✗ Nichts", callback_data="cancel")])

    text = "\n".join(lines)
    if len(text) > 4000:
        text = text[:3900] + "\n_(gekürzt)_"
    await progress.edit_text(
        text, parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard_rows) if keyboard_rows else None,
    )


async def _present_completion_candidates(text: str, cfg: Config, msg) -> None:
    """Find matching open tasks and let the user CONFIRM which to close. Never auto-close."""
    progress = await msg.reply_text("🔎 Suche passende Tasks...")
    candidates, reason = await asyncio.to_thread(intent_mod.match_tasks_for_completion, text, cfg)
    if not candidates:
        await progress.edit_text(
            "Keine passende offene Task gefunden. Falls du was Neues meintest, "
            "formuliere es als Notiz."
        )
        return

    buttons = []
    for t in candidates[:5]:
        content = t["content"][:45]
        buttons.append([InlineKeyboardButton(f"✅ {content}", callback_data=f"close:{t['id']}")])
    if len(candidates) > 1:
        ids = ",".join(t["id"] for t in candidates[:5])
        buttons.append([InlineKeyboardButton(f"✅✅ Alle {len(candidates[:5])} schließen", callback_data=f"closeall:{ids}")])
    buttons.append([InlineKeyboardButton("✗ Abbrechen", callback_data="cancel")])

    hint = f"_{reason}_\n\n" if reason else ""
    await progress.edit_text(
        f"{hint}Welche Task(s) als erledigt markieren?",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(buttons),
    )


async def cmd_ask(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Force a general-knowledge / research answer (bypass the auto-router)."""
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        await update.message.reply_text("Nicht autorisiert.")
        return
    question = " ".join(ctx.args) if ctx.args else ""
    if not question.strip():
        await update.message.reply_text(
            "Nutzung: `/ask <frage>` — allgemeine Frage / Recherche. "
            "Für Fragen an deine Notizen schreib einfach ohne Slash.",
            parse_mode="Markdown",
        )
        return
    await _answer_ask(question, cfg, update.message)


# --- Interactive devtask / agenttask -----------------------------------------------------------
# Flow: classify -> read-only PLAN pass (sees what already exists, decides if clarifying questions
# are needed) -> ask via Telegram -> user answers (1 free-text round) -> confirm -> RESUME the same
# claude session to execute. Single-user, so one dialog slot at a time makes the answer-intercept
# in handle_text/handle_voice unambiguous.
_DIALOG_TTL = 2 * 3600  # awaiting-answers expires after 2h


def _dialog(ctx) -> dict | None:
    return ctx.application.bot_data.get("active_dialog")


def _set_dialog(ctx, d: dict) -> None:
    ctx.application.bot_data["active_dialog"] = d


def _clear_dialog(ctx) -> None:
    ctx.application.bot_data.pop("active_dialog", None)


def _confirm_kb(token: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Ausführen", callback_data=f"go:{token}"),
        InlineKeyboardButton("✗ Abbrechen", callback_data="cancel"),
    ]])


def _format_questions(d: dict) -> str:
    lines = []
    if d.get("existing_work"):
        lines.append(f"📂 Schon da: {d['existing_work']}")
    if d.get("plan"):
        lines.append(f"📝 Plan: {d['plan']}")
    lines.append("")
    lines.append("❓ Bevor ich loslege:")
    for i, q in enumerate(d.get("questions") or [], 1):
        choices = q.get("choices") or []
        suffix = f"  ({' / '.join(choices)})" if choices else ""
        lines.append(f"{i}. {q.get('question', '')}{suffix}")
    lines.append("")
    lines.append("_Antworte einfach frei (alle Fragen in einer Nachricht). Oder /cancel._")
    return "\n".join(lines)


def _format_confirm(d: dict) -> str:
    head = "🛠️ *Dev-Task*" if d["kind"] == "devtask" else "🤖 *Aufgabe*"
    lines = [head]
    if d["kind"] == "devtask" and d.get("repo"):
        lines.append(f"📁 `{Path(d['repo']).name}` — neuer Branch, kein Push (reviewbar)")
    if d.get("existing_work"):
        lines.append(f"📂 Schon da: {d['existing_work']}")
    lines.append(f"📝 {d.get('plan') or d['task']}")
    if d.get("answers"):
        lines.append(f"💬 Deine Vorgabe: {d['answers'].strip()}")
    lines.append("")
    lines.append("Ausführen?")
    return "\n".join(lines)


async def _present_devtask(classification: dict, cfg: Config, msg, ctx) -> None:
    """Start the interactive plan -> clarify -> confirm dialog for a code task."""
    repo_hint = (classification.get("dev_repo") or "").strip()
    task = (classification.get("dev_task") or "").strip()
    if not task:
        await msg.reply_text("🛠️ Was genau soll Claude Code umsetzen — und in welchem Projekt?")
        return
    repo = await asyncio.to_thread(devtask_mod.resolve_repo, repo_hint)
    if not repo:
        await msg.reply_text(
            f"🛠️ Projekt '{repo_hint}' nicht als Git-Repo unter `{devtask_mod.DEV_ROOT}` gefunden. "
            "Welches Projekt genau?", parse_mode="Markdown")
        return
    await _start_task_dialog(ctx, msg, "devtask", task, repo=str(repo))


async def _present_agenttask(classification: dict, cfg: Config, msg, ctx) -> None:
    """Start the interactive plan -> clarify -> confirm dialog for a general executor task."""
    task = (classification.get("agent_task") or "").strip()
    if not task:
        await msg.reply_text("🤖 Was genau soll ich ausführen (z.B. 'zieh meine Notion-Habits in den Vault')?")
        return
    await _start_task_dialog(ctx, msg, "agenttask", task)


async def _start_task_dialog(ctx, msg, kind: str, task: str, repo: str | None = None) -> None:
    """Concurrency-gate, then kick off the read-only PLAN pass in the background."""
    existing = _dialog(ctx)
    if existing and existing.get("state") in ("planning", "awaiting_answers", "awaiting_confirm"):
        await msg.reply_text("Erst die aktuelle Aufgabe abschließen oder /cancel.")
        return
    token = uuid4().hex[:8]
    label = "🛠️ Plane Dev-Task" if kind == "devtask" else "🤖 Plane Aufgabe"
    prog = await msg.reply_text(f"{label} — schaue erst, was schon da ist...")
    _set_dialog(ctx, {
        "token": token, "kind": kind, "task": task, "repo": repo,
        "state": "planning", "session_id": "", "questions": [], "answers": "",
        "plan": "", "existing_work": "", "recommended_default": "", "created_ts": time.time(),
    })
    asyncio.create_task(_run_plan_bg(ctx, token, kind, task, repo, prog))


async def _run_plan_bg(ctx, token: str, kind: str, task: str, repo: str | None, prog) -> None:
    try:
        if kind == "devtask":
            res = await asyncio.to_thread(interactive_mod.plan, interactive_mod.DEVTASK, task, Path(repo))
        else:
            res = await asyncio.to_thread(interactive_mod.plan, interactive_mod.AGENTTASK, task, interactive_mod.HOME)
    except Exception as e:
        log.exception("plan bg failed")
        _clear_dialog(ctx)
        await _safe_edit(prog, f"❌ Planung fehlgeschlagen: {e}")
        return
    d = _dialog(ctx)
    if not d or d.get("token") != token:
        return  # cancelled or superseded while planning
    if res.get("error"):
        _clear_dialog(ctx)
        await _safe_edit(prog, f"⚠️ Planung: {res['error']}")
        return
    d["session_id"] = res.get("session_id", "")
    d["plan"] = res.get("plan", "")
    d["existing_work"] = res.get("existing_work", "")
    d["questions"] = res.get("questions") or []
    d["recommended_default"] = res.get("recommended_default", "")
    if d["questions"]:
        d["state"] = "awaiting_answers"
        _set_dialog(ctx, d)
        await _safe_edit(prog, _format_questions(d))
    else:
        d["state"] = "awaiting_confirm"
        _set_dialog(ctx, d)
        await _safe_edit_kb(prog, _format_confirm(d), _confirm_kb(token))


async def _consume_dialog_answer(text: str, msg, ctx) -> bool:
    """If a clarify dialog is awaiting answers, treat this message as the answer batch and move to
    confirmation. Returns True if consumed (caller must skip normal classification)."""
    d = _dialog(ctx)
    if not d or d.get("state") != "awaiting_answers":
        return False
    if time.time() - d.get("created_ts", 0) > _DIALOG_TTL:
        _clear_dialog(ctx)
        await msg.reply_text("Dialog abgelaufen — schick die Aufgabe nochmal.")
        return True
    d["answers"] = f"{d.get('answers', '')}\n{text}".strip()
    d["state"] = "awaiting_confirm"
    _set_dialog(ctx, d)
    await _safe_reply_kb(msg, _format_confirm(d), _confirm_kb(d["token"]))
    return True


async def handle_interactive_go(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()
    token = q.data.split(":", 1)[1]
    d = _dialog(ctx)
    if not d or d.get("token") != token:
        await q.edit_message_text("Aufgabe abgelaufen.")
        return
    kind = d["kind"]
    # free the dialog slot now — execution is fire-and-forget and asks nothing more
    _clear_dialog(ctx)
    icon = "🛠️" if kind == "devtask" else "🤖"
    await q.edit_message_text(
        f"{icon} Läuft im Hintergrund — ich melde mich. _Du kannst weiter fragen._",
        parse_mode="Markdown")
    n_q = len(d.get("questions") or [])
    if kind == "devtask":
        repo = Path(d["repo"])
        jid = jobs_mod.start("devtask", repo.name)
        asyncio.create_task(_run_devtask_bg(repo, d["task"], q.message, jid,
                                            session_id=d["session_id"], answers=d["answers"], n_q=n_q))
    else:
        jid = jobs_mod.start("agenttask", d["task"][:60])
        asyncio.create_task(_run_agenttask_bg(d["task"], q.message, jid,
                                              session_id=d["session_id"], answers=d["answers"], n_q=n_q))


async def _run_devtask_bg(repo: Path, task: str, msg, jid: int | None = None,
                          session_id: str = "", answers: str = "", n_q: int = 0) -> None:
    outcome = "fertig"
    try:
        res = await asyncio.to_thread(devtask_mod.run_devtask, repo, task, session_id, answers)
        if res.get("error"):
            outcome = f"error: {res['error']}"
            await msg.reply_text(f"⚠️ Dev-Task: {res['error']}")
            return
        outcome = f"fertig: branch {res['branch']} (geklärt: {n_q} Fragen)"
        out = (f"🛠️ Fertig in `{repo.name}` (Branch `{res['branch']}`).\n\n"
               f"*Geändert:*\n{res.get('changed', '') or '(nichts)'}\n\n"
               f"{res.get('report', '')[:1200]}\n\n"
               f"_Branch lokal zum Prüfen — kein Push. Merge wenn's passt._")
        await _safe_reply(msg, out)
    except Exception as e:
        log.exception("devtask bg failed")
        outcome = f"error: {e}"
        await msg.reply_text(f"❌ Dev-Task-Fehler: {e}")
    finally:
        transcript_mod.record("devtask:run", f"{repo.name}: {task}", None, outcome)
        if jid is not None:
            jobs_mod.finish(jid)


async def _run_agenttask_bg(task: str, msg, jid: int | None = None,
                            session_id: str = "", answers: str = "", n_q: int = 0) -> None:
    outcome = f"fertig (geklärt: {n_q} Fragen)"
    try:
        res = await asyncio.to_thread(agenttask_mod.run_agenttask, task, session_id, answers)
        if res.get("error"):
            outcome = f"error: {res['error']}"
            await msg.reply_text(f"⚠️ Aufgabe: {res['error']}\n\n{res.get('report','')[:800]}")
            return
        await _safe_reply(msg, f"🤖 *Fertig.*\n\n{res.get('report', '')[:1800]}")
    except Exception as e:
        log.exception("agenttask bg failed")
        outcome = f"error: {e}"
        await msg.reply_text(f"❌ Aufgabe-Fehler: {e}")
    finally:
        transcript_mod.record("agenttask:run", task, None, outcome)
        if jid is not None:
            jobs_mod.finish(jid)


async def cmd_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Abort a pending interactive devtask/agenttask dialog."""
    if _dialog(ctx):
        _clear_dialog(ctx)
        await update.message.reply_text("Abgebrochen.")
    else:
        await update.message.reply_text("Nichts offen zum Abbrechen.")


_ACTION_INTENTS = {"podcast", "overview", "stats", "synthesize", "draft", "finddoc", "mailme"}


async def _route_action(intent: str, text: str, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Route a natural-language action intent to its slash-command handler, so the user
    never has to remember slash commands. Arg-taking actions get the message as their args."""
    if intent in ("draft", "finddoc", "mailme"):
        ctx.args = text.split()
    handler = {
        "podcast": cmd_podcast,
        "overview": cmd_overview,
        "stats": cmd_stats,
        "synthesize": cmd_synthesize,
        "draft": cmd_draft,
        "finddoc": cmd_finddoc,
        "mailme": cmd_mailme,
    }[intent]
    await handler(update, ctx)


HELP_TEXT = """🧠 *Echo — was ich kann* (einfach normal sagen, keine Befehle nötig)

*Festhalten & Wissen*
• Idee/Gedanke → landet als Notiz im richtigen Vault (+ Todoist-Task)
• „Was waren meine Ideen zu X?" → Antwort aus deinen Notizen
• „Was ist ein SAFE-Note?" → allgemeine Antwort / Web-Recherche (im Hintergrund)

*Organisieren*
• „erledigt: X" → schließt passende Tasks
• „was soll ich zuerst machen?" → priorisiert deine Tasks
• „morgen 15 Uhr Zahnarzt" → Kalender-Termin
• „check meine Mails" → Gmail-Triage

*Output*
• „mach mir einen Podcast" → Audio-Briefing
• „lies mir das vor" / `/voice on` → Sprachantworten
• „schreib mir ein Anschreiben für …" → Entwurf in deinem Stil
• „finde mein Steuerdokument" → Doc-Suche (Disk + Mail)
• „maile mir das Briefing" → E-Mail an dich
• „zeig mir eine Übersicht / meine Stats" → Dashboard / XP

*Wissen & Dev*
• „fass meine Woche zusammen" → Synthese ins SecondBrain-Wiki
• „baue X in repo Y" → Claude-Code-Agent (mit Bestätigung)

*Proaktiv*: morgens Fokus-Nudge, abends Habit-Check-in.
*Status*: „wie lange dauert das noch?" → echter Job-Status.
"""


async def _do_prioritize(text: str, cfg: Config, msg) -> None:
    progress = await msg.reply_text("🎯 Priorisiere deine Tasks...")
    out = await asyncio.to_thread(intent_mod.rank_tasks, cfg, text)
    await _safe_edit(progress, out)


async def _mirror_notion_bg(date: str, entry: str, msg) -> None:
    """Mirror a habit check-in to Notion via the agent bridge (slow, background)."""
    ok = await asyncio.to_thread(notionsync_mod.mirror_habit_log, date, entry)
    if ok:
        await msg.reply_text("🔁 Auch in Notion gespiegelt.")


async def _maybe_answer_followup(classification: dict, cfg: Config, msg, history: str = "") -> None:
    """Multi-intent: if the message also held a question on top of a capture/action, answer it."""
    q = (classification.get("also_question") or "").strip()
    if q:
        await _answer_ask(q, cfg, msg, history)


async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Plain-text message → intent classifier → route."""
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    msg = update.message
    text = (msg.text or "").strip()
    if not text:
        return

    # If a clarify dialog is awaiting answers, this message IS the answer — handle before anything else.
    if not text.startswith("/") and await _consume_dialog_answer(text, msg, ctx):
        return

    # Evening habit check-in capture: if Echo just asked (within 4h), log this reply to the vault.
    import time as _t
    ck_ts = state_mod.load().get("pending_habit_checkin_ts")
    if ck_ts and (_t.time() - ck_ts) < 4 * 3600 and not text.startswith("/"):
        state_mod.set_key("pending_habit_checkin_ts", 0)
        try:
            path = await asyncio.to_thread(proactive_mod.log_checkin, text)
            shortterm_mod.add("user", text)
            shortterm_mod.add("echo", "[Habit-Check-in geloggt]")
            await msg.reply_text(f"✅ Im Habits-Vault eingetragen ({path.name}). Gut gemacht.")
            asyncio.create_task(_mirror_notion_bg(path.stem.replace("-checkin", ""), text, msg))
        except Exception as e:
            log.exception("checkin log failed")
            await msg.reply_text(f"❌ Konnte Check-in nicht loggen: {e}")
        return

    # Single LLM call: intent + (if note) classification, with recent-conversation context
    history = shortterm_mod.recent_text()
    classification = await asyncio.to_thread(classify, text, cfg, history)
    intent = classification.get("intent", "note")
    log.info("text intent: %s for %r", intent, text[:80])
    _log_event(intent, classification, text, "text")
    shortterm_mod.add("user", text)
    outcome = ""
    if intent in ("devtask", "agenttask"):
        outcome = f"{intent}: Bestätigung angefragt"

    try:
        if intent == "query":
            await _answer_query(text, cfg, msg, history)
            return
        if intent == "complete":
            await _present_completion_candidates(text, cfg, msg)
            await _maybe_answer_followup(classification, cfg, msg, history)
            return
        if intent == "event":
            await _present_event(classification, cfg, msg, ctx)
            await _maybe_answer_followup(classification, cfg, msg, history)
            return
        if intent == "mail":
            await _handle_mail(cfg, msg, ctx, text, classification)
            return
        if intent == "news":
            await _send_news(cfg, msg)
            return
        if intent == "ask":
            await _answer_ask(text, cfg, msg, history)
            return
        if intent == "status":
            await msg.reply_text(jobs_mod.status_text())
            return
        if intent == "devtask":
            await _present_devtask(classification, cfg, msg, ctx)
            return
        if intent == "agenttask":
            await _present_agenttask(classification, cfg, msg, ctx)
            return
        if intent == "prioritize":
            await _do_prioritize(text, cfg, msg)
            return
        if intent == "help":
            await msg.reply_text(HELP_TEXT, parse_mode="Markdown")
            return
        if intent in _ACTION_INTENTS:
            await _route_action(intent, text, update, ctx)
            return
        await _ingest_text(text, cfg, msg, classification=classification)
        outcome = "note geschrieben"
        await _maybe_answer_followup(classification, cfg, msg, history)
    except Exception as e:
        outcome = f"error: {e}"
        raise
    finally:
        transcript_mod.record("text", text, classification, outcome or f"-> {intent}")


def _tasks_keyboard(tasks: list) -> InlineKeyboardMarkup | None:
    if not tasks:
        return None
    rows = []
    for t in tasks:
        rows.append([
            InlineKeyboardButton(f"✅ {t.content[:35]}", callback_data=f"done:{t.id}"),
            InlineKeyboardButton("🔗", url=t.url),
        ])
    return InlineKeyboardMarkup(rows)


def _format_ingest_reply(classification: dict, tasks: list, note_path, cfg: Config) -> str:
    vault_name = classification["vault"]
    conf = classification.get("confidence", 0)
    title = classification.get("title", "")
    tags = ", ".join(classification.get("tags", []))
    rel = note_path.relative_to(cfg.vault_root)
    lines = [f"✅ *{title}*", "", f"📁 `{vault_name}` (conf {conf:.2f})"]
    if tags:
        lines.append(f"🏷️ {tags}")
    lines.append(f"📄 `{rel}`")
    if tasks:
        lines.append("")
        lines.append(f"📋 *{len(tasks)} Task(s):*")
        task_meta = classification.get("tasks", [])
        for i, t in enumerate(tasks):
            meta = task_meta[i] if i < len(task_meta) else {}
            pri = meta.get("priority", 0)
            pri_label = {4: "🔴", 3: "🟠", 2: "🟡", 1: "⚪"}.get(pri, "")
            label_str = " ".join(f"@{l}" for l in t.labels) if t.labels else ""
            lines.append(f"  {pri_label} {t.content} {label_str}".rstrip())
    return "\n".join(lines)


async def _ingest_text(text: str, cfg: Config, msg, classification: dict | None = None) -> None:
    """Treat plain text the same way as a transcript — classify + write + tasks."""
    progress = await msg.reply_text("🧠 Klassifiziere...")
    try:
        if classification is None:
            classification = await asyncio.to_thread(classify, text, cfg)
        tasks = await _create_tasks(classification)
        related = await asyncio.to_thread(find_related, text, classification["vault"], cfg)

        note_path = await asyncio.to_thread(write_note, text, classification, cfg, tasks=tasks, related=related)
        try:
            await asyncio.to_thread(
                store.upsert_note,
                cfg.data_dir / "store.db",
                str(note_path.resolve()),
                classification["vault"],
                classification.get("title", ""),
                classification.get("summary", ""),
                text, None,
            )
        except Exception as e:
            log.error("index failed: %s", e)

        await progress.edit_text(
            _format_ingest_reply(classification, tasks, note_path, cfg),
            parse_mode="Markdown",
            reply_markup=_tasks_keyboard(tasks),
        )
    except LLMError as e:
        await progress.edit_text(f"❌ Fehler: {e}")
    except Exception as e:
        log.exception("text ingest failed")
        await progress.edit_text(f"❌ Unerwarteter Fehler: {e}")


async def handle_done_callback(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q or not q.data or not q.data.startswith("done:"):
        return
    task_id = q.data.split(":", 1)[1]
    try:
        await asyncio.to_thread(td.close_task, task_id)
        await q.answer("Done ✓")
        if q.message and q.message.text:
            await q.edit_message_text(
                q.message.text + "\n\n☑️ _Task closed in Todoist_",
                parse_mode="Markdown",
            )
    except Exception as e:
        log.exception("close failed: %s", e)
        await q.answer(f"Failed: {e}", show_alert=True)


async def cmd_mail(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    await _handle_mail(cfg, update.message, ctx)


async def _send_news(cfg: Config, msg) -> None:
    progress = await msg.reply_text("📰 Hole News...")
    try:
        text = await asyncio.to_thread(news_mod.build_news_briefing, cfg)
        if len(text) > 4000:
            text = text[:3900] + "\n_(gekürzt)_"
        await progress.edit_text(text, parse_mode="Markdown", disable_web_page_preview=True)
    except Exception as e:
        log.exception("news failed")
        await progress.edit_text(f"❌ News-Fehler: {e}")


async def cmd_news(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    await _send_news(cfg, update.message)


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update, ctx.application.bot_data["cfg"]):
        return
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")


async def cmd_prioritize(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    await _do_prioritize(" ".join(ctx.args) if ctx.args else "", cfg, update.message)


def _mail_scope_hint(e: Exception) -> str:
    s = str(e).lower()
    if "insufficient" in s or "scope" in s or "403" in s:
        return ("📧 Mail-Versand noch nicht autorisiert. Einmalig im Terminal:\n"
                "`.venv/bin/python scripts/google_auth.py` (neu einloggen, Senden erlauben).")
    return f"❌ Mail-Fehler: {e}"


async def cmd_mailme(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """`/mailme` → briefing per Mail. `/mailme <thema>` → recherchiert + mailt das Ergebnis."""
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    if not gcal.is_configured():
        await update.message.reply_text("📧 Gmail nicht verbunden (erst Google-OAuth).")
        return
    arg = " ".join(ctx.args).strip() if ctx.args else ""

    if not arg or arg.lower() in ("briefing", "news", "daily"):
        progress = await update.message.reply_text("📧 Baue Briefing und maile es dir...")
        try:
            text = await asyncio.to_thread(briefing_mod.build_briefing, cfg)
            await asyncio.to_thread(gcal.send_self, "Echo — Dein Briefing", text)
            await progress.edit_text("📧 Briefing an dein Gmail geschickt.")
        except Exception as e:
            log.exception("mailme briefing failed")
            await progress.edit_text(_mail_scope_hint(e))
        return

    ack = f"🔍 Recherchiere '{arg}' im Hintergrund — maile dir das Ergebnis in ~2-3 Min."
    await update.message.reply_text(ack)
    shortterm_mod.add("echo", ack)
    jid = jobs_mod.start("mail-research", arg[:60])
    asyncio.create_task(_mail_research_bg(arg, cfg, update.message, jid))


async def _mail_research_bg(topic: str, cfg: Config, msg, jid: int | None = None) -> None:
    """Deep web research off-handler, then email the result to the user."""
    try:
        answer = await asyncio.to_thread(ask_mod.run_research, topic, cfg, "")
        await asyncio.to_thread(gcal.send_self, f"Echo — Recherche: {topic}", answer)
        await msg.reply_text(f"📧 Recherche zu '{topic}' an dein Gmail geschickt.")
    except Exception as e:
        log.exception("mailme research failed")
        await msg.reply_text(_mail_scope_hint(e))
    finally:
        if jid is not None:
            jobs_mod.finish(jid)


async def cmd_synthesize(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Weekly synthesis: stage recent notes → ingest into SecondBrain wiki → re-index into RAG."""
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    ack = ("🧠 Wochen-Synthese läuft im Hintergrund: Notizen → SecondBrain-Wiki → RAG re-indexieren. "
           "Dauert ein paar Minuten, ich melde mich.")
    await update.message.reply_text(ack, parse_mode="Markdown")
    shortterm_mod.add("echo", ack)
    jid = jobs_mod.start("synthesize", "Wochen-Synthese")
    asyncio.create_task(_synthesize_bg(cfg, update.message, jid))


async def _synthesize_bg(cfg: Config, msg, jid: int | None = None) -> None:
    try:
        res = await asyncio.to_thread(secondbrain_mod.synthesize_week, cfg)
        out = (f"🧠 Synthese fertig.\n"
               f"📥 {res['staged']} neue Notizen gestaged · 📚 {res['wiki_indexed']} Wiki-Seiten re-indexiert.")
        report = (res.get("ingest_report") or "").strip()
        if report:
            out += "\n\n" + report[:1500]
        await _safe_reply(msg, out)
    except Exception as e:
        log.exception("synthesize failed")
        await msg.reply_text(f"❌ Synthese-Fehler: {e}")
    finally:
        if jid is not None:
            jobs_mod.finish(jid)


async def _synthesis_job(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Scheduled weekly (guards to Sundays) — runs the synthesis + notifies the chat."""
    from datetime import datetime
    if datetime.now().weekday() != 6:  # 6 = Sunday
        return
    cfg: Config = ctx.application.bot_data["cfg"]
    st = state_mod.load()
    chat = st.get("chat_id")
    try:
        res = await asyncio.to_thread(secondbrain_mod.synthesize_week, cfg)
        if chat:
            await ctx.bot.send_message(
                chat,
                f"🧠 Wöchentliche Synthese: {res['staged']} Notizen → Wiki, "
                f"{res['wiki_indexed']} Seiten re-indexiert.",
            )
    except Exception as e:
        log.exception("weekly synthesis failed: %s", e)


async def _morning_nudge_job(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    from datetime import date
    st = state_mod.load()
    chat = st.get("chat_id")
    if not st.get("proactive_enabled", True) or not chat:
        return
    try:
        await ctx.bot.send_message(chat, proactive_mod.morning_text(), parse_mode="Markdown")
        state_mod.set_key("last_morning_nudge_sent", date.today().isoformat())
    except Exception as e:
        log.exception("morning nudge failed: %s", e)


async def _catch_up_tick(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Periodic safety net (runs every ~20 min). Daily jobs are SKIPPED if the Mac slept — or the
    process was down — through their fire time (APScheduler misfire grace is ~1s). This tick
    re-sends any of today's briefing / morning nudge / evening check-in that is past-due-but-unsent,
    each only inside its own time window and date-guarded so it goes out at most once per day."""
    from datetime import date, datetime
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    today = date.today().isoformat()
    st = state_mod.load()
    if not st.get("chat_id"):
        return
    if now.hour < 12:  # morning window: briefing + morning nudge
        bh, bm = (int(x) for x in st.get("briefing_time", "07:30").split(":"))
        if (st.get("briefing_enabled") and st.get("last_briefing_sent") != today
                and (now.hour, now.minute) >= (bh, bm)):
            log.info("catch-up: missed daily briefing, sending now")
            await _briefing_job(ctx)
        if (st.get("proactive_enabled", True) and st.get("last_morning_nudge_sent") != today
                and now.hour >= 8):
            log.info("catch-up: missed morning nudge, sending now")
            await _morning_nudge_job(ctx)
    elif (now.hour, now.minute) >= (21, 30):  # evening window: 21:30 to midnight
        if st.get("proactive_enabled", True) and st.get("last_evening_nudge_sent") != today:
            log.info("catch-up: missed evening nudge, sending now")
            await _evening_nudge_job(ctx)


async def _evening_nudge_job(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    st = state_mod.load()
    chat = st.get("chat_id")
    if not st.get("proactive_enabled", True) or not chat:
        return
    try:
        import time as _t
        from datetime import date
        state_mod.set_key("pending_habit_checkin_ts", _t.time())
        await ctx.bot.send_message(chat, proactive_mod.evening_text(), parse_mode="Markdown")
        state_mod.set_key("last_evening_nudge_sent", date.today().isoformat())
    except Exception as e:
        log.exception("evening nudge failed: %s", e)


def _pending_move(ctx) -> dict:
    return ctx.application.bot_data.setdefault("pending_move", {})


async def cmd_inbox(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    progress = await update.message.reply_text("📥 Prüfe Inbox...")
    candidates = await asyncio.to_thread(review_mod.review_candidates, cfg, 5)
    if not candidates:
        await progress.edit_text(f"✨ Inbox ({cfg.default_vault}) leer — nichts zu reviewen.")
        return
    await progress.edit_text(f"📥 {len(candidates)} Notiz(en) im Misc-Inbox. Review:")
    for cand in candidates:
        suggestion = await asyncio.to_thread(review_mod.suggest_vault, cand["text"], cfg)
        target = suggestion.get("vault", cfg.default_vault)
        conf = suggestion.get("confidence", 0)
        reason = suggestion.get("reason", "")
        token = uuid4().hex[:8]
        _pending_move(ctx)[token] = {"path": cand["path"], "target": target}
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton(f"→ {target} ({conf:.0%})", callback_data=f"mv:{token}"),
            InlineKeyboardButton("⏭ Skip", callback_data="cancel"),
        ]])
        await update.message.reply_text(
            f"*{cand['title']}*\n_Vorschlag: {target} — {reason[:50]}_",
            parse_mode="Markdown", reply_markup=kb,
        )


async def handle_move_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q or not q.data or not q.data.startswith("mv:"):
        return
    cfg: Config = ctx.application.bot_data["cfg"]
    token = q.data.split(":", 1)[1]
    pending = _pending_move(ctx).pop(token, None)
    if not pending:
        await q.answer("Abgelaufen")
        return
    try:
        await asyncio.to_thread(review_mod.move_note, pending["path"], pending["target"], cfg)
        await q.answer("Verschoben ✓")
        await q.edit_message_text(q.message.text + f"\n\n✅ → {pending['target']}", parse_mode="Markdown")
    except Exception as e:
        log.exception("move failed")
        await q.answer(f"Fehler: {e}", show_alert=True)


async def _send_chunked(msg, text: str) -> None:
    """Send Markdown text to Telegram, splitting on line boundaries under the 4096 limit."""
    chunk = ""
    for line in text.split("\n"):
        if len(chunk) + len(line) + 1 > 3800:
            await msg.reply_text(chunk, parse_mode="Markdown")
            chunk = ""
        chunk += line + "\n"
    if chunk.strip():
        await msg.reply_text(chunk, parse_mode="Markdown")


def _memory_vault_dir(cfg: Config) -> Path:
    """Personal_Vault if configured, else the default vault."""
    name = "Personal_Vault" if "Personal_Vault" in cfg.vaults else cfg.default_vault
    return cfg.vaults[name].path


async def cmd_memory(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    grouped = memory_mod.list_structured()
    total = sum(len(v) for v in grouped.values())
    if not total:
        await update.message.reply_text("Noch nichts gemerkt. Echo lernt beim Notizen-Aufnehmen.")
        return
    lines = [f"🧠 *Was ich über dich weiß ({total}):*", ""]
    for t, items in grouped.items():
        label = memory_mod.TYPE_LABELS.get(t, t)
        lines.append(f"{memory_mod.TYPE_ICONS.get(t, '·')} *{label}:*")
        for f in items:
            lines.append(f"  `{f['id']}` {f['text']}")
        lines.append("")
    lines.append("_Bearbeiten: `/editmemory <id> <text>` · Löschen: `/forget <id>` · "
                 "Obsidian-Übersicht: `/memorymd`_")
    await _send_chunked(update.message, "\n".join(lines))


async def cmd_editmemory(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    args = ctx.args or []
    if len(args) < 2 or not args[0].isdigit():
        await update.message.reply_text(
            "Nutzung: `/editmemory <id> <neuer text>`", parse_mode="Markdown")
        return
    fact_id = int(args[0])
    new_text = " ".join(args[1:]).strip()
    if memory_mod.edit_fact(fact_id, new_text):
        await update.message.reply_text(f"✏️ Fakt `{fact_id}` aktualisiert.", parse_mode="Markdown")
    else:
        await update.message.reply_text(f"Kein Fakt mit id `{fact_id}`.", parse_mode="Markdown")


async def cmd_mergememory(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    args = ctx.args or []
    ids = [int(a) for a in args if a.isdigit()]
    if len(ids) < 2:
        await update.message.reply_text(
            "Nutzung: `/mergememory <id-behalten> <id> [id ...]`", parse_mode="Markdown")
        return
    survivor = memory_mod.merge_facts(ids)
    if survivor:
        await update.message.reply_text(
            f"🔗 {len(ids) - 1} Fakt(en) in `{survivor['id']}` zusammengeführt.",
            parse_mode="Markdown")
    else:
        await update.message.reply_text("Zusammenführen fehlgeschlagen (id unbekannt?).")


async def cmd_memorymd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    path = await asyncio.to_thread(memory_mod.export_markdown, _memory_vault_dir(cfg))
    try:
        rel = path.relative_to(cfg.vault_root)
    except ValueError:
        rel = path
    await update.message.reply_text(f"📝 Übersicht aktualisiert: `{rel}`", parse_mode="Markdown")


async def cmd_forget(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    arg = " ".join(ctx.args).strip() if ctx.args else ""
    if not arg:
        await update.message.reply_text(
            "Nutzung: `/forget <id>` oder `/forget <stichwort>`", parse_mode="Markdown")
        return
    if arg.isdigit() and memory_mod.delete_fact(int(arg)):
        await update.message.reply_text(f"🗑️ Fakt `{arg}` gelöscht.", parse_mode="Markdown")
        return
    n = memory_mod.forget(arg)
    await update.message.reply_text(f"🗑️ {n} Fakt(en) gelöscht.")


def _save_draft_note(brief: str, result: dict, cfg: Config):
    """Persist a generated draft into Self_Vault/inbox so it is not lost."""
    import re
    from datetime import datetime, timezone
    base = agents_mod.self_vault_dir(cfg) / "inbox"
    base.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).astimezone()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", brief.lower()).strip("-")[:50] or "draft"
    path = base / f"{now.strftime('%Y-%m-%d-%H%M')}-draft-{slug}.md"
    fm = [
        "---",
        f'created: {now.isoformat(timespec="seconds")}',
        "source: draft-agent",
        f'agent: {result.get("agent", "")}',
        "tags: [draft, anschreiben]",
        "---",
        "",
        f"# Draft: {brief}",
        "",
        result.get("text", ""),
        "",
    ]
    path.write_text("\n".join(fm), encoding="utf-8")
    return path


async def cmd_draft(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate a humanized German draft in the user's style from the Self_Vault.

    Usage: /draft <brief>
    First line = brief (what to write). Any following lines = job posting / source material.
    """
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        await update.message.reply_text("Nicht autorisiert.")
        return
    raw = " ".join(ctx.args) if ctx.args else ""
    # Prefer the full message text so multi-line postings survive (args collapses whitespace).
    full = (update.message.text or "").split(maxsplit=1)
    raw = full[1].strip() if len(full) > 1 else raw.strip()
    if not raw:
        await update.message.reply_text(
            "Nutzung: `/draft <briefing>`\n"
            "Erste Zeile = was geschrieben werden soll, danach optional die Stellenausschreibung.\n"
            "Bsp: `/draft Anschreiben für Founders Associate bei Moonscale`",
            parse_mode="Markdown",
        )
        return

    lines = raw.split("\n", 1)
    brief = lines[0].strip()
    posting = lines[1].strip() if len(lines) > 1 else ""

    progress = await update.message.reply_text("✍️ Schreibe Entwurf in deinem Stil...")
    try:
        result = await asyncio.to_thread(agents_mod.draft, brief, cfg, posting)
        text = (result.get("text") or "").strip() or "Kein Entwurf erzeugt."

        note_path = None
        try:
            note_path = await asyncio.to_thread(_save_draft_note, brief, result, cfg)
        except Exception as e:
            log.error("saving draft note failed: %s", e)

        footer = f"\n\n_🤖 {result.get('agent', 'Draft')}-Agent"
        if result.get("missing_self_vault"):
            footer += " · ⚠️ Self_Vault leer"
        if note_path:
            try:
                footer += f" · 📄 `{note_path.relative_to(cfg.vault_root)}`"
            except Exception:
                pass
        footer += "_"

        out = text + footer
        if len(out) > 4000:
            await update.message.reply_text(text[:4000])
            if len(text) > 4000:
                await update.message.reply_text(text[4000:8000])
            await progress.edit_text(footer.strip(), parse_mode="Markdown")
        else:
            await progress.edit_text(out, parse_mode="Markdown")
    except Exception as e:
        log.exception("draft failed")
        await progress.edit_text(f"❌ Fehler: {e}")


async def handle_clean_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q or not q.data or not q.data.startswith("clean:"):
        return
    token = q.data.split(":", 1)[1]
    ids = _pending_clean(ctx).pop(token, None)
    if not ids:
        await q.answer("Abgelaufen")
        return
    trashed = 0
    for mid in ids:
        try:
            await asyncio.to_thread(gcal.trash_mail, mid)
            trashed += 1
        except Exception as e:
            log.error("trash failed %s: %s", mid, e)
    await q.answer(f"{trashed} in Papierkorb")
    await q.edit_message_text(f"🗑️ {trashed} Mails in Papierkorb (wiederherstellbar in Gmail).")


async def handle_mail_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q or not q.data:
        return
    kind, _, token = q.data.partition(":")
    pending = _pending_mail(ctx).get(token)
    if not pending:
        await q.answer("Abgelaufen")
        return

    if kind == "mailt":
        created = 0
        for t in pending.get("tasks", []):
            try:
                await asyncio.to_thread(
                    td.create_task,
                    content=t["content"],
                    due_string=t.get("due_string") or None,
                    priority=t.get("priority") or None,
                )
                created += 1
            except Exception as e:
                log.error("mail task failed: %s", e)
        await q.answer(f"{created} Tasks")
        await q.edit_message_text(q.message.text + f"\n\n✅ {created} Task(s) angelegt.", parse_mode="Markdown")
    elif kind == "maile":
        from datetime import datetime
        created = 0
        for e in pending.get("events", []):
            try:
                start = datetime.fromisoformat(e["start"])
                end = datetime.fromisoformat(e["end"]) if e.get("end") else None
                await asyncio.to_thread(gcal.create_event, e["summary"], start, end, "", e.get("location", ""))
                created += 1
            except Exception as ex:
                log.error("mail event failed: %s", ex)
        await q.answer(f"{created} Termine")
        await q.edit_message_text(q.message.text + f"\n\n✅ {created} Termin(e) eingetragen.", parse_mode="Markdown")


async def handle_event_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q or not q.data or not q.data.startswith("ev:"):
        return
    token = q.data.split(":", 1)[1]
    pending = _pending_events(ctx).pop(token, None)
    if not pending:
        await q.answer("Abgelaufen")
        await q.edit_message_text("⌛ Termin-Anfrage abgelaufen. Nochmal sagen.")
        return

    from datetime import datetime
    try:
        start_dt = datetime.fromisoformat(pending["start"])
        end_dt = datetime.fromisoformat(pending["end"]) if pending.get("end") else None
        result = await asyncio.to_thread(
            gcal.create_event,
            pending["summary"], start_dt, end_dt,
            "", pending.get("location", ""),
        )
        await q.answer("Eingetragen ✓")
        link = result.get("htmlLink", "")
        await q.edit_message_text(
            f"📅 *{result['summary']}* eingetragen.\n[Im Kalender öffnen]({link})",
            parse_mode="Markdown",
        )
    except Exception as e:
        log.exception("create event failed")
        await q.answer(f"Fehler: {e}", show_alert=True)
        await q.edit_message_text(f"❌ Konnte Termin nicht eintragen: {e}")


async def handle_completion_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q or not q.data:
        return
    data = q.data

    if data == "cancel":
        await q.answer("Abgebrochen")
        _clear_dialog(ctx)  # also abort any pending interactive devtask/agenttask dialog
        await q.edit_message_text("✗ Abgebrochen.")
        return

    if data.startswith("close:"):
        task_ids = [data.split(":", 1)[1]]
    elif data.startswith("closeall:"):
        task_ids = [t for t in data.split(":", 1)[1].split(",") if t]
    else:
        return

    closed, failed = [], []
    for tid in task_ids:
        try:
            await asyncio.to_thread(td.close_task, tid)
            closed.append(tid)
        except Exception as e:
            log.error("close %s failed: %s", tid, e)
            failed.append(tid)

    await q.answer(f"{len(closed)} geschlossen")
    txt = f"☑️ {len(closed)} Task(s) als erledigt markiert."
    if failed:
        txt += f"\n⚠️ {len(failed)} fehlgeschlagen."
    await q.edit_message_text(txt)


async def cmd_indexdocs(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """(Re-)build the document index from disk + email. Idempotent."""
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    progress = await update.message.reply_text("📚 Indexiere Dokumente (Disk + Mail)...")
    try:
        disk = await asyncio.to_thread(docsearch_mod.index_disk, cfg, None)
        mail = await asyncio.to_thread(docsearch_mod.index_email_docs, cfg)
        lines = [
            "📚 *Dokument-Index aktualisiert*",
            f"📂 Disk: {disk['indexed']} indexiert, {disk['skipped']} übersprungen",
            f"   _{disk.get('root','')}_",
        ]
        if mail.get("blocked"):
            lines.append("📧 Mail: übersprungen (Gmail nicht verbunden)")
        else:
            lines.append(f"📧 Mail: {mail['indexed']} Nachrichten mit Anhängen")
        await progress.edit_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        log.exception("indexdocs failed")
        await progress.edit_text(f"❌ Index-Fehler: {e}")


async def cmd_finddoc(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Search indexed documents, summarize the top hit, link results into a vault."""
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    query = " ".join(ctx.args).strip() if ctx.args else ""
    if not query:
        await update.message.reply_text(
            "Nutzung: `/finddoc <suche>` — z.B. `/finddoc Steuerbescheid 2024`.\n"
            "Index zuerst mit `/indexdocs` bauen.",
            parse_mode="Markdown",
        )
        return
    progress = await update.message.reply_text("🔎 Durchsuche Dokumente...")
    try:
        result = await asyncio.to_thread(docsearch_mod.find_docs, query, cfg)
    except Exception as e:
        log.exception("finddoc failed")
        await progress.edit_text(f"❌ Fehler: {e}")
        return

    if not result.get("indexed"):
        await progress.edit_text("Kein Dokument-Index vorhanden. Erst `/indexdocs` ausführen.", parse_mode="Markdown")
        return
    hits = result.get("hits", [])
    if not hits:
        await progress.edit_text("Keine passenden Dokumente gefunden.")
        return

    note_path = None
    try:
        note_path = await asyncio.to_thread(docsearch_mod.write_doc_note, query, result, cfg)
    except Exception as e:
        log.error("doc note write failed: %s", e)

    lines = [f"📄 *{len(hits)} Treffer für* _{query}_", "", result.get("summary", ""), "", "*Dokumente:*"]
    for h in hits:
        if h.vault == "email":
            lines.append(f"  📧 {h.title[:60]} — _{h.summary[:50]}_")
        else:
            lines.append(f"  📄 {h.title[:60]}")
    if note_path:
        try:
            rel = note_path.relative_to(cfg.vault_root)
            lines.append(f"\n📝 Notiz: `{rel}`")
        except Exception:
            pass
    text = "\n".join(lines)
    if len(text) > 4000:
        text = text[:3900] + "\n_(gekürzt)_"
    await progress.edit_text(text, parse_mode="Markdown")


async def cmd_podcast(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Turn the daily briefing into a German audio podcast and send it as voice/audio."""
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        await update.message.reply_text("Nicht autorisiert.")
        return
    if update.effective_chat:
        state_mod.set_key("chat_id", update.effective_chat.id)
    progress = await update.message.reply_text("🎙️ Baue Podcast (Briefing → Skript → Audio)...")
    jid = jobs_mod.start("podcast", "Briefing-Podcast")
    try:
        result = await asyncio.to_thread(podcast_mod.build_podcast, cfg)
        await progress.edit_text(
            f"🎧 Podcast fertig ({result.duration:.0f}s, {result.num_segments} Beiträge, {result.backend}).",
        )
        with result.path.open("rb") as f:
            await update.message.reply_audio(
                audio=f,
                title="Echo Daily",
                performer="Echo",
                caption="🎧 Dein Tages-Briefing als Podcast.",
            )
    except Exception as e:
        log.exception("podcast failed")
        await progress.edit_text(f"❌ Podcast-Fehler: {e}")
    finally:
        jobs_mod.finish(jid)


async def cmd_overview(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Aggregate everything fed into Echo → refresh the Obsidian dashboard + reply with a summary."""
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    progress = await update.message.reply_text("📊 Baue Übersicht...")
    try:
        stats = await asyncio.to_thread(overview_mod.aggregate, cfg)
        await asyncio.to_thread(overview_mod.write_dashboard, cfg, stats)
        text = overview_mod.build_telegram(stats)
        if len(text) > 4000:
            text = text[:3900] + "\n_(gekürzt)_"
        await progress.edit_text(text, parse_mode="Markdown")
    except Exception as e:
        log.exception("overview failed")
        await progress.edit_text(f"❌ Übersicht-Fehler: {e}")


async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    progress = await update.message.reply_text("📊 Berechne Fortschritt...")
    try:
        await asyncio.to_thread(stats_mod.backfill, cfg)
        stats = await asyncio.to_thread(stats_mod.compute, cfg)
        summary = stats_mod.format_summary(stats)
        chart_path = cfg.data_dir / "stats_chart.png"
        png = await asyncio.to_thread(stats_mod.render_chart, stats, chart_path)
        if png and png.exists():
            await progress.delete()
            with png.open("rb") as fh:
                await update.message.reply_photo(photo=fh, caption=summary, parse_mode="Markdown")
        else:
            await progress.edit_text(summary, parse_mode="Markdown")
    except Exception as e:
        log.exception("stats failed")
        await progress.edit_text(f"❌ Stats-Fehler: {e}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # silence httpx URL leak (contains bot token)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cfg = Config.load()
    log.info("loaded %d vaults: %s", len(cfg.vaults), list(cfg.vaults))
    log.info("default vault: %s", cfg.default_vault)
    log.info("whisper model: %s", cfg.whisper_model_path)

    req = HTTPXRequest(connect_timeout=30.0, read_timeout=30.0, write_timeout=30.0, pool_timeout=10.0)
    app = (
        ApplicationBuilder()
        .token(cfg.telegram_token)
        .request(req)
        .get_updates_request(req)
        .build()
    )
    app.bot_data["cfg"] = cfg
    # Ensure vector store schema exists before serving
    store.init_schema(cfg.data_dir / "store.db")
    events_mod.init_schema()
    # Bridge: index the curated SecondBrain wiki into Echo's RAG (high-signal pages)
    try:
        n = secondbrain_mod.index_wiki(cfg, reindex=True)
        log.info("indexed %d SecondBrain wiki pages into RAG", n)
    except Exception as e:
        log.warning("SecondBrain wiki index at startup failed: %s", e)

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("ask", cmd_ask))
    app.add_handler(CommandHandler("briefing", cmd_briefing))
    app.add_handler(CommandHandler("briefingtime", cmd_briefingtime))
    app.add_handler(CommandHandler("mail", cmd_mail))
    app.add_handler(CommandHandler("news", cmd_news))
    app.add_handler(CommandHandler("memory", cmd_memory))
    app.add_handler(CommandHandler("forget", cmd_forget))
    app.add_handler(CommandHandler("editmemory", cmd_editmemory))
    app.add_handler(CommandHandler("mergememory", cmd_mergememory))
    app.add_handler(CommandHandler("memorymd", cmd_memorymd))
    app.add_handler(CommandHandler("inbox", cmd_inbox))
    app.add_handler(CommandHandler("draft", cmd_draft))
    app.add_handler(CommandHandler("indexdocs", cmd_indexdocs))
    app.add_handler(CommandHandler("finddoc", cmd_finddoc))
    app.add_handler(CommandHandler("podcast", cmd_podcast))
    app.add_handler(CommandHandler("overview", cmd_overview))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("mailme", cmd_mailme))
    app.add_handler(CommandHandler("synthesize", cmd_synthesize))
    app.add_handler(CommandHandler("prioritize", cmd_prioritize))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CommandHandler("voice", cmd_voice))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_done_callback, pattern=r"^done:"))
    app.add_handler(CallbackQueryHandler(handle_event_callback, pattern=r"^ev:"))
    app.add_handler(CallbackQueryHandler(handle_mail_callback, pattern=r"^(mailt:|maile:)"))
    app.add_handler(CallbackQueryHandler(handle_clean_callback, pattern=r"^clean:"))
    app.add_handler(CallbackQueryHandler(handle_move_callback, pattern=r"^mv:"))
    app.add_handler(CallbackQueryHandler(handle_interactive_go, pattern=r"^go:"))
    app.add_handler(CallbackQueryHandler(handle_completion_callback, pattern=r"^(close:|closeall:|cancel$)"))

    _reschedule_briefing(app)

    # Weekly SecondBrain synthesis — fires daily 20:00, the job itself runs only on Sundays.
    from datetime import time as _dtime
    from zoneinfo import ZoneInfo as _ZoneInfo
    app.job_queue.run_daily(
        _synthesis_job,
        time=_dtime(hour=20, minute=0, tzinfo=_ZoneInfo("Europe/Berlin")),
        name="weekly_synthesis",
    )
    # Proactive habit nudges (Habits_Vault-grounded). Toggle via state "proactive_enabled".
    _tz = _ZoneInfo("Europe/Berlin")
    # misfire_grace_time: still fire if the loop was briefly busy at the exact time (default ~1s
    # silently drops it). The catch_up_tick is the wider net; this keeps the on-time job forgiving.
    _grace = {"misfire_grace_time": 3600}
    app.job_queue.run_daily(_morning_nudge_job, time=_dtime(hour=8, minute=0, tzinfo=_tz),
                            name="morning_nudge", job_kwargs=_grace)
    app.job_queue.run_daily(_evening_nudge_job, time=_dtime(hour=21, minute=30, tzinfo=_tz),
                            name="evening_nudge", job_kwargs=_grace)
    # Safety net for briefing/nudges missed because the Mac slept (or the process was down) at the
    # scheduled time — APScheduler does not retro-fire. Re-checks every 20 min, date-guarded.
    app.job_queue.run_repeating(_catch_up_tick, interval=1200, first=8, name="catch_up_tick")

    log.info("Echo bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
