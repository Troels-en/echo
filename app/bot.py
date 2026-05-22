"""Telegram bot entrypoint. Voice messages → transcribe → classify → write note + Todoist task."""
from __future__ import annotations

import asyncio
import logging
import subprocess
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
from . import store, rag, ask as ask_mod, intent as intent_mod, gcal, briefing as briefing_mod, state as state_mod, mailtriage, memory as memory_mod, news as news_mod, review as review_mod

log = logging.getLogger(__name__)


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
        await progress.edit_text(text, parse_mode="Markdown")
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
    cfg: Config = ctx.application.bot_data["cfg"]
    st = state_mod.load()
    if not st.get("briefing_enabled") or not st.get("chat_id"):
        return
    try:
        text = await asyncio.to_thread(briefing_mod.build_briefing, cfg)
        await ctx.bot.send_message(st["chat_id"], text, parse_mode="Markdown")
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

    try:
        tg_file = await voice.get_file()
        await tg_file.download_to_drive(custom_path=str(ogg_path))

        await progress.edit_text("🎙️ Transkribiere...")
        wav_path = await asyncio.to_thread(_to_wav, ogg_path)
        transcript = await asyncio.to_thread(transcribe, wav_path, cfg.whisper_model_path)
        if not transcript.strip():
            await progress.edit_text("⚠️ Leere Transkription.")
            return

        await progress.edit_text(f"🧠 Verstehe...\n\n_{transcript[:200]}_", parse_mode="Markdown")
        # Single LLM call: intent + (if note) classification
        classification = await asyncio.to_thread(classify, transcript, cfg)
        intent = classification.get("intent", "note")
        log.info("voice intent: %s", intent)
        if intent == "query":
            await progress.delete()
            await _answer_query(transcript, cfg, msg)
            return
        if intent == "complete":
            await progress.delete()
            await _present_completion_candidates(transcript, cfg, msg)
            return
        if intent == "event":
            await progress.delete()
            await _present_event(classification, cfg, msg, ctx)
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
            await _answer_ask(transcript, cfg, msg)
            return

        tasks = await _create_tasks(classification)
        related = await asyncio.to_thread(find_related, transcript, classification["vault"], cfg)

        note_path = await asyncio.to_thread(
            write_note, transcript, classification, cfg, tasks=tasks, related=related,
        )

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
    except (TranscribeError, LLMError) as e:
        log.exception("voice handler failed")
        await progress.edit_text(f"❌ Fehler: {e}")
    except Exception as e:
        log.exception("unexpected error")
        await progress.edit_text(f"❌ Unerwarteter Fehler: {e}")
    finally:
        for p in [ogg_path, ogg_path.with_suffix(".wav")]:
            p.unlink(missing_ok=True)


async def _answer_query(question: str, cfg: Config, msg) -> None:
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
        await progress.edit_text(out, parse_mode="Markdown")
    except Exception as e:
        log.exception("ask failed")
        await progress.edit_text(f"❌ Fehler: {e}")


async def _answer_ask(question: str, cfg: Config, msg) -> None:
    """General question → LLM (with optional web research) → reply + always save to vault."""
    progress = await msg.reply_text("🤔 Denke nach...")
    try:
        result = await asyncio.to_thread(ask_mod.smart_answer, question, cfg)
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
                rel = note_path.relative_to(cfg.vault_root)
                footer += f" · 📄 `{rel}`"
            except Exception:
                pass
        footer += "_"

        out = answer + footer
        if len(out) > 4000:
            out = answer[: 3900 - len(footer)] + "\n\n_(gekürzt)_" + footer
        await progress.edit_text(out, parse_mode="Markdown")
    except Exception as e:
        log.exception("ask (general) failed")
        await progress.edit_text(f"❌ Fehler: {e}")


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


async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Plain-text message → intent classifier → route."""
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    msg = update.message
    text = (msg.text or "").strip()
    if not text:
        return

    # Single LLM call: intent + (if note) classification
    classification = await asyncio.to_thread(classify, text, cfg)
    intent = classification.get("intent", "note")
    log.info("text intent: %s for %r", intent, text[:80])

    if intent == "query":
        await _answer_query(text, cfg, msg)
        return
    if intent == "complete":
        await _present_completion_candidates(text, cfg, msg)
        return
    if intent == "event":
        await _present_event(classification, cfg, msg, ctx)
        return
    if intent == "mail":
        await _handle_mail(cfg, msg, ctx, text, classification)
        return
    if intent == "news":
        await _send_news(cfg, msg)
        return
    if intent == "ask":
        await _answer_ask(text, cfg, msg)
        return
    await _ingest_text(text, cfg, msg, classification=classification)


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


async def cmd_memory(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    facts = memory_mod.all_facts()
    if not facts:
        await update.message.reply_text("Noch nichts gemerkt. Echo lernt beim Notizen-Aufnehmen.")
        return
    by_type: dict[str, list[str]] = {}
    for f in facts:
        by_type.setdefault(f["type"], []).append(f["text"])
    lines = [f"🧠 *Was ich über dich weiß ({len(facts)}):*", ""]
    icons = {"person": "👤", "preference": "❤️", "project": "🚀", "pattern": "🔁", "fact": "·"}
    for t, items in by_type.items():
        lines.append(f"{icons.get(t,'·')} *{t}:*")
        for it in items:
            lines.append(f"  • {it}")
    lines.append("\n_`/forget <stichwort>` zum Löschen._")
    txt = "\n".join(lines)
    await update.message.reply_text(txt[:4000], parse_mode="Markdown")


async def cmd_forget(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = ctx.application.bot_data["cfg"]
    if not _is_allowed(update, cfg):
        return
    sub = " ".join(ctx.args).strip() if ctx.args else ""
    if not sub:
        await update.message.reply_text("Nutzung: `/forget <stichwort>`", parse_mode="Markdown")
        return
    n = memory_mod.forget(sub)
    await update.message.reply_text(f"🗑️ {n} Fakt(en) gelöscht.")


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


async def handle_completion_callback(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q or not q.data:
        return
    data = q.data

    if data == "cancel":
        await q.answer("Abgebrochen")
        await q.edit_message_text("✗ Abgebrochen — nichts geschlossen.")
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

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("ask", cmd_ask))
    app.add_handler(CommandHandler("briefing", cmd_briefing))
    app.add_handler(CommandHandler("briefingtime", cmd_briefingtime))
    app.add_handler(CommandHandler("mail", cmd_mail))
    app.add_handler(CommandHandler("news", cmd_news))
    app.add_handler(CommandHandler("memory", cmd_memory))
    app.add_handler(CommandHandler("forget", cmd_forget))
    app.add_handler(CommandHandler("inbox", cmd_inbox))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_done_callback, pattern=r"^done:"))
    app.add_handler(CallbackQueryHandler(handle_event_callback, pattern=r"^ev:"))
    app.add_handler(CallbackQueryHandler(handle_mail_callback, pattern=r"^(mailt:|maile:)"))
    app.add_handler(CallbackQueryHandler(handle_clean_callback, pattern=r"^clean:"))
    app.add_handler(CallbackQueryHandler(handle_move_callback, pattern=r"^mv:"))
    app.add_handler(CallbackQueryHandler(handle_completion_callback, pattern=r"^(close:|closeall:|cancel$)"))

    _reschedule_briefing(app)

    log.info("Echo bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
