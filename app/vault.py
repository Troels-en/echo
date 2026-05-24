"""Classify transcript and write markdown note into the right vault.
Optionally also creates a Todoist task if the note is actionable.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import Config, VaultSpec, REPO_ROOT
from .llm import call_json
from . import store, memory

log = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.7
# L2 distance on normalized vectors: d = 2(1-cos). MiniLM clusters related content
# around 0.9-1.1; d<1.1 ≈ cos>0.45 captures related notes without linking noise.
RELATED_MAX_DISTANCE = 1.1
RELATED_MAX_LINKS = 5


def _load_vault_routing() -> dict:
    """Load vaults.yml; fall back to the shipped example for fresh clones."""
    path = REPO_ROOT / "config" / "vaults.yml"
    if not path.exists():
        path = REPO_ROOT / "config" / "vaults.example.yml"
    with path.open() as f:
        return yaml.safe_load(f)


CLASSIFY_PROMPT = """You are the router AND classifier for a personal voice-vault assistant. Do BOTH in one step.

NOW (user local time, Europe/Berlin): {now}

WHAT YOU KNOW ABOUT THE USER (use to disambiguate names/projects; do NOT re-extract these):
{memory}

RECENT CONVERSATION (most recent last; use ONLY to resolve references like "das", "dazu",
"weißt du was ich meine", follow-up questions — do NOT re-file old turns):
{history}

STEP 1 — INTENT. Decide what the user wants:
- "query"    — asking a question wanting an answer from their notes ("Was waren meine besten Ideen?", "Wie war mein letzter Lauf?")
- "complete" — reporting finished task(s) ("Habe X gekauft", "Bewerbung verschickt", "die letzten drei Punkte erledigt")
- "event"    — wants a CALENDAR appointment with a specific date/time ("Trag morgen 15 Uhr Zahnarzt ein", "Meeting Donnerstag 10 Uhr")
- "mail"      — wants the assistant to check/read/triage their email ("Check meine Mails", "Was ist in meinem Postfach", "übernimm meine Mails")
- "news"      — wants news / latest developments ("Was gibt's Neues", "news", "Entwicklungen in KI", "Accelerator-Deadlines")
- "ask"       — a GENERAL question about the world, facts, how-to, research — NOT about the user's own notes ("Was ist ein SAFE-Note?", "Erklär mir Vector-DBs", "Recherchier die besten Whisper-Modelle", "Wie funktioniert X?"). Use this when answering needs general knowledge or live research, not the user's vault.
- "podcast"   — wants the briefing as an AUDIO podcast ("mach mir einen Podcast", "lies mir das als Podcast vor", "Audio-Briefing")
- "overview"  — wants a dashboard/overview of everything fed into Echo ("zeig mir eine Übersicht", "was hab ich alles drin", "Dashboard")
- "stats"     — wants usage/progress statistics ("zeig mir meine Stats", "wie viel nutze ich dich", "mein Fortschritt", "XP")
- "draft"     — wants a written draft in the user's style ("schreib mir ein Anschreiben für …", "entwirf eine Mail an …", "formulier mir …"). The full request text is the brief.
- "finddoc"   — wants to find/search a DOCUMENT on disk or in email ("finde mein Steuerdokument", "such die Gehaltsabrechnung", "wo ist der Vertrag von …")
- "synthesize"— wants to curate/synthesize their captured notes into the long-term knowledge wiki ("synthetisiere meine Notizen", "bau das ins Wiki ein", "kuratier meine Woche", "verarbeite meine letzten Notizen", "update mein Wissens-Wiki"). NOTE: this runs automatically every week — only route here on an explicit curate/wiki request. A casual "was hab ich diese Woche gemacht / fass meine Woche zusammen" is a "query" (read + summarize), NOT synthesize.
- "mailme"    — wants something emailed to themselves ("maile mir das Briefing", "schick mir das per Mail", "per Email an mich")
- "status"    — asking whether a background job is still running / how long it takes / if you're done ("wie lange noch", "läuft das noch", "bist du fertig", "status", "wie lange dauert das")
- "devtask"   — wants Echo to trigger a real CODING/dev task via Claude Code in one of their repos ("baue Feature X in repo Y", "fix den Bug in Z", "lass Claude Code … umsetzen", "implementier … in <projekt>"). Only for actual code/dev work on a named project, NOT general notes.
- "agenttask"  — wants Echo to EXECUTE a multi-step action on their knowledge/data using its tools (Notion + Obsidian vaults + files), NOT just note it. e.g. "zieh meine Notion-Habits in den Vault", "übertrag X aus Notion nach Y", "räum SecondBrain auf", "fass meine Finance-Notizen zu einer Seite zusammen", "sync …". Use this when the user asks Echo to DO/transfer/organize/consolidate something across Notion/vaults — not for code (devtask), not for a simple capture (note).
- "prioritize"— wants their OPEN tasks ranked / what to do first ("was soll ich zuerst machen", "priorisier meine Tasks", "was ist am wichtigsten", "womit anfangen", "was ist dringend")
- "help"      — wants to know what Echo can do ("was kannst du", "hilfe", "welche Befehle", "help", "was geht alles")
- "note"     — capturing a new thought, idea, observation, or future task (default)

STEP 2 — only if intent is "note", classify it into a vault AND extract ALL distinct tasks.

VAULTS (name → typical content):
{vault_list}

LABELS (apply by MEANING, not by vault):
{labels}

INPUT:
\"\"\"{transcript}\"\"\"

Return ONLY a JSON object, no prose:
{{
  "intent": "query" | "complete" | "event" | "mail" | "news" | "ask" | "podcast" | "overview" | "stats" | "draft" | "finddoc" | "synthesize" | "mailme" | "status" | "devtask" | "prioritize" | "help" | "note",
  "dev_repo": "<if intent=devtask: the project/repo name the user named, else empty>",
  "dev_task": "<if intent=devtask: the concrete coding task in one clear sentence, else empty>",
  "agent_task": "<if intent=agenttask: the action to execute in one clear sentence, else empty>",
  "also_question": "<if the message ALSO contains a separate question on top of a note/task/event/complete (e.g. 'priorisierst du Tasks? Übrigens ich muss mich bewerben'), put that question here so it gets answered too; else empty>",
  "intent_confidence": <float 0..1>,
  "vault": "<vault name from list, or empty if intent != note>",
  "confidence": <float 0..1>,
  "title": "<short imperative title for the note, max 8 words>",
  "tags": ["<3 to 5 short obsidian tags>"],
  "summary": "<one sentence>",
  "tasks": [
    {{
      "content": "<ONE concrete action, imperative>",
      "due_string": "<'today'|'tomorrow'|'next monday'..'next sunday'|'<N> days'|'<DD MMM>' or empty>",
      "priority": <4=urgent+important, 3=important not urgent, 2=urgent not important, 1=neither>,
      "labels": [<0..N labels chosen by MEANING of THIS task>]
    }}
  ],
  "event": {{
    "summary": "<event title if intent=event, else empty>",
    "start": "<ISO 8601 datetime resolved against NOW, e.g. 2026-05-22T15:00:00, or empty>",
    "end": "<ISO 8601 or empty (defaults to +1h)>",
    "location": "<if mentioned, else empty>"
  }},
  "new_facts": [
    {{"text": "<durable fact about the user worth remembering long-term>", "type": "person|preference|project|pattern"}}
  ],
  "mail_action": {{
    "action": "<if intent=mail: 'triage' (general check) | 'search' (looking for specific mail) | 'clean' (tidy inbox/remove junk), else empty>",
    "search_terms": "<Gmail search query if action=search, e.g. 'from:katha' or 'Langdock', else empty>"
  }}
}}

Rules:
- Default intent to "note" if unsure. "query" only if asking for info FROM the user's OWN notes. "ask" if it's a general/world/research question not about their notes. "complete" only if reporting DONE work.
- MULTI-INTENT: a message can hold BOTH a capture/action (note/task/event/complete) AND a separate question. Pick the capture/action as the primary "intent", AND put the question into "also_question" so it gets answered too. Don't drop the question. If the message is ONLY a question, use "ask"/"query" as intent and leave "also_question" empty. "event" only if a specific date/time for an appointment is given — resolve relative dates ("morgen", "Donnerstag") against NOW into absolute ISO.
- CRITICAL — SPLIT TASKS: if the input mentions multiple distinct actions, output ONE task object PER action. "Ich muss X und Y" → two tasks. Never merge two actions into one task.
- A task is a concrete action to DO. Ideas, reflections, observations → NO task (empty "tasks" array).
- Choose labels PER TASK by meaning. A personal job application = "Karriere", NOT "Career-Buddy". Career-Buddy label is ONLY for the user's Career-Buddy product itself.
- priority defaults to 3; use 4 only if urgent/deadline/today.
- Pick "Misc_Vault" if no vault fits. Be conservative with confidence.
- new_facts: ONLY durable, reusable facts. Empty list if nothing durable. Never restate facts already in WHAT YOU KNOW.
  Types: "person" (e.g. "Katha is a recruiter at Langdock"), "project" (e.g. "building Career-Buddy startup"), "preference", "pattern".
  IMPORTANT — capture CORRECTIONS as type "pattern": if the user corrects routing/labeling or states a rule ("Bewerbungen gehören nach Karriere nicht Career-Buddy", "Lauf-Notizen sind Fitness nicht Journal", "nenn das immer X"), save it as a pattern fact so you route correctly next time."""


def _slug(text: str, max_len: int = 50) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len] or "note"


def _vault_list(vaults: dict[str, VaultSpec]) -> str:
    lines = []
    for v in vaults.values():
        kw = ", ".join(v.keywords) if v.keywords else "(fallback)"
        lines.append(f"- {v.name}: {kw}")
    return "\n".join(lines)


def classify(transcript: str, cfg: Config, history: str = "") -> dict:
    """Single LLM call: returns intent + (if note) full vault classification.
    history: recent conversation turns (see app/shortterm) to resolve follow-ups."""
    routing = _load_vault_routing()
    label_defs = routing.get("labels", {})
    if label_defs:
        labels_str = "\n".join(f"- {k}: {v}" for k, v in label_defs.items())
    else:
        labels_str = ", ".join(routing.get("available_labels", []))

    now = datetime.now(timezone.utc).astimezone()
    mem_context = memory.context() or "(noch nichts bekannt)"
    prompt = CLASSIFY_PROMPT.format(
        vault_list=_vault_list(cfg.vaults),
        labels=labels_str,
        now=now.strftime("%A %Y-%m-%d %H:%M"),
        memory=mem_context,
        history=history or "(kein vorheriger Kontext)",
        transcript=transcript.replace('"""', "'''"),
    )
    result = call_json(prompt, primary=cfg.llm_primary, fallback=cfg.llm_fallback)

    new_facts = result.get("new_facts") or []
    if new_facts:
        try:
            n = memory.add_facts(new_facts)
            if n:
                log.info("memory: +%d facts", n)
        except Exception as e:
            log.warning("memory add failed: %s", e)

    intent = result.get("intent", "note")
    if intent not in ("query", "complete", "note", "event", "mail", "news", "ask",
                      "podcast", "overview", "stats", "draft", "finddoc", "synthesize",
                      "mailme", "status", "devtask", "agenttask", "prioritize", "help"):
        intent = "note"
    result["intent"] = intent

    if intent != "note":
        return result  # vault routing irrelevant for query/complete/event/ask

    vault = result.get("vault") or cfg.default_vault
    if vault not in cfg.vaults:
        log.warning("LLM returned unknown vault %r → %s", vault, cfg.default_vault)
        vault = cfg.default_vault
        result["confidence"] = min(result.get("confidence", 0.0), 0.5)
    result["vault"] = vault

    if result.get("confidence", 0.0) < CONFIDENCE_THRESHOLD:
        result["original_vault"] = vault
        result["vault"] = cfg.default_vault

    return result


def vault_todoist_config(vault_name: str) -> tuple[str | None, list[str], bool]:
    """Return (todoist_project_name, default_labels, create_tasks)."""
    routing = _load_vault_routing()
    spec = (routing.get("vaults") or {}).get(vault_name, {})
    return (
        spec.get("todoist_project"),
        spec.get("default_labels", []) or [],
        spec.get("create_tasks", True),
    )


def find_related(query: str, vault: str, cfg: Config, exclude_path: str | None = None) -> list:
    """Return same-vault notes related to query (for [[wikilinks]]). Obsidian links are intra-vault."""
    db = cfg.data_dir / "store.db"
    if not db.exists():
        return []
    try:
        hits = store.search(db, query, k=RELATED_MAX_LINKS + 2, vault=vault)
    except Exception as e:
        log.warning("find_related failed: %s", e)
        return []
    out = []
    for h in hits:
        if exclude_path and h.path == exclude_path:
            continue
        if h.distance > RELATED_MAX_DISTANCE:
            continue
        out.append(h)
        if len(out) >= RELATED_MAX_LINKS:
            break
    return out


def _wikilink(hit) -> str:
    basename = Path(hit.path).stem  # filename without .md
    title = (hit.title or basename).replace("]", "").replace("[", "")
    return f"[[{basename}|{title}]]"


def write_note(
    transcript: str,
    classification: dict,
    cfg: Config,
    tasks: list | None = None,
    related: list | None = None,
) -> Path:
    """tasks: list of todoist Task objects. related: list of store.Hit for [[wikilinks]]."""
    tasks = tasks or []
    related = related or []
    vault_name = classification["vault"]
    spec = cfg.vaults[vault_name]
    now = datetime.now(timezone.utc).astimezone()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M")

    title = classification.get("title") or transcript[:50]
    slug = _slug(title)
    inbox = spec.path / "inbox"
    inbox.mkdir(exist_ok=True)
    note_path = inbox / f"{date_str}-{time_str}-{slug}.md"

    tags = classification.get("tags", [])
    summary = classification.get("summary", "")

    fm = [
        "---",
        f'created: {now.isoformat(timespec="seconds")}',
        f"source: voice",
        f'confidence: {classification.get("confidence", 0):.2f}',
        f'tags: [{", ".join(tags)}]',
    ]
    if "original_vault" in classification:
        fm.append(f'original_vault: {classification["original_vault"]}')
    if tasks:
        fm.append(f'todoist_task_ids: [{", ".join(t.id for t in tasks)}]')
    fm.append("---")

    body = ["", f"# {title}", ""]
    if summary:
        body += [f"> {summary}", ""]
    if tasks:
        body.append("## Tasks")
        for t in tasks:
            body.append(f"- [ ] [{t.content}]({t.url})")
        body.append("")
    if related:
        body.append("## Related")
        for h in related:
            body.append(f"- {_wikilink(h)}")
        body.append("")
    body += ["## Transcript", "", transcript, ""]

    note_path.write_text("\n".join(fm + body), encoding="utf-8")
    log.info("wrote note: %s (%d tasks, %d links)", note_path, len(tasks), len(related))
    return note_path


def write_answer_note(question: str, result: dict, cfg: Config) -> Path:
    """Persist an 'ask' answer (Q + A) into the vault with an importance ranking.
    Tagged #echo/answer so a weekly cleanup job can re-sort / prune by importance.
    """
    vault_name = result.get("vault") or cfg.default_vault
    if vault_name not in cfg.vaults:
        vault_name = cfg.default_vault
    spec = cfg.vaults[vault_name]
    now = datetime.now(timezone.utc).astimezone()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M")

    title = result.get("title") or question[:50]
    slug = _slug(title)
    inbox = spec.path / "inbox"
    inbox.mkdir(exist_ok=True)
    note_path = inbox / f"{date_str}-{time_str}-{slug}.md"

    tags = list(result.get("tags", [])) + ["echo/answer"]
    fm = [
        "---",
        f'created: {now.isoformat(timespec="seconds")}',
        "source: ask",
        f'importance: {result.get("importance", 3)}',
        f'web_research: {str(result.get("used_web", False)).lower()}',
        f'tags: [{", ".join(tags)}]',
        "---",
    ]
    body = ["", f"# {title}", "", "## Frage", "", question, "", "## Antwort", "", result.get("answer", ""), ""]
    note_path.write_text("\n".join(fm + body), encoding="utf-8")
    log.info("wrote answer note: %s (importance=%s, web=%s)", note_path,
             result.get("importance"), result.get("used_web"))
    return note_path
