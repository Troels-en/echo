"""Document search: index the owner's documents from disk AND email into a SEPARATE
vector store (data/docs.db), search them semantically, summarize the top hit with the
LLM, and link the results into a vault.

Reuses app.store (vector index, db_path-parameterized) and app.gcal (existing Gmail
OAuth). Never touches the notes store.db.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .llm import call_text
from . import store

log = logging.getLogger(__name__)

DOC_EXTS = {".pdf", ".docx", ".txt", ".md"}
# attachments worth surfacing (superset — we only index metadata for these)
ATTACH_EXTS = DOC_EXTS | {".doc", ".xlsx", ".xls", ".pptx", ".ppt"}
MAX_TEXT_CHARS = 20000  # cap extracted text stored/embedded per doc
SKIP_DIRS = {".git", "node_modules", ".obsidian", "$RECYCLE.BIN", ".Trash", "Library"}


def db_path(cfg: Config) -> Path:
    """Separate index — keeps documents out of the notes store.db."""
    return cfg.data_dir / "docs.db"


# --------------------------------------------------------------------------- disk

def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return _extract_pdf(path)
        if suffix == ".docx":
            return _extract_docx(path)
        if suffix in (".txt", ".md"):
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        log.warning("extract failed %s: %s", path, e)
    return ""


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    total = 0
    for page in reader.pages:
        t = page.extract_text() or ""
        if t:
            parts.append(t)
            total += len(t)
        if total > MAX_TEXT_CHARS:
            break
    return "\n".join(parts)


def _extract_docx(path: Path) -> str:
    import docx

    d = docx.Document(str(path))
    return "\n".join(p.text for p in d.paragraphs if p.text)


def _iter_files(root: Path):
    for p in root.rglob("*"):
        if p.is_dir() or p.is_symlink():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in DOC_EXTS:
            yield p


def index_disk(cfg: Config, root: Path | None = None) -> dict:
    """Walk root, extract text from documents, upsert into docs.db (source='disk').
    Idempotent: the absolute path is the key (full re-scan, re-runnable safely).
    """
    root = root or cfg.doc_search_root
    db = db_path(cfg)
    store.init_schema(db)
    if not root.exists():
        log.warning("doc root missing: %s", root)
        return {"indexed": 0, "skipped": 0, "total": 0, "root": str(root)}

    indexed = skipped = 0
    for path in _iter_files(root):
        text = extract_text(path)
        if not text.strip():
            skipped += 1
            continue
        store.upsert_note(
            db,
            path=str(path.resolve()),
            vault="disk",
            title=path.name,
            summary=text.strip()[:300],
            content=text[:MAX_TEXT_CHARS],
            created=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        )
        indexed += 1
    log.info("disk index: %d indexed, %d skipped (root=%s)", indexed, skipped, root)
    return {"indexed": indexed, "skipped": skipped, "total": indexed + skipped, "root": str(root)}


# --------------------------------------------------------------------------- email

def _iter_attachment_names(payload: dict):
    fn = payload.get("filename") or ""
    if fn:
        yield fn
    for part in payload.get("parts", []) or []:
        yield from _iter_attachment_names(part)


def list_email_docs(max_results: int = 25, query: str = "has:attachment") -> list[dict]:
    """List recent Gmail messages carrying document attachments. Metadata only — no
    download (downloading is optional per the spec). Reuses gcal's OAuth client."""
    from . import gcal

    svc = gcal._gmail()
    listing = svc.users().messages().list(
        userId="me", q=query, maxResults=max_results,
    ).execute()
    out = []
    for ref in listing.get("messages", []):
        msg = svc.users().messages().get(userId="me", id=ref["id"], format="full").execute()
        payload = msg.get("payload", {})
        names = [
            n for n in _iter_attachment_names(payload)
            if Path(n).suffix.lower() in ATTACH_EXTS
        ]
        if not names:
            continue
        out.append({
            "id": ref["id"],
            "from": gcal._header(payload, "From"),
            "subject": gcal._header(payload, "Subject"),
            "date": gcal._header(payload, "Date"),
            "snippet": msg.get("snippet", ""),
            "filenames": names,
        })
    return out


def index_email_docs(cfg: Config, max_results: int = 25, query: str = "has:attachment") -> dict:
    """Index Gmail document-attachment messages into docs.db (source='email')."""
    from . import gcal

    if not gcal.is_configured():
        return {"indexed": 0, "skipped": 0, "total": 0, "blocked": "gmail-not-configured"}
    db = db_path(cfg)
    store.init_schema(db)
    mails = list_email_docs(max_results=max_results, query=query)
    indexed = 0
    for m in mails:
        files = ", ".join(m["filenames"])
        content = (
            f"Email von {m['from']}\nBetreff: {m['subject']}\nDatum: {m['date']}\n"
            f"Anhänge: {files}\n\n{m['snippet']}"
        )
        store.upsert_note(
            db,
            path=f"gmail:{m['id']}",
            vault="email",
            title=m["subject"] or files,
            summary=f"Anhänge: {files} — von {m['from']}",
            content=content,
            created=m["date"],
        )
        indexed += 1
    log.info("email doc index: %d messages", indexed)
    return {"indexed": indexed, "skipped": 0, "total": indexed}


# --------------------------------------------------------------------------- search

SUMMARY_PROMPT = """Du fasst ein offizielles Dokument des Nutzers zusammen.

Fasse das folgende Dokument in 2-4 deutschen Sätzen zusammen: worum es geht, die wichtigsten
Fakten (Beträge, Fristen, Absender) und was der Nutzer ggf. tun muss.

DOKUMENT ({title}):
\"\"\"{content}\"\"\"

Antworte NUR mit der Zusammenfassung, ohne Vorspann."""


def summarize_doc(title: str, content: str, cfg: Config) -> str:
    prompt = SUMMARY_PROMPT.format(title=title, content=content[:6000])
    try:
        return call_text(prompt, primary=cfg.llm_primary, fallback=cfg.llm_fallback).strip()
    except Exception as e:
        log.warning("summarize failed: %s", e)
        return content.strip()[:300]


def find_docs(query: str, cfg: Config, k: int = 5) -> dict:
    """Semantic search over indexed documents; LLM-summarize the top hit.
    Returns {hits, summary, indexed}."""
    db = db_path(cfg)
    if not db.exists():
        return {"hits": [], "summary": "", "indexed": False}
    hits = store.search(db, query, k=k)
    summary = summarize_doc(hits[0].title, hits[0].content, cfg) if hits else ""
    return {"hits": hits, "summary": summary, "indexed": True}


# --------------------------------------------------------------------------- vault note

def _slug(text: str, max_len: int = 50) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len] or "doc"


def _hit_line(h: store.Hit) -> str:
    if h.vault == "email":
        mid = h.path.split(":", 1)[1]
        link = f"https://mail.google.com/mail/u/0/#all/{mid}"
        return f"- 📧 [{h.title}]({link}) — {h.summary}"
    uri = Path(h.path).as_uri()  # encodes spaces; Obsidian opens file:// links
    return f"- 📄 [{h.title}]({uri}) — `{h.path}`"


def write_doc_note(query: str, result: dict, cfg: Config, vault_name: str | None = None) -> Path | None:
    """Write a linked note pointing to the matching documents (disk paths / email links)."""
    hits = result.get("hits") or []
    if not hits:
        return None
    vault_name = vault_name or ("Misc_Vault" if "Misc_Vault" in cfg.vaults else cfg.default_vault)
    spec = cfg.vaults[vault_name]
    now = datetime.now(timezone.utc).astimezone()
    inbox = spec.path / "inbox"
    inbox.mkdir(exist_ok=True)
    note_path = inbox / f"{now:%Y-%m-%d-%H%M}-doc-{_slug(query)}.md"

    fm = [
        "---",
        f'created: {now.isoformat(timespec="seconds")}',
        "source: docsearch",
        f"query: {query!r}",
        "tags: [echo/doc]",
        "---",
    ]
    body = [
        "",
        f"# Dokumentensuche: {query}",
        "",
        "## Zusammenfassung (Top-Treffer)",
        "",
        result.get("summary", ""),
        "",
        "## Treffer",
        "",
    ]
    body += [_hit_line(h) for h in hits]
    body.append("")
    note_path.write_text("\n".join(fm + body), encoding="utf-8")
    log.info("wrote doc note: %s", note_path)
    return note_path
