"""Walk all *_Vault folders, parse markdown, index into store. Idempotent."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config
from app import store

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
TITLE_RE = re.compile(r"^# (.+)$", re.MULTILINE)
SUMMARY_RE = re.compile(r"^> (.+)$", re.MULTILINE)


def parse_note(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    fm_match = FRONTMATTER_RE.match(text)
    body = text[fm_match.end():] if fm_match else text
    fm = fm_match.group(1) if fm_match else ""

    title_match = TITLE_RE.search(body)
    title = title_match.group(1).strip() if title_match else path.stem
    summary_match = SUMMARY_RE.search(body)
    summary = summary_match.group(1).strip() if summary_match else ""

    created = ""
    for line in fm.splitlines():
        if line.startswith("created:"):
            created = line.split(":", 1)[1].strip()
            break

    return {
        "title": title,
        "summary": summary,
        "content": body.strip(),
        "created": created,
    }


def main(reindex: bool = False) -> None:
    cfg = Config.load()
    db = cfg.data_dir / "store.db"
    store.init_schema(db)

    added = 0
    skipped = 0
    failed = 0
    for vault_name, spec in cfg.vaults.items():
        if not spec.path.exists():
            continue
        for md_path in spec.path.rglob("*.md"):
            if ".obsidian" in md_path.parts or "_pipeline" in md_path.parts:
                continue
            abs_path = str(md_path.resolve())
            if not reindex and store.path_exists(db, abs_path):
                skipped += 1
                continue
            try:
                parsed = parse_note(md_path)
                store.upsert_note(
                    db,
                    path=abs_path,
                    vault=vault_name,
                    title=parsed["title"],
                    summary=parsed["summary"],
                    content=parsed["content"],
                    created=parsed["created"],
                )
                added += 1
                if added % 10 == 0:
                    print(f"  indexed {added}...")
            except Exception as e:
                print(f"  FAIL {md_path}: {e}")
                failed += 1

    total = store.count(db)
    print(f"\nDone. added={added} skipped={skipped} failed={failed}  total_in_store={total}")


if __name__ == "__main__":
    reindex = "--reindex" in sys.argv
    main(reindex=reindex)
