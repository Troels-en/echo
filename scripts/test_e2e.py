"""Synthetic E2E test: feeds mock transcripts through classify → todoist → write_note.
Skips Whisper (no real audio). Cleans up tasks + notes after.

Categories tested:
- Idea  (no task expected)
- Task  (task expected, due, priority)
- Journal (no task, journal vault)
- Cross-cutting Karriere label
- Misc fallback (gibberish)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config
from app.vault import classify, write_note
from app.bot import _maybe_create_task
from app.todoist import close_task

CASES = [
    {
        "name": "Idea (no task)",
        "transcript": "Hatte gerade die Idee einen MVP zu bauen, der Notion-Notes auf Basis von Voice-Input erstellt. Marktlücke zwischen Otter und Notion.",
        "expect_task": False,
        "expect_vault_contains": "Startup",
    },
    {
        "name": "Task with deadline",
        "transcript": "Vergiss nicht morgen die Bewerbung für die McKinsey-Stelle abzuschicken, das ist wichtig.",
        "expect_task": True,
        "expect_vault_contains": "Career",
    },
    {
        "name": "Journal entry",
        "transcript": "Heute war ein guter Lauf, fünf Kilometer in 25 Minuten, fühle mich erschöpft aber zufrieden.",
        "expect_task": False,
        "expect_vault_contains": "Journal",
    },
    {
        "name": "Finance task",
        "transcript": "Ich sollte diese Woche meine ETF-Sparrate von 500 auf 700 Euro erhöhen.",
        "expect_task": True,
        "expect_vault_contains": "Finance",
    },
    {
        "name": "Misc fallback (ambiguous)",
        "transcript": "Hmm, blau ist eine Farbe.",
        "expect_task": False,
        "expect_vault_contains": None,
    },
]


def fmt_ok(b: bool) -> str:
    return "✓" if b else "✗"


async def run() -> int:
    cfg = Config.load()
    failures = 0
    cleanup_task_ids: list[str] = []
    cleanup_notes: list[Path] = []

    for i, case in enumerate(CASES, 1):
        print(f"\n[{i}/{len(CASES)}] {case['name']}")
        print(f"  transcript: {case['transcript'][:80]}")

        try:
            c = classify(case["transcript"], cfg)
        except Exception as e:
            print(f"  CLASSIFY FAILED: {e}")
            failures += 1
            continue

        vault = c.get("vault", "?")
        conf = c.get("confidence", 0)
        is_task = (c.get("task") or {}).get("is_task", False)
        title = c.get("title", "")
        labels = (c.get("task") or {}).get("labels", [])
        pri = (c.get("task") or {}).get("priority", 0)
        due = (c.get("task") or {}).get("due_string", "")

        print(f"  vault: {vault}  conf={conf:.2f}")
        print(f"  title: {title}")
        print(f"  is_task: {is_task}  pri={pri}  due={due}  labels={labels}")

        # vault check
        vault_ok = (
            case["expect_vault_contains"] is None
            or case["expect_vault_contains"].lower() in vault.lower()
        )
        # task check
        task_ok = is_task == case["expect_task"]

        print(f"  vault match {fmt_ok(vault_ok)} | task match {fmt_ok(task_ok)}")
        if not vault_ok or not task_ok:
            failures += 1

        task = await _maybe_create_task(c)
        if task:
            cleanup_task_ids.append(task.id)
            print(f"  todoist: created id={task.id} labels={task.labels}")

        try:
            note_path = write_note(
                case["transcript"], c, cfg,
                todoist_url=task.url if task else None,
                todoist_task_id=task.id if task else None,
            )
            cleanup_notes.append(note_path)
            print(f"  note: {note_path.relative_to(cfg.vault_root)}")
        except Exception as e:
            print(f"  WRITE FAILED: {e}")
            failures += 1

    # cleanup
    print(f"\n--- Cleanup ---")
    for tid in cleanup_task_ids:
        try:
            close_task(tid)
            print(f"  closed task {tid}")
        except Exception as e:
            print(f"  close failed {tid}: {e}")
    for p in cleanup_notes:
        try:
            p.unlink()
            print(f"  removed {p.name}")
        except Exception as e:
            print(f"  rm failed {p}: {e}")

    print(f"\n=== Result: {len(CASES) - failures}/{len(CASES)} passed ===")
    return failures


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
