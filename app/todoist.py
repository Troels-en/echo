"""Thin Todoist REST API client. Sync — called via asyncio.to_thread from bot."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import httpx
from dotenv import load_dotenv
from pathlib import Path as _Path

load_dotenv(_Path(__file__).resolve().parent.parent / ".env")

log = logging.getLogger(__name__)

API = "https://api.todoist.com/api/v1"


class TodoistError(RuntimeError):
    pass


def _token() -> str:
    t = os.getenv("TODOIST_API_TOKEN", "").strip()
    if not t or t.startswith("PASTE_"):
        raise TodoistError("TODOIST_API_TOKEN not set")
    return t


def _hdr() -> dict:
    return {"Authorization": f"Bearer {_token()}", "Content-Type": "application/json"}


@dataclass
class Task:
    id: str
    content: str
    url: str
    project_id: str | None
    labels: list[str]


@lru_cache(maxsize=1)
def _projects() -> dict[str, str]:
    """name → id"""
    with httpx.Client(timeout=15.0) as c:
        r = c.get(f"{API}/projects", headers=_hdr())
        r.raise_for_status()
        return {p["name"]: p["id"] for p in r.json()["results"]}


@lru_cache(maxsize=1)
def _labels_set() -> set[str]:
    with httpx.Client(timeout=15.0) as c:
        r = c.get(f"{API}/labels", headers=_hdr())
        r.raise_for_status()
        return {l["name"] for l in r.json()["results"]}


def project_id(name: str | None) -> str | None:
    if not name:
        return None
    return _projects().get(name)


def create_task(
    content: str,
    project: str | None = None,
    labels: list[str] | None = None,
    due_string: str | None = None,
    priority: int | None = None,
    description: str | None = None,
) -> Task:
    """priority: 1 = lowest, 4 = highest (Todoist convention)."""
    body: dict = {"content": content}
    pid = project_id(project)
    if pid:
        body["project_id"] = pid
    if labels:
        valid = _labels_set()
        body["labels"] = [l for l in labels if l in valid]
    if due_string:
        body["due_string"] = due_string
    if priority and 1 <= priority <= 4:
        body["priority"] = priority
    if description:
        body["description"] = description

    log.info("todoist create: %s", body)
    with httpx.Client(timeout=15.0) as c:
        r = c.post(f"{API}/tasks", headers=_hdr(), json=body)
        if r.status_code == 400 and "due_string" in body:
            log.warning("todoist 400 with due_string=%r, retrying without due: %s",
                        body["due_string"], r.text[:200])
            body.pop("due_string", None)
            r = c.post(f"{API}/tasks", headers=_hdr(), json=body)
        r.raise_for_status()
        d = r.json()
    return Task(
        id=d["id"],
        content=d["content"],
        url=d.get("url", f"https://todoist.com/showTask?id={d['id']}"),
        project_id=d.get("project_id"),
        labels=d.get("labels", []),
    )


def close_task(task_id: str) -> None:
    with httpx.Client(timeout=15.0) as c:
        r = c.post(f"{API}/tasks/{task_id}/close", headers=_hdr())
        if r.status_code not in (200, 204):
            r.raise_for_status()
    log.info("todoist closed: %s", task_id)
