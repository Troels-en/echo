"""Create Todoist projects + labels. Idempotent."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

TOKEN = os.environ["TODOIST_API_TOKEN"]
API = "https://api.todoist.com/api/v1"

PROJECTS = ["Startups", "Engineering", "Career", "Fitness"]
LABELS = ["Uni", "Privat", "Karriere", "Finanzen", "Career-Buddy"]


def _hdr() -> dict:
    return {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


def list_projects(client: httpx.Client) -> list[dict]:
    r = client.get(f"{API}/projects", headers=_hdr())
    r.raise_for_status()
    return r.json()["results"]


def list_labels(client: httpx.Client) -> list[dict]:
    r = client.get(f"{API}/labels", headers=_hdr())
    r.raise_for_status()
    return r.json()["results"]


def create_project(client: httpx.Client, name: str) -> dict:
    r = client.post(f"{API}/projects", headers=_hdr(), json={"name": name})
    r.raise_for_status()
    return r.json()


def create_label(client: httpx.Client, name: str) -> dict:
    r = client.post(f"{API}/labels", headers=_hdr(), json={"name": name})
    r.raise_for_status()
    return r.json()


def main() -> None:
    with httpx.Client(timeout=15.0) as client:
        existing_projects = {p["name"]: p["id"] for p in list_projects(client)}
        existing_labels = {l["name"]: l["id"] for l in list_labels(client)}

        print(f"Found {len(existing_projects)} projects, {len(existing_labels)} labels.\n")

        print("PROJECTS:")
        for name in PROJECTS:
            if name in existing_projects:
                print(f"  skip {name:15s} id={existing_projects[name]}")
                continue
            try:
                p = create_project(client, name)
                print(f"  +new {name:15s} id={p['id']}")
            except httpx.HTTPStatusError as e:
                print(f"  FAIL {name:15s} {e.response.status_code} {e.response.text[:120]}")

        print("\nLABELS:")
        for name in LABELS:
            if name in existing_labels:
                print(f"  skip {name:15s} id={existing_labels[name]}")
                continue
            try:
                l = create_label(client, name)
                print(f"  +new {name:15s} id={l['id']}")
            except httpx.HTTPStatusError as e:
                print(f"  FAIL {name:15s} {e.response.status_code} {e.response.text[:120]}")

        print("\nFinal state:")
        for p in list_projects(client):
            print(f"  P {p['name']:20s} {p['id']}")
        for l in list_labels(client):
            print(f"  L {l['name']:20s} {l['id']}")


if __name__ == "__main__":
    main()
