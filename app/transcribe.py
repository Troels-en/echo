"""Whisper transcription. Prefers persistent whisper-server (model in RAM);
falls back to whisper-cli subprocess if the server is not reachable.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


class TranscribeError(RuntimeError):
    pass


def _server_url() -> str:
    host = os.getenv("WHISPER_HOST", "127.0.0.1")
    port = os.getenv("WHISPER_PORT", "8910")
    return f"http://{host}:{port}/inference"


def _via_server(audio_path: Path, language: str) -> str | None:
    """POST audio to whisper-server. Returns text, or None if server unreachable."""
    url = _server_url()
    try:
        with audio_path.open("rb") as f:
            files = {"file": (audio_path.name, f, "audio/wav")}
            data = {"response_format": "json", "language": language}
            r = httpx.post(url, files=files, data=data, timeout=180.0)
        r.raise_for_status()
        try:
            return (r.json().get("text") or "").strip()
        except Exception:
            return r.text.strip()
    except (httpx.ConnectError, httpx.ConnectTimeout):
        log.warning("whisper-server not reachable at %s, falling back to CLI", url)
        return None
    except httpx.HTTPStatusError as e:
        raise TranscribeError(f"whisper-server error {e.response.status_code}: {e.response.text[:300]}")


def _via_cli(audio_path: Path, model_path: Path, language: str) -> str:
    if not model_path.exists():
        raise TranscribeError(f"whisper model not found at {model_path}")
    cmd = [
        "whisper-cli", "-m", str(model_path), "-f", str(audio_path),
        "-l", language, "--no-prints", "-otxt",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise TranscribeError(f"whisper-cli failed (exit {result.returncode}): {result.stderr[-500:]}")
    txt_path = audio_path.with_suffix(audio_path.suffix + ".txt")
    if not txt_path.exists():
        raise TranscribeError(f"transcript file missing: {txt_path}")
    text = txt_path.read_text(encoding="utf-8").strip()
    txt_path.unlink(missing_ok=True)
    return text


def transcribe(audio_path: Path, model_path: Path, language: str = "auto") -> str:
    if not audio_path.exists():
        raise TranscribeError(f"audio file not found: {audio_path}")
    text = _via_server(audio_path, language)
    if text is not None:
        return text
    return _via_cli(audio_path, model_path, language)
