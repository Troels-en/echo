"""Text→speech via ElevenLabs, transcoded to Telegram-native OGG/Opus voice.

ElevenLabs returns mp3; ffmpeg transcodes to mono Opus-in-Ogg so Telegram shows it
as a real voice memo (waveform + playback speed), not a generic audio attachment.
"""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from uuid import uuid4

import httpx

from .config import Config

log = logging.getLogger(__name__)

API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


class TTSError(Exception):
    pass


def available(cfg: Config) -> bool:
    """True when an ElevenLabs key is configured."""
    return bool(cfg.elevenlabs_api_key)


def shorten_for_speech(text: str, max_chars: int) -> str:
    """Strip markdown noise and cap length for a spoken summary.

    Removes the bot's markdown footer/links, code fences and emphasis markers, then
    truncates on a sentence boundary near `max_chars` (falls back to a word boundary).
    """
    t = text.strip()
    # drop the trailing "_…footer…_" the bot appends (web tag · stars · path)
    t = re.sub(r"\n+_[^\n]*_\s*$", "", t)
    t = t.replace("```", " ")
    t = re.sub(r"[`*#>_]", "", t)          # markdown emphasis / headings / code
    t = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", t)  # [label](url) → label
    t = re.sub(r"\n{2,}", ". ", t)         # paragraph breaks → sentence breaks
    t = re.sub(r"\s+", " ", t).strip()

    if len(t) <= max_chars:
        return t

    head = t[:max_chars]
    cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if cut >= max_chars * 0.5:
        return head[: cut + 1].strip()
    sp = head.rfind(" ")
    return (head[:sp] if sp > 0 else head).strip() + "…"


def _transcode_to_opus(src: Path) -> Path:
    """mp3 → mono Ogg/Opus (Telegram voice format)."""
    dst = src.with_suffix(".ogg")
    res = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-c:a", "libopus", "-b:a", "32k", "-ar", "48000", "-ac", "1",
         str(dst)],
        capture_output=True, text=True, timeout=120,
    )
    if res.returncode != 0 or not dst.exists():
        raise TTSError(f"ffmpeg transcode failed: {res.stderr[-300:]}")
    return dst


def synthesize(text: str, cfg: Config, max_chars: int | None = None) -> Path:
    """Synthesize `text` to a Telegram-ready OGG/Opus file. Returns the file path.

    Raises TTSError if no key is set or the API/transcode fails. Caller should guard
    with `available(cfg)` first when a graceful skip is desired.
    """
    if not available(cfg):
        raise TTSError("ELEVENLABS_API_KEY not set")

    spoken = shorten_for_speech(text, max_chars or cfg.tts_max_chars)
    if not spoken:
        raise TTSError("nothing to synthesize")

    out_dir = cfg.data_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp3_path = out_dir / f"tts-{uuid4().hex}.mp3"

    url = API_URL.format(voice_id=cfg.elevenlabs_voice_id)
    headers = {
        "xi-api-key": cfg.elevenlabs_api_key,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }
    payload = {
        "text": spoken,
        "model_id": cfg.elevenlabs_model,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    try:
        with httpx.Client(timeout=60.0) as c:
            r = c.post(url, headers=headers, json=payload)
            r.raise_for_status()
            mp3_path.write_bytes(r.content)
    except httpx.HTTPError as e:
        raise TTSError(f"ElevenLabs request failed: {e}") from e

    try:
        return _transcode_to_opus(mp3_path)
    finally:
        mp3_path.unlink(missing_ok=True)
