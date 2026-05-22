"""Turn a briefing into an audio podcast: LLM writes a German script, TTS synthesizes it,
ffmpeg stitches the segments into one MP3.

Backends (auto-selected):
- ElevenLabs (`ELEVENLABS_API_KEY`) — natural multi-voice, preferred when the key is present.
- macOS `say` — always available locally, used as a reliable fallback (de_DE voices).

Self-contained on purpose: reads its own env vars instead of touching the shared Config,
so it merges cleanly with the other parallel sessions.
"""
from __future__ import annotations

import logging
import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from . import briefing as briefing_mod
from . import llm

log = logging.getLogger(__name__)

def _env(name: str, default: str) -> str:
    """os.getenv but treat an empty/whitespace value as unset (falls back to default)."""
    val = (os.getenv(name) or "").strip()
    return val or default


# Two German hosts. ElevenLabs uses voice IDs; macOS `say` uses voice names.
# multilingual_v2 speaks German with any voice; defaults are stock ElevenLabs voices
# present on every account (Sarah = female host A, Brian = male host B).
ELEVEN_MODEL = _env("ELEVENLABS_MODEL", "eleven_multilingual_v2")
ELEVEN_VOICE_A = _env("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
ELEVEN_VOICE_B = _env("ELEVENLABS_VOICE_ID_B", "nPczCjzI2devNBz1zQrb")
SAY_VOICE_A = _env("PODCAST_SAY_VOICE_A", "Anna")
SAY_VOICE_B = _env("PODCAST_SAY_VOICE_B", "Reed")


class PodcastError(RuntimeError):
    pass


@dataclass
class Segment:
    speaker: str  # "A" or "B"
    text: str


@dataclass
class PodcastResult:
    path: Path
    duration: float
    backend: str
    num_segments: int
    script: list[Segment]

    def script_text(self) -> str:
        return "\n".join(f"[{s.speaker}] {s.text}" for s in self.script)


_SCRIPT_PROMPT = """Du bist Headautor eines kurzen deutschen Tages-Podcasts namens "Echo Daily".
Verwandle das folgende persönliche Briefing in ein natürliches, lockeres Gespräch zwischen \
zwei Moderatoren: Host A (Anna) und Host B (Reed). Sprich den Hörer direkt an (Du-Form).

Regeln:
- Rein Deutsch, gesprochene Sprache, keine Aufzählungszeichen, keine Emojis, keine Markdown-Zeichen.
- Kurzer Begrüßungs-Einstieg, dann gehen die beiden die Punkte durch, am Ende ein knapper Abschluss.
- Halte es kompakt: etwa 8 bis 14 Wortbeiträge insgesamt, jeder 1-3 Sätze.
- Wechsle die Sprecher natürlich ab. Erfinde keine Fakten, die nicht im Briefing stehen.

Antworte AUSSCHLIESSLICH mit JSON in genau diesem Format:
{"segments": [{"speaker": "A", "text": "..."}, {"speaker": "B", "text": "..."}]}

BRIEFING:
---
%s
---"""


def generate_script(briefing_text: str, cfg: Config) -> list[Segment]:
    """LLM turns briefing text into an alternating two-host German dialogue."""
    prompt = _SCRIPT_PROMPT % briefing_text.strip()
    data = llm.call_json(prompt, primary=cfg.llm_primary, fallback=cfg.llm_fallback)
    raw = data.get("segments") or []
    segments: list[Segment] = []
    for s in raw:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        speaker = "B" if str(s.get("speaker", "A")).strip().upper().startswith("B") else "A"
        segments.append(Segment(speaker=speaker, text=text))
    if not segments:
        raise PodcastError("LLM returned no usable script segments")
    return segments


def _eleven_available() -> bool:
    return bool(os.getenv("ELEVENLABS_API_KEY", "").strip())


def _synth_eleven(text: str, speaker: str, out_wav: Path) -> None:
    """Synthesize one segment via ElevenLabs, decode to a normalized wav."""
    import httpx

    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    voice_id = ELEVEN_VOICE_B if speaker == "B" else ELEVEN_VOICE_A
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": text,
        "model_id": ELEVEN_MODEL,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    headers = {"xi-api-key": api_key, "accept": "audio/mpeg", "content-type": "application/json"}
    with httpx.Client(timeout=120.0) as c:
        r = c.post(url, json=payload, headers=headers)
        if r.status_code != 200:
            raise PodcastError(f"ElevenLabs {r.status_code}: {r.text[:200]}")
        mp3 = out_wav.with_suffix(".seg.mp3")
        mp3.write_bytes(r.content)
    _to_norm_wav(mp3, out_wav)
    mp3.unlink(missing_ok=True)


def _synth_say(text: str, speaker: str, out_wav: Path) -> None:
    """Synthesize one segment via macOS `say`, decode to a normalized wav."""
    voice = SAY_VOICE_B if speaker == "B" else SAY_VOICE_A
    aiff = out_wav.with_suffix(".seg.aiff")
    res = subprocess.run(
        ["say", "-v", voice, "-o", str(aiff), text],
        capture_output=True, text=True, timeout=120,
    )
    if res.returncode != 0:
        raise PodcastError(f"say failed (voice={voice}): {res.stderr[-200:]}")
    _to_norm_wav(aiff, out_wav)
    aiff.unlink(missing_ok=True)


def _to_norm_wav(src: Path, dst: Path) -> None:
    """Decode any audio to a uniform PCM wav so segments concat cleanly."""
    res = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", "44100", "-ac", "1", str(dst)],
        capture_output=True, text=True, timeout=120,
    )
    if res.returncode != 0:
        raise PodcastError(f"ffmpeg decode failed: {res.stderr[-200:]}")


def _stitch(wavs: list[Path], out_mp3: Path) -> None:
    """Concat uniform wavs into a single MP3 via the ffmpeg concat demuxer."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        list_path = Path(f.name)
        for w in wavs:
            f.write(f"file '{w.resolve()}'\n")
    try:
        res = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
             "-c:a", "libmp3lame", "-b:a", "128k", str(out_mp3)],
            capture_output=True, text=True, timeout=300,
        )
        if res.returncode != 0:
            raise PodcastError(f"ffmpeg concat failed: {res.stderr[-200:]}")
    finally:
        list_path.unlink(missing_ok=True)


def _duration(mp3: Path) -> float:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(mp3)],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return float(res.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def synthesize(segments: list[Segment], out_mp3: Path) -> str:
    """Synthesize all segments and stitch to one MP3. Returns the backend name used."""
    if _eleven_available():
        backend, synth = "elevenlabs", _synth_eleven
    elif platform.system() == "Darwin":
        backend, synth = "macos-say", _synth_say
    else:
        raise PodcastError(
            "No TTS backend: set ELEVENLABS_API_KEY or run on macOS (say)."
        )

    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        wavs: list[Path] = []
        for i, seg in enumerate(segments):
            wav = tmpdir / f"seg_{i:03d}.wav"
            synth(seg.text, seg.speaker, wav)
            wavs.append(wav)
        _stitch(wavs, out_mp3)
    return backend


def build_podcast(cfg: Config, briefing_text: str | None = None) -> PodcastResult:
    """End-to-end: briefing text -> German script -> synthesized MP3."""
    if briefing_text is None:
        briefing_text = briefing_mod.build_briefing(cfg)

    segments = generate_script(briefing_text, cfg)
    out_mp3 = cfg.data_dir / "podcast" / "briefing.mp3"
    backend = synthesize(segments, out_mp3)
    dur = _duration(out_mp3)
    log.info("podcast built: %s (%.1fs, %s, %d segments)", out_mp3, dur, backend, len(segments))
    return PodcastResult(
        path=out_mp3, duration=dur, backend=backend,
        num_segments=len(segments), script=segments,
    )
