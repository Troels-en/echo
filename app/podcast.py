"""Turn a briefing into an audio podcast: LLM writes a German script, TTS synthesizes it,
ffmpeg stitches the segments into one MP3.

Backends, chosen by PODCAST_BACKEND (default "auto"):
- "gemini"     — Gemini multi-speaker TTS (`GOOGLE_API_KEY`). Two distinct voices, one API
                 call. Official, stable. PRIMARY when a key is present.
- "notebooklm" — real NotebookLM Audio Overview via notebooklm-py (browser-cookie auth, run
                 `notebooklm login --browser-cookies chrome` once). Unofficial, can break.
- "elevenlabs" — per-segment ElevenLabs synthesis (`ELEVENLABS_API_KEY`).
- "say"        — macOS `say`, always-available local fallback (de_DE voices).
- "auto"       — gemini → elevenlabs → say, by what's configured; falls back on runtime error.

Self-contained on purpose: reads its own env vars instead of touching the shared Config.
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

# Gemini multi-speaker TTS (Path B). Two prebuilt voices; speaker labels in the transcript
# must match the names registered in the multi-speaker config below.
GEMINI_TTS_MODEL = _env("GEMINI_TTS_MODEL", "gemini-2.5-pro-tts")
GEMINI_VOICE_A = _env("GEMINI_VOICE_A", "Kore")   # host A (female)
GEMINI_VOICE_B = _env("GEMINI_VOICE_B", "Puck")   # host B (male)
_SPEAKER_NAME = {"A": "Anna", "B": "Reed"}

PODCAST_BACKEND = _env("PODCAST_BACKEND", "auto").lower()


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


def _gemini_available() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY", "").strip())


def _synth_gemini_multi(segments: list[Segment], out_mp3: Path) -> None:
    """Render the whole two-host dialogue in ONE Gemini multi-speaker TTS call → MP3.
    Gemini returns raw PCM (24kHz, 16-bit, mono); ffmpeg packs it into MP3.
    """
    from google import genai
    from google.genai import types

    transcript = "\n".join(f"{_SPEAKER_NAME[s.speaker]}: {s.text}" for s in segments)
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY", "").strip())
    resp = client.models.generate_content(
        model=GEMINI_TTS_MODEL,
        contents=transcript,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker=_SPEAKER_NAME["A"],
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=GEMINI_VOICE_A))),
                        types.SpeakerVoiceConfig(
                            speaker=_SPEAKER_NAME["B"],
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=GEMINI_VOICE_B))),
                    ]
                )
            ),
        ),
    )
    try:
        data = resp.candidates[0].content.parts[0].inline_data.data
    except (AttributeError, IndexError, TypeError) as e:
        raise PodcastError(f"Gemini TTS returned no audio: {e}")
    if isinstance(data, str):  # SDK may hand back base64 text
        import base64
        data = base64.b64decode(data)

    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".pcm", delete=False) as f:
        pcm_path = Path(f.name)
        f.write(data)
    try:
        res = subprocess.run(
            ["ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1",
             "-i", str(pcm_path), "-c:a", "libmp3lame", "-b:a", "128k", str(out_mp3)],
            capture_output=True, text=True, timeout=180,
        )
        if res.returncode != 0:
            raise PodcastError(f"ffmpeg PCM→MP3 failed: {res.stderr[-200:]}")
    finally:
        pcm_path.unlink(missing_ok=True)


def _build_notebooklm(briefing_text: str, out_mp3: Path) -> None:
    """Real NotebookLM Audio Overview via notebooklm-py. Needs cookies from
    `notebooklm login --browser-cookies chrome`. Unofficial — may break.
    """
    import asyncio
    from datetime import datetime

    async def _run() -> None:
        from notebooklm import NotebookLMClient
        async with await NotebookLMClient.from_storage() as client:
            title = f"Echo Daily {datetime.now():%Y-%m-%d %H:%M}"
            nb = await client.notebooks.create(title)
            await client.sources.add_text(nb.id, briefing_text, wait=True)
            status = await client.artifacts.generate_audio(
                nb.id,
                instructions="Kurzer, lockerer deutscher Tagesüberblick zwischen zwei Moderatoren. Sprich den Hörer mit Du an.",
            )
            await client.artifacts.wait_for_completion(nb.id, status.task_id)
            out_mp3.parent.mkdir(parents=True, exist_ok=True)
            await client.artifacts.download_audio(nb.id, str(out_mp3))

    try:
        asyncio.run(_run())
    except FileNotFoundError as e:
        raise PodcastError(
            "NotebookLM nicht eingeloggt. Einmalig ausführen: "
            "`notebooklm login --browser-cookies chrome`."
        ) from e
    except Exception as e:
        raise PodcastError(f"NotebookLM-Backend fehlgeschlagen: {e}") from e
    if not out_mp3.exists():
        raise PodcastError("NotebookLM lieferte keine Audiodatei.")


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


def _script_backend_chain() -> list[str]:
    """Ordered script-based backends to try, honoring PODCAST_BACKEND."""
    if PODCAST_BACKEND in ("gemini", "elevenlabs", "say"):
        return [PODCAST_BACKEND]
    chain: list[str] = []
    if _gemini_available():
        chain.append("gemini")
    if _eleven_available():
        chain.append("elevenlabs")
    if platform.system() == "Darwin":
        chain.append("say")
    if not chain:
        raise PodcastError(
            "Kein TTS-Backend: setze GOOGLE_API_KEY (Gemini) oder ELEVENLABS_API_KEY, "
            "oder laufe auf macOS (say)."
        )
    return chain


def _synth_one_backend(segments: list[Segment], out_mp3: Path, backend: str) -> str:
    """Synthesize the whole script with a single named backend. Returns backend label."""
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    if backend == "gemini":
        _synth_gemini_multi(segments, out_mp3)
        return "gemini-tts"
    if backend == "elevenlabs":
        synth = _synth_eleven
    elif backend == "say":
        synth = _synth_say
    else:
        raise PodcastError(f"Unknown TTS backend: {backend!r}")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        wavs: list[Path] = []
        for i, seg in enumerate(segments):
            wav = tmpdir / f"seg_{i:03d}.wav"
            synth(seg.text, seg.speaker, wav)
            wavs.append(wav)
        _stitch(wavs, out_mp3)
    return "macos-say" if backend == "say" else backend


def synthesize(segments: list[Segment], out_mp3: Path) -> str:
    """Synthesize the script, trying backends in order, falling back on error."""
    chain = _script_backend_chain()
    last_err: Exception | None = None
    for backend in chain:
        try:
            return _synth_one_backend(segments, out_mp3, backend)
        except Exception as e:
            log.warning("podcast backend %s failed: %s", backend, e)
            last_err = e
    raise PodcastError(f"All podcast backends failed: {last_err}")


def build_podcast(cfg: Config, briefing_text: str | None = None) -> PodcastResult:
    """End-to-end: briefing text -> audio podcast MP3.

    PODCAST_BACKEND=notebooklm uses the real NotebookLM Audio Overview (no script step);
    otherwise an LLM writes a two-host German script and a TTS backend renders it.
    On NotebookLM failure, falls back to the script+TTS path so /podcast always returns audio.
    """
    if briefing_text is None:
        briefing_text = briefing_mod.build_briefing(cfg)
    out_mp3 = cfg.data_dir / "podcast" / "briefing.mp3"

    if PODCAST_BACKEND == "notebooklm":
        try:
            _build_notebooklm(briefing_text, out_mp3)
            dur = _duration(out_mp3)
            log.info("podcast built: %s (%.1fs, notebooklm)", out_mp3, dur)
            return PodcastResult(path=out_mp3, duration=dur, backend="notebooklm",
                                 num_segments=0, script=[])
        except PodcastError as e:
            log.warning("NotebookLM failed, falling back to script+TTS: %s", e)

    segments = generate_script(briefing_text, cfg)
    backend = synthesize(segments, out_mp3)
    dur = _duration(out_mp3)
    log.info("podcast built: %s (%.1fs, %s, %d segments)", out_mp3, dur, backend, len(segments))
    return PodcastResult(
        path=out_mp3, duration=dur, backend=backend,
        num_segments=len(segments), script=segments,
    )
