"""Load .env + vaults.yml into a typed config object."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")


@dataclass
class VaultSpec:
    name: str
    path: Path
    keywords: list[str] = field(default_factory=list)
    default: bool = False


@dataclass
class Config:
    telegram_token: str
    vault_root: Path
    allowed_user_ids: set[int]
    llm_mode: str
    llm_primary: str
    llm_fallback: str
    llm_deep: str
    whisper_backend: str
    whisper_model: str
    whisper_model_path: Path
    vaults: dict[str, VaultSpec]
    default_vault: str
    data_dir: Path
    ask_model: str
    ask_web_timeout: int
    doc_search_root: Path
    elevenlabs_api_key: str
    elevenlabs_voice_id: str
    elevenlabs_model: str
    tts_max_chars: int

    @classmethod
    def load(cls) -> "Config":
        vault_root = Path(os.path.expanduser(os.getenv("VAULT_ROOT", "~/"))).resolve()

        vaults_path = REPO_ROOT / "config" / "vaults.yml"
        if not vaults_path.exists():
            vaults_path = REPO_ROOT / "config" / "vaults.example.yml"
        with vaults_path.open() as f:
            raw = yaml.safe_load(f)

        vaults: dict[str, VaultSpec] = {}
        default_vault = "Misc_Vault"
        for name, spec in (raw.get("vaults") or {}).items():
            path = vault_root / name
            if not path.exists():
                continue
            vaults[name] = VaultSpec(
                name=name,
                path=path,
                keywords=spec.get("keywords", []) or [],
                default=bool(spec.get("default", False)),
            )
            if spec.get("default"):
                default_vault = name

        allowed_raw = os.getenv("ALLOWED_USER_IDS", "").strip()
        allowed = {int(x) for x in allowed_raw.split(",") if x.strip().isdigit()}

        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token or token.startswith("PASTE_"):
            raise RuntimeError("TELEGRAM_BOT_TOKEN not set in .env")

        data_dir = REPO_ROOT / "data"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "audio").mkdir(exist_ok=True)
        (data_dir / "models").mkdir(exist_ok=True)

        model_name = os.getenv("WHISPER_MODEL", "large-v3-turbo")
        model_path = data_dir / "models" / f"ggml-{model_name}.bin"

        return cls(
            telegram_token=token,
            vault_root=vault_root,
            allowed_user_ids=allowed,
            llm_mode=os.getenv("LLM_MODE", "cli"),
            llm_primary=os.getenv("LLM_PRIMARY", "codex"),
            llm_fallback=os.getenv("LLM_FALLBACK", "claude"),
            llm_deep=os.getenv("LLM_DEEP", "claude"),
            whisper_backend=os.getenv("WHISPER_BACKEND", "whisper-cpp"),
            whisper_model=model_name,
            whisper_model_path=model_path,
            vaults=vaults,
            default_vault=default_vault,
            data_dir=data_dir,
            ask_model=os.getenv("ASK_MODEL", "sonnet"),
            ask_web_timeout=int(os.getenv("ASK_WEB_TIMEOUT", "300")),
            doc_search_root=Path(os.path.expanduser(os.getenv("DOC_SEARCH_ROOT", "~/Documents"))).resolve(),
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", "").strip(),
            elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID", "").strip() or "JBFqnCBsd6RMkjVDRZzb",
            elevenlabs_model=os.getenv("ELEVENLABS_MODEL", "").strip() or "eleven_multilingual_v2",
            tts_max_chars=int(os.getenv("TTS_MAX_CHARS", "").strip() or "600"),
        )
