"""Personal style agents: produce humanized German drafts in the user's own voice.

Each agent assembles a prompt from the Self_Vault (writing-style rules + profile facts)
plus a task brief, then calls the existing LLM layer. Start with one drafting agent
(Anschreiben / Bewerbung); the registry makes adding more topic agents trivial.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .llm import call_text

log = logging.getLogger(__name__)

SELF_VAULT = "Self_Vault"
STYLE_FILE = "writing-style.md"
# Profile files mined into prompt context, in priority order.
PROFILE_FILES = ("about-me.md", "cv-facts.md", "preferences.md")


def self_vault_dir(cfg: Config) -> Path:
    """Self_Vault lives under VAULT_ROOT, registered in vaults.yml like any vault."""
    return cfg.vault_root / SELF_VAULT


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        log.warning("self-vault file missing: %s", path)
        return ""


def load_self_context(cfg: Config) -> tuple[str, str]:
    """Return (style_rules, profile_context) from the Self_Vault markdown."""
    base = self_vault_dir(cfg)
    style = _read(base / STYLE_FILE)
    parts = []
    for name in PROFILE_FILES:
        text = _read(base / name)
        if text:
            parts.append(text)
    return style, "\n\n---\n\n".join(parts)


@dataclass(frozen=True)
class Agent:
    name: str
    description: str
    instructions: str  # task-specific framing appended to the shared prompt


ANSCHREIBEN = Agent(
    name="Anschreiben",
    description="Bewerbungs- / Motivationsschreiben auf Deutsch im Stil des Nutzers",
    instructions=(
        "Schreibe ein vollständiges deutsches Anschreiben.\n"
        "Aufbau: Hook (ein echtes, konkretes Detail aus seiner Geschichte) → was er mitbringt "
        "(konkrete Stationen mit echten Zahlen aus den CV-Fakten) → warum genau diese Firma/Rolle "
        "(spezifisch auf das Posting bezogen, nicht generisch) → kurzer Abschluss mit Gesprächswunsch.\n"
        "Eine Seite. Anrede 'Sehr geehrte Damen und Herren,' wenn kein Ansprechpartner genannt ist, "
        "sonst die genannte Person. Gruß 'Mit freundlichen Grüßen'.\n"
        "Erfinde keine Fakten oder Zahlen; nutze nur, was im Profil steht. Wenn eine Info fehlt "
        "(z. B. konkretes Eintrittsdatum), lass eine klar erkennbare Lücke wie [Eintrittsdatum] statt zu raten."
    ),
)

# Generic fallback agent for any other German drafting task (email, LinkedIn message, etc.).
GENERIC = Agent(
    name="Text",
    description="Allgemeiner deutscher Text im Stil des Nutzers",
    instructions=(
        "Schreibe den angeforderten Text auf Deutsch im Stil des Nutzers. Halte dich an die Stilregeln, "
        "bleib knapp und konkret, erfinde keine Fakten. Markiere fehlende Infos als [Platzhalter]."
    ),
)

AGENTS: dict[str, Agent] = {
    "anschreiben": ANSCHREIBEN,
    "text": GENERIC,
}
DEFAULT_AGENT = "anschreiben"


_PROMPT = """Du schreibst im Namen und im persönlichen Stil von Troels Enigk. Du bist sein {agent_name}-Agent.
Dein Output klingt menschlich und nach IHM, nicht nach KI.

STILREGELN (strikt befolgen):
{style}

WAS DU ÜBER IHN WEISST (nur diese Fakten verwenden, nichts erfinden):
{profile}

AUFGABE:
{instructions}

BRIEFING DES NUTZERS:
\"\"\"{brief}\"\"\"
{posting_block}
Gib NUR den fertigen deutschen Text aus. Keine Vorrede, keine Erklärung, keine Meta-Kommentare,
keine Markdown-Codeblöcke. Beginne direkt mit dem Text."""


def _select_agent(brief: str, agent: str | None) -> Agent:
    if agent and agent.lower() in AGENTS:
        return AGENTS[agent.lower()]
    low = brief.lower()
    if any(k in low for k in ("anschreiben", "bewerbung", "motivationsschreiben", "cover letter", "stelle")):
        return ANSCHREIBEN
    return AGENTS[DEFAULT_AGENT]


def draft(brief: str, cfg: Config, posting: str = "", agent: str | None = None) -> dict:
    """Produce a humanized German draft.

    brief:   what to write (e.g. "Anschreiben für Founders Associate bei Moonscale").
    posting: optional job-posting text or extra source material to ground the draft.
    agent:   force a specific agent by key; else inferred from the brief.

    Returns {text, agent, used_posting, missing_self_vault}.
    """
    style, profile = load_self_context(cfg)
    chosen = _select_agent(brief, agent)

    posting_block = ""
    if posting.strip():
        posting_block = (
            "\n\nSTELLENAUSSCHREIBUNG / QUELLENMATERIAL "
            "(nur belegbare Aussagen daraus über die Firma verwenden):\n"
            f'"""{posting.strip()}"""\n'
        )

    prompt = _PROMPT.format(
        agent_name=chosen.name,
        style=style or "(keine Stildatei gefunden — schreibe knapp, ehrlich, ohne KI-Floskeln)",
        profile=profile or "(kein Profil gefunden)",
        instructions=chosen.instructions,
        brief=brief.replace('"""', "'''"),
        posting_block=posting_block,
    )

    text = call_text(prompt, primary=cfg.llm_primary, fallback=cfg.llm_fallback)
    text = _strip_fences(text)

    return {
        "text": text,
        "agent": chosen.name,
        "used_posting": bool(posting.strip()),
        "missing_self_vault": not style and not profile,
    }


def _strip_fences(text: str) -> str:
    """Remove a wrapping ```...``` block if the model added one anyway."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t
