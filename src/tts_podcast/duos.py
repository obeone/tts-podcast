"""
Named voice duos for the tts-podcast pipeline.

A *duo* bundles the two speaker configurations (name, prebuilt Gemini voice,
and baseline personality) that drive both the dialogue prompt and the TTS
preamble.  Instead of editing ``gemini.speaker1`` / ``gemini.speaker2`` by
hand for every episode, users pick a duo by name — built-in or defined in
their YAML config — via ``gemini.default_duo`` or the ``--duo`` CLI flag.

The personalities are intentionally written in English (even when the
dialogue language is e.g. French): Gemini handles language-mixed
meta-instructions robustly, and English fragments stay tight.

The built-in duo *slugs* (``warm``, ``contrast``, ``explorer``, ``journalist``,
``debate``) are a stable contract — renaming or removing one is a breaking
change for users who reference them in YAML or on the command line.  Each
voice is annotated with its official one-word descriptor; pair the descriptor
with the personality so the voice acting reinforces the character.  See
https://ai.google.dev/gemini-api/docs/speech-generation for voice previews.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import click

logger = logging.getLogger(__name__)


#: The duo selected when neither ``--duo`` nor ``gemini.default_duo`` is set
#: and no legacy ``gemini.speaker1`` / ``speaker2`` block is present.
DEFAULT_DUO = "contrast"


#: Built-in duos, available out of the box without any configuration.  A user
#: ``gemini.duos`` mapping is merged over this dict (same slug overrides the
#: built-in; new slugs extend it).  Each entry carries a human-readable
#: ``description`` plus ``speaker1`` / ``speaker2`` blocks shaped exactly like
#: the legacy ``gemini.speakerN`` config (``name`` / ``voice`` / ``personality``).
BUILTIN_DUOS: dict[str, dict[str, Any]] = {
    "warm": {
        "description": "Warm host + friendly co-host — accessible, mainstream feel.",
        "speaker1": {
            "name": "Alex",
            "voice": "Sulafat",  # Warm
            "personality": "warm, welcoming, makes complex tech feel human and inviting",
        },
        "speaker2": {
            "name": "Jordan",
            "voice": "Achird",  # Friendly
            "personality": "friendly, witty, asks the questions the listener is thinking",
        },
    },
    "contrast": {
        "description": "Upbeat host + firm co-host — high timbre contrast (Google's own pairing).",
        "speaker1": {
            "name": "Theo",
            "voice": "Puck",  # Upbeat
            "personality": "energetic, curious, quick to get excited about tech innovations",
        },
        "speaker2": {
            "name": "Nadia",
            "voice": "Kore",  # Firm
            "personality": "firm, analytical, grounds the hype with evidence and historical context",
        },
    },
    "explorer": {
        "description": "Excitable explorer + knowledgeable expert — vulgarisation-friendly.",
        "speaker1": {
            "name": "Sam",
            "voice": "Fenrir",  # Excitable
            "personality": "exuberant, rapid-fire curiosity, loves wild what-ifs and tangents",
        },
        "speaker2": {
            "name": "Vera",
            "voice": "Sadaltager",  # Knowledgeable
            "personality": "measured domain expert, precise, gently corrects overreach",
        },
    },
    "journalist": {
        "description": "Bright reporter + smooth analyst — fast-paced tech-journalism feel.",
        "speaker1": {
            "name": "Nora",
            "voice": "Zephyr",  # Bright
            "personality": "bright, punchy, drives the narrative forward with momentum",
        },
        "speaker2": {
            "name": "Marc",
            "voice": "Algieba",  # Smooth
            "personality": "smooth, reflective, adds depth and nuance to every point",
        },
    },
    "debate": {
        "description": (
            "Opposing viewpoints — techno-optimist vs hard-nosed skeptic. "
            "Best combined with --preset debate."
        ),
        "speaker1": {
            "name": "Robin",
            "voice": "Laomedeia",  # Upbeat
            "personality": (
                "techno-optimist; champions the upside, the opportunity, and what "
                "becomes newly possible; argues in good faith for adoption"
            ),
        },
        "speaker2": {
            "name": "Sasha",
            "voice": "Algenib",  # Gravelly
            "personality": (
                "hard-nosed skeptic; probes risks, costs, hype and failure modes; "
                "steel-mans the case against and refuses to let claims slide"
            ),
        },
    },
}


def _validate_speaker(duo_name: str, role: str, speaker: Any) -> dict[str, Any]:
    """
    Validate a single speaker block of a duo.

    Parameters
    ----------
    duo_name : str
        Slug of the duo being validated (used in error messages).
    role : str
        Either ``"speaker1"`` or ``"speaker2"``.
    speaker : Any
        The candidate speaker mapping.

    Returns
    -------
    dict[str, Any]
        The validated speaker mapping (same object).

    Raises
    ------
    click.BadParameter
        When *speaker* is not a mapping or is missing ``name`` / ``voice``.
    """
    if not isinstance(speaker, dict):
        raise click.BadParameter(
            f"Duo {duo_name!r} {role} must be a mapping with 'name' and 'voice'."
        )
    for field in ("name", "voice"):
        if not speaker.get(field):
            raise click.BadParameter(
                f"Duo {duo_name!r} {role} is missing required field {field!r}."
            )
    return speaker


def available_duos(config_duos: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """
    Return the full set of duos: built-ins overlaid with config-defined ones.

    Parameters
    ----------
    config_duos : dict[str, Any] or None, optional
        The ``gemini.duos`` mapping from the loaded config, if any.  Entries
        with a slug matching a built-in override it; new slugs extend the set.

    Returns
    -------
    dict[str, dict[str, Any]]
        A deep copy of the merged duo registry (safe for the caller to mutate).
    """
    merged: dict[str, dict[str, Any]] = copy.deepcopy(BUILTIN_DUOS)
    if config_duos:
        if not isinstance(config_duos, dict):
            raise click.BadParameter(
                f"gemini.duos must be a mapping of duo-name → duo, got "
                f"{type(config_duos).__name__}."
            )
        for name, duo in config_duos.items():
            if not isinstance(duo, dict):
                raise click.BadParameter(
                    f"Duo {name!r} must be a mapping, got {type(duo).__name__}."
                )
            merged[str(name)] = copy.deepcopy(duo)
    return merged


def resolve_duo(
    name: str | None,
    config_duos: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]] | None:
    """
    Resolve a duo *name* to its ``{"speaker1": ..., "speaker2": ...}`` blocks.

    This is the single source of truth for duo validation across every entry
    point (CLI ``--duo``, ``gemini.default_duo`` in YAML, the ``config init``
    wizard) — Click's ``Choice`` cannot validate a config value, so a typo
    arriving via YAML would otherwise fall through silently.

    Parameters
    ----------
    name : str or None
        Duo slug (case-insensitive) to resolve.  ``None`` and the empty string
        both mean "no duo selected" and return ``None`` so the caller can fall
        back to legacy ``gemini.speakerN`` blocks.
    config_duos : dict[str, Any] or None, optional
        The ``gemini.duos`` mapping from the loaded config, if any.

    Returns
    -------
    dict or None
        A deep-copied mapping with ``speaker1`` and ``speaker2`` keys when
        *name* resolves; ``None`` when no duo was requested.

    Raises
    ------
    click.BadParameter
        When *name* is a non-empty string that matches no duo, or when the
        resolved duo is structurally invalid.  The message lists valid slugs.
    """
    if name is None:
        return None
    stripped = name.strip().lower()
    if stripped == "":
        return None

    registry = available_duos(config_duos)
    if stripped not in registry:
        valid = ", ".join(sorted(registry.keys()))
        raise click.BadParameter(f"Unknown duo {name!r}. Valid duos: {valid}.")

    duo = copy.deepcopy(registry[stripped])
    speaker1 = _validate_speaker(stripped, "speaker1", duo.get("speaker1"))
    speaker2 = _validate_speaker(stripped, "speaker2", duo.get("speaker2"))
    return {"speaker1": speaker1, "speaker2": speaker2}


def describe_duos(config_duos: dict[str, Any] | None = None) -> list[tuple[str, str, str, str]]:
    """
    Summarise every available duo for human-facing listing (CLI ``duos`` cmd).

    Parameters
    ----------
    config_duos : dict[str, Any] or None, optional
        The ``gemini.duos`` mapping from the loaded config, if any.

    Returns
    -------
    list[tuple[str, str, str, str]]
        One ``(slug, description, speaker1_summary, speaker2_summary)`` tuple
        per duo, built-ins first (in declaration order) then config-only
        slugs.  Each speaker summary reads ``"Name (Voice)"``.
    """
    registry = available_duos(config_duos)
    ordered = list(BUILTIN_DUOS.keys())
    ordered += [name for name in registry if name not in BUILTIN_DUOS]

    rows: list[tuple[str, str, str, str]] = []
    for slug in ordered:
        duo = registry[slug]
        desc = str(duo.get("description", ""))
        sp1 = duo.get("speaker1", {}) or {}
        sp2 = duo.get("speaker2", {}) or {}
        sp1_summary = f"{sp1.get('name', '?')} ({sp1.get('voice', '?')})"
        sp2_summary = f"{sp2.get('name', '?')} ({sp2.get('voice', '?')})"
        rows.append((slug, desc, sp1_summary, sp2_summary))
    return rows
