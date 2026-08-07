"""
Named voice duos for the tts-podcast pipeline.

A *duo* bundles the two speaker configurations (name, prebuilt Gemini voice,
and baseline personality) that drive both the dialogue prompt and the TTS
preamble.  Instead of editing ``gemini.speaker1`` / ``gemini.speaker2`` by
hand for every episode, users pick a duo by name — built-in or defined in
their YAML config — via ``gemini.default_duo`` or the ``--duo`` CLI flag.

Two optional layers refine how a duo *sounds* without touching what it
*says*:

* a per-speaker ``voice_direction`` string (a Director's Note covering
  register, tempo, articulation and breathing).  It is read **only** by
  :func:`tts_podcast.tts_generator._build_tts_prompt`; it must never reach
  the dialogue prompt, which is the mirror image of the existing
  ``style_overlay`` key (dialogue-prompt-only, never read by the TTS path).
* duo-level ``scene`` and ``pace`` strings, used as *defaults* for
  ``gemini.tts_style.scene`` / ``gemini.tts_style.pace``.  A value already
  present in the user's own config always wins.

The personalities are intentionally written in English (even when the
dialogue language is e.g. French): Gemini handles language-mixed
meta-instructions robustly, and English fragments stay tight.

They also have two consumers with different sentence frames, and a
personality string must read correctly in both.  The TTS preamble renders
``"{name} is {personality}."`` while the dialogue prompt renders
``"- {name}: {personality}"``, so every personality is a noun phrase opening
with a determiner (``a`` / ``an`` / ``the``).  A bare job title such as
``"desk anchor; presses..."`` reads fine as a bullet but ships
``"Nora is desk anchor;"`` to the TTS model, and a garbled opening sentence
is exactly what pushes it back toward one averaged delivery.
:func:`tests.test_duos.TestPersonalityGrammar` guards the rule.

The built-in duo *slugs* (``warm``, ``contrast``, ``explorer``, ``journalist``,
``debate``) are a stable contract — renaming or removing one is a breaking
change for users who reference them in YAML or on the command line.  Each
voice is annotated with its official one-word descriptor; pair the descriptor
with the personality so the voice acting reinforces the character.  See
https://ai.google.dev/gemini-api/docs/speech-generation for voice previews.

Every built-in duo is a *different conversational dynamic* (transmission,
tempo flip, co-discovery, role asymmetry, open conflict), not the same
"excited host plus calm analyst" pairing under five names: that sameness is
what makes episodes blur together.  Two rules keep them apart and are worth
preserving when editing this registry:

* the ten voices are distinct and so are their ten official descriptors, so
  no two duos share a timbre family;
* within a duo the two ``voice_direction`` notes are opposed on register,
  tempo, articulation *and* breathing, and each note describes a steady trait
  rather than a trajectory (a note that ramps up would fight the
  anti-crescendo instruction in the TTS director's notes).

The ``voice_direction`` strings also sit inside a byte budget: they are
prepended to every TTS request alongside ``llm_summarizer._MAX_CHUNK_BYTES``
of dialogue.  Keep each one under ~160 characters and re-measure
:func:`tts_podcast.tts_generator._build_tts_prompt` before growing them.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import click

logger = logging.getLogger(__name__)


#: The duo selected when neither ``--duo`` nor ``gemini.default_duo`` is set
#: and no legacy ``gemini.speaker1`` / ``speaker2`` block is present.  It stays
#: ``"contrast"``: the tempo-flip dynamic (an idea machine against an editor)
#: is the most broadly usable default and its two voices sit at opposite ends
#: of the pitch-mobility axis, so a first episode never sounds mono-voiced.
DEFAULT_DUO = "contrast"


#: Built-in duos, available out of the box without any configuration.  A user
#: ``gemini.duos`` mapping is merged over this dict (same slug overrides the
#: built-in; new slugs extend it).  Each entry carries a human-readable
#: ``description`` plus ``speaker1`` / ``speaker2`` blocks shaped exactly like
#: the legacy ``gemini.speakerN`` config (``name`` / ``voice`` / ``personality``,
#: plus an optional TTS-only ``voice_direction``).  A duo may also carry
#: optional top-level ``scene`` and ``pace`` strings used as ``tts_style``
#: defaults.
BUILTIN_DUOS: dict[str, dict[str, Any]] = {
    # Dynamic: transmission.  A settled storyteller hands the thread to the
    # listener's stand-in; no doubt and no counter-argument.  Long turn, short
    # reaction; the only duo with real silence in it.
    "warm": {
        "description": (
            "Transmission, not debate: a settled storyteller and the newcomer "
            "who says it back in his own words."
        ),
        "scene": "two armchairs, one microphone, late evening",
        "pace": "unhurried, with real silences",
        "speaker1": {
            "name": "Vera",
            "voice": "Gacrux",  # Mature
            "personality": (
                "a seasoned storyteller who explains through things she has "
                "watched happen first-hand and never rushes to the point"
            ),
            "voice_direction": (
                "Low chest register, two thirds of normal tempo. Long rounded "
                "vowels, legato. A full beat of silence at each stop, deep "
                "breath between thoughts."
            ),
        },
        "speaker2": {
            "name": "Milo",
            "voice": "Achird",  # Friendly
            "personality": (
                "the listener's stand-in, hearing it for the first time, saying "
                "it back in his own words and asking the plain question"
            ),
            "voice_direction": (
                "Light mid-high register, no chest weight. Faster than ordinary "
                "speech, in short runs. Crisp attacks, clipped endings, quick "
                "shallow breaths."
            ),
        },
    },
    # Dynamic: tempo flip.  An idea machine generating volume against an editor
    # compressing it; the two agree, they just run at different speeds.
    "contrast": {
        "description": (
            "Tempo flips at every turn: an idea machine in bursts against an "
            "editor who lands it in one sentence."
        ),
        "scene": "a small studio, a whiteboard covered in half-erased diagrams",
        "pace": "brisk, changing speed at every handover",
        "speaker1": {
            "name": "Theo",
            "voice": "Puck",  # Upbeat
            "personality": (
                "an idea machine who offers three framings of a problem in a row, "
                "thinks out loud and swaps a metaphor for a better one"
            ),
            "voice_direction": (
                "Upper-mid register with wide pitch travel inside one sentence. "
                "Fast staccato bursts, hard first-syllable attack, sharp breath "
                "before each burst."
            ),
        },
        "speaker2": {
            "name": "Nadia",
            "voice": "Kore",  # Firm
            "personality": (
                "an editor who lets two ideas go past, keeps the third, then says "
                "it back in one sentence with nothing left over"
            ),
            "voice_direction": (
                "Narrow level register resolving downward at every stop. Steady "
                "metronomic tempo, even stress, inaudible breath, one unbroken "
                "line per turn."
            ),
        },
    },
    # Dynamic: co-discovery.  Two non-experts approach the same thing from
    # opposite ends; nobody holds the answer and nobody corrects the other.
    "explorer": {
        "description": (
            "Co-discovery with no expert: two non-experts coming at the same "
            "thing from opposite ends."
        ),
        "scene": "a kitchen table late at night, one laptop between them",
        "pace": "quick when something clicks, slow when it does not",
        "speaker1": {
            "name": "Iris",
            "voice": "Achernar",  # Soft
            "personality": (
                "an intuitive thinker who works in images, asks what a thing would "
                "look or feel like at that scale, and gets there sideways"
            ),
            "voice_direction": (
                "Mid register at library volume, close to the mic. Slow, with "
                "pauses inside the sentence; soft attacks, audible intake, lines "
                "trailing off."
            ),
        },
        "speaker2": {
            "name": "Sam",
            "voice": "Sadachbia",  # Lively
            "personality": (
                "a systems thinker who asks what breaks, what this plugs into and "
                "what happens two steps later, then reports back"
            ),
            "voice_direction": (
                "High register, projected as if to a room. Brisk and at the front "
                "of the beat; hard consonants, finished endings, quick breath and "
                "no run-up."
            ),
        },
    },
    # Dynamic: role asymmetry.  One asks and runs the clock, the other answers
    # at length under sourcing rules.  Formal register, no small talk.
    "journalist": {
        "description": (
            "Desk and field: an anchor asking short questions, a correspondent "
            "reporting long, sourced answers."
        ),
        "scene": "a broadcast studio, the anchor at the desk and the correspondent on a clean line",
        "pace": "brisk and controlled, news tempo",
        "speaker1": {
            "name": "Nora",
            "voice": "Pulcherrima",  # Forward
            "personality": (
                "a desk anchor who presses with short direct questions, holds the "
                "correspondent to the claim and runs the clock"
            ),
            "voice_direction": (
                "Mid-high forward placement, projected. Short clipped sentences at "
                "news tempo, clean attack on the first word, silent breath, flat "
                "endings."
            ),
        },
        "speaker2": {
            "name": "Marc",
            "voice": "Charon",  # Informative
            "personality": (
                "a field correspondent who reports what he has verified in tight "
                "blocks, attributes every claim and flags the unknowns"
            ),
            "voice_direction": (
                "Bottom of the register, resonant. Half that tempo, long even "
                "blocks; rounded sustained vowels, breath taken at the commas, no "
                "rise at the end."
            ),
        },
    },
    # Dynamic: open conflict.  Two advocates claiming the floor, one escalating
    # upward and the other escalating downward on the same axis.
    "debate": {
        "description": (
            "Open conflict: techno-optimist against hard-nosed skeptic. "
            "Best combined with --preset debate."
        ),
        "scene": "a bare table, two microphones facing each other",
        "pace": "quick exchanges, no dead air",
        "speaker1": {
            "name": "Robin",
            "voice": "Autonoe",  # Bright
            "personality": (
                "a techno-optimist who champions the upside, the opportunity and "
                "what becomes newly possible, arguing in good faith"
            ),
            "voice_direction": (
                "Bright upper register, leaning into the mic. Quick, no gap between "
                "sentences; pitch climbing through the clause, breath snatched on "
                "the fly."
            ),
        },
        "speaker2": {
            "name": "Sasha",
            "voice": "Algenib",  # Gravelly
            "personality": (
                "a hard-nosed skeptic who probes risks, costs, hype and failure "
                "modes, steel-mans the case against and lets nothing slide"
            ),
            "voice_direction": (
                "Low gravelly register, dropping in volume instead of rising. Slow, "
                "a long beat before each answer; blunt consonants, falling endings."
            ),
        },
    },
}


def validate_speaker(label: str, role: str, speaker: Any) -> dict[str, Any]:
    """
    Validate a single speaker block, whatever its source.

    The same rules apply to a duo speaker and to a legacy
    ``gemini.speaker1`` / ``speaker2`` block: both end up in
    ``gemini_cfg["speakerN"]`` and are read by the same consumers, so a
    malformed ``voice_direction`` must be rejected up front rather than
    crashing inside the TTS thread pool after the dialogue has been paid for.

    Parameters
    ----------
    label : str
        Human-readable origin of the block, used to prefix error messages
        (e.g. ``"Duo 'warm'"`` or ``"Config gemini"``).
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
        When *speaker* is not a mapping, is missing ``name`` / ``voice``, or
        carries a non-string ``voice_direction``.
    """
    if not isinstance(speaker, dict):
        raise click.BadParameter(
            f"{label} {role} must be a mapping with 'name' and 'voice'."
        )
    for field in ("name", "voice"):
        if not speaker.get(field):
            raise click.BadParameter(
                f"{label} {role} is missing required field {field!r}."
            )
    # voice_direction is optional; when present it must be a plain string so
    # the TTS preamble can interpolate it directly.  YAML makes a non-string
    # easy to hit: an unquoted `voice_direction: 1.5` or a `- ` list.
    direction = speaker.get("voice_direction")
    if direction is not None and not isinstance(direction, str):
        raise click.BadParameter(
            f"{label} {role} field 'voice_direction' must be a string, "
            f"got {type(direction).__name__}."
        )
    return speaker


def _validate_optional_text(duo_name: str, field: str, value: Any) -> str | None:
    """
    Validate an optional duo-level free-text field (``scene`` / ``pace``).

    Parameters
    ----------
    duo_name : str
        Slug of the duo being validated (used in error messages).
    field : str
        Name of the field being validated, e.g. ``"scene"``.
    value : Any
        The candidate value, possibly ``None``.

    Returns
    -------
    str or None
        The value unchanged when it is a non-empty string, ``None`` when it is
        absent or blank.

    Raises
    ------
    click.BadParameter
        When *value* is present but is not a string.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise click.BadParameter(
            f"Duo {duo_name!r} field {field!r} must be a string, "
            f"got {type(value).__name__}."
        )
    # Treat a blank string as "not set" so an empty YAML value does not
    # override the built-in tts_style behaviour with nothing.
    return value if value.strip() else None


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
) -> dict[str, Any] | None:
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
        *name* resolves; ``None`` when no duo was requested.  Each speaker
        block carries its optional ``voice_direction`` through unchanged.
        When the duo declares a non-blank top-level ``scene`` and/or ``pace``,
        the corresponding key is also present at the top level of the returned
        mapping (absent otherwise, so callers can distinguish "not set" from
        "set to empty").  These are *defaults* for ``gemini.tts_style``; the
        caller must let the user's own config win.

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
    label = f"Duo {stripped!r}"
    speaker1 = validate_speaker(label, "speaker1", duo.get("speaker1"))
    speaker2 = validate_speaker(label, "speaker2", duo.get("speaker2"))
    resolved: dict[str, Any] = {"speaker1": speaker1, "speaker2": speaker2}

    # Surface duo-level tts_style defaults only when actually declared, so the
    # caller can tell "the duo has an opinion" from "the duo is silent".
    for field in ("scene", "pace"):
        value = _validate_optional_text(stripped, field, duo.get(field))
        if value is not None:
            resolved[field] = value
    return resolved


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
