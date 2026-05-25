"""
Curated style presets and helpers for the tts-podcast dialogue prompt.

The presets are short, opinionated English prompt fragments that influence the
overall tone of the generated dialogue.  They are intentionally written in
English (even when the dialogue language is e.g. French) because Gemini handles
language-mixed meta-instructions robustly and English fragments stay tight.

The five presets are a stable contract: renaming or removing one is a breaking
change for users who reference them in their YAML config.
"""

from __future__ import annotations

import logging

import click

logger = logging.getLogger(__name__)


STYLE_PRESETS: dict[str, str] = {
    "casual": (
        "Adopt a relaxed, conversational tone — like two friends chatting over "
        "coffee. Favour everyday language, light digressions, and the occasional "
        "self-deprecating aside. Avoid jargon unless it earns its keep."
    ),
    "academic": (
        "Adopt a measured, academic register. Define terms precisely, attribute "
        "claims to their sources, and structure each turn around a clear "
        "thesis-evidence-implication arc. Prefer rigorous phrasing over "
        "rhetorical flourish; concede uncertainty explicitly when warranted."
    ),
    "humorous": (
        "Lean into wit, wordplay, and well-timed irony. Find absurdities in the "
        "subject matter and call them out without being mean-spirited. Keep the "
        "facts honest — the humour rides on top of the substance, never replaces it."
    ),
    "debate": (
        "Frame the conversation as a structured debate: one host advocates a "
        "position, the other steel-mans the opposite. Surface disagreements "
        "explicitly, name the underlying assumptions, and let neither side "
        "off the hook. End each thread with a brief synthesis of where the "
        "tension genuinely sits."
    ),
    "vulgarized": (
        "Translate technical material for a curious non-expert audience. Use "
        "analogies and concrete examples; spell out acronyms on first use; "
        "break down each new concept into the smallest digestible step. Assume "
        "the listener is smart but unfamiliar with the field's jargon."
    ),
}


_PRESET_NONE_SENTINEL = "none"


def validate_preset(name: str | None) -> str | None:
    """
    Resolve a preset name to its prompt fragment.

    The function is the single source of truth for preset validation across
    every entry point (CLI, YAML config, ``config init`` wizard) — Click's
    ``Choice`` only validates CLI input, so a typo arriving via config would
    silently fall through without this check.

    Parameters
    ----------
    name : str or None
        Preset key (case-insensitive) to resolve.  ``None``, the empty string,
        and the literal sentinel ``"none"`` all mean "no preset selected".

    Returns
    -------
    str or None
        The preset's prompt fragment when *name* is a valid key; ``None`` when
        no preset is requested.

    Raises
    ------
    click.BadParameter
        When *name* is a non-empty string that does not match any preset key.
        The error message lists the valid keys.
    """
    if name is None:
        return None
    stripped = name.strip().lower()
    if stripped == "" or stripped == _PRESET_NONE_SENTINEL:
        return None
    if stripped in STYLE_PRESETS:
        return STYLE_PRESETS[stripped]
    valid = ", ".join(sorted(STYLE_PRESETS.keys()))
    raise click.BadParameter(
        f"Unknown preset {name!r}. Valid presets: {valid} (or 'none' to disable)."
    )


def truncate_with_warning(
    text: str | None,
    field: str,
    *,
    cap: int = 500,
) -> str | None:
    """
    Truncate a free-text field to *cap* characters, warning when truncated.

    Used by all four free-text CLI surfaces (``--style``, ``--speaker1-style``,
    ``--speaker2-style``, ``--angle``) to keep the prompt's steering signal
    coherent — a several-thousand-character free-text injection would drown
    the article content and shift the model's attention away from its core task.

    Parameters
    ----------
    text : str or None
        Input text.  Returned untouched when ``None`` or already within *cap*.
    field : str
        Field name used in the warning message (so users know which flag
        triggered the truncation).
    cap : int, keyword-only, optional
        Maximum length in characters, by default 500.

    Returns
    -------
    str or None
        The original text when it fits, the truncated text otherwise, or
        ``None`` when the input was ``None``.
    """
    if text is None:
        return None
    if len(text) <= cap:
        return text
    logger.warning(
        "Field %r exceeds %d characters (%d) — truncating.",
        field,
        cap,
        len(text),
    )
    return text[:cap]
