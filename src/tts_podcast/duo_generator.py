"""
AI-powered voice duo generator for the tts-podcast pipeline.

Uses Google Gemini with structured output (JSON schema + voice enum) to
select a contextually appropriate two-voice pair from the 30 prebuilt
Gemini TTS voices, given the content and tone of the episode's sources.

The model acts as a casting director: it reads the content signal (titles,
summaries, optionally research notes) and picks the two voices whose
one-word descriptor best matches the episode's tone.  It also names the
hosts and writes brief personality strings.

This is the first module in the codebase to use ``response_mime_type`` /
``response_schema``; all other call patterns (client creation, retry
decorator, token tracking) are identical to
:mod:`tts_podcast.llm_summarizer`.

Convention (mirrored from duos.py):
    Personality strings are ALWAYS written in English, regardless of the
    podcast language.  The ``language`` parameter is used only as tone
    context (e.g. "the podcast is in French") — never as the personality
    writing language.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from google import genai
from google.genai import types

from tts_podcast.duos import GEMINI_VOICES, _validate_speaker
from tts_podcast.retry import gemini_retry

if TYPE_CHECKING:
    from tts_podcast.models import Source
    from tts_podcast.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

# Maximum characters of full_text per source included in the prompt.
# Keeps the prompt well-bounded even with many long articles.
_MAX_FULL_TEXT_CHARS = 1500

# Maximum number of sources for which we include a full-text excerpt.
# Beyond this count, only titles and summaries are shown to the model.
_MAX_SOURCES_FULL = 4

# The JSON schema describing the expected structured output.
# Both voice fields are constrained to an ENUM of valid Gemini voice names
# (list(GEMINI_VOICES)) so the API cannot return a hallucinated voice name
# that would silently break the TTS stage later in the pipeline.
_DUO_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    description="A generated podcast voice duo with two speakers.",
    properties={
        "description": types.Schema(
            type=types.Type.STRING,
            description="One-sentence description of the duo's dynamic and tone.",
        ),
        "speaker1": types.Schema(
            type=types.Type.OBJECT,
            description="First speaker configuration.",
            properties={
                "name": types.Schema(
                    type=types.Type.STRING,
                    description="Host first name (short, memorable).",
                ),
                "voice": types.Schema(
                    type=types.Type.STRING,
                    description="Prebuilt Gemini TTS voice name.",
                    # Constrain to valid voices — prevents hallucinated names
                    # that would cause a TTS API error at generation time.
                    enum=list(GEMINI_VOICES),
                ),
                "personality": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "Short personality string in English (2-3 traits), "
                        "matching the voice descriptor."
                    ),
                ),
            },
            required=["name", "voice", "personality"],
        ),
        "speaker2": types.Schema(
            type=types.Type.OBJECT,
            description="Second speaker configuration.",
            properties={
                "name": types.Schema(
                    type=types.Type.STRING,
                    description="Host first name (short, memorable).",
                ),
                "voice": types.Schema(
                    type=types.Type.STRING,
                    description="Prebuilt Gemini TTS voice name.",
                    # Same enum constraint as speaker1.
                    enum=list(GEMINI_VOICES),
                ),
                "personality": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "Short personality string in English (2-3 traits), "
                        "matching the voice descriptor."
                    ),
                ),
            },
            required=["name", "voice", "personality"],
        ),
    },
    required=["description", "speaker1", "speaker2"],
)


def _build_voice_catalogue() -> str:
    """
    Build a formatted string listing all available voices with their descriptors.

    Reads directly from :data:`tts_podcast.duos.GEMINI_VOICES` so the
    catalogue stays in sync without a separate hardcoded list.

    Returns
    -------
    str
        Multi-line string with one ``"  Name (Descriptor)"`` entry per voice,
        suitable for inclusion in the Gemini system instruction.
    """
    lines = [f"  {name} ({descriptor})" for name, descriptor in GEMINI_VOICES.items()]
    return "\n".join(lines)


def _build_system_instruction() -> str:
    """
    Build the system instruction for the duo-generation call.

    The instruction explains the casting-director role, lists every available
    voice with its descriptor, and provides tone-matching guidance so the
    model can reason from content tone to voice descriptor.

    Returns
    -------
    str
        System instruction text to pass as
        ``GenerateContentConfig.system_instruction``.
    """
    voice_catalogue = _build_voice_catalogue()

    return f"""\
You are a casting director for a two-host podcast. Your job is to select a \
complementary voice pair from the available prebuilt Gemini TTS voices and \
assign each host a short name and personality.

## Available voices (Name → Descriptor)

{voice_catalogue}

## Tone-matching guidance

Match the voice descriptor to the content's emotional register:
- Grave / analytical content  → Even, Mature, Informative, Firm, Gravelly
- Light / enthusiast content  → Bright, Upbeat, Breezy, Lively, Youthful
- Warm / narrative content    → Warm, Friendly, Gentle, Soft, Easy-going
- Fast / journalistic content → Clear, Forward, Smooth, Casual

Pick two voices with contrasting descriptors for good sonic variety. \
The two voices MUST be different from each other.

## Personality convention

Write ALL personality strings in English — even when the podcast language is \
French or another language. This is a hard convention: English personalities \
mix robustly with non-English dialogue prompts and match the convention used \
throughout the duos module.
"""


def _build_prompt(
    sources: list[Source],
    research_notes: str,
    language: str,
) -> str:
    """
    Build the user-turn prompt for the duo-generation call.

    Encodes enough content signal (titles, summaries, truncated full text
    for the first few sources) so the model can reason about tone and pick
    matching voice descriptors.

    Parameters
    ----------
    sources : list[Source]
        Scraped / loaded sources for this episode.
    research_notes : str
        Accumulated research notes from the research stage (may be empty).
    language : str
        Natural language of the podcast (e.g. ``"French"``), used as tone
        context only — not the language for personality strings.

    Returns
    -------
    str
        The user-turn prompt text ready to pass to ``generate_content``.
    """
    lines: list[str] = []

    lines.append(f"The podcast will be in **{language}**.")
    lines.append("")
    lines.append("## Episode sources")
    lines.append("")

    for i, src in enumerate(sources, start=1):
        lines.append(f"### Source {i}: {src.title or src.url}")
        if src.summary:
            lines.append(f"**Summary:** {src.summary}")
        # Include a full-text excerpt only for the first N sources to cap
        # prompt size; for the rest, the title + summary is sufficient signal.
        if i <= _MAX_SOURCES_FULL and src.full_text:
            snippet = src.full_text[:_MAX_FULL_TEXT_CHARS]
            if len(src.full_text) > _MAX_FULL_TEXT_CHARS:
                snippet += "…"
            lines.append(f"**Excerpt:** {snippet}")
        lines.append("")

    if research_notes and research_notes.strip():
        lines.append("## Research notes")
        lines.append("")
        # Cap research notes at 2000 chars to avoid oversized prompts.
        notes_snippet = research_notes[:2000]
        if len(research_notes) > 2000:
            notes_snippet += "…"
        lines.append(notes_snippet)
        lines.append("")

    lines.append(
        "Based on the tone, subject matter, and emotional register of the content above, "
        "choose a complementary two-voice duo from the available voices. "
        "Pick voices whose descriptor matches the content's tone "
        "(e.g. grave / technical content → Even, Mature, Informative; "
        "energetic / pop-science content → Upbeat, Bright, Excitable). "
        "The two voices MUST be different. "
        "Write the personality strings in **English** regardless of the podcast language."
    )

    return "\n".join(lines)


def generate_duo(
    sources: list[Source],
    research_notes: str,
    gemini_cfg: dict[str, Any],
    token_tracker: TokenTracker | None = None,
    *,
    language: str = "French",
) -> dict[str, Any]:
    """
    Generate a contextually appropriate voice duo using Gemini structured output.

    Analyses the episode's sources (and optional research notes) to pick two
    prebuilt Gemini TTS voices whose descriptors match the content's tone.
    Returns a duo dict shaped exactly like entries in
    :data:`tts_podcast.duos.BUILTIN_DUOS`.

    The API call uses structured output (``response_mime_type="application/json"``
    with a ``response_schema`` whose voice fields are constrained to an ENUM of
    all valid :data:`~tts_podcast.duos.GEMINI_VOICES` keys).  This guarantees
    the returned voice names are real Gemini voices that the TTS stage can use
    without any extra validation round-trip.

    Personality strings are ALWAYS generated in English, regardless of
    ``language`` — this mirrors the convention documented at the top of
    :mod:`tts_podcast.duos`.

    Parameters
    ----------
    sources : list[Source]
        Scraped or locally loaded sources for this episode.  Each source's
        ``title``, ``summary``, and the first :data:`_MAX_FULL_TEXT_CHARS`
        characters of ``full_text`` are fed to the model as content signal.
    research_notes : str
        Accumulated research notes from :mod:`tts_podcast.research` (may be
        empty string when research was not run).
    gemini_cfg : dict[str, Any]
        Loaded Gemini configuration dict (from
        :func:`tts_podcast.config.load_config`).
        Required keys: ``api_key``, ``text_model``.  Optional: ``service_tier``.
    token_tracker : TokenTracker or None, optional
        When provided, records the token usage of the Gemini call for cost
        accounting.  Passing ``None`` silently skips tracking.
    language : str, optional
        Natural language of the podcast (default ``"French"``).  Used as tone
        context in the prompt only — does not change the language of the
        generated personality strings (always English).

    Returns
    -------
    dict[str, Any]
        A duo dict with keys ``"description"``, ``"speaker1"``, ``"speaker2"``.
        Each speaker block contains ``"name"``, ``"voice"``, and
        ``"personality"``.  Shaped identically to
        :data:`tts_podcast.duos.BUILTIN_DUOS` entries.

    Raises
    ------
    RuntimeError
        When the Gemini response cannot be parsed into a valid duo dict, or
        when the returned voice is not in
        :data:`~tts_podcast.duos.GEMINI_VOICES`.
    click.BadParameter
        Propagated from :func:`tts_podcast.duos._validate_speaker` when a
        speaker block is missing ``name`` or ``voice``.
    google.genai.errors.ServerError
        Re-raised after :data:`tts_podcast.retry._MAX_ATTEMPTS` failed
        attempts (handled transparently by
        :func:`~tts_podcast.retry.gemini_retry`).

    Examples
    --------
    >>> duo = generate_duo(sources, "", gemini_cfg, language="French")
    >>> set(duo.keys()) == {"description", "speaker1", "speaker2"}
    True
    >>> duo["speaker1"]["voice"] in GEMINI_VOICES
    True
    """
    prompt = _build_prompt(sources, research_notes, language)
    system_instruction = _build_system_instruction()

    logger.info(
        "Generating voice duo with model '%s' for %d source(s) (language=%s).",
        gemini_cfg["text_model"],
        len(sources),
        language,
    )
    logger.debug("Duo-generation prompt (%d chars):\n%s", len(prompt), prompt)

    client = genai.Client(api_key=gemini_cfg["api_key"])
    service_tier = gemini_cfg.get("service_tier")

    # Mirror the config_kwargs pattern from llm_summarizer.generate_dialogue
    # (lines 789-814) for consistency across all Gemini text calls.
    config_kwargs: dict[str, Any] = {
        # Structured output: constrain response to our JSON schema.
        # The voice fields in _DUO_RESPONSE_SCHEMA use enum=list(GEMINI_VOICES)
        # so the model CANNOT return a hallucinated voice name.
        "response_mime_type": "application/json",
        "response_schema": _DUO_RESPONSE_SCHEMA,
        "system_instruction": system_instruction,
        # Duo generation is short; 512 tokens is more than enough.
        "max_output_tokens": 512,
    }
    if service_tier:
        # Service tier is passed as a custom HTTP header (Gemini text API
        # only; TTS never uses it — see CLAUDE.md key invariants).
        config_kwargs["http_options"] = types.HttpOptions(
            headers={"x-goog-api-service-tier": service_tier},
        )

    @gemini_retry
    def _call_api() -> Any:
        return client.models.generate_content(
            model=gemini_cfg["text_model"],
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )

    response = _call_api()

    # Record token usage for cost accounting when a tracker is wired in.
    if token_tracker is not None:
        token_tracker.record_usage(gemini_cfg["text_model"], response.usage_metadata)

    # Parse the structured output.
    # Prefer response.parsed when the SDK populates it (newer SDK versions
    # with response_schema support); fall back to json.loads(response.text).
    raw: Any = None
    if hasattr(response, "parsed") and response.parsed is not None:
        raw = response.parsed
    else:
        text = response.text or ""
        if not text.strip():
            raise RuntimeError(
                "Gemini returned an empty response for duo generation. "
                "Check that the text model supports structured output."
            )
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Gemini returned non-JSON for duo generation: {text[:200]!r}"
            ) from exc

    # Normalise: the SDK may return a dict-like Pydantic object; coerce to dict.
    if not isinstance(raw, dict):
        try:
            raw = dict(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot convert Gemini structured output to dict: {raw!r}"
            ) from exc

    # Defensive voice check: the enum in the schema should prevent invalid
    # voices, but if the SDK bypasses it (older SDK version / schema ignored),
    # we catch it here with a clear error rather than a cryptic TTS failure.
    for role in ("speaker1", "speaker2"):
        speaker = raw.get(role, {})
        voice = speaker.get("voice", "") if isinstance(speaker, dict) else ""
        if voice and voice not in GEMINI_VOICES:
            raise RuntimeError(
                f"Gemini returned voice {voice!r} for {role} which is not in "
                f"GEMINI_VOICES. Valid voices: {sorted(GEMINI_VOICES)}"
            )

    # Validate structure via the canonical duos validator.
    # Raises click.BadParameter on missing name/voice fields.
    _validate_speaker("auto", "speaker1", raw.get("speaker1", {}))
    _validate_speaker("auto", "speaker2", raw.get("speaker2", {}))

    duo: dict[str, Any] = {
        "description": str(raw.get("description", "")),
        "speaker1": dict(raw["speaker1"]),
        "speaker2": dict(raw["speaker2"]),
    }

    logger.info(
        "Generated duo: %s (%s) + %s (%s) — %s",
        duo["speaker1"].get("name"),
        duo["speaker1"].get("voice"),
        duo["speaker2"].get("name"),
        duo["speaker2"].get("voice"),
        duo.get("description", ""),
    )

    return duo
