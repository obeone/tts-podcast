"""
LLM-based dialogue generator for the tts-podcast tool.

Uses Google Gemini to turn one or more scraped articles into a
conversational two-host podcast dialogue, then splits the output into
byte-size-bounded chunks ready for TTS processing.

Optionally injects complementary research notes (produced by
:mod:`tts_podcast.research`) into the prompt so the hosts can weave the
extra context naturally into the discussion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from google import genai
from google.genai import types

from tts_podcast.retry import gemini_retry
from tts_podcast.style_presets import truncate_with_warning, validate_preset

if TYPE_CHECKING:
    from rich.progress import Progress
    from tts_podcast.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

# Maximum UTF-8 byte size for a single dialogue chunk sent to TTS.
# Set to 3000 to leave ~600-800 bytes of headroom for the TTS preamble
# that tts_generator.py prepends before sending to the Gemini TTS API.
_MAX_CHUNK_BYTES = 3000

# Maximum number of generation attempts before raising a hard error.
_DIALOGUE_MAX_ATTEMPTS = 3

_THINKING_LEVEL_VALID = {"minimal", "low", "medium", "high"}


_AUDIO_TAGS_GUIDANCE = """\
- When a speaker turn has a distinct emotional or pacing colour, start it with \
an English audio tag in square brackets (e.g., "[enthusiasm] Incroyable !"). \
Skip the opening tag when the line is plainly neutral — don't force a tag \
just to have one.
- You may also sprinkle additional audio tags mid-turn when they add real \
expressive value — sparingly, never more than one every couple of sentences.
  * Non-verbal vocalizations: [laughs], [gasp], [whispers]
  * Pacing: [short pause], [long pause], [slow], [fast]
  * Emotion: [enthusiasm], [curiosity], [amusement], [seriousness], \
[cautious], [relief], [awe], [confusion]
- Audio tags MUST be in English and enclosed in square brackets, even when the \
surrounding dialogue is in another language.
- Never place two tags back to back — always separate them with spoken text or \
punctuation.
- Do NOT also add a French parenthetical cue describing the same effect when a \
bracketed audio tag is used."""

_PAREN_CUES_GUIDANCE = """\
- EVERY speaker turn MUST start with an inline emotional cue in parentheses \
(e.g., "(avec enthousiasme) Incroyable !"). This sets the delivery tone for \
the turn.
- You may also add additional parenthetical cues mid-turn to guide delivery \
(e.g., "(sceptique)", "(posément)", "(en pesant ses mots)", \
"(après une courte pause)"). Vary them naturally according to the content and \
pace of the discussion."""

_EXAMPLE_AUDIO_TAGS = """\
{speaker1}: [neutral] {speaker2}, on a une annonce de Google cette semaine sur leur API.
{speaker2}: [curiosity] Ah ? [short pause] Qu'est-ce qu'ils publient ?
{speaker1}: [animated] Une réduction de latence de cinquante pour cent sur les requêtes longues — c'est pas rien.
{speaker2}: [thoughtful] D'accord. [short pause] Ils expliquent comment ils y arrivent, ou c'est un chiffre marketing ?
{speaker1}: [matter-of-fact] Ils parlent d'un nouveau routage côté inférence. Le détail technique reste vague.
{speaker2}: [skeptical] Mouais. [short pause] On verra à l'usage."""

_EXAMPLE_PAREN_CUES = """\
{speaker1}: (de manière neutre) {speaker2}, on a une annonce de Google cette semaine sur leur API.
{speaker2}: (curieux) Ah ? Qu'est-ce qu'ils publient ?
{speaker1}: (avec entrain) Une réduction de latence de cinquante pour cent sur les requêtes longues — c'est pas rien.
{speaker2}: (en réfléchissant) D'accord. Ils expliquent comment ils y arrivent, ou c'est un chiffre marketing ?
{speaker1}: (de façon factuelle) Ils parlent d'un nouveau routage côté inférence. Le détail technique reste vague.
{speaker2}: (sceptique) Mouais. On verra à l'usage."""

_RESEARCH_SECTION_TEMPLATE = """\
Complementary research (from Google Search grounding — the dialogue MUST \
incorporate these findings substantively; when you credit a source out loud, \
use only a short reference such as the domain or publication name, never the \
full URL):
{notes}

"""

_SYSTEM_PROMPT_TEMPLATE = """\
You are a podcast script writer. Your job is to create an engaging, \
conversational podcast dialogue between two hosts: {speaker1} and {speaker2}.

Host personalities:
- {speaker1}: {speaker1_personality}
- {speaker2}: {speaker2_personality}
{speaker_adjustments_block}
Instructions:
- The article(s) below are the central topic of the episode.
- Discuss them in depth: explain what they are about, why they matter, \
and explore implications or connections to broader trends.
- The entire dialogue MUST be written in {language}.
- Target episode length: about {target_minutes:.0f} minutes of spoken \
conversation (~{target_words} words at a {wpm} wpm conversational pace).
- Hard minimum: ~{min_words} words (~{min_minutes:.0f} min). Do NOT wrap up \
shorter than this.
- Soft maximum: ~{max_words} words (~{max_minutes:.0f} min). Wrap up before \
exceeding this; trim depth on secondary points rather than blowing past it.
- Keep the tone informative but lively — like two curious friends catching up on tech news.
{style_block}{angle_line}{research_directive_line}- Reflect each host's personality in their speaking style and reactions.
{delivery_cues_guidance}
- Use shorter sentences for excitement, longer ones for analysis.
- Never read a full URL aloud — they sound awful in speech. When a speaker \
needs to credit a source, use only a short spoken reference: a bare domain \
(e.g. "example.com"), a publication name, or the article title. Strip the \
"https://", "www.", paths, query strings, and tracking parameters. If even \
the short form would be clunky, just describe the source ("the official \
blog", "a recent Wired piece") instead of naming it.
- End the dialogue with a brief conclusion that recaps the key takeaway and \
adds a witty remark, favourite point, or thought-provoking sign-off.

STRICT OUTPUT FORMAT:
- Each line must follow exactly this pattern: SpeakerName: dialogue text
- Alternate between {speaker1} and {speaker2}.
- Delivery cues go INSIDE the dialogue text, never in the SpeakerName prefix.
- Do NOT add blank lines between turns.
- Do NOT add any introduction or conclusion outside of the dialogue format.

Example output format:
{example_dialogue}

{research_section}Articles:
{articles}
"""


def _audio_tags_enabled(gemini_cfg: dict) -> bool:
    """
    Decide whether inline English audio tags should be requested from the LLM.

    Policy:

    - ``gemini.tts_style.audio_tags`` set to ``"on"``/``True`` → always enabled.
    - Set to ``"off"``/``False`` → always disabled.
    - Set to ``"auto"``, missing, or any other value → enabled iff the
      configured ``tts_model`` name starts with ``"gemini-3"`` (Gemini 3.x
      Flash TTS introduced bracketed audio-tag support).

    Parameters
    ----------
    gemini_cfg : dict
        Resolved Gemini configuration section.

    Returns
    -------
    bool
        ``True`` when audio tags should be included in the dialogue prompt.
    """
    setting = gemini_cfg.get("tts_style", {}).get("audio_tags", "auto")
    if isinstance(setting, bool):
        return setting
    normalised = str(setting).strip().lower()
    if normalised in {"on", "true", "yes", "1", "enabled"}:
        return True
    if normalised in {"off", "false", "no", "0", "disabled"}:
        return False
    if normalised not in {"auto", ""}:
        logger.warning(
            "Unknown gemini.tts_style.audio_tags value %r — falling back to auto-detection.",
            setting,
        )
    tts_model = str(gemini_cfg.get("tts_model", "")).lower()
    return tts_model.startswith("gemini-3")


def _build_thinking_config(
    model: str,
    thinking_level: str | None,
    thinking_budget: int | None,
) -> types.ThinkingConfig | None:
    """
    Return a ``types.ThinkingConfig`` for the model family, or ``None``.

    Gemini 3.x models use ``thinking_level`` (a string enum); Gemini 2.5 and
    other models use ``thinking_budget`` (an integer).  Passing the wrong field
    for a model family is silently ignored with a warning so callers can set
    both in config without errors.

    Parameters
    ----------
    model : str
        The Gemini model name (e.g. ``"gemini-3.5-flash"``).
    thinking_level : str or None
        Desired thinking level for Gemini 3.x models.  Accepted values
        (case-insensitive): ``"minimal"``, ``"low"``, ``"medium"``,
        ``"high"``.  ``None`` or empty string means "not set".
    thinking_budget : int or None
        Thinking token budget for Gemini 2.5 / other models.  ``0`` disables
        thinking; ``-1`` means dynamic.  ``None`` means "not set".

    Returns
    -------
    types.ThinkingConfig or None
        A configured ``ThinkingConfig`` instance, or ``None`` when no
        configuration applies (not set, wrong family, or invalid value).
    """
    is_3x = str(model).startswith("gemini-3")

    # Normalise: treat empty string as "not set"
    level_normalised: str | None = None
    if thinking_level is not None and str(thinking_level).strip():
        level_normalised = str(thinking_level).strip().lower()

    budget_set = thinking_budget is not None

    if is_3x:
        if level_normalised is not None:
            if level_normalised not in _THINKING_LEVEL_VALID:
                logger.warning(
                    "Invalid thinking_level %r for model %r — must be one of %s. "
                    "Ignoring.",
                    thinking_level,
                    model,
                    sorted(_THINKING_LEVEL_VALID),
                )
                return None
            return types.ThinkingConfig(thinking_level=level_normalised)
        if budget_set:
            logger.warning(
                "thinking_budget is ignored for Gemini 3.x model %r — "
                "use thinking_level (minimal|low|medium|high) instead.",
                model,
            )
        return None
    else:
        if budget_set:
            return types.ThinkingConfig(thinking_budget=int(thinking_budget))
        if level_normalised is not None:
            logger.warning(
                "thinking_level is ignored for non-3.x model %r — "
                "use thinking_budget (int tokens) instead.",
                model,
            )
        return None


def _has_speaker_turns(
    text: str,
    speaker1_name: str,
    speaker2_name: str,
) -> bool:
    """
    Return ``True`` iff *text* contains at least one valid speaker turn.

    A valid speaker turn is a stripped line that starts with
    ``"<SpeakerName>:"``.  This mirrors the boundary detection used by
    :func:`_split_dialogue_into_chunks`.

    Parameters
    ----------
    text : str
        The raw dialogue text returned by the LLM.
    speaker1_name : str
        Name of the first speaker.
    speaker2_name : str
        Name of the second speaker.

    Returns
    -------
    bool
        ``True`` when at least one line starts with a recognised speaker prefix.
    """
    prefix1 = f"{speaker1_name}:"
    prefix2 = f"{speaker2_name}:"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix1) or stripped.startswith(prefix2):
            return True
    return False


def _render_speaker_adjustments_block(
    speaker1_name: str,
    speaker2_name: str,
    speaker1_overlay: str | None,
    speaker2_overlay: str | None,
) -> str:
    """
    Render the per-episode speaker-adjustments block.

    Returns the empty string when both overlays are absent so that the
    surrounding prompt stays byte-identical to the no-overlay case.

    Parameters
    ----------
    speaker1_name, speaker2_name : str
        Host display names.
    speaker1_overlay, speaker2_overlay : str or None
        Episode-specific overlay text for each speaker.  ``None`` and the
        empty string both mean "no overlay".

    Returns
    -------
    str
        Either ``""`` or a block starting and ending with ``"\\n"`` that lists
        only the populated overlays.
    """
    overlays: list[tuple[str, str]] = []
    if speaker1_overlay:
        overlays.append((speaker1_name, speaker1_overlay))
    if speaker2_overlay:
        overlays.append((speaker2_name, speaker2_overlay))
    if not overlays:
        return ""
    bullets = "\n".join(f"- {name}: {text}" for name, text in overlays)
    return f"\nEpisode-specific adjustments:\n{bullets}\n"


def _render_style_block(
    preset_fragment: str | None,
    style_text: str | None,
) -> str:
    """
    Render the ``Stylistic guidance:`` sub-block inside ``Instructions:``.

    Returns the empty string when neither input is set, preserving prompt
    byte-identity for runs that don't use the style flags.

    Parameters
    ----------
    preset_fragment : str or None
        Resolved prompt fragment for the selected preset, if any.
    style_text : str or None
        Free-text style guidance, if any.

    Returns
    -------
    str
        Either ``""`` or a block starting with ``"\\n"`` (to leave a blank
        line before the header) and ending with ``"\\n\\n"`` (to separate
        from the bullet that follows).
    """
    parts: list[str] = []
    if preset_fragment:
        parts.append(preset_fragment.strip())
    if style_text:
        parts.append(style_text.strip())
    if not parts:
        return ""
    body = "\n\n".join(parts)
    return f"\nStylistic guidance:\n{body}\n\n"


def _render_angle_line(angle: str | None) -> str:
    """
    Render the ``- Episode angle:`` bullet inside ``Instructions:``.

    Returns the empty string when *angle* is unset so the existing bullet
    list stays untouched.

    Parameters
    ----------
    angle : str or None
        Free-text episode angle (already truncated).

    Returns
    -------
    str
        Either ``""`` or a single bullet line ending with ``"\\n"``.
    """
    if not angle:
        return ""
    return (
        f"- Episode angle: {angle.strip()}. Weave this through the "
        "conversation; don't just mention it once.\n"
    )


def _render_research_directive(research_notes: str) -> str:
    """
    Render the research-integration directive bullet inside ``Instructions:``.

    Returns the empty string when *research_notes* is blank so the prompt
    stays byte-identical to runs without research.

    Parameters
    ----------
    research_notes : str
        The raw research notes string (may be empty or whitespace-only).

    Returns
    -------
    str
        Either ``""`` or a single bullet line ending with ``"\\n"`` instructing
        the model to incorporate the research findings substantively.
    """
    if not research_notes.strip():
        return ""
    return (
        "- The research notes above contain key findings that the dialogue MUST "
        "incorporate substantively — cover the main findings, bring the outside "
        "facts into the conversation; a single passing mention is not enough.\n"
    )


def _build_prompt(
    articles: list,
    speaker1_name: str,
    speaker2_name: str,
    speaker1_personality: str = "",
    speaker2_personality: str = "",
    min_minutes: float = 6.0,
    target_minutes: float = 8.0,
    max_minutes: float = 14.0,
    words_per_minute: int = 150,
    language: str = "French",
    audio_tags: bool = False,
    research_notes: str = "",
    *,
    preset: str | None = None,
    style_text: str | None = None,
    speaker1_overlay: str | None = None,
    speaker2_overlay: str | None = None,
    angle: str | None = None,
) -> str:
    """
    Build the full LLM prompt from the article list, speaker names, and personalities.

    Parameters
    ----------
    articles : list
        A list of article-like objects with ``title``, ``url``, and
        ``full_text`` (or ``summary``) attributes.
    speaker1_name : str
        Display name of the first podcast host.
    speaker2_name : str
        Display name of the second podcast host.
    speaker1_personality : str, optional
        Short description of the first host's personality and speaking style.
    speaker2_personality : str, optional
        Short description of the second host's personality and speaking style.
    min_minutes : float, optional
        Lower bound on the target episode length, in minutes.  Translated
        to a word count using *words_per_minute* and presented to the LLM
        as a hard minimum.
    target_minutes : float, optional
        Desired episode length, in minutes.
    max_minutes : float, optional
        Upper bound on the target episode length, in minutes.  Presented
        to the LLM as a soft maximum.
    words_per_minute : int, optional
        Conversational pace used to translate minutes to word counts in
        the prompt, by default 150.
    language : str, optional
        Language for the generated dialogue, by default ``"French"``.
    audio_tags : bool, optional
        When ``True``, instruct the LLM to use bracketed audio tags
        (e.g. ``"[enthusiasm]"``) rather than parenthetical cues.
    research_notes : str, optional
        Complementary research notes (markdown bullet points) to inject into
        the prompt under a "Complementary research" section.  Empty string
        disables the injection.
    preset : str or None, keyword-only, optional
        Style preset name resolved via :func:`validate_preset`.  Adds a
        ``Stylistic guidance:`` sub-block inside ``Instructions:`` when set.
    style_text : str or None, keyword-only, optional
        Free-text style guidance.  Truncated to 500 chars.  Renders inside
        the same ``Stylistic guidance:`` sub-block as *preset*; both may be
        set, in which case the preset fragment comes first.
    speaker1_overlay, speaker2_overlay : str or None, keyword-only, optional
        Per-speaker style overlays.  Truncated to 500 chars.  Render in a
        dedicated ``Episode-specific adjustments:`` block placed between
        ``Host personalities:`` and ``Instructions:``.  Never mutate the
        baseline personality strings.
    angle : str or None, keyword-only, optional
        Free-text episode angle.  Truncated to 500 chars.  Renders as a
        ``- Episode angle: …`` bullet inside ``Instructions:``.

    Returns
    -------
    str
        The formatted prompt string ready to send to Gemini.
    """
    article_blocks: list[str] = []
    for i, article in enumerate(articles, start=1):
        title = getattr(article, "title", f"Article {i}")
        if getattr(article, "kind", "url") == "search":
            block = f"[{i}] Topic of investigation: {title}\n(See research notes below for findings.)"
        else:
            text = getattr(article, "full_text", "") or getattr(article, "summary", "")
            url = getattr(article, "url", "")
            block = f"[{i}] {title}\nURL: {url}\n{text}"
        article_blocks.append(block)

    articles_text = "\n\n".join(article_blocks)

    delivery_cues_guidance = (
        _AUDIO_TAGS_GUIDANCE if audio_tags else _PAREN_CUES_GUIDANCE
    )
    example_template = _EXAMPLE_AUDIO_TAGS if audio_tags else _EXAMPLE_PAREN_CUES
    example_dialogue = example_template.format(
        speaker1=speaker1_name, speaker2=speaker2_name
    )

    research_section = (
        _RESEARCH_SECTION_TEMPLATE.format(notes=research_notes.strip())
        if research_notes.strip()
        else ""
    )

    preset_fragment = validate_preset(preset)
    style_text_resolved = truncate_with_warning(style_text, "style")
    speaker1_overlay_resolved = truncate_with_warning(
        speaker1_overlay, "speaker1-style"
    )
    speaker2_overlay_resolved = truncate_with_warning(
        speaker2_overlay, "speaker2-style"
    )
    angle_resolved = truncate_with_warning(angle, "angle")

    speaker_adjustments_block = _render_speaker_adjustments_block(
        speaker1_name,
        speaker2_name,
        speaker1_overlay_resolved,
        speaker2_overlay_resolved,
    )
    style_block = _render_style_block(preset_fragment, style_text_resolved)
    angle_line = _render_angle_line(angle_resolved)
    research_directive_line = _render_research_directive(research_notes)

    min_words = max(1, round(min_minutes * words_per_minute))
    target_words = max(min_words, round(target_minutes * words_per_minute))
    max_words = max(target_words, round(max_minutes * words_per_minute))

    return _SYSTEM_PROMPT_TEMPLATE.format(
        speaker1=speaker1_name,
        speaker2=speaker2_name,
        speaker1_personality=speaker1_personality or "enthusiastic and curious",
        speaker2_personality=speaker2_personality or "analytical and thoughtful",
        min_minutes=min_minutes,
        target_minutes=target_minutes,
        max_minutes=max_minutes,
        min_words=min_words,
        target_words=target_words,
        max_words=max_words,
        wpm=words_per_minute,
        language=language,
        delivery_cues_guidance=delivery_cues_guidance,
        example_dialogue=example_dialogue,
        research_section=research_section,
        articles=articles_text,
        speaker_adjustments_block=speaker_adjustments_block,
        style_block=style_block,
        angle_line=angle_line,
        research_directive_line=research_directive_line,
    )


def _split_dialogue_into_chunks(
    dialogue_text: str,
    speaker1_name: str,
    speaker2_name: str,
    max_bytes: int = _MAX_CHUNK_BYTES,
) -> list[DialogueChunk]:
    """
    Split a raw dialogue string into byte-bounded chunks.

    Chunks are split only at speaker-turn boundaries (lines beginning with
    a speaker name followed by a colon).  A turn is never split mid-line.

    Parameters
    ----------
    dialogue_text : str
        The raw multi-line dialogue produced by the LLM.
    speaker1_name : str
        Name of the first speaker, used to detect turn boundaries.
    speaker2_name : str
        Name of the second speaker, used to detect turn boundaries.
    max_bytes : int, optional
        Maximum UTF-8 byte size per chunk, by default 3000.

    Returns
    -------
    list[DialogueChunk]
        Ordered list of dialogue chunks, each within the byte limit.
    """
    speaker_prefixes = (
        f"{speaker1_name}:",
        f"{speaker2_name}:",
    )

    # Collect turn lines (lines that start a new speaker turn) and their
    # accumulated content (a turn may theoretically span multiple lines,
    # though the prompt requests one line per turn).
    turns: list[str] = []
    current_lines: list[str] = []

    for line in dialogue_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_new_turn = any(stripped.startswith(prefix) for prefix in speaker_prefixes)
        if is_new_turn and current_lines:
            turns.append("\n".join(current_lines))
            current_lines = [stripped]
        else:
            current_lines.append(stripped)

    if current_lines:
        turns.append("\n".join(current_lines))

    logger.debug("Dialogue split into %d speaker turns.", len(turns))

    # Pack turns into byte-bounded chunks
    chunks: list[DialogueChunk] = []
    current_chunk_lines: list[str] = []
    current_size = 0

    for turn in turns:
        turn_bytes = len(turn.encode("utf-8")) + 1  # +1 for newline separator

        if current_chunk_lines and current_size + turn_bytes > max_bytes:
            # Flush current chunk
            chunk_text = "\n".join(current_chunk_lines)
            chunks.append(DialogueChunk(text=chunk_text, index=len(chunks)))
            logger.debug(
                "Chunk %d created: %d UTF-8 bytes.",
                len(chunks) - 1,
                len(chunk_text.encode("utf-8")),
            )
            current_chunk_lines = [turn]
            current_size = turn_bytes
        else:
            current_chunk_lines.append(turn)
            current_size += turn_bytes

    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines)
        chunks.append(DialogueChunk(text=chunk_text, index=len(chunks)))
        logger.debug(
            "Chunk %d created: %d UTF-8 bytes.",
            len(chunks) - 1,
            len(chunk_text.encode("utf-8")),
        )

    logger.info("Dialogue split into %d chunk(s).", len(chunks))
    return chunks


@dataclass
class DialogueChunk:
    """
    A byte-size-bounded segment of podcast dialogue.

    Attributes
    ----------
    text : str
        The raw dialogue text for this chunk.
    index : int
        Zero-based position of this chunk in the full dialogue sequence.
    """

    text: str
    index: int


def generate_dialogue(
    articles: list,
    gemini_cfg: dict,
    speaker1_name: str,
    speaker2_name: str,
    token_tracker: TokenTracker | None = None,
    progress: Progress | None = None,
    task_id: Any = None,
    research_notes: str = "",
) -> list[DialogueChunk]:
    """
    Generate a two-host podcast dialogue from a list of articles.

    Sends all articles to Gemini and parses the resulting dialogue into
    byte-bounded :class:`DialogueChunk` objects suitable for TTS.

    Parameters
    ----------
    articles : list
        A list of article-like objects with ``title``, ``url``, and
        ``full_text`` / ``summary`` attributes.
    gemini_cfg : dict
        Resolved Gemini configuration section from the YAML config, containing
        at minimum ``api_key`` and ``text_model`` keys.  Optional keys:
        ``speaker1.personality``, ``speaker2.personality``, ``language``
        (default ``"French"``), and ``dialogue`` sub-section.
    speaker1_name : str
        Display name of the first podcast host (used in the prompt and as a
        speaker-turn boundary marker).
    speaker2_name : str
        Display name of the second podcast host.
    token_tracker : TokenTracker or None, optional
        When provided, records prompt and candidates token counts for this call.
    progress : rich.progress.Progress or None, optional
        A rich :class:`~rich.progress.Progress` instance for displaying a
        spinner.  When provided, ``task_id`` must also be supplied and will be
        marked complete after the API call returns.
    task_id : Any, optional
        Task identifier returned by ``progress.add_task()``.
    research_notes : str, optional
        Complementary research notes to inject into the prompt.  When empty,
        no research section is added (no-research path).

    Returns
    -------
    list[DialogueChunk]
        Ordered list of dialogue chunks ready for TTS processing.

    Raises
    ------
    RuntimeError
        If the Gemini API returns an empty or missing response text.

    Examples
    --------
    >>> chunks = generate_dialogue(articles, gemini_cfg, "Alex", "Jordan")
    >>> len(chunks) > 0
    True
    """
    speaker1_personality = gemini_cfg.get("speaker1", {}).get("personality", "")
    speaker2_personality = gemini_cfg.get("speaker2", {}).get("personality", "")
    language = gemini_cfg.get("language", "French")

    style_cfg = gemini_cfg.get("style", {}) or {}
    preset = style_cfg.get("preset")
    style_text = style_cfg.get("text")
    angle = style_cfg.get("angle")
    speaker1_overlay = gemini_cfg.get("speaker1", {}).get("style_overlay")
    speaker2_overlay = gemini_cfg.get("speaker2", {}).get("style_overlay")

    dialogue_cfg = gemini_cfg.get("dialogue", {})
    target_minutes = float(dialogue_cfg.get("target_duration_minutes", 8.0))
    wpm = int(dialogue_cfg.get("words_per_minute", 150))
    min_minutes = float(
        dialogue_cfg.get("min_duration_minutes", round(target_minutes * 0.7, 1))
    )
    max_minutes = float(
        dialogue_cfg.get("max_duration_minutes", round(target_minutes * 1.5, 1))
    )

    thinking_level = dialogue_cfg.get("thinking_level")
    thinking_budget = dialogue_cfg.get("thinking_budget")
    thinking_cfg = _build_thinking_config(
        gemini_cfg["text_model"], thinking_level, thinking_budget
    )

    audio_tags = _audio_tags_enabled(gemini_cfg)
    if audio_tags:
        logger.info("Audio tags enabled for dialogue prompt (tts_model=%s).", gemini_cfg.get("tts_model"))

    prompt = _build_prompt(
        articles,
        speaker1_name,
        speaker2_name,
        speaker1_personality=speaker1_personality,
        speaker2_personality=speaker2_personality,
        min_minutes=min_minutes,
        target_minutes=target_minutes,
        max_minutes=max_minutes,
        words_per_minute=wpm,
        language=language,
        audio_tags=audio_tags,
        research_notes=research_notes,
        preset=preset,
        style_text=style_text,
        speaker1_overlay=speaker1_overlay,
        speaker2_overlay=speaker2_overlay,
        angle=angle,
    )

    logger.info(
        "Sending %d article(s) to Gemini model '%s'%s.",
        len(articles),
        gemini_cfg["text_model"],
        " with research notes" if research_notes.strip() else "",
    )
    logger.debug("Dialogue prompt (%d chars):\n%s", len(prompt), prompt)

    client = genai.Client(api_key=gemini_cfg["api_key"])
    service_tier = gemini_cfg.get("service_tier")

    config_kwargs: dict[str, Any] = {"max_output_tokens": 8192}
    if service_tier:
        config_kwargs["http_options"] = types.HttpOptions(
            headers={"x-goog-api-service-tier": service_tier},
        )
    if thinking_cfg is not None:
        config_kwargs["thinking_config"] = thinking_cfg

    @gemini_retry
    def _call_api():
        return client.models.generate_content(
            model=gemini_cfg["text_model"],
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )

    dialogue_text: str = ""
    ok = False
    for attempt in range(1, _DIALOGUE_MAX_ATTEMPTS + 1):
        response = _call_api()

        if token_tracker is not None:
            token_tracker.record_usage(gemini_cfg["text_model"], response.usage_metadata)

        dialogue_text = response.text or ""
        ok = bool(dialogue_text) and _has_speaker_turns(
            dialogue_text, speaker1_name, speaker2_name
        )
        if ok:
            break
        snippet = dialogue_text[:200].replace("\n", "\\n")
        logger.warning(
            "Attempt %d/%d: dialogue has no speaker turns (or is empty). "
            "Snippet: %r. Retrying.",
            attempt,
            _DIALOGUE_MAX_ATTEMPTS,
            snippet,
        )

    if not ok:
        snippet = dialogue_text[:200].replace("\n", "\\n")
        raise RuntimeError(
            f"Gemini returned no properly-formatted dialogue (no speaker turns) "
            f"after {_DIALOGUE_MAX_ATTEMPTS} attempt(s). "
            f"Last response snippet: {snippet!r}"
        )

    if progress is not None and task_id is not None:
        progress.advance(task_id)
        if token_tracker is not None:
            progress.update(
                task_id,
                description=f"[cyan]Dialogue[/cyan] — {token_tracker.live_line()}",
            )

    logger.info("Received dialogue of %d chars from Gemini.", len(dialogue_text))

    return _split_dialogue_into_chunks(dialogue_text, speaker1_name, speaker2_name)
