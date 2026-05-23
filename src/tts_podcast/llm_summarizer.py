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

if TYPE_CHECKING:
    from rich.progress import Progress
    from tts_podcast.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

# Maximum UTF-8 byte size for a single dialogue chunk sent to TTS.
# Set to 3000 to leave ~600-800 bytes of headroom for the TTS preamble
# that tts_generator.py prepends before sending to the Gemini TTS API.
_MAX_CHUNK_BYTES = 3000


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
Complementary research (from Google Search grounding — use to enrich the \
discussion, weave naturally into the dialogue, and cite sources by URL when \
relevant facts come from these notes):
{notes}

"""

_SYSTEM_PROMPT_TEMPLATE = """\
You are a podcast script writer. Your job is to create an engaging, \
conversational podcast dialogue between two hosts: {speaker1} and {speaker2}.

Host personalities:
- {speaker1}: {speaker1_personality}
- {speaker2}: {speaker2_personality}

Instructions:
- The article(s) below are the central topic of the episode.
- Discuss them in depth: explain what they are about, why they matter, \
and explore implications or connections to broader trends.
- The entire dialogue MUST be written in {language}.
- The total dialogue must be at least {target_word_count} words.
- Keep the tone informative but lively — like two curious friends catching up on tech news.
- Reflect each host's personality in their speaking style and reactions.
{delivery_cues_guidance}
- Use shorter sentences for excitement, longer ones for analysis.
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


def _build_prompt(
    articles: list,
    speaker1_name: str,
    speaker2_name: str,
    speaker1_personality: str = "",
    speaker2_personality: str = "",
    target_word_count: int = 1200,
    language: str = "French",
    audio_tags: bool = False,
    research_notes: str = "",
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
    target_word_count : int, optional
        Minimum total word count for the generated dialogue, by default 1200.
    language : str, optional
        Language for the generated dialogue, by default ``"French"``.
    audio_tags : bool, optional
        When ``True``, instruct the LLM to use bracketed audio tags
        (e.g. ``"[enthusiasm]"``) rather than parenthetical cues.
    research_notes : str, optional
        Complementary research notes (markdown bullet points) to inject into
        the prompt under a "Complementary research" section.  Empty string
        disables the injection.

    Returns
    -------
    str
        The formatted prompt string ready to send to Gemini.
    """
    article_blocks: list[str] = []
    for i, article in enumerate(articles, start=1):
        text = getattr(article, "full_text", "") or getattr(article, "summary", "")
        title = getattr(article, "title", f"Article {i}")
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

    return _SYSTEM_PROMPT_TEMPLATE.format(
        speaker1=speaker1_name,
        speaker2=speaker2_name,
        speaker1_personality=speaker1_personality or "enthusiastic and curious",
        speaker2_personality=speaker2_personality or "analytical and thoughtful",
        target_word_count=target_word_count,
        language=language,
        delivery_cues_guidance=delivery_cues_guidance,
        example_dialogue=example_dialogue,
        research_section=research_section,
        articles=articles_text,
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

    dialogue_cfg = gemini_cfg.get("dialogue", {})
    target_word_count = dialogue_cfg.get("target_word_count", 1200)

    audio_tags = _audio_tags_enabled(gemini_cfg)
    if audio_tags:
        logger.info("Audio tags enabled for dialogue prompt (tts_model=%s).", gemini_cfg.get("tts_model"))

    prompt = _build_prompt(
        articles,
        speaker1_name,
        speaker2_name,
        speaker1_personality=speaker1_personality,
        speaker2_personality=speaker2_personality,
        target_word_count=target_word_count,
        language=language,
        audio_tags=audio_tags,
        research_notes=research_notes,
    )

    logger.info(
        "Sending %d article(s) to Gemini model '%s'%s.",
        len(articles),
        gemini_cfg["text_model"],
        " with research notes" if research_notes.strip() else "",
    )
    logger.debug("Prompt length: %d chars.", len(prompt))

    client = genai.Client(api_key=gemini_cfg["api_key"])
    service_tier = gemini_cfg.get("service_tier")

    config_kwargs: dict[str, Any] = {"max_output_tokens": 8192}
    if service_tier:
        config_kwargs["http_options"] = types.HttpOptions(
            headers={"x-goog-api-service-tier": service_tier},
        )

    @gemini_retry
    def _call_api():
        return client.models.generate_content(
            model=gemini_cfg["text_model"],
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )

    response = _call_api()

    if token_tracker is not None:
        token_tracker.record_usage(gemini_cfg["text_model"], response.usage_metadata)

    if progress is not None and task_id is not None:
        progress.advance(task_id)
        if token_tracker is not None:
            progress.update(
                task_id,
                description=f"[cyan]Dialogue[/cyan] — {token_tracker.live_line()}",
            )

    dialogue_text = response.text
    if not dialogue_text:
        raise RuntimeError("Gemini returned an empty dialogue response.")

    logger.info("Received dialogue of %d chars from Gemini.", len(dialogue_text))

    return _split_dialogue_into_chunks(dialogue_text, speaker1_name, speaker2_name)
