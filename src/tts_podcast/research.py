"""
Iterative Google Search-grounded research for the tts-podcast pipeline.

Runs one or more rounds of Gemini generation with the ``google_search`` tool
enabled.  Round 1 looks for complementary angles around the input articles;
each subsequent round receives the previous rounds' notes and is asked to
drill into the gaps that remain.

The resulting :class:`ResearchReport` aggregates per-round notes, the
queries Gemini actually issued, and any grounding citations.  The combined
notes string is intended to be fed back into
:func:`tts_podcast.llm_summarizer.generate_dialogue` via its
``research_notes`` keyword argument.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tts_podcast.llm_client import build_model_string, complete

from tts_podcast.style_presets import truncate_with_warning

if TYPE_CHECKING:
    from rich.progress import Progress
    from tts_podcast.models import Source
    from tts_podcast.settings import LlmSettings
    from tts_podcast.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

# Default model when the LLM settings carry no explicit research/text model.
_DEFAULT_RESEARCH_MODEL_FALLBACK = "gemini-2.5-flash"


_ROUND_1_PROMPT = """\
You are a research assistant for a podcast. Given the article(s) below, \
identify the most interesting complementary angles a curious listener would \
want to know — background context, recent developments, contradicting takes, \
technical depth the article skips, related work. Use Google Search \
aggressively. Cite every fact you add by URL. Produce at most 800 words of \
bullet-point notes in {language}.
{angle_block}
Articles:
{articles}
"""

_ROUND_1_SEARCH_PROMPT = """\
You are a research assistant for a podcast. The host wants to record an \
episode about the topic below. Build SUBSTANTIVE, COMPREHENSIVE coverage \
of the topic itself: background and context, key facts and figures, current \
state of affairs, notable recent developments, the main perspectives and \
debates, and concrete examples that could make the episode engaging. \
Use Google Search aggressively — search from multiple angles. Cite every \
fact you add by URL. Produce thorough bullet-point notes in {language}; \
aim for depth rather than brevity so the host has ample material for a \
full-length episode.
{angle_block}
Topic:
{articles}
"""

_ROUND_N_PROMPT = """\
You are a research assistant for a podcast. Below: the original article(s) \
plus the research notes from previous rounds. Identify gaps, uncertainties, \
and unanswered questions the previous rounds left open. Use Google Search to \
drill into those gaps specifically. Do NOT repeat facts already covered. \
Produce at most 600 words of new notes in {language}. Cite every fact by URL.

Articles:
{articles}

Previous research notes (rounds 1…{prev_round}):
{previous_notes}
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    """
    A single grounding citation extracted from Gemini's response metadata.

    Attributes
    ----------
    title : str
        Page title reported by Google Search.
    uri : str
        URL of the cited page.
    """

    title: str
    uri: str


@dataclass
class ResearchRound:
    """
    Notes and metadata produced by a single research round.

    Attributes
    ----------
    index : int
        Zero-based round index (round 1 has ``index == 0``).
    query_hint : str
        Short label describing what this round was asked to investigate.
    notes : str
        Markdown notes returned by Gemini.
    citations : list[Citation]
        Grounding citations extracted from the response metadata.
    raw_search_queries : list[str]
        Queries Gemini issued to Google Search during this round.
    """

    index: int
    query_hint: str
    notes: str
    citations: list[Citation] = field(default_factory=list)
    raw_search_queries: list[str] = field(default_factory=list)


@dataclass
class ResearchReport:
    """
    Aggregate output of :func:`conduct_research`.

    Attributes
    ----------
    rounds : list[ResearchRound]
        Per-round notes in execution order.  Empty list when zero rounds
        were requested or all rounds returned nothing.
    combined_notes : str
        All rounds' notes concatenated under per-round headers, suitable
        for injection into the dialogue prompt.
    """

    rounds: list[ResearchRound] = field(default_factory=list)
    combined_notes: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_articles(sources: list[Source]) -> str:
    """
    Build the ``Articles:`` block injected into every research prompt.

    Parameters
    ----------
    sources : list[Source]
        Scraped articles to summarise for the research model.

    Returns
    -------
    str
        Multi-line text where each article is preceded by its index, title,
        URL, and the available body text (full text preferred, summary as
        fallback).
    """
    blocks: list[str] = []
    for i, src in enumerate(sources, start=1):
        if getattr(src, "kind", "url") == "search":
            blocks.append(f"[{i}] Topic to investigate: {src.title}")
        else:
            body = src.full_text or src.summary or ""
            blocks.append(f"[{i}] {src.title}\nURL: {src.url}\n{body}")
    return "\n\n".join(blocks)


def _build_combined_notes(rounds: list[ResearchRound]) -> str:
    """
    Concatenate per-round notes into a single string with headers.

    Parameters
    ----------
    rounds : list[ResearchRound]
        Rounds whose ``notes`` should be combined.

    Returns
    -------
    str
        Markdown text where each round is prefixed by a level-3 header
        such as ``### Research round 1``.  Empty rounds are skipped.
    """
    parts: list[str] = []
    for r in rounds:
        notes = r.notes.strip()
        if not notes:
            continue
        parts.append(f"### Research round {r.index + 1}\n\n{notes}")
    return "\n\n".join(parts)


def _run_single_round(
    *,
    prompt: str,
    model_string: str,
    bare_model: str,
    query_hint: str,
    round_index: int,
    api_key: str | None,
    api_base: str | None,
    extra_headers: dict[str, str] | None,
    token_tracker: TokenTracker | None,
) -> ResearchRound:
    """
    Execute one grounded LLM call and capture the result.

    The call is routed through :func:`tts_podcast.llm_client.complete` with
    Google Search grounding enabled.  Grounding is Gemini-native: on non-Gemini
    providers the tool is silently omitted and the round runs on the model's own
    knowledge (no citations/queries).

    Parameters
    ----------
    prompt : str
        Fully formatted prompt for this round.
    model_string : str
        LiteLLM model string (``"<provider>/<model>"``) for the call.
    bare_model : str
        Bare model name used as the token-tracking / pricing key.
    query_hint : str
        Short hint stored in the returned :class:`ResearchRound` for logging.
    round_index : int
        Zero-based index of this round.
    api_key : str or None
        Provider API key (``None`` lets LiteLLM read the provider env var).
    api_base : str or None
        Optional provider base-URL override.
    extra_headers : dict[str, str] or None
        Passthrough HTTP headers (e.g. the Gemini service-tier header).
    token_tracker : TokenTracker or None
        When provided, records prompt and completion token usage.

    Returns
    -------
    ResearchRound
        Populated round (notes may be empty when the model returned
        nothing).
    """
    logger.debug(
        "Research round %d prompt (%d chars):\n%s",
        round_index + 1,
        len(prompt),
        prompt,
    )

    result = complete(
        model=model_string,
        prompt=prompt,
        api_key=api_key,
        api_base=api_base,
        extra_headers=extra_headers,
        enable_google_search=True,
    )

    if token_tracker is not None:
        token_tracker.record(bare_model, result.input_tokens, result.output_tokens)

    notes = (result.text or "").strip()
    if result.grounding is not None:
        citations = [
            Citation(title=title, uri=uri)
            for title, uri in result.grounding.citations
        ]
        queries = list(result.grounding.queries)
    else:
        citations = []
        queries = []

    if not notes:
        logger.warning(
            "Research round %d returned no notes (model=%s).",
            round_index + 1,
            model_string,
        )
    else:
        logger.info(
            "Research round %d produced %d chars of notes, %d citation(s), %d query(ies).",
            round_index + 1,
            len(notes),
            len(citations),
            len(queries),
        )

    return ResearchRound(
        index=round_index,
        query_hint=query_hint,
        notes=notes,
        citations=citations,
        raw_search_queries=queries,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def conduct_research(
    sources: list[Source],
    rounds: int,
    gemini_cfg: dict,
    llm_cfg: LlmSettings,
    token_tracker: TokenTracker | None = None,
    progress: Progress | None = None,
    task_id: Any = None,
    *,
    angle: str | None = None,
) -> ResearchReport:
    """
    Run *rounds* sequential research rounds using the LLM + Google Search.

    Round 1 looks for complementary angles; each subsequent round receives
    the previous rounds' notes and is asked to drill into the gaps.  Search
    grounding is Gemini-native — see :func:`_run_single_round`.

    Parameters
    ----------
    sources : list[Source]
        Scraped articles serving as the central topic.
    rounds : int
        Number of research rounds to perform.  ``0`` short-circuits and
        returns an empty :class:`ResearchReport` without any API call.
        Must be non-negative.
    gemini_cfg : dict
        Resolved configuration section, used here only for ``language``
        (default ``"French"``).  Model selection and auth come from *llm_cfg*.
    llm_cfg : LlmSettings
        Resolved provider-agnostic LLM settings.  The research model is
        ``research_model`` when set, else ``text_model``, else
        ``"gemini-2.5-flash"``.
    token_tracker : TokenTracker or None, optional
        When provided, records token usage for every research call.
    progress : rich.progress.Progress or None, optional
        Progress instance whose task is advanced once per completed round.
    task_id : Any, optional
        Task identifier returned by ``progress.add_task()``.
    angle : str or None, keyword-only, optional
        Free-text episode angle to prioritise during round 1.  Round N>=2
        inherits the angle implicitly through ``previous_notes`` and is NOT
        re-prompted, so that gap-analysis rounds stay neutral.  Truncated to
        500 chars.

    Returns
    -------
    ResearchReport
        Aggregate of per-round notes plus a single ``combined_notes``
        string ready for the dialogue prompt.

    Raises
    ------
    ValueError
        If *rounds* is negative.
    """
    if rounds < 0:
        raise ValueError(f"rounds must be non-negative, got {rounds}.")

    if rounds == 0:
        logger.info("Research disabled (rounds=0); skipping all API calls.")
        return ResearchReport(rounds=[], combined_notes="")

    if not sources:
        logger.warning("No sources provided to conduct_research; skipping.")
        return ResearchReport(rounds=[], combined_notes="")

    language = gemini_cfg.get("language", "French")
    bare_model = (
        llm_cfg.research_model
        or llm_cfg.text_model
        or _DEFAULT_RESEARCH_MODEL_FALLBACK
    )
    model_string = build_model_string(llm_cfg.provider, bare_model)
    api_key = llm_cfg.api_key
    api_base = llm_cfg.api_base
    extra_headers = llm_cfg.extra_headers

    articles_text = _format_articles(sources)
    completed_rounds: list[ResearchRound] = []

    angle_resolved = truncate_with_warning(angle, "angle")
    if angle_resolved:
        angle_block = (
            f"\nAngle to emphasize: {angle_resolved.strip()}. Prioritise "
            "sources and notes that illuminate this angle.\n"
        )
    else:
        angle_block = ""

    logger.info(
        "Conducting %d research round(s) with model '%s' on %d source(s).",
        rounds,
        model_string,
        len(sources),
    )

    for i in range(rounds):
        if i == 0:
            all_search = all(
                getattr(s, "kind", "url") == "search" for s in sources
            )
            if all_search:
                prompt = _ROUND_1_SEARCH_PROMPT.format(
                    language=language,
                    articles=articles_text,
                    angle_block=angle_block,
                )
                query_hint = "comprehensive topic coverage (search-only)"
            else:
                prompt = _ROUND_1_PROMPT.format(
                    language=language,
                    articles=articles_text,
                    angle_block=angle_block,
                )
                query_hint = "initial complementary angles"
        else:
            previous_notes = _build_combined_notes(completed_rounds) or "(no prior notes)"
            prompt = _ROUND_N_PROMPT.format(
                language=language,
                articles=articles_text,
                prev_round=i,
                previous_notes=previous_notes,
            )
            query_hint = f"gap analysis after round {i}"

        round_result = _run_single_round(
            prompt=prompt,
            model_string=model_string,
            bare_model=bare_model,
            query_hint=query_hint,
            round_index=i,
            api_key=api_key,
            api_base=api_base,
            extra_headers=extra_headers,
            token_tracker=token_tracker,
        )
        completed_rounds.append(round_result)

        if progress is not None and task_id is not None:
            progress.advance(task_id)
            if token_tracker is not None:
                progress.update(
                    task_id,
                    description=f"[cyan]Research[/cyan] — {token_tracker.live_line()}",
                )

    combined = _build_combined_notes(completed_rounds)
    return ResearchReport(rounds=completed_rounds, combined_notes=combined)
