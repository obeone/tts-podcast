"""
Two-stage breadth-first link following for the tts-podcast pipeline.

After the seed documents have been scraped or loaded, this module discovers
and traverses interesting links found *inside* them, up to an optional depth.
The pages it keeps enrich BOTH the research stage and the dialogue, so it runs
before either of them in ``cli.py``.

Selection is two-stage:

1. **Heuristic, before fetch** — :func:`tts_podcast.link_extractor.is_followable_link`
   is a cheap URL filter dropping obvious junk (non-http schemes, anchors,
   asset extensions, ad/social/tracker hosts, auth/commerce paths) while
   keeping every real-content category.
2. **LLM, after fetch** — one Gemini call (:func:`_judge_sources`) classifies
   each fetched page's *actual content* against the seed topic into ``"core"``
   (strongly on-topic), ``"supporting"`` (useful complementary source), or
   ``"irrelevant"`` (dropped).

The traversal is a level-by-level BFS: from the current frontier, gather and
heuristically filter candidate links, fetch them, judge them, keep the
non-irrelevant ones (recording the verdict on each :class:`~tts_podcast.models.Source`),
and recurse into those kept pages until ``depth`` levels have been processed.
A ``seen`` set of normalized URLs (seeded with the seed sources' own URLs)
guarantees cycle safety and avoids re-fetching pages.

The whole stage is best-effort: every failure (scrape error, malformed API
response, JSON parse failure) is logged and degraded gracefully so a
``--follow-links`` run never aborts the pipeline.  When the relevance judge
cannot produce a verdict it fails *open* — keeping content as ``"supporting"``
rather than silently dropping it.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse, urlunparse

from google import genai
from google.genai import types

from tts_podcast.link_extractor import extract_links_from_text, is_followable_link
from tts_podcast.retry import gemini_retry
from tts_podcast.style_presets import truncate_with_warning
from tts_podcast.web_scraper import scrape_urls

if TYPE_CHECKING:
    from rich.progress import Progress

    from tts_podcast.models import Source
    from tts_podcast.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

# Default model when neither follow.model nor gemini.text_model is configured.
_DEFAULT_FOLLOW_MODEL_FALLBACK = "gemini-2.5-flash"

# How many characters of each fetched body are shown to the relevance judge.
# Enough to characterise the content without bloating the prompt.
_JUDGE_BODY_CHARS = 2000

# The three relevance verdicts the judge may return.
_VALID_VERDICTS: frozenset[str] = frozenset({"core", "supporting", "irrelevant"})

_JUDGE_PROMPT = """\
You are curating sources for a podcast about the topic below. For EACH fetched \
page, judge how its real content relates to that topic and classify it as one \
of exactly three labels:
- "core": strongly on-topic; the content could directly join the main subject \
of the episode.
- "supporting": useful complementary source that adds context, background, or \
a related angle.
- "irrelevant": off-topic, navigational, promotional, or otherwise not worth \
including.

Main topic:
{topic}

Fetched pages:
{pages}

Return ONLY a JSON ARRAY with one object per fetched page, each of the form \
{{"url": "...", "label": "core|supporting|irrelevant"}}. Echo back the exact \
URL given for that page (verbatim, as it appears above) and choose exactly one \
label. Example: [{{"url": "https://a.example/x", "label": "core"}}, \
{{"url": "https://b.example/y", "label": "irrelevant"}}].
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_url(url: str) -> str | None:
    """
    Normalize a URL for cycle/dedup detection, or ``None`` when not followable.

    Normalization lowercases the host, strips the fragment, and removes a
    single trailing slash from the path so that ``https://A.com/x`` and
    ``https://a.com/x/#frag`` collapse to the same key.  Only ``http``/
    ``https`` URLs are normalizable; anything else (and any parse failure)
    returns ``None`` so the caller treats it as non-followable.

    Parameters
    ----------
    url : str
        The URL to normalize.

    Returns
    -------
    str or None
        The normalized URL string, or ``None`` when the URL is not an
        ``http``/``https`` URL or cannot be parsed.
    """
    try:
        parsed = urlparse(url)
    except Exception:  # noqa: BLE001 — malformed URL: not followable.
        return None

    if parsed.scheme not in ("http", "https"):
        return None

    host = parsed.netloc.lower()
    path = parsed.path
    # Strip a single trailing slash so "/x" and "/x/" are the same node, but
    # keep a bare-root "/" intact rather than producing an empty path.
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    # Drop the fragment; keep scheme/query/params so distinct pages stay distinct.
    return urlunparse((parsed.scheme.lower(), host, path, parsed.params, parsed.query, ""))


def _build_topic(seed_sources: list[Source]) -> str:
    """
    Build a concise topic-context string from the seed sources.

    Combines each seed's title and summary (full text is deliberately avoided
    here — the summary is enough to anchor the topic and keeps the judge prompt
    small).  The aggregate is truncated to keep the relevance call focused.

    Parameters
    ----------
    seed_sources : list[Source]
        The seed (already-scraped) sources that define the episode topic.

    Returns
    -------
    str
        A multi-line ``title — summary`` block, truncated to a safe length.
    """
    parts: list[str] = []
    for src in seed_sources:
        title = (src.title or "").strip()
        summary = (src.summary or "").strip()
        if title and summary:
            parts.append(f"{title} — {summary}")
        elif title:
            parts.append(title)
        elif summary:
            parts.append(summary)
    topic = "\n".join(parts)
    # Reuse the shared truncation helper for a consistent cap + warning.
    return truncate_with_warning(topic, "follow-links topic", cap=2000) or ""


def _gather_candidates(
    frontier: list[Source],
    seen: set[str],
    max_links: int,
) -> list[str]:
    """
    Collect followable, not-yet-seen candidate URLs from a frontier.

    For each source, builds a unified candidate sequence from two sources:

    1. ``Source.links`` — structured outbound URLs captured at scrape time via
       trafilatura's ``include_links=True`` markdown pass.  HTML pages carry
       their hrefs here because plain-text extraction drops all hyperlinks.
    2. ``extract_links_from_text(full_text or summary)`` — bare URLs found in
       the body text, which covers plain-text/markdown/PDF sources whose
       ``links`` list is empty.

    Both lists are concatenated (structured links first so they take priority
    within the per-level cap), then the stage-1 heuristic is applied and
    dedups are tracked via the shared ``seen`` set of normalized URLs.
    Newly accepted URLs are immediately added to ``seen`` so the same target
    is never queued twice, and the level is capped at *max_links*.

    Parameters
    ----------
    frontier : list[Source]
        Sources whose bodies are scanned for outgoing links.
    seen : set[str]
        Normalized URLs already queued or fetched; mutated in place.
    max_links : int
        Maximum number of candidates to return for this level.

    Returns
    -------
    list[str]
        Original (non-normalized) candidate URLs to fetch, at most
        *max_links* of them, in first-seen order.
    """
    candidates: list[str] = []
    for src in frontier:
        # Combine structured links captured at scrape time (Source.links, populated
        # by trafilatura's include_links markdown pass) with bare URLs extracted from
        # the plain text body.  HTML sources carry their hrefs in Source.links because
        # plain-text extraction drops all hyperlinks — without this union the follower
        # would find 0 candidates for normal HTML pages (e.g. Wikipedia).
        # Plain-text/markdown/PDF sources have links=[] but may carry bare URLs in
        # full_text that extract_links_from_text finds, so both paths keep working.
        seq = list(getattr(src, "links", []) or []) + extract_links_from_text(
            src.full_text or src.summary or ""
        )
        for url in seq:
            if len(candidates) >= max_links:
                return candidates
            if not is_followable_link(url):
                continue
            norm = _normalize_url(url)
            if norm is None or norm in seen:
                continue
            seen.add(norm)
            candidates.append(url)
    return candidates


def _judge_sources(
    topic: str,
    fetched_sources: list[Source],
    gemini_cfg: dict,
    token_tracker: TokenTracker | None,
) -> dict[str, str]:
    """
    Classify each fetched page's content against the topic via one Gemini call.

    Mirrors the research module's Gemini-call pattern (client construction,
    ``@gemini_retry`` wrapper, service-tier header, token tracking) but uses
    structured JSON output.  The response schema is a STABLE ARRAY of
    ``{"url": ..., "label": ...}`` objects (not URL-keyed properties): a
    per-call schema whose property *names* are the fetched URLs is a dynamic
    schema the Gemini API may reject or ignore, which would silently turn the
    judge into a no-op.  The array is parsed back into a ``{url: label}`` map.

    The call fails *open*: on any error or unparseable response, every fetched
    URL defaults to ``"supporting"`` so useful content is kept rather than
    silently dropped.  Unknown/missing labels are likewise coerced to
    ``"supporting"``, and any fetched URL the model omitted from the array
    defaults to ``"supporting"`` as well.

    Parameters
    ----------
    topic : str
        The seed topic context produced by :func:`_build_topic`.
    fetched_sources : list[Source]
        Successfully fetched pages to classify.
    gemini_cfg : dict
        Resolved Gemini configuration.  Uses ``api_key``, ``service_tier``,
        and the follow model resolved from ``follow.model`` →
        ``text_model`` → :data:`_DEFAULT_FOLLOW_MODEL_FALLBACK`.
    token_tracker : TokenTracker or None
        When provided, records token usage for the judging call.

    Returns
    -------
    dict[str, str]
        Mapping from each fetched source URL to one of ``"core"``,
        ``"supporting"``, or ``"irrelevant"``.  Every input URL is present.
    """
    # Fail-open default: keep everything as "supporting".
    default_verdicts = {src.url: "supporting" for src in fetched_sources}
    if not fetched_sources:
        return {}

    follow_cfg = gemini_cfg.get("follow", {}) or {}
    model = (
        follow_cfg.get("model")
        or gemini_cfg.get("text_model")
        or _DEFAULT_FOLLOW_MODEL_FALLBACK
    )
    api_key = gemini_cfg.get("api_key")
    service_tier = gemini_cfg.get("service_tier") or None

    if not api_key:
        logger.warning(
            "No Gemini api_key in config; keeping all %d followed page(s) as 'supporting'.",
            len(fetched_sources),
        )
        return default_verdicts

    # Build the per-page block: a stable URL header + a truncated body so the
    # judge sees the real content without an oversized prompt.
    page_blocks: list[str] = []
    for i, src in enumerate(fetched_sources, start=1):
        body = (src.full_text or src.summary or "")[:_JUDGE_BODY_CHARS]
        page_blocks.append(f"[{i}] URL: {src.url}\nTitle: {src.title}\n{body}")
    prompt = _JUDGE_PROMPT.format(topic=topic, pages="\n\n".join(page_blocks))

    try:
        client = genai.Client(api_key=api_key)

        config_kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
            # Structured output: a STABLE array of {url, label} objects. The
            # schema never embeds the per-call URLs as property names, so the
            # API can validate it the same way on every request (a URL-keyed
            # object schema is dynamic and may be rejected/ignored).
            "response_schema": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "url": {"type": "STRING"},
                        "label": {
                            "type": "STRING",
                            "enum": ["core", "supporting", "irrelevant"],
                        },
                    },
                    "required": ["url", "label"],
                },
            },
        }
        if service_tier:
            config_kwargs["http_options"] = types.HttpOptions(
                headers={"x-goog-api-service-tier": service_tier},
            )

        @gemini_retry
        def _call_api():
            return client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )

        response = _call_api()

        if token_tracker is not None:
            token_tracker.record_usage(model, response.usage_metadata)

        raw = (response.text or "").strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError(f"expected a JSON array, got {type(parsed).__name__}")
    except Exception as exc:  # noqa: BLE001 — fail open on any judging failure.
        logger.warning(
            "Relevance judging failed (%s); keeping all %d page(s) as 'supporting'.",
            exc,
            len(fetched_sources),
        )
        return default_verdicts

    # Flatten the model's array into a {url: label} map, keyed by the exact URL
    # string the model echoed. Stray items (non-dict, missing url) are ignored;
    # unknown labels collapse to "supporting".
    by_url: dict[str, str] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not isinstance(url, str):
            continue
        label = item.get("label")
        if isinstance(label, str) and label.lower() in _VALID_VERDICTS:
            by_url[url] = label.lower()
        else:
            by_url[url] = "supporting"

    # Match each fetched URL to the model's answer: exact match wins; any URL
    # the model omitted (no exact match) defaults to "supporting".
    verdicts: dict[str, str] = {}
    for src in fetched_sources:
        verdicts[src.url] = by_url.get(src.url, "supporting")
    return verdicts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def follow_links(
    seed_sources: list[Source],
    *,
    depth: int,
    gemini_cfg: dict,
    scrape_timeout: int,
    user_agent: str,
    cloak_fallback: bool,
    token_tracker: TokenTracker | None = None,
    max_links_per_level: int = 5,
    max_links_total: int = 20,
    progress: Progress | None = None,
    task_id: Any = None,
) -> list[Source]:
    """
    Discover and traverse interesting links from *seed_sources* via two-stage BFS.

    For each level up to *depth*: gather followable, not-yet-seen links from the
    current frontier (stage-1 heuristic), fetch them, judge their content
    against the seed topic (stage-2 LLM), drop the ``"irrelevant"`` ones, record
    the verdict on the rest, and recurse into them.  A ``seen`` set of
    normalized URLs (seeded with the seeds' own URLs) guarantees cycle safety
    and avoids re-fetching.

    Never raises: every per-level failure is logged and the traversal degrades
    gracefully (returning whatever was kept so far).

    Parameters
    ----------
    seed_sources : list[Source]
        The already-scraped seed sources to start from.
    depth : int
        Number of link-following hops.  ``<= 0`` returns an empty list without
        any work.
    gemini_cfg : dict
        Resolved Gemini configuration, threaded to :func:`_judge_sources`.
    scrape_timeout : int
        HTTP timeout (seconds) forwarded to
        :func:`tts_podcast.web_scraper.scrape_urls`.
    user_agent : str
        ``User-Agent`` header forwarded to the scraper.
    cloak_fallback : bool
        Whether the scraper may retry blocked pages via the CloakBrowser
        stealth fallback.
    token_tracker : TokenTracker or None, optional
        When provided, records token usage for every judging call.
    max_links_per_level : int, keyword-only, optional
        Maximum number of links fetched per BFS level, by default 5.
    max_links_total : int, keyword-only, optional
        Global budget on the cumulative number of links fetched across ALL
        levels, by default 20.  Each level's candidate list is truncated so
        the running total never exceeds this cap; once the budget is
        exhausted the traversal stops.  This bounds total cost independently
        of *depth* × *max_links_per_level*.
    progress : rich.progress.Progress or None, optional
        Progress instance whose task is advanced once per completed level.
    task_id : Any, optional
        Task identifier returned by ``progress.add_task()``.

    Returns
    -------
    list[Source]
        The kept (``"core"`` + ``"supporting"``) sources across all levels,
        in traversal order, each with ``.relevance`` set.  Seed sources are
        never included.
    """
    if depth <= 0:
        return []
    if not seed_sources:
        return []

    # Seed the seen set with the seeds' own normalized URLs so a followed page
    # that links straight back to a seed is never re-fetched.
    seen: set[str] = set()
    for src in seed_sources:
        norm = _normalize_url(src.url)
        if norm is not None:
            seen.add(norm)

    topic = _build_topic(seed_sources)

    kept: list[Source] = []
    frontier = seed_sources
    # Cumulative count of links fetched across all levels, capped at
    # max_links_total so cost stays bounded regardless of depth.
    fetched_total = 0

    for level in range(depth):
        remaining = max_links_total - fetched_total
        if remaining <= 0:
            logger.info("global follow budget %d reached", max_links_total)
            break

        candidates = _gather_candidates(frontier, seen, max_links_per_level)
        if not candidates:
            logger.info(
                "Link-following level %d: no followable candidates; stopping.",
                level + 1,
            )
            break

        # Truncate this level's candidates so the running total never exceeds
        # the global budget.
        if len(candidates) > remaining:
            logger.info("global follow budget %d reached", max_links_total)
            candidates = candidates[:remaining]
        fetched_total += len(candidates)

        logger.info(
            "Link-following level %d: fetching %d candidate link(s).",
            level + 1,
            len(candidates),
        )

        try:
            fetched = scrape_urls(
                candidates,
                timeout=scrape_timeout,
                user_agent=user_agent,
                use_cloak_fallback=cloak_fallback,
            )
        except Exception as exc:  # noqa: BLE001 — never let a scrape abort the run.
            logger.warning("Link-following level %d scrape failed: %s", level + 1, exc)
            break

        ok = [s for s in fetched if s.scraped_ok]
        if not ok:
            logger.info(
                "Link-following level %d: none of %d link(s) yielded content.",
                level + 1,
                len(candidates),
            )
            # Still advance progress for this level before stopping.
            _advance(progress, task_id, token_tracker)
            break

        verdicts = _judge_sources(topic, ok, gemini_cfg, token_tracker)

        next_frontier: list[Source] = []
        for src in ok:
            verdict = verdicts.get(src.url, "supporting")
            if verdict == "irrelevant":
                continue
            src.relevance = verdict
            kept.append(src)
            next_frontier.append(src)

        logger.info(
            "Link-following level %d: kept %d/%d fetched page(s).",
            level + 1,
            len(next_frontier),
            len(ok),
        )

        _advance(progress, task_id, token_tracker)

        if not next_frontier:
            break
        frontier = next_frontier

    logger.info("Link-following kept %d source(s) total.", len(kept))
    return kept


def _advance(
    progress: Progress | None,
    task_id: Any,
    token_tracker: TokenTracker | None,
) -> None:
    """
    Advance the progress task by one level and refresh its live cost line.

    Mirrors the research module's progress-update idiom so the follow stage
    shows the running token spend in its description.

    Parameters
    ----------
    progress : rich.progress.Progress or None
        Progress instance, or ``None`` when progress reporting is disabled.
    task_id : Any
        Task identifier returned by ``progress.add_task()``.
    token_tracker : TokenTracker or None
        When provided, its ``live_line()`` is appended to the description.
    """
    if progress is None or task_id is None:
        return
    progress.advance(task_id)
    if token_tracker is not None:
        progress.update(
            task_id,
            description=f"[cyan]Following links[/cyan] — {token_tracker.live_line()}",
        )
