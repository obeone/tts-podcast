"""
Shared data models for the tts-podcast tool.

Holds dataclasses used across multiple pipeline stages so that producer
modules (web scraper) and consumer modules (research, summariser, report)
can depend on a single, neutral location.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Source:
    """
    Represents a single input source for the podcast pipeline.

    Attributes
    ----------
    url : str
        The URL of the source article, a ``file://`` URI for local files,
        or a ``search://`` URI for web-search queries.
    title : str
        Title extracted from the page metadata, or a fallback derived from
        the URL when no title was found.
    summary : str
        Short blurb (first few sentences of ``full_text``); empty when the
        page could not be scraped.
    full_text : str
        Main article body extracted by the scraper, default ``""``.
    scraped_ok : bool
        ``True`` when scraping returned non-empty content, ``False``
        otherwise.
    kind : str
        Input kind — one of ``"url"`` (default, fetched from the web),
        ``"file"`` (read from a local file path), or ``"search"`` (a
        natural-language query to be investigated via web research).
    relevance : str or None
        Content-relevance verdict assigned by the link-following stage
        (:mod:`tts_podcast.link_follower`).  ``"core"`` for pages strongly
        on-topic with the seed subject, ``"supporting"`` for useful
        complementary sources, and ``None`` for primary/seed inputs that
        were never judged (URLs, files, search queries, and the seeds
        themselves).  Stored on the source so the verdict can be reused
        across the research and dialogue stages.
    links : list[str]
        Outbound hyperlink targets found in the article body (absolute
        http(s) URLs), preserved at scrape time.  Populated by
        :func:`~tts_podcast.web_scraper._extract_from_html` (and the HTML
        path in :mod:`tts_podcast.local_loader`) via trafilatura's
        ``include_links=True`` markdown extraction, because plain-text
        extraction drops all hyperlinks.  Empty for sources whose reader
        does not capture links (txt, md, pdf, search synthetics).
        Consumed by the link-following stage
        (:func:`~tts_podcast.link_follower._gather_candidates`).
    """

    url: str
    title: str = ""
    summary: str = ""
    full_text: str = field(default="")
    scraped_ok: bool = field(default=False)
    kind: str = field(default="url")
    relevance: str | None = field(default=None)
    links: list[str] = field(default_factory=list)
