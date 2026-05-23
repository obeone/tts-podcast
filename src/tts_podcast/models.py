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
    """

    url: str
    title: str = ""
    summary: str = ""
    full_text: str = field(default="")
    scraped_ok: bool = field(default=False)
    kind: str = field(default="url")
