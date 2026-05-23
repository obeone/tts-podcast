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
    Represents a single article fetched from an arbitrary URL.

    Attributes
    ----------
    url : str
        The URL of the source article.
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
    """

    url: str
    title: str = ""
    summary: str = ""
    full_text: str = field(default="")
    scraped_ok: bool = field(default=False)
