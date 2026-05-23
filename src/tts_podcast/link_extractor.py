"""
Link extractor and categoriser for podcast source articles.

Scans article URLs and body text to extract and classify links into
categories such as GitHub repositories, Hugging Face models, arXiv papers,
and general source articles.  The structured output is consumed by the
report generator to produce a categorised reference section.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tts_podcast.models import Source

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CategorisedLink:
    """
    A single URL with its category and display label.

    Attributes
    ----------
    url : str
        The resolved URL.
    label : str
        Human-readable label (typically the article title or inferred name).
    category : str
        One of ``"repo"``, ``"model"``, ``"paper"``, ``"source"``, or
        ``"other"``.
    """

    url: str
    label: str
    category: str


@dataclass
class LinkReport:
    """
    Aggregated, categorised links extracted from a set of articles.

    Attributes
    ----------
    repos : list[CategorisedLink]
        GitHub / GitLab repository links.
    models : list[CategorisedLink]
        Hugging Face model or model-card links.
    papers : list[CategorisedLink]
        arXiv or academic paper links.
    sources : list[CategorisedLink]
        Primary source article links (one per article).
    other : list[CategorisedLink]
        Links that do not match any specific category.
    """

    repos: list[CategorisedLink] = field(default_factory=list)
    models: list[CategorisedLink] = field(default_factory=list)
    papers: list[CategorisedLink] = field(default_factory=list)
    sources: list[CategorisedLink] = field(default_factory=list)
    other: list[CategorisedLink] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Return the total number of links across all categories."""
        return (
            len(self.repos)
            + len(self.models)
            + len(self.papers)
            + len(self.sources)
            + len(self.other)
        )


# ---------------------------------------------------------------------------
# URL patterns for categorisation
# ---------------------------------------------------------------------------

_REPO_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^https?://github\.com/[\w.\-]+/[\w.\-]+",
        r"^https?://gitlab\.com/[\w.\-]+/[\w.\-]+",
        r"^https?://bitbucket\.org/[\w.\-]+/[\w.\-]+",
    )
)

_MODEL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^https?://huggingface\.co/[\w.\-]+/[\w.\-]+",
        r"^https?://hf\.co/[\w.\-]+/[\w.\-]+",
    )
)

_PAPER_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^https?://arxiv\.org/",
        r"^https?://papers\.ssrn\.com/",
        r"^https?://openreview\.net/",
        r"^https?://aclanthology\.org/",
        r"^https?://dl\.acm\.org/",
    )
)

# Regex for finding URLs embedded in article body text.
_URL_RE = re.compile(r"https?://[^\s\)\]>\"']+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def categorise_url(url: str) -> str:
    """
    Determine the category of a single URL.

    Parameters
    ----------
    url : str
        The URL to categorise.

    Returns
    -------
    str
        One of ``"repo"``, ``"model"``, ``"paper"``, or ``"other"``.

    Examples
    --------
    >>> categorise_url("https://github.com/user/repo")
    'repo'
    >>> categorise_url("https://arxiv.org/abs/2401.12345")
    'paper'
    >>> categorise_url("https://example.com/blog")
    'other'
    """
    for pat in _REPO_PATTERNS:
        if pat.match(url):
            return "repo"
    for pat in _MODEL_PATTERNS:
        if pat.match(url):
            return "model"
    for pat in _PAPER_PATTERNS:
        if pat.match(url):
            return "paper"
    return "other"


def extract_links_from_text(text: str) -> list[str]:
    """
    Extract all HTTP(S) URLs from a block of text.

    Trailing punctuation (periods, commas, parentheses) is stripped from
    each match to avoid capturing sentence-ending characters.

    Parameters
    ----------
    text : str
        The text to scan for URLs.

    Returns
    -------
    list[str]
        Deduplicated list of URLs in the order they first appear.

    Examples
    --------
    >>> extract_links_from_text("Visit https://example.com for info.")
    ['https://example.com']
    """
    urls: list[str] = []
    seen: set[str] = set()
    for match in _URL_RE.finditer(text):
        url = match.group(0).rstrip(".,;:!?)")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def extract_links(sources: list[Source]) -> LinkReport:
    """
    Extract and categorise all links from a list of source articles.

    For each source, the primary ``source.url`` is added as a ``"source"``
    link.  Additional URLs found in ``source.full_text`` (or
    ``source.summary`` when full text is empty) are categorised and added
    to the appropriate bucket.

    Parameters
    ----------
    sources : list[Source]
        Scraped sources produced by :func:`~tts_podcast.web_scraper.scrape_urls`.

    Returns
    -------
    LinkReport
        A :class:`LinkReport` with links grouped by category.

    Examples
    --------
    >>> from tts_podcast.models import Source
    >>> s = Source(url="https://example.com/post", title="Demo",
    ...            summary="See https://github.com/x/y")
    >>> report = extract_links([s])
    >>> len(report.sources)
    1
    >>> report.sources[0].url
    'https://example.com/post'
    """
    report = LinkReport()
    seen_urls: set[str] = set()

    for source in sources:
        # 1. Primary article URL → source
        if source.url and source.url not in seen_urls:
            seen_urls.add(source.url)
            report.sources.append(
                CategorisedLink(
                    url=source.url,
                    label=source.title,
                    category="source",
                )
            )

        # 2. Scan body text for additional URLs
        body = source.full_text or source.summary or ""
        for url in extract_links_from_text(body):
            if url in seen_urls:
                continue
            seen_urls.add(url)

            category = categorise_url(url)
            link = CategorisedLink(url=url, label=source.title, category=category)

            if category == "repo":
                report.repos.append(link)
            elif category == "model":
                report.models.append(link)
            elif category == "paper":
                report.papers.append(link)
            else:
                report.other.append(link)

    logger.info(
        "Extracted %d link(s): %d source(s), %d repo(s), %d model(s), "
        "%d paper(s), %d other.",
        report.total,
        len(report.sources),
        len(report.repos),
        len(report.models),
        len(report.papers),
        len(report.other),
    )
    return report
