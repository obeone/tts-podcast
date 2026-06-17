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
from urllib.parse import urlparse

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
# Followability heuristic (stage-1 pre-fetch filter for link-following)
# ---------------------------------------------------------------------------

# Binary / asset file extensions that never carry article-like content worth
# fetching.  ``.pdf`` and ``.html`` are deliberately KEPT (they are real
# content) and therefore absent from this set.
_ASSET_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".svg",
        ".webp",
        ".ico",
        ".css",
        ".js",
        ".mjs",
        ".woff",
        ".woff2",
        ".ttf",
        ".mp4",
        ".mp3",
        ".zip",
    }
)

# Host substrings for ad networks, social sharers, and trackers whose links
# are noise rather than content.  Matched as substrings against the lowercased
# ``netloc`` so e.g. ``m.facebook.com`` and ``www.facebook.com`` both hit.
_JUNK_HOST_SUBSTRINGS: tuple[str, ...] = (
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "t.co",
    "doubleclick.net",
    "googlesyndication.com",
    "googletagmanager.com",
    "pinterest.",
)

# (host_substring, path_prefix) pairs for share/submit endpoints that live on
# otherwise-legitimate hosts — only the specific sharing path is junk.
_JUNK_HOST_PATH_PAIRS: tuple[tuple[str, str], ...] = (
    ("linkedin.com", "/share"),
    ("reddit.com", "/submit"),
)

# Path prefixes that signal a non-article destination (auth, commerce, sharing)
# regardless of host.
_JUNK_PATH_PREFIXES: tuple[str, ...] = (
    "/login",
    "/signup",
    "/cart",
    "/share",
)


def is_followable_link(url: str) -> bool:
    """
    Decide whether a URL is worth fetching during link following (stage 1).

    This is the cheap, pre-fetch heuristic of the two-stage link-following
    selection: it inspects only the URL string (no network call) and keeps
    everything that *looks* like real content — articles, papers, repos,
    models, and generic pages — while dropping obvious junk.  A second,
    content-aware LLM stage (in :mod:`tts_podcast.link_follower`) makes the
    final relevance call on what survives this filter.

    A URL is dropped when any of the following holds:

    - its scheme is not ``http``/``https`` (e.g. ``mailto:``, ``tel:``,
      ``javascript:``, or a bare ``#anchor`` with no scheme);
    - it is a same-page anchor (starts with ``#``);
    - its path ends in an asset/binary extension in :data:`_ASSET_EXTENSIONS`
      (``.pdf`` and ``.html`` are intentionally KEPT);
    - its host matches a known ad/social/tracker host
      (:data:`_JUNK_HOST_SUBSTRINGS`) or a host-specific share/submit endpoint;
    - its path begins with an obvious non-article prefix
      (:data:`_JUNK_PATH_PREFIXES`).

    Parameters
    ----------
    url : str
        The candidate URL to evaluate.

    Returns
    -------
    bool
        ``True`` when the URL passes the heuristic and should be fetched,
        ``False`` when it should be skipped.

    Examples
    --------
    >>> is_followable_link("https://example.com/article-about-ai")
    True
    >>> is_followable_link("https://arxiv.org/abs/2401.12345")
    True
    >>> is_followable_link("https://github.com/user/repo")
    True
    >>> is_followable_link("https://example.com/paper.pdf")
    True
    >>> is_followable_link("mailto:hello@example.com")
    False
    >>> is_followable_link("#section-2")
    False
    >>> is_followable_link("https://example.com/logo.png")
    False
    >>> is_followable_link("https://cdn.example.com/app.js")
    False
    >>> is_followable_link("https://facebook.com/sharer/sharer.php")
    False
    >>> is_followable_link("https://example.com/login")
    False
    """
    if not url:
        return False

    # Same-page anchors carry no scheme and reference the current document.
    if url.startswith("#"):
        return False

    try:
        parsed = urlparse(url)
    except Exception:  # noqa: BLE001 — a malformed URL is simply not followable.
        return False

    # Only plain web schemes are fetchable; this rejects mailto/tel/javascript
    # and any other non-http(s) scheme in one check.
    if parsed.scheme not in ("http", "https"):
        return False

    host = parsed.netloc.lower()
    path = parsed.path.lower()

    # Asset/binary extensions: look at the final path segment's suffix.
    last_segment = path.rsplit("/", 1)[-1]
    if "." in last_segment:
        ext = last_segment[last_segment.rfind(".") :]
        if ext in _ASSET_EXTENSIONS:
            return False

    # Known ad/social/tracker hosts (substring match covers www./m. variants).
    for junk in _JUNK_HOST_SUBSTRINGS:
        if junk in host:
            return False

    # Host-specific share/submit endpoints (the host itself is fine elsewhere).
    for junk_host, junk_path in _JUNK_HOST_PATH_PAIRS:
        if junk_host in host and path.startswith(junk_path):
            return False

    # Obvious non-article paths (auth, commerce, sharing) on any host.
    for prefix in _JUNK_PATH_PREFIXES:
        if path.startswith(prefix):
            return False

    return True


# ---------------------------------------------------------------------------
# Relevance label (shared across dialogue / research / report)
# ---------------------------------------------------------------------------


def relevance_label(relevance: str | None) -> str:
    """
    Map a :attr:`Source.relevance` verdict to a short human-readable label.

    The link-following stage annotates each followed page with a relevance
    verdict (``"core"`` / ``"supporting"``).  Several downstream consumers
    (dialogue prompt, research prompt, report) surface that verdict, so the
    mapping lives in one place to stay consistent.  Seed/primary inputs carry
    ``relevance is None`` and intentionally map to the empty string so their
    rendered blocks stay byte-identical (no annotation).

    Parameters
    ----------
    relevance : str or None
        A :attr:`tts_podcast.models.Source.relevance` value.

    Returns
    -------
    str
        ``""`` for ``None`` (and any unknown value), ``"core source"`` for
        ``"core"``, ``"supporting source"`` for ``"supporting"``.

    Examples
    --------
    >>> relevance_label(None)
    ''
    >>> relevance_label("core")
    'core source'
    >>> relevance_label("supporting")
    'supporting source'
    """
    if relevance == "core":
        return "core source"
    if relevance == "supporting":
        return "supporting source"
    return ""


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
        # 1. Primary article URL → source (skip non-http schemes like file:// and search://)
        if source.url and source.url not in seen_urls and source.url.startswith(("http://", "https://")):
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
