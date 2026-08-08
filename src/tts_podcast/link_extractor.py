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
from html import unescape
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlparse

if TYPE_CHECKING:
    from tts_podcast.models import Source

logger = logging.getLogger(__name__)

# Absolute http(s) target of a markdown link, i.e. the URL in [text](https://…),
# which is what trafilatura emits with include_links=True.
_MD_LINK_RE = re.compile(r"\]\((https?://[^)\s]+)\)")

# href of any HTML anchor, single or double quoted.  A regex rather than a real
# parser is deliberate: lxml is only a transitive dependency (via trafilatura),
# and the cost of a missed exotic anchor is one lost candidate among many, not
# a failure — every href collected here still has to clear is_followable_link
# and the relevance judge.
_ANCHOR_HREF_RE = re.compile(r"""<a\b[^>]*?\bhref\s*=\s*["']([^"']+)["']""", re.IGNORECASE)

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


def _unescape_fully(href: str, limit: int = 3) -> str:
    """
    Undo HTML entity escaping on *href*, including double escaping.

    A single :func:`html.unescape` is the correct reading of the HTML spec, but
    pages that build links by templating an already-escaped URL emit
    ``&amp;amp;``, which one pass leaves as ``&amp;``.  The result is the same
    destination reaching us under two spellings, so it survives deduplication
    and burns two candidate slots on one page (observed on a real newsletter).
    Unescaping to a fixed point collapses them.

    The theoretical cost is a URL whose query string legitimately contains the
    literal text ``&amp;``; that URL would be rewritten. It is not a real risk
    here, and the alternative is fetching the same page twice.

    Parameters
    ----------
    href : str
        Raw href as it appeared in the document.
    limit : int, optional
        Maximum unescape passes, by default 3.  Bounds the loop rather than
        trusting the input to converge.

    Returns
    -------
    str
        The href with entities resolved.
    """
    for _ in range(limit):
        unescaped = unescape(href)
        if unescaped == href:
            break
        href = unescaped
    return href


def _site_of(url: str) -> str:
    """
    Return a coarse site key for *url*, used only to order candidates.

    Compares the last two labels of the hostname, so ``advertise.tldr.tech``
    and ``www.tldr.tech`` share a key with ``tldr.tech``.  This is a deliberate
    approximation: a multi-part public suffix such as ``co.uk`` will lump
    unrelated sites together.  The consequence is a candidate ordered slightly
    worse, never one dropped, so a real public-suffix list would buy nothing
    worth its weight here.

    Parameters
    ----------
    url : str
        Absolute URL to key.

    Returns
    -------
    str
        Lowercased ``"second-level.tld"``, or ``""`` when the host is
        unparseable.
    """
    try:
        host = urlparse(url).netloc.lower().split(":")[0]
    except ValueError:
        return ""
    labels = [label for label in host.split(".") if label]
    return ".".join(labels[-2:]) if len(labels) >= 2 else host


def collect_document_links(markdown: str, html: str, base_url: str | None = None) -> list[str]:
    """
    Collect a document's outbound links, article body first, whole page after.

    Link following needs candidates, and the two available sources disagree
    about what counts.  trafilatura's ``include_links=True`` markdown pass is
    the *precise* one: it scopes links to the article body it detected, so
    nav and footer junk never appears.  The raw anchor list is the *complete*
    one: it sees every ``<a href>`` on the page, junk included.

    Neither alone is enough.  On a page whose body detector under-selects (a
    newsletter, an aggregator, a link roundup — precisely the pages worth
    following links from), the markdown pass can return almost nothing while
    the page carries a dozen real article links.  Observed on a TLDR AI issue:
    69 KB of HTML, 25 distinct anchors, and a detected body of 2.3 KB holding
    exactly one link, the sponsor ad.  No trafilatura setting recovers the
    rest; ``favor_recall``, ``no_fallback`` and ``include_comments`` all
    return the same fragment.

    So this returns the union, ordered rather than filtered: body links first,
    then bare URLs in the body text, then everything else on the page.  Order
    is the whole mechanism.  Consumers take a bounded prefix of this list
    (``link_follower`` caps candidates per hop), so on a page where body
    detection works, the body links are consumed first and the page tail is
    never reached.  On a page where it fails, the tail is what saves the run.
    Junk is not this function's problem: :func:`is_followable_link` and the
    LLM relevance judge are the two gates that follow.

    Anchor hrefs are HTML-unescaped before deduplication, so a page emitting
    ``&amp;`` in query strings does not yield the same URL twice.

    Parameters
    ----------
    markdown : str
        Output of trafilatura's ``include_links=True, output_format="markdown"``
        pass, or ``""`` when that pass failed or returned nothing.
    html : str
        The raw HTML of the same document, scanned for anchors the body pass
        did not surface.
    base_url : str or None, optional
        Document URL used to resolve relative hrefs.  When ``None`` (a local
        file, where relative links cannot be resolved to anything fetchable)
        only absolute http(s) anchors are kept.

    Returns
    -------
    list[str]
        Deduplicated absolute http(s) URLs, body links first, in first-seen
        order.
    """
    links: list[str] = []
    seen: set[str] = set()

    def _add(href: str) -> None:
        """Append *href* unless an equal string was already collected."""
        if href not in seen:
            seen.add(href)
            links.append(href)

    # 1. Markdown link targets: the highest-confidence signal, so they lead.
    for match in _MD_LINK_RE.finditer(markdown):
        _add(_unescape_fully(match.group(1)))
    # 2. Bare URLs sitting in the body text rather than behind an anchor.
    for href in extract_links_from_text(markdown):
        _add(_unescape_fully(href))

    # 3. The rest of the page's anchors, as the recall backstop, off-site
    #    first.  Document order would be the obvious choice and is the wrong
    #    one: site chrome sits at the top of the markup, so it would fill the
    #    candidate budget before a single article link was reached.  On the
    #    newsletter that motivated this, the first five anchors are the
    #    masthead, the newsletter index, the ad-sales page and the blog.
    #    A link leaving the document's own domain is far likelier to be
    #    content, because linking outward is what these pages are for, so
    #    off-site anchors are promoted ahead of same-site ones.  Both groups
    #    are kept: this reorders the tail, it never drops from it.
    base_site = _site_of(base_url) if base_url else ""
    offsite: list[str] = []
    onsite: list[str] = []
    for match in _ANCHOR_HREF_RE.finditer(html):
        href = _unescape_fully(match.group(1).strip())
        if base_url:
            href = urljoin(base_url, href)
        if not href.lower().startswith(("http://", "https://")):
            continue
        if base_site and _site_of(href) == base_site:
            onsite.append(href)
        else:
            offsite.append(href)
    for href in (*offsite, *onsite):
        _add(href)
    return links


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
