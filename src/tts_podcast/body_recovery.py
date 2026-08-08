"""
Article-body extraction with recovery for under-selected trafilatura bodies.

trafilatura's main extractor designates a single subtree of the page as the
article body.  On a page assembled from several sibling sections (a
newsletter, an aggregator, a link roundup) it can lock onto one of them and
report that fragment as the whole article.  Nothing in the return value says
so: the caller gets a plausible-looking string and writes an episode from a
tenth of the page, sponsor copy included.

Measured on a TLDR AI issue: 69,501 bytes of HTML, ``trafilatura.extract``
returns 2,324 characters (masthead plus one of five sections), while
``trafilatura.baseline`` over the same HTML returns 4,072 characters covering
every section.  ``favor_recall``, ``no_fallback`` and ``include_comments``
all return the identical 2,324-character fragment, so no extractor setting
recovers it.

The detector is therefore the baseline comparison itself, not a ratio against
the fetched HTML: the byte ratio is useless here, since even a perfect
extraction of that page would be about 6.5 % of its markup (the other 93 % is
scripts, styles and Tailwind class soup).  What separates a truncated body
from a healthy one is how much more text a second, structure-blind pass finds.
Measured across a spread of real pages, ``baseline / extract`` sits at 0.5-1.15
when extraction works (Wikipedia 0.52, LWN 0.73, martinfowler 0.93, The Verge
0.98, BBC and Hacker News 1.00, GitHub 1.03, uv docs 1.04, blog.python.org
1.14) and jumps to 1.75 on the truncated TLDR page, hence the 1.5 threshold in
:data:`_MIN_RECOVERY_GAIN`.

Baseline text is cruder than a successful extraction (it separates block
elements less carefully and keeps some boilerplate), so it is used only when
it is substantially larger, i.e. only when the alternative is an episode
written from a fragment.
"""

from __future__ import annotations

import logging

import trafilatura

logger = logging.getLogger(__name__)

# How much longer the baseline pass must be before its text replaces the main
# extraction.  See the module docstring for the measurements behind this value:
# healthy pages top out around 1.15, the known-bad page sits at 1.75.
_MIN_RECOVERY_GAIN = 1.5


def _baseline_text(html: str) -> str:
    """
    Return trafilatura's baseline extraction of *html*, or ``""`` on failure.

    :func:`trafilatura.baseline` applies a simple structure-blind heuristic
    (JSON-LD article body, ``<article>`` elements, quotes, then every ``<p>``),
    which is exactly what makes it a useful second opinion on a page whose
    body detector under-selected.

    Parameters
    ----------
    html : str
        Raw HTML of the page.

    Returns
    -------
    str
        The baseline text, or an empty string when baseline extraction finds
        nothing or raises.
    """
    try:
        _, text, _ = trafilatura.baseline(html)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Baseline extraction failed: %s", exc)
        return ""
    return text or ""


def extract_body_text(html: str, *, origin: str) -> str:
    """
    Extract the main article text, recovering from a truncated body.

    Runs the normal :func:`trafilatura.extract` pass, then cross-checks it
    against :func:`trafilatura.baseline`.  When the baseline text is at least
    :data:`_MIN_RECOVERY_GAIN` times longer, the main extraction is treated as
    truncated: a warning naming both lengths is logged and the baseline text is
    returned instead.  Otherwise the main extraction is returned byte for byte,
    so pages that extract cleanly are unaffected.

    Never raises.

    Parameters
    ----------
    html : str
        Raw HTML of the page.
    origin : str
        URL or file path of the page, used in log messages only.

    Returns
    -------
    str
        The article text, empty when trafilatura extracted nothing.  An empty
        main extraction is returned as-is rather than recovered: that is the
        signature of an access error or a JS-only shell, which the caller
        handles by failing the scrape (and, when enabled, retrying through the
        CloakBrowser fallback), not by scraping the page furniture.

    Examples
    --------
    >>> extract_body_text("<html><body><p>Hello.</p></body></html>", origin="x")
    'Hello.'
    """
    text = trafilatura.extract(html) or ""
    if not text:
        return ""

    recovered = _baseline_text(html)
    if len(recovered) < len(text) * _MIN_RECOVERY_GAIN:
        return text

    logger.warning(
        "Article body looks truncated for %s: trafilatura kept %d chars, but a "
        "baseline pass over the same HTML yields %d (%.1fx more). Using the "
        "baseline text instead.",
        origin,
        len(text),
        len(recovered),
        len(recovered) / len(text),
    )
    return recovered
