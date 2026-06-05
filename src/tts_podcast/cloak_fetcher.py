"""
Optional stealth-browser fallback for fetching pages that block plain HTTP.

Wraps `CloakBrowser <https://github.com/CloakHQ/cloakbrowser>`_, a modified
Chromium that evades bot-detection systems (Cloudflare, reCAPTCHA, etc.) at
the C++ fingerprint level.  It is an **optional** dependency: install it with
``uv sync --extra cloak`` (or ``pip install tts-podcast[cloak]``).  When the
package is absent or fails at runtime, every function here degrades gracefully
to ``None``/``False`` so the caller can fall back to its normal behaviour
without the pipeline ever aborting.

The first ``cloakbrowser`` import also triggers a one-off ~200 MB Chromium
binary download, which is why this path is opt-in (``scraping.cloak_fallback``)
and only reached after trafilatura fails to extract content.
"""

from __future__ import annotations

import logging

from tts_podcast.user_agent import BROWSER_USER_AGENT

logger = logging.getLogger(__name__)

# Minimum wall-clock budget (seconds) granted to the stealth browser.  Anti-bot
# interstitials (Cloudflare Turnstile, JS challenges) routinely take longer than
# the plain-HTTP scrape timeout, so we floor the navigation deadline here.
_MIN_NAV_TIMEOUT_SECONDS = 30

# Guard so the "not installed" hint is logged at most once per process.
_warned_unavailable = False


def is_available() -> bool:
    """
    Report whether the optional ``cloakbrowser`` package can be imported.

    Uses :func:`importlib.util.find_spec` so the heavy package (and its
    Chromium binary download) is never actually imported just to probe
    availability.

    Returns
    -------
    bool
        ``True`` when ``cloakbrowser`` is installed, ``False`` otherwise.

    Examples
    --------
    >>> is_available()  # doctest: +SKIP
    False
    """
    import importlib.util

    return importlib.util.find_spec("cloakbrowser") is not None


def fetch_html(
    url: str,
    timeout: int = 10,
    user_agent: str = BROWSER_USER_AGENT,
) -> str | None:
    """
    Fetch the rendered HTML of a URL through the CloakBrowser stealth Chromium.

    Launches a humanised headless Chromium, navigates to ``url``, and returns
    the fully rendered DOM as an HTML string.  Never raises: any failure
    (package missing, launch error, navigation timeout, anti-bot block) is
    logged and converted into ``None`` so the caller can fall back.

    Parameters
    ----------
    url : str
        The URL to fetch.
    timeout : int, optional
        Navigation timeout in seconds, by default 10.  Floored to
        :data:`_MIN_NAV_TIMEOUT_SECONDS` to give challenge pages time to
        resolve, then converted to milliseconds for Playwright.
    user_agent : str, optional
        ``User-Agent`` advertised by the browser context, by default a
        browser-like Chrome UA.

    Returns
    -------
    str or None
        The rendered HTML on success, or ``None`` when the page could not be
        fetched (including when ``cloakbrowser`` is not installed).

    Examples
    --------
    >>> html = fetch_html("https://example.com")  # doctest: +SKIP
    >>> html is None or "<html" in html.lower()  # doctest: +SKIP
    True
    """
    global _warned_unavailable

    try:
        from cloakbrowser import launch
    except ImportError:
        if not _warned_unavailable:
            logger.warning(
                "cloakbrowser is not installed; skipping stealth fallback. "
                "Install it with `uv sync --extra cloak` to enable it."
            )
            _warned_unavailable = True
        return None

    nav_timeout_ms = max(timeout, _MIN_NAV_TIMEOUT_SECONDS) * 1000
    browser = None
    try:
        logger.info("cloakbrowser: launching stealth Chromium for %s", url)
        browser = launch(headless=True, humanize=True)
        context = browser.new_context(user_agent=user_agent)
        page = context.new_page()
        page.goto(url, timeout=nav_timeout_ms, wait_until="domcontentloaded")
        html = page.content()
        if not html:
            logger.warning("cloakbrowser: empty document returned for %s", url)
            return None
        return html
    except Exception as exc:  # noqa: BLE001
        logger.warning("cloakbrowser: failed to fetch %s: %s", url, exc)
        return None
    finally:
        if browser is not None:
            try:
                browser.close()
            except Exception:  # noqa: BLE001
                pass
