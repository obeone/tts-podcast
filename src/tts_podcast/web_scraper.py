"""
Web scraper module for fetching full article text from arbitrary URLs.

Uses trafilatura to fetch and extract the main content + metadata from a
web page.  Multiple URLs can be scraped concurrently via a thread pool.
A scrape failure (network error, no extractable content) is reported via
the ``Source.scraped_ok`` flag so the caller can decide how to react.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import trafilatura
from trafilatura.settings import use_config

from tts_podcast import cloak_fetcher
from tts_podcast.models import Source
from tts_podcast.user_agent import BROWSER_USER_AGENT

if TYPE_CHECKING:
    from rich.progress import Progress

logger = logging.getLogger(__name__)

# Fallback summary length (characters) when none is available from the page.
_SUMMARY_CHARS = 500


def _title_fallback(url: str) -> str:
    """
    Return a best-effort display title derived from a URL.

    Used when trafilatura cannot extract a ``<title>`` from the page.

    Parameters
    ----------
    url : str
        The source URL.

    Returns
    -------
    str
        ``"<host><path>"`` with leading/trailing slashes trimmed, e.g.
        ``"example.com/blog/foo"``.  Falls back to the bare URL if parsing
        fails.
    """
    try:
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if path:
            return f"{parsed.netloc}/{path}"
        return parsed.netloc or url
    except Exception:  # noqa: BLE001
        return url


def _extract_from_html(url: str, downloaded: str) -> Source:
    """
    Build a :class:`Source` from already-downloaded HTML via trafilatura.

    Shared by the plain-HTTP path and the CloakBrowser fallback so both
    produce identically-shaped sources.

    Parameters
    ----------
    url : str
        The source URL (used for the title fallback).
    downloaded : str
        Raw HTML to extract the article body and metadata from.

    Returns
    -------
    Source
        Populated source with ``scraped_ok=True`` when non-empty content was
        extracted; otherwise a source with empty ``full_text``/``summary`` and
        ``scraped_ok=False``.
    """
    text = trafilatura.extract(downloaded)
    metadata = trafilatura.extract_metadata(downloaded)
    title = (metadata.title if metadata and metadata.title else "") or _title_fallback(url)

    if not text:
        return Source(url=url, title=title)

    summary = text[:_SUMMARY_CHARS].strip()
    return Source(
        url=url,
        title=title,
        summary=summary,
        full_text=text,
        scraped_ok=True,
    )


def _scrape_with_trafilatura(url: str, timeout: int, user_agent: str) -> Source:
    """
    Download and extract one URL using trafilatura only (no fallback).

    Never raises; network and extraction errors are converted into a
    ``scraped_ok=False`` source.

    Parameters
    ----------
    url : str
        The URL of the article to scrape.
    timeout : int
        HTTP request timeout in seconds.
    user_agent : str
        ``User-Agent`` header to send with the request.

    Returns
    -------
    Source
        Populated source; ``scraped_ok=False`` on any failure.
    """
    try:
        logger.debug("Fetching URL: %s", url)
        cfg = use_config()
        cfg.set("DEFAULT", "USER_AGENTS", user_agent)
        cfg.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(timeout))
        downloaded = trafilatura.fetch_url(url, no_ssl=True, config=cfg)
        if downloaded is None:
            logger.warning("trafilatura.fetch_url returned None for URL: %s", url)
            return Source(url=url, title=_title_fallback(url))

        source = _extract_from_html(url, downloaded)
        if source.scraped_ok:
            logger.info("Successfully scraped %d chars from: %s", len(source.full_text), url)
        else:
            logger.warning("trafilatura.extract returned no content for URL: %s", url)
        return source

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to scrape URL %s: %s", url, exc)
        return Source(url=url, title=_title_fallback(url))


def scrape_url(
    url: str,
    timeout: int = 10,
    user_agent: str = BROWSER_USER_AGENT,
    use_cloak_fallback: bool = False,
) -> Source:
    """
    Fetch one URL and return a populated :class:`Source`.

    Uses trafilatura to download the page, extract the main article body,
    and read metadata for the title.  ``Source.scraped_ok`` is ``True`` when
    non-empty content was extracted, ``False`` otherwise.  Never raises;
    network and extraction errors are converted into ``scraped_ok=False``.

    When ``use_cloak_fallback`` is set and trafilatura yields no content — the
    typical signature of an access error (403/429, Cloudflare, or a JS-only
    page) — the URL is retried through the optional CloakBrowser stealth
    Chromium (:mod:`tts_podcast.cloak_fetcher`).  If that package is missing or
    also fails, the original trafilatura result is returned unchanged.

    Parameters
    ----------
    url : str
        The URL of the article to scrape.
    timeout : int, optional
        HTTP request timeout in seconds, by default 10.
    user_agent : str, optional
        ``User-Agent`` header to send with the request, by default a
        browser-like Chrome UA.
    use_cloak_fallback : bool, optional
        When ``True``, retry failed scrapes through the CloakBrowser stealth
        browser, by default ``False``.

    Returns
    -------
    Source
        Populated source object.  When extraction fails, ``full_text`` and
        ``summary`` are empty strings, ``title`` is a URL-derived fallback,
        and ``scraped_ok`` is ``False``.

    Examples
    --------
    >>> src = scrape_url("https://example.com/article")
    >>> src.scraped_ok
    True
    """
    source = _scrape_with_trafilatura(url, timeout, user_agent)
    if source.scraped_ok or not use_cloak_fallback:
        return source

    logger.info(
        "trafilatura could not extract %s — retrying via CloakBrowser stealth fallback.",
        url,
    )
    html = cloak_fetcher.fetch_html(url, timeout=timeout, user_agent=user_agent)
    if not html:
        return source

    cloak_source = _extract_from_html(url, html)
    if cloak_source.scraped_ok:
        logger.info(
            "CloakBrowser fallback recovered %d chars from: %s",
            len(cloak_source.full_text),
            url,
        )
        return cloak_source

    logger.warning("CloakBrowser fallback still produced no content for URL: %s", url)
    return source


def scrape_urls(
    urls: list[str],
    timeout: int = 10,
    user_agent: str = BROWSER_USER_AGENT,
    progress: Progress | None = None,
    task_id: Any = None,
    use_cloak_fallback: bool = False,
) -> list[Source]:
    """
    Scrape multiple URLs in parallel and return :class:`Source` objects.

    Up to 10 URLs are scraped concurrently using a thread pool.  Results
    are returned in the same order as the input list.

    Parameters
    ----------
    urls : list[str]
        URLs to fetch.
    timeout : int, optional
        HTTP request timeout in seconds passed to :func:`scrape_url`,
        by default 10.
    user_agent : str, optional
        ``User-Agent`` header to send with article requests, by default
        a browser-like Chrome UA.
    use_cloak_fallback : bool, optional
        Forwarded to :func:`scrape_url`; when ``True``, failed scrapes are
        retried through the optional CloakBrowser stealth browser, by default
        ``False``.
    progress : rich.progress.Progress or None, optional
        A rich :class:`~rich.progress.Progress` instance.  When provided,
        ``task_id`` must also be supplied and will be advanced once per
        completed article.
    task_id : Any, optional
        Task identifier returned by ``progress.add_task()``.

    Returns
    -------
    list[Source]
        Populated source objects in the same order as the input URLs.
    """
    total = len(urls)
    if total == 0:
        return []

    max_workers = min(10, total)
    logger.info("Scraping %d URL(s) with up to %d worker(s)…", total, max_workers)

    results: dict[int, Source] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                scrape_url, url, timeout, user_agent, use_cloak_fallback=use_cloak_fallback
            ): i
            for i, url in enumerate(urls)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            results[idx] = future.result()
            if progress is not None and task_id is not None:
                progress.advance(task_id)

    logger.info("Scraping complete (%d URL(s) processed).", total)
    return [results[i] for i in range(total)]
