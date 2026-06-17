"""
Local document loader for the tts-podcast pipeline.

Reads local files (plain text, Markdown, HTML, PDF) and returns populated
:class:`~tts_podcast.models.Source` objects with ``kind="file"``.  All
reader functions catch errors internally and surface them via
``Source.scraped_ok=False`` rather than raising.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import re

import trafilatura

from tts_podcast.link_extractor import extract_links_from_text
from tts_podcast.models import Source

# Regex for absolute http(s) links inside markdown output from trafilatura.
# Mirrors the same pattern in web_scraper so both paths parse identically.
_MD_LINK_RE = re.compile(r"\]\((https?://[^)\s]+)\)")

if TYPE_CHECKING:
    from rich.progress import Progress

logger = logging.getLogger(__name__)

_SUMMARY_CHARS = 500

_TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
_HTML_EXTENSIONS = {".html", ".htm"}
_PDF_EXTENSIONS = {".pdf"}


# ---------------------------------------------------------------------------
# Internal readers
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    """
    Read a plain-text or Markdown file as UTF-8.

    Parameters
    ----------
    path : Path
        Path to the file.

    Returns
    -------
    str
        File contents; replacement characters are used for invalid bytes.
    """
    return path.read_text(encoding="utf-8", errors="replace")


def _read_html(path: Path) -> tuple[str, list[str]]:
    """
    Read an HTML file and extract its main content and body links via trafilatura.

    Runs two trafilatura passes: a plain-text extraction for ``full_text``
    (keeping it clean and link-free) and a markdown extraction with
    ``include_links=True`` to capture hyperlinks scoped to the article body
    (nav/footer links are excluded by trafilatura's body detector).  The link
    list is deduplicated, preserving order.

    Parameters
    ----------
    path : Path
        Path to the HTML file.

    Returns
    -------
    tuple[str, list[str]]
        A ``(text, links)`` pair where *text* is the extracted plain text
        (``""`` when trafilatura finds no content) and *links* is a
        deduplicated list of absolute http(s) URLs found in the article body.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = trafilatura.extract(raw) or ""

    # Capture article-body links via a second trafilatura pass with
    # include_links=True.  Plain-text extraction drops all hrefs, so the
    # link-following stage would find 0 candidates without this step.
    links: list[str] = []
    try:
        links_md = trafilatura.extract(raw, include_links=True, output_format="markdown") or ""
    except Exception:  # noqa: BLE001
        links_md = ""
    if links_md:
        seen_links: set[str] = set()
        for m in _MD_LINK_RE.finditer(links_md):
            href = m.group(1)
            if href not in seen_links:
                seen_links.add(href)
                links.append(href)
        for href in extract_links_from_text(links_md):
            if href not in seen_links:
                seen_links.add(href)
                links.append(href)

    return text, links


def _read_pdf(path: Path) -> str:
    """
    Extract text from a PDF using pypdf.

    Parameters
    ----------
    path : Path
        Path to the PDF file.

    Returns
    -------
    str
        Page texts joined with ``"\\n\\n"``.  Pages that raise during
        extraction are skipped with a warning.

    Raises
    ------
    RuntimeError
        When ``pypdf`` is not installed.
    """
    try:
        from pypdf import PdfReader  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required to read PDF files. "
            "Install it with: uv add pypdf"
        ) from exc

    reader = PdfReader(str(path))
    page_texts: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to extract text from PDF page %d of %s: %s", i, path, exc)
            continue
        if text.strip():
            page_texts.append(text)
    return "\n\n".join(page_texts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_local_file(path: Path) -> Source:
    """
    Load a single local file and return a populated :class:`Source`.

    Dispatches to the appropriate reader based on file extension.  Unknown
    extensions are read as plain text with a warning.  The function never
    raises; all errors are converted into ``scraped_ok=False``.

    Parameters
    ----------
    path : Path
        Path to the local file.  Must exist.

    Returns
    -------
    Source
        Populated source with ``kind="file"`` and
        ``url="file://<absolute_path>"``.  ``scraped_ok`` is ``True`` when
        non-empty content was extracted, ``False`` otherwise.

    Raises
    ------
    FileNotFoundError
        When *path* does not exist.
    """
    abs_path = path.resolve()
    url = f"file://{abs_path}"

    if not path.exists():
        raise FileNotFoundError(f"Local file not found: {path}")

    ext = path.suffix.lower()
    # links is populated only for HTML files; other readers leave it empty
    # (txt/md may still contain bare URLs that the follower picks up via
    # extract_links_from_text on full_text; PDF links are not extractable).
    links: list[str] = []
    try:
        if ext in _TEXT_EXTENSIONS:
            text = _read_text(path)
        elif ext in _HTML_EXTENSIONS:
            text, links = _read_html(path)
        elif ext in _PDF_EXTENSIONS:
            text = _read_pdf(path)
        else:
            logger.warning(
                "Unknown file extension %r for %s — reading as plain text.", ext, path
            )
            text = _read_text(path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load local file %s: %s", path, exc)
        return Source(url=url, title=path.stem, kind="file", scraped_ok=False)

    if not text.strip():
        logger.warning("No text extracted from local file: %s", path)
        return Source(url=url, title=path.stem, kind="file", scraped_ok=False)

    summary = text[:_SUMMARY_CHARS].strip()
    logger.info("Loaded %d chars from local file: %s", len(text), path)
    return Source(
        url=url,
        title=path.stem,
        summary=summary,
        full_text=text,
        scraped_ok=True,
        kind="file",
        links=links,
    )


def load_local_files(
    paths: list[Path],
    *,
    progress: Progress | None = None,
    task_id: Any = None,
) -> list[Source]:
    """
    Load multiple local files sequentially and return :class:`Source` objects.

    Parameters
    ----------
    paths : list[Path]
        Local file paths to load.
    progress : rich.progress.Progress or None, optional
        A rich :class:`~rich.progress.Progress` instance.  When provided,
        ``task_id`` must also be supplied and will be advanced once per file.
    task_id : Any, optional
        Task identifier returned by ``progress.add_task()``.

    Returns
    -------
    list[Source]
        Populated source objects in the same order as the input paths.
    """
    results: list[Source] = []
    for path in paths:
        results.append(load_local_file(path))
        if progress is not None and task_id is not None:
            progress.advance(task_id)
    return results
