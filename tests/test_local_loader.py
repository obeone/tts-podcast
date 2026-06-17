"""
Tests for the local_loader module.

Verifies local file loading for txt, md, html, pdf, missing files, empty
files, and unknown extensions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tts_podcast.local_loader import load_local_file, load_local_files


# ---------------------------------------------------------------------------
# load_local_file tests
# ---------------------------------------------------------------------------


class TestLoadLocalFilePlainText:
    """Tests for plain text (.txt) file loading."""

    def test_txt_round_trip(self, tmp_path: Path) -> None:
        """load_local_file reads a .txt file and returns its content."""
        f = tmp_path / "article.txt"
        f.write_text("Hello, world!", encoding="utf-8")

        src = load_local_file(f)

        assert src.scraped_ok is True
        assert src.full_text == "Hello, world!"
        assert src.title == "article"
        assert src.kind == "file"
        assert src.url == f"file://{f.resolve()}"

    def test_md_round_trip(self, tmp_path: Path) -> None:
        """load_local_file reads a .md file and returns its content."""
        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nBody text.", encoding="utf-8")

        src = load_local_file(f)

        assert src.scraped_ok is True
        assert src.full_text == "# Title\n\nBody text."
        assert src.title == "readme"
        assert src.kind == "file"

    def test_markdown_extension(self, tmp_path: Path) -> None:
        """load_local_file handles .markdown extension like .md."""
        f = tmp_path / "doc.markdown"
        f.write_text("Some markdown content.", encoding="utf-8")

        src = load_local_file(f)

        assert src.scraped_ok is True
        assert src.full_text == "Some markdown content."

    def test_summary_capped_at_500_chars(self, tmp_path: Path) -> None:
        """Summary is capped at 500 characters."""
        f = tmp_path / "long.txt"
        f.write_text("A" * 2000, encoding="utf-8")

        src = load_local_file(f)

        assert len(src.summary) == 500
        assert len(src.full_text) == 2000


class TestLoadLocalFileHtml:
    """Tests for HTML file loading."""

    def test_html_calls_trafilatura_extract(self, tmp_path: Path) -> None:
        """load_local_file passes raw HTML to trafilatura.extract (plain pass)."""
        f = tmp_path / "page.html"
        html_content = "<html><body><p>Hello</p></body></html>"
        f.write_text(html_content, encoding="utf-8")

        # _read_html calls trafilatura.extract twice: once plain (no kwargs) for
        # full_text, and once with include_links=True / output_format="markdown"
        # to capture hrefs.  We return "" for the second call so it is a no-op.
        def _fake_extract(raw, **kwargs):
            if kwargs:
                return ""
            return "Hello"

        with patch("tts_podcast.local_loader.trafilatura.extract", side_effect=_fake_extract):
            src = load_local_file(f)

        assert src.scraped_ok is True
        assert src.full_text == "Hello"
        assert src.kind == "file"

    def test_htm_extension(self, tmp_path: Path) -> None:
        """load_local_file handles .htm extension like .html."""
        f = tmp_path / "page.htm"
        f.write_text("<html><body><p>Content</p></body></html>", encoding="utf-8")

        def _fake_extract(raw, **kwargs):
            return "" if kwargs else "Content"

        with patch("tts_podcast.local_loader.trafilatura.extract", side_effect=_fake_extract):
            src = load_local_file(f)

        assert src.scraped_ok is True
        assert src.full_text == "Content"

    def test_html_extract_returns_none_gives_failed_source(self, tmp_path: Path) -> None:
        """When trafilatura.extract returns None for plain text, scraped_ok is False."""
        f = tmp_path / "empty.html"
        f.write_text("<html></html>", encoding="utf-8")

        with patch("tts_podcast.local_loader.trafilatura.extract", return_value=None):
            src = load_local_file(f)

        assert src.scraped_ok is False

    def test_html_links_populated_from_markdown_pass(self, tmp_path: Path) -> None:
        """load_local_file populates Source.links from the include_links markdown pass."""
        f = tmp_path / "article.html"
        f.write_text(
            "<html><body><p>Body. "
            '<a href="https://alpha.example/one">one</a> '
            '<a href="https://beta.example/two">two</a>'
            "</p></body></html>",
            encoding="utf-8",
        )

        # Simulate trafilatura returning markdown with link syntax on the second call.
        def _fake_extract(raw, **kwargs):
            if kwargs.get("include_links"):
                return "Body. [one](https://alpha.example/one) [two](https://beta.example/two)"
            return "Body. one two"

        with patch("tts_podcast.local_loader.trafilatura.extract", side_effect=_fake_extract):
            src = load_local_file(f)

        assert src.scraped_ok is True
        assert "https://alpha.example/one" in src.links
        assert "https://beta.example/two" in src.links
        # full_text must stay clean — no markdown link syntax.
        assert "](" not in src.full_text

    def test_html_links_empty_when_markdown_pass_fails(self, tmp_path: Path) -> None:
        """Source.links is empty when the include_links pass raises an exception."""
        f = tmp_path / "page.html"
        f.write_text("<html><body><p>Text</p></body></html>", encoding="utf-8")

        call_count = 0

        def _fake_extract(raw, **kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("include_links"):
                raise RuntimeError("trafilatura error")
            return "Text"

        with patch("tts_podcast.local_loader.trafilatura.extract", side_effect=_fake_extract):
            src = load_local_file(f)

        assert src.scraped_ok is True
        assert src.links == []


class TestLoadLocalFilePdf:
    """Tests for PDF file loading."""

    def _make_fake_reader(self, page_texts: list[str]) -> MagicMock:
        """Build a fake PdfReader whose pages have extract_text()."""
        pages = []
        for text in page_texts:
            page = MagicMock()
            page.extract_text.return_value = text
            pages.append(page)
        reader = MagicMock()
        reader.pages = pages
        return reader

    def test_pdf_pages_joined_with_double_newline(self, tmp_path: Path) -> None:
        """PDF pages are joined with '\\n\\n'."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4 fake")

        fake_reader = self._make_fake_reader(["Page one text.", "Page two text."])

        with patch("pypdf.PdfReader", return_value=fake_reader):
            src = load_local_file(f)

        assert src.scraped_ok is True
        assert src.full_text == "Page one text.\n\nPage two text."

    def test_pdf_pages_actually_joined(self, tmp_path: Path) -> None:
        """Directly test _read_pdf joins non-empty pages with double newline."""
        from tts_podcast.local_loader import _read_pdf

        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4 fake")

        fake_reader = self._make_fake_reader(["First page.", "Second page."])

        with patch("pypdf.PdfReader", return_value=fake_reader):
            result = _read_pdf(f)

        assert result == "First page.\n\nSecond page."

    def test_pdf_skips_empty_pages(self, tmp_path: Path) -> None:
        """Pages with empty text are skipped in the join."""
        from tts_podcast.local_loader import _read_pdf

        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4 fake")

        fake_reader = self._make_fake_reader(["Page 1.", "", "Page 3."])

        with patch("pypdf.PdfReader", return_value=fake_reader):
            result = _read_pdf(f)

        assert result == "Page 1.\n\nPage 3."

    def test_pdf_page_extraction_error_is_skipped(self, tmp_path: Path, caplog) -> None:
        """A page that raises during extract_text is skipped with a warning."""
        from tts_podcast.local_loader import _read_pdf

        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4 fake")

        bad_page = MagicMock()
        bad_page.extract_text.side_effect = RuntimeError("corrupt page")
        good_page = MagicMock()
        good_page.extract_text.return_value = "Good content."

        fake_reader = MagicMock()
        fake_reader.pages = [bad_page, good_page]

        with patch("pypdf.PdfReader", return_value=fake_reader):
            with caplog.at_level(logging.WARNING):
                result = _read_pdf(f)

        assert result == "Good content."
        assert any("Failed to extract text from PDF page" in r.message for r in caplog.records)


class TestLoadLocalFileMissingAndEmpty:
    """Tests for missing files and empty files."""

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """load_local_file raises FileNotFoundError for a non-existent file."""
        f = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError):
            load_local_file(f)

    def test_empty_file_returns_failed_source(self, tmp_path: Path) -> None:
        """An empty file yields scraped_ok=False without raising."""
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")

        src = load_local_file(f)

        assert src.scraped_ok is False
        assert src.kind == "file"


class TestLoadLocalFileUnknownExtension:
    """Tests for unknown file extensions."""

    def test_unknown_extension_reads_as_text(self, tmp_path: Path, caplog) -> None:
        """An unknown extension falls back to text reading with a warning."""
        f = tmp_path / "data.xyz"
        f.write_text("Some data content.", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            src = load_local_file(f)

        assert src.scraped_ok is True
        assert src.full_text == "Some data content."
        assert any("Unknown file extension" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# load_local_files tests
# ---------------------------------------------------------------------------


class TestLoadLocalFiles:
    """Tests for load_local_files()."""

    def test_returns_sources_in_input_order(self, tmp_path: Path) -> None:
        """load_local_files preserves input order."""
        files = []
        for i in range(3):
            f = tmp_path / f"file{i}.txt"
            f.write_text(f"Content {i}", encoding="utf-8")
            files.append(f)

        sources = load_local_files(files)

        assert len(sources) == 3
        for i, src in enumerate(sources):
            assert src.full_text == f"Content {i}"

    def test_empty_input_returns_empty_list(self, tmp_path: Path) -> None:
        """load_local_files returns empty list for empty input."""
        assert load_local_files([]) == []

    def test_advances_progress(self, tmp_path: Path) -> None:
        """load_local_files advances the progress task once per file."""
        f = tmp_path / "file.txt"
        f.write_text("Content", encoding="utf-8")

        progress = MagicMock()
        task_id = MagicMock()

        load_local_files([f], progress=progress, task_id=task_id)

        progress.advance.assert_called_once_with(task_id)
