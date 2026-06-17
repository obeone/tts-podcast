"""
Tests for the web_scraper module.

Verifies article scraping behaviour with mocked trafilatura calls.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import trafilatura

from tts_podcast.web_scraper import scrape_url, scrape_urls


_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# scrape_url tests
# ---------------------------------------------------------------------------


class TestScrapeUrl:
    """Unit tests for scrape_url()."""

    def test_returns_source_on_success(self) -> None:
        """scrape_url returns a populated Source on success."""
        meta = SimpleNamespace(title="Real Title")
        with (
            patch("tts_podcast.web_scraper.trafilatura.fetch_url", return_value="<html>") as mock_fetch,
            patch("tts_podcast.web_scraper.trafilatura.extract", return_value="Article body text") as mock_extract,
            patch("tts_podcast.web_scraper.trafilatura.extract_metadata", return_value=meta),
        ):
            src = scrape_url("https://example.com/article")

        assert src.url == "https://example.com/article"
        assert src.title == "Real Title"
        assert src.full_text == "Article body text"
        assert src.summary == "Article body text"
        assert src.scraped_ok is True
        args, kwargs = mock_fetch.call_args
        assert args == ("https://example.com/article",)
        assert kwargs["no_ssl"] is True
        assert kwargs["config"].get("DEFAULT", "USER_AGENTS") == _BROWSER_USER_AGENT
        # extract is now called twice: once plain (for full_text) and once with
        # include_links=True / output_format="markdown" (for Source.links).
        assert mock_extract.call_count == 2
        first_call_args, first_call_kwargs = mock_extract.call_args_list[0]
        assert first_call_args == ("<html>",)
        assert first_call_kwargs == {}

    def test_falls_back_to_url_title_when_metadata_missing(self) -> None:
        """scrape_url uses a URL-derived title when extract_metadata returns no title."""
        with (
            patch("tts_podcast.web_scraper.trafilatura.fetch_url", return_value="<html>"),
            patch("tts_podcast.web_scraper.trafilatura.extract", return_value="Body"),
            patch("tts_podcast.web_scraper.trafilatura.extract_metadata", return_value=None),
        ):
            src = scrape_url("https://example.com/blog/post")

        assert src.scraped_ok is True
        assert src.title == "example.com/blog/post"

    def test_returns_failed_source_when_fetch_returns_none(self) -> None:
        """scrape_url returns scraped_ok=False when trafilatura.fetch_url returns None."""
        with patch("tts_podcast.web_scraper.trafilatura.fetch_url", return_value=None):
            src = scrape_url("https://example.com/article")

        assert src.scraped_ok is False
        assert src.full_text == ""
        assert src.summary == ""
        assert src.url == "https://example.com/article"

    def test_returns_failed_source_when_fetch_raises(self) -> None:
        """scrape_url returns scraped_ok=False when fetch_url throws."""
        with patch(
            "tts_podcast.web_scraper.trafilatura.fetch_url",
            side_effect=RuntimeError("network error"),
        ):
            src = scrape_url("https://example.com/article")

        assert src.scraped_ok is False
        assert src.full_text == ""

    def test_returns_failed_source_when_extract_returns_none(self) -> None:
        """scrape_url returns scraped_ok=False when extract returns nothing."""
        with (
            patch("tts_podcast.web_scraper.trafilatura.fetch_url", return_value="<html>"),
            patch("tts_podcast.web_scraper.trafilatura.extract", return_value=None),
            patch("tts_podcast.web_scraper.trafilatura.extract_metadata", return_value=None),
        ):
            src = scrape_url("https://example.com/article")

        assert src.scraped_ok is False

    def test_summary_capped_at_500_chars(self) -> None:
        """When full_text is long, summary is the first 500 chars."""
        long_text = "A" * 2000
        with (
            patch("tts_podcast.web_scraper.trafilatura.fetch_url", return_value="<html>"),
            patch("tts_podcast.web_scraper.trafilatura.extract", return_value=long_text),
            patch("tts_podcast.web_scraper.trafilatura.extract_metadata", return_value=None),
        ):
            src = scrape_url("https://example.com/article")

        assert len(src.summary) == 500
        assert src.full_text == long_text

    def test_passes_custom_user_agent_via_config(self) -> None:
        """scrape_url propagates the configured User-Agent and timeout."""
        captured: dict[str, str] = {}

        def fake_fetch(url, no_ssl=False, config=None, options=None):
            captured["ua"] = config.get("DEFAULT", "USER_AGENTS")
            captured["timeout"] = config.get("DEFAULT", "DOWNLOAD_TIMEOUT")
            return "<html>"

        with (
            patch("tts_podcast.web_scraper.trafilatura.fetch_url", side_effect=fake_fetch),
            patch("tts_podcast.web_scraper.trafilatura.extract", return_value="Body"),
            patch("tts_podcast.web_scraper.trafilatura.extract_metadata", return_value=None),
        ):
            scrape_url("https://example.com/a", timeout=42, user_agent="custom/9")

        assert captured["ua"] == "custom/9"
        assert captured["timeout"] == "42"

    def test_trafilatura_signature_does_not_include_headers(self) -> None:
        """Regression: fetch_url must not be called with an unsupported headers= kwarg."""
        from unittest.mock import create_autospec

        autospec_fetch = create_autospec(trafilatura.fetch_url, return_value="<html>")
        with (
            patch("tts_podcast.web_scraper.trafilatura.fetch_url", autospec_fetch),
            patch("tts_podcast.web_scraper.trafilatura.extract", return_value="Body"),
            patch("tts_podcast.web_scraper.trafilatura.extract_metadata", return_value=None),
        ):
            src = scrape_url("https://example.com/a")

        assert src.scraped_ok is True
        _, kwargs = autospec_fetch.call_args
        assert "headers" not in kwargs


# ---------------------------------------------------------------------------
# CloakBrowser fallback tests
# ---------------------------------------------------------------------------


class TestCloakFallback:
    """Behaviour of the optional CloakBrowser stealth fallback in scrape_url()."""

    def test_fallback_recovers_content_when_trafilatura_fails(self) -> None:
        """When trafilatura yields nothing, the cloak HTML is extracted instead."""
        with (
            patch("tts_podcast.web_scraper.trafilatura.fetch_url", return_value=None),
            patch("tts_podcast.web_scraper.cloak_fetcher.fetch_html", return_value="<cloak-html>") as mock_cloak,
            patch("tts_podcast.web_scraper.trafilatura.extract", return_value="Recovered body"),
            patch("tts_podcast.web_scraper.trafilatura.extract_metadata", return_value=None),
        ):
            src = scrape_url("https://example.com/blocked", use_cloak_fallback=True)

        assert src.scraped_ok is True
        assert src.full_text == "Recovered body"
        mock_cloak.assert_called_once()
        _, kwargs = mock_cloak.call_args
        assert kwargs.get("user_agent") == _BROWSER_USER_AGENT

    def test_fallback_not_invoked_when_disabled(self) -> None:
        """With use_cloak_fallback=False the stealth browser is never called."""
        with (
            patch("tts_podcast.web_scraper.trafilatura.fetch_url", return_value=None),
            patch("tts_podcast.web_scraper.cloak_fetcher.fetch_html") as mock_cloak,
        ):
            src = scrape_url("https://example.com/blocked", use_cloak_fallback=False)

        assert src.scraped_ok is False
        mock_cloak.assert_not_called()

    def test_fallback_not_invoked_when_trafilatura_succeeds(self) -> None:
        """A successful trafilatura scrape skips the cloak fallback entirely."""
        with (
            patch("tts_podcast.web_scraper.trafilatura.fetch_url", return_value="<html>"),
            patch("tts_podcast.web_scraper.trafilatura.extract", return_value="Body"),
            patch("tts_podcast.web_scraper.trafilatura.extract_metadata", return_value=None),
            patch("tts_podcast.web_scraper.cloak_fetcher.fetch_html") as mock_cloak,
        ):
            src = scrape_url("https://example.com/ok", use_cloak_fallback=True)

        assert src.scraped_ok is True
        mock_cloak.assert_not_called()

    def test_returns_failed_source_when_cloak_unavailable(self) -> None:
        """When cloak returns None (e.g. not installed), the failed source stands."""
        with (
            patch("tts_podcast.web_scraper.trafilatura.fetch_url", return_value=None),
            patch("tts_podcast.web_scraper.cloak_fetcher.fetch_html", return_value=None),
        ):
            src = scrape_url("https://example.com/blocked", use_cloak_fallback=True)

        assert src.scraped_ok is False
        assert src.full_text == ""

    def test_keeps_failed_source_when_cloak_html_extracts_nothing(self) -> None:
        """If cloak returns HTML but extraction is still empty, source stays failed."""
        with (
            patch("tts_podcast.web_scraper.trafilatura.fetch_url", return_value=None),
            patch("tts_podcast.web_scraper.cloak_fetcher.fetch_html", return_value="<cloak-html>"),
            patch("tts_podcast.web_scraper.trafilatura.extract", return_value=None),
            patch("tts_podcast.web_scraper.trafilatura.extract_metadata", return_value=None),
        ):
            src = scrape_url("https://example.com/blocked", use_cloak_fallback=True)

        assert src.scraped_ok is False


# ---------------------------------------------------------------------------
# scrape_urls tests
# ---------------------------------------------------------------------------


class TestScrapeUrls:
    """Unit tests for scrape_urls()."""

    def test_returns_results_in_input_order(self) -> None:
        """scrape_urls preserves the input URL order in the returned list."""
        urls = [f"https://example.com/{i}" for i in range(5)]

        with patch(
            "tts_podcast.web_scraper.scrape_url",
            side_effect=lambda url, *_a, **_k: SimpleNamespace(url=url, scraped_ok=True),
        ):
            results = scrape_urls(urls)

        assert [r.url for r in results] == urls

    def test_empty_input_returns_empty_list(self) -> None:
        """scrape_urls returns an empty list when given no URLs."""
        assert scrape_urls([]) == []

    def test_passes_custom_user_agent_to_scrape_url(self) -> None:
        """scrape_urls forwards the configured UA to each per-URL call."""
        urls = ["https://a.com"]

        with patch(
            "tts_podcast.web_scraper.scrape_url",
            return_value=SimpleNamespace(url="https://a.com", scraped_ok=True),
        ) as mock_scrape:
            scrape_urls(urls, user_agent="custom/7")

        mock_scrape.assert_called_once_with(
            "https://a.com", 10, "custom/7", use_cloak_fallback=False
        )

    def test_forwards_cloak_fallback_flag_to_scrape_url(self) -> None:
        """scrape_urls forwards use_cloak_fallback to each per-URL call."""
        with patch(
            "tts_podcast.web_scraper.scrape_url",
            return_value=SimpleNamespace(url="https://a.com", scraped_ok=True),
        ) as mock_scrape:
            scrape_urls(["https://a.com"], use_cloak_fallback=True)

        _, kwargs = mock_scrape.call_args
        assert kwargs["use_cloak_fallback"] is True


# ---------------------------------------------------------------------------
# Source.links population tests
# ---------------------------------------------------------------------------


class TestSourceLinksPopulation:
    """_extract_from_html populates Source.links from the include_links markdown pass."""

    def test_links_extracted_from_html_article_body(self) -> None:
        """Links found in the article body are stored in Source.links."""
        html = (
            "<html><body>"
            "<p>Article text. "
            '<a href="https://alpha.example/one">one</a> '
            '<a href="https://beta.example/two">two</a>'
            "</p></body></html>"
        )

        def _fake_extract(downloaded, **kwargs):
            if kwargs.get("include_links"):
                return (
                    "Article text. "
                    "[one](https://alpha.example/one) "
                    "[two](https://beta.example/two)"
                )
            return "Article text. one two"

        meta = SimpleNamespace(title="Test Article")
        with (
            patch("tts_podcast.web_scraper.trafilatura.fetch_url", return_value=html),
            patch("tts_podcast.web_scraper.trafilatura.extract", side_effect=_fake_extract),
            patch("tts_podcast.web_scraper.trafilatura.extract_metadata", return_value=meta),
        ):
            src = scrape_url("https://example.com/article")

        assert src.scraped_ok is True
        assert "https://alpha.example/one" in src.links
        assert "https://beta.example/two" in src.links

    def test_full_text_stays_clean_no_markdown_syntax(self) -> None:
        """full_text must not contain markdown link syntax even when links are captured."""
        html = "<html><body><p>Body. <a href='https://x.example/y'>link</a></p></body></html>"

        def _fake_extract(downloaded, **kwargs):
            if kwargs.get("include_links"):
                return "Body. [link](https://x.example/y)"
            return "Body. link"

        with (
            patch("tts_podcast.web_scraper.trafilatura.fetch_url", return_value=html),
            patch("tts_podcast.web_scraper.trafilatura.extract", side_effect=_fake_extract),
            patch("tts_podcast.web_scraper.trafilatura.extract_metadata", return_value=None),
        ):
            src = scrape_url("https://example.com/article")

        assert src.scraped_ok is True
        assert "](" not in src.full_text

    def test_links_empty_when_no_hrefs_in_article(self) -> None:
        """Source.links is empty when the markdown pass finds no links."""
        with (
            patch("tts_podcast.web_scraper.trafilatura.fetch_url", return_value="<html>"),
            patch("tts_podcast.web_scraper.trafilatura.extract", return_value="Plain body text"),
            patch("tts_podcast.web_scraper.trafilatura.extract_metadata", return_value=None),
        ):
            src = scrape_url("https://example.com/article")

        assert src.scraped_ok is True
        assert src.links == []

    def test_links_empty_on_failed_scrape(self) -> None:
        """A failed scrape (no content) still produces an empty Source.links."""
        with patch("tts_podcast.web_scraper.trafilatura.fetch_url", return_value=None):
            src = scrape_url("https://example.com/article")

        assert src.scraped_ok is False
        assert src.links == []
