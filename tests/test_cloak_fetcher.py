"""
Tests for the optional CloakBrowser stealth-fetch fallback.

The real ``cloakbrowser`` package is not a test dependency; these tests inject
a fake module into ``sys.modules`` to exercise the success path and rely on its
genuine absence to exercise graceful degradation.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tts_podcast import cloak_fetcher


class TestIsAvailable:
    """Unit tests for cloak_fetcher.is_available()."""

    def test_true_when_spec_found(self) -> None:
        """is_available reports True when the package spec resolves."""
        with patch("importlib.util.find_spec", return_value=object()):
            assert cloak_fetcher.is_available() is True

    def test_false_when_spec_missing(self) -> None:
        """is_available reports False when the package cannot be located."""
        with patch("importlib.util.find_spec", return_value=None):
            assert cloak_fetcher.is_available() is False


class TestFetchHtml:
    """Unit tests for cloak_fetcher.fetch_html()."""

    def test_returns_none_when_package_absent(self) -> None:
        """A missing cloakbrowser import degrades to None without raising."""
        # Ensure the import fails deterministically even if installed.
        with patch.dict(sys.modules, {"cloakbrowser": None}):
            assert cloak_fetcher.fetch_html("https://example.com") is None

    def test_returns_rendered_html_on_success(self) -> None:
        """The happy path returns page.content() and closes the browser."""
        page = MagicMock()
        page.content.return_value = "<html>cloak</html>"
        context = MagicMock()
        context.new_page.return_value = page
        browser = MagicMock()
        browser.new_context.return_value = context
        fake_launch = MagicMock(return_value=browser)
        fake_module = SimpleNamespace(launch=fake_launch)

        with patch.dict(sys.modules, {"cloakbrowser": fake_module}):
            html = cloak_fetcher.fetch_html(
                "https://example.com", timeout=10, user_agent="custom/9"
            )

        assert html == "<html>cloak</html>"
        browser.new_context.assert_called_once_with(user_agent="custom/9")
        page.goto.assert_called_once()
        # Navigation deadline is floored to the 30 s minimum (in ms).
        _, goto_kwargs = page.goto.call_args
        assert goto_kwargs["timeout"] == 30_000
        browser.close.assert_called_once()

    def test_returns_none_and_closes_browser_on_navigation_error(self) -> None:
        """A navigation failure is swallowed and the browser is still closed."""
        page = MagicMock()
        page.goto.side_effect = RuntimeError("blocked")
        context = MagicMock()
        context.new_page.return_value = page
        browser = MagicMock()
        browser.new_context.return_value = context
        fake_module = SimpleNamespace(launch=MagicMock(return_value=browser))

        with patch.dict(sys.modules, {"cloakbrowser": fake_module}):
            html = cloak_fetcher.fetch_html("https://example.com")

        assert html is None
        browser.close.assert_called_once()

    def test_returns_none_when_content_empty(self) -> None:
        """An empty document is treated as a failure."""
        page = MagicMock()
        page.content.return_value = ""
        context = MagicMock()
        context.new_page.return_value = page
        browser = MagicMock()
        browser.new_context.return_value = context
        fake_module = SimpleNamespace(launch=MagicMock(return_value=browser))

        with patch.dict(sys.modules, {"cloakbrowser": fake_module}):
            assert cloak_fetcher.fetch_html("https://example.com") is None
        browser.close.assert_called_once()
