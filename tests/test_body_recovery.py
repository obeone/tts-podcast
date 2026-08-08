"""
Tests for the body_recovery module.

Runs real trafilatura extraction against the synthetic pages in
``tests/fixtures/truncating_page.py`` (one that reproduces the
under-selection bug of issue #21, one that extracts cleanly), and mocks
trafilatura only where the point is the guard rail rather than the extractor.
"""

from __future__ import annotations

import logging

import pytest
import trafilatura

from tests.fixtures.truncating_page import (
    ARTICLE_TITLES,
    newsletter_html,
    plain_article_html,
)
from tts_podcast.body_recovery import _MIN_RECOVERY_GAIN, extract_body_text


class TestHealthyPage:
    """A page trafilatura handles correctly must come back untouched."""

    def test_returns_main_extraction_unchanged(self) -> None:
        html = plain_article_html()
        assert extract_body_text(html, origin="https://example.com/a") == trafilatura.extract(html)

    def test_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="tts_podcast.body_recovery"):
            extract_body_text(plain_article_html(), origin="https://example.com/a")
        assert caplog.records == []


class TestTruncatedPage:
    """Regression cover for issue #21: sections dropped from the body."""

    def test_main_extraction_really_is_truncated(self) -> None:
        """Guard the premise: without recovery, articles go missing."""
        text = trafilatura.extract(newsletter_html()) or ""
        missing = [t for t in ARTICLE_TITLES if t not in text]
        assert missing, "synthetic page no longer reproduces the under-selection bug"

    def test_recovers_every_section(self) -> None:
        text = extract_body_text(newsletter_html(), origin="https://example.com/digest")
        assert [t for t in ARTICLE_TITLES if t not in text] == []

    def test_warns_with_both_lengths(self, caplog: pytest.LogCaptureFixture) -> None:
        html = newsletter_html()
        with caplog.at_level(logging.WARNING, logger="tts_podcast.body_recovery"):
            recovered = extract_body_text(html, origin="https://example.com/digest")

        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "https://example.com/digest" in message
        assert str(len(trafilatura.extract(html) or "")) in message
        assert str(len(recovered)) in message


class TestGuardRails:
    """Behaviour around the baseline comparison itself."""

    def test_empty_extraction_is_not_recovered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty body means a failed fetch, not a truncated one."""
        monkeypatch.setattr(trafilatura, "extract", lambda *a, **k: None)
        called = False

        def _baseline(*_args: object, **_kwargs: object) -> tuple[None, str, int]:
            nonlocal called
            called = True
            return None, "page furniture " * 100, 1400

        monkeypatch.setattr(trafilatura, "baseline", _baseline)

        assert extract_body_text("<html></html>", origin="x") == ""
        assert called is False

    def test_baseline_failure_keeps_main_extraction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(trafilatura, "extract", lambda *a, **k: "main body")

        def _boom(*_args: object, **_kwargs: object) -> tuple[None, str, int]:
            raise ValueError("lxml said no")

        monkeypatch.setattr(trafilatura, "baseline", _boom)

        assert extract_body_text("<html></html>", origin="x") == "main body"

    @pytest.mark.parametrize(
        ("gain", "expect_recovery"),
        [(_MIN_RECOVERY_GAIN - 0.2, False), (_MIN_RECOVERY_GAIN, True)],
    )
    def test_threshold_is_inclusive(
        self, monkeypatch: pytest.MonkeyPatch, gain: float, expect_recovery: bool
    ) -> None:
        main = "m" * 1000
        recovered = "b" * int(1000 * gain)
        monkeypatch.setattr(trafilatura, "extract", lambda *a, **k: main)
        monkeypatch.setattr(
            trafilatura, "baseline", lambda *a, **k: (None, recovered, len(recovered))
        )

        result = extract_body_text("<html></html>", origin="x")
        assert result == (recovered if expect_recovery else main)

    def test_shorter_baseline_never_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The common case: baseline keeps less than the real extractor."""
        monkeypatch.setattr(trafilatura, "extract", lambda *a, **k: "x" * 5000)
        monkeypatch.setattr(trafilatura, "baseline", lambda *a, **k: (None, "y" * 2600, 2600))

        assert extract_body_text("<html></html>", origin="x") == "x" * 5000
