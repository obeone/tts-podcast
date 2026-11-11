"""
Tests for :mod:`tts_podcast.tts_models`.

``uses_speech_metadata`` decides which TTS request schema a model gets, so a
wrong answer silently routes a model to the wrong prompt format.
"""

from __future__ import annotations

import pytest

from tts_podcast.tts_models import uses_speech_metadata


class TestUsesSpeechMetadata:
    """Version-based detection of the speech-metadata request schema."""

    @pytest.mark.parametrize(
        "model", ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"]
    )
    def test_true_for_gemini_3_8_models(self, model):
        """Both Gemini 3.8 TTS variants use the speech-metadata schema."""
        assert uses_speech_metadata(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "gemini-3.1-flash-tts-preview",
            "gemini-2.5-flash-preview-tts",
            "gemini-2.5-pro-preview-tts",
        ],
    )
    def test_false_for_legacy_models(self, model):
        """Gemini 2.5 and 3.1 TTS models keep the text-preamble schema."""
        assert uses_speech_metadata(model) is False

    @pytest.mark.parametrize(
        "model",
        [
            "gemini-3.9-flash-tts",
            "gemini-4.0-flash-tts",
            "gemini-4-flash-tts",
            "gemini-3.10-flash-tts",
        ],
    )
    def test_true_for_later_versions(self, model):
        """Later versions qualify, and the minor is compared as a number."""
        assert uses_speech_metadata(model) is True

    @pytest.mark.parametrize("model", [None, "", "gpt-4o-tts", "tts-3.9"])
    def test_false_for_missing_or_non_gemini_names(self, model):
        """Unknown, empty or non-Gemini values stay on the legacy path."""
        assert uses_speech_metadata(model) is False

    def test_case_and_surrounding_whitespace_are_tolerated(self):
        """Mixed case and padding do not change the verdict."""
        assert uses_speech_metadata("  Gemini-3.8-Flash-TTS \n") is True
        assert uses_speech_metadata("  GEMINI-3.1-FLASH-TTS-PREVIEW ") is False
