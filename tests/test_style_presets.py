"""
Tests for the style_presets module.

Verifies the curated preset list shape, validate_preset() resolution and error
behaviour, and the truncate_with_warning() helper.
"""

from __future__ import annotations

import logging

import click
import pytest

from tts_podcast.style_presets import (
    STYLE_PRESETS,
    truncate_with_warning,
    validate_preset,
)


class TestStylePresets:
    """The curated preset list is a stable contract."""

    def test_exact_five_presets(self):
        """STYLE_PRESETS must contain exactly the documented five keys."""
        assert set(STYLE_PRESETS.keys()) == {
            "casual",
            "academic",
            "humorous",
            "debate",
            "vulgarized",
        }

    def test_all_fragments_non_empty(self):
        """Every preset must ship with a non-empty prompt fragment."""
        for key, fragment in STYLE_PRESETS.items():
            assert fragment.strip(), f"Preset {key!r} has empty fragment"


class TestValidatePreset:
    """validate_preset must resolve known keys, ignore disables, error on typos."""

    def test_returns_none_for_none(self):
        assert validate_preset(None) is None

    def test_returns_none_for_empty_string(self):
        assert validate_preset("") is None

    def test_returns_none_for_whitespace(self):
        assert validate_preset("   ") is None

    def test_returns_none_for_sentinel_none(self):
        assert validate_preset("none") is None

    def test_sentinel_is_case_insensitive(self):
        assert validate_preset("NONE") is None
        assert validate_preset("None") is None

    def test_returns_fragment_for_known_preset(self):
        fragment = validate_preset("academic")
        assert fragment is not None
        assert fragment == STYLE_PRESETS["academic"]

    def test_lowercases_known_preset(self):
        assert validate_preset("ACADEMIC") == STYLE_PRESETS["academic"]

    def test_raises_bad_parameter_for_unknown(self):
        with pytest.raises(click.BadParameter) as exc_info:
            validate_preset("nosuchpreset")
        msg = str(exc_info.value)
        for key in STYLE_PRESETS:
            assert key in msg, f"Error message missing valid preset {key!r}"


class TestTruncateWithWarning:
    """truncate_with_warning enforces a per-field length cap and warns on hit."""

    def test_returns_none_for_none(self):
        assert truncate_with_warning(None, "style") is None

    def test_returns_empty_for_empty(self):
        assert truncate_with_warning("", "style") == ""

    def test_returns_text_unchanged_when_short(self):
        assert truncate_with_warning("short text", "style") == "short text"

    def test_returns_text_unchanged_at_cap(self):
        text = "a" * 500
        assert truncate_with_warning(text, "style") == text

    def test_truncates_when_over_cap(self, caplog):
        text = "x" * 600
        with caplog.at_level(logging.WARNING, logger="tts_podcast.style_presets"):
            result = truncate_with_warning(text, "style")
        assert result == "x" * 500
        assert len(result) == 500

    def test_warning_names_field(self, caplog):
        with caplog.at_level(logging.WARNING, logger="tts_podcast.style_presets"):
            truncate_with_warning("y" * 700, "angle")
        assert any("angle" in rec.message for rec in caplog.records)

    def test_custom_cap_respected(self):
        result = truncate_with_warning("a" * 50, "style", cap=20)
        assert result == "a" * 20
