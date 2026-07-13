"""
Unit tests for :mod:`tts_podcast.duo_generator`.

The provider-agnostic ``complete()`` seam (:mod:`tts_podcast.llm_client`) is
mocked — no network access.

Coverage targets
----------------
* Happy path: returned voices ∈ GEMINI_VOICES; personalities non-empty;
  description non-empty.
* Token tracker: ``record`` is called with the bare model id and token counts.
* Voice validation: RuntimeError on voice not in GEMINI_VOICES (schema bypass).
* Empty response: RuntimeError on blank result text.
* Non-JSON response: RuntimeError on garbage text.
* No tracker: passing ``token_tracker=None`` is safe (no AttributeError).
* Extra headers: ``llm_cfg.extra_headers`` (e.g. the service-tier shim) reach
  ``complete()`` verbatim; ``None`` when unset.
* Prompt content: source titles and research notes appear in the user prompt.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tts_podcast.duo_generator import _build_prompt, generate_duo
from tts_podcast.duos import GEMINI_VOICES
from tts_podcast.llm_client import LlmResult
from tts_podcast.models import Source
from tts_podcast.settings import LlmSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_source(
    title: str = "Article about something",
    summary: str = "A quick summary.",
    full_text: str = "The full text of the article.",
    url: str = "https://example.com/art",
) -> Source:
    """Build a minimal Source for testing."""
    return Source(
        url=url,
        title=title,
        summary=summary,
        full_text=full_text,
        scraped_ok=True,
        kind="url",
    )


def _fake_gemini_cfg() -> dict[str, Any]:
    """Build a minimal gemini config dict (unused for auth/model selection now)."""
    return {}


LLM_SETTINGS = LlmSettings(
    provider="gemini",
    text_model="gemini-2.5-flash",
    research_model=None,
    api_key="fake-api-key",
    api_base=None,
    temperature=None,
    extra_headers=None,
)


def _valid_voice() -> str:
    """Return the first voice in GEMINI_VOICES (deterministic)."""
    return next(iter(GEMINI_VOICES))


def _second_valid_voice() -> str:
    """Return the second voice in GEMINI_VOICES (different from the first)."""
    it = iter(GEMINI_VOICES)
    next(it)
    return next(it)


def _make_llm_result(
    speaker1_voice: str | None = None,
    speaker2_voice: str | None = None,
    text_override: str | None = None,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> LlmResult:
    """
    Build a neutral :class:`~tts_podcast.llm_client.LlmResult` for mocking ``complete``.

    Parameters
    ----------
    speaker1_voice:
        Voice name for speaker1; defaults to a valid voice.
    speaker2_voice:
        Voice name for speaker2; defaults to a different valid voice.
    text_override:
        When given, used verbatim as the result text (allows injecting
        invalid JSON or an empty string); otherwise a valid duo JSON blob
        is generated from *speaker1_voice* / *speaker2_voice*.
    input_tokens, output_tokens:
        Token counts to report.
    """
    if text_override is not None:
        text = text_override
    else:
        v1 = speaker1_voice or _valid_voice()
        v2 = speaker2_voice or _second_valid_voice()
        duo_dict = {
            "description": "A calm analytical duo.",
            "speaker1": {"name": "Alex", "voice": v1, "personality": "calm and precise"},
            "speaker2": {"name": "Jordan", "voice": v2, "personality": "warm and curious"},
        }
        text = json.dumps(duo_dict)

    return LlmResult(text=text, input_tokens=input_tokens, output_tokens=output_tokens, grounding=None)


# ---------------------------------------------------------------------------
# Fixture: patched complete()
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_complete():
    """Yield the mock for ``tts_podcast.duo_generator.complete``."""
    with patch("tts_podcast.duo_generator.complete") as m:
        yield m


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestGenerateDuoHappyPath:
    """Core behaviour on a well-formed structured-output result."""

    def test_returns_dict_with_expected_keys(self, mock_complete):
        mock_complete.return_value = _make_llm_result()

        duo = generate_duo([_fake_source()], "", _fake_gemini_cfg(), LLM_SETTINGS)

        assert set(duo.keys()) == {"description", "speaker1", "speaker2"}

    def test_voices_in_gemini_voices(self, mock_complete):
        mock_complete.return_value = _make_llm_result()

        duo = generate_duo([_fake_source()], "", _fake_gemini_cfg(), LLM_SETTINGS)

        assert duo["speaker1"]["voice"] in GEMINI_VOICES
        assert duo["speaker2"]["voice"] in GEMINI_VOICES

    def test_personalities_non_empty(self, mock_complete):
        mock_complete.return_value = _make_llm_result()

        duo = generate_duo([_fake_source()], "", _fake_gemini_cfg(), LLM_SETTINGS)

        assert duo["speaker1"]["personality"].strip()
        assert duo["speaker2"]["personality"].strip()

    def test_description_non_empty(self, mock_complete):
        mock_complete.return_value = _make_llm_result()

        duo = generate_duo([_fake_source()], "", _fake_gemini_cfg(), LLM_SETTINGS)

        assert duo["description"].strip()

    def test_speaker_names_present(self, mock_complete):
        mock_complete.return_value = _make_llm_result()

        duo = generate_duo([_fake_source()], "", _fake_gemini_cfg(), LLM_SETTINGS)

        assert duo["speaker1"]["name"]
        assert duo["speaker2"]["name"]


# ---------------------------------------------------------------------------
# Token tracker tests
# ---------------------------------------------------------------------------

class TestTokenTracker:
    """token_tracker.record is called with the bare model id and token counts."""

    def test_record_called_with_correct_model(self, mock_complete):
        mock_complete.return_value = _make_llm_result(input_tokens=111, output_tokens=22)

        tracker = MagicMock()
        generate_duo([_fake_source()], "", _fake_gemini_cfg(), LLM_SETTINGS, tracker)

        tracker.record.assert_called_once_with(LLM_SETTINGS.text_model, 111, 22)

    def test_no_tracker_does_not_raise(self, mock_complete):
        """Passing token_tracker=None skips tracking silently."""
        mock_complete.return_value = _make_llm_result()

        # Should not raise AttributeError or anything else.
        generate_duo(
            [_fake_source()], "", _fake_gemini_cfg(), LLM_SETTINGS, token_tracker=None
        )


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------

class TestGenerateDuoErrors:
    """RuntimeError / BadParameter raised on malformed structured-output results."""

    def test_voice_not_in_gemini_voices_raises_runtime_error(self, mock_complete):
        """Voice returned by the model but not in GEMINI_VOICES → RuntimeError."""
        duo_dict = {
            "description": "desc",
            "speaker1": {"name": "X", "voice": "HallucinatedVoice", "personality": "x"},
            "speaker2": {
                "name": "Y",
                "voice": _second_valid_voice(),
                "personality": "y",
            },
        }
        mock_complete.return_value = LlmResult(text=json.dumps(duo_dict))

        with pytest.raises(RuntimeError, match="HallucinatedVoice"):
            generate_duo([_fake_source()], "", _fake_gemini_cfg(), LLM_SETTINGS)

    def test_empty_response_text_raises_runtime_error(self, mock_complete):
        """Blank result text → RuntimeError."""
        mock_complete.return_value = _make_llm_result(text_override="")

        with pytest.raises(RuntimeError, match="empty response"):
            generate_duo([_fake_source()], "", _fake_gemini_cfg(), LLM_SETTINGS)

    def test_non_json_response_raises_runtime_error(self, mock_complete):
        """Garbage text (not JSON) → RuntimeError."""
        mock_complete.return_value = _make_llm_result(
            text_override="Sorry, I cannot do that."
        )

        with pytest.raises(RuntimeError, match="non-JSON"):
            generate_duo([_fake_source()], "", _fake_gemini_cfg(), LLM_SETTINGS)

    def test_missing_voice_field_raises_bad_parameter(self, mock_complete):
        """Speaker block missing 'voice' key → click.BadParameter from _validate_speaker."""
        import click

        duo_dict = {
            "description": "desc",
            "speaker1": {"name": "Alex", "personality": "calm"},  # no voice
            "speaker2": {
                "name": "Jordan",
                "voice": _second_valid_voice(),
                "personality": "warm",
            },
        }
        mock_complete.return_value = LlmResult(text=json.dumps(duo_dict))

        with pytest.raises(click.BadParameter):
            generate_duo([_fake_source()], "", _fake_gemini_cfg(), LLM_SETTINGS)


# ---------------------------------------------------------------------------
# Extra-headers passthrough tests
# ---------------------------------------------------------------------------

class TestExtraHeaders:
    """llm_cfg.extra_headers (e.g. the service-tier shim) reach complete() verbatim."""

    def test_extra_headers_passed_through(self, mock_complete):
        """When llm_cfg.extra_headers is set, it is forwarded to complete()."""
        mock_complete.return_value = _make_llm_result()
        llm_cfg = replace(
            LLM_SETTINGS, extra_headers={"x-goog-api-service-tier": "dynamic"}
        )

        generate_duo([_fake_source()], "", _fake_gemini_cfg(), llm_cfg)

        _, kwargs = mock_complete.call_args
        assert kwargs["extra_headers"] == {"x-goog-api-service-tier": "dynamic"}

    def test_no_extra_headers_passes_none(self, mock_complete):
        """When llm_cfg.extra_headers is unset, complete() receives None."""
        mock_complete.return_value = _make_llm_result()

        generate_duo([_fake_source()], "", _fake_gemini_cfg(), LLM_SETTINGS)

        _, kwargs = mock_complete.call_args
        assert kwargs["extra_headers"] is None


# ---------------------------------------------------------------------------
# Prompt content tests
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    """_build_prompt encodes the right content signals."""

    def test_source_title_in_prompt(self):
        src = _fake_source(title="Resignation letter analysis")
        prompt = _build_prompt([src], "", "French")
        assert "Resignation letter analysis" in prompt

    def test_research_notes_in_prompt(self):
        src = _fake_source()
        notes = "Key finding: the mood is grave and somber."
        prompt = _build_prompt([src], notes, "French")
        assert "grave and somber" in prompt

    def test_language_in_prompt(self):
        src = _fake_source()
        prompt = _build_prompt([src], "", "Japanese")
        assert "Japanese" in prompt

    def test_empty_research_notes_not_included(self):
        src = _fake_source()
        prompt = _build_prompt([src], "", "French")
        assert "Research notes" not in prompt

    def test_full_text_truncated_beyond_limit(self):
        """full_text longer than _MAX_FULL_TEXT_CHARS is truncated with ellipsis."""
        from tts_podcast.duo_generator import _MAX_FULL_TEXT_CHARS

        long_text = "A" * (_MAX_FULL_TEXT_CHARS + 500)
        src = _fake_source(full_text=long_text)
        prompt = _build_prompt([src], "", "French")
        assert "A" * _MAX_FULL_TEXT_CHARS in prompt
        assert "…" in prompt

    def test_beyond_max_sources_no_full_text(self):
        """Sources beyond _MAX_SOURCES_FULL do not include full_text excerpts."""
        from tts_podcast.duo_generator import _MAX_SOURCES_FULL

        sources = [
            _fake_source(
                title=f"Source {i}",
                full_text=f"UNIQUE_FULLTEXT_{i}",
                url=f"https://example.com/s{i}",
            )
            for i in range(_MAX_SOURCES_FULL + 2)
        ]
        prompt = _build_prompt(sources, "", "French")
        # Full text for sources within the limit should appear.
        assert "UNIQUE_FULLTEXT_0" in prompt
        # Full text beyond the limit should NOT appear.
        beyond_idx = _MAX_SOURCES_FULL + 1
        assert f"UNIQUE_FULLTEXT_{beyond_idx}" not in prompt
