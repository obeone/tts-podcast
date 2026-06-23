"""
Unit tests for :mod:`tts_podcast.duo_generator`.

All Gemini API calls are mocked — no network access.

Coverage targets
----------------
* Happy path: returned voices ∈ GEMINI_VOICES; personalities non-empty;
  description non-empty.
* Token tracker: ``record_usage`` is called with the correct model id.
* Structured-output fast path: ``response.parsed`` (newer SDK) is preferred
  over ``response.text``.
* JSON fallback path: ``response.parsed is None`` → ``json.loads(response.text)``.
* Voice validation: RuntimeError on voice not in GEMINI_VOICES (schema bypass).
* Empty response: RuntimeError on blank ``response.text``.
* Non-JSON response: RuntimeError on garbage text.
* No tracker: passing ``token_tracker=None`` is safe (no AttributeError).
* Service tier: ``http_options`` header is set when ``service_tier`` is present;
  absent when not set.
* Retry decorator wired: ``@gemini_retry`` wraps the inner API call.
* Prompt content: source titles and research notes appear in the user prompt.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tts_podcast.duo_generator import _build_prompt, generate_duo
from tts_podcast.duos import GEMINI_VOICES
from tts_podcast.models import Source


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


def _fake_gemini_cfg(service_tier: str | None = None) -> dict[str, Any]:
    """Build a minimal gemini config dict."""
    cfg: dict[str, Any] = {
        "api_key": "fake-api-key",
        "text_model": "gemini-2.5-flash",
    }
    if service_tier is not None:
        cfg["service_tier"] = service_tier
    return cfg


def _valid_voice() -> str:
    """Return the first voice in GEMINI_VOICES (deterministic)."""
    return next(iter(GEMINI_VOICES))


def _second_valid_voice() -> str:
    """Return the second voice in GEMINI_VOICES (different from the first)."""
    it = iter(GEMINI_VOICES)
    next(it)
    return next(it)


def _make_response(
    speaker1_voice: str | None = None,
    speaker2_voice: str | None = None,
    use_parsed: bool = True,
    text_override: str | None = None,
) -> MagicMock:
    """
    Build a mock Gemini response object.

    Parameters
    ----------
    speaker1_voice:
        Voice name for speaker1; defaults to a valid voice.
    speaker2_voice:
        Voice name for speaker2; defaults to a different valid voice.
    use_parsed:
        When True, ``response.parsed`` is set (fast path).
        When False, ``response.parsed is None`` and ``response.text`` is used.
    text_override:
        When given, sets ``response.text`` to this value regardless of
        ``use_parsed``; allows injecting invalid JSON or empty strings.
    """
    v1 = speaker1_voice or _valid_voice()
    v2 = speaker2_voice or _second_valid_voice()

    duo_dict = {
        "description": "A calm analytical duo.",
        "speaker1": {"name": "Alex", "voice": v1, "personality": "calm and precise"},
        "speaker2": {"name": "Jordan", "voice": v2, "personality": "warm and curious"},
    }

    response = MagicMock()
    response.usage_metadata = MagicMock()

    if text_override is not None:
        response.text = text_override
        response.parsed = None
    elif use_parsed:
        response.parsed = duo_dict
        response.text = json.dumps(duo_dict)
    else:
        response.parsed = None
        response.text = json.dumps(duo_dict)

    return response


# ---------------------------------------------------------------------------
# Fixture: patched genai.Client
# ---------------------------------------------------------------------------

@pytest.fixture()
def patched_client(request):
    """
    Yield a factory (response_or_fn) -> MagicMock(client).

    Patches ``genai.Client`` so that ``client.models.generate_content``
    returns the given response (or calls the given callable).
    """
    # Indirect parameterization not needed here; simple fixture returning factory.
    with patch("tts_podcast.duo_generator.genai.Client") as MockClient:
        client_instance = MagicMock()
        MockClient.return_value = client_instance
        yield client_instance, MockClient


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestGenerateDuoHappyPath:
    """Core behaviour on a well-formed Gemini response."""

    def test_returns_dict_with_expected_keys(self, patched_client):
        client_instance, _ = patched_client
        client_instance.models.generate_content.return_value = _make_response()

        duo = generate_duo([_fake_source()], "", _fake_gemini_cfg())

        assert set(duo.keys()) == {"description", "speaker1", "speaker2"}

    def test_voices_in_gemini_voices(self, patched_client):
        client_instance, _ = patched_client
        client_instance.models.generate_content.return_value = _make_response()

        duo = generate_duo([_fake_source()], "", _fake_gemini_cfg())

        assert duo["speaker1"]["voice"] in GEMINI_VOICES
        assert duo["speaker2"]["voice"] in GEMINI_VOICES

    def test_personalities_non_empty(self, patched_client):
        client_instance, _ = patched_client
        client_instance.models.generate_content.return_value = _make_response()

        duo = generate_duo([_fake_source()], "", _fake_gemini_cfg())

        assert duo["speaker1"]["personality"].strip()
        assert duo["speaker2"]["personality"].strip()

    def test_description_non_empty(self, patched_client):
        client_instance, _ = patched_client
        client_instance.models.generate_content.return_value = _make_response()

        duo = generate_duo([_fake_source()], "", _fake_gemini_cfg())

        assert duo["description"].strip()

    def test_speaker_names_present(self, patched_client):
        client_instance, _ = patched_client
        client_instance.models.generate_content.return_value = _make_response()

        duo = generate_duo([_fake_source()], "", _fake_gemini_cfg())

        assert duo["speaker1"]["name"]
        assert duo["speaker2"]["name"]

    def test_json_fallback_path(self, patched_client):
        """When response.parsed is None, json.loads(response.text) is used."""
        client_instance, _ = patched_client
        client_instance.models.generate_content.return_value = _make_response(
            use_parsed=False
        )

        duo = generate_duo([_fake_source()], "", _fake_gemini_cfg())

        assert duo["speaker1"]["voice"] in GEMINI_VOICES


# ---------------------------------------------------------------------------
# Token tracker tests
# ---------------------------------------------------------------------------

class TestTokenTracker:
    """token_tracker.record_usage is called correctly."""

    def test_record_usage_called_with_correct_model(self, patched_client):
        client_instance, _ = patched_client
        response = _make_response()
        client_instance.models.generate_content.return_value = response

        tracker = MagicMock()
        cfg = _fake_gemini_cfg()
        generate_duo([_fake_source()], "", cfg, tracker)

        tracker.record_usage.assert_called_once_with(
            cfg["text_model"], response.usage_metadata
        )

    def test_no_tracker_does_not_raise(self, patched_client):
        """Passing token_tracker=None skips tracking silently."""
        client_instance, _ = patched_client
        client_instance.models.generate_content.return_value = _make_response()

        # Should not raise AttributeError or anything else.
        generate_duo([_fake_source()], "", _fake_gemini_cfg(), token_tracker=None)


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------

class TestGenerateDuoErrors:
    """RuntimeError / BadParameter raised on malformed Gemini responses."""

    def test_voice_not_in_gemini_voices_raises_runtime_error(self, patched_client):
        """Voice returned by SDK but not in GEMINI_VOICES → RuntimeError."""
        client_instance, _ = patched_client
        response = _make_response(speaker1_voice="HallucinatedVoice")
        # Force the parsed path to bypass schema enum (simulate old SDK).
        response.parsed = {
            "description": "desc",
            "speaker1": {"name": "X", "voice": "HallucinatedVoice", "personality": "x"},
            "speaker2": {
                "name": "Y",
                "voice": _second_valid_voice(),
                "personality": "y",
            },
        }
        client_instance.models.generate_content.return_value = response

        with pytest.raises(RuntimeError, match="HallucinatedVoice"):
            generate_duo([_fake_source()], "", _fake_gemini_cfg())

    def test_empty_response_text_raises_runtime_error(self, patched_client):
        """Blank response.text with no parsed object → RuntimeError."""
        client_instance, _ = patched_client
        client_instance.models.generate_content.return_value = _make_response(
            text_override=""
        )

        with pytest.raises(RuntimeError, match="empty response"):
            generate_duo([_fake_source()], "", _fake_gemini_cfg())

    def test_non_json_response_raises_runtime_error(self, patched_client):
        """Garbage text from SDK (not JSON) → RuntimeError."""
        client_instance, _ = patched_client
        client_instance.models.generate_content.return_value = _make_response(
            text_override="Sorry, I cannot do that."
        )

        with pytest.raises(RuntimeError, match="non-JSON"):
            generate_duo([_fake_source()], "", _fake_gemini_cfg())

    def test_missing_voice_field_raises_bad_parameter(self, patched_client):
        """Speaker block missing 'voice' key → click.BadParameter from _validate_speaker."""
        import click

        client_instance, _ = patched_client
        response = MagicMock()
        response.usage_metadata = MagicMock()
        response.parsed = {
            "description": "desc",
            "speaker1": {"name": "Alex", "personality": "calm"},  # no voice
            "speaker2": {
                "name": "Jordan",
                "voice": _second_valid_voice(),
                "personality": "warm",
            },
        }
        response.text = json.dumps(response.parsed)
        client_instance.models.generate_content.return_value = response

        with pytest.raises(click.BadParameter):
            generate_duo([_fake_source()], "", _fake_gemini_cfg())


# ---------------------------------------------------------------------------
# Service tier tests
# ---------------------------------------------------------------------------

class TestServiceTier:
    """http_options header is set when service_tier is configured."""

    def test_service_tier_header_passed(self, patched_client):
        """When service_tier is set, http_options header is included in config."""
        client_instance, _ = patched_client
        client_instance.models.generate_content.return_value = _make_response()

        generate_duo(
            [_fake_source()],
            "",
            _fake_gemini_cfg(service_tier="dynamic"),
        )

        # generate_content is called as generate_content(model=..., contents=..., config=...)
        _, kwargs = client_instance.models.generate_content.call_args
        config = kwargs["config"]
        # Verify http_options header contains service tier.
        assert config.http_options is not None
        assert config.http_options.headers["x-goog-api-service-tier"] == "dynamic"

    def test_no_service_tier_no_http_options(self, patched_client):
        """When service_tier is absent, http_options is not set on the config."""
        client_instance, _ = patched_client
        client_instance.models.generate_content.return_value = _make_response()

        generate_duo([_fake_source()], "", _fake_gemini_cfg())

        _, kwargs = client_instance.models.generate_content.call_args
        config = kwargs["config"]
        # When no service_tier, the config should have no http_options key.
        assert not hasattr(config, "http_options") or config.http_options is None


# ---------------------------------------------------------------------------
# Retry wiring tests
# ---------------------------------------------------------------------------

class TestRetryWiring:
    """@gemini_retry is active on the inner API call."""

    def test_gemini_retry_retries_on_server_error(self, patched_client):
        """A ServerError on first call is retried; second call succeeds."""
        from google.genai import errors as genai_errors

        client_instance, _ = patched_client
        success_response = _make_response()
        server_error = genai_errors.ServerError("503 Service Unavailable", {"error": {}})

        client_instance.models.generate_content.side_effect = [
            server_error,
            success_response,
        ]

        # gemini_retry uses tenacity which calls tenacity.nap.sleep internally.
        with patch("tenacity.nap.sleep"):
            duo = generate_duo([_fake_source()], "", _fake_gemini_cfg())

        assert duo["speaker1"]["voice"] in GEMINI_VOICES
        assert client_instance.models.generate_content.call_count == 2

    def test_client_error_not_retried(self, patched_client):
        """4xx (non-ServerError) exceptions propagate immediately without retry."""
        from google.genai import errors as genai_errors

        client_instance, _ = patched_client
        # Use ClientError (4xx) — not retried by @gemini_retry.
        client_error = genai_errors.ClientError(
            "400 Bad Request",
            {"error": {"status": "INVALID_ARGUMENT"}},
        )
        client_instance.models.generate_content.side_effect = client_error

        with pytest.raises(genai_errors.ClientError):
            generate_duo([_fake_source()], "", _fake_gemini_cfg())

        # Should have been called exactly once — no retry.
        assert client_instance.models.generate_content.call_count == 1


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
