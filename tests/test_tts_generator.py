"""
Tests for the tts_generator module.

The TTS preamble built by :func:`tts_podcast.tts_generator._build_tts_prompt`
reads ``gemini_cfg["speakerN"]["personality"]`` verbatim.  The new
``style_overlay`` key introduced for the dialogue prompt MUST NOT leak into
this preamble — voice acting and dialogue-content steering are intentionally
separate concerns.  This file pins that invariant with a regression test.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tts_podcast.llm_summarizer import DialogueChunk
from tts_podcast.tts_generator import (
    _DEFAULT_TTS_TEMPERATURE,
    _build_tts_prompt,
    generate_audio_chunks,
)


_BASE_CFG = {
    "tts_model": "gemini-2.5-flash-preview-tts",
    "language": "French",
    "speaker1": {
        "name": "Alex",
        "voice": "Puck",
        "personality": "calm and curious",
    },
    "speaker2": {
        "name": "Jordan",
        "voice": "Charon",
        "personality": "measured and analytical",
    },
}


class TestTtsPreambleInvariant:
    """The TTS preamble must NEVER reflect dialogue-side overlays."""

    def test_tts_preamble_unaffected_by_speaker_overlay(self):
        cfg = {
            **_BASE_CFG,
            "speaker1": {
                **_BASE_CFG["speaker1"],
                # The overlay key exists for the dialogue prompt only; the TTS
                # path must continue to read `personality` verbatim.
                "style_overlay": "extremely angry, shouting throughout",
            },
        }
        prompt = _build_tts_prompt("Alex: Bonjour.\nJordan: Salut.", cfg)
        # The baseline personality string reaches the preamble verbatim.
        assert "Alex is calm and curious." in prompt
        # The overlay text MUST NOT leak into the preamble.
        assert "extremely angry" not in prompt
        assert "shouting" not in prompt

    def test_tts_preamble_reads_baseline_personality(self):
        prompt = _build_tts_prompt("Alex: Hi.\nJordan: Hi.", _BASE_CFG)
        assert "Alex is calm and curious." in prompt
        assert "Jordan is measured and analytical." in prompt

    def test_tts_preamble_falls_back_to_defaults_when_personality_absent(self):
        cfg = {
            **_BASE_CFG,
            "speaker1": {"name": "Alex", "voice": "Puck"},  # no personality
            "speaker2": {"name": "Jordan", "voice": "Charon"},
        }
        prompt = _build_tts_prompt("Alex: Hi.\nJordan: Hi.", cfg)
        # Defaults defined in _build_tts_prompt
        assert "Alex is enthusiastic and curious." in prompt
        assert "Jordan is analytical and thoughtful." in prompt


class TestTtsPreambleSteadiness:
    """
    The preamble must steer the TTS model away from the per-chunk energy
    escalation (the "sawtooth" where a host ramps up and stops breathing as
    the chunk goes on).  These assertions pin the director's-note wording that
    counteracts it.
    """

    def test_preamble_requests_steady_energy_and_breathing(self):
        prompt = _build_tts_prompt("Alex: Hi.\nJordan: Hi.", _BASE_CFG)
        # Steady energy: no build-up / crescendo across the chunk.
        assert "steady" in prompt
        assert "crescendo" in prompt
        # Explicit breathing / pausing instruction.
        assert "Breathe" in prompt
        # Guardrail against the exact reported symptom.
        assert "frantic" in prompt


def _fake_tts_response(pcm: bytes = b"pcm-bytes"):
    """
    Build a MagicMock shaped like a Gemini TTS ``generate_content`` response.

    The audio bytes live at
    ``response.candidates[0].content.parts[0].inline_data.data`` and
    ``usage_metadata`` is present so token accounting stays happy.

    Parameters
    ----------
    pcm : bytes, optional
        Raw PCM payload to expose on the fake response.

    Returns
    -------
    unittest.mock.MagicMock
        A response mock compatible with ``_generate_chunk_audio``.
    """
    response = MagicMock()
    response.candidates[0].content.parts[0].inline_data.data = pcm
    response.usage_metadata = None
    return response


class TestTtsTemperature:
    """The TTS sampling temperature defaults low and honours config overrides."""

    def test_default_temperature_is_calmer_than_one(self):
        # A high temperature (1.0) is what let prosody drift toward
        # over-expression; the default must stay meaningfully below it.
        assert _DEFAULT_TTS_TEMPERATURE < 1.0

    def test_default_temperature_reaches_the_api(self):
        cfg = {**_BASE_CFG, "api_key": "test-key"}  # no tts_style.temperature
        with patch("tts_podcast.tts_generator.genai.Client") as client_cls:
            client = client_cls.return_value
            client.models.generate_content.return_value = _fake_tts_response()
            generate_audio_chunks([DialogueChunk(text="Alex: Hi.", index=0)], cfg)

        config = client.models.generate_content.call_args.kwargs["config"]
        assert config.temperature == _DEFAULT_TTS_TEMPERATURE

    def test_config_temperature_override_wins(self):
        cfg = {
            **_BASE_CFG,
            "api_key": "test-key",
            "tts_style": {"temperature": 0.35},
        }
        with patch("tts_podcast.tts_generator.genai.Client") as client_cls:
            client = client_cls.return_value
            client.models.generate_content.return_value = _fake_tts_response()
            generate_audio_chunks([DialogueChunk(text="Alex: Hi.", index=0)], cfg)

        config = client.models.generate_content.call_args.kwargs["config"]
        assert config.temperature == 0.35
