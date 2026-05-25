"""
Tests for the tts_generator module.

The TTS preamble built by :func:`tts_podcast.tts_generator._build_tts_prompt`
reads ``gemini_cfg["speakerN"]["personality"]`` verbatim.  The new
``style_overlay`` key introduced for the dialogue prompt MUST NOT leak into
this preamble — voice acting and dialogue-content steering are intentionally
separate concerns.  This file pins that invariant with a regression test.
"""

from __future__ import annotations

from tts_podcast.tts_generator import _build_tts_prompt


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
