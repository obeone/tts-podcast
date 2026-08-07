"""
Tests for the tts_generator module.

The TTS preamble built by :func:`tts_podcast.tts_generator._build_tts_prompt`
reads ``gemini_cfg["speakerN"]["personality"]`` verbatim.  The
``style_overlay`` key introduced for the dialogue prompt MUST NOT leak into
this preamble — voice acting and dialogue-content steering are intentionally
separate concerns.  The ``voice_direction`` key is the mirror image: it belongs
to this preamble only and must never reach the dialogue prompt.  This file pins
both invariants, the byte-identical rendering for configs that declare no
``voice_direction``, and the preamble byte budget.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tts_podcast.duos import BUILTIN_DUOS
from tts_podcast.llm_summarizer import _MAX_CHUNK_BYTES, DialogueChunk
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

#: The preamble ``_build_tts_prompt`` produced for ``_BASE_CFG`` *before*
#: ``voice_direction`` existed, captured from the pre-change implementation.
#: A config that declares no ``voice_direction`` must still render exactly
#: this, byte for byte: existing users' episodes must not change how they
#: sound just because the feature landed.  Editing the director's notes on
#: purpose means updating this constant in the same commit, deliberately.
_LEGACY_PREAMBLE = (
    "Audio profile: Two hosts of a French tech podcast, speaking in French.\n"
    "Alex is calm and curious.\n"
    "Jordan is measured and analytical.\n"
    "Director's notes: Conversational pace — natural; speak clearly, allow a natural "
    "beat between sentences so the listener can absorb each idea. Keep each host's "
    "energy level steady from the first line to the last: do not build up, speed up, "
    "or crescendo as the passage goes on. Breathe — leave a clear pause at every "
    "sentence boundary and never run turns together. React genuinely and stay in "
    "character; an enthusiastic host stays warmly enthusiastic, not increasingly "
    "frantic. Honour any emotional cues written in parentheses in the dialogue.\n\n"
)


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


class TestVoiceDirectionInPreamble:
    """
    ``voice_direction`` is the whole point of the duo rework: without it both
    hosts are rendered with the same delivery, which is what made every duo
    sound alike.  It must reach the preamble and be attributed by name.
    """

    _CFG = {
        **_BASE_CFG,
        "speaker1": {
            **_BASE_CFG["speaker1"],
            "voice_direction": "Low chest register, slow, legato, deep breaths.",
        },
        "speaker2": {
            **_BASE_CFG["speaker2"],
            "voice_direction": "Bright high register, fast staccato, clipped endings.",
        },
    }

    def test_each_direction_is_attributed_to_its_own_host(self):
        prompt = _build_tts_prompt("Alex: Bonjour.\nJordan: Salut.", self._CFG)
        assert (
            "Voice direction for Alex: Low chest register, slow, legato, deep breaths."
            in prompt
        )
        assert (
            "Voice direction for Jordan: Bright high register, fast staccato, "
            "clipped endings." in prompt
        )

    def test_direction_follows_its_own_personality_line(self):
        # Order matters: an unattributed or misplaced note lets the model
        # average both deliveries back into one voice.
        lines = _build_tts_prompt("", self._CFG).splitlines()
        assert lines[1] == "Alex is calm and curious."
        assert lines[2].startswith("Voice direction for Alex: ")
        assert lines[3] == "Jordan is measured and analytical."
        assert lines[4].startswith("Voice direction for Jordan: ")

    def test_only_the_declaring_speaker_gets_a_direction(self):
        cfg = {
            **_BASE_CFG,
            "speaker1": {
                **_BASE_CFG["speaker1"],
                "voice_direction": "Low chest register, slow.",
            },
        }
        prompt = _build_tts_prompt("", cfg)
        assert "Voice direction for Alex: Low chest register, slow." in prompt
        assert "Voice direction for Jordan" not in prompt

    def test_personality_is_still_read_verbatim_alongside_a_direction(self):
        prompt = _build_tts_prompt("", self._CFG)
        assert "Alex is calm and curious." in prompt
        assert "Jordan is measured and analytical." in prompt

    @pytest.mark.parametrize("slug", sorted(BUILTIN_DUOS))
    def test_builtin_duo_directions_render(self, slug):
        duo = BUILTIN_DUOS[slug]
        cfg = {
            "tts_model": "gemini-2.5-flash-preview-tts",
            "language": "French",
            "speaker1": duo["speaker1"],
            "speaker2": duo["speaker2"],
            "tts_style": {"scene": duo["scene"], "pace": duo["pace"]},
        }
        prompt = _build_tts_prompt("", cfg)
        for role in ("speaker1", "speaker2"):
            speaker = duo[role]
            expected = f"Voice direction for {speaker['name']}: {speaker['voice_direction']}"
            assert expected in prompt
        assert f"Scene: {duo['scene']}" in prompt
        assert f"Conversational pace — {duo['pace']};" in prompt


class TestLegacyPreambleUnchanged:
    """
    A config with no ``voice_direction`` must render byte for byte what it
    rendered before the key existed.
    """

    def test_no_direction_renders_the_legacy_preamble(self):
        assert _build_tts_prompt("", _BASE_CFG) == _LEGACY_PREAMBLE

    def test_dialogue_text_is_appended_unchanged(self):
        chunk = "Alex: Bonjour.\nJordan: Salut."
        assert _build_tts_prompt(chunk, _BASE_CFG) == _LEGACY_PREAMBLE + chunk

    @pytest.mark.parametrize("empty", [None, "", "   ", "\n"])
    def test_blank_direction_is_indistinguishable_from_absent(self, empty):
        cfg = {
            **_BASE_CFG,
            "speaker1": {**_BASE_CFG["speaker1"], "voice_direction": empty},
            "speaker2": {**_BASE_CFG["speaker2"], "voice_direction": empty},
        }
        assert _build_tts_prompt("", cfg) == _LEGACY_PREAMBLE

    @pytest.mark.parametrize("bad", [42, 1.5, ["low", "slow"], {"register": "low"}])
    def test_non_string_direction_does_not_crash_the_tts_thread_pool(self, bad):
        # The CLI rejects this up front (duos.validate_speaker), but a caller
        # assembling the config in Python must never get an AttributeError
        # raised inside a worker thread, after the dialogue has been billed.
        cfg = {**_BASE_CFG, "speaker1": {**_BASE_CFG["speaker1"], "voice_direction": bad}}
        prompt = _build_tts_prompt("", cfg)
        assert f"Voice direction for Alex: {bad}" in prompt


class TestPreambleByteBudget:
    """
    Gemini TTS caps the request text around 4000 bytes.  Every request is
    ``preamble + chunk``, and the chunker bounds the chunk at
    ``llm_summarizer._MAX_CHUNK_BYTES``, so the two together are the budget.
    Growing the preamble (longer directions, a longer scene) without lowering
    the chunk bound is how requests start failing mid-episode.
    """

    #: Hard ceiling imposed by the TTS API, in bytes.
    TTS_TEXT_LIMIT = 4000
    #: Safety margin required on top of the ceiling, in bytes.
    REQUIRED_HEADROOM = 200

    @staticmethod
    def _preamble_bytes(cfg: dict) -> int:
        """
        Measure the rendered preamble for *cfg*, in UTF-8 bytes.

        Parameters
        ----------
        cfg : dict
            A resolved ``gemini`` config section.

        Returns
        -------
        int
            Length of the preamble alone (the dialogue chunk is empty).
        """
        return len(_build_tts_prompt("", cfg).encode("utf-8"))

    @pytest.mark.parametrize("slug", sorted(BUILTIN_DUOS))
    @pytest.mark.parametrize("language", ["French", "Brazilian Portuguese"])
    def test_builtin_duo_fits_the_budget(self, slug, language):
        duo = BUILTIN_DUOS[slug]
        cfg = {
            "tts_model": "gemini-2.5-flash-preview-tts",
            "language": language,
            "speaker1": duo["speaker1"],
            "speaker2": duo["speaker2"],
            "tts_style": {"scene": duo["scene"], "pace": duo["pace"]},
        }
        total = _MAX_CHUNK_BYTES + self._preamble_bytes(cfg)
        assert total <= self.TTS_TEXT_LIMIT - self.REQUIRED_HEADROOM, (
            f"Duo {slug!r} in {language}: {total} bytes worst case, over the "
            f"{self.TTS_TEXT_LIMIT - self.REQUIRED_HEADROOM}-byte working ceiling."
        )

    @staticmethod
    def _envelope_cfg(language: str, pace: str) -> dict:
        """
        Build the synthetic maximal config the byte budget was sized against.

        Not a real duo: every field is pushed to the documented ceiling (two
        160-char voice directions, a 200-char scene, 11-char host names,
        125-char personalities) so the result bounds any duo a user may write.

        Parameters
        ----------
        language : str
            Dialogue language name, which appears twice in the preamble header.
        pace : str
            The ``tts_style.pace`` string.

        Returns
        -------
        dict
            A resolved ``gemini`` config section.
        """
        return {
            "tts_model": "gemini-2.5-flash-preview-tts",
            "language": language,
            "speaker1": {
                "name": "N" * 11,
                "voice": "Puck",
                "personality": "p" * 125,
                "voice_direction": "d" * 160,
            },
            "speaker2": {
                "name": "M" * 11,
                "voice": "Kore",
                "personality": "q" * 125,
                "voice_direction": "e" * 160,
            },
            "tts_style": {"scene": "s" * 200, "pace": pace},
        }

    def test_documented_envelope_keeps_the_required_headroom(self):
        # The envelope _MAX_CHUNK_BYTES was actually sized against: a short
        # language name and a ~25-char pace.
        cfg = self._envelope_cfg("French", "unhurried and slow")
        total = _MAX_CHUNK_BYTES + self._preamble_bytes(cfg)
        assert total <= self.TTS_TEXT_LIMIT - self.REQUIRED_HEADROOM, (
            f"Documented envelope: {total} bytes. Lower _MAX_CHUNK_BYTES in "
            "llm_summarizer.py and update its comment plus CLAUDE.md."
        )

    def test_stretched_envelope_keeps_the_required_headroom(self):
        # Same envelope stretched on the two axes the documented figure did
        # not cover: a long language name (the preamble header prints it twice)
        # and a 50-char pace, which the built-in duos already reach.  This is
        # the real worst case a user-authored duo can hit while staying inside
        # the documented per-field maxima, so it carries the same headroom
        # guarantee rather than merely staying under the hard API ceiling.
        cfg = self._envelope_cfg(
            "Brazilian Portuguese", "quick when something clicks, slow when it does not"
        )
        total = _MAX_CHUNK_BYTES + self._preamble_bytes(cfg)
        assert total <= self.TTS_TEXT_LIMIT - self.REQUIRED_HEADROOM, (
            f"Stretched envelope: {total} bytes, over the "
            f"{self.TTS_TEXT_LIMIT - self.REQUIRED_HEADROOM}-byte working ceiling. "
            "Lower _MAX_CHUNK_BYTES in llm_summarizer.py and update its comment "
            "plus CLAUDE.md."
        )


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
