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

import logging
from unittest.mock import MagicMock, patch

import pytest

from tts_podcast.duos import BUILTIN_DUOS
from tts_podcast.llm_summarizer import (
    _MAX_CHUNK_BYTES,
    _MIN_CHUNK_BYTES,
    _TTS_PREAMBLE_HEADROOM,
    _TTS_TEXT_LIMIT,
    DialogueChunk,
    _resolve_chunk_budget,
)
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


def _budget_cfg(
    *,
    language: str = "French",
    scene: str | None = None,
    pace: str | None = None,
    name_chars: int = 11,
    personality_chars: int = 125,
    direction_chars: int | None = 160,
) -> dict:
    """
    Build a synthetic config whose preamble size is controlled field by field.

    Not a real duo: every field is a filler string of an exact length, so a
    test can dial the rendered preamble up or down on one axis at a time.

    Parameters
    ----------
    language : str, optional
        Dialogue language name.  The preamble header prints it twice, so it is
        a real axis of the byte budget, not a cosmetic detail.
    scene : str or None, optional
        Literal ``tts_style.scene`` value.  ``None`` omits the key entirely.
    pace : str or None, optional
        Literal ``tts_style.pace`` value.  ``None`` omits the key, which makes
        the preamble fall back to ``"natural"``.
    name_chars : int, optional
        Length of each host name.
    personality_chars : int, optional
        Length of each ``personality`` string.
    direction_chars : int or None, optional
        Length of each ``voice_direction``.  ``None`` omits the key, which is
        the legacy shape that predates the feature.

    Returns
    -------
    dict
        A resolved ``gemini`` config section.
    """
    speakers = {}
    for role, name_char, personality_char, voice in (
        ("speaker1", "N", "p", "Puck"),
        ("speaker2", "M", "q", "Kore"),
    ):
        block: dict = {
            "name": name_char * name_chars,
            "voice": voice,
            "personality": personality_char * personality_chars,
        }
        if direction_chars is not None:
            block["voice_direction"] = "d" * direction_chars
        speakers[role] = block

    tts_style: dict = {}
    if scene is not None:
        tts_style["scene"] = scene
    if pace is not None:
        tts_style["pace"] = pace

    cfg: dict = {
        "tts_model": "gemini-2.5-flash-preview-tts",
        "language": language,
        **speakers,
    }
    if tts_style:
        cfg["tts_style"] = tts_style
    return cfg


class TestPreambleByteBudget:
    """
    Gemini TTS caps the request text around 4000 bytes, and every request is
    ``preamble + chunk``.

    The chunk bound is no longer a static constant sized against a worst-case
    preamble: ``llm_summarizer._resolve_chunk_budget`` measures the preamble
    the *active* config actually renders and hands the remainder to the
    chunker.  So the property to pin is not "one constant plus the worst
    preamble fits", it is "for any config, that config's own budget plus its
    own preamble fits".  A static bound would be wrong in both directions: too
    small for the configs that declare nothing (extra requests, extra audio
    splice points, the preamble re-billed on each), too large the day someone
    ships a longer scene.
    """

    #: Hard ceiling imposed by the TTS API, in bytes.  Stated here
    #: independently of the source constant on purpose: these tests are the
    #: statement of the guarantee, not a mirror of the implementation.
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

    def _assert_request_fits(self, cfg: dict, label: str) -> int:
        """
        Assert that *cfg*'s resolved budget leaves the required headroom.

        Parameters
        ----------
        cfg : dict
            A resolved ``gemini`` config section.
        label : str
            Human-readable identifier used in the failure message.

        Returns
        -------
        int
            The resolved budget, so callers can make further assertions on it.
        """
        budget = _resolve_chunk_budget(cfg)
        preamble = self._preamble_bytes(cfg)
        total = budget + preamble
        ceiling = self.TTS_TEXT_LIMIT - self.REQUIRED_HEADROOM
        assert total <= ceiling, (
            f"{label}: budget {budget} + preamble {preamble} = {total} bytes, over "
            f"the {ceiling}-byte working ceiling.  Either the preamble grew or "
            "_resolve_chunk_budget stopped accounting for it."
        )
        return budget

    # -- 1. every built-in duo, in every language -------------------------

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
        budget = self._assert_request_fits(cfg, f"Duo {slug!r} in {language}")
        # A built-in duo must never be so verbose that it hits the floor: that
        # would mean shipping a duo that shreds every episode into tiny chunks.
        assert budget > _MIN_CHUNK_BYTES, (
            f"Duo {slug!r} in {language} is verbose enough to hit the chunk floor."
        )

    @pytest.mark.parametrize("slug", sorted(BUILTIN_DUOS))
    @pytest.mark.parametrize("language", ["French", "Brazilian Portuguese"])
    def test_builtin_duo_resolves_without_warning(self, slug, language, caplog):
        duo = BUILTIN_DUOS[slug]
        cfg = {
            "tts_model": "gemini-2.5-flash-preview-tts",
            "language": language,
            "speaker1": duo["speaker1"],
            "speaker2": duo["speaker2"],
            "tts_style": {"scene": duo["scene"], "pace": duo["pace"]},
        }
        with caplog.at_level(logging.WARNING, logger="tts_podcast.llm_summarizer"):
            _resolve_chunk_budget(cfg)
        assert caplog.records == [], (
            f"Duo {slug!r} in {language} triggered an over-budget warning."
        )

    # -- 2. synthetic envelopes, including the pathological one -----------

    def test_documented_envelope_keeps_the_required_headroom(self):
        # The envelope the original static bound was sized against: a short
        # language name and a ~18-char pace.
        cfg = _budget_cfg(language="French", scene="s" * 200, pace="unhurried and slow")
        self._assert_request_fits(cfg, "Documented envelope")

    def test_stretched_envelope_keeps_the_required_headroom(self):
        # Same envelope stretched on the two axes the documented figure did
        # not cover: a long language name (the preamble header prints it twice)
        # and a 50-char pace, which the built-in duos already reach.  This is
        # the real worst case a user-authored duo can hit while staying inside
        # the documented per-field maxima.
        cfg = _budget_cfg(
            language="Brazilian Portuguese",
            scene="s" * 200,
            pace="quick when something clicks, slow when it does not",
        )
        budget = self._assert_request_fits(cfg, "Stretched envelope")
        # The stretched envelope must still leave a usable chunk, otherwise the
        # documented per-field maxima are themselves over budget.
        assert budget > _MIN_CHUNK_BYTES

    def test_oversized_preamble_clamps_to_the_floor_and_warns(self, caplog):
        # Well past every documented maximum: the preamble alone eats most of
        # the request, so no positive budget can keep the headroom.  The
        # contract there is not "it fits" (it cannot) but "it degrades loudly":
        # clamp to the floor and say which fields to shorten, at chunking time,
        # rather than raise a non-retryable 4xx from a TTS worker thread after
        # the dialogue has already been generated and billed.
        cfg = _budget_cfg(
            language="Brazilian Portuguese",
            scene="s" * 500,
            pace="p" * 120,
            personality_chars=300,
            direction_chars=400,
        )
        preamble = self._preamble_bytes(cfg)
        assert preamble > self.TTS_TEXT_LIMIT - self.REQUIRED_HEADROOM - _MIN_CHUNK_BYTES, (
            "Fixture is no longer pathological; make it bigger."
        )

        with caplog.at_level(logging.WARNING, logger="tts_podcast.llm_summarizer"):
            budget = _resolve_chunk_budget(cfg)

        assert budget == _MIN_CHUNK_BYTES
        messages = [record.getMessage() for record in caplog.records]
        assert len(messages) == 1, f"Expected exactly one warning, got {messages!r}"
        warning = messages[0]
        assert str(preamble) in warning, (
            "The warning must name the measured preamble size, otherwise the user "
            "cannot tell how far over budget they are."
        )
        # It must also name the fields that are actionable.
        assert "scene" in warning
        assert "voice_direction" in warning
        assert "personality" in warning

    # -- 3. regression guard: legacy configs chunk exactly as before ------

    def test_legacy_config_resolves_to_the_pre_change_bound(self):
        # The point of resolving the budget instead of hard-coding a worst
        # case.  _BASE_CFG declares no voice_direction, no scene and no pace,
        # and TestLegacyPreambleUnchanged pins that it renders byte for byte
        # the pre-feature preamble.  It must therefore also chunk byte for
        # byte as before: same chunk count, same number of TTS requests, same
        # number of splice points between independently generated audio
        # segments (this project already fought per-chunk prosody drift, so
        # every extra seam is an audio-quality cost, not just latency).
        assert _MAX_CHUNK_BYTES == 3000, (
            "3000 is the pre-voice_direction chunk bound.  Changing it changes "
            "how every legacy config is chunked."
        )
        assert _resolve_chunk_budget(_BASE_CFG) == _MAX_CHUNK_BYTES

    def test_a_legacy_config_above_the_line_does_lose_budget(self):
        # The counterexample class, pinned so the guarantee above is read with
        # its condition attached.  The condition is a byte threshold, not "the
        # config declares no voice_direction": the pre-change code spent the
        # whole 3000 bytes on the chunk and reserved no headroom at all, so a
        # config that never had a voice_direction, but whose preamble already
        # sat above (_TTS_TEXT_LIMIT - _TTS_PREAMBLE_HEADROOM -
        # _MAX_CHUNK_BYTES), gives up the difference even though that preamble
        # is byte for byte what it always rendered.  125-char personalities and
        # 11-char names are inside the documented per-field maxima, so this is
        # an ordinary config, not a pathological one.
        cfg = _budget_cfg(direction_chars=None)
        assert "voice_direction" not in cfg["speaker1"]
        assert "voice_direction" not in cfg["speaker2"]

        threshold = self.TTS_TEXT_LIMIT - self.REQUIRED_HEADROOM - _MAX_CHUNK_BYTES
        preamble = self._preamble_bytes(cfg)
        assert preamble > threshold, (
            f"Fixture no longer sits above the {threshold}-byte line "
            f"(preamble {preamble}); it cannot demonstrate the counterexample."
        )

        budget = _resolve_chunk_budget(cfg)
        assert budget < _MAX_CHUNK_BYTES
        # Still a mild loss, not a cliff: worth documenting, not worth alarm.
        assert budget == self.TTS_TEXT_LIMIT - self.REQUIRED_HEADROOM - preamble
        assert budget > _MIN_CHUNK_BYTES

    def test_legacy_config_resolves_without_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="tts_podcast.llm_summarizer"):
            _resolve_chunk_budget(_BASE_CFG)
        assert caplog.records == []

    @pytest.mark.parametrize("empty", [None, "", "   "])
    def test_blank_direction_resolves_like_no_direction(self, empty):
        # Blank values render nothing (TestLegacyPreambleUnchanged), so they
        # must not cost a single byte of budget either.
        cfg = {
            **_BASE_CFG,
            "speaker1": {**_BASE_CFG["speaker1"], "voice_direction": empty},
            "speaker2": {**_BASE_CFG["speaker2"], "voice_direction": empty},
        }
        assert _resolve_chunk_budget(cfg) == _MAX_CHUNK_BYTES

    def test_adding_a_direction_costs_budget(self):
        # The complement of the guard above: a config that *does* declare
        # directions must resolve strictly lower, otherwise the budget is not
        # actually being measured.
        directed = {
            **_BASE_CFG,
            "speaker1": {**_BASE_CFG["speaker1"], "voice_direction": "d" * 160},
            "speaker2": {**_BASE_CFG["speaker2"], "voice_direction": "e" * 160},
            "tts_style": {"scene": "s" * 200},
        }
        assert _resolve_chunk_budget(directed) < _resolve_chunk_budget(_BASE_CFG)

    # -- 4. the clamp holds at both ends, monotonically in between --------

    def test_budget_decreases_monotonically_as_the_preamble_grows(self):
        # Property check over a sweep of scene lengths: the budget must never
        # go *up* when the preamble grows, must stay inside the clamp at every
        # point, and must keep the required headroom for as long as the floor
        # is not engaged.
        # Lean everywhere except the scene, so the sweep starts on the upper
        # clamp and the scene is the only variable.
        scene_lengths = [0, 40, 200, 400, 800, 1200, 1800, 2400, 3200]
        cfgs = [
            _budget_cfg(
                scene="s" * n,
                name_chars=4,
                personality_chars=20,
                direction_chars=None,
            )
            for n in scene_lengths
        ]
        preambles = [self._preamble_bytes(cfg) for cfg in cfgs]
        budgets = [_resolve_chunk_budget(cfg) for cfg in cfgs]

        # The sweep must actually span both clamps, otherwise it proves little.
        assert budgets[0] == _MAX_CHUNK_BYTES
        assert budgets[-1] == _MIN_CHUNK_BYTES

        for i in range(1, len(budgets)):
            assert preambles[i] > preambles[i - 1], (
                f"scene={scene_lengths[i]} did not grow the preamble; the sweep "
                "is not exercising what it claims to."
            )
            assert budgets[i] <= budgets[i - 1], (
                f"Budget rose from {budgets[i - 1]} to {budgets[i]} while the "
                f"preamble grew from {preambles[i - 1]} to {preambles[i]} bytes."
            )

        for length, preamble, budget in zip(scene_lengths, preambles, budgets, strict=True):
            assert _MIN_CHUNK_BYTES <= budget <= _MAX_CHUNK_BYTES, (
                f"scene={length}: budget {budget} outside the clamp."
            )
            if budget > _MIN_CHUNK_BYTES:
                # Floor not engaged, so the headroom guarantee applies.
                assert budget + preamble <= self.TTS_TEXT_LIMIT - self.REQUIRED_HEADROOM

    def test_budget_is_exact_between_the_clamps(self):
        # Between the two clamps the budget is whatever is left of the request
        # after the preamble and the safety margin — no rounding, no slack
        # silently added or dropped.
        cfg = _budget_cfg(scene="s" * 400)
        preamble = self._preamble_bytes(cfg)
        expected = self.TTS_TEXT_LIMIT - self.REQUIRED_HEADROOM - preamble
        assert _MIN_CHUNK_BYTES < expected < _MAX_CHUNK_BYTES, (
            "Fixture no longer lands between the clamps; adjust the scene length."
        )
        assert _resolve_chunk_budget(cfg) == expected

    def test_module_constants_agree_with_the_stated_guarantee(self):
        # The class constants above are deliberately independent of the source
        # ones.  This is the single place the two are tied together, so
        # loosening the safety margin in llm_summarizer.py fails here instead
        # of silently weakening every assertion in this class.
        assert _TTS_TEXT_LIMIT == self.TTS_TEXT_LIMIT
        assert _TTS_PREAMBLE_HEADROOM >= self.REQUIRED_HEADROOM
        assert _MIN_CHUNK_BYTES < _MAX_CHUNK_BYTES

    def test_sparse_config_still_yields_a_budget(self):
        # generate_dialogue accepts configs with no speaker blocks at all
        # (names arrive as arguments), while _build_tts_prompt indexes them.
        # Resolving the budget must not turn that into a KeyError.
        assert (
            _resolve_chunk_budget({"tts_model": "gemini-2.5-flash-preview-tts"})
            == _MAX_CHUNK_BYTES
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
