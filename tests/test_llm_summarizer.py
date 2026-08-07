"""
Tests for the llm_summarizer module.

Verifies dialogue generation, chunking, and research-notes injection
behaviour with a mocked Gemini client.
"""

from __future__ import annotations

import logging
import inspect
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tts_podcast.duos import BUILTIN_DUOS
from tts_podcast.llm_summarizer import (
    _MAX_CHUNK_BYTES,
    DialogueChunk,
    _audio_tags_enabled,
    _build_prompt,
    _build_thinking_config,
    _has_speaker_turns,
    _resolve_chunk_budget,
    _split_dialogue_into_chunks,
    generate_dialogue,
)
from tts_podcast.style_presets import STYLE_PRESETS
from tts_podcast.tts_generator import _build_tts_prompt


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeArticle:
    """Minimal article stub for testing."""

    title: str
    url: str
    summary: str
    full_text: str = ""


GEMINI_CFG = {
    "api_key": "test-api-key",
    "text_model": "gemini-2.5-flash",
    "tts_model": "gemini-2.5-flash-preview-tts",
    "speaker1": {"name": "Alex", "voice": "Puck"},
    "speaker2": {"name": "Jordan", "voice": "Charon"},
}

SAMPLE_ARTICLES = [
    FakeArticle(
        title="Rust hits 1.0 stability milestone",
        url="https://example.com/rust",
        summary="Rust language announces major stability improvements.",
        full_text="Rust language announces major stability improvements in version 1.0.",
    ),
]

SHORT_DIALOGUE = """\
Alex: Hey Jordan, ready to dive into today's article?
Jordan: Absolutely! What caught your eye?
Alex: There's a fascinating piece about Rust hitting stability milestones.
Jordan: Oh interesting! Tell me more.
Alex: The language team says performance improved by 40 percent.
Jordan: That's huge for systems programming.
"""


def _mock_genai_response(text: str):
    """
    Build a mock genai module whose Client.models.generate_content returns text.

    Parameters
    ----------
    text : str
        The dialogue text the mock should return.

    Returns
    -------
    MagicMock
        A mock that mimics the genai module interface.
    """
    mock_response = MagicMock()
    mock_response.text = text

    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    mock_client_instance = MagicMock()
    mock_client_instance.models = mock_model

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client_instance

    return mock_genai


def _captured_prompt(mock_genai) -> str:
    """Return the prompt string that was sent to the mocked Gemini client."""
    call = mock_genai.Client.return_value.models.generate_content.call_args
    return call.kwargs.get("contents") or call.args[1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateDialogue:
    """Unit tests for generate_dialogue()."""

    def test_returns_non_empty_list_of_chunks(self):
        """generate_dialogue returns at least one DialogueChunk on success."""
        mock_genai = _mock_genai_response(SHORT_DIALOGUE)

        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            chunks = generate_dialogue(SAMPLE_ARTICLES, GEMINI_CFG, "Alex", "Jordan")

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, DialogueChunk) for c in chunks)

    def test_chunks_contain_text(self):
        """Every returned DialogueChunk has non-empty text."""
        mock_genai = _mock_genai_response(SHORT_DIALOGUE)

        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            chunks = generate_dialogue(SAMPLE_ARTICLES, GEMINI_CFG, "Alex", "Jordan")

        for chunk in chunks:
            assert chunk.text.strip() != ""

    def test_chunks_have_sequential_indices(self):
        """DialogueChunk objects are indexed sequentially from 0."""
        mock_genai = _mock_genai_response(SHORT_DIALOGUE)

        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            chunks = generate_dialogue(SAMPLE_ARTICLES, GEMINI_CFG, "Alex", "Jordan")

        for expected_index, chunk in enumerate(chunks):
            assert chunk.index == expected_index

    def test_genai_client_called_with_correct_model(self):
        """Gemini client is called with the model specified in gemini_cfg."""
        mock_genai = _mock_genai_response(SHORT_DIALOGUE)

        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            generate_dialogue(SAMPLE_ARTICLES, GEMINI_CFG, "Alex", "Jordan")

        mock_genai.Client.assert_called_once_with(api_key="test-api-key")
        call_kwargs = mock_genai.Client.return_value.models.generate_content.call_args
        assert call_kwargs.kwargs.get("model") == "gemini-2.5-flash"

    def test_passes_max_output_tokens(self):
        """generate_dialogue must pass max_output_tokens=8192 to the Gemini API."""
        mock_genai = _mock_genai_response(SHORT_DIALOGUE)

        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            generate_dialogue(SAMPLE_ARTICLES, GEMINI_CFG, "Alex", "Jordan")

        call_kwargs = mock_genai.Client.return_value.models.generate_content.call_args.kwargs
        config_obj = call_kwargs.get("config")
        assert config_obj is not None
        assert config_obj.max_output_tokens == 8192


class TestDurationConfig:
    """generate_dialogue must translate duration config into the prompt."""

    def test_default_duration_appears_in_prompt(self):
        """With no dialogue config, defaults (8 min target, 150 wpm) show up."""
        mock_genai = _mock_genai_response(SHORT_DIALOGUE)

        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            generate_dialogue(SAMPLE_ARTICLES, GEMINI_CFG, "Alex", "Jordan")

        prompt = _captured_prompt(mock_genai)
        # Target word count = 8 * 150 = 1200
        assert "1200" in prompt
        # Duration label
        assert "8 minutes" in prompt or "8 min" in prompt

    def test_custom_target_duration_propagated(self):
        """Explicit target_duration_minutes drives target word count."""
        cfg = {
            **GEMINI_CFG,
            "dialogue": {"target_duration_minutes": 12, "words_per_minute": 150},
        }
        mock_genai = _mock_genai_response(SHORT_DIALOGUE)

        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            generate_dialogue(SAMPLE_ARTICLES, cfg, "Alex", "Jordan")

        prompt = _captured_prompt(mock_genai)
        # 12 * 150 = 1800
        assert "1800" in prompt
        # Default bounds: 70% (8.4 min → ~1260 words) and 150% (18 min → 2700 words)
        assert "2700" in prompt
        assert "1260" in prompt

    def test_explicit_min_max_overrides_defaults(self):
        """min/max_duration_minutes from config drive the bounds verbatim."""
        cfg = {
            **GEMINI_CFG,
            "dialogue": {
                "target_duration_minutes": 10,
                "min_duration_minutes": 5,
                "max_duration_minutes": 20,
                "words_per_minute": 140,
            },
        }
        mock_genai = _mock_genai_response(SHORT_DIALOGUE)

        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            generate_dialogue(SAMPLE_ARTICLES, cfg, "Alex", "Jordan")

        prompt = _captured_prompt(mock_genai)
        # 5 * 140 = 700, 10 * 140 = 1400, 20 * 140 = 2800
        assert "700" in prompt
        assert "1400" in prompt
        assert "2800" in prompt
        assert "140 wpm" in prompt


class TestResearchNotesInjection:
    """Verify generate_dialogue injects research notes into the prompt only when provided."""

    def test_no_research_section_when_notes_empty(self):
        """No 'Complementary research' header appears when research_notes is empty."""
        mock_genai = _mock_genai_response(SHORT_DIALOGUE)

        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            generate_dialogue(SAMPLE_ARTICLES, GEMINI_CFG, "Alex", "Jordan")

        prompt = _captured_prompt(mock_genai)
        assert "Complementary research" not in prompt

    def test_research_notes_appear_in_prompt(self):
        """When research_notes is supplied, its text appears in the prompt before Articles."""
        notes = "### Research round 1\n\n- Background fact A (https://src/a)\n- Recent dev B (https://src/b)"
        mock_genai = _mock_genai_response(SHORT_DIALOGUE)

        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            generate_dialogue(
                SAMPLE_ARTICLES, GEMINI_CFG, "Alex", "Jordan",
                research_notes=notes,
            )

        prompt = _captured_prompt(mock_genai)
        assert "Complementary research" in prompt
        assert "Background fact A" in prompt
        assert "Recent dev B" in prompt
        assert prompt.index("Complementary research") < prompt.index("Articles:")

    def test_whitespace_only_notes_treated_as_empty(self):
        """A whitespace-only research_notes string is treated as no-research."""
        mock_genai = _mock_genai_response(SHORT_DIALOGUE)

        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            generate_dialogue(
                SAMPLE_ARTICLES, GEMINI_CFG, "Alex", "Jordan",
                research_notes="   \n\t  ",
            )

        prompt = _captured_prompt(mock_genai)
        assert "Complementary research" not in prompt


class TestAudioTagsEnabled:
    """Unit tests for the _audio_tags_enabled helper."""

    def test_auto_detects_gemini_3_tts_model(self):
        cfg = {"tts_model": "gemini-3.1-flash-preview-tts"}
        assert _audio_tags_enabled(cfg) is True

    def test_auto_rejects_gemini_2_5_tts_model(self):
        cfg = {"tts_model": "gemini-2.5-flash-preview-tts"}
        assert _audio_tags_enabled(cfg) is False

    def test_explicit_on_overrides_unsupported_model(self):
        cfg = {
            "tts_model": "gemini-2.5-flash-preview-tts",
            "tts_style": {"audio_tags": "on"},
        }
        assert _audio_tags_enabled(cfg) is True

    def test_explicit_off_overrides_supported_model(self):
        cfg = {
            "tts_model": "gemini-3.1-flash-preview-tts",
            "tts_style": {"audio_tags": "off"},
        }
        assert _audio_tags_enabled(cfg) is False

    def test_missing_tts_model_defaults_off(self):
        assert _audio_tags_enabled({}) is False


# ---------------------------------------------------------------------------
# Style / overlay / angle injections
# ---------------------------------------------------------------------------


def _build_prompt_default_kwargs(**overrides):
    """Return _build_prompt kwargs matching the snapshot fixture inputs."""
    base = {
        "articles": SAMPLE_ARTICLES,
        "speaker1_name": "Alex",
        "speaker2_name": "Jordan",
        "speaker1_personality": "enthusiastic and curious",
        "speaker2_personality": "analytical and thoughtful",
        "min_minutes": 6.0,
        "target_minutes": 8.0,
        "max_minutes": 14.0,
        "words_per_minute": 150,
        "language": "French",
        "audio_tags": False,
        "research_notes": "",
    }
    base.update(overrides)
    return base


class TestStyleInjections:
    """Preset and free-text style guidance render inside Instructions."""

    def test_preset_injected(self):
        prompt = _build_prompt(**_build_prompt_default_kwargs(preset="academic"))
        assert "Stylistic guidance:" in prompt
        assert STYLE_PRESETS["academic"].strip() in prompt

    def test_style_free_text_injected(self):
        prompt = _build_prompt(
            **_build_prompt_default_kwargs(style_text="extra rigorous, dry tone")
        )
        assert "Stylistic guidance:" in prompt
        assert "extra rigorous, dry tone" in prompt

    def test_preset_plus_style_compose(self):
        prompt = _build_prompt(
            **_build_prompt_default_kwargs(
                preset="academic",
                style_text="but extra dry",
            )
        )
        assert "Stylistic guidance:" in prompt
        # Preset fragment first, free text after.
        preset_pos = prompt.index(STYLE_PRESETS["academic"].strip())
        text_pos = prompt.index("but extra dry")
        assert preset_pos < text_pos

    def test_no_style_means_no_header(self):
        prompt = _build_prompt(**_build_prompt_default_kwargs())
        assert "Stylistic guidance:" not in prompt


class TestSpeakerOverlay:
    """Per-speaker overlays render in a dedicated block, never mutate personality."""

    def test_speaker_overlay_in_dedicated_block(self):
        prompt = _build_prompt(
            **_build_prompt_default_kwargs(speaker1_overlay="more skeptical than usual")
        )
        assert "Episode-specific adjustments:" in prompt
        assert "- Alex: more skeptical than usual" in prompt
        # Overlay text must NOT be inlined into the Host personalities bullet.
        host_block_end = prompt.index("Episode-specific adjustments:")
        host_block = prompt[: host_block_end]
        assert "more skeptical than usual" not in host_block

    def test_both_overlays_listed(self):
        prompt = _build_prompt(
            **_build_prompt_default_kwargs(
                speaker1_overlay="X for Alex",
                speaker2_overlay="Y for Jordan",
            )
        )
        assert "- Alex: X for Alex" in prompt
        assert "- Jordan: Y for Jordan" in prompt

    def test_only_one_overlay_renders_one_bullet(self):
        prompt = _build_prompt(
            **_build_prompt_default_kwargs(speaker2_overlay="only Jordan")
        )
        assert "Episode-specific adjustments:" in prompt
        assert "- Jordan: only Jordan" in prompt
        assert "- Alex:" not in prompt.split("Episode-specific adjustments:")[1].split(
            "Instructions:"
        )[0]

    def test_speaker_overlay_does_not_mutate_personality(self):
        """generate_dialogue must NEVER write to gemini_cfg['speakerN']['personality']."""
        cfg = {
            **GEMINI_CFG,
            "speaker1": {**GEMINI_CFG["speaker1"], "personality": "original P1", "style_overlay": "overlay P1"},
            "speaker2": {**GEMINI_CFG["speaker2"], "personality": "original P2"},
        }
        original_p1 = cfg["speaker1"]["personality"]
        original_p2 = cfg["speaker2"]["personality"]
        mock_genai = _mock_genai_response(SHORT_DIALOGUE)

        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            generate_dialogue(SAMPLE_ARTICLES, cfg, "Alex", "Jordan")

        assert cfg["speaker1"]["personality"] == original_p1
        assert cfg["speaker2"]["personality"] == original_p2


class TestAngleInjection:
    """Angle text reaches the dialogue prompt regardless of research presence."""

    def test_angle_in_dialogue_prompt(self):
        prompt = _build_prompt(
            **_build_prompt_default_kwargs(angle="the economic implications")
        )
        assert "Episode angle: the economic implications" in prompt

    def test_angle_in_dialogue_prompt_without_research(self):
        prompt = _build_prompt(
            **_build_prompt_default_kwargs(
                angle="regulatory bite",
                research_notes="",
            )
        )
        assert "Episode angle: regulatory bite" in prompt
        assert "Complementary research" not in prompt

    def test_no_angle_means_no_header(self):
        prompt = _build_prompt(**_build_prompt_default_kwargs())
        assert "Episode angle:" not in prompt


@pytest.mark.parametrize(
    "field,kwarg",
    [
        ("style", "style_text"),
        ("speaker1-style", "speaker1_overlay"),
        ("speaker2-style", "speaker2_overlay"),
        ("angle", "angle"),
    ],
)
class TestTruncationWarningPerField:
    """600-char input is truncated to 500 with the field name in the warning."""

    def test_truncation_emits_warning_named_by_field(self, caplog, field, kwarg):
        long_text = "a" * 600
        with caplog.at_level(logging.WARNING, logger="tts_podcast.style_presets"):
            _build_prompt(**_build_prompt_default_kwargs(**{kwarg: long_text}))
        matching = [rec for rec in caplog.records if field in rec.message]
        assert matching, f"No warning mentioning field {field!r} in {caplog.records}"

    def test_truncated_value_reaches_prompt(self, caplog, field, kwarg):
        long_text = "a" * 600
        with caplog.at_level(logging.WARNING, logger="tts_podcast.style_presets"):
            prompt = _build_prompt(**_build_prompt_default_kwargs(**{kwarg: long_text}))
        # 500 a's must appear; 600 a's must not.
        assert "a" * 500 in prompt
        assert "a" * 600 not in prompt


class TestPromptSectionOrder:
    """All four injection points render in the documented order."""

    def test_order_when_all_options_set(self):
        prompt = _build_prompt(
            **_build_prompt_default_kwargs(
                preset="academic",
                style_text="extra dry",
                speaker1_overlay="overlay1",
                speaker2_overlay="overlay2",
                angle="big picture",
            )
        )
        positions = {
            "Host personalities:": prompt.index("Host personalities:"),
            "Episode-specific adjustments:": prompt.index("Episode-specific adjustments:"),
            "Instructions:": prompt.index("Instructions:"),
            "tone bullet": prompt.index("Keep the tone informative"),
            "Stylistic guidance:": prompt.index("Stylistic guidance:"),
            "Episode angle:": prompt.index("Episode angle:"),
            "Articles:": prompt.index("Articles:"),
        }
        # Top-level order
        assert positions["Host personalities:"] < positions["Episode-specific adjustments:"]
        assert positions["Episode-specific adjustments:"] < positions["Instructions:"]
        assert positions["Instructions:"] < positions["Articles:"]
        # Inside Instructions: tone bullet → Stylistic guidance → Episode angle
        assert positions["Instructions:"] < positions["tone bullet"]
        assert positions["tone bullet"] < positions["Stylistic guidance:"]
        assert positions["Stylistic guidance:"] < positions["Episode angle:"]
        assert positions["Episode angle:"] < positions["Articles:"]

    def test_no_block_header_double_emitted(self):
        prompt = _build_prompt(
            **_build_prompt_default_kwargs(
                preset="academic",
                style_text="extra dry",
                speaker1_overlay="x",
                speaker2_overlay="y",
                angle="z",
            )
        )
        for header in (
            "Stylistic guidance:",
            "Episode-specific adjustments:",
            "Host personalities:",
            "Instructions:",
            "Articles:",
        ):
            assert prompt.count(header) == 1, f"Header {header!r} appears multiple times"


class TestResearchDirective:
    """Research directive bullet appears iff research_notes is non-empty."""

    def test_research_directive_present_when_notes_provided(self):
        """When research_notes is non-empty, directive bullet appears in prompt."""
        notes = "- Key finding A\n- Key finding B"
        prompt = _build_prompt(**_build_prompt_default_kwargs(research_notes=notes))
        assert "MUST incorporate substantively" in prompt

    def test_research_directive_absent_when_notes_empty(self):
        """When research_notes is empty, directive bullet is absent."""
        prompt = _build_prompt(**_build_prompt_default_kwargs(research_notes=""))
        assert "MUST incorporate substantively" not in prompt

    def test_research_directive_absent_when_notes_whitespace(self):
        """When research_notes is whitespace-only, directive bullet is absent."""
        prompt = _build_prompt(**_build_prompt_default_kwargs(research_notes="   \n\t  "))
        assert "MUST incorporate substantively" not in prompt

    def test_research_directive_in_instructions_block(self):
        """Directive bullet appears inside the Instructions block, before Articles."""
        notes = "- Key finding"
        prompt = _build_prompt(**_build_prompt_default_kwargs(research_notes=notes))
        instructions_pos = prompt.index("Instructions:")
        directive_pos = prompt.index("MUST incorporate substantively")
        articles_pos = prompt.index("Articles:")
        assert instructions_pos < directive_pos < articles_pos


class TestVoiceDirectionIsolation:
    """
    Mirror of the ``style_overlay`` invariant.

    ``style_overlay`` steers what the hosts say and is dialogue-prompt-only;
    ``voice_direction`` steers how they sound and is TTS-only.  A voice
    direction leaking into the dialogue prompt would make the text model write
    stage directions about register and breathing into the script itself.
    """

    _DIRECTION_1 = "ZZDIRECTIONONE low chest register, slow, legato, deep breaths"
    _DIRECTION_2 = "ZZDIRECTIONTWO bright high register, fast staccato, clipped"

    def _cfg_with_directions(self) -> dict:
        """
        Return a Gemini config whose two speakers both carry a voice direction.

        Returns
        -------
        dict
            A copy of ``GEMINI_CFG`` with ``voice_direction`` on both speakers.
        """
        return {
            **GEMINI_CFG,
            "speaker1": {**GEMINI_CFG["speaker1"], "voice_direction": self._DIRECTION_1},
            "speaker2": {**GEMINI_CFG["speaker2"], "voice_direction": self._DIRECTION_2},
        }

    def test_voice_direction_never_reaches_the_dialogue_prompt(self):
        mock_genai = _mock_genai_response(SHORT_DIALOGUE)
        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            generate_dialogue(SAMPLE_ARTICLES, self._cfg_with_directions(), "Alex", "Jordan")

        prompt = _captured_prompt(mock_genai)
        assert self._DIRECTION_1 not in prompt
        assert self._DIRECTION_2 not in prompt
        assert "voice direction" not in prompt.lower()

    def test_dialogue_prompt_is_identical_with_and_without_directions(self):
        mock_plain = _mock_genai_response(SHORT_DIALOGUE)
        with patch("tts_podcast.llm_summarizer.genai", mock_plain):
            generate_dialogue(SAMPLE_ARTICLES, GEMINI_CFG, "Alex", "Jordan")

        mock_directed = _mock_genai_response(SHORT_DIALOGUE)
        with patch("tts_podcast.llm_summarizer.genai", mock_directed):
            generate_dialogue(SAMPLE_ARTICLES, self._cfg_with_directions(), "Alex", "Jordan")

        assert _captured_prompt(mock_directed) == _captured_prompt(mock_plain)

    @pytest.mark.parametrize("slug", sorted(BUILTIN_DUOS))
    def test_no_builtin_duo_direction_reaches_the_dialogue_prompt(self, slug):
        duo = BUILTIN_DUOS[slug]
        name1 = duo["speaker1"]["name"]
        name2 = duo["speaker2"]["name"]
        cfg = {**GEMINI_CFG, "speaker1": duo["speaker1"], "speaker2": duo["speaker2"]}
        # The response must carry this duo's own speaker turns, otherwise the
        # turn validation rejects it and retries instead of reaching the assert.
        dialogue = f"{name1}: Bonjour.\n{name2}: Salut.\n"
        mock_genai = _mock_genai_response(dialogue)
        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            generate_dialogue(SAMPLE_ARTICLES, cfg, name1, name2)

        prompt = _captured_prompt(mock_genai)
        for role in ("speaker1", "speaker2"):
            assert duo[role]["voice_direction"] not in prompt
            # The baseline personality is what the dialogue prompt *does* use.
            assert duo[role]["personality"] in prompt

    def test_dialogue_prompt_builders_never_mention_the_key(self):
        # Static guard: neither the prompt builder nor its caller may read
        # voice_direction, whatever the runtime path.
        assert "voice_direction" not in inspect.signature(_build_prompt).parameters
        assert "voice_direction" not in inspect.getsource(_build_prompt)
        assert "voice_direction" not in inspect.getsource(generate_dialogue)


class TestNoFlagsByteIdentical:
    """Backward-compat snapshot guarantee: defaults produce the frozen baseline."""

    def test_no_flags_byte_identical(self):
        fixture = Path(__file__).parent / "fixtures" / "dialogue_prompt_no_overlay.txt"
        expected = fixture.read_text(encoding="utf-8")
        # The fixture was generated with these specific articles — replicate.
        from tests.fixtures.regen_dialogue_prompt import _FIXTURE_ARTICLES
        got = _build_prompt(
            articles=_FIXTURE_ARTICLES,
            speaker1_name="Alex",
            speaker2_name="Jordan",
            speaker1_personality="enthusiastic and curious",
            speaker2_personality="analytical and thoughtful",
            min_minutes=6.0,
            target_minutes=8.0,
            max_minutes=14.0,
            words_per_minute=150,
            language="French",
            audio_tags=False,
            research_notes="",
        )
        assert got == expected, "Backward-compat regression: default prompt drifted from fixture."


# ---------------------------------------------------------------------------
# Tests for _build_thinking_config
# ---------------------------------------------------------------------------


class TestBuildThinkingConfig:
    """Unit tests for the _build_thinking_config helper."""

    def test_3x_model_valid_level_returns_thinking_config(self):
        """3.x model + valid thinking_level returns ThinkingConfig with that level."""
        result = _build_thinking_config("gemini-3.5-flash", "low", None)
        assert result is not None
        # The SDK converts the string to a ThinkingLevel enum; compare via .value.
        assert result.thinking_level.value == "LOW"

    def test_3x_model_level_normalised_lowercase(self):
        """thinking_level is accepted case-insensitively (SDK normalises to enum)."""
        result = _build_thinking_config("gemini-3.5-flash", "LOW", None)
        assert result is not None
        assert result.thinking_level.value == "LOW"

    def test_3x_model_invalid_level_returns_none(self):
        """3.x model + invalid thinking_level logs a warning and returns None."""
        result = _build_thinking_config("gemini-3.5-flash", "turbo", None)
        assert result is None

    def test_25_model_budget_zero_returns_thinking_config(self):
        """2.5 model + thinking_budget=0 returns ThinkingConfig with budget 0."""
        result = _build_thinking_config("gemini-2.5-flash", None, 0)
        assert result is not None
        assert result.thinking_budget == 0

    def test_25_model_budget_positive(self):
        """2.5 model + positive thinking_budget returns ThinkingConfig."""
        result = _build_thinking_config("gemini-2.5-flash", None, 1024)
        assert result is not None
        assert result.thinking_budget == 1024

    def test_nothing_set_returns_none(self):
        """Neither level nor budget set returns None for any model."""
        assert _build_thinking_config("gemini-3.5-flash", None, None) is None
        assert _build_thinking_config("gemini-2.5-flash", None, None) is None

    def test_3x_model_only_budget_set_returns_none(self):
        """3.x model + only thinking_budget set returns None (budget ignored)."""
        result = _build_thinking_config("gemini-3.5-flash", None, 512)
        assert result is None

    def test_25_model_only_level_set_returns_none(self):
        """Non-3.x model + only thinking_level set returns None (level ignored)."""
        result = _build_thinking_config("gemini-2.5-flash", "low", None)
        assert result is None

    def test_empty_string_level_treated_as_not_set(self):
        """Empty string thinking_level is treated as not set."""
        assert _build_thinking_config("gemini-3.5-flash", "", None) is None

    def test_all_valid_levels_accepted(self):
        """All four valid thinking levels are accepted for 3.x models."""
        for level in ("minimal", "low", "medium", "high"):
            result = _build_thinking_config("gemini-3.5-flash", level, None)
            assert result is not None, f"Expected ThinkingConfig for level={level!r}"
            # SDK normalises to a ThinkingLevel enum; compare via .value.
            assert result.thinking_level.value == level.upper()


# ---------------------------------------------------------------------------
# Tests for _has_speaker_turns
# ---------------------------------------------------------------------------


class TestHasSpeakerTurns:
    """Unit tests for the _has_speaker_turns guard helper."""

    def test_valid_dialogue_detected(self):
        text = "Alex: Hello!\nJordan: Hi there!"
        assert _has_speaker_turns(text, "Alex", "Jordan") is True

    def test_empty_text_returns_false(self):
        assert _has_speaker_turns("", "Alex", "Jordan") is False

    def test_no_speaker_prefix_returns_false(self):
        text = "This is just a planning note without any speaker turns."
        assert _has_speaker_turns(text, "Alex", "Jordan") is False

    def test_only_first_speaker_returns_true(self):
        text = "Alex: I am the only speaker here."
        assert _has_speaker_turns(text, "Alex", "Jordan") is True

    def test_only_second_speaker_returns_true(self):
        text = "Jordan: Just me today."
        assert _has_speaker_turns(text, "Alex", "Jordan") is True

    def test_leading_whitespace_stripped(self):
        text = "  Alex: This line has leading spaces."
        assert _has_speaker_turns(text, "Alex", "Jordan") is True


# ---------------------------------------------------------------------------
# Tests for the guardrail retry logic in generate_dialogue
# ---------------------------------------------------------------------------


def _make_fake_response(text: str) -> MagicMock:
    """Build a single fake genai response object."""
    r = MagicMock()
    r.text = text
    r.usage_metadata = MagicMock()
    return r


class TestDialogueGuardrail:
    """Retry guardrail: no speaker turns triggers retry; raises after all attempts."""

    def test_all_bad_responses_raises_runtime_error(self):
        """When every attempt returns text with no speaker turns, RuntimeError is raised."""
        bad_response = _make_fake_response(
            "I am thinking about the structure of the dialogue..."
        )
        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content.return_value = bad_response

        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            with pytest.raises(RuntimeError, match="no speaker turns"):
                generate_dialogue(SAMPLE_ARTICLES, GEMINI_CFG, "Alex", "Jordan")

    def test_retry_on_bad_then_good_returns_chunks(self):
        """When the first attempt is bad but the second is valid, chunks are returned."""
        bad_response = _make_fake_response("Planning: let me think about this...")
        good_response = _make_fake_response(SHORT_DIALOGUE)

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content.side_effect = [bad_response, good_response]

        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            chunks = generate_dialogue(SAMPLE_ARTICLES, GEMINI_CFG, "Alex", "Jordan")

        assert len(chunks) > 0
        assert all(isinstance(c, DialogueChunk) for c in chunks)
        # Two API calls were made (one bad, one good)
        assert mock_client.models.generate_content.call_count == 2

    def test_empty_response_also_triggers_retry(self):
        """An empty response.text is treated as a failed attempt."""
        empty_response = _make_fake_response("")
        good_response = _make_fake_response(SHORT_DIALOGUE)

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content.side_effect = [empty_response, good_response]

        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            chunks = generate_dialogue(SAMPLE_ARTICLES, GEMINI_CFG, "Alex", "Jordan")

        assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Tests for thinking_config wiring in generate_dialogue
# ---------------------------------------------------------------------------


GEMINI_CFG_3X = {
    "api_key": "test-api-key",
    "text_model": "gemini-3.5-flash",
    "tts_model": "gemini-3.1-flash-tts-preview",
    "speaker1": {"name": "Alex", "voice": "Puck"},
    "speaker2": {"name": "Jordan", "voice": "Charon"},
    "dialogue": {"thinking_level": "low"},
}


class TestThinkingConfigWiring:
    """thinking_config is passed to generate_content when configured."""

    def test_thinking_config_passed_for_3x_model_with_level(self):
        """GenerateContentConfig carries thinking_config when thinking_level is set."""
        mock_genai = _mock_genai_response(SHORT_DIALOGUE)

        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            generate_dialogue(SAMPLE_ARTICLES, GEMINI_CFG_3X, "Alex", "Jordan")

        call_kwargs = mock_genai.Client.return_value.models.generate_content.call_args.kwargs
        config_obj = call_kwargs.get("config")
        assert config_obj is not None
        assert config_obj.thinking_config is not None
        # SDK normalises to a ThinkingLevel enum; compare via .value.
        assert config_obj.thinking_config.thinking_level.value == "LOW"

    def test_no_thinking_config_when_not_set(self):
        """GenerateContentConfig has no thinking_config when dialogue section is absent."""
        mock_genai = _mock_genai_response(SHORT_DIALOGUE)

        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            generate_dialogue(SAMPLE_ARTICLES, GEMINI_CFG, "Alex", "Jordan")

        call_kwargs = mock_genai.Client.return_value.models.generate_content.call_args.kwargs
        config_obj = call_kwargs.get("config")
        assert config_obj is not None
        # thinking_config should be absent or None
        thinking = getattr(config_obj, "thinking_config", None)
        assert thinking is None


class TestChunkBudgetWiring:
    """
    ``generate_dialogue`` must chunk with the budget resolved from the active
    config, not with the module-level upper bound.

    Testing ``_resolve_chunk_budget`` in isolation is not enough: a budget that
    is computed and then never passed down is exactly the silent failure this
    project has been bitten by before (the same shape as a Gemini call that
    forgets to thread ``token_tracker`` through and silently undercounts cost).
    Here the symptom would be invisible too — chunks keep being produced, they
    are simply too big, and the TTS API rejects them mid-episode with a 4xx the
    retry decorator deliberately does not retry.
    """

    #: A duo that ships two voice directions plus a scene and a pace, i.e. the
    #: configuration shape whose preamble is large enough to move the budget.
    _DUO = BUILTIN_DUOS["explorer"]

    @classmethod
    def _heavy_cfg(cls) -> dict:
        """
        Return a Gemini config whose preamble is big enough to lower the budget.

        Returns
        -------
        dict
            ``GEMINI_CFG`` with a built-in duo's speakers, scene and pace.
        """
        return {
            **GEMINI_CFG,
            "speaker1": cls._DUO["speaker1"],
            "speaker2": cls._DUO["speaker2"],
            "tts_style": {"scene": cls._DUO["scene"], "pace": cls._DUO["pace"]},
        }

    @staticmethod
    def _long_dialogue(name1: str, name2: str, turns: int = 120) -> str:
        """
        Build a dialogue long enough to span several chunks.

        Parameters
        ----------
        name1, name2 : str
            Speaker names, alternated one per line.
        turns : int, optional
            Number of speaker turns to generate.

        Returns
        -------
        str
            A dialogue in the strict ``SpeakerName: text`` output format.
        """
        lines = []
        for i in range(turns):
            speaker = name1 if i % 2 == 0 else name2
            # ~110 bytes per turn: well under any budget, so no single turn can
            # overflow a chunk on its own and mask the bound under test.
            lines.append(f"{speaker}: Point numero {i}, {'du contenu parle ' * 5}voila.")
        return "\n".join(lines) + "\n"

    def _capture_split_kwargs(self, cfg: dict) -> dict:
        """
        Run ``generate_dialogue`` and return the chunker's keyword arguments.

        Parameters
        ----------
        cfg : dict
            Resolved Gemini config section to run against.

        Returns
        -------
        dict
            The kwargs ``generate_dialogue`` passed to
            ``_split_dialogue_into_chunks``.
        """
        mock_genai = _mock_genai_response(SHORT_DIALOGUE)
        with patch("tts_podcast.llm_summarizer.genai", mock_genai), patch(
            "tts_podcast.llm_summarizer._split_dialogue_into_chunks"
        ) as split:
            generate_dialogue(SAMPLE_ARTICLES, cfg, "Alex", "Jordan")
        assert split.call_count == 1
        return split.call_args.kwargs

    def test_resolved_budget_reaches_the_chunker(self):
        cfg = self._heavy_cfg()
        expected = _resolve_chunk_budget(cfg)
        # Guard the guard: if this duo happened to resolve to the upper bound,
        # the assertion below would also pass with the budget left unwired.
        assert expected < _MAX_CHUNK_BYTES, (
            "Fixture no longer has a preamble large enough to move the budget; "
            "pick a heavier config."
        )
        assert self._capture_split_kwargs(cfg)["max_bytes"] == expected

    def test_legacy_config_still_chunks_at_the_upper_bound(self):
        # GEMINI_CFG declares no voice_direction, no scene and no pace: the
        # pre-change chunking must be preserved exactly.
        assert self._capture_split_kwargs(GEMINI_CFG)["max_bytes"] == _MAX_CHUNK_BYTES

    def test_legacy_chunking_is_byte_identical_to_the_pre_change_split(self):
        # The regression guard end to end: same chunk texts, same chunk count,
        # therefore the same number of TTS requests and the same audio splice
        # points a legacy user got before the budget became dynamic.
        dialogue = self._long_dialogue("Alex", "Jordan")
        mock_genai = _mock_genai_response(dialogue)
        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            chunks = generate_dialogue(SAMPLE_ARTICLES, GEMINI_CFG, "Alex", "Jordan")

        pre_change = _split_dialogue_into_chunks(
            dialogue, "Alex", "Jordan", max_bytes=3000
        )
        assert [c.text for c in chunks] == [c.text for c in pre_change]
        assert len(chunks) > 1, "Fixture is too short to exercise chunking at all."

    def test_a_heavy_config_produces_more_chunks_than_a_legacy_one(self):
        # The flip side: the budget must actually bite when the preamble is
        # large, otherwise requests would go out over the TTS text limit.
        dialogue = self._long_dialogue("Alex", "Jordan")
        cfg = self._heavy_cfg()

        mock_heavy = _mock_genai_response(dialogue)
        with patch("tts_podcast.llm_summarizer.genai", mock_heavy):
            heavy_chunks = generate_dialogue(SAMPLE_ARTICLES, cfg, "Alex", "Jordan")

        mock_legacy = _mock_genai_response(dialogue)
        with patch("tts_podcast.llm_summarizer.genai", mock_legacy):
            legacy_chunks = generate_dialogue(
                SAMPLE_ARTICLES, GEMINI_CFG, "Alex", "Jordan"
            )

        assert len(heavy_chunks) > len(legacy_chunks)

    @pytest.mark.parametrize("slug", sorted(BUILTIN_DUOS))
    def test_every_generated_request_fits_the_tts_text_limit(self, slug):
        # The end-to-end statement of the guarantee: take the chunks the
        # pipeline really produces, render the request the TTS stage really
        # sends, and check the bytes on the wire.
        duo = BUILTIN_DUOS[slug]
        name1 = duo["speaker1"]["name"]
        name2 = duo["speaker2"]["name"]
        cfg = {
            **GEMINI_CFG,
            "speaker1": duo["speaker1"],
            "speaker2": duo["speaker2"],
            "tts_style": {"scene": duo["scene"], "pace": duo["pace"]},
        }
        dialogue = self._long_dialogue(name1, name2)
        mock_genai = _mock_genai_response(dialogue)
        with patch("tts_podcast.llm_summarizer.genai", mock_genai):
            chunks = generate_dialogue(SAMPLE_ARTICLES, cfg, name1, name2)

        assert len(chunks) > 1
        for chunk in chunks:
            request_bytes = len(_build_tts_prompt(chunk.text, cfg).encode("utf-8"))
            assert request_bytes <= 3800, (
                f"Duo {slug!r} chunk {chunk.index}: {request_bytes}-byte TTS request, "
                "over the 3800-byte working ceiling."
            )

    def test_explicit_max_bytes_wins_over_the_resolved_budget(self):
        # The CLI resolves the budget up front, before research and the
        # dialogue call are billed, and hands it down.  If generate_dialogue
        # ignored it and re-resolved, that early resolution would be decorative.
        cfg = self._heavy_cfg()
        mock_genai = _mock_genai_response(SHORT_DIALOGUE)
        with patch("tts_podcast.llm_summarizer.genai", mock_genai), patch(
            "tts_podcast.llm_summarizer._split_dialogue_into_chunks"
        ) as split:
            generate_dialogue(
                SAMPLE_ARTICLES, cfg, "Alex", "Jordan", max_bytes=1234
            )
        assert split.call_args.kwargs["max_bytes"] == 1234

    def test_explicit_max_bytes_skips_the_measurement_entirely(self):
        # A caller that already knows the budget must not pay for rendering the
        # preamble a second time, and must not trigger a duplicate over-budget
        # warning for a config the CLI has already reported on.
        cfg = self._heavy_cfg()
        mock_genai = _mock_genai_response(SHORT_DIALOGUE)
        with patch("tts_podcast.llm_summarizer.genai", mock_genai), patch(
            "tts_podcast.llm_summarizer._resolve_chunk_budget"
        ) as resolve:
            generate_dialogue(
                SAMPLE_ARTICLES, cfg, "Alex", "Jordan", max_bytes=2000
            )
        assert resolve.call_count == 0

    def test_max_bytes_defaults_to_resolving_from_the_config(self):
        # Direct library callers pass nothing and must keep the measured
        # behaviour rather than silently fall back to the upper bound.
        cfg = self._heavy_cfg()
        expected = _resolve_chunk_budget(cfg)
        assert expected < _MAX_CHUNK_BYTES, (
            "Fixture no longer has a preamble large enough to move the budget."
        )
        assert self._capture_split_kwargs(cfg)["max_bytes"] == expected


class TestOversizedSpeakerTurn:
    """
    A speaker turn is never split mid-line, so a turn bigger than the whole
    chunk budget is emitted alone and over budget.

    No safety margin can absorb that (the turn can be arbitrarily long), and it
    is the one request the TTS API is knowably liable to reject.  The contract
    is therefore: emit it, but say so, at chunking time, where the offending
    turn is still identifiable, rather than as a non-retryable 4xx from a
    worker thread once the dialogue is already billed.
    """

    @staticmethod
    def _dialogue_with_a_monologue(turn_bytes: int) -> str:
        """
        Build a dialogue containing one deliberately oversized turn.

        Parameters
        ----------
        turn_bytes : int
            Approximate UTF-8 size of the oversized turn.

        Returns
        -------
        str
            A dialogue in the strict ``SpeakerName: text`` output format.
        """
        return (
            "Alex: Short opener.\n"
            f"Jordan: {'mot ' * (turn_bytes // 4)}fin.\n"
            "Alex: Short closer.\n"
        )

    def test_oversized_turn_is_emitted_unsplit(self):
        dialogue = self._dialogue_with_a_monologue(3000)
        chunks = _split_dialogue_into_chunks(dialogue, "Alex", "Jordan", max_bytes=2000)
        oversized = [c for c in chunks if len(c.text.encode("utf-8")) > 2000]
        assert len(oversized) == 1, (
            "The long turn must land alone in its own chunk, unsplit: splitting "
            "mid-turn would cut a sentence in half in the audio."
        )
        assert oversized[0].text.startswith("Jordan:")

    def test_oversized_turn_warns_once_and_names_the_turn(self, caplog):
        dialogue = self._dialogue_with_a_monologue(3000)
        with caplog.at_level(logging.WARNING, logger="tts_podcast.llm_summarizer"):
            _split_dialogue_into_chunks(dialogue, "Alex", "Jordan", max_bytes=2000)
        messages = [record.getMessage() for record in caplog.records]
        assert len(messages) == 1, f"Expected exactly one warning, got {messages!r}"
        assert "2000" in messages[0]
        assert "Jordan:" in messages[0], (
            "The warning must quote the start of the offending turn, otherwise "
            "the user cannot find it in a 10000-word script."
        )

    def test_no_warning_when_every_turn_fits(self):
        # The complement: a normal script must stay silent, or the warning is
        # noise and gets ignored the day it matters.
        with patch.object(logging.getLogger("tts_podcast.llm_summarizer"), "warning") as warn:
            _split_dialogue_into_chunks(
                SHORT_DIALOGUE, "Alex", "Jordan", max_bytes=_MAX_CHUNK_BYTES
            )
        assert warn.call_count == 0
