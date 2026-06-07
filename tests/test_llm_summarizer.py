"""
Tests for the llm_summarizer module.

Verifies dialogue generation, chunking, and research-notes injection
behaviour with a mocked Gemini client.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tts_podcast.llm_summarizer import (
    DialogueChunk,
    _audio_tags_enabled,
    _build_prompt,
    generate_dialogue,
)
from tts_podcast.style_presets import STYLE_PRESETS


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
