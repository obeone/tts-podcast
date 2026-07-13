"""
Tests for the research module.

Verifies the round-0 short-circuit, round-1 / round-N prompt construction,
grounding metadata extraction, and the iterative chaining of notes from
prior rounds into subsequent prompts.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tts_podcast.llm_client import Grounding, LlmResult
from tts_podcast.models import Source
from tts_podcast.research import (
    Citation,
    ResearchReport,
    _build_combined_notes,
    conduct_research,
)
from tts_podcast.settings import LlmSettings


GEMINI_CFG = {
    "language": "French",
}

LLM_SETTINGS = LlmSettings(
    provider="gemini",
    text_model="gemini-2.5-flash",
    research_model=None,
    api_key="test-key",
    api_base=None,
    temperature=None,
    extra_headers=None,
)

SAMPLE_SOURCES = [
    Source(
        url="https://example.com/article",
        title="Sample Article",
        summary="Short summary.",
        full_text="Full body text of the sample article.",
        scraped_ok=True,
    ),
]


def _llm_result(
    text: str,
    citations: list[tuple[str, str]] | None = None,
    queries: list[str] | None = None,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> LlmResult:
    """
    Build a neutral :class:`~tts_podcast.llm_client.LlmResult` for mocking ``complete``.

    Parameters
    ----------
    text : str
        Text content the mock should return.
    citations : list[tuple[str, str]] or None
        Optional ``(title, uri)`` pairs to expose as grounding citations.
    queries : list[str] or None
        Optional search queries to expose as grounding queries.
    input_tokens, output_tokens : int, optional
        Token counts to report.

    Returns
    -------
    LlmResult
        A result carrying *text*, token counts, and optional grounding
        (``None`` when neither *citations* nor *queries* is given).
    """
    grounding = None
    if citations or queries:
        grounding = Grounding(citations=citations or [], queries=queries or [])
    return LlmResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        grounding=grounding,
    )


# ---------------------------------------------------------------------------
# Round 0 short-circuit
# ---------------------------------------------------------------------------


class TestRound0:
    """Round 0 must not call the API and must return an empty report."""

    def test_returns_empty_report(self):
        with patch("tts_podcast.research.complete") as mock_complete:
            report = conduct_research(
                SAMPLE_SOURCES, rounds=0, gemini_cfg=GEMINI_CFG, llm_cfg=LLM_SETTINGS
            )

        assert isinstance(report, ResearchReport)
        assert report.rounds == []
        assert report.combined_notes == ""
        mock_complete.assert_not_called()

    def test_negative_rounds_raises(self):
        with pytest.raises(ValueError):
            conduct_research(
                SAMPLE_SOURCES, rounds=-1, gemini_cfg=GEMINI_CFG, llm_cfg=LLM_SETTINGS
            )


# ---------------------------------------------------------------------------
# Round 1 prompt construction
# ---------------------------------------------------------------------------


class TestRound1Prompt:
    """The first round must inject the article(s) and language into the prompt."""

    def test_prompt_includes_articles_and_language(self):
        with patch(
            "tts_podcast.research.complete", return_value=_llm_result("Round 1 notes")
        ) as mock_complete:
            conduct_research(
                SAMPLE_SOURCES, rounds=1, gemini_cfg=GEMINI_CFG, llm_cfg=LLM_SETTINGS
            )

        prompt = mock_complete.call_args.kwargs["prompt"]
        assert "Sample Article" in prompt
        assert "https://example.com/article" in prompt
        assert "French" in prompt
        assert "Google Search" in prompt

    def test_uses_text_model_when_research_model_missing(self):
        with patch(
            "tts_podcast.research.complete", return_value=_llm_result("notes")
        ) as mock_complete:
            conduct_research(
                SAMPLE_SOURCES, rounds=1, gemini_cfg=GEMINI_CFG, llm_cfg=LLM_SETTINGS
            )

        assert mock_complete.call_args.kwargs["model"] == "gemini/gemini-2.5-flash"

    def test_uses_research_model_override(self):
        llm_cfg = replace(LLM_SETTINGS, research_model="gemini-2.5-pro")
        with patch(
            "tts_podcast.research.complete", return_value=_llm_result("notes")
        ) as mock_complete:
            conduct_research(SAMPLE_SOURCES, rounds=1, gemini_cfg=GEMINI_CFG, llm_cfg=llm_cfg)

        assert mock_complete.call_args.kwargs["model"] == "gemini/gemini-2.5-pro"


# ---------------------------------------------------------------------------
# Round N (N >= 2) prompt construction
# ---------------------------------------------------------------------------


class TestRoundNPrompt:
    """Subsequent rounds must include the previous rounds' notes verbatim."""

    def test_round_2_prompt_contains_round_1_notes(self):
        round1_notes = "- Initial fact about quantum (https://q.test/1)\n- Background on language (https://lang.test/2)"
        round2_notes = "- Follow-up gap on quantum (https://q.test/3)"

        responses = [
            _llm_result(round1_notes, citations=[("Q Test", "https://q.test/1")]),
            _llm_result(round2_notes),
        ]

        with patch("tts_podcast.research.complete", side_effect=responses) as mock_complete:
            report = conduct_research(
                SAMPLE_SOURCES, rounds=2, gemini_cfg=GEMINI_CFG, llm_cfg=LLM_SETTINGS
            )

        assert len(report.rounds) == 2

        calls = mock_complete.call_args_list
        round_2_prompt = calls[1].kwargs["prompt"]

        assert "Initial fact about quantum" in round_2_prompt
        assert "Background on language" in round_2_prompt
        assert "Previous research notes" in round_2_prompt
        assert "gaps" in round_2_prompt.lower()

    def test_three_rounds_chain_combines_all_prior_notes(self):
        """Round 3's prompt must include round-1 AND round-2 notes."""
        notes_1 = "- R1 fact"
        notes_2 = "- R2 fact"
        responses = [
            _llm_result(notes_1),
            _llm_result(notes_2),
            _llm_result("- R3 fact"),
        ]

        with patch("tts_podcast.research.complete", side_effect=responses) as mock_complete:
            conduct_research(
                SAMPLE_SOURCES, rounds=3, gemini_cfg=GEMINI_CFG, llm_cfg=LLM_SETTINGS
            )

        calls = mock_complete.call_args_list
        round_3_prompt = calls[2].kwargs["prompt"]

        assert "R1 fact" in round_3_prompt
        assert "R2 fact" in round_3_prompt


# ---------------------------------------------------------------------------
# Grounding metadata extraction
# ---------------------------------------------------------------------------


class TestGroundingExtraction:
    """Verify conduct_research maps LlmResult.grounding into ResearchRound fields."""

    def test_citations_and_queries_mapped_from_grounding(self):
        """Grounding citations/queries on the LlmResult land verbatim on the round."""
        with patch(
            "tts_podcast.research.complete",
            return_value=_llm_result(
                "notes",
                citations=[("Title 1", "https://a"), ("Title 2", "https://b")],
                queries=["query one", "query two"],
            ),
        ):
            report = conduct_research(
                SAMPLE_SOURCES, rounds=1, gemini_cfg=GEMINI_CFG, llm_cfg=LLM_SETTINGS
            )

        round_ = report.rounds[0]
        assert round_.citations == [
            Citation(title="Title 1", uri="https://a"),
            Citation(title="Title 2", uri="https://b"),
        ]
        assert round_.raw_search_queries == ["query one", "query two"]

    def test_no_grounding_yields_empty_citations_and_queries(self):
        """A None grounding (e.g. non-Gemini provider) degrades to empty lists."""
        with patch(
            "tts_podcast.research.complete",
            return_value=_llm_result("notes"),
        ):
            report = conduct_research(
                SAMPLE_SOURCES, rounds=1, gemini_cfg=GEMINI_CFG, llm_cfg=LLM_SETTINGS
            )

        round_ = report.rounds[0]
        assert round_.citations == []
        assert round_.raw_search_queries == []


# ---------------------------------------------------------------------------
# Combined notes
# ---------------------------------------------------------------------------


class TestCombinedNotes:
    """Verify the per-round notes are concatenated under round headers."""

    def test_combined_notes_includes_round_headers(self):
        responses = [_llm_result("- Fact A"), _llm_result("- Fact B")]

        with patch("tts_podcast.research.complete", side_effect=responses):
            report = conduct_research(
                SAMPLE_SOURCES, rounds=2, gemini_cfg=GEMINI_CFG, llm_cfg=LLM_SETTINGS
            )

        assert "Research round 1" in report.combined_notes
        assert "Research round 2" in report.combined_notes
        assert "Fact A" in report.combined_notes
        assert "Fact B" in report.combined_notes

    def test_combined_notes_skips_empty_rounds(self):
        rounds = [
            SimpleNamespace(index=0, notes="- Fact A"),
            SimpleNamespace(index=1, notes=""),
            SimpleNamespace(index=2, notes="- Fact C"),
        ]
        combined = _build_combined_notes(rounds)

        assert "Fact A" in combined
        assert "Fact C" in combined
        assert "Research round 2" not in combined  # empty round skipped


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------


class TestTokenTrackerIntegration:
    """When provided, the token tracker records usage for every research call."""

    def test_tracker_records_each_round(self):
        from tts_podcast.token_tracker import TokenTracker

        tracker = TokenTracker()
        responses = [_llm_result("Notes 1"), _llm_result("Notes 2")]

        with patch("tts_podcast.research.complete", side_effect=responses):
            conduct_research(
                SAMPLE_SOURCES,
                rounds=2,
                gemini_cfg=GEMINI_CFG,
                llm_cfg=LLM_SETTINGS,
                token_tracker=tracker,
            )

        # 2 rounds × 100 input + 50 output tokens each
        summary = tracker.summary()
        assert "200" in summary  # total input
        assert "100" in summary  # total output


# ---------------------------------------------------------------------------
# Angle injection (round 1 only)
# ---------------------------------------------------------------------------


class TestAngleInjection:
    """Angle reaches round-1 prompt only — never re-injected into round N>=2."""

    def test_angle_in_round1_prompt(self):
        with patch(
            "tts_podcast.research.complete", return_value=_llm_result("Notes 1")
        ) as mock_complete:
            conduct_research(
                SAMPLE_SOURCES,
                rounds=1,
                gemini_cfg=GEMINI_CFG,
                llm_cfg=LLM_SETTINGS,
                angle="the regulatory implications",
            )
        prompt = mock_complete.call_args.kwargs["prompt"]
        assert "Angle to emphasize: the regulatory implications" in prompt

    def test_angle_header_NOT_re_injected_in_round_n_prompt(self):
        """Round-N prompt MUST NOT carry the literal 'Angle to emphasize:' header."""
        responses = [
            _llm_result("Round 1 baseline notes"),
            _llm_result("Round 2 gap notes"),
        ]
        with patch("tts_podcast.research.complete", side_effect=responses) as mock_complete:
            conduct_research(
                SAMPLE_SOURCES,
                rounds=2,
                gemini_cfg=GEMINI_CFG,
                llm_cfg=LLM_SETTINGS,
                angle="economy",
            )
        calls = mock_complete.call_args_list
        round_1 = calls[0].kwargs["prompt"]
        round_2 = calls[1].kwargs["prompt"]
        # Round 1 has the header; round 2 must not (the angle stays out of the
        # gap-analysis directive; it would only survive via previous_notes).
        assert "Angle to emphasize:" in round_1
        assert "Angle to emphasize:" not in round_2

    def test_round1_no_angle_keeps_byte_identical_prompt(self):
        """No-angle path: prompt has no 'Angle to emphasize:' line, byte-identical to baseline."""
        with patch(
            "tts_podcast.research.complete", return_value=_llm_result("notes")
        ) as mock_complete:
            conduct_research(
                SAMPLE_SOURCES, rounds=1, gemini_cfg=GEMINI_CFG, llm_cfg=LLM_SETTINGS
            )
        prompt = mock_complete.call_args.kwargs["prompt"]
        assert "Angle to emphasize:" not in prompt

    def test_angle_plus_search_input(self):
        """Search source + angle => round-1 prompt contains both the topic and the angle."""
        search_source = Source(
            url="search://AI%20economy",
            title="Web search: AI economy",
            summary="Topic to investigate via web research: AI economy",
            full_text="Topic to investigate via web research: AI economy",
            scraped_ok=True,
            kind="search",
        )
        with patch(
            "tts_podcast.research.complete", return_value=_llm_result("notes")
        ) as mock_complete:
            conduct_research(
                [search_source],
                rounds=1,
                gemini_cfg=GEMINI_CFG,
                llm_cfg=LLM_SETTINGS,
                angle="regulatory impact",
            )
        prompt = mock_complete.call_args.kwargs["prompt"]
        assert "AI economy" in prompt
        assert "Angle to emphasize: regulatory impact" in prompt


# ---------------------------------------------------------------------------
# Search-only round-1 prompt selection
# ---------------------------------------------------------------------------


SEARCH_SOURCE = Source(
    url="search://quantum%20computing",
    title="Web search: quantum computing",
    summary="Topic to investigate via web research: quantum computing",
    full_text="Topic to investigate via web research: quantum computing",
    scraped_ok=True,
    kind="search",
)


class TestSearchOnlyRound1Prompt:
    """Round-1 prompt selection: search-only vs article-centric."""

    def test_all_search_sources_use_search_prompt(self):
        """When every source is kind=='search', _ROUND_1_SEARCH_PROMPT must be used."""
        with patch(
            "tts_podcast.research.complete", return_value=_llm_result("notes")
        ) as mock_complete:
            conduct_research(
                [SEARCH_SOURCE], rounds=1, gemini_cfg=GEMINI_CFG, llm_cfg=LLM_SETTINGS
            )

        prompt = mock_complete.call_args.kwargs["prompt"]
        # Distinctive markers of _ROUND_1_SEARCH_PROMPT
        assert "SUBSTANTIVE, COMPREHENSIVE" in prompt
        assert "Topic:" in prompt
        # Must NOT look like the article-centric prompt
        assert "complementary angles" not in prompt

    def test_url_sources_use_article_prompt(self):
        """When sources are kind=='url' (default), _ROUND_1_PROMPT must be used."""
        with patch(
            "tts_podcast.research.complete", return_value=_llm_result("notes")
        ) as mock_complete:
            conduct_research(
                SAMPLE_SOURCES, rounds=1, gemini_cfg=GEMINI_CFG, llm_cfg=LLM_SETTINGS
            )

        prompt = mock_complete.call_args.kwargs["prompt"]
        # Distinctive markers of _ROUND_1_PROMPT
        assert "complementary angles" in prompt
        assert "Articles:" in prompt
        # Must NOT look like the search-only prompt
        assert "SUBSTANTIVE, COMPREHENSIVE" not in prompt

    def test_mixed_sources_use_article_prompt(self):
        """A mix of search + url sources must fall back to _ROUND_1_PROMPT."""
        url_source = SAMPLE_SOURCES[0]
        mixed = [SEARCH_SOURCE, url_source]
        with patch(
            "tts_podcast.research.complete", return_value=_llm_result("notes")
        ) as mock_complete:
            conduct_research(mixed, rounds=1, gemini_cfg=GEMINI_CFG, llm_cfg=LLM_SETTINGS)

        prompt = mock_complete.call_args.kwargs["prompt"]
        assert "complementary angles" in prompt
        assert "SUBSTANTIVE, COMPREHENSIVE" not in prompt

    def test_search_prompt_round_n_unchanged(self):
        """Round N>=2 must still use _ROUND_N_PROMPT regardless of source kind."""
        responses = [_llm_result("round 1 notes"), _llm_result("round 2 notes")]
        with patch("tts_podcast.research.complete", side_effect=responses) as mock_complete:
            conduct_research(
                [SEARCH_SOURCE], rounds=2, gemini_cfg=GEMINI_CFG, llm_cfg=LLM_SETTINGS
            )

        calls = mock_complete.call_args_list
        round_2_prompt = calls[1].kwargs["prompt"]

        # Round 2 must use _ROUND_N_PROMPT (gap-analysis)
        assert "Previous research notes" in round_2_prompt
        assert "gaps" in round_2_prompt.lower()
        assert "SUBSTANTIVE, COMPREHENSIVE" not in round_2_prompt
