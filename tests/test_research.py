"""
Tests for the research module.

Verifies the round-0 short-circuit, round-1 / round-N prompt construction,
grounding metadata extraction, and the iterative chaining of notes from
prior rounds into subsequent prompts.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tts_podcast.models import Source
from tts_podcast.research import (
    Citation,
    ResearchReport,
    _build_combined_notes,
    _extract_citations,
    conduct_research,
)


GEMINI_CFG = {
    "api_key": "test-key",
    "text_model": "gemini-2.5-flash",
    "language": "French",
}

SAMPLE_SOURCES = [
    Source(
        url="https://example.com/article",
        title="Sample Article",
        summary="Short summary.",
        full_text="Full body text of the sample article.",
        scraped_ok=True,
    ),
]


def _mock_response(text: str, citations=None, queries=None):
    """
    Build a Gemini-like response with optional grounding metadata.

    Parameters
    ----------
    text : str
        Text content the mock should return.
    citations : list[tuple[str, str]] or None
        Optional (title, uri) pairs to expose under grounding_chunks.
    queries : list[str] or None
        Optional list of strings exposed under web_search_queries.

    Returns
    -------
    SimpleNamespace
        Object mimicking google-genai's response with usage_metadata and
        candidates[0].grounding_metadata fields.
    """
    chunks = []
    for title, uri in citations or []:
        chunks.append(SimpleNamespace(web=SimpleNamespace(title=title, uri=uri)))

    metadata = SimpleNamespace(
        grounding_chunks=chunks,
        web_search_queries=queries or [],
    )
    candidate = SimpleNamespace(grounding_metadata=metadata)
    return SimpleNamespace(
        text=text,
        candidates=[candidate],
        usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=50),
    )


def _mock_genai(responses):
    """
    Build a mocked genai module whose generate_content yields the given responses in order.

    Parameters
    ----------
    responses : list
        Sequence of response objects returned successively.

    Returns
    -------
    MagicMock
        Mock genai module suitable for patching ``tts_podcast.research.genai``.
    """
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = responses

    mock_client = MagicMock()
    mock_client.models = mock_model

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client
    return mock_genai


# ---------------------------------------------------------------------------
# Round 0 short-circuit
# ---------------------------------------------------------------------------


class TestRound0:
    """Round 0 must not call the API and must return an empty report."""

    def test_returns_empty_report(self):
        with patch("tts_podcast.research.genai") as mock_genai:
            report = conduct_research(SAMPLE_SOURCES, rounds=0, gemini_cfg=GEMINI_CFG)

        assert isinstance(report, ResearchReport)
        assert report.rounds == []
        assert report.combined_notes == ""
        mock_genai.Client.assert_not_called()

    def test_negative_rounds_raises(self):
        with pytest.raises(ValueError):
            conduct_research(SAMPLE_SOURCES, rounds=-1, gemini_cfg=GEMINI_CFG)


# ---------------------------------------------------------------------------
# Round 1 prompt construction
# ---------------------------------------------------------------------------


class TestRound1Prompt:
    """The first round must inject the article(s) and language into the prompt."""

    def test_prompt_includes_articles_and_language(self):
        mock_genai = _mock_genai([_mock_response("Round 1 notes")])

        with patch("tts_podcast.research.genai", mock_genai):
            conduct_research(SAMPLE_SOURCES, rounds=1, gemini_cfg=GEMINI_CFG)

        call = mock_genai.Client.return_value.models.generate_content.call_args
        prompt = call.kwargs["contents"]
        assert "Sample Article" in prompt
        assert "https://example.com/article" in prompt
        assert "French" in prompt
        assert "Google Search" in prompt

    def test_uses_text_model_when_research_model_missing(self):
        mock_genai = _mock_genai([_mock_response("notes")])

        with patch("tts_podcast.research.genai", mock_genai):
            conduct_research(SAMPLE_SOURCES, rounds=1, gemini_cfg=GEMINI_CFG)

        call = mock_genai.Client.return_value.models.generate_content.call_args
        assert call.kwargs["model"] == "gemini-2.5-flash"

    def test_uses_research_model_override(self):
        cfg = {**GEMINI_CFG, "research": {"model": "gemini-2.5-pro"}}
        mock_genai = _mock_genai([_mock_response("notes")])

        with patch("tts_podcast.research.genai", mock_genai):
            conduct_research(SAMPLE_SOURCES, rounds=1, gemini_cfg=cfg)

        call = mock_genai.Client.return_value.models.generate_content.call_args
        assert call.kwargs["model"] == "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# Round N (N >= 2) prompt construction
# ---------------------------------------------------------------------------


class TestRoundNPrompt:
    """Subsequent rounds must include the previous rounds' notes verbatim."""

    def test_round_2_prompt_contains_round_1_notes(self):
        round1_notes = "- Initial fact about quantum (https://q.test/1)\n- Background on language (https://lang.test/2)"
        round2_notes = "- Follow-up gap on quantum (https://q.test/3)"

        responses = [
            _mock_response(round1_notes, citations=[("Q Test", "https://q.test/1")]),
            _mock_response(round2_notes),
        ]
        mock_genai = _mock_genai(responses)

        with patch("tts_podcast.research.genai", mock_genai):
            report = conduct_research(SAMPLE_SOURCES, rounds=2, gemini_cfg=GEMINI_CFG)

        assert len(report.rounds) == 2

        calls = mock_genai.Client.return_value.models.generate_content.call_args_list
        round_2_prompt = calls[1].kwargs["contents"]

        assert "Initial fact about quantum" in round_2_prompt
        assert "Background on language" in round_2_prompt
        assert "Previous research notes" in round_2_prompt
        assert "gaps" in round_2_prompt.lower()

    def test_three_rounds_chain_combines_all_prior_notes(self):
        """Round 3's prompt must include round-1 AND round-2 notes."""
        notes_1 = "- R1 fact"
        notes_2 = "- R2 fact"
        responses = [
            _mock_response(notes_1),
            _mock_response(notes_2),
            _mock_response("- R3 fact"),
        ]
        mock_genai = _mock_genai(responses)

        with patch("tts_podcast.research.genai", mock_genai):
            conduct_research(SAMPLE_SOURCES, rounds=3, gemini_cfg=GEMINI_CFG)

        calls = mock_genai.Client.return_value.models.generate_content.call_args_list
        round_3_prompt = calls[2].kwargs["contents"]

        assert "R1 fact" in round_3_prompt
        assert "R2 fact" in round_3_prompt


# ---------------------------------------------------------------------------
# Grounding metadata extraction
# ---------------------------------------------------------------------------


class TestCitationExtraction:
    """Verify _extract_citations parses grounding_chunks and web_search_queries."""

    def test_extracts_citations_and_queries(self):
        response = _mock_response(
            "notes",
            citations=[("Title 1", "https://a"), ("Title 2", "https://b")],
            queries=["query one", "query two"],
        )

        citations, queries = _extract_citations(response)

        assert citations == [
            Citation(title="Title 1", uri="https://a"),
            Citation(title="Title 2", uri="https://b"),
        ]
        assert queries == ["query one", "query two"]

    def test_empty_grounding_metadata_returns_empty(self):
        response = SimpleNamespace(candidates=[SimpleNamespace(grounding_metadata=None)])
        citations, queries = _extract_citations(response)
        assert citations == []
        assert queries == []

    def test_skips_chunks_without_uri(self):
        chunk_no_uri = SimpleNamespace(web=SimpleNamespace(title="X", uri=""))
        chunk_ok = SimpleNamespace(web=SimpleNamespace(title="Y", uri="https://y"))
        metadata = SimpleNamespace(
            grounding_chunks=[chunk_no_uri, chunk_ok],
            web_search_queries=[],
        )
        response = SimpleNamespace(candidates=[SimpleNamespace(grounding_metadata=metadata)])

        citations, _ = _extract_citations(response)
        assert citations == [Citation(title="Y", uri="https://y")]


# ---------------------------------------------------------------------------
# Combined notes
# ---------------------------------------------------------------------------


class TestCombinedNotes:
    """Verify the per-round notes are concatenated under round headers."""

    def test_combined_notes_includes_round_headers(self):
        mock_genai = _mock_genai([
            _mock_response("- Fact A"),
            _mock_response("- Fact B"),
        ])

        with patch("tts_podcast.research.genai", mock_genai):
            report = conduct_research(SAMPLE_SOURCES, rounds=2, gemini_cfg=GEMINI_CFG)

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
        mock_genai = _mock_genai([
            _mock_response("Notes 1"),
            _mock_response("Notes 2"),
        ])

        with patch("tts_podcast.research.genai", mock_genai):
            conduct_research(
                SAMPLE_SOURCES, rounds=2, gemini_cfg=GEMINI_CFG, token_tracker=tracker,
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
        mock_genai = _mock_genai([_mock_response("Notes 1")])
        with patch("tts_podcast.research.genai", mock_genai):
            conduct_research(
                SAMPLE_SOURCES,
                rounds=1,
                gemini_cfg=GEMINI_CFG,
                angle="the regulatory implications",
            )
        prompt = mock_genai.Client.return_value.models.generate_content.call_args.kwargs[
            "contents"
        ]
        assert "Angle to emphasize: the regulatory implications" in prompt

    def test_angle_header_NOT_re_injected_in_round_n_prompt(self):
        """Round-N prompt MUST NOT carry the literal 'Angle to emphasize:' header."""
        mock_genai = _mock_genai([
            _mock_response("Round 1 baseline notes"),
            _mock_response("Round 2 gap notes"),
        ])
        with patch("tts_podcast.research.genai", mock_genai):
            conduct_research(
                SAMPLE_SOURCES,
                rounds=2,
                gemini_cfg=GEMINI_CFG,
                angle="economy",
            )
        calls = mock_genai.Client.return_value.models.generate_content.call_args_list
        round_1 = calls[0].kwargs["contents"]
        round_2 = calls[1].kwargs["contents"]
        # Round 1 has the header; round 2 must not (the angle stays out of the
        # gap-analysis directive; it would only survive via previous_notes).
        assert "Angle to emphasize:" in round_1
        assert "Angle to emphasize:" not in round_2

    def test_round1_no_angle_keeps_byte_identical_prompt(self):
        """No-angle path: prompt has no 'Angle to emphasize:' line, byte-identical to baseline."""
        mock_genai = _mock_genai([_mock_response("notes")])
        with patch("tts_podcast.research.genai", mock_genai):
            conduct_research(SAMPLE_SOURCES, rounds=1, gemini_cfg=GEMINI_CFG)
        prompt = mock_genai.Client.return_value.models.generate_content.call_args.kwargs[
            "contents"
        ]
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
        mock_genai = _mock_genai([_mock_response("notes")])
        with patch("tts_podcast.research.genai", mock_genai):
            conduct_research(
                [search_source],
                rounds=1,
                gemini_cfg=GEMINI_CFG,
                angle="regulatory impact",
            )
        prompt = mock_genai.Client.return_value.models.generate_content.call_args.kwargs[
            "contents"
        ]
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
        mock_genai = _mock_genai([_mock_response("notes")])
        with patch("tts_podcast.research.genai", mock_genai):
            conduct_research([SEARCH_SOURCE], rounds=1, gemini_cfg=GEMINI_CFG)

        prompt = mock_genai.Client.return_value.models.generate_content.call_args.kwargs[
            "contents"
        ]
        # Distinctive markers of _ROUND_1_SEARCH_PROMPT
        assert "SUBSTANTIVE, COMPREHENSIVE" in prompt
        assert "Topic:" in prompt
        # Must NOT look like the article-centric prompt
        assert "complementary angles" not in prompt

    def test_url_sources_use_article_prompt(self):
        """When sources are kind=='url' (default), _ROUND_1_PROMPT must be used."""
        mock_genai = _mock_genai([_mock_response("notes")])
        with patch("tts_podcast.research.genai", mock_genai):
            conduct_research(SAMPLE_SOURCES, rounds=1, gemini_cfg=GEMINI_CFG)

        prompt = mock_genai.Client.return_value.models.generate_content.call_args.kwargs[
            "contents"
        ]
        # Distinctive markers of _ROUND_1_PROMPT
        assert "complementary angles" in prompt
        assert "Articles:" in prompt
        # Must NOT look like the search-only prompt
        assert "SUBSTANTIVE, COMPREHENSIVE" not in prompt

    def test_mixed_sources_use_article_prompt(self):
        """A mix of search + url sources must fall back to _ROUND_1_PROMPT."""
        url_source = SAMPLE_SOURCES[0]
        mixed = [SEARCH_SOURCE, url_source]
        mock_genai = _mock_genai([_mock_response("notes")])
        with patch("tts_podcast.research.genai", mock_genai):
            conduct_research(mixed, rounds=1, gemini_cfg=GEMINI_CFG)

        prompt = mock_genai.Client.return_value.models.generate_content.call_args.kwargs[
            "contents"
        ]
        assert "complementary angles" in prompt
        assert "SUBSTANTIVE, COMPREHENSIVE" not in prompt

    def test_search_prompt_round_n_unchanged(self):
        """Round N>=2 must still use _ROUND_N_PROMPT regardless of source kind."""
        mock_genai = _mock_genai([
            _mock_response("round 1 notes"),
            _mock_response("round 2 notes"),
        ])
        with patch("tts_podcast.research.genai", mock_genai):
            conduct_research([SEARCH_SOURCE], rounds=2, gemini_cfg=GEMINI_CFG)

        calls = mock_genai.Client.return_value.models.generate_content.call_args_list
        round_2_prompt = calls[1].kwargs["contents"]

        # Round 2 must use _ROUND_N_PROMPT (gap-analysis)
        assert "Previous research notes" in round_2_prompt
        assert "gaps" in round_2_prompt.lower()
        assert "SUBSTANTIVE, COMPREHENSIVE" not in round_2_prompt
