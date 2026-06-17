"""
Tests for the link-following module.

Covers the stage-1 ``is_followable_link`` heuristic, the BFS traversal in
``follow_links`` (depth, dedup/cycle safety, the relevance verdict, the
per-level cap, and the no-candidate short-circuit), and the stage-2
``_judge_sources`` Gemini call.  Like the rest of the suite, the Gemini SDK is
mocked — no test hits the network.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tts_podcast.link_follower as link_follower
from tts_podcast.link_extractor import is_followable_link
from tts_podcast.link_follower import _judge_sources, follow_links
from tts_podcast.models import Source

GEMINI_CFG = {
    "api_key": "test-key",
    "text_model": "gemini-2.5-flash",
    "language": "French",
}


def _source(url: str, *, body: str = "", title: str = "", ok: bool = True) -> Source:
    """
    Build a scraped :class:`Source` for tests.

    Parameters
    ----------
    url : str
        Source URL.
    body : str, optional
        Full-text body (where outgoing links are scanned from), by default "".
    title : str, optional
        Source title, by default derived from the URL.
    ok : bool, optional
        Value for ``scraped_ok``, by default ``True``.

    Returns
    -------
    Source
        A populated source object.
    """
    return Source(
        url=url,
        title=title or url,
        summary=body[:200],
        full_text=body,
        scraped_ok=ok,
    )


# ---------------------------------------------------------------------------
# Stage 1 — is_followable_link heuristic
# ---------------------------------------------------------------------------


class TestIsFollowableLink:
    """The pre-fetch heuristic keeps real content and drops obvious junk."""

    def test_keeps_real_content_urls(self):
        assert is_followable_link("https://example.com/some-article")
        assert is_followable_link("https://arxiv.org/abs/2401.12345")
        assert is_followable_link("https://github.com/user/repo")
        assert is_followable_link("https://huggingface.co/org/model")
        assert is_followable_link("https://example.com/paper.pdf")
        assert is_followable_link("https://example.com/page.html")

    def test_drops_non_http_schemes(self):
        assert not is_followable_link("mailto:hello@example.com")
        assert not is_followable_link("tel:+15551234")
        assert not is_followable_link("javascript:void(0)")

    def test_drops_same_page_anchor(self):
        assert not is_followable_link("#section-2")

    def test_drops_asset_extensions(self):
        assert not is_followable_link("https://example.com/logo.png")
        assert not is_followable_link("https://cdn.example.com/styles.css")
        assert not is_followable_link("https://cdn.example.com/app.js")
        assert not is_followable_link("https://example.com/archive.zip")

    def test_drops_ad_social_tracker_hosts(self):
        assert not is_followable_link("https://www.facebook.com/sharer")
        assert not is_followable_link("https://twitter.com/intent/tweet")
        assert not is_followable_link("https://x.com/someone/status/1")
        assert not is_followable_link("https://ad.doubleclick.net/abc")

    def test_drops_non_article_paths(self):
        assert not is_followable_link("https://example.com/login")
        assert not is_followable_link("https://example.com/signup")


# ---------------------------------------------------------------------------
# follow_links — BFS behaviour (judge mocked)
# ---------------------------------------------------------------------------


class TestFollowLinksDepth1:
    """Depth-1 traversal: heuristic filter, fetch, judge, keep non-irrelevant."""

    def test_keeps_only_non_irrelevant_and_skips_junk(self):
        seed = _source(
            "https://seed.example/post",
            body=(
                "See https://good.example/a and https://good.example/b "
                "and also https://facebook.com/sharer for sharing."
            ),
        )

        fetched = [
            _source("https://good.example/a", body="content a"),
            _source("https://good.example/b", body="content b"),
        ]
        scrape_mock = MagicMock(return_value=fetched)
        judge_mock = MagicMock(
            return_value={
                "https://good.example/a": "core",
                "https://good.example/b": "irrelevant",
            }
        )

        with (
            patch.object(link_follower, "scrape_urls", scrape_mock),
            patch.object(link_follower, "_judge_sources", judge_mock),
        ):
            kept = follow_links(
                [seed],
                depth=1,
                gemini_cfg=GEMINI_CFG,
                scrape_timeout=10,
                user_agent="ua",
                cloak_fallback=False,
            )

        # Only the non-irrelevant page is kept, with its verdict recorded.
        assert len(kept) == 1
        assert kept[0].url == "https://good.example/a"
        assert kept[0].relevance == "core"

        # The junk facebook link was never fetched.
        fetched_urls = scrape_mock.call_args.args[0]
        assert "https://facebook.com/sharer" not in fetched_urls
        assert fetched_urls == ["https://good.example/a", "https://good.example/b"]


class TestFollowLinksDedup:
    """A fetched page linking back to a seed must not be re-fetched."""

    def test_cycle_back_to_seed_is_not_refetched(self):
        seed = _source(
            "https://seed.example/post",
            body="Go to https://child.example/x",
        )
        # The fetched child links back to the seed AND to a brand-new page.
        child = _source(
            "https://child.example/x",
            body="back to https://seed.example/post and https://child.example/y",
        )

        scrape_mock = MagicMock(side_effect=[[child], [_source("https://child.example/y")]])
        judge_mock = MagicMock(
            side_effect=[
                {"https://child.example/x": "supporting"},
                {"https://child.example/y": "supporting"},
            ]
        )

        with (
            patch.object(link_follower, "scrape_urls", scrape_mock),
            patch.object(link_follower, "_judge_sources", judge_mock),
        ):
            kept = follow_links(
                [seed],
                depth=2,
                gemini_cfg=GEMINI_CFG,
                scrape_timeout=10,
                user_agent="ua",
                cloak_fallback=False,
            )

        # Level 2 fetched only the new page, never the seed it cycled back to.
        level2_urls = scrape_mock.call_args_list[1].args[0]
        assert "https://seed.example/post" not in level2_urls
        assert level2_urls == ["https://child.example/y"]
        assert {k.url for k in kept} == {
            "https://child.example/x",
            "https://child.example/y",
        }


class TestFollowLinksDepth2:
    """Depth-2 recurses into the links of pages kept at level 1."""

    def test_recurses_into_kept_pages(self):
        seed = _source("https://seed.example/post", body="L1 https://l1.example/a")
        l1 = _source("https://l1.example/a", body="L2 https://l2.example/b")
        l2 = _source("https://l2.example/b", body="leaf")

        scrape_mock = MagicMock(side_effect=[[l1], [l2]])
        judge_mock = MagicMock(
            side_effect=[
                {"https://l1.example/a": "core"},
                {"https://l2.example/b": "supporting"},
            ]
        )

        with (
            patch.object(link_follower, "scrape_urls", scrape_mock),
            patch.object(link_follower, "_judge_sources", judge_mock),
        ):
            kept = follow_links(
                [seed],
                depth=2,
                gemini_cfg=GEMINI_CFG,
                scrape_timeout=10,
                user_agent="ua",
                cloak_fallback=False,
            )

        assert scrape_mock.call_count == 2
        assert scrape_mock.call_args_list[1].args[0] == ["https://l2.example/b"]
        assert [k.url for k in kept] == [
            "https://l1.example/a",
            "https://l2.example/b",
        ]


class TestFollowLinksCap:
    """max_links_per_level bounds how many links a single level fetches."""

    def test_cap_respected(self):
        links = " ".join(f"https://good.example/{i}" for i in range(10))
        seed = _source("https://seed.example/post", body=links)

        def _fake_scrape(urls, **kwargs):
            return [_source(u) for u in urls]

        scrape_mock = MagicMock(side_effect=_fake_scrape)
        judge_mock = MagicMock(side_effect=lambda *a, **k: {})

        with (
            patch.object(link_follower, "scrape_urls", scrape_mock),
            patch.object(link_follower, "_judge_sources", judge_mock),
        ):
            follow_links(
                [seed],
                depth=1,
                gemini_cfg=GEMINI_CFG,
                scrape_timeout=10,
                user_agent="ua",
                cloak_fallback=False,
                max_links_per_level=3,
            )

        assert len(scrape_mock.call_args.args[0]) == 3


class TestFollowLinksGlobalCap:
    """max_links_total bounds the cumulative fetch count across all levels."""

    def test_global_budget_caps_total_fetches(self):
        # Each level's body yields fresh candidates, so without a global cap a
        # depth-3 run would fetch far more than 2 pages.
        seed = _source(
            "https://seed.example/post",
            body="https://l1.example/a https://l1.example/b https://l1.example/c",
        )

        def _fake_scrape(urls, **kwargs):
            # Each fetched page links to brand-new candidates for the next level.
            out = []
            for u in urls:
                child = u.rstrip("/") + "/child"
                out.append(_source(u, body=f"{child} {child}-2"))
            return out

        scrape_mock = MagicMock(side_effect=_fake_scrape)
        judge_mock = MagicMock(
            side_effect=lambda topic, ok, *a, **k: {s.url: "supporting" for s in ok}
        )

        with (
            patch.object(link_follower, "scrape_urls", scrape_mock),
            patch.object(link_follower, "_judge_sources", judge_mock),
        ):
            follow_links(
                [seed],
                depth=3,
                gemini_cfg=GEMINI_CFG,
                scrape_timeout=10,
                user_agent="ua",
                cloak_fallback=False,
                max_links_per_level=5,
                max_links_total=2,
            )

        # Sum every URL passed to scrape across all levels: never exceeds 2.
        total_fetched = sum(
            len(call.args[0]) for call in scrape_mock.call_args_list
        )
        assert total_fetched <= 2


class TestFollowLinksFromStructuredLinks:
    """Candidates are discovered from Source.links even when full_text has no URLs."""

    def test_discovers_candidates_from_source_links_field(self):
        """follow_links fetches URLs in Source.links when full_text contains no bare URLs."""
        # Simulate an HTML seed whose plain-text body has no bare URLs at all
        # (the real scenario: trafilatura strips hrefs from full_text).
        seed = Source(
            url="https://seed.example/post",
            title="Seed",
            summary="plain text no urls",
            full_text="plain text no urls",
            scraped_ok=True,
            links=["https://a.example/x", "https://b.example/y"],
        )

        fetched = [
            _source("https://a.example/x", body="content a"),
            _source("https://b.example/y", body="content b"),
        ]
        scrape_mock = MagicMock(return_value=fetched)
        judge_mock = MagicMock(
            return_value={
                "https://a.example/x": "core",
                "https://b.example/y": "supporting",
            }
        )

        with (
            patch.object(link_follower, "scrape_urls", scrape_mock),
            patch.object(link_follower, "_judge_sources", judge_mock),
        ):
            kept = follow_links(
                [seed],
                depth=1,
                gemini_cfg=GEMINI_CFG,
                scrape_timeout=10,
                user_agent="ua",
                cloak_fallback=False,
            )

        # Both links from Source.links were fetched and kept.
        assert len(kept) == 2
        fetched_urls = scrape_mock.call_args.args[0]
        assert "https://a.example/x" in fetched_urls
        assert "https://b.example/y" in fetched_urls

    def test_deduplicates_links_across_structured_and_text(self):
        """A URL that appears in both Source.links and full_text is fetched only once."""
        seed = Source(
            url="https://seed.example/post",
            title="Seed",
            summary="see https://a.example/x",
            full_text="see https://a.example/x",
            scraped_ok=True,
            links=["https://a.example/x"],
        )

        scrape_mock = MagicMock(return_value=[_source("https://a.example/x")])
        judge_mock = MagicMock(return_value={"https://a.example/x": "core"})

        with (
            patch.object(link_follower, "scrape_urls", scrape_mock),
            patch.object(link_follower, "_judge_sources", judge_mock),
        ):
            follow_links(
                [seed],
                depth=1,
                gemini_cfg=GEMINI_CFG,
                scrape_timeout=10,
                user_agent="ua",
                cloak_fallback=False,
            )

        fetched_urls = scrape_mock.call_args.args[0]
        # The URL appears only once despite being in both lists.
        assert fetched_urls.count("https://a.example/x") == 1


class TestFollowLinksNoCandidates:
    """No followable candidates → empty result, no fetch, no judge."""

    def test_no_candidates_returns_empty(self):
        # Body has only a junk link → nothing followable.
        seed = _source(
            "https://seed.example/post",
            body="only junk https://facebook.com/sharer here",
        )
        scrape_mock = MagicMock()
        judge_mock = MagicMock()

        with (
            patch.object(link_follower, "scrape_urls", scrape_mock),
            patch.object(link_follower, "_judge_sources", judge_mock),
        ):
            kept = follow_links(
                [seed],
                depth=1,
                gemini_cfg=GEMINI_CFG,
                scrape_timeout=10,
                user_agent="ua",
                cloak_fallback=False,
            )

        assert kept == []
        scrape_mock.assert_not_called()
        judge_mock.assert_not_called()

    def test_depth_zero_returns_empty(self):
        seed = _source("https://seed.example/post", body="https://good.example/a")
        scrape_mock = MagicMock()
        with patch.object(link_follower, "scrape_urls", scrape_mock):
            assert (
                follow_links(
                    [seed],
                    depth=0,
                    gemini_cfg=GEMINI_CFG,
                    scrape_timeout=10,
                    user_agent="ua",
                    cloak_fallback=False,
                )
                == []
            )
        scrape_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Stage 2 — _judge_sources Gemini call
# ---------------------------------------------------------------------------


def _mock_judge_response(text: str):
    """
    Build a Gemini-like response whose ``.text`` is the given JSON string.

    Parameters
    ----------
    text : str
        JSON payload to expose under ``.text``.

    Returns
    -------
    SimpleNamespace
        Object mimicking google-genai's response with ``usage_metadata``.
    """
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=50),
    )


def _mock_genai(response):
    """
    Build a mocked genai module whose generate_content returns *response*.

    Parameters
    ----------
    response : Any
        Response object to return from ``generate_content``.

    Returns
    -------
    MagicMock
        Mock genai module suitable for patching ``tts_podcast.link_follower.genai``.
    """
    mock_model = MagicMock()
    mock_model.generate_content.return_value = response
    mock_client = MagicMock()
    mock_client.models = mock_model
    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client
    return mock_genai


class TestJudgeSources:
    """The stage-2 judge parses JSON, records usage, and fails open."""

    def test_parses_json_array_and_records_usage(self):
        from tts_podcast.token_tracker import TokenTracker

        fetched = [
            _source("https://a.example/x", body="alpha"),
            _source("https://b.example/y", body="beta"),
        ]
        # The judge now returns a STABLE ARRAY of {url, label} objects.
        response = _mock_judge_response(
            '[{"url": "https://a.example/x", "label": "core"}, '
            '{"url": "https://b.example/y", "label": "irrelevant"}]'
        )
        mock_genai = _mock_genai(response)
        tracker = TokenTracker()

        with patch.object(link_follower, "genai", mock_genai):
            verdicts = _judge_sources("topic", fetched, GEMINI_CFG, tracker)

        # The array is flattened back to the correct mapping (NOT all-"supporting").
        assert verdicts == {
            "https://a.example/x": "core",
            "https://b.example/y": "irrelevant",
        }
        # Token usage was recorded for the judging call.
        summary = tracker.summary()
        assert "100" in summary
        assert "50" in summary

    def test_unparseable_response_fails_open_to_supporting(self):
        fetched = [_source("https://a.example/x", body="alpha")]
        response = _mock_judge_response("not json at all")
        mock_genai = _mock_genai(response)

        with patch.object(link_follower, "genai", mock_genai):
            verdicts = _judge_sources("topic", fetched, GEMINI_CFG, None)

        assert verdicts == {"https://a.example/x": "supporting"}

    def test_unknown_label_coerced_to_supporting(self):
        fetched = [_source("https://a.example/x", body="alpha")]
        response = _mock_judge_response(
            '[{"url": "https://a.example/x", "label": "maybe"}]'
        )
        mock_genai = _mock_genai(response)

        with patch.object(link_follower, "genai", mock_genai):
            verdicts = _judge_sources("topic", fetched, GEMINI_CFG, None)

        assert verdicts == {"https://a.example/x": "supporting"}

    def test_omitted_url_defaults_to_supporting(self):
        # The model only judged one of the two fetched pages; the omitted URL
        # must default to "supporting" rather than vanish.
        fetched = [
            _source("https://a.example/x", body="alpha"),
            _source("https://b.example/y", body="beta"),
        ]
        response = _mock_judge_response(
            '[{"url": "https://a.example/x", "label": "core"}]'
        )
        mock_genai = _mock_genai(response)

        with patch.object(link_follower, "genai", mock_genai):
            verdicts = _judge_sources("topic", fetched, GEMINI_CFG, None)

        assert verdicts == {
            "https://a.example/x": "core",
            "https://b.example/y": "supporting",
        }
