"""
Tests for :mod:`tts_podcast.link_extractor` link collection.

Focused on :func:`collect_document_links`, the union of trafilatura's
article-body link pass and the document's raw anchors.  The body pass alone
was the original implementation, and it silently starved link following on
exactly the pages worth following links from.
"""

from __future__ import annotations

from tts_podcast.link_extractor import collect_document_links

# A newsletter shaped like the page that exposed the bug: trafilatura's body
# detector locks onto the sponsor paragraph and reports it as the whole
# article, so the markdown pass yields one link while the page carries a
# dozen.  Observed on a real TLDR AI issue: 69 KB of HTML, 25 distinct
# anchors, a 2.3 KB detected body, one captured link, and it was the ad.
#
# The site chrome deliberately sits at the top of the markup and uses plain
# <div>s, as the real page does: it has no <nav> and no <footer>, so nothing
# can be stripped by tag name.  The sponsor href is double-escaped, also as
# the real page does.
_NEWSLETTER_HTML = """
<html><body>
  <div class="masthead">
    <a href="https://tldr.example/">Home</a>
    <a href="/newsletters">Newsletters</a>
    <a href="https://advertise.tldr.example/">Advertise</a>
    <a href="/blog">Blog</a>
  </div>
  <div class="sponsor">
    <p>Invest in AI infra (Sponsor)
      <a href="https://sponsor.example/thematic?utm_source=NL&amp;amp;utm_medium=newsletter">Learn more</a>
    </p>
  </div>
  <div class="issue">
    <a href="https://openai.example/index/gpt-improvements">GPT improvements</a>
    <a href="https://arxiv.example/abs/2608.05000">A paper</a>
    <a href="https://github.example/someone/loopx">A repo</a>
    <a href="/relative/story">A relative story</a>
    <a href="mailto:hi@tldr.example">Contact</a>
  </div>
</body></html>
"""

_BASE_URL = "https://tldr.example/ai/2026-08-07"

# What the body pass produced for that page: the ad, and nothing else.
_NEWSLETTER_MARKDOWN = (
    "Invest in AI infra (Sponsor) "
    "[Learn more](https://sponsor.example/thematic?utm_source=NL&utm_medium=newsletter)"
)


class TestCollectDocumentLinks:
    """The body pass leads; the raw anchors are the recall backstop."""

    def test_article_links_survive_a_starved_body_pass(self) -> None:
        """
        The regression: a body pass yielding only the sponsor must not cost us
        the actual article links.

        Before the anchor fallback, this page produced exactly one candidate
        and it was the ad, so a --follow-links run spent its whole budget on
        the sponsor's landing page.
        """
        links = collect_document_links(_NEWSLETTER_MARKDOWN, _NEWSLETTER_HTML, base_url=_BASE_URL)
        for expected in (
            "https://openai.example/index/gpt-improvements",
            "https://arxiv.example/abs/2608.05000",
            "https://github.example/someone/loopx",
        ):
            assert expected in links, f"{expected} was lost before the heuristic could judge it"

    def test_article_links_fit_inside_a_default_candidate_budget(self) -> None:
        """
        Recovering the links is not enough if the site chrome is consumed
        first.  ``follow.max_links_per_level`` defaults to 5, and the chrome
        here is four anchors sitting above the content, so plain document
        order would spend the whole hop on the masthead.
        """
        links = collect_document_links(_NEWSLETTER_MARKDOWN, _NEWSLETTER_HTML, base_url=_BASE_URL)
        assert "https://openai.example/index/gpt-improvements" in links[:5]

    def test_body_links_come_first(self) -> None:
        """
        Ordering is the whole mechanism: consumers take a bounded prefix, so a
        body link must never be displaced by a page-tail anchor.
        """
        links = collect_document_links(_NEWSLETTER_MARKDOWN, _NEWSLETTER_HTML, base_url=_BASE_URL)
        assert links[0].startswith("https://sponsor.example/thematic")

    def test_offsite_anchors_outrank_same_site_chrome(self) -> None:
        """A link leaving the document's domain beats one pointing back into it."""
        links = collect_document_links("", _NEWSLETTER_HTML, base_url=_BASE_URL)
        offsite = links.index("https://openai.example/index/gpt-improvements")
        onsite = links.index("https://tldr.example/")
        assert offsite < onsite
        # Ordered, not filtered: the chrome is still available further down.
        assert "https://advertise.tldr.example/" in links

    def test_double_escaped_entities_collapse_to_one_url(self) -> None:
        """
        The sponsor href arrives as ``&amp;amp;`` from the anchor and ``&``
        from the markdown pass.  A single unescape pass leaves ``&amp;``, so
        both spellings survive deduplication and the follower fetches the same
        page twice.
        """
        links = collect_document_links(_NEWSLETTER_MARKDOWN, _NEWSLETTER_HTML, base_url=_BASE_URL)
        sponsor = [u for u in links if "sponsor.example" in u]
        assert sponsor == ["https://sponsor.example/thematic?utm_source=NL&utm_medium=newsletter"]

    def test_relative_hrefs_resolve_against_the_base_url(self) -> None:
        links = collect_document_links("", _NEWSLETTER_HTML, base_url=_BASE_URL)
        assert "https://tldr.example/relative/story" in links

    def test_relative_hrefs_are_dropped_without_a_base_url(self) -> None:
        """A local file's relative link resolves to nothing the follower can fetch."""
        links = collect_document_links("", _NEWSLETTER_HTML)
        assert not [u for u in links if "relative/story" in u]
        # Absolute anchors in the same document still come through.
        assert "https://arxiv.example/abs/2608.05000" in links

    def test_non_http_schemes_are_dropped(self) -> None:
        links = collect_document_links("", _NEWSLETTER_HTML, base_url=_BASE_URL)
        assert not [u for u in links if u.startswith("mailto:")]

    def test_subdomains_count_as_the_same_site(self) -> None:
        """``advertise.tldr.example`` is chrome, not an outbound content link."""
        links = collect_document_links("", _NEWSLETTER_HTML, base_url=_BASE_URL)
        assert links.index("https://arxiv.example/abs/2608.05000") < links.index(
            "https://advertise.tldr.example/"
        )

    def test_order_is_first_seen_and_deduplicated(self) -> None:
        html = (
            '<a href="https://a.example/1">a</a>'
            '<a href="https://b.example/2">b</a>'
            '<a href="https://a.example/1">a again</a>'
        )
        assert collect_document_links("", html) == [
            "https://a.example/1",
            "https://b.example/2",
        ]

    def test_single_quoted_hrefs_are_collected(self) -> None:
        assert collect_document_links("", "<a href='https://a.example/1'>a</a>") == [
            "https://a.example/1"
        ]

    def test_empty_document_yields_no_links(self) -> None:
        assert collect_document_links("", "") == []
