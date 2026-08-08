"""
Synthetic HTML pages for the body-recovery tests.

:func:`newsletter_html` reproduces the shape that makes trafilatura's body
detector under-select (issue #21): several sibling ``<section>`` elements of
unequal size, each holding linked articles.  The extractor designates one of
them as the article body and drops the rest.  :func:`plain_article_html` is
the control: an ordinary single-article page that extracts cleanly.

Lives under ``tests/fixtures/`` so pytest does not collect it as a test
module (see ``tests/conftest.py``); import it from the test modules that need
a page with a known extraction behaviour.
"""

from __future__ import annotations

_WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey"
).split()

#: Titles of every article embedded in :func:`newsletter_html`, in page order.
ARTICLE_TITLES = [
    "Headline one",
    "Headline two",
    "Headline three",
    "Deep dive one",
    "Deep dive two",
    "Engineering one",
    "Engineering two",
    "Engineering three",
    "Engineering four",
]


def filler(seed: int, words: int = 40) -> str:
    """
    Build a deterministic sentence that is distinct for every *seed*.

    Distinctness matters: trafilatura deduplicates repeated paragraphs, so a
    page of identical filler would shrink the main extraction for a reason
    that has nothing to do with the bug under test.

    Parameters
    ----------
    seed : int
        Selects the word sequence.
    words : int, optional
        Sentence length in words, by default 40.

    Returns
    -------
    str
        A capitalised, full-stopped sentence.
    """
    picked = (_WORDS[(seed * 7 + i * 13) % len(_WORDS)] for i in range(words))
    return " ".join(picked).capitalize() + "."


def _section(title: str, items: list[tuple[str, int]]) -> str:
    """
    Render one newsletter section as HTML.

    Parameters
    ----------
    title : str
        Section heading.
    items : list[tuple[str, int]]
        ``(article title, paragraph length in words)`` pairs.

    Returns
    -------
    str
        A ``<section>`` element holding one ``<article>`` per item.
    """
    articles = "".join(
        f'<article class="mt-3"><a class="font-bold" href="https://example.com/{i}">'
        f"<h3>{name}</h3></a>"
        f'<div class="newsletter-html"><p>{filler(i + len(title), length)}</p></div>'
        "</article>"
        for i, (name, length) in enumerate(items)
    )
    return f"<section><header><div>x</div><h3>{title}</h3></header>{articles}</section>"


def newsletter_html() -> str:
    """
    Build a multi-section page whose body detector under-selects.

    Returns
    -------
    str
        A complete HTML document holding every title in
        :data:`ARTICLE_TITLES`.
    """
    sections = (
        _section("Headlines", [(t, 40) for t in ARTICLE_TITLES[:3]])
        + _section("Deep dives", [(t, 45) for t in ARTICLE_TITLES[3:5]])
        + _section("Engineering", [(t, 60) for t in ARTICLE_TITLES[5:]])
    )
    return (
        "<!DOCTYPE html><html><head><title>Daily digest</title></head><body>"
        '<nav><a href="/archive">Archive</a></nav>'
        '<div class="content"><h1>Daily digest</h1>'
        f"{sections}</div>"
        '<footer><a href="/privacy">Privacy</a></footer></body></html>'
    )


def plain_article_html() -> str:
    """
    Build an ordinary single-article page that trafilatura extracts cleanly.

    Returns
    -------
    str
        A complete HTML document.
    """
    paragraphs = "".join(f"<p>{filler(i, 60)}</p>" for i in range(6))
    return (
        "<!DOCTYPE html><html><head><title>An article</title></head><body>"
        '<nav><a href="/">Home</a></nav>'
        f"<article><h1>An article</h1>{paragraphs}</article>"
        "<footer>Footer text</footer></body></html>"
    )
