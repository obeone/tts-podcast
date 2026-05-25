"""
Regenerate the dialogue-prompt baseline fixture.

The fixture ``tests/fixtures/dialogue_prompt_no_overlay.txt`` is a frozen
snapshot of :func:`tts_podcast.llm_summarizer._build_prompt` rendered with a
fixed set of inputs and all new style/overlay/angle params at their defaults.
It is used by ``test_no_flags_byte_identical`` to guarantee that adding the
style/angle controls never accidentally drifts the prompt for runs that don't
use the new flags.

Run via::

    uv run python -m tests.fixtures.regen_dialogue_prompt

Regenerate intentionally when (and only when) ``_SYSTEM_PROMPT_TEMPLATE`` is
edited on purpose.  Review ``git diff`` on the fixture before committing.

The initial seed of ``dialogue_prompt_no_overlay.txt`` was rendered against
the **pre-refactor** ``_build_prompt`` — this script was created and run
*before* the style/overlay/angle placeholders were added to the template,
so the on-disk fixture is a true baseline of the original prompt shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tts_podcast.llm_summarizer import _build_prompt


@dataclass
class _FixtureArticle:
    """Minimal article-like stub used to seed the snapshot prompt."""

    title: str
    url: str
    summary: str
    full_text: str


_FIXTURE_ARTICLES = [
    _FixtureArticle(
        title="Rust hits 1.0 stability milestone",
        url="https://example.com/rust",
        summary="Rust language announces major stability improvements.",
        full_text=(
            "Rust language announces major stability improvements in version 1.0. "
            "The team highlights performance gains and a stronger guarantee of "
            "backwards compatibility going forward."
        ),
    ),
]

_FIXTURE_PATH = Path(__file__).parent / "dialogue_prompt_no_overlay.txt"


def regen() -> Path:
    """
    Render the baseline prompt and write it to the fixture file.

    Returns
    -------
    pathlib.Path
        Absolute path to the fixture that was (re)written.
    """
    prompt = _build_prompt(
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
    _FIXTURE_PATH.write_text(prompt, encoding="utf-8")
    return _FIXTURE_PATH


if __name__ == "__main__":
    path = regen()
    print(f"Wrote {path} ({path.stat().st_size} bytes)")
