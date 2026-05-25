"""
CLI integration tests.

Exercise the Click command surface with all collaborators mocked at the
``tts_podcast.cli`` module boundary so the tests stay hermetic.  They focus
specifically on the new style / overlay / angle flags introduced by the
``2026-05-23-style-and-angle-cli`` plan.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tts_podcast.cli import cli
from tts_podcast.models import Source
from tts_podcast.research import ResearchReport


def _write_config(tmp_path: Path) -> Path:
    """
    Drop a minimal YAML config into *tmp_path* and return its absolute path.

    The wizard's pricing block is omitted to keep the file short; the cost
    summary will simply have no rate entries.
    """
    cfg = textwrap.dedent("""\
        web:
          user_agent: TestUA
          timeout_seconds: 5
        gemini:
          api_key_env: TTS_TEST_API_KEY
          text_model: gemini-2.5-flash
          tts_model: gemini-2.5-flash-preview-tts
          language: French
          speaker1:
            name: Alex
            voice: Puck
            personality: "calm and curious"
          speaker2:
            name: Jordan
            voice: Charon
            personality: "measured and analytical"
        research:
          rounds_default: 0
        scraping:
          timeout_seconds: 5
        output:
          dir: "."
          format: mp3
        pricing: {}
        """)
    path = tmp_path / "config.yaml"
    path.write_text(cfg, encoding="utf-8")
    return path


@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    """Provide a populated env var + tmp config + CliRunner ready to invoke."""
    monkeypatch.setenv("TTS_TEST_API_KEY", "fake-key-for-tests")
    config_path = _write_config(tmp_path)
    return CliRunner(), config_path


def _fake_source() -> Source:
    """A scraped source the mocked scrape_urls returns."""
    return Source(
        url="https://example.com/article",
        title="Test article",
        summary="summary",
        full_text="full text",
        scraped_ok=True,
        kind="url",
    )


class TestStyleFlagsWiring:
    """Verify CLI style flags reach the downstream functions correctly."""

    def test_angle_threaded_to_conduct_research(self, cli_env):
        runner, config_path = cli_env
        with patch("tts_podcast.cli.scrape_urls", return_value=[_fake_source()]), \
             patch("tts_podcast.cli.conduct_research", return_value=ResearchReport()) as mock_research, \
             patch("tts_podcast.cli.generate_dialogue", return_value=[]), \
             patch("tts_podcast.cli.generate_audio_chunks", return_value=[]):
            result = runner.invoke(
                cli,
                [
                    "run",
                    "-c", str(config_path),
                    "-R", "1",  # force research to run so we can assert the call
                    "-A",  # no audio
                    "-n",  # dry-run
                    "--angle", "the economic implications",
                    "https://example.com/article",
                ],
            )
        assert result.exit_code == 0, result.output
        assert mock_research.called, "conduct_research was not invoked"
        kwargs = mock_research.call_args.kwargs
        assert kwargs.get("angle") == "the economic implications"

    def test_speaker_style_does_not_mutate_personality(self, cli_env):
        runner, config_path = cli_env
        captured = {}

        def _capture_generate(_articles, gemini_cfg, *args, **kwargs):
            # Snapshot the personality keys *and* the new overlay key at call time.
            captured["speaker1_personality"] = gemini_cfg["speaker1"]["personality"]
            captured["speaker2_personality"] = gemini_cfg["speaker2"]["personality"]
            captured["speaker1_overlay"] = gemini_cfg["speaker1"].get("style_overlay")
            captured["speaker2_overlay"] = gemini_cfg["speaker2"].get("style_overlay")
            return []

        with patch("tts_podcast.cli.scrape_urls", return_value=[_fake_source()]), \
             patch("tts_podcast.cli.conduct_research", return_value=ResearchReport()), \
             patch("tts_podcast.cli.generate_dialogue", side_effect=_capture_generate), \
             patch("tts_podcast.cli.generate_audio_chunks", return_value=[]):
            result = runner.invoke(
                cli,
                [
                    "run",
                    "-c", str(config_path),
                    "-A",
                    "-n",
                    "--speaker1-style", "more skeptical than usual",
                    "--speaker2-style", "extra warm",
                    "https://example.com/article",
                ],
            )
        assert result.exit_code == 0, result.output
        # Baseline personalities preserved verbatim — TTS preamble stays clean.
        assert captured["speaker1_personality"] == "calm and curious"
        assert captured["speaker2_personality"] == "measured and analytical"
        # Overlay landed in the dedicated key only.
        assert captured["speaker1_overlay"] == "more skeptical than usual"
        assert captured["speaker2_overlay"] == "extra warm"

    def test_preset_and_style_reach_dialogue_via_gemini_cfg(self, cli_env):
        runner, config_path = cli_env
        captured = {}

        def _capture(_articles, gemini_cfg, *args, **kwargs):
            captured["style"] = gemini_cfg.get("style", {})
            return []

        with patch("tts_podcast.cli.scrape_urls", return_value=[_fake_source()]), \
             patch("tts_podcast.cli.conduct_research", return_value=ResearchReport()), \
             patch("tts_podcast.cli.generate_dialogue", side_effect=_capture), \
             patch("tts_podcast.cli.generate_audio_chunks", return_value=[]):
            result = runner.invoke(
                cli,
                [
                    "run",
                    "-c", str(config_path),
                    "-A",
                    "-n",
                    "--preset", "academic",
                    "--style", "extra dry",
                    "--angle", "regulatory",
                    "https://example.com/article",
                ],
            )
        assert result.exit_code == 0, result.output
        assert captured["style"]["preset"] == "academic"
        assert captured["style"]["text"] == "extra dry"
        assert captured["style"]["angle"] == "regulatory"

    def test_no_style_flags_leaves_gemini_cfg_untouched(self, cli_env):
        """When no new flags are passed, gemini.style and style_overlay are absent."""
        runner, config_path = cli_env
        captured = {}

        def _capture(_articles, gemini_cfg, *args, **kwargs):
            captured["has_style"] = "style" in gemini_cfg
            captured["speaker1_has_overlay"] = "style_overlay" in gemini_cfg["speaker1"]
            captured["speaker2_has_overlay"] = "style_overlay" in gemini_cfg["speaker2"]
            return []

        with patch("tts_podcast.cli.scrape_urls", return_value=[_fake_source()]), \
             patch("tts_podcast.cli.conduct_research", return_value=ResearchReport()), \
             patch("tts_podcast.cli.generate_dialogue", side_effect=_capture), \
             patch("tts_podcast.cli.generate_audio_chunks", return_value=[]):
            result = runner.invoke(
                cli,
                [
                    "run",
                    "-c", str(config_path),
                    "-A",
                    "-n",
                    "https://example.com/article",
                ],
            )
        assert result.exit_code == 0, result.output
        assert captured["has_style"] is False
        assert captured["speaker1_has_overlay"] is False
        assert captured["speaker2_has_overlay"] is False

    def test_preset_none_sentinel_clears_configured_preset(self, cli_env):
        """`--preset none` should write None into gemini.style.preset."""
        runner, config_path = cli_env
        captured = {}

        def _capture(_articles, gemini_cfg, *args, **kwargs):
            captured["preset"] = gemini_cfg.get("style", {}).get("preset")
            return []

        with patch("tts_podcast.cli.scrape_urls", return_value=[_fake_source()]), \
             patch("tts_podcast.cli.conduct_research", return_value=ResearchReport()), \
             patch("tts_podcast.cli.generate_dialogue", side_effect=_capture), \
             patch("tts_podcast.cli.generate_audio_chunks", return_value=[]):
            result = runner.invoke(
                cli,
                [
                    "run",
                    "-c", str(config_path),
                    "-A",
                    "-n",
                    "--preset", "none",
                    "https://example.com/article",
                ],
            )
        assert result.exit_code == 0, result.output
        # validate_preset() converts "none" to None when llm_summarizer reads it,
        # but at the CLI layer the string "none" is what gets stored — the
        # resolution happens later in _build_prompt.
        assert captured["preset"] == "none"

    def test_preset_none_resolves_to_no_stylistic_guidance_in_prompt(self, cli_env):
        """End-to-end: --preset none (sentinel) leaves no 'Stylistic guidance:' header in the prompt.

        Composition check that the CLI layer stores "none" and the
        ``_build_prompt`` layer resolves it back to None via ``validate_preset``.
        """
        runner, config_path = cli_env
        captured = {}

        def _capture(articles, gemini_cfg, speaker1_name, speaker2_name, **kwargs):
            # Build the actual prompt the way generate_dialogue would, so we
            # exercise the validate_preset("none") -> None resolution path.
            from tts_podcast.llm_summarizer import _build_prompt
            prompt = _build_prompt(
                articles=articles,
                speaker1_name=speaker1_name,
                speaker2_name=speaker2_name,
                preset=gemini_cfg.get("style", {}).get("preset"),
            )
            captured["prompt"] = prompt
            return []

        with patch("tts_podcast.cli.scrape_urls", return_value=[_fake_source()]), \
             patch("tts_podcast.cli.conduct_research", return_value=ResearchReport()), \
             patch("tts_podcast.cli.generate_dialogue", side_effect=_capture), \
             patch("tts_podcast.cli.generate_audio_chunks", return_value=[]):
            result = runner.invoke(
                cli,
                [
                    "run",
                    "-c", str(config_path),
                    "-A",
                    "-n",
                    "--preset", "none",
                    "https://example.com/article",
                ],
            )
        assert result.exit_code == 0, result.output
        assert "Stylistic guidance:" not in captured["prompt"]

    def test_preset_unknown_exits_2(self, cli_env):
        runner, config_path = cli_env
        result = runner.invoke(
            cli,
            [
                "run",
                "-c", str(config_path),
                "-A",
                "-n",
                "--preset", "nosuchpreset",
                "https://example.com/article",
            ],
        )
        assert result.exit_code == 2
        # Click's default Choice error embeds the invalid value and lists
        # the valid choices — both signals must surface.
        assert "nosuchpreset" in result.output
        assert "academic" in result.output  # at least one valid choice listed
