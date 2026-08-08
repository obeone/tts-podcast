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


class TestReportOptIn:
    """The report folder is now opt-in via --report (off by default)."""

    def _patches(self):
        """Mock the whole audio path so the run reaches the export/report stage."""
        return (
            patch("tts_podcast.cli.scrape_urls", return_value=[_fake_source()]),
            patch("tts_podcast.cli._check_ffmpeg"),
            patch("tts_podcast.cli.generate_dialogue", return_value=[]),
            patch("tts_podcast.cli.generate_audio_chunks", return_value=[b"pcm"]),
            patch("tts_podcast.cli.export_audio", return_value=Path("episode.mp3")),
            patch("tts_podcast.cli.generate_report", return_value=Path("tts_x")),
        )

    def test_report_omitted_by_default(self, cli_env):
        runner, config_path = cli_env
        scrape, ffmpeg, dialogue, tts, export, report = self._patches()
        with scrape, ffmpeg, dialogue, tts, export, report as mock_report:
            result = runner.invoke(
                cli,
                ["run", "-c", str(config_path), "https://example.com/article"],
            )
        assert result.exit_code == 0, result.output
        assert not mock_report.called, "report folder must not be generated by default"

    def test_report_generated_with_flag(self, cli_env):
        runner, config_path = cli_env
        scrape, ffmpeg, dialogue, tts, export, report = self._patches()
        with scrape, ffmpeg, dialogue, tts, export, report as mock_report:
            result = runner.invoke(
                cli,
                ["run", "-c", str(config_path), "--report", "https://example.com/article"],
            )
        assert result.exit_code == 0, result.output
        assert mock_report.called, "--report should generate the report folder"


class TestOutputFile:
    """`--output` chooses the audio filename, or streams to stdout with `-`."""

    def _patches(self):
        return (
            patch("tts_podcast.cli.scrape_urls", return_value=[_fake_source()]),
            patch("tts_podcast.cli._check_ffmpeg"),
            patch("tts_podcast.cli.generate_dialogue", return_value=[]),
            patch("tts_podcast.cli.generate_audio_chunks", return_value=[b"pcm"]),
        )

    def test_bare_name_routed_to_output_dir(self, cli_env):
        runner, config_path = cli_env
        scrape, ffmpeg, dialogue, tts = self._patches()
        with scrape, ffmpeg, dialogue, tts, \
             patch("tts_podcast.cli.export_audio", return_value=Path("show.mp3")) as mock_export:
            result = runner.invoke(
                cli,
                ["run", "-c", str(config_path), "-O", "show.mp3", "https://example.com/article"],
            )
        assert result.exit_code == 0, result.output
        # output_dir is "." in the test config → bare name lands there.
        assert Path(mock_export.call_args.args[1]) == Path("show.mp3")
        assert mock_export.call_args.kwargs["fmt"] == "mp3"

    def test_extension_drives_format(self, cli_env):
        runner, config_path = cli_env
        scrape, ffmpeg, dialogue, tts = self._patches()
        with scrape, ffmpeg, dialogue, tts, \
             patch("tts_podcast.cli.export_audio", return_value=Path("show.wav")) as mock_export:
            result = runner.invoke(
                cli,
                ["run", "-c", str(config_path), "-O", "show.wav", "https://example.com/article"],
            )
        assert result.exit_code == 0, result.output
        assert mock_export.call_args.kwargs["fmt"] == "wav"

    def test_dash_streams_to_stdout(self, cli_env):
        runner, config_path = cli_env
        scrape, ffmpeg, dialogue, tts = self._patches()
        with scrape, ffmpeg, dialogue, tts, \
             patch("tts_podcast.cli.encode_audio", return_value=b"BINARYAUDIO") as mock_encode, \
             patch("tts_podcast.cli.export_audio") as mock_export:
            result = runner.invoke(
                cli,
                ["run", "-c", str(config_path), "-O", "-", "https://example.com/article"],
            )
        assert result.exit_code == 0, result.output
        assert mock_encode.called, "stdout mode must encode in memory"
        assert not mock_export.called, "stdout mode must not write a file"
        assert b"BINARYAUDIO" in result.stdout_bytes


class TestChunkBudgetResolvedBeforeBilling:
    """
    The chunk byte budget reads only the resolved config (speakers, language,
    ``tts_style``), never the generated text, so the CLI resolves it before the
    research and dialogue stages run.

    That ordering is the whole point of the over-budget warning: resolved at
    the point of use it would land after the user has already paid for research
    and for the dialogue, which is exactly the failure it was meant to replace.
    """

    def test_budget_is_resolved_before_research_runs(self, cli_env):
        runner, config_path = cli_env
        order: list[str] = []

        def _resolve(gemini_cfg):
            order.append("budget")
            return 2500

        def _research(*_args, **_kwargs):
            order.append("research")
            return ResearchReport()

        with patch("tts_podcast.cli.scrape_urls", return_value=[_fake_source()]), \
             patch("tts_podcast.cli._resolve_chunk_budget", side_effect=_resolve), \
             patch("tts_podcast.cli.conduct_research", side_effect=_research), \
             patch("tts_podcast.cli.generate_dialogue", return_value=[]), \
             patch("tts_podcast.cli.generate_audio_chunks", return_value=[]):
            result = runner.invoke(
                cli,
                [
                    "run",
                    "-c", str(config_path),
                    "-R", "1",
                    "-A",
                    "-n",
                    "https://example.com/article",
                ],
            )
        assert result.exit_code == 0, result.output
        assert order == ["budget", "research"], (
            f"Expected the budget to be resolved before research, got {order!r}."
        )

    def test_resolved_budget_is_passed_to_generate_dialogue(self, cli_env):
        runner, config_path = cli_env
        captured: dict = {}

        def _capture(_articles, _gemini_cfg, *_args, **kwargs):
            captured["max_bytes"] = kwargs.get("max_bytes")
            return []

        with patch("tts_podcast.cli.scrape_urls", return_value=[_fake_source()]), \
             patch("tts_podcast.cli._resolve_chunk_budget", return_value=2345), \
             patch("tts_podcast.cli.conduct_research", return_value=ResearchReport()), \
             patch("tts_podcast.cli.generate_dialogue", side_effect=_capture), \
             patch("tts_podcast.cli.generate_audio_chunks", return_value=[]):
            result = runner.invoke(
                cli,
                ["run", "-c", str(config_path), "-A", "-n", "https://example.com/article"],
            )
        assert result.exit_code == 0, result.output
        assert captured["max_bytes"] == 2345, (
            "A budget computed and then not threaded through is the same silent "
            "failure as a Gemini call that forgets its token_tracker."
        )

    def test_budget_reflects_the_duo_supplied_tts_style(self, cli_env):
        # The duo fills tts_style.scene / pace just above the resolution point.
        # Resolving before that fill would measure a preamble the run never
        # sends and hand out a budget that is too generous.
        runner, config_path = cli_env
        captured: dict = {}

        def _capture_cfg(gemini_cfg):
            captured["scene"] = gemini_cfg.get("tts_style", {}).get("scene")
            captured["voice_direction"] = gemini_cfg["speaker1"].get("voice_direction")
            return 2500

        with patch("tts_podcast.cli.scrape_urls", return_value=[_fake_source()]), \
             patch("tts_podcast.cli._resolve_chunk_budget", side_effect=_capture_cfg), \
             patch("tts_podcast.cli.conduct_research", return_value=ResearchReport()), \
             patch("tts_podcast.cli.generate_dialogue", return_value=[]), \
             patch("tts_podcast.cli.generate_audio_chunks", return_value=[]):
            result = runner.invoke(
                cli,
                [
                    "run",
                    "-c", str(config_path),
                    "--duo", "explorer",
                    "-A",
                    "-n",
                    "https://example.com/article",
                ],
            )
        assert result.exit_code == 0, result.output
        assert captured["scene"], "Duo scene had not been applied yet at resolution time."
        assert captured["voice_direction"], (
            "Duo voice_direction had not been applied yet at resolution time."
        )


class TestFollowCapFlags:
    """`--follow-max-links[-per-hop]` override the config caps, and are validated."""

    def _config_with_follow(self, tmp_path: Path, block: str) -> Path:
        """Write the base config with an extra top-level ``follow:`` block appended."""
        path = _write_config(tmp_path)
        path.write_text(
            path.read_text(encoding="utf-8") + textwrap.dedent(block),
            encoding="utf-8",
        )
        return path

    def _run(self, runner, config_path, extra_args):
        """Invoke ``run`` with the follow stage mocked, returning (result, mock)."""
        with patch("tts_podcast.cli.scrape_urls", return_value=[_fake_source()]), \
             patch("tts_podcast.link_follower.follow_links", return_value=[]) as mock_follow, \
             patch("tts_podcast.cli.generate_dialogue", return_value=[]), \
             patch("tts_podcast.cli.generate_audio_chunks", return_value=[]):
            result = runner.invoke(
                cli,
                [
                    "run", "-c", str(config_path), "-A", "-n",
                    *extra_args,
                    "https://example.com/article",
                ],
            )
        return result, mock_follow

    def test_defaults_when_config_and_flags_are_silent(self, cli_env):
        runner, config_path = cli_env
        result, mock_follow = self._run(runner, config_path, ["-L"])
        assert result.exit_code == 0, result.output
        kwargs = mock_follow.call_args.kwargs
        assert kwargs["max_links_per_level"] == 5
        assert kwargs["max_links_total"] == 20

    def test_config_values_are_used(self, cli_env, tmp_path):
        runner, _ = cli_env
        config_path = self._config_with_follow(tmp_path, """
            follow:
              max_links_per_level: 3
              max_links_total: 7
            """)
        result, mock_follow = self._run(runner, config_path, ["-L"])
        assert result.exit_code == 0, result.output
        kwargs = mock_follow.call_args.kwargs
        assert kwargs["max_links_per_level"] == 3
        assert kwargs["max_links_total"] == 7

    def test_flags_beat_config(self, cli_env, tmp_path):
        runner, _ = cli_env
        config_path = self._config_with_follow(tmp_path, """
            follow:
              max_links_per_level: 3
              max_links_total: 7
            """)
        result, mock_follow = self._run(
            runner,
            config_path,
            ["-L", "--follow-max-links", "8", "--follow-max-links-per-hop", "4"],
        )
        assert result.exit_code == 0, result.output
        kwargs = mock_follow.call_args.kwargs
        assert kwargs["max_links_per_level"] == 4
        assert kwargs["max_links_total"] == 8

    @pytest.mark.parametrize("flag", ["--follow-max-links", "--follow-max-links-per-hop"])
    def test_non_positive_flag_exits_1(self, cli_env, flag):
        runner, config_path = cli_env
        result, mock_follow = self._run(runner, config_path, ["-L", flag, "0"])
        assert result.exit_code == 1
        assert flag in result.output
        assert not mock_follow.called, "an invalid cap must abort before any fetching"

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("max_links_total", "0"),
            ("max_links_per_level", "-1"),
            ("max_links_total", "'a handful'"),
        ],
    )
    def test_invalid_config_value_exits_1(self, cli_env, tmp_path, key, value):
        runner, _ = cli_env
        config_path = self._config_with_follow(tmp_path, f"""
            follow:
              {key}: {value}
            """)
        result, mock_follow = self._run(runner, config_path, ["-L"])
        assert result.exit_code == 1
        # The message must name the config key, not the CLI flag: the user has
        # nothing to fix on the command line here.
        assert f"follow.{key}" in result.output
        assert not mock_follow.called

    def test_caps_without_follow_links_do_nothing(self, cli_env):
        runner, config_path = cli_env
        result, mock_follow = self._run(runner, config_path, ["--follow-max-links", "8"])
        assert result.exit_code == 0, result.output
        assert not mock_follow.called, "the follow stage must stay off without -L"
