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


# ---------------------------------------------------------------------------
# --duo auto path
# ---------------------------------------------------------------------------

def _make_generated_duo(s1_name: str = "Mira", s2_name: str = "Ravi") -> dict:
    """
    Build a minimal dict that ``generate_duo`` would return.

    Uses the first two voices from GEMINI_VOICES so the names are valid.
    """
    from tts_podcast.duos import GEMINI_VOICES

    voices = list(GEMINI_VOICES)
    return {
        "description": "A thoughtful duo for the content.",
        "speaker1": {
            "name": s1_name,
            "voice": voices[0],
            "personality": "calm and analytical",
        },
        "speaker2": {
            "name": s2_name,
            "voice": voices[1],
            "personality": "warm and curious",
        },
    }


class TestDuoAuto:
    """
    ``--duo auto`` defers duo generation until after scraping + research
    and injects into ``gemini_cfg`` at the single injection point.
    """

    def _base_patches(self):
        """Shared mocks for the standard run pipeline (no audio write)."""
        return (
            patch("tts_podcast.cli.scrape_urls", return_value=[_fake_source()]),
            patch("tts_podcast.cli.conduct_research", return_value=ResearchReport()),
            patch("tts_podcast.cli.generate_dialogue", return_value=[]),
            patch("tts_podcast.cli.generate_audio_chunks", return_value=[]),
        )

    def test_generate_duo_called_when_auto(self, cli_env):
        """``generate_duo`` must be invoked exactly once when ``--duo auto`` is passed."""
        runner, config_path = cli_env
        scrape, research, dialogue, tts = self._base_patches()
        with scrape, research, dialogue, tts, \
             patch(
                 "tts_podcast.cli.generate_duo",
                 return_value=_make_generated_duo(),
             ) as mock_gen_duo:
            result = runner.invoke(
                cli,
                [
                    "run",
                    "-c", str(config_path),
                    "-A",
                    "-n",
                    "--duo", "auto",
                    "https://example.com/article",
                ],
            )
        assert result.exit_code == 0, result.output
        assert mock_gen_duo.called, "generate_duo must be called in auto mode"
        assert mock_gen_duo.call_count == 1

    def test_generate_duo_not_called_for_named_duo(self, cli_env):
        """In non-auto mode ``generate_duo`` must never be called."""
        runner, config_path = cli_env
        scrape, research, dialogue, tts = self._base_patches()
        with scrape, research, dialogue, tts, \
             patch("tts_podcast.cli.generate_duo") as mock_gen_duo:
            result = runner.invoke(
                cli,
                [
                    "run",
                    "-c", str(config_path),
                    "-A",
                    "-n",
                    "--duo", "warm",
                    "https://example.com/article",
                ],
            )
        assert result.exit_code == 0, result.output
        assert not mock_gen_duo.called, "generate_duo must not be called for named duo"

    def test_generate_duo_not_called_without_duo_flag(self, cli_env):
        """Without any ``--duo`` flag, ``generate_duo`` must not be called."""
        runner, config_path = cli_env
        scrape, research, dialogue, tts = self._base_patches()
        with scrape, research, dialogue, tts, \
             patch("tts_podcast.cli.generate_duo") as mock_gen_duo:
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
        assert not mock_gen_duo.called

    def test_speaker_names_recalculated_from_generated_duo(self, cli_env):
        """
        After ``generate_duo`` injects names, ``generate_dialogue`` receives
        the generated speaker names, not the config defaults.
        """
        runner, config_path = cli_env
        captured: dict = {}

        generated = _make_generated_duo(s1_name="Mira", s2_name="Ravi")

        def _capture_dialogue(_articles, gemini_cfg, speaker1_name, speaker2_name, **_kw):
            captured["s1"] = speaker1_name
            captured["s2"] = speaker2_name
            return []

        scrape, research, dialogue, tts = self._base_patches()
        with scrape, research, \
             patch("tts_podcast.cli.generate_dialogue", side_effect=_capture_dialogue), \
             tts, \
             patch("tts_podcast.cli.generate_duo", return_value=generated):
            result = runner.invoke(
                cli,
                [
                    "run",
                    "-c", str(config_path),
                    "-A",
                    "-n",
                    "--duo", "auto",
                    "https://example.com/article",
                ],
            )
        assert result.exit_code == 0, result.output
        assert captured["s1"] == "Mira"
        assert captured["s2"] == "Ravi"

    def test_auto_duo_injects_into_gemini_cfg_speaker_blocks(self, cli_env):
        """
        ``generate_duo``'s result is written into ``gemini_cfg["speaker1/2"]``
        (the single injection point), so ``generate_dialogue`` sees the new voices.
        """
        runner, config_path = cli_env
        captured: dict = {}

        generated = _make_generated_duo()

        def _capture_dialogue(_articles, gemini_cfg, *_args, **_kw):
            captured["voice1"] = gemini_cfg["speaker1"]["voice"]
            captured["voice2"] = gemini_cfg["speaker2"]["voice"]
            return []

        scrape, research, _, tts = self._base_patches()
        with scrape, research, \
             patch("tts_podcast.cli.generate_dialogue", side_effect=_capture_dialogue), \
             tts, \
             patch("tts_podcast.cli.generate_duo", return_value=generated):
            result = runner.invoke(
                cli,
                [
                    "run",
                    "-c", str(config_path),
                    "-A",
                    "-n",
                    "--duo", "auto",
                    "https://example.com/article",
                ],
            )
        assert result.exit_code == 0, result.output
        from tts_podcast.duos import GEMINI_VOICES

        assert captured["voice1"] in GEMINI_VOICES
        assert captured["voice2"] in GEMINI_VOICES
        # Voices must match what generate_duo returned.
        assert captured["voice1"] == generated["speaker1"]["voice"]
        assert captured["voice2"] == generated["speaker2"]["voice"]

    def test_auto_duo_does_not_mutate_personality(self, cli_env):
        """
        The personality strings from ``generate_duo`` reach ``generate_dialogue``
        as-is via ``gemini_cfg``; the TTS preamble path reads ``personality``
        verbatim and must never see a mutated value.
        """
        runner, config_path = cli_env
        captured: dict = {}
        generated = _make_generated_duo()
        original_p1 = generated["speaker1"]["personality"]
        original_p2 = generated["speaker2"]["personality"]

        def _capture_dialogue(_articles, gemini_cfg, *_args, **_kw):
            captured["p1"] = gemini_cfg["speaker1"]["personality"]
            captured["p2"] = gemini_cfg["speaker2"]["personality"]
            return []

        scrape, research, _, tts = self._base_patches()
        with scrape, research, \
             patch("tts_podcast.cli.generate_dialogue", side_effect=_capture_dialogue), \
             tts, \
             patch("tts_podcast.cli.generate_duo", return_value=generated):
            result = runner.invoke(
                cli,
                [
                    "run",
                    "-c", str(config_path),
                    "-A",
                    "-n",
                    "--duo", "auto",
                    "https://example.com/article",
                ],
            )
        assert result.exit_code == 0, result.output
        # personality must be the verbatim string from the generated duo dict.
        assert captured["p1"] == original_p1
        assert captured["p2"] == original_p2
        # The generated dict itself must not have been mutated.
        assert generated["speaker1"]["personality"] == original_p1
        assert generated["speaker2"]["personality"] == original_p2
