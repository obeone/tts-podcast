"""
Tests for the named voice-duo system.

Two layers:
* unit tests for :mod:`tts_podcast.duos` (resolution, validation, merging);
* CLI integration tests for the ``--duo`` / ``gemini.default_duo`` precedence
  and backward-compatibility with legacy ``gemini.speakerN`` blocks, with every
  collaborator mocked at the ``tts_podcast.cli`` boundary.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from tts_podcast.cli import cli
from tts_podcast.duos import (
    BUILTIN_DUOS,
    DEFAULT_DUO,
    available_duos,
    describe_duos,
    resolve_duo,
)
from tts_podcast.models import Source
from tts_podcast.research import ResearchReport


# ---------------------------------------------------------------------------
# Unit tests — tts_podcast.duos
# ---------------------------------------------------------------------------


class TestResolveDuo:
    def test_builtin_default_resolves_to_contrast_pairing(self):
        resolved = resolve_duo(DEFAULT_DUO)
        assert resolved is not None
        assert resolved["speaker1"]["voice"] == "Puck"
        assert resolved["speaker2"]["voice"] == "Kore"

    @pytest.mark.parametrize("name", [None, "", "   "])
    def test_blank_name_returns_none(self, name):
        # None / empty mean "no duo selected" so the caller can fall back to
        # legacy speakerN blocks.
        assert resolve_duo(name) is None

    def test_unknown_name_raises_listing_valid(self):
        with pytest.raises(click.BadParameter) as exc:
            resolve_duo("does-not-exist")
        msg = exc.value.format_message()
        assert "does-not-exist" in msg
        # Every built-in slug is offered as a valid choice.
        for slug in BUILTIN_DUOS:
            assert slug in msg

    def test_case_insensitive(self):
        assert resolve_duo("WARM") == resolve_duo("warm")

    def test_returns_deepcopy_not_shared_state(self):
        resolved = resolve_duo("warm")
        resolved["speaker1"]["voice"] = "Mutated"
        # The built-in registry must be untouched by caller mutation.
        assert BUILTIN_DUOS["warm"]["speaker1"]["voice"] == "Sulafat"
        assert resolve_duo("warm")["speaker1"]["voice"] == "Sulafat"

    def test_config_duo_overrides_builtin_same_slug(self):
        config_duos = {
            "warm": {
                "speaker1": {"name": "A", "voice": "Kore", "personality": "x"},
                "speaker2": {"name": "B", "voice": "Puck", "personality": "y"},
            }
        }
        resolved = resolve_duo("warm", config_duos)
        assert resolved["speaker1"]["voice"] == "Kore"
        assert resolved["speaker2"]["voice"] == "Puck"

    def test_config_duo_extends_with_new_slug(self):
        config_duos = {
            "custom": {
                "speaker1": {"name": "A", "voice": "Zephyr", "personality": "x"},
                "speaker2": {"name": "B", "voice": "Algenib", "personality": "y"},
            }
        }
        resolved = resolve_duo("custom", config_duos)
        assert resolved["speaker1"]["voice"] == "Zephyr"

    def test_config_duo_missing_required_field_raises(self):
        config_duos = {
            "broken": {
                "speaker1": {"name": "A", "personality": "x"},  # no voice
                "speaker2": {"name": "B", "voice": "Puck", "personality": "y"},
            }
        }
        with pytest.raises(click.BadParameter):
            resolve_duo("broken", config_duos)

    def test_config_duos_not_a_mapping_raises(self):
        with pytest.raises(click.BadParameter):
            available_duos(["not", "a", "mapping"])  # type: ignore[arg-type]


class TestDescribeDuos:
    def test_builtins_listed_first_in_declaration_order(self):
        rows = describe_duos()
        slugs = [row[0] for row in rows]
        assert slugs == list(BUILTIN_DUOS.keys())

    def test_config_only_slugs_appended_after_builtins(self):
        config_duos = {
            "zzz_custom": {
                "speaker1": {"name": "A", "voice": "Zephyr", "personality": "x"},
                "speaker2": {"name": "B", "voice": "Algenib", "personality": "y"},
            }
        }
        rows = describe_duos(config_duos)
        slugs = [row[0] for row in rows]
        assert slugs[: len(BUILTIN_DUOS)] == list(BUILTIN_DUOS.keys())
        assert slugs[-1] == "zzz_custom"

    def test_speaker_summary_format(self):
        rows = {row[0]: row for row in describe_duos()}
        _, _, sp1, sp2 = rows["debate"]
        assert sp1 == "Robin (Laomedeia)"
        assert sp2 == "Sasha (Algenib)"


# ---------------------------------------------------------------------------
# CLI integration tests — duo precedence & backward compatibility
# ---------------------------------------------------------------------------


def _config(tmp_path: Path, voices_block: str) -> Path:
    """Write a minimal config whose gemini voices section is *voices_block*."""
    cfg = textwrap.dedent(
        """\
        web:
          user_agent: TestUA
          timeout_seconds: 5
        gemini:
          api_key_env: TTS_TEST_API_KEY
          text_model: gemini-2.5-flash
          tts_model: gemini-2.5-flash-preview-tts
          language: French
        {voices}
        research:
          rounds_default: 0
        scraping:
          timeout_seconds: 5
        output:
          dir: "."
          format: mp3
        pricing: {{}}
        """
    ).format(voices=textwrap.indent(textwrap.dedent(voices_block), "  "))
    path = tmp_path / "config.yaml"
    path.write_text(cfg, encoding="utf-8")
    return path


def _fake_source() -> Source:
    return Source(
        url="https://example.com/article",
        title="Test article",
        summary="summary",
        full_text="full text",
        scraped_ok=True,
        kind="url",
    )


def _run_capture_speakers(runner: CliRunner, config_path: Path, extra_args: list[str]):
    """Invoke `run` with mocked collaborators, capturing the resolved speakers."""
    captured: dict = {}

    def _capture(_articles, gemini_cfg, sp1_name, sp2_name, *args, **kwargs):
        captured["s1_name"] = sp1_name
        captured["s2_name"] = sp2_name
        captured["s1_voice"] = gemini_cfg["speaker1"]["voice"]
        captured["s2_voice"] = gemini_cfg["speaker2"]["voice"]
        captured["s1_overlay"] = gemini_cfg["speaker1"].get("style_overlay")
        return []

    with patch("tts_podcast.cli.scrape_urls", return_value=[_fake_source()]), \
         patch("tts_podcast.cli.conduct_research", return_value=ResearchReport()), \
         patch("tts_podcast.cli.generate_dialogue", side_effect=_capture), \
         patch("tts_podcast.cli.generate_audio_chunks", return_value=[]):
        result = runner.invoke(
            cli,
            ["run", "-c", str(config_path), "-A", "-n", *extra_args,
             "https://example.com/article"],
        )
    return result, captured


@pytest.fixture
def runner_env(monkeypatch):
    monkeypatch.setenv("TTS_TEST_API_KEY", "fake-key-for-tests")
    return CliRunner()


class TestDuoCliPrecedence:
    def test_builtin_default_when_no_speakers_no_duo(self, runner_env, tmp_path):
        # Neither default_duo nor legacy speakers → built-in 'contrast'.
        config_path = _config(tmp_path, "# no voices configured\n")
        result, cap = _run_capture_speakers(runner_env, config_path, [])
        assert result.exit_code == 0, result.output
        assert (cap["s1_voice"], cap["s2_voice"]) == ("Puck", "Kore")
        assert (cap["s1_name"], cap["s2_name"]) == ("Theo", "Nadia")

    def test_legacy_speakers_preserved_when_no_duo(self, runner_env, tmp_path):
        # Backward compat: a config with only speaker1/speaker2 is untouched.
        config_path = _config(
            tmp_path,
            """\
            speaker1:
              name: Old1
              voice: Puck
              personality: legacy one
            speaker2:
              name: Old2
              voice: Charon
              personality: legacy two
            """,
        )
        result, cap = _run_capture_speakers(runner_env, config_path, [])
        assert result.exit_code == 0, result.output
        assert (cap["s1_voice"], cap["s2_voice"]) == ("Puck", "Charon")
        assert (cap["s1_name"], cap["s2_name"]) == ("Old1", "Old2")

    def test_default_duo_from_config(self, runner_env, tmp_path):
        config_path = _config(tmp_path, "default_duo: contrast\n")
        result, cap = _run_capture_speakers(runner_env, config_path, [])
        assert result.exit_code == 0, result.output
        assert (cap["s1_voice"], cap["s2_voice"]) == ("Puck", "Kore")
        assert (cap["s1_name"], cap["s2_name"]) == ("Theo", "Nadia")

    def test_cli_duo_overrides_default_duo(self, runner_env, tmp_path):
        config_path = _config(tmp_path, "default_duo: contrast\n")
        result, cap = _run_capture_speakers(runner_env, config_path, ["--duo", "journalist"])
        assert result.exit_code == 0, result.output
        assert (cap["s1_voice"], cap["s2_voice"]) == ("Zephyr", "Algieba")

    def test_cli_duo_overrides_legacy_speakers(self, runner_env, tmp_path):
        config_path = _config(
            tmp_path,
            """\
            speaker1:
              name: Old1
              voice: Puck
              personality: legacy one
            speaker2:
              name: Old2
              voice: Charon
              personality: legacy two
            """,
        )
        result, cap = _run_capture_speakers(runner_env, config_path, ["--duo", "debate"])
        assert result.exit_code == 0, result.output
        assert (cap["s1_voice"], cap["s2_voice"]) == ("Laomedeia", "Algenib")
        assert (cap["s1_name"], cap["s2_name"]) == ("Robin", "Sasha")

    def test_invalid_duo_exits_with_error(self, runner_env, tmp_path):
        config_path = _config(tmp_path, "default_duo: warm\n")
        result, _ = _run_capture_speakers(runner_env, config_path, ["--duo", "bogus"])
        assert result.exit_code != 0
        assert "bogus" in result.output

    def test_speaker_style_overlay_composes_with_duo(self, runner_env, tmp_path):
        # --speakerN-style still lands on the duo-resolved speaker.
        config_path = _config(tmp_path, "default_duo: warm\n")
        result, cap = _run_capture_speakers(
            runner_env, config_path, ["--speaker1-style", "extra punchy"]
        )
        assert result.exit_code == 0, result.output
        assert cap["s1_voice"] == "Sulafat"
        assert cap["s1_overlay"] == "extra punchy"


class TestDuosCommand:
    def test_lists_all_builtins_with_default_marker(self, runner_env, tmp_path, monkeypatch):
        # Point the default-config lookup at an empty dir so no user config leaks in.
        monkeypatch.setattr("tts_podcast.cli._DEFAULT_CONFIG", tmp_path / "nope.yaml")
        result = runner_env.invoke(cli, ["duos"])
        assert result.exit_code == 0, result.output
        for slug in BUILTIN_DUOS:
            assert slug in result.output
        assert "[default]" in result.output
