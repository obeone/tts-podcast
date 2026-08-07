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
from tts_podcast.tts_generator import _build_tts_prompt


#: The 30 prebuilt voices offered by the Gemini speech-generation API, pinned
#: here as the reference set.  A built-in duo naming anything outside this set
#: fails at request time with an API error, so the check belongs in the suite
#: rather than in a reviewer's memory.
#: See https://ai.google.dev/gemini-api/docs/speech-generation
OFFICIAL_GEMINI_VOICES = frozenset(
    {
        "Achernar",
        "Achird",
        "Algenib",
        "Algieba",
        "Alnilam",
        "Aoede",
        "Autonoe",
        "Callirrhoe",
        "Charon",
        "Despina",
        "Enceladus",
        "Erinome",
        "Fenrir",
        "Gacrux",
        "Iapetus",
        "Kore",
        "Laomedeia",
        "Leda",
        "Orus",
        "Puck",
        "Pulcherrima",
        "Rasalgethi",
        "Sadachbia",
        "Sadaltager",
        "Schedar",
        "Sulafat",
        "Umbriel",
        "Vindemiatrix",
        "Zephyr",
        "Zubenelgenubi",
    }
)

#: Upper bound on a single ``voice_direction`` string, in characters.  The TTS
#: preamble byte budget (see ``tests/test_tts_generator.py``) was measured with
#: this envelope; longer notes require re-measuring and lowering
#: ``llm_summarizer._MAX_CHUNK_BYTES``.
MAX_VOICE_DIRECTION_CHARS = 160


def _voices_of(slug: str) -> tuple[str, str]:
    """
    Return the ``(speaker1, speaker2)`` voices declared by a built-in duo.

    Parameters
    ----------
    slug : str
        Built-in duo slug, e.g. ``"debate"``.

    Returns
    -------
    tuple[str, str]
        The two prebuilt Gemini voice names.
    """
    duo = BUILTIN_DUOS[slug]
    return duo["speaker1"]["voice"], duo["speaker2"]["voice"]


def _names_of(slug: str) -> tuple[str, str]:
    """
    Return the ``(speaker1, speaker2)`` host names declared by a built-in duo.

    Parameters
    ----------
    slug : str
        Built-in duo slug, e.g. ``"debate"``.

    Returns
    -------
    tuple[str, str]
        The two host display names.
    """
    duo = BUILTIN_DUOS[slug]
    return duo["speaker1"]["name"], duo["speaker2"]["name"]


# ---------------------------------------------------------------------------
# Unit tests — tts_podcast.duos
# ---------------------------------------------------------------------------


class TestResolveDuo:
    def test_builtin_default_resolves_to_its_registry_pairing(self):
        resolved = resolve_duo(DEFAULT_DUO)
        assert resolved is not None
        assert (resolved["speaker1"]["voice"], resolved["speaker2"]["voice"]) == _voices_of(
            DEFAULT_DUO
        )
        assert (resolved["speaker1"]["name"], resolved["speaker2"]["name"]) == _names_of(
            DEFAULT_DUO
        )

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
        original = BUILTIN_DUOS["warm"]["speaker1"]["voice"]
        resolved = resolve_duo("warm")
        resolved["speaker1"]["voice"] = "Mutated"
        # The built-in registry must be untouched by caller mutation.
        assert BUILTIN_DUOS["warm"]["speaker1"]["voice"] == original
        assert resolve_duo("warm")["speaker1"]["voice"] == original

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
        debate = BUILTIN_DUOS["debate"]
        assert sp1 == f"{debate['speaker1']['name']} ({debate['speaker1']['voice']})"
        assert sp2 == f"{debate['speaker2']['name']} ({debate['speaker2']['voice']})"


class TestVoiceDirection:
    """
    The optional per-speaker ``voice_direction`` key.

    It is a Director's Note (register, tempo, articulation, breathing) read
    only by the TTS path.  ``duos`` only has to carry it through untouched and
    reject a non-string value.
    """

    @pytest.mark.parametrize("slug", sorted(BUILTIN_DUOS))
    @pytest.mark.parametrize("role", ["speaker1", "speaker2"])
    def test_every_builtin_speaker_declares_a_voice_direction(self, slug, role):
        direction = BUILTIN_DUOS[slug][role].get("voice_direction")
        assert isinstance(direction, str) and direction.strip(), (
            f"Duo {slug!r} {role} has no voice_direction; both hosts of a duo need "
            "one or the model renders them with the same delivery."
        )

    @pytest.mark.parametrize("slug", sorted(BUILTIN_DUOS))
    def test_every_builtin_declares_scene_and_pace(self, slug):
        duo = BUILTIN_DUOS[slug]
        for field in ("scene", "pace"):
            value = duo.get(field)
            assert isinstance(value, str) and value.strip(), (
                f"Duo {slug!r} declares no {field!r}; the duo-level tts_style "
                "defaults are what give each duo its own room and tempo."
            )

    @pytest.mark.parametrize("slug", sorted(BUILTIN_DUOS))
    def test_resolve_duo_passes_voice_direction_through(self, slug):
        resolved = resolve_duo(slug)
        for role in ("speaker1", "speaker2"):
            assert (
                resolved[role]["voice_direction"]
                == BUILTIN_DUOS[slug][role]["voice_direction"]
            )

    @pytest.mark.parametrize("slug", sorted(BUILTIN_DUOS))
    def test_resolve_duo_surfaces_scene_and_pace(self, slug):
        resolved = resolve_duo(slug)
        assert resolved["scene"] == BUILTIN_DUOS[slug]["scene"]
        assert resolved["pace"] == BUILTIN_DUOS[slug]["pace"]

    def test_config_duo_voice_direction_passes_through(self):
        config_duos = {
            "custom": {
                "speaker1": {
                    "name": "A",
                    "voice": "Zephyr",
                    "personality": "x",
                    "voice_direction": "Low register, slow, legato, deep breaths.",
                },
                "speaker2": {"name": "B", "voice": "Algenib", "personality": "y"},
            }
        }
        resolved = resolve_duo("custom", config_duos)
        assert resolved["speaker1"]["voice_direction"] == (
            "Low register, slow, legato, deep breaths."
        )

    def test_duo_without_voice_direction_still_resolves(self):
        # Legacy shape: a user duo written before voice_direction existed must
        # keep resolving, and must not gain a synthesised direction.
        config_duos = {
            "legacy": {
                "speaker1": {"name": "A", "voice": "Zephyr", "personality": "x"},
                "speaker2": {"name": "B", "voice": "Algenib", "personality": "y"},
            }
        }
        resolved = resolve_duo("legacy", config_duos)
        assert resolved["speaker1"]["voice"] == "Zephyr"
        assert "voice_direction" not in resolved["speaker1"]
        assert "voice_direction" not in resolved["speaker2"]

    def test_duo_without_scene_or_pace_stays_silent(self):
        # Absent means absent: the caller must be able to tell "the duo has no
        # opinion" from "the duo says empty", so no key is added.
        config_duos = {
            "legacy": {
                "speaker1": {"name": "A", "voice": "Zephyr", "personality": "x"},
                "speaker2": {"name": "B", "voice": "Algenib", "personality": "y"},
            }
        }
        resolved = resolve_duo("legacy", config_duos)
        assert "scene" not in resolved
        assert "pace" not in resolved

    def test_blank_scene_is_treated_as_unset(self):
        config_duos = {
            "blank": {
                "scene": "   ",
                "speaker1": {"name": "A", "voice": "Zephyr", "personality": "x"},
                "speaker2": {"name": "B", "voice": "Algenib", "personality": "y"},
            }
        }
        assert "scene" not in resolve_duo("blank", config_duos)

    def test_non_string_voice_direction_raises(self):
        config_duos = {
            "broken": {
                "speaker1": {
                    "name": "A",
                    "voice": "Zephyr",
                    "personality": "x",
                    "voice_direction": 42,
                },
                "speaker2": {"name": "B", "voice": "Algenib", "personality": "y"},
            }
        }
        with pytest.raises(click.BadParameter) as exc:
            resolve_duo("broken", config_duos)
        assert "voice_direction" in exc.value.format_message()

    def test_non_string_scene_raises(self):
        config_duos = {
            "broken": {
                "scene": ["not", "a", "string"],
                "speaker1": {"name": "A", "voice": "Zephyr", "personality": "x"},
                "speaker2": {"name": "B", "voice": "Algenib", "personality": "y"},
            }
        }
        with pytest.raises(click.BadParameter) as exc:
            resolve_duo("broken", config_duos)
        assert "scene" in exc.value.format_message()


class TestAcousticHygiene:
    """
    The contract that keeps the five built-in duos from sounding alike.

    The reported symptom was "the voices of the duos tend to sound the same
    every time": duos reusing the same voice, the same host name, or a voice
    that does not exist are all ways back into it.
    """

    def test_no_voice_repeated_across_builtins(self):
        voices = [
            duo[role]["voice"]
            for duo in BUILTIN_DUOS.values()
            for role in ("speaker1", "speaker2")
        ]
        duplicates = sorted({v for v in voices if voices.count(v) > 1})
        assert not duplicates, f"Voices reused across built-in duos: {duplicates}"
        assert len(voices) == 2 * len(BUILTIN_DUOS)

    def test_no_first_name_repeated_across_builtins(self):
        names = [
            duo[role]["name"]
            for duo in BUILTIN_DUOS.values()
            for role in ("speaker1", "speaker2")
        ]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        assert not duplicates, f"Host names reused across built-in duos: {duplicates}"

    @pytest.mark.parametrize("slug", sorted(BUILTIN_DUOS))
    def test_every_builtin_voice_is_an_official_gemini_voice(self, slug):
        for role in ("speaker1", "speaker2"):
            voice = BUILTIN_DUOS[slug][role]["voice"]
            assert voice in OFFICIAL_GEMINI_VOICES, (
                f"Duo {slug!r} {role} uses voice {voice!r}, which is not one of the "
                "30 prebuilt Gemini voices."
            )

    @pytest.mark.parametrize("slug", sorted(BUILTIN_DUOS))
    def test_voice_directions_stay_within_the_measured_budget(self, slug):
        # The preamble byte budget was measured against this envelope; growing
        # a direction beyond it invalidates _MAX_CHUNK_BYTES.
        for role in ("speaker1", "speaker2"):
            direction = BUILTIN_DUOS[slug][role]["voice_direction"]
            assert len(direction) <= MAX_VOICE_DIRECTION_CHARS, (
                f"Duo {slug!r} {role} voice_direction is {len(direction)} chars, over "
                f"the {MAX_VOICE_DIRECTION_CHARS}-char budget."
            )

    @pytest.mark.parametrize("slug", sorted(BUILTIN_DUOS))
    def test_the_two_directions_of_a_duo_differ(self, slug):
        duo = BUILTIN_DUOS[slug]
        assert duo["speaker1"]["voice_direction"] != duo["speaker2"]["voice_direction"]


class TestPersonalityGrammar:
    """
    A personality string has two consumers with two different sentence frames.

    The TTS preamble renders ``"{name} is {personality}."`` and the dialogue
    prompt renders ``"- {name}: {personality}"``.  A bare job title reads fine
    as a bullet but ships ``"Nora is desk anchor;"`` to the TTS model, and a
    garbled character description is what makes it fall back on one averaged
    delivery — the very bug the duos rework exists to fix.
    """

    #: Determiners a personality may open with so ``"{name} is …"`` parses.
    _DETERMINERS = ("a ", "an ", "the ")

    @pytest.mark.parametrize("slug", sorted(BUILTIN_DUOS))
    def test_personality_reads_as_a_noun_phrase(self, slug):
        for role in ("speaker1", "speaker2"):
            personality = BUILTIN_DUOS[slug][role]["personality"]
            assert personality.startswith(self._DETERMINERS), (
                f"Duo {slug!r} {role} personality {personality!r} does not open with a "
                "determiner, so the TTS preamble renders ungrammatical English "
                f"(\"{BUILTIN_DUOS[slug][role]['name']} is {personality}.\")."
            )

    @pytest.mark.parametrize("slug", sorted(BUILTIN_DUOS))
    def test_rendered_tts_sentence_is_grammatical(self, slug):
        # Render through the real preamble builder rather than restating the
        # template, so the two cannot drift apart.
        duo = BUILTIN_DUOS[slug]
        cfg = {
            "language": "French",
            "speaker1": duo["speaker1"],
            "speaker2": duo["speaker2"],
        }
        prompt = _build_tts_prompt("", cfg)
        for role in ("speaker1", "speaker2"):
            speaker = duo[role]
            sentence = f"{speaker['name']} is {speaker['personality']}."
            assert sentence in prompt
            # A semicolon inside the clause is the tell of the old job-title
            # style ("Nora is desk anchor; presses with ...").
            assert ";" not in speaker["personality"], (
                f"Duo {slug!r} {role} personality uses a semicolon list, which reads "
                "as broken English after \"is\"."
            )


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
        captured["s1_direction"] = gemini_cfg["speaker1"].get("voice_direction")
        captured["s2_direction"] = gemini_cfg["speaker2"].get("voice_direction")
        captured["tts_style"] = dict(gemini_cfg.get("tts_style") or {})
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
        assert (cap["s1_voice"], cap["s2_voice"]) == _voices_of(DEFAULT_DUO)
        assert (cap["s1_name"], cap["s2_name"]) == _names_of(DEFAULT_DUO)

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

    @pytest.mark.parametrize(
        "bad_direction",
        ["voice_direction: 42", "voice_direction:\n                - low\n                - slow"],
    )
    def test_legacy_non_string_voice_direction_fails_fast(
        self, runner_env, tmp_path, bad_direction
    ):
        # Legacy speakerN blocks bypass resolve_duo, so without an explicit
        # check a YAML scalar or list here only surfaced as an AttributeError
        # inside the TTS thread pool, after the dialogue was already billed.
        config_path = _config(
            tmp_path,
            f"""\
            speaker1:
              name: Old1
              voice: Puck
              personality: legacy one
              {bad_direction}
            speaker2:
              name: Old2
              voice: Charon
              personality: legacy two
            """,
        )
        result, _ = _run_capture_speakers(runner_env, config_path, [])
        assert result.exit_code != 0
        assert "voice_direction" in result.output
        assert "must be a string" in result.output

    def test_default_duo_from_config(self, runner_env, tmp_path):
        config_path = _config(tmp_path, "default_duo: contrast\n")
        result, cap = _run_capture_speakers(runner_env, config_path, [])
        assert result.exit_code == 0, result.output
        assert (cap["s1_voice"], cap["s2_voice"]) == _voices_of("contrast")
        assert (cap["s1_name"], cap["s2_name"]) == _names_of("contrast")

    def test_cli_duo_overrides_default_duo(self, runner_env, tmp_path):
        config_path = _config(tmp_path, "default_duo: contrast\n")
        result, cap = _run_capture_speakers(runner_env, config_path, ["--duo", "journalist"])
        assert result.exit_code == 0, result.output
        assert (cap["s1_voice"], cap["s2_voice"]) == _voices_of("journalist")

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
        assert (cap["s1_voice"], cap["s2_voice"]) == _voices_of("debate")
        assert (cap["s1_name"], cap["s2_name"]) == _names_of("debate")

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
        assert cap["s1_voice"] == _voices_of("warm")[0]
        assert cap["s1_overlay"] == "extra punchy"


class TestDuoTtsStyleDefaults:
    """
    A duo's ``scene`` / ``pace`` are *defaults*: the user's config always wins.
    """

    def test_duo_supplies_scene_and_pace_when_config_is_silent(self, runner_env, tmp_path):
        config_path = _config(tmp_path, "default_duo: journalist\n")
        result, cap = _run_capture_speakers(runner_env, config_path, [])
        assert result.exit_code == 0, result.output
        assert cap["tts_style"]["scene"] == BUILTIN_DUOS["journalist"]["scene"]
        assert cap["tts_style"]["pace"] == BUILTIN_DUOS["journalist"]["pace"]

    def test_user_tts_style_wins_over_duo_defaults(self, runner_env, tmp_path):
        config_path = _config(
            tmp_path,
            """\
            default_duo: journalist
            tts_style:
              scene: my own scene
              pace: my own pace
            """,
        )
        result, cap = _run_capture_speakers(runner_env, config_path, [])
        assert result.exit_code == 0, result.output
        assert cap["tts_style"]["scene"] == "my own scene"
        assert cap["tts_style"]["pace"] == "my own pace"

    def test_duo_fills_only_the_missing_key(self, runner_env, tmp_path):
        config_path = _config(
            tmp_path,
            """\
            default_duo: journalist
            tts_style:
              pace: my own pace
            """,
        )
        result, cap = _run_capture_speakers(runner_env, config_path, [])
        assert result.exit_code == 0, result.output
        assert cap["tts_style"]["pace"] == "my own pace"
        assert cap["tts_style"]["scene"] == BUILTIN_DUOS["journalist"]["scene"]

    def test_voice_direction_reaches_the_resolved_speakers(self, runner_env, tmp_path):
        # The CLI is the single injection point; the TTS stage reads the
        # direction back out of gemini_cfg["speakerN"].
        config_path = _config(tmp_path, "default_duo: debate\n")
        result, cap = _run_capture_speakers(runner_env, config_path, [])
        assert result.exit_code == 0, result.output
        assert cap["s1_direction"] == BUILTIN_DUOS["debate"]["speaker1"]["voice_direction"]
        assert cap["s2_direction"] == BUILTIN_DUOS["debate"]["speaker2"]["voice_direction"]

    def test_legacy_speakers_get_no_duo_tts_style(self, runner_env, tmp_path):
        # No duo resolved → nothing injected, existing episodes sound unchanged.
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
        assert cap["tts_style"] == {}
        assert cap["s1_direction"] is None
        assert cap["s2_direction"] is None


class TestDuosCommand:
    def test_lists_all_builtins_with_default_marker(self, runner_env, tmp_path, monkeypatch):
        # Point the default-config lookup at an empty dir so no user config leaks in.
        monkeypatch.setattr("tts_podcast.cli._DEFAULT_CONFIG", tmp_path / "nope.yaml")
        result = runner_env.invoke(cli, ["duos"])
        assert result.exit_code == 0, result.output
        for slug in BUILTIN_DUOS:
            assert slug in result.output
        assert "[default]" in result.output
