"""
Tests for the MOSS-TTSD HTTP backend (:mod:`tts_podcast.tts.moss_backend`).

The backend talks to a self-hosted SGLang fork's **native** API
(``POST {api_base}/generate``, not an OpenAI-compatible ``/v1/audio/speech``
route). Coverage, by layer:

* Text rewriting — ``_to_moss_text`` / ``_turn_to_moss`` (speaker-prefix ->
  ``[Sn]`` inline tags, delivery-cue stripping, continuation-line folding).
* ``_build_audio_data`` — the flat per-speaker reference-clip list.
* ``_reference_prefix`` — the ``[Sn] <transcript>`` text prefix (no
  structured reference-transcript field exists on this API).
* ``_build_body`` / ``_headers`` — the ``/generate`` request shape.
* ``_decode_wav`` — base64 + WAV-header decoding into PCM and
  :class:`~tts_podcast.tts.base.AudioFormat`.
* ``synthesize`` — the mocked end-to-end HTTP round trip, including error
  paths, plus the :func:`~tts_podcast.tts.resolve_tts_backend` factory wiring.
"""

from __future__ import annotations

import base64
import io
import wave
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from tts_podcast.llm_summarizer import DialogueChunk
from tts_podcast.tts import resolve_tts_backend
from tts_podcast.tts.base import AudioFormat
from tts_podcast.tts.moss_backend import MossTtsBackend

GEMINI_CFG = {"speaker1": {"name": "Alex"}, "speaker2": {"name": "Jordan"}}


def _make_wav_b64(
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
    frames: bytes = b"\x01\x00\x02\x00",
) -> str:
    """
    Build a base64-encoded in-memory WAV file for decode-path tests.

    Parameters
    ----------
    sample_rate : int, optional
        WAV header sample rate, by default 16000.
    channels : int, optional
        WAV header channel count, by default 1.
    sample_width : int, optional
        WAV header bytes-per-sample, by default 2.
    frames : bytes, optional
        Raw PCM frame bytes to embed, by default a 4-byte sample.

    Returns
    -------
    str
        Base64 ASCII text of the full WAV container.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(frames)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class TestMossTextRewriting:
    """`_to_moss_text` / `_turn_to_moss` — dialogue lines to `[Sn]` tags."""

    def test_basic_two_speaker_turns(self):
        backend = MossTtsBackend({"api_base": "http://x"}, GEMINI_CFG)
        result = backend._to_moss_text("Alex: Salut\nJordan: Oui")
        assert result == "[S1] Salut [S2] Oui"

    def test_delivery_cues_stripped_by_default(self):
        backend = MossTtsBackend({"api_base": "http://x"}, GEMINI_CFG)
        result = backend._to_moss_text("Alex: (curieux) Salut [enthusiasm] Jordan")
        assert result == "[S1] Salut Jordan"

    def test_delivery_cues_kept_when_disabled(self):
        moss_cfg = {"api_base": "http://x", "strip_delivery_cues": False}
        backend = MossTtsBackend(moss_cfg, GEMINI_CFG)
        result = backend._to_moss_text("Alex: (curieux) Salut [enthusiasm] Jordan")
        assert "(curieux)" in result
        assert "[enthusiasm]" in result

    def test_continuation_line_folds_into_previous_turn(self):
        backend = MossTtsBackend({"api_base": "http://x"}, GEMINI_CFG)
        result = backend._to_moss_text("Alex: Salut\nla suite du texte")
        assert result == "[S1] Salut la suite du texte"

    def test_leading_orphan_line_attributed_to_s1(self):
        backend = MossTtsBackend({"api_base": "http://x"}, GEMINI_CFG)
        result = backend._to_moss_text("orphan line first\nJordan: Oui")
        assert result == "[S1] orphan line first [S2] Oui"


class TestBuildAudioData:
    """`_build_audio_data` — flat per-speaker reference-clip list."""

    def test_two_speakers_flat_list_speaker1_first(self):
        moss_cfg = {
            "api_base": "http://x",
            "speaker1": {"ref_audio": "/tmp/alex.wav", "ref_text": "Bonjour"},
            "speaker2": {"ref_audio": "/tmp/jordan.wav", "ref_text": "Salut"},
        }
        backend = MossTtsBackend(moss_cfg, GEMINI_CFG)
        assert backend._build_audio_data() == ["/tmp/alex.wav", "/tmp/jordan.wav"]

    def test_speaker_without_ref_audio_is_skipped(self):
        moss_cfg = {
            "api_base": "http://x",
            "speaker1": {"ref_text": "Bonjour"},  # no ref_audio
            "speaker2": {"ref_audio": "/tmp/jordan.wav", "ref_text": "Salut"},
        }
        backend = MossTtsBackend(moss_cfg, GEMINI_CFG)
        assert backend._build_audio_data() == ["/tmp/jordan.wav"]

    def test_no_reference_audio_returns_empty_list(self):
        backend = MossTtsBackend({"api_base": "http://x"}, GEMINI_CFG)
        assert backend._build_audio_data() == []


class TestReferencePrefix:
    """`_reference_prefix` — the `[Sn] <transcript>` text prefix."""

    def test_tagged_mode_includes_every_speaker_with_ref_audio(self):
        moss_cfg = {
            "api_base": "http://x",
            "speaker1": {"ref_audio": "/tmp/a.wav", "ref_text": "one"},
            "speaker2": {"ref_audio": "/tmp/b.wav", "ref_text": "two"},
        }
        backend = MossTtsBackend(moss_cfg, GEMINI_CFG)
        assert backend._reference_prefix() == "[S1] one [S2] two"

    def test_tagged_mode_skips_speaker_without_ref_audio(self):
        moss_cfg = {
            "api_base": "http://x",
            "speaker1": {"ref_text": "one"},  # no ref_audio -> no [S1] prefix
            "speaker2": {"ref_audio": "/tmp/b.wav", "ref_text": "two"},
        }
        backend = MossTtsBackend(moss_cfg, GEMINI_CFG)
        assert backend._reference_prefix() == "[S2] two"

    def test_none_mode_yields_empty_string(self):
        moss_cfg = {
            "api_base": "http://x",
            "ref_prefix_mode": "none",
            "speaker1": {"ref_audio": "/tmp/a.wav", "ref_text": "one"},
        }
        backend = MossTtsBackend(moss_cfg, GEMINI_CFG)
        assert backend._reference_prefix() == ""

    def test_empty_when_no_transcripts_configured(self):
        backend = MossTtsBackend({"api_base": "http://x"}, GEMINI_CFG)
        assert backend._reference_prefix() == ""


class TestBuildBody:
    """`_build_body` — the native `/generate` JSON request body."""

    def test_contains_text_and_stream_false(self):
        backend = MossTtsBackend({"api_base": "http://x"}, GEMINI_CFG)
        body = backend._build_body("[S1] Salut", [])
        assert body["text"] == "[S1] Salut"
        assert body["stream"] is False

    def test_reference_prefix_prepended_to_text(self):
        moss_cfg = {
            "api_base": "http://x",
            "speaker1": {"ref_audio": "/tmp/a.wav", "ref_text": "Bonjour"},
        }
        backend = MossTtsBackend(moss_cfg, GEMINI_CFG)
        body = backend._build_body("[S1] Salut", ["/tmp/a.wav"])
        assert body["text"] == "[S1] Bonjour [S1] Salut"

    def test_audio_data_included_only_when_non_empty(self):
        backend = MossTtsBackend({"api_base": "http://x"}, GEMINI_CFG)
        body_without = backend._build_body("[S1] Salut", [])
        assert "audio_data" not in body_without

        body_with = backend._build_body("[S1] Salut", ["/tmp/a.wav"])
        assert body_with["audio_data"] == ["/tmp/a.wav"]

    def test_sampling_params_nested_with_only_present_keys(self):
        moss_cfg = {"api_base": "http://x", "temperature": 0.7, "top_p": 0.9}
        backend = MossTtsBackend(moss_cfg, GEMINI_CFG)
        body = backend._build_body("[S1] Salut", [])
        assert body["sampling_params"] == {"temperature": 0.7, "top_p": 0.9}

    def test_sampling_params_absent_when_no_keys_configured(self):
        backend = MossTtsBackend({"api_base": "http://x"}, GEMINI_CFG)
        body = backend._build_body("[S1] Salut", [])
        assert "sampling_params" not in body

    def test_extra_body_merged(self):
        moss_cfg = {
            "api_base": "http://x",
            "extra_body": {"custom_field": "custom_value"},
        }
        backend = MossTtsBackend(moss_cfg, GEMINI_CFG)
        body = backend._build_body("[S1] Salut", [])
        assert body["custom_field"] == "custom_value"

    def test_no_model_voice_or_response_format_keys(self):
        backend = MossTtsBackend({"api_base": "http://x"}, GEMINI_CFG)
        body = backend._build_body("[S1] Salut", [])
        assert "model" not in body
        assert "voice" not in body
        assert "response_format" not in body


class TestHeaders:
    """`_headers` — bearer auth is optional."""

    def test_no_api_key_no_authorization_header(self):
        backend = MossTtsBackend({"api_base": "http://x"}, GEMINI_CFG)
        headers = backend._headers()
        assert "Authorization" not in headers

    def test_api_key_sets_bearer_authorization(self):
        backend = MossTtsBackend({"api_base": "http://x", "api_key": "K"}, GEMINI_CFG)
        headers = backend._headers()
        assert headers["Authorization"] == "Bearer K"


class TestDecodeWav:
    """`_decode_wav` — base64 + WAV-header decoding."""

    def test_decodes_pcm_and_format_from_wav_header(self):
        frames = b"\x01\x00\x02\x00\x03\x00"
        payload = _make_wav_b64(sample_rate=16000, channels=1, sample_width=2, frames=frames)
        pcm, fmt = MossTtsBackend._decode_wav(payload, 0)
        assert pcm == frames
        assert fmt == AudioFormat(sample_rate=16000, channels=1, sample_width=2)

    def test_non_base64_payload_raises_runtime_error(self):
        with pytest.raises(RuntimeError):
            MossTtsBackend._decode_wav("not valid base64 !!!", 0)

    def test_non_wav_payload_raises_runtime_error(self):
        payload = base64.b64encode(b"definitely not a wav container").decode("ascii")
        with pytest.raises(RuntimeError):
            MossTtsBackend._decode_wav(payload, 0)


class TestSynthesize:
    """`synthesize` — mocked `httpx.Client` HTTP round trip against `/generate`."""

    def _make_client_mock(self, patcher):
        """Wire a patched ``httpx.Client`` context manager and return the mock client."""
        return patcher.return_value.__enter__.return_value

    def test_returns_decoded_pcm_in_order_and_sets_audio_format(self):
        backend = MossTtsBackend({"api_base": "http://x"}, GEMINI_CFG)
        chunks = [
            DialogueChunk(text="Alex: Hi\nJordan: Yo", index=0),
            DialogueChunk(text="Alex: Bye\nJordan: Later", index=1),
        ]
        wav_hi = _make_wav_b64(sample_rate=16000, frames=b"\x01\x02")
        wav_bye = _make_wav_b64(sample_rate=16000, frames=b"\x03\x04")

        def fake_post(url, json, headers):
            # Select the response by content rather than call order: chunks
            # run on a thread pool, so physical call order is not guaranteed
            # to match submission order.
            payload = wav_hi if "Hi" in json["text"] else wav_bye
            return SimpleNamespace(
                status_code=200,
                json=lambda: {"text": payload, "meta_info": {}},
                text="",
            )

        with patch("httpx.Client") as mock_client_cls:
            mock_client = self._make_client_mock(mock_client_cls)
            mock_client.post.side_effect = fake_post
            result = backend.synthesize(chunks)

        assert result == [b"\x01\x02", b"\x03\x04"]
        assert backend.audio_format == AudioFormat(
            sample_rate=16000, channels=1, sample_width=2
        )

    def test_posts_to_generate_endpoint_with_speaker_tags_in_body(self):
        backend = MossTtsBackend({"api_base": "http://x"}, GEMINI_CFG)
        chunks = [DialogueChunk(text="Alex: Hi\nJordan: Yo", index=0)]
        wav = _make_wav_b64(sample_rate=16000, frames=b"\x01\x02")
        with patch("httpx.Client") as mock_client_cls:
            mock_client = self._make_client_mock(mock_client_cls)
            mock_client.post.return_value = SimpleNamespace(
                status_code=200,
                json=lambda: {"text": wav, "meta_info": {}},
                text="",
            )
            backend.synthesize(chunks)

        call = mock_client.post.call_args
        assert call.args[0] == "http://x/generate"
        assert "[S1]" in call.kwargs["json"]["text"]
        assert "[S2]" in call.kwargs["json"]["text"]

    def test_http_500_raises_runtime_error(self):
        backend = MossTtsBackend({"api_base": "http://x"}, GEMINI_CFG)
        chunks = [DialogueChunk(text="Alex: Hi\nJordan: Yo", index=0)]
        with patch("httpx.Client") as mock_client_cls:
            mock_client = self._make_client_mock(mock_client_cls)
            mock_client.post.return_value = SimpleNamespace(status_code=500, text="boom")
            with pytest.raises(RuntimeError):
                backend.synthesize(chunks)

    def test_envelope_without_text_field_raises_runtime_error(self):
        backend = MossTtsBackend({"api_base": "http://x"}, GEMINI_CFG)
        chunks = [DialogueChunk(text="Alex: Hi\nJordan: Yo", index=0)]
        with patch("httpx.Client") as mock_client_cls:
            mock_client = self._make_client_mock(mock_client_cls)
            mock_client.post.return_value = SimpleNamespace(
                status_code=200,
                json=lambda: {"meta_info": {}},
                text="",
            )
            with pytest.raises(RuntimeError):
                backend.synthesize(chunks)

    def test_inconsistent_audio_formats_across_chunks_raises(self):
        """
        Two chunks whose WAVs disagree on sample rate cannot be concatenated.

        Regression guard: the error message sorts on a plain
        ``(sample_rate, channels, sample_width)`` tuple rather than on the
        frozen-but-unordered :class:`~tts_podcast.tts.base.AudioFormat`
        itself, so the intended ``RuntimeError`` surfaces instead of a
        ``TypeError`` from a failed ``sorted()``.
        """
        backend = MossTtsBackend({"api_base": "http://x"}, GEMINI_CFG)
        chunks = [
            DialogueChunk(text="Alex: Hi\nJordan: Yo", index=0),
            DialogueChunk(text="Alex: Bye\nJordan: Later", index=1),
        ]
        wav_16k = _make_wav_b64(sample_rate=16000, frames=b"\x01\x02")
        wav_24k = _make_wav_b64(sample_rate=24000, frames=b"\x03\x04")

        def fake_post(url, json, headers):
            payload = wav_16k if "Hi" in json["text"] else wav_24k
            return SimpleNamespace(
                status_code=200,
                json=lambda: {"text": payload, "meta_info": {}},
                text="",
            )

        with patch("httpx.Client") as mock_client_cls:
            mock_client = self._make_client_mock(mock_client_cls)
            mock_client.post.side_effect = fake_post
            with pytest.raises(RuntimeError, match="inconsistent audio formats"):
                backend.synthesize(chunks)

    def test_empty_chunks_returns_empty_list_without_http_call(self):
        backend = MossTtsBackend({"api_base": "http://x"}, GEMINI_CFG)
        with patch("httpx.Client") as mock_client_cls:
            result = backend.synthesize([])
        assert result == []
        mock_client_cls.assert_not_called()


class TestFactoryAndConstruction:
    """`resolve_tts_backend` factory wiring + construction invariants."""

    def test_resolve_tts_backend_returns_moss_instance(self):
        cfg = {"tts": {"backend": "moss", "moss": {"api_base": "http://x"}}}
        backend = resolve_tts_backend(cfg, GEMINI_CFG)
        assert isinstance(backend, MossTtsBackend)

    def test_missing_api_base_raises_bad_parameter(self):
        with pytest.raises(click.BadParameter):
            MossTtsBackend({}, GEMINI_CFG)

    def test_invalid_ref_prefix_mode_raises_bad_parameter(self):
        with pytest.raises(click.BadParameter):
            MossTtsBackend(
                {"api_base": "http://x", "ref_prefix_mode": "bogus"}, GEMINI_CFG
            )
