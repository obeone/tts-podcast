"""
Tests for the pluggable TTS backend abstraction (:mod:`tts_podcast.tts`).

Three layers:
* :class:`~tts_podcast.tts.gemini_backend.GeminiTtsBackend` — fixed audio
  format, Protocol conformance, and delegation to
  :func:`tts_podcast.tts_generator.generate_audio_chunks`.
* :func:`~tts_podcast.tts.resolve_tts_backend` — the factory that picks a
  backend from config.
* :func:`tts_podcast.audio_exporter._combine_pcm` — honouring a custom
  :class:`~tts_podcast.tts.base.AudioFormat` instead of the Gemini default.
"""

from __future__ import annotations

from unittest.mock import patch

import click
import pytest

from tts_podcast.audio_exporter import _combine_pcm
from tts_podcast.tts import resolve_tts_backend
from tts_podcast.tts.base import AudioFormat, TtsBackend
from tts_podcast.tts.gemini_backend import GeminiTtsBackend


class TestGeminiTtsBackend:
    def test_audio_format_is_fixed_gemini_shape(self):
        backend = GeminiTtsBackend({})
        assert backend.audio_format == AudioFormat(
            sample_rate=24000, channels=1, sample_width=2
        )

    def test_satisfies_tts_backend_protocol(self):
        backend = GeminiTtsBackend({})
        assert isinstance(backend, TtsBackend)

    def test_synthesize_delegates_to_generate_audio_chunks(self):
        gemini_cfg = {"api_key": "K", "tts_model": "gemini-2.5-flash-preview-tts"}
        backend = GeminiTtsBackend(gemini_cfg)
        expected = [b"pcm-bytes"]

        with patch(
            "tts_podcast.tts.gemini_backend.generate_audio_chunks",
            return_value=expected,
        ) as mock_generate:
            chunks = ["chunk1", "chunk2"]
            result = backend.synthesize(
                chunks, token_tracker="tracker", progress="progress", task_id="task"
            )

        mock_generate.assert_called_once()
        call_args = mock_generate.call_args.args
        assert call_args[0] == chunks
        assert call_args[1] is gemini_cfg
        assert result == expected


class TestResolveTtsBackend:
    def test_default_backend_returns_gemini_instance(self):
        gemini_cfg = {"api_key": "K"}
        backend = resolve_tts_backend({}, gemini_cfg)
        assert isinstance(backend, GeminiTtsBackend)
        assert backend._gemini_cfg is gemini_cfg

    def test_explicit_gemini_backend_returns_gemini_instance(self):
        gemini_cfg = {"api_key": "K"}
        cfg = {"tts": {"backend": "gemini"}}
        backend = resolve_tts_backend(cfg, gemini_cfg)
        assert isinstance(backend, GeminiTtsBackend)

    def test_moss_backend_raises_bad_parameter(self):
        cfg = {"tts": {"backend": "moss"}}
        with pytest.raises(click.BadParameter):
            resolve_tts_backend(cfg, {})

    def test_unknown_backend_raises_bad_parameter(self):
        cfg = {"tts": {"backend": "festival"}}
        with pytest.raises(click.BadParameter):
            resolve_tts_backend(cfg, {})


class TestCombinePcmAudioFormat:
    def test_custom_audio_format_is_honoured(self):
        custom_format = AudioFormat(sample_rate=48000, channels=2, sample_width=2)
        chunks = [b"\x00" * 8, b"\x00" * 8]
        combined = _combine_pcm(chunks, custom_format)
        assert combined.frame_rate == 48000
        assert combined.channels == 2
        assert combined.sample_width == 2

    def test_default_audio_format_is_gemini_shape(self):
        chunks = [b"\x00" * 4, b"\x00" * 4]
        combined = _combine_pcm(chunks)
        assert combined.frame_rate == 24000
        assert combined.channels == 1
