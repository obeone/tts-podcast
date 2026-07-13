"""
Tests for the MOSS-TTSD HTTP backend (:mod:`tts_podcast.tts.moss_backend`).

Four layers:
* Text rewriting — ``_to_moss_text`` / ``_turn_to_moss`` (speaker-prefix ->
  ``[Sn]`` inline tags, delivery-cue stripping, continuation-line folding).
* ``_build_references`` — per-speaker reference-audio entries.
* ``_build_body`` / ``_headers`` — the HTTP request shape.
* ``synthesize`` — the mocked end-to-end HTTP round trip, including error
  paths, plus the :func:`~tts_podcast.tts.resolve_tts_backend` factory wiring.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from tts_podcast.llm_summarizer import DialogueChunk
from tts_podcast.tts import resolve_tts_backend
from tts_podcast.tts.base import AudioFormat
from tts_podcast.tts.moss_backend import MossTtsBackend

GEMINI_CFG = {"speaker1": {"name": "Alex"}, "speaker2": {"name": "Jordan"}}


class TestMossTextRewriting:
    """`_to_moss_text` / `_turn_to_moss` — dialogue lines to `[Sn]` tags."""

    def test_basic_two_speaker_turns(self):
        backend = MossTtsBackend({"api_base": "http://x/v1"}, GEMINI_CFG)
        result = backend._to_moss_text("Alex: Salut\nJordan: Oui")
        assert result == "[S1] Salut [S2] Oui"

    def test_delivery_cues_stripped_by_default(self):
        backend = MossTtsBackend({"api_base": "http://x/v1"}, GEMINI_CFG)
        result = backend._to_moss_text("Alex: (curieux) Salut [enthusiasm] Jordan")
        assert result == "[S1] Salut Jordan"

    def test_delivery_cues_kept_when_disabled(self):
        moss_cfg = {"api_base": "http://x/v1", "strip_delivery_cues": False}
        backend = MossTtsBackend(moss_cfg, GEMINI_CFG)
        result = backend._to_moss_text("Alex: (curieux) Salut [enthusiasm] Jordan")
        assert "(curieux)" in result
        assert "[enthusiasm]" in result

    def test_continuation_line_folds_into_previous_turn(self):
        backend = MossTtsBackend({"api_base": "http://x/v1"}, GEMINI_CFG)
        result = backend._to_moss_text("Alex: Salut\nla suite du texte")
        assert result == "[S1] Salut la suite du texte"

    def test_leading_orphan_line_attributed_to_s1(self):
        backend = MossTtsBackend({"api_base": "http://x/v1"}, GEMINI_CFG)
        result = backend._to_moss_text("orphan line first\nJordan: Oui")
        assert result == "[S1] orphan line first [S2] Oui"


class TestBuildReferences:
    """`_build_references` — per-speaker reference-audio entries."""

    def test_two_speakers_with_references(self):
        moss_cfg = {
            "api_base": "http://x/v1",
            "speaker1": {"ref_audio": "/tmp/alex.wav", "ref_text": "Bonjour"},
            "speaker2": {"ref_audio": "/tmp/jordan.wav", "ref_text": "Salut"},
        }
        backend = MossTtsBackend(moss_cfg, GEMINI_CFG)
        references = backend._build_references()
        assert references == [
            {"audio_path": "/tmp/alex.wav", "text": "[S1] Bonjour"},
            {"audio_path": "/tmp/jordan.wav", "text": "[S2] Salut"},
        ]

    def test_custom_reference_audio_field(self):
        moss_cfg = {
            "api_base": "http://x/v1",
            "reference_audio_field": "audio",
            "speaker1": {"ref_audio": "/tmp/alex.wav", "ref_text": "Bonjour"},
        }
        backend = MossTtsBackend(moss_cfg, GEMINI_CFG)
        references = backend._build_references()
        assert references == [{"audio": "/tmp/alex.wav", "text": "[S1] Bonjour"}]

    def test_speaker_without_ref_audio_is_skipped(self):
        moss_cfg = {
            "api_base": "http://x/v1",
            "speaker1": {"ref_text": "Bonjour"},
            "speaker2": {"ref_audio": "/tmp/jordan.wav", "ref_text": "Salut"},
        }
        backend = MossTtsBackend(moss_cfg, GEMINI_CFG)
        references = backend._build_references()
        assert references == [{"audio_path": "/tmp/jordan.wav", "text": "[S2] Salut"}]

    def test_no_reference_audio_returns_empty_list(self):
        backend = MossTtsBackend({"api_base": "http://x/v1"}, GEMINI_CFG)
        assert backend._build_references() == []


class TestBuildBody:
    """`_build_body` — the `/audio/speech` JSON request body."""

    def test_contains_required_keys(self):
        backend = MossTtsBackend({"api_base": "http://x/v1"}, GEMINI_CFG)
        body = backend._build_body("[S1] Salut", [])
        assert body["model"]
        assert body["input"] == "[S1] Salut"
        assert body["response_format"] == "pcm"
        assert "voice" in body

    def test_references_included_only_when_non_empty(self):
        backend = MossTtsBackend({"api_base": "http://x/v1"}, GEMINI_CFG)
        body_without = backend._build_body("[S1] Salut", [])
        assert "references" not in body_without

        refs = [{"audio_path": "/tmp/a.wav", "text": "[S1] hi"}]
        body_with = backend._build_body("[S1] Salut", refs)
        assert body_with["references"] == refs

    def test_sampling_keys_forwarded_only_when_present(self):
        moss_cfg = {
            "api_base": "http://x/v1",
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 42,
        }
        backend = MossTtsBackend(moss_cfg, GEMINI_CFG)
        body = backend._build_body("[S1] Salut", [])
        assert body["temperature"] == 0.7
        assert body["top_p"] == 0.9
        assert body["seed"] == 42
        assert "top_k" not in body
        assert "repetition_penalty" not in body
        assert "max_new_tokens" not in body

    def test_extra_body_merged(self):
        moss_cfg = {
            "api_base": "http://x/v1",
            "extra_body": {"custom_field": "custom_value"},
        }
        backend = MossTtsBackend(moss_cfg, GEMINI_CFG)
        body = backend._build_body("[S1] Salut", [])
        assert body["custom_field"] == "custom_value"


class TestHeaders:
    """`_headers` — bearer auth is optional."""

    def test_no_api_key_no_authorization_header(self):
        backend = MossTtsBackend({"api_base": "http://x/v1"}, GEMINI_CFG)
        headers = backend._headers()
        assert "Authorization" not in headers

    def test_api_key_sets_bearer_authorization(self):
        backend = MossTtsBackend({"api_base": "http://x/v1", "api_key": "K"}, GEMINI_CFG)
        headers = backend._headers()
        assert headers["Authorization"] == "Bearer K"


class TestSynthesize:
    """`synthesize` — mocked `httpx.Client` HTTP round trip."""

    def _make_client_mock(self, patcher):
        """Wire a patched ``httpx.Client`` context manager and return the mock client."""
        return patcher.return_value.__enter__.return_value

    def test_returns_pcm_bytes_in_order(self):
        backend = MossTtsBackend({"api_base": "http://x/v1"}, GEMINI_CFG)
        chunks = [
            DialogueChunk(text="Alex: Hi\nJordan: Yo", index=0),
            DialogueChunk(text="Alex: Bye\nJordan: Later", index=1),
        ]
        with patch("httpx.Client") as mock_client_cls:
            mock_client = self._make_client_mock(mock_client_cls)
            mock_client.post.side_effect = [
                SimpleNamespace(status_code=200, content=b"\x01\x02", text=""),
                SimpleNamespace(status_code=200, content=b"\x03\x04", text=""),
            ]
            result = backend.synthesize(chunks)

        assert result == [b"\x01\x02", b"\x03\x04"]

    def test_posts_to_expected_url_with_speaker_tags_in_body(self):
        backend = MossTtsBackend({"api_base": "http://x/v1"}, GEMINI_CFG)
        chunks = [DialogueChunk(text="Alex: Hi\nJordan: Yo", index=0)]
        with patch("httpx.Client") as mock_client_cls:
            mock_client = self._make_client_mock(mock_client_cls)
            mock_client.post.return_value = SimpleNamespace(
                status_code=200, content=b"\x01\x02", text=""
            )
            backend.synthesize(chunks)

        call = mock_client.post.call_args
        assert call.args[0] == "http://x/v1/audio/speech"
        assert "[S1]" in call.kwargs["json"]["input"]
        assert "[S2]" in call.kwargs["json"]["input"]

    def test_server_error_status_raises_runtime_error(self):
        backend = MossTtsBackend({"api_base": "http://x/v1"}, GEMINI_CFG)
        chunks = [DialogueChunk(text="Alex: Hi\nJordan: Yo", index=0)]
        with patch("httpx.Client") as mock_client_cls:
            mock_client = self._make_client_mock(mock_client_cls)
            mock_client.post.return_value = SimpleNamespace(
                status_code=500, content=b"", text="boom"
            )
            with pytest.raises(RuntimeError):
                backend.synthesize(chunks)

    def test_empty_body_raises_runtime_error(self):
        backend = MossTtsBackend({"api_base": "http://x/v1"}, GEMINI_CFG)
        chunks = [DialogueChunk(text="Alex: Hi\nJordan: Yo", index=0)]
        with patch("httpx.Client") as mock_client_cls:
            mock_client = self._make_client_mock(mock_client_cls)
            mock_client.post.return_value = SimpleNamespace(
                status_code=200, content=b"", text=""
            )
            with pytest.raises(RuntimeError):
                backend.synthesize(chunks)

    def test_empty_chunks_returns_empty_list_without_http_call(self):
        backend = MossTtsBackend({"api_base": "http://x/v1"}, GEMINI_CFG)
        with patch("httpx.Client") as mock_client_cls:
            result = backend.synthesize([])
        assert result == []
        mock_client_cls.assert_not_called()


class TestFactoryAndConstruction:
    """`resolve_tts_backend` factory wiring + construction invariants."""

    def test_resolve_tts_backend_returns_moss_instance(self):
        cfg = {"tts": {"backend": "moss", "moss": {"api_base": "http://x/v1"}}}
        backend = resolve_tts_backend(cfg, GEMINI_CFG)
        assert isinstance(backend, MossTtsBackend)

    def test_missing_api_base_raises_bad_parameter(self):
        with pytest.raises(click.BadParameter):
            MossTtsBackend({}, GEMINI_CFG)

    def test_default_audio_format(self):
        backend = MossTtsBackend({"api_base": "http://x/v1"}, GEMINI_CFG)
        assert backend.audio_format == AudioFormat(
            sample_rate=24000, channels=1, sample_width=2
        )

    def test_custom_sample_rate(self):
        moss_cfg = {"api_base": "http://x/v1", "sample_rate": 48000}
        backend = MossTtsBackend(moss_cfg, GEMINI_CFG)
        assert backend.audio_format == AudioFormat(
            sample_rate=48000, channels=1, sample_width=2
        )
