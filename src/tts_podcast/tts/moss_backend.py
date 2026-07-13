"""
MOSS-TTSD text-to-speech backend (self-hosted, HTTP).

Renders the podcast dialogue through a self-hosted `MOSS-TTSD
<https://github.com/OpenMOSS/MOSS-TTSD>`_ server (OpenMOSS' spoken-dialogue
model) exposed over an OpenAI-compatible ``POST /v1/audio/speech`` endpoint —
as served by vLLM-Omni or SGLang-Omni.  The heavy 8B model runs on the
operator's GPU box; this backend is only a thin HTTP client, so the project
pulls no torch/transformers weight.

Mapping from the pipeline to MOSS-TTSD
--------------------------------------
- Each :class:`~tts_podcast.llm_summarizer.DialogueChunk` (lines like
  ``"Alex: …"`` / ``"Jordan: …"``) is rewritten into MOSS' native inline
  speaker-tag form ``"[S1] … [S2] …"``: speaker 1 → ``[S1]``, speaker 2 →
  ``[S2]``.
- Voices are set by zero-shot cloning: an optional short reference clip plus
  its transcript per speaker, passed in a ``references`` array (one entry per
  speaker, each transcript prefixed with its ``[Sn]`` tag).
- ``response_format: "pcm"`` returns raw 24 kHz / mono / 16-bit LE PCM, which
  is exactly what :mod:`tts_podcast.audio_exporter` consumes — no conversion.

Unverified server contract
---------------------------
The *multi-speaker* reference-audio request shape for MOSS-TTSD v1.0 over HTTP
is **not** officially documented (only single-speaker examples are published).
The default here follows the SGLang-Omni cookbook (``references: [{audio_path,
text}]``); OpenMOSS' own legacy client instead used a base64 ``audio`` key.
Because the exact field name is server-dependent, it is configurable
(``reference_audio_field``), and an arbitrary ``extra_body`` mapping is merged
into every request so operators can match their actual server without a code
change.  Verify voice cloning against your running server before relying on it.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from tts_podcast.tts.base import AudioFormat

if TYPE_CHECKING:
    from rich.progress import Progress

    from tts_podcast.llm_summarizer import DialogueChunk
    from tts_podcast.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

# Default number of concurrent requests to the MOSS server.  Kept low: an 8B
# model on a single GPU serialises work anyway, and over-parallelising risks
# OOM on the server.  Override via `tts.moss.max_workers`.
_DEFAULT_MAX_WORKERS = 2

# Default per-request HTTP timeout (seconds).  MOSS inference is slow; a full
# chunk can take tens of seconds.  Override via `tts.moss.timeout_seconds`.
_DEFAULT_TIMEOUT = 300.0

# MOSS-TTSD "pcm" output is 24 kHz / mono / 16-bit LE — matches the exporter.
_DEFAULT_SAMPLE_RATE = 24_000

# Sampling parameters forwarded to the server when present in the moss config.
_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens", "seed")

# Matches a leading delivery cue like "(avec enthousiasme) " at the start of a turn.
_LEADING_PAREN_CUE = re.compile(r"^\s*\([^)]*\)\s*")
# Matches inline bracketed audio-tag tokens like "[enthusiasm]" / "[short pause]".
# These collide with MOSS' own [S1]/[S2] tags and must be stripped before the
# speaker prefixes are rewritten (order matters — see _to_moss_text).
_BRACKET_TAG = re.compile(r"\[[^\]]*\]")


class MossTtsBackend:
    """
    :class:`~tts_podcast.tts.base.TtsBackend` backed by a self-hosted MOSS-TTSD server.

    Parameters
    ----------
    moss_cfg : dict
        The ``tts.moss`` config sub-section.  Recognised keys:

        - ``api_base`` (str, required): OpenAI-compatible base URL, e.g.
          ``"http://gpu-box:8091/v1"``.
        - ``api_key`` (str, optional): bearer token; omit for an open local
          server.
        - ``model`` (str): model name the server expects, e.g.
          ``"OpenMOSS-Team/MOSS-TTSD-v1.0"``.
        - ``sample_rate`` (int): PCM sample rate for the exporter (default
          24000).
        - ``max_workers`` (int): concurrent requests (default 2).
        - ``timeout_seconds`` (float): per-request timeout (default 300).
        - ``reference_audio_field`` (str): the key holding the clip inside each
          ``references`` entry (default ``"audio_path"``; some servers use
          ``"audio"`` with a base64 data URI).
        - ``speaker1`` / ``speaker2`` ({``ref_audio``, ``ref_text``}): optional
          per-speaker reference clip (path / URL / data URI) and its transcript.
        - ``strip_delivery_cues`` (bool): strip Gemini-style ``(cue)`` and
          ``[tag]`` delivery markers from the text (default ``True`` — MOSS
          does not use them and ``[tag]`` collides with ``[Sn]``).
        - ``extra_body`` (dict): arbitrary extra fields merged into every
          request body (server-specific escape hatch).
    gemini_cfg : dict
        The resolved speaker configuration (duo-injected).  Only the speaker
        *names* are read, to map ``"<name>:"`` turn prefixes onto ``[S1]`` /
        ``[S2]``.

    Raises
    ------
    click.BadParameter
        When ``api_base`` is missing.
    """

    def __init__(self, moss_cfg: dict, gemini_cfg: dict) -> None:
        import click

        self._cfg = moss_cfg
        api_base = str(moss_cfg.get("api_base") or "").rstrip("/")
        if not api_base:
            raise click.BadParameter(
                "tts.moss.api_base is required for the moss TTS backend "
                "(e.g. http://localhost:8091/v1)."
            )
        self._api_base = api_base
        self._speaker1_name = gemini_cfg.get("speaker1", {}).get("name", "Alex")
        self._speaker2_name = gemini_cfg.get("speaker2", {}).get("name", "Jordan")
        self.audio_format = AudioFormat(
            sample_rate=int(moss_cfg.get("sample_rate", _DEFAULT_SAMPLE_RATE)),
            channels=1,
            sample_width=2,
        )

    # ------------------------------------------------------------------
    # Text rewriting
    # ------------------------------------------------------------------

    def _turn_to_moss(self, line: str) -> str | None:
        """
        Rewrite one dialogue line ``"<name>: text"`` into ``"[Sn] text"``.

        Returns ``None`` for a line that is not a recognised speaker turn (it is
        appended to the previous turn's text by the caller).

        Parameters
        ----------
        line : str
            A single stripped dialogue line.

        Returns
        -------
        str or None
            The ``"[Sn] …"`` rendering, or ``None`` when *line* has no known
            speaker prefix.
        """
        strip_cues = bool(self._cfg.get("strip_delivery_cues", True))
        for name, tag in (
            (self._speaker1_name, "[S1]"),
            (self._speaker2_name, "[S2]"),
        ):
            prefix = f"{name}:"
            if line.startswith(prefix):
                text = line[len(prefix):].strip()
                if strip_cues:
                    # Order: drop the leading (cue), then any inline [tag] so a
                    # stray tag can never be mistaken for a speaker tag below.
                    text = _LEADING_PAREN_CUE.sub("", text)
                    text = _BRACKET_TAG.sub("", text)
                    text = re.sub(r"\s{2,}", " ", text).strip()
                return f"{tag} {text}".strip()
        return None

    def _to_moss_text(self, chunk_text: str) -> str:
        """
        Convert a full dialogue chunk into MOSS ``[S1]/[S2]`` inline-tag text.

        Non-turn continuation lines are folded into the preceding turn so no
        text is dropped.

        Parameters
        ----------
        chunk_text : str
            Raw multi-line dialogue chunk (``"<name>: …"`` per line).

        Returns
        -------
        str
            Single-line MOSS input, e.g. ``"[S1] Salut… [S2] Oui…"``.
        """
        parts: list[str] = []
        for raw in chunk_text.splitlines():
            line = raw.strip()
            if not line:
                continue
            rendered = self._turn_to_moss(line)
            if rendered is not None:
                parts.append(rendered)
            elif parts:
                # Continuation of the current turn — append its text.
                parts[-1] = f"{parts[-1]} {line}".strip()
            else:
                # Leading orphan line with no speaker: attribute it to [S1].
                parts.append(f"[S1] {line}")
        return " ".join(parts)

    def _build_references(self) -> list[dict[str, Any]]:
        """
        Build the ``references`` array from the per-speaker reference config.

        One entry per speaker that has a ``ref_audio`` set; the transcript is
        prefixed with the matching ``[Sn]`` tag.  Returns an empty list when no
        reference audio is configured (MOSS then generates voices without
        cloning — usable but not repeatable across runs).

        Returns
        -------
        list[dict[str, Any]]
            Reference entries shaped ``{<reference_audio_field>: clip, "text":
            "[Sn] transcript"}``.
        """
        field = str(self._cfg.get("reference_audio_field", "audio_path"))
        references: list[dict[str, Any]] = []
        for spk_key, tag in (("speaker1", "[S1]"), ("speaker2", "[S2]")):
            spk = self._cfg.get(spk_key, {}) or {}
            audio = spk.get("ref_audio")
            if not audio:
                continue
            text = str(spk.get("ref_text", "")).strip()
            references.append({field: audio, "text": f"{tag} {text}".strip()})
        return references

    def _build_body(self, moss_text: str, references: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Assemble the ``/v1/audio/speech`` request body for one chunk.

        Parameters
        ----------
        moss_text : str
            The chunk rendered as MOSS ``[S1]/[S2]`` input.
        references : list[dict[str, Any]]
            Pre-built per-speaker reference entries (may be empty).

        Returns
        -------
        dict[str, Any]
            The JSON request body.
        """
        body: dict[str, Any] = {
            "model": self._cfg.get("model", "OpenMOSS-Team/MOSS-TTSD-v1.0"),
            "input": moss_text,
            # Raw PCM straight into the exporter (24 kHz / mono / 16-bit LE).
            "response_format": "pcm",
            # OpenAI schema requires a `voice`; MOSS ignores it when cloning.
            "voice": self._cfg.get("voice", "default"),
        }
        if references:
            body["references"] = references
        for key in _SAMPLING_KEYS:
            if key in self._cfg:
                body[key] = self._cfg[key]
        extra = self._cfg.get("extra_body")
        if isinstance(extra, dict):
            body.update(extra)
        return body

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Return the request headers, adding bearer auth when an api_key is set."""
        headers = {"Content-Type": "application/json"}
        api_key = self._cfg.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _synthesize_one(self, client: Any, chunk: DialogueChunk, references: list[dict]) -> bytes:
        """
        Send one chunk to the MOSS server and return its raw PCM bytes.

        Parameters
        ----------
        client : httpx.Client
            An open HTTP client (shared across chunks).
        chunk : DialogueChunk
            The dialogue chunk to synthesise.
        references : list[dict]
            Pre-built reference entries (identical for every chunk).

        Returns
        -------
        bytes
            Raw PCM audio for this chunk.

        Raises
        ------
        RuntimeError
            When the server returns a non-2xx status or an empty body.
        """
        moss_text = self._to_moss_text(chunk.text)
        body = self._build_body(moss_text, references)
        logger.debug(
            "MOSS chunk %d: %d chars of [Sn] text -> POST %s/audio/speech",
            chunk.index,
            len(moss_text),
            self._api_base,
        )
        response = client.post(
            f"{self._api_base}/audio/speech",
            json=body,
            headers=self._headers(),
        )
        if response.status_code >= 400:
            snippet = response.text[:300]
            raise RuntimeError(
                f"MOSS server returned HTTP {response.status_code} for chunk "
                f"{chunk.index}: {snippet!r}"
            )
        pcm = response.content
        if not pcm:
            raise RuntimeError(
                f"MOSS server returned an empty body for chunk {chunk.index}."
            )
        logger.info("MOSS chunk %d: received %d bytes of PCM audio", chunk.index, len(pcm))
        return pcm

    def synthesize(
        self,
        chunks: list[DialogueChunk],
        token_tracker: TokenTracker | None = None,
        progress: Progress | None = None,
        task_id: Any = None,
    ) -> list[bytes]:
        """
        Render *chunks* via the MOSS-TTSD HTTP server.

        Chunks are sent as independent requests (up to ``max_workers`` in
        parallel) and reassembled in input order.  ``token_tracker`` is accepted
        for protocol symmetry but unused — a self-hosted MOSS server is not
        token-billed.

        Parameters
        ----------
        chunks : list[DialogueChunk]
            Ordered dialogue chunks to synthesise.
        token_tracker : TokenTracker or None, optional
            Ignored (no per-token billing for a self-hosted server).
        progress : rich.progress.Progress or None, optional
            Progress instance advanced once per completed chunk.
        task_id : Any, optional
            Task identifier returned by ``progress.add_task()``.

        Returns
        -------
        list[bytes]
            Raw PCM blobs, one per input chunk, in input order, matching
            :attr:`audio_format`.
        """
        import httpx

        if not chunks:
            return []

        references = self._build_references()
        if not references:
            logger.warning(
                "No MOSS reference audio configured (tts.moss.speakerN.ref_audio) — "
                "voices will not be cloned or stable across runs."
            )

        max_workers = min(int(self._cfg.get("max_workers", _DEFAULT_MAX_WORKERS)), len(chunks))
        timeout = float(self._cfg.get("timeout_seconds", _DEFAULT_TIMEOUT))
        logger.info(
            "Generating MOSS TTS for %d chunk(s) with up to %d worker(s)…",
            len(chunks),
            max_workers,
        )

        results: dict[int, bytes] = {}
        with httpx.Client(timeout=timeout) as client:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(self._synthesize_one, client, chunk, references): i
                    for i, chunk in enumerate(chunks)
                }
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    results[idx] = future.result()
                    if progress is not None and task_id is not None:
                        progress.advance(task_id)

        return [results[i] for i in range(len(chunks))]
