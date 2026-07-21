"""
MOSS-TTSD text-to-speech backend (self-hosted SGLang server, HTTP).

Renders the podcast dialogue through a self-hosted `MOSS-TTSD
<https://github.com/OpenMOSS/MOSS-TTSD>`_ server (OpenMOSS' 8B spoken-dialogue
model) served by the project's own SGLang fork (`OpenMOSS/sglang`, branch
``moss-ttsd-v1.0-with-cat``).  The heavy model runs on the operator's GPU box;
this backend is only a thin HTTP client, so the project pulls no torch weight.

The wire protocol (verified against the fork's source)
------------------------------------------------------
That server exposes SGLang's **native** API, not an OpenAI-compatible one:
``/v1/audio/speech`` does **not** exist (the only OpenAI audio route is
``/v1/audio/transcriptions``, which is ASR).  TTS goes through:

``POST {api_base}/generate``
    Body: ``{"text": ..., "audio_data": [...], "sampling_params": {...}}``.
    There is no ``model`` field (the server is single-model, fixed by
    ``--model-path``) and no ``response_format``.
    Response: ``{"text": "<base64-encoded WAV file>", "meta_info": {...}}`` —
    a full WAV container, **not** raw headerless PCM.

Mapping from the pipeline to MOSS-TTSD
--------------------------------------
- Each :class:`~tts_podcast.llm_summarizer.DialogueChunk` (lines like
  ``"Alex: …"`` / ``"Jordan: …"``) is rewritten into MOSS' native inline
  speaker-tag form ``"[S1] … [S2] …"``: speaker 1 → ``[S1]``, speaker 2 →
  ``[S2]``.  These tags are a convention of the model's text template; the
  server does not parse them structurally.
- Voices come from zero-shot cloning: the per-speaker reference clips go in
  ``audio_data`` (a list, one entry per speaker).  The fork exposes **no**
  structured field for a reference transcript, so the transcripts are prepended
  to ``text`` as a prefix, per the MOSS-TTSD model card.
- The returned WAV is decoded here into raw PCM frames, and
  :attr:`audio_format` is set from the WAV header itself, so the exporter never
  has to assume a sample rate.

Unverified at build time
------------------------
The fork's multimodal processor appeared, on a static read, to replace the
prompt with audio placeholder tokens when ``audio_data`` is attached, and it was
not possible to confirm from source alone that the ``text`` (with its ``[Sn]``
tags and transcript prefix) survives into the token stream.  That is why the
text-assembly knobs below (:data:`_REF_PREFIX_MODES`, ``extra_body``) are
configurable: smoke-test against your running server and adjust.
"""

from __future__ import annotations

import base64
import io
import logging
import re
import wave
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

# Fallback format, only used if a response somehow carries no parseable WAV
# header.  The real format is read off each returned WAV.
_FALLBACK_AUDIO_FORMAT = AudioFormat(sample_rate=24_000, channels=1, sample_width=2)

# Sampling knobs forwarded inside the nested `sampling_params` object.
_SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "max_new_tokens",
)

# Matches a leading delivery cue like "(avec enthousiasme) " at the start of a turn.
_LEADING_PAREN_CUE = re.compile(r"^\s*\([^)]*\)\s*")
# Matches inline bracketed audio-tag tokens like "[enthusiasm]" / "[short pause]".
# These collide with MOSS' own [S1]/[S2] tags and must be stripped before the
# speaker prefixes are rewritten (order matters — see _turn_to_moss).
_BRACKET_TAG = re.compile(r"\[[^\]]*\]")

#: How the per-speaker reference transcripts are prepended to `text`.  The fork
#: has no structured `ref_text` field, so the transcript must ride along inside
#: the text.  "tagged" follows the MOSS-TTSD model card (each transcript behind
#: its own [Sn] tag); "none" sends no prefix at all.  Configurable because the
#: exact expected shape could not be confirmed from source.
_REF_PREFIX_MODES = ("tagged", "none")


class MossTtsBackend:
    """
    :class:`~tts_podcast.tts.base.TtsBackend` backed by a self-hosted MOSS-TTSD server.

    Parameters
    ----------
    moss_cfg : dict
        The ``tts.moss`` config sub-section.  Recognised keys:

        - ``api_base`` (str, required): server root URL, e.g.
          ``"https://moss-ttsd.example.org"``.  No ``/v1`` suffix: the native
          SGLang API is used, and the client POSTs to ``{api_base}/generate``.
        - ``api_key`` (str, optional): bearer token.  SGLang only enforces one
          when launched with ``--api-key``; omit for an open server.
        - ``max_workers`` (int): concurrent requests (default 2).
        - ``timeout_seconds`` (float): per-request timeout (default 300).
        - ``speaker1`` / ``speaker2`` ({``ref_audio``, ``ref_text``}): optional
          per-speaker reference clip (path / URL / base64 data URI) and its
          transcript, for zero-shot voice cloning.
        - ``ref_prefix_mode`` (str): ``"tagged"`` (default) prepends each
          reference transcript behind its ``[Sn]`` tag; ``"none"`` sends no
          transcript prefix.
        - ``strip_delivery_cues`` (bool): strip Gemini-style ``(cue)`` and
          ``[tag]`` delivery markers from the text (default ``True`` — MOSS does
          not use them and ``[tag]`` collides with ``[Sn]``).
        - ``temperature`` / ``top_p`` / ``top_k`` / ``repetition_penalty`` /
          ``max_new_tokens``: forwarded inside ``sampling_params``.
        - ``extra_body`` (dict): arbitrary extra top-level fields merged into
          every request (server-specific escape hatch).
    gemini_cfg : dict
        The resolved speaker configuration (duo-injected).  Only the speaker
        *names* are read, to map ``"<name>:"`` turn prefixes onto ``[S1]`` /
        ``[S2]``.

    Raises
    ------
    click.BadParameter
        When ``api_base`` is missing or ``ref_prefix_mode`` is invalid.
    """

    def __init__(self, moss_cfg: dict, gemini_cfg: dict) -> None:
        import click

        self._cfg = moss_cfg
        api_base = str(moss_cfg.get("api_base") or "").rstrip("/")
        if not api_base:
            raise click.BadParameter(
                "tts.moss.api_base is required for the moss TTS backend "
                "(e.g. https://moss-ttsd.example.org)."
            )
        self._api_base = api_base

        mode = str(moss_cfg.get("ref_prefix_mode", "tagged")).strip().lower()
        if mode not in _REF_PREFIX_MODES:
            raise click.BadParameter(
                f"tts.moss.ref_prefix_mode must be one of {list(_REF_PREFIX_MODES)}, "
                f"got {mode!r}."
            )
        self._ref_prefix_mode = mode

        self._speaker1_name = gemini_cfg.get("speaker1", {}).get("name", "Alex")
        self._speaker2_name = gemini_cfg.get("speaker2", {}).get("name", "Jordan")

        # Provisional: overwritten in `synthesize` with the format read off the
        # WAV the server actually returns.  The CLI reads `audio_format` only
        # after `synthesize`, so the real value is always the one used.
        self.audio_format: AudioFormat = _FALLBACK_AUDIO_FORMAT

    # ------------------------------------------------------------------
    # Text assembly
    # ------------------------------------------------------------------

    def _turn_to_moss(self, line: str) -> str | None:
        """
        Rewrite one dialogue line ``"<name>: text"`` into ``"[Sn] text"``.

        Returns ``None`` for a line that is not a recognised speaker turn (the
        caller folds it into the previous turn).

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
                    # stray tag can never be mistaken for a speaker tag.
                    text = _LEADING_PAREN_CUE.sub("", text)
                    text = _BRACKET_TAG.sub("", text)
                    text = re.sub(r"\s{2,}", " ", text).strip()
                return f"{tag} {text}".strip()
        return None

    def _to_moss_text(self, chunk_text: str) -> str:
        """
        Convert a dialogue chunk into MOSS ``[S1]/[S2]`` inline-tag text.

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
                parts[-1] = f"{parts[-1]} {line}".strip()
            else:
                # Leading orphan line with no speaker: attribute it to [S1].
                parts.append(f"[S1] {line}")
        return " ".join(parts)

    def _reference_prefix(self) -> str:
        """
        Build the reference-transcript prefix prepended to ``text``.

        The SGLang fork exposes no structured per-reference transcript field, so
        the MOSS-TTSD convention is to carry each speaker's reference transcript
        in the text itself, behind that speaker's ``[Sn]`` tag, ahead of the
        dialogue to synthesise.  Returns ``""`` when disabled or when no
        transcripts are configured.

        Returns
        -------
        str
            E.g. ``"[S1] transcript one [S2] transcript two"``, or ``""``.
        """
        if self._ref_prefix_mode == "none":
            return ""
        parts: list[str] = []
        for spk_key, tag in (("speaker1", "[S1]"), ("speaker2", "[S2]")):
            spk = self._cfg.get(spk_key, {}) or {}
            # Only speakers that actually contribute a clip may prefix a
            # transcript: the transcript describes the clip.
            if not spk.get("ref_audio"):
                continue
            text = str(spk.get("ref_text", "")).strip()
            if text:
                parts.append(f"{tag} {text}")
        return " ".join(parts)

    def _build_audio_data(self) -> list[str]:
        """
        Collect the per-speaker reference clips for ``audio_data``.

        The native ``/generate`` API takes a flat list of reference clips (one
        entry per speaker, in speaker order); there is no ``{audio, text}``
        object form on this fork.

        Returns
        -------
        list[str]
            Reference clips (path / URL / data URI), speaker1 first.  Empty when
            no reference audio is configured.
        """
        clips: list[str] = []
        for spk_key in ("speaker1", "speaker2"):
            spk = self._cfg.get(spk_key, {}) or {}
            audio = spk.get("ref_audio")
            if audio:
                clips.append(str(audio))
        return clips

    def _sampling_params(self) -> dict[str, Any]:
        """
        Build the nested ``sampling_params`` object from the moss config.

        Returns
        -------
        dict[str, Any]
            Only the sampling keys actually present in the config.
        """
        params: dict[str, Any] = {}
        for key in _SAMPLING_KEYS:
            if key in self._cfg:
                params[key] = self._cfg[key]
        return params

    def _build_body(self, moss_text: str, audio_data: list[str]) -> dict[str, Any]:
        """
        Assemble the ``/generate`` request body for one chunk.

        Parameters
        ----------
        moss_text : str
            The chunk rendered as MOSS ``[S1]/[S2]`` input.
        audio_data : list[str]
            Pre-built reference clips (may be empty).

        Returns
        -------
        dict[str, Any]
            The JSON request body.
        """
        prefix = self._reference_prefix()
        text = f"{prefix} {moss_text}".strip() if prefix else moss_text

        body: dict[str, Any] = {"text": text, "stream": False}
        if audio_data:
            body["audio_data"] = audio_data
        sampling = self._sampling_params()
        if sampling:
            body["sampling_params"] = sampling
        extra = self._cfg.get("extra_body")
        if isinstance(extra, dict):
            body.update(extra)
        return body

    # ------------------------------------------------------------------
    # HTTP + audio decoding
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Return the request headers, adding bearer auth when an api_key is set."""
        headers = {"Content-Type": "application/json"}
        api_key = self._cfg.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @staticmethod
    def _decode_wav(payload: str, chunk_index: int) -> tuple[bytes, AudioFormat]:
        """
        Decode the server's base64 WAV payload into raw PCM frames plus its format.

        The native SGLang response carries a complete WAV container (RIFF
        header) base64-encoded in the ``text`` field, so the sample rate,
        channel count, and sample width are read from the header rather than
        assumed.

        Parameters
        ----------
        payload : str
            Base64-encoded WAV bytes.
        chunk_index : int
            Chunk index, for error messages.

        Returns
        -------
        tuple[bytes, AudioFormat]
            ``(pcm_frames, audio_format)``.

        Raises
        ------
        RuntimeError
            When the payload is not valid base64 or not a parseable WAV.
        """
        try:
            wav_bytes = base64.b64decode(payload, validate=True)
        except (ValueError, TypeError) as exc:
            raise RuntimeError(
                f"MOSS server returned a non-base64 payload for chunk {chunk_index}: {exc}"
            ) from exc

        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
                fmt = AudioFormat(
                    sample_rate=wav.getframerate(),
                    channels=wav.getnchannels(),
                    sample_width=wav.getsampwidth(),
                )
                pcm = wav.readframes(wav.getnframes())
        except wave.Error as exc:
            raise RuntimeError(
                f"MOSS server returned a payload that is not a valid WAV for "
                f"chunk {chunk_index}: {exc}"
            ) from exc

        return pcm, fmt

    def _synthesize_one(
        self, client: Any, chunk: DialogueChunk, audio_data: list[str]
    ) -> tuple[bytes, AudioFormat]:
        """
        Send one chunk to the MOSS server and return its PCM frames and format.

        Parameters
        ----------
        client : httpx.Client
            An open HTTP client (shared across chunks).
        chunk : DialogueChunk
            The dialogue chunk to synthesise.
        audio_data : list[str]
            Pre-built reference clips (identical for every chunk).

        Returns
        -------
        tuple[bytes, AudioFormat]
            Raw PCM frames for this chunk, and the format read off its WAV.

        Raises
        ------
        RuntimeError
            On a non-2xx status, a malformed JSON envelope, or an undecodable
            audio payload.
        """
        moss_text = self._to_moss_text(chunk.text)
        body = self._build_body(moss_text, audio_data)
        logger.debug(
            "MOSS chunk %d: %d chars of [Sn] text -> POST %s/generate",
            chunk.index,
            len(body["text"]),
            self._api_base,
        )
        response = client.post(
            f"{self._api_base}/generate",
            json=body,
            headers=self._headers(),
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"MOSS server returned HTTP {response.status_code} for chunk "
                f"{chunk.index}: {response.text[:300]!r}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"MOSS server returned non-JSON for chunk {chunk.index}: "
                f"{response.text[:200]!r}"
            ) from exc

        # The native envelope is {"text": "<base64 wav>", "meta_info": {...}}.
        audio_b64 = payload.get("text") if isinstance(payload, dict) else None
        if not audio_b64:
            raise RuntimeError(
                f"MOSS server response for chunk {chunk.index} carried no audio "
                f"in its 'text' field: {str(payload)[:200]!r}"
            )

        pcm, fmt = self._decode_wav(audio_b64, chunk.index)
        logger.info(
            "MOSS chunk %d: %d bytes of PCM (%d Hz, %d ch, %d-bit)",
            chunk.index,
            len(pcm),
            fmt.sample_rate,
            fmt.channels,
            fmt.sample_width * 8,
        )
        return pcm, fmt

    def synthesize(
        self,
        chunks: list[DialogueChunk],
        token_tracker: TokenTracker | None = None,
        progress: Progress | None = None,
        task_id: Any = None,
    ) -> list[bytes]:
        """
        Render *chunks* via the MOSS-TTSD SGLang server.

        Chunks are sent as independent ``POST /generate`` requests (up to
        ``max_workers`` in parallel) and reassembled in input order.  Each
        response is a base64 WAV, decoded here to raw PCM;
        :attr:`audio_format` is then set from the WAV header so the exporter
        uses the server's true format.  ``token_tracker`` is accepted for
        protocol symmetry but unused: a self-hosted server is not token-billed.

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

        Raises
        ------
        RuntimeError
            When the server errors, or when chunks come back in inconsistent
            audio formats (they cannot be concatenated).
        """
        import httpx

        if not chunks:
            return []

        audio_data = self._build_audio_data()
        if not audio_data:
            logger.warning(
                "No MOSS reference audio configured (tts.moss.speakerN.ref_audio) — "
                "voices will not be cloned or stable across runs."
            )

        max_workers = min(
            int(self._cfg.get("max_workers", _DEFAULT_MAX_WORKERS)), len(chunks)
        )
        timeout = float(self._cfg.get("timeout_seconds", _DEFAULT_TIMEOUT))
        logger.info(
            "Generating MOSS TTS for %d chunk(s) with up to %d worker(s)…",
            len(chunks),
            max_workers,
        )

        results: dict[int, bytes] = {}
        formats: dict[int, AudioFormat] = {}
        with httpx.Client(timeout=timeout) as client:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(self._synthesize_one, client, chunk, audio_data): i
                    for i, chunk in enumerate(chunks)
                }
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    pcm, fmt = future.result()
                    results[idx] = pcm
                    formats[idx] = fmt

                    if progress is not None and task_id is not None:
                        progress.advance(task_id)

        # Every chunk is concatenated into one track, so they must agree.
        distinct = set(formats.values())
        if len(distinct) > 1:
            # Sort on a plain tuple: AudioFormat is frozen but not ordered, so
            # sorting the dataclasses themselves would raise TypeError here and
            # mask the real error.
            summary = sorted(
                (f.sample_rate, f.channels, f.sample_width) for f in distinct
            )
            raise RuntimeError(
                f"MOSS server returned inconsistent audio formats across chunks "
                f"(sample_rate/channels/sample_width: {summary}); they cannot be "
                f"concatenated."
            )
        # Adopt the server's real format (read off the WAV header).  The CLI
        # reads `audio_format` only after synthesize, so the exporter always
        # sees this value rather than the provisional fallback.
        self.audio_format = distinct.pop()
        logger.info(
            "MOSS audio format: %d Hz, %d channel(s), %d-bit",
            self.audio_format.sample_rate,
            self.audio_format.channels,
            self.audio_format.sample_width * 8,
        )

        return [results[i] for i in range(len(chunks))]
