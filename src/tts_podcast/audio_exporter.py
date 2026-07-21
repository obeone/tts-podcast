"""
Audio exporter — converts raw PCM chunks to an MP3 or WAV file.

TTS backends emit raw PCM whose format (sample rate / channels / bit depth) is
declared by the backend via :class:`~tts_podcast.tts.base.AudioFormat` — Gemini
TTS is 24 kHz / mono / 16-bit LE, but the moss backend may differ.  This module
assembles the individual per-chunk bytes objects into a single
:class:`~pydub.AudioSegment` using the given format, then exports the result to
the requested container via ffmpeg.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from pydub import AudioSegment

from tts_podcast.tts.base import AudioFormat

logger = logging.getLogger(__name__)

#: Default raw-PCM format when a caller does not specify one — the historical
#: Gemini TTS shape (24 kHz, mono, 16-bit signed little-endian), so existing
#: callers keep their behaviour.
_DEFAULT_AUDIO_FORMAT = AudioFormat(sample_rate=24_000, channels=1, sample_width=2)


def _combine_pcm(
    pcm_chunks: list[bytes],
    audio_format: AudioFormat = _DEFAULT_AUDIO_FORMAT,
) -> AudioSegment:
    """
    Concatenate raw PCM chunks into a single :class:`~pydub.AudioSegment`.

    Parameters
    ----------
    pcm_chunks : list[bytes]
        Ordered list of raw PCM audio chunks produced by a TTS backend.
    audio_format : AudioFormat, optional
        The raw-PCM format of *pcm_chunks*.  Defaults to the Gemini shape
        (24 kHz / mono / 16-bit LE).

    Returns
    -------
    pydub.AudioSegment
        The concatenated audio segment.

    Raises
    ------
    ValueError
        If *pcm_chunks* is empty.
    """
    if not pcm_chunks:
        raise ValueError("pcm_chunks must not be empty.")

    logger.info("Assembling %d audio chunk(s)…", len(pcm_chunks))

    segments: list[AudioSegment] = []
    for i, pcm in enumerate(pcm_chunks):
        seg = AudioSegment(
            data=pcm,
            sample_width=audio_format.sample_width,
            frame_rate=audio_format.sample_rate,
            channels=audio_format.channels,
        )
        logger.debug("Chunk %d: %.2f s", i, seg.duration_seconds)
        segments.append(seg)

    combined = segments[0]
    for seg in segments[1:]:
        combined = combined + seg

    return combined


def export_audio(
    pcm_chunks: list[bytes],
    output_path: str | Path,
    fmt: str = "mp3",
    audio_format: AudioFormat = _DEFAULT_AUDIO_FORMAT,
) -> Path:
    """
    Concatenate raw PCM chunks and export to an audio file.

    Parameters
    ----------
    pcm_chunks : list[bytes]
        Ordered list of raw PCM audio chunks produced by a TTS backend.
    output_path : str | Path
        Destination file path.  Parent directories are created automatically.
    fmt : str, optional
        Output format passed to pydub (``"mp3"`` or ``"wav"``).
        Defaults to ``"mp3"``.
    audio_format : AudioFormat, optional
        The raw-PCM format of *pcm_chunks* (pass the active backend's
        ``audio_format``).  Defaults to the Gemini shape.

    Returns
    -------
    Path
        Absolute path of the written audio file.

    Raises
    ------
    ValueError
        If *pcm_chunks* is empty.

    Examples
    --------
    >>> path = export_audio(chunks, "output/episode.mp3", fmt="mp3")
    >>> path.exists()
    True
    """
    combined = _combine_pcm(pcm_chunks, audio_format)

    logger.info(
        "Total duration: %.2f s — exporting as %s",
        combined.duration_seconds,
        fmt.upper(),
    )

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    combined.export(str(out), format=fmt)
    logger.info("Audio saved to %s", out)

    return out


def encode_audio(
    pcm_chunks: list[bytes],
    fmt: str = "mp3",
    audio_format: AudioFormat = _DEFAULT_AUDIO_FORMAT,
) -> bytes:
    """
    Concatenate raw PCM chunks and encode to an in-memory audio blob.

    Mirrors :func:`export_audio` but returns the encoded bytes instead of
    writing a file — used to stream the podcast to stdout (``--output -``).

    Parameters
    ----------
    pcm_chunks : list[bytes]
        Ordered list of raw PCM audio chunks produced by a TTS backend.
    fmt : str, optional
        Output format passed to pydub (``"mp3"`` or ``"wav"``).
        Defaults to ``"mp3"``.
    audio_format : AudioFormat, optional
        The raw-PCM format of *pcm_chunks* (pass the active backend's
        ``audio_format``).  Defaults to the Gemini shape.

    Returns
    -------
    bytes
        The encoded audio in the requested format.

    Raises
    ------
    ValueError
        If *pcm_chunks* is empty.
    """
    combined = _combine_pcm(pcm_chunks, audio_format)

    logger.info(
        "Total duration: %.2f s — encoding as %s for stdout",
        combined.duration_seconds,
        fmt.upper(),
    )

    buffer = io.BytesIO()
    combined.export(buffer, format=fmt)
    return buffer.getvalue()
