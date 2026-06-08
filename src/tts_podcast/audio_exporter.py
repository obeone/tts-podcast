"""
Audio exporter — converts raw PCM chunks to an MP3 or WAV file.

Gemini TTS produces raw PCM data at 24 kHz, mono, 16-bit signed
little-endian.  This module assembles the individual per-chunk bytes objects
into a single :class:`~pydub.AudioSegment`, then exports the result to the
requested format via ffmpeg.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from pydub import AudioSegment

logger = logging.getLogger(__name__)

# Gemini TTS output format constants
_SAMPLE_RATE = 24_000   # Hz
_SAMPLE_WIDTH = 2       # bytes (16-bit)
_CHANNELS = 1           # mono


def _combine_pcm(pcm_chunks: list[bytes]) -> AudioSegment:
    """
    Concatenate raw PCM chunks into a single :class:`~pydub.AudioSegment`.

    Parameters
    ----------
    pcm_chunks : list[bytes]
        Ordered list of raw PCM audio chunks (24 kHz, mono, 16-bit LE)
        produced by :func:`~tts_podcast.tts_generator.generate_audio_chunks`.

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
            sample_width=_SAMPLE_WIDTH,
            frame_rate=_SAMPLE_RATE,
            channels=_CHANNELS,
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
) -> Path:
    """
    Concatenate raw PCM chunks and export to an audio file.

    Parameters
    ----------
    pcm_chunks : list[bytes]
        Ordered list of raw PCM audio chunks (24 kHz, mono, 16-bit LE)
        produced by :func:`~tts_podcast.tts_generator.generate_audio_chunks`.
    output_path : str | Path
        Destination file path.  Parent directories are created automatically.
    fmt : str, optional
        Output format passed to pydub (``"mp3"`` or ``"wav"``).
        Defaults to ``"mp3"``.

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
    combined = _combine_pcm(pcm_chunks)

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


def encode_audio(pcm_chunks: list[bytes], fmt: str = "mp3") -> bytes:
    """
    Concatenate raw PCM chunks and encode to an in-memory audio blob.

    Mirrors :func:`export_audio` but returns the encoded bytes instead of
    writing a file — used to stream the podcast to stdout (``--output -``).

    Parameters
    ----------
    pcm_chunks : list[bytes]
        Ordered list of raw PCM audio chunks (24 kHz, mono, 16-bit LE).
    fmt : str, optional
        Output format passed to pydub (``"mp3"`` or ``"wav"``).
        Defaults to ``"mp3"``.

    Returns
    -------
    bytes
        The encoded audio in the requested format.

    Raises
    ------
    ValueError
        If *pcm_chunks* is empty.
    """
    combined = _combine_pcm(pcm_chunks)

    logger.info(
        "Total duration: %.2f s — encoding as %s for stdout",
        combined.duration_seconds,
        fmt.upper(),
    )

    buffer = io.BytesIO()
    combined.export(buffer, format=fmt)
    return buffer.getvalue()
