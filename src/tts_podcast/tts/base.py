"""
Common interface for the pluggable text-to-speech backends.

The pipeline synthesises a list of :class:`~tts_podcast.llm_summarizer.DialogueChunk`
objects into raw PCM audio, then hands those bytes to
:mod:`tts_podcast.audio_exporter` for encoding.  Historically that was hard-wired
to Gemini multi-speaker TTS (24 kHz / mono / 16-bit).  This module defines the
neutral contract both backends implement so the CLI can pick one from config:

- :class:`AudioFormat` — the raw-PCM shape a backend emits, so the exporter is
  no longer pinned to Gemini's constants.
- :class:`TtsBackend` — the ``synthesize`` protocol every backend satisfies.

Each concrete backend is constructed with its own resolved config and knows its
own :attr:`~TtsBackend.audio_format`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rich.progress import Progress

    from tts_podcast.llm_summarizer import DialogueChunk
    from tts_podcast.token_tracker import TokenTracker


@dataclass(frozen=True)
class AudioFormat:
    """
    Raw-PCM audio format emitted by a TTS backend.

    Attributes
    ----------
    sample_rate : int
        Sample rate in Hz (e.g. ``24000``).
    channels : int
        Channel count (``1`` = mono, ``2`` = stereo).
    sample_width : int
        Bytes per sample (``2`` = 16-bit signed little-endian PCM).
    """

    sample_rate: int
    channels: int
    sample_width: int


@runtime_checkable
class TtsBackend(Protocol):
    """
    Protocol implemented by every speech backend.

    A backend turns dialogue chunks into ordered raw-PCM byte blobs and
    advertises the format of those bytes via :attr:`audio_format` so the
    exporter can assemble them without assuming Gemini's constants.
    """

    #: The raw-PCM format this backend emits.
    audio_format: AudioFormat

    def synthesize(
        self,
        chunks: list[DialogueChunk],
        token_tracker: TokenTracker | None = None,
        progress: Progress | None = None,
        task_id: Any = None,
    ) -> list[bytes]:
        """
        Synthesise dialogue chunks into ordered raw-PCM byte blobs.

        Parameters
        ----------
        chunks : list[DialogueChunk]
            Ordered dialogue chunks to render.
        token_tracker : TokenTracker or None, optional
            When provided, the backend records any billable usage it incurs.
        progress : rich.progress.Progress or None, optional
            Progress instance advanced once per completed chunk.
        task_id : Any, optional
            Task identifier returned by ``progress.add_task()``.

        Returns
        -------
        list[bytes]
            One raw-PCM blob per input chunk, in input order, matching
            :attr:`audio_format`.
        """
        ...
