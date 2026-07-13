"""
Gemini multi-speaker TTS backend.

Thin adapter that exposes the existing
:func:`tts_podcast.tts_generator.generate_audio_chunks` implementation through
the :class:`~tts_podcast.tts.base.TtsBackend` protocol.  The synthesis logic,
the natural-language preamble, and the hard "personality is read verbatim"
invariant all live unchanged in :mod:`tts_podcast.tts_generator`; this class only
wraps it and declares the fixed Gemini output format (24 kHz / mono / 16-bit LE).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tts_podcast.tts.base import AudioFormat
from tts_podcast.tts_generator import generate_audio_chunks

if TYPE_CHECKING:
    from rich.progress import Progress

    from tts_podcast.llm_summarizer import DialogueChunk
    from tts_podcast.token_tracker import TokenTracker

#: Gemini multi-speaker TTS always returns 24 kHz, mono, 16-bit signed LE PCM.
_GEMINI_AUDIO_FORMAT = AudioFormat(sample_rate=24_000, channels=1, sample_width=2)


class GeminiTtsBackend:
    """
    :class:`~tts_podcast.tts.base.TtsBackend` backed by Gemini multi-speaker TTS.

    Parameters
    ----------
    gemini_cfg : dict
        Resolved Gemini configuration section carrying ``api_key``,
        ``tts_model``, ``speaker1`` / ``speaker2`` (``name`` / ``voice`` /
        ``personality``), ``language``, and optional ``tts_style``.  Passed
        straight through to :func:`~tts_podcast.tts_generator.generate_audio_chunks`.
    """

    #: This backend's fixed raw-PCM output format.
    audio_format: AudioFormat = _GEMINI_AUDIO_FORMAT

    def __init__(self, gemini_cfg: dict) -> None:
        self._gemini_cfg = gemini_cfg

    def synthesize(
        self,
        chunks: list[DialogueChunk],
        token_tracker: TokenTracker | None = None,
        progress: Progress | None = None,
        task_id: Any = None,
    ) -> list[bytes]:
        """
        Render *chunks* via Gemini multi-speaker TTS.

        Delegates verbatim to
        :func:`~tts_podcast.tts_generator.generate_audio_chunks`; see that
        function for the parallelism, preamble, and token-accounting details.

        Parameters
        ----------
        chunks : list[DialogueChunk]
            Ordered dialogue chunks to synthesise.
        token_tracker : TokenTracker or None, optional
            Records per-call TTS token usage when provided.
        progress : rich.progress.Progress or None, optional
            Progress instance advanced once per completed chunk.
        task_id : Any, optional
            Task identifier returned by ``progress.add_task()``.

        Returns
        -------
        list[bytes]
            Raw PCM (24 kHz / mono / 16-bit LE) blobs, one per input chunk.
        """
        return generate_audio_chunks(
            chunks,
            self._gemini_cfg,
            token_tracker=token_tracker,
            progress=progress,
            task_id=task_id,
        )
