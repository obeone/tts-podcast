"""
Pluggable text-to-speech backends for the tts-podcast pipeline.

Public surface:

- :class:`~tts_podcast.tts.base.AudioFormat` / :class:`~tts_podcast.tts.base.TtsBackend`
  — the neutral contract.
- :func:`resolve_tts_backend` — the factory that turns the loaded config into a
  concrete backend instance (``gemini`` or ``moss``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from tts_podcast.settings import resolve_tts_settings
from tts_podcast.tts.base import AudioFormat, TtsBackend

if TYPE_CHECKING:
    from typing import Any

__all__ = [
    "AudioFormat",
    "TtsBackend",
    "resolve_tts_backend",
]


def resolve_tts_backend(cfg: dict[str, "Any"], gemini_cfg: dict) -> TtsBackend:
    """
    Build the active TTS backend from the loaded config.

    The backend is chosen by ``tts.backend`` (defaulting to ``"gemini"`` when
    the ``tts:`` section is absent, so legacy configs render through Gemini TTS
    unchanged).

    Parameters
    ----------
    cfg : dict
        The fully loaded, env-resolved configuration mapping.
    gemini_cfg : dict
        The resolved Gemini configuration section (speakers already injected by
        the duo layer).  Consumed by :class:`GeminiTtsBackend`.

    Returns
    -------
    TtsBackend
        A ready-to-use backend instance.

    Raises
    ------
    click.BadParameter
        When ``tts.backend`` names an unknown or unavailable backend.
    """
    settings = resolve_tts_settings(cfg)

    if settings.backend == "gemini":
        # Lazy import keeps `tts.base` (and the exporter that imports it) free of
        # the heavy google-genai chain that GeminiTtsBackend pulls in.
        from tts_podcast.tts.gemini_backend import GeminiTtsBackend

        return GeminiTtsBackend(gemini_cfg)

    if settings.backend == "moss":
        raise click.BadParameter(
            "The 'moss' TTS backend is not available in this build."
        )

    raise click.BadParameter(
        f"Unknown tts.backend {settings.backend!r}. Valid backends: gemini, moss."
    )
