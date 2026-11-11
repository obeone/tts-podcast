"""
TTS model capability detection.

Gemini TTS models come in two request schemas, and both the dialogue writer
(:mod:`tts_podcast.llm_summarizer`) and the synthesiser
(:mod:`tts_podcast.tts_generator`) must agree on which one is in play:

* **Preamble schema** (``gemini-2.5-*-tts``, ``gemini-3.1-flash-tts-preview``):
  one text block per request.  A natural-language preamble (audio profile,
  director's notes) is followed by ``Name: line`` turns, and delivery cues are
  written inline in the text.
* **Speech-metadata schema** (Gemini 3.8 TTS and later): the text is a
  *verbatim transcript*.  Anything written in it may be spoken aloud, speaker
  labels and stage directions included, so each turn is sent as its own part
  carrying ``speech_metadata`` (``speaker`` plus a short ``style``), and only
  point-in-time vocal events stay inline, as angle-bracket tags.

This module has no heavy imports on purpose, so either side can import it at
module scope without pulling in the Gemini SDK.
"""

from __future__ import annotations

import re

# First model generation that uses the speech-metadata schema.
_SPEECH_METADATA_SINCE = (3, 8)

# Leading "gemini-<major>.<minor>" of a model name.  The minor part is optional
# so a hypothetical "gemini-4-flash-tts" still parses.
_MODEL_VERSION_RE = re.compile(r"^gemini-(\d+)(?:\.(\d+))?")


def uses_speech_metadata(tts_model: object) -> bool:
    """
    Tell whether a TTS model expects the speech-metadata request schema.

    Parameters
    ----------
    tts_model : object
        The configured ``gemini.tts_model`` value.  Anything that is not a
        recognisable ``gemini-<major>.<minor>`` name is treated as a legacy
        model, which keeps unknown or missing values on the historical path.

    Returns
    -------
    bool
        ``True`` for Gemini 3.8 TTS and later (structured turns with
        ``speech_metadata``), ``False`` for earlier models (text preamble).

    Examples
    --------
    >>> uses_speech_metadata("gemini-3.8-flash-tts")
    True
    >>> uses_speech_metadata("gemini-3.8-flash-lite-tts")
    True
    >>> uses_speech_metadata("gemini-3.1-flash-tts-preview")
    False
    >>> uses_speech_metadata(None)
    False
    """
    match = _MODEL_VERSION_RE.match(str(tts_model or "").strip().lower())
    if match is None:
        return False
    version = (int(match.group(1)), int(match.group(2) or 0))
    return version >= _SPEECH_METADATA_SINCE
