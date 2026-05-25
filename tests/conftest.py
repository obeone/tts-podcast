"""
Project-wide pytest configuration.

Currently used to keep test-support assets under ``tests/fixtures/`` from being
collected as tests.  The snapshot fixture for the dialogue prompt lives there
and must NEVER be regenerated as a side effect of a test run — otherwise the
byte-identical guarantee on :func:`tts_podcast.llm_summarizer._build_prompt`
silently evaporates.
"""

from __future__ import annotations


collect_ignore_glob = ["fixtures/*"]
