"""
Retry utilities for LLM and TTS API calls.

Provides pre-configured tenacity retry decorators that transparently retry on
transient server-side failures with exponential back-off:

- :data:`gemini_retry` — for direct ``google-genai`` SDK calls (the Gemini TTS
  backend), retrying on :class:`google.genai.errors.ServerError` (HTTP 5xx).
- :data:`llm_retry` — for provider-agnostic text calls routed through LiteLLM,
  retrying on LiteLLM's transient exception types (5xx, connection, timeout).

Both share the same back-off schedule (2 s → 60 s, 5 attempts).
"""

from __future__ import annotations

import logging

from google.genai import errors as genai_errors
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

#: Maximum number of attempts before giving up (1 original + N-1 retries).
_MAX_ATTEMPTS = 5


def _is_retryable(exc: BaseException) -> bool:
    """
    Return True if *exc* is a transient Gemini server-side error.

    Parameters
    ----------
    exc : BaseException
        The exception to evaluate.

    Returns
    -------
    bool
        True for :class:`google.genai.errors.ServerError` (HTTP 5xx), which
        covers 503 Service Unavailable as well as other transient failures.
        False for all other exception types (client errors, timeouts, etc.).
    """
    return isinstance(exc, genai_errors.ServerError)


def _is_retryable_llm(exc: BaseException) -> bool:
    """
    Return True if *exc* is a transient LiteLLM error worth retrying.

    LiteLLM is imported lazily inside this predicate so the (heavy) import is
    only paid when an exception actually needs classifying — importing
    ``retry`` at module load stays cheap, and environments without LiteLLM
    installed simply treat every error as non-retryable.

    Retryable: server-side 5xx (:class:`litellm.InternalServerError`,
    :class:`litellm.ServiceUnavailableError`), connection failures
    (:class:`litellm.APIConnectionError`), and timeouts
    (:class:`litellm.Timeout`).  Deliberately NOT retried: 4xx client errors
    (bad request, auth, not found) and rate limits — mirroring the 5xx-only
    policy of :func:`_is_retryable`.

    Parameters
    ----------
    exc : BaseException
        The exception to evaluate.

    Returns
    -------
    bool
        ``True`` for transient LiteLLM errors, ``False`` otherwise.
    """
    try:
        import litellm
    except ImportError:
        return False

    retryable = (
        litellm.InternalServerError,
        litellm.ServiceUnavailableError,
        litellm.APIConnectionError,
        litellm.Timeout,
    )
    return isinstance(exc, retryable)


#: Retry decorator for direct Gemini (``google-genai``) API calls.
#:
#: Behaviour:
#: - Retries on any :class:`~google.genai.errors.ServerError` (HTTP 5xx).
#: - Waits 2 s after the first failure, doubling each time up to 60 s.
#: - Gives up after ``_MAX_ATTEMPTS`` total attempts and re-raises the error.
#: - Logs a WARNING before each sleep so the caller is kept informed.
gemini_retry = retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


#: Retry decorator for provider-agnostic text calls routed through LiteLLM.
#:
#: Same back-off schedule as :data:`gemini_retry` (2 s → 60 s, 5 attempts) but
#: keyed on LiteLLM's transient exception types (see :func:`_is_retryable_llm`)
#: so it works across every provider LiteLLM supports (Gemini, OpenAI,
#: Anthropic, Ollama, …).
llm_retry = retry(
    retry=retry_if_exception(_is_retryable_llm),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
