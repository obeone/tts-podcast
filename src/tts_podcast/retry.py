"""
Retry utilities for Google Gemini API calls.

Provides a pre-configured tenacity retry decorator that transparently retries
on transient server-side failures (HTTP 5xx) with exponential back-off.
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


#: Retry decorator for Gemini API calls.
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
