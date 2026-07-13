"""
Provider-agnostic text-generation layer built on LiteLLM.

Every text call in the pipeline (dialogue script, iterative research, voice-duo
casting) used to talk to Google Gemini directly via the ``google-genai`` SDK.
This module wraps :func:`litellm.completion` so the same three call patterns
work against any LiteLLM-supported provider (Gemini, OpenAI, Anthropic, Ollama,
…) selected purely from configuration.

The three call patterns collapse into a single :func:`complete` entry point:

- **plain text** — dialogue generation (:mod:`tts_podcast.llm_summarizer`).
- **structured output** — voice-duo casting (:mod:`tts_podcast.duo_generator`),
  via ``response_format`` with a JSON schema whose voice fields are enum-locked.
- **Google Search grounding** — research (:mod:`tts_podcast.research`), via the
  native Gemini ``googleSearch`` tool.  Grounding metadata (citations + issued
  search queries) is read back from LiteLLM's Gemini-specific hidden params.

``complete`` always returns a neutral :class:`LlmResult` — the raw provider
response never leaks upward, so call sites stay provider-agnostic.

Notes
-----
- ``litellm`` is a heavy import (it pulls ``openai`` + ``tokenizers``); it is
  imported lazily inside :func:`complete` so CLI startup stays fast, matching
  the ``TYPE_CHECKING`` gating used elsewhere in the codebase.
- Google Search grounding is a **Gemini-only** capability.  For non-Gemini
  providers the tool is silently omitted and :attr:`LlmResult.grounding` is
  ``None`` — research then runs on the model's own knowledge (degraded, not
  broken).  See :func:`complete`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from tts_podcast.retry import llm_retry

logger = logging.getLogger(__name__)


# Mapping from a bare provider name to the LiteLLM model-string prefix.  The
# prefix before the first "/" is how LiteLLM routes to a provider.  ``gemini/``
# targets Google AI Studio (simple API key); a bare model name would default to
# Vertex AI (full GCP creds), so the prefix is mandatory.  ``ollama_chat/`` is
# LiteLLM's recommended prefix for local Ollama chat models.
_PROVIDER_PREFIXES: dict[str, str] = {
    "gemini": "gemini/",
    "openai": "openai/",
    "anthropic": "anthropic/",
    "ollama": "ollama_chat/",
}


@dataclass
class Grounding:
    """
    Search-grounding metadata extracted from a provider response.

    Attributes
    ----------
    citations : list[tuple[str, str]]
        ``(title, uri)`` pairs for every cited web source.
    queries : list[str]
        The search queries the model actually issued.
    """

    citations: list[tuple[str, str]] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)


@dataclass
class LlmResult:
    """
    Neutral result of a single :func:`complete` call.

    Attributes
    ----------
    text : str
        The generated text (``choices[0].message.content``).  For structured
        output this is the raw JSON string, ready for ``json.loads``.
    input_tokens : int
        Prompt token count reported by the provider (0 when unavailable).
    output_tokens : int
        Completion token count reported by the provider (0 when unavailable).
    grounding : Grounding or None
        Populated only when Google Search grounding ran and returned metadata
        (Gemini only); ``None`` otherwise.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    grounding: Grounding | None = None


def build_model_string(provider: str | None, model: str) -> str:
    """
    Build the LiteLLM ``"<provider>/<model>"`` string from parts.

    A *model* that already contains a ``/`` is assumed to be fully qualified
    and returned unchanged, so power users can specify exotic routes verbatim
    in config (e.g. ``vertex_ai/gemini-2.5-flash``).

    Parameters
    ----------
    provider : str or None
        Provider name (case-insensitive).  ``None`` / empty defaults to
        ``"gemini"`` to preserve the pre-abstraction behaviour.
    model : str
        Bare model name (e.g. ``"gemini-2.5-flash"``) or an already-qualified
        LiteLLM model string.

    Returns
    -------
    str
        The LiteLLM model string (e.g. ``"gemini/gemini-2.5-flash"``).
    """
    if "/" in model:
        return model
    name = (provider or "gemini").strip().lower()
    prefix = _PROVIDER_PREFIXES.get(name, f"{name}/")
    return f"{prefix}{model}"


def is_gemini_model(model_string: str) -> bool:
    """
    Return ``True`` when *model_string* routes to a Gemini provider.

    Used to gate Gemini-only features (Google Search grounding, the
    ``service_tier`` header) so they are never sent to a provider that would
    reject them.

    Parameters
    ----------
    model_string : str
        A LiteLLM model string as produced by :func:`build_model_string`.

    Returns
    -------
    bool
        ``True`` for ``gemini/…`` and ``vertex_ai/…gemini…`` routes.
    """
    lowered = model_string.lower()
    return lowered.startswith("gemini/") or (
        lowered.startswith("vertex_ai/") and "gemini" in lowered
    )


def _extract_grounding(response: Any) -> Grounding | None:
    """
    Pull Google Search grounding metadata out of a LiteLLM Gemini response.

    LiteLLM attaches the native Gemini grounding blocks to
    ``response._hidden_params["vertex_ai_grounding_metadata"]`` (a list of
    dicts — the key says ``vertex_ai`` even on the ``gemini/`` AI-Studio route
    because both share LiteLLM's transformation code).  Each block may carry
    ``webSearchQueries`` and ``groundingChunks`` (``{"web": {"uri", "title"}}``).

    The reader is deliberately defensive: the hidden-params path is a
    version-sensitive escape hatch, so any missing/renamed field degrades to an
    empty result rather than raising.

    Parameters
    ----------
    response : Any
        The object returned by :func:`litellm.completion`.

    Returns
    -------
    Grounding or None
        Populated grounding when at least one citation or query was found,
        else ``None``.
    """
    hidden = getattr(response, "_hidden_params", None) or {}
    blocks = hidden.get("vertex_ai_grounding_metadata") or []
    if isinstance(blocks, dict):  # be liberal: some versions return a single dict
        blocks = [blocks]

    citations: list[tuple[str, str]] = []
    queries: list[str] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue
        for q in block.get("webSearchQueries") or []:
            queries.append(str(q))
        for chunk in block.get("groundingChunks") or []:
            web = chunk.get("web") if isinstance(chunk, dict) else None
            if not isinstance(web, dict):
                continue
            uri = web.get("uri") or ""
            title = web.get("title") or uri
            if uri:
                citations.append((str(title), str(uri)))

    if citations or queries:
        return Grounding(citations=citations, queries=queries)
    return None


def _build_messages(prompt: str, system_instruction: str | None) -> list[dict[str, str]]:
    """
    Assemble the OpenAI-style ``messages`` list from a prompt and system text.

    Parameters
    ----------
    prompt : str
        The user-turn content.
    system_instruction : str or None
        Optional system-role content prepended before the user turn.

    Returns
    -------
    list[dict[str, str]]
        The ``messages`` payload for :func:`litellm.completion`.
    """
    messages: list[dict[str, str]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    return messages


def complete(
    *,
    model: str,
    prompt: str,
    api_key: str | None = None,
    api_base: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_instruction: str | None = None,
    response_format: Any = None,
    reasoning_effort: str | None = None,
    extra_headers: dict[str, str] | None = None,
    enable_google_search: bool = False,
) -> LlmResult:
    """
    Run one provider-agnostic completion through LiteLLM and normalise the result.

    Wraps :func:`litellm.completion` with the shared 5xx retry policy
    (:data:`tts_podcast.retry.llm_retry`) and returns a neutral
    :class:`LlmResult`.

    Parameters
    ----------
    model : str
        LiteLLM model string (``"<provider>/<model>"``); build it with
        :func:`build_model_string`.
    prompt : str
        User-turn content.
    api_key : str or None, optional
        Provider API key.  When ``None``, LiteLLM falls back to the provider's
        conventional environment variable.
    api_base : str or None, optional
        Override the provider base URL (e.g. a local Ollama or a self-hosted
        OpenAI-compatible endpoint).
    temperature : float or None, optional
        Sampling temperature; omitted from the call when ``None``.
    max_tokens : int or None, optional
        Maximum completion tokens; omitted when ``None``.
    system_instruction : str or None, optional
        System-role content prepended to the messages.
    response_format : Any, optional
        Structured-output specification passed straight to LiteLLM — a Pydantic
        model, or a ``{"type": "json_schema", ...}`` / ``{"type": "json_object"}``
        dict.  On providers with native schema enforcement (Gemini, OpenAI,
        Anthropic) enum constraints are hard-enforced; elsewhere LiteLLM
        downgrades to JSON mode, so callers must keep Python-side validation.
    reasoning_effort : str or None, optional
        ``"minimal" | "low" | "medium" | "high"`` — LiteLLM maps this to each
        provider's thinking/reasoning control (Gemini thinking budget, OpenAI
        reasoning effort, …).  Omitted when ``None``.
    extra_headers : dict[str, str] or None, optional
        Arbitrary HTTP headers forwarded verbatim to the provider (provider org
        ids, Anthropic beta flags, the Gemini ``x-goog-api-service-tier`` tier
        header, …).  It is the caller's responsibility to only set headers the
        target provider accepts.  Omitted when ``None``/empty.
    enable_google_search : bool, optional
        When ``True`` and *model* is a Gemini route, attaches the native
        ``googleSearch`` grounding tool.  Ignored (with a warning) for
        non-Gemini providers.

    Returns
    -------
    LlmResult
        Neutral result carrying the text, token counts, and optional grounding.

    Raises
    ------
    litellm.exceptions.APIError
        Re-raised after the retry budget is exhausted, or immediately for
        non-retryable (4xx) errors.
    """
    # Lazy import: keep the heavy litellm import off the CLI startup path.
    import litellm

    gemini_route = is_gemini_model(model)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": _build_messages(prompt, system_instruction),
    }
    if api_key is not None:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        kwargs["response_format"] = response_format
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    # Passthrough headers, forwarded verbatim (the caller owns provider
    # compatibility).  Covers the Gemini service-tier header and any
    # provider-specific header a future backend needs.
    if extra_headers:
        kwargs["extra_headers"] = dict(extra_headers)

    # Google Search grounding is Gemini-native.  Attach the tool only on a
    # Gemini route; warn (once per call) if a caller asked for it elsewhere.
    if enable_google_search:
        if gemini_route:
            kwargs["tools"] = [{"googleSearch": {}}]
            # Opt out of LiteLLM's MCP-tool handler: whenever `tools` is set it
            # eagerly imports litellm.responses.mcp.*, which transitively needs
            # fastapi (the proxy extra we deliberately don't install).  We never
            # pass MCP tools, so skipping that path keeps the base install lean.
            kwargs["_skip_mcp_handler"] = True
        else:
            logger.warning(
                "Google Search grounding requested but model %r is not a Gemini "
                "route — running without grounding.",
                model,
            )

    @llm_retry
    def _call() -> Any:
        return litellm.completion(**kwargs)

    response = _call()

    # --- Text -----------------------------------------------------------
    text = ""
    try:
        text = response.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError):
        logger.warning("LiteLLM response carried no message content.")

    # --- Usage ----------------------------------------------------------
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    # --- Grounding (Gemini only) ---------------------------------------
    grounding = _extract_grounding(response) if gemini_route else None

    return LlmResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        grounding=grounding,
    )
