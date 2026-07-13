"""
Resolved runtime settings for the provider-agnostic LLM and TTS layers.

The YAML config grew a clean two-section shape:

``llm:``
    Which provider/model answers the *text* calls (dialogue, research, duo
    casting) and how to authenticate.

``tts:``
    Which speech backend renders audio (``gemini`` or ``moss``) and its
    per-backend options.

Both sections are optional: when absent, they are **synthesised from the legacy
``gemini:`` block** so pre-existing configs keep working untouched.  This module
holds the small resolver functions that turn the raw loaded config into typed
settings objects, isolating that back-compat logic from the CLI and the call
sites.

The LLM API key and the Gemini-TTS API key are resolved **independently** — a
run can generate its dialogue with OpenAI while still rendering audio through
Gemini TTS, so the two keys must never be conflated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LlmSettings:
    """
    Effective settings for the provider-agnostic text layer.

    Attributes
    ----------
    provider : str
        Provider name (``"gemini"`` / ``"openai"`` / ``"anthropic"`` /
        ``"ollama"`` / …).  Defaults to ``"gemini"``.
    text_model : str
        Bare model name for dialogue and duo generation (e.g.
        ``"gemini-2.5-flash"``).
    research_model : str or None
        Optional dedicated model for the research stage.  ``None`` means "reuse
        *text_model*".
    api_key : str or None
        Resolved provider API key (already env-expanded by the config loader),
        or ``None`` to let LiteLLM read the provider's conventional env var.
    api_base : str or None
        Optional base-URL override (local Ollama, self-hosted OpenAI-compatible
        endpoint, …).
    temperature : float or None
        Optional sampling temperature applied to text calls.
    extra_headers : dict[str, str] or None
        Arbitrary HTTP headers passed straight through to the provider on every
        text call (provider org ids, Anthropic beta flags, the Gemini
        ``x-goog-api-service-tier`` tier header, …).  ``None`` when unset.
    """

    provider: str
    text_model: str
    research_model: str | None
    api_key: str | None
    api_base: str | None
    temperature: float | None
    extra_headers: dict[str, str] | None


def resolve_llm_settings(cfg: dict[str, Any]) -> LlmSettings:
    """
    Resolve the effective LLM settings from the config.

    Precedence: the new ``llm:`` section wins field-by-field; any field it
    leaves unset falls back to the legacy ``gemini:`` block so old configs keep
    working.  The result always has a non-empty ``provider`` (defaults to
    ``"gemini"``).

    Parameters
    ----------
    cfg : dict
        The fully loaded, env-resolved configuration mapping.

    Returns
    -------
    LlmSettings
        Typed, resolved LLM settings.

    Raises
    ------
    KeyError
        Never raised directly, but callers should ensure ``text_model`` is set
        either under ``llm:`` or legacy ``gemini.text_model`` — otherwise
        ``text_model`` is ``None`` and the first API call will fail loudly.
    """
    gemini = cfg.get("gemini", {}) or {}
    llm = cfg.get("llm", {}) or {}
    gemini_research = gemini.get("research", {}) or {}
    top_research = cfg.get("research", {}) or {}

    provider = (llm.get("provider") or "gemini").strip().lower()
    text_model = llm.get("text_model") or gemini.get("text_model")
    # Research model precedence: llm.research_model > gemini.research.model >
    # legacy top-level research.model (which the CLI historically merged in).
    research_model = (
        llm.get("research_model")
        or gemini_research.get("model")
        or top_research.get("model")
    )
    api_key = llm.get("api_key") or gemini.get("api_key")
    api_base = llm.get("api_base")
    temperature = llm.get("temperature")

    # Arbitrary passthrough headers: the new, provider-agnostic way to send
    # things like the Gemini service tier or an Anthropic beta flag.
    extra_headers: dict[str, str] = dict(llm.get("extra_headers") or {})
    # Back-compat shim: the legacy Gemini-only `gemini.service_tier` becomes the
    # `x-goog-api-service-tier` header (Gemini provider only), unless an explicit
    # llm.extra_headers already set it.
    legacy_tier = gemini.get("service_tier")
    if (
        legacy_tier
        and provider == "gemini"
        and "x-goog-api-service-tier" not in extra_headers
    ):
        extra_headers["x-goog-api-service-tier"] = str(legacy_tier)

    return LlmSettings(
        provider=provider,
        text_model=text_model,
        research_model=research_model,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        extra_headers=extra_headers or None,
    )
