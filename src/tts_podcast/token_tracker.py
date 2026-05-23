"""
Token usage and cost tracker for Gemini API calls.

Accumulates ``prompt_token_count`` and ``candidates_token_count`` across
multiple API calls, computes an estimated cost in USD based on configurable
per-model pricing, and produces a human-readable summary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _ModelUsage:
    """Raw token counters for a single model.

    Attributes
    ----------
    input_tokens : int
        Cumulative prompt (input) token count.
    output_tokens : int
        Cumulative candidates (output) token count.
    calls : int
        Number of API calls recorded for this model.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


class TokenTracker:
    """
    Accumulate Gemini API token usage and estimate costs.

    Usage
    -----
    1. Optionally call :meth:`set_pricing` with a per-model rate table.
    2. After each ``generate_content`` call, pass ``response.usage_metadata``
       to :meth:`record_usage`.
    3. Call :meth:`summary` or :meth:`total_cost` for reporting.

    Parameters
    ----------
    pricing : dict[str, dict[str, float]] or None
        Optional initial pricing table.  Keys are model names; values are
        dicts with ``input_per_1m`` and/or ``output_per_1m`` (USD per 1M
        tokens).  Can also be set via :meth:`set_pricing`.

    Examples
    --------
    >>> tracker = TokenTracker()
    >>> tracker.set_pricing({"gemini-flash": {"input_per_1m": 0.075, "output_per_1m": 0.30}})
    >>> tracker.record("gemini-flash", input_tokens=1000, output_tokens=200)
    >>> print(tracker.summary())
    """

    def __init__(
        self,
        pricing: dict[str, dict] | None = None,
        service_tier: str | None = None,
    ) -> None:
        self._usage: dict[str, _ModelUsage] = {}
        self._raw_pricing: dict[str, dict] = pricing or {}
        self._service_tier = service_tier
        self._pricing: dict[str, dict[str, float]] = self._resolve_pricing(
            self._raw_pricing, self._service_tier,
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_pricing(
        raw: dict[str, dict],
        service_tier: str | None,
    ) -> dict[str, dict[str, float]]:
        """
        Resolve tier-aware pricing into a flat per-model table.

        Supports two formats per model:

        * **Flat** — ``{"input_per_1m": float, "output_per_1m": float}``
        * **Tier-aware** — ``{"standard": {…}, "flex": {…}, "priority": {…}}``

        When tier-aware entries are present, the active *service_tier* selects
        the matching sub-dict.  Falls back to ``"standard"`` if the requested
        tier is missing, then to the first available sub-tier.

        Parameters
        ----------
        raw : dict[str, dict]
            Raw pricing table (may mix flat and tier-aware entries).
        service_tier : str or None
            Active service tier (``"flex"``, ``"priority"``, or ``None`` for
            standard).

        Returns
        -------
        dict[str, dict[str, float]]
            Flat pricing table keyed by model name.
        """
        resolved: dict[str, dict[str, float]] = {}
        tier = service_tier or "standard"

        for model, value in raw.items():
            if "input_per_1m" in value:
                # Flat format
                resolved[model] = value
            elif isinstance(value, dict):
                # Tier-aware format
                if tier in value:
                    resolved[model] = value[tier]
                elif "standard" in value:
                    resolved[model] = value["standard"]
                else:
                    # Fallback to first available tier
                    first = next(iter(value.values()), {})
                    if isinstance(first, dict):
                        resolved[model] = first

        return resolved

    def set_pricing(self, pricing: dict[str, dict[str, float]]) -> None:
        """
        Set or replace the per-model pricing table.

        Parameters
        ----------
        pricing : dict[str, dict[str, float]]
            Maps model names to ``{"input_per_1m": float, "output_per_1m": float}``.
        """
        self._pricing = pricing

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, model: str, input_tokens: int, output_tokens: int = 0) -> None:
        """
        Add token counts for a completed API call.

        Parameters
        ----------
        model : str
            Gemini model name (e.g. ``"gemini-2.5-flash"``).
        input_tokens : int
            Prompt token count for this call.
        output_tokens : int, optional
            Candidates token count for this call, by default 0.
        """
        if model not in self._usage:
            self._usage[model] = _ModelUsage()
        usage = self._usage[model]
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens
        usage.calls += 1
        logger.debug(
            "Token recorded — model=%s input=%d output=%d (total calls=%d)",
            model,
            input_tokens,
            output_tokens,
            usage.calls,
        )

    def record_usage(self, model: str, usage_metadata: Any) -> None:
        """
        Convenience wrapper that reads ``usage_metadata`` from a Gemini response.

        Silently ignores ``None`` metadata so callers need not guard against it.

        Parameters
        ----------
        model : str
            Gemini model name.
        usage_metadata : google.genai.types.GenerateContentResponseUsageMetadata or None
            The ``response.usage_metadata`` object.  Uses
            ``prompt_token_count`` and ``candidates_token_count`` attributes.
        """
        if usage_metadata is None:
            return
        input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
        self.record(model, input_tokens, output_tokens)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def total_cost(self) -> float:
        """
        Return the total estimated cost in USD across all recorded models.

        Returns
        -------
        float
            Estimated cost in USD, or 0.0 if no pricing data is configured.
        """
        total = 0.0
        for model, usage in self._usage.items():
            pricing = self._pricing.get(model, {})
            input_rate = pricing.get("input_per_1m", 0.0)
            output_rate = pricing.get("output_per_1m", 0.0)
            total += (usage.input_tokens * input_rate + usage.output_tokens * output_rate) / 1_000_000
        return total

    def _model_cost(self, model: str) -> float:
        """Return the estimated cost in USD for a single model."""
        usage = self._usage.get(model)
        if not usage:
            return 0.0
        pricing = self._pricing.get(model, {})
        input_rate = pricing.get("input_per_1m", 0.0)
        output_rate = pricing.get("output_per_1m", 0.0)
        return (usage.input_tokens * input_rate + usage.output_tokens * output_rate) / 1_000_000

    def summary(self) -> str:
        """
        Return a multi-line human-readable token and cost summary.

        Returns
        -------
        str
            Formatted summary string, one line per model plus a totals line.
        """
        if not self._usage:
            return "Token usage: no API calls recorded."

        lines = ["Token usage:"]
        total_input = 0
        total_output = 0

        for model, usage in self._usage.items():
            cost = self._model_cost(model)
            call_word = "call" if usage.calls == 1 else "calls"
            cost_str = f"${cost:.4f}" if self._pricing.get(model) else "n/a"
            lines.append(
                f"  {model}: {usage.input_tokens:,} in + {usage.output_tokens:,} out"
                f" ({usage.calls} {call_word}) — {cost_str}"
            )
            total_input += usage.input_tokens
            total_output += usage.output_tokens

        total_cost = self.total_cost()
        total_cost_str = f"${total_cost:.4f}" if self._pricing else "n/a"
        lines.append(
            f"  ─── Total: {total_input:,} in + {total_output:,} out — {total_cost_str}"
        )
        return "\n".join(lines)

    def live_line(self) -> str:
        """
        Return a compact one-line token/cost string suitable for progress displays.

        Returns
        -------
        str
            E.g. ``"tokens: 12,345 in / 3,210 out — $0.0123"``.
        """
        total_input = sum(u.input_tokens for u in self._usage.values())
        total_output = sum(u.output_tokens for u in self._usage.values())
        cost = self.total_cost()
        cost_str = f" — ${cost:.4f}" if self._pricing else ""
        return f"tokens: {total_input:,} in / {total_output:,} out{cost_str}"
