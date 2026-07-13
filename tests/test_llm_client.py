"""
Tests for the provider-agnostic LLM client (:mod:`tts_podcast.llm_client`).

These exercise the thin seam over ``litellm.completion``: model-string
building, the request kwargs assembled for each call shape (plain, structured,
grounded), and the neutral :class:`LlmResult` extraction (text, token usage,
grounding).  ``litellm.completion`` is patched so nothing hits the network.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tts_podcast.llm_client import (
    Grounding,
    LlmResult,
    build_model_string,
    complete,
    is_gemini_model,
)


def _fake_response(
    text: str = "hello",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    hidden: dict | None = None,
):
    """Build a minimal litellm-shaped response object for patching."""
    choice = SimpleNamespace(message=SimpleNamespace(content=text))
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
        _hidden_params=hidden or {},
    )


class TestBuildModelString:
    """Provider-prefix construction for the LiteLLM model string."""

    @pytest.mark.parametrize(
        "provider,model,expected",
        [
            ("gemini", "gemini-2.5-flash", "gemini/gemini-2.5-flash"),
            ("openai", "gpt-4o", "openai/gpt-4o"),
            ("anthropic", "claude-sonnet-4-5", "anthropic/claude-sonnet-4-5"),
            ("ollama", "llama3.1", "ollama_chat/llama3.1"),
            (None, "gemini-2.5-flash", "gemini/gemini-2.5-flash"),
            ("GEMINI", "gemini-2.5-flash", "gemini/gemini-2.5-flash"),
            ("custom", "m", "custom/m"),
        ],
    )
    def test_prefixes(self, provider, model, expected):
        assert build_model_string(provider, model) == expected

    def test_already_qualified_passthrough(self):
        assert build_model_string("gemini", "vertex_ai/gemini-2.5-flash") == (
            "vertex_ai/gemini-2.5-flash"
        )


class TestIsGeminiModel:
    """Gemini-route detection used to gate grounding and headers."""

    def test_gemini_route(self):
        assert is_gemini_model("gemini/gemini-2.5-flash") is True

    def test_vertex_gemini_route(self):
        assert is_gemini_model("vertex_ai/gemini-2.5-flash") is True

    def test_non_gemini(self):
        assert is_gemini_model("openai/gpt-4o") is False


class TestCompleteRequest:
    """The kwargs assembled for litellm.completion."""

    def test_google_search_gemini_sets_tool_and_skips_mcp(self):
        # Regression: passing `tools` makes litellm import its MCP handler,
        # which needs fastapi (the proxy extra we don't install).  The
        # `_skip_mcp_handler` flag must be sent to avoid that import.
        with patch("litellm.completion", return_value=_fake_response()) as m:
            complete(
                model="gemini/gemini-2.5-flash",
                prompt="p",
                enable_google_search=True,
            )
        kwargs = m.call_args.kwargs
        assert kwargs["tools"] == [{"googleSearch": {}}]
        assert kwargs["_skip_mcp_handler"] is True

    def test_google_search_non_gemini_omits_tool(self):
        with patch("litellm.completion", return_value=_fake_response()) as m:
            complete(model="openai/gpt-4o", prompt="p", enable_google_search=True)
        kwargs = m.call_args.kwargs
        assert "tools" not in kwargs
        assert "_skip_mcp_handler" not in kwargs

    def test_optional_kwargs_are_forwarded(self):
        with patch("litellm.completion", return_value=_fake_response()) as m:
            complete(
                model="openai/gpt-4o",
                prompt="p",
                api_key="K",
                api_base="http://x",
                temperature=0.7,
                max_tokens=512,
                reasoning_effort="low",
                extra_headers={"x-test": "1"},
                system_instruction="sys",
            )
        kwargs = m.call_args.kwargs
        assert kwargs["api_key"] == "K"
        assert kwargs["api_base"] == "http://x"
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 512
        assert kwargs["reasoning_effort"] == "low"
        assert kwargs["extra_headers"] == {"x-test": "1"}
        assert kwargs["messages"][0] == {"role": "system", "content": "sys"}
        assert kwargs["messages"][-1] == {"role": "user", "content": "p"}

    def test_absent_optionals_are_omitted(self):
        with patch("litellm.completion", return_value=_fake_response()) as m:
            complete(model="openai/gpt-4o", prompt="p")
        kwargs = m.call_args.kwargs
        for absent in (
            "api_key",
            "api_base",
            "temperature",
            "max_tokens",
            "reasoning_effort",
            "extra_headers",
            "response_format",
        ):
            assert absent not in kwargs


class TestCompleteResult:
    """Extraction of the neutral LlmResult."""

    def test_text_and_usage(self):
        with patch(
            "litellm.completion",
            return_value=_fake_response("dialogue", prompt_tokens=42, completion_tokens=7),
        ):
            result = complete(model="openai/gpt-4o", prompt="p")
        assert isinstance(result, LlmResult)
        assert result.text == "dialogue"
        assert result.input_tokens == 42
        assert result.output_tokens == 7

    def test_grounding_extracted_on_gemini(self):
        hidden = {
            "vertex_ai_grounding_metadata": [
                {
                    "webSearchQueries": ["q1", "q2"],
                    "groundingChunks": [
                        {"web": {"uri": "https://a", "title": "A"}},
                        {"web": {"uri": "https://b"}},  # title falls back to uri
                    ],
                }
            ]
        }
        with patch("litellm.completion", return_value=_fake_response(hidden=hidden)):
            result = complete(
                model="gemini/gemini-2.5-flash", prompt="p", enable_google_search=True
            )
        assert isinstance(result.grounding, Grounding)
        assert result.grounding.queries == ["q1", "q2"]
        assert result.grounding.citations == [
            ("A", "https://a"),
            ("https://b", "https://b"),
        ]

    def test_no_grounding_on_non_gemini(self):
        hidden = {"vertex_ai_grounding_metadata": [{"webSearchQueries": ["q"]}]}
        with patch("litellm.completion", return_value=_fake_response(hidden=hidden)):
            result = complete(model="openai/gpt-4o", prompt="p")
        assert result.grounding is None

    def test_grounding_none_when_absent(self):
        with patch("litellm.completion", return_value=_fake_response(hidden={})):
            result = complete(model="gemini/gemini-2.5-flash", prompt="p")
        assert result.grounding is None
