"""
Tests for :mod:`tts_podcast.settings` — the provider-agnostic LLM/TTS resolvers.

Two layers:
* :func:`~tts_podcast.settings.resolve_llm_settings` — legacy ``gemini:``
  fallback, the new ``llm:`` section winning field-by-field, provider
  normalisation, and the ``service_tier`` back-compat shim.
* :func:`~tts_podcast.settings.resolve_tts_settings` — backend selection
  defaulting to ``"gemini"`` and passthrough of the ``tts.moss`` sub-config.
"""

from __future__ import annotations

from tts_podcast.settings import resolve_llm_settings, resolve_tts_settings


class TestResolveLlmSettingsLegacy:
    def test_legacy_only_config_resolves_gemini_defaults(self):
        cfg = {"gemini": {"api_key": "K", "text_model": "gemini-2.5-flash"}}
        settings = resolve_llm_settings(cfg)
        assert settings.provider == "gemini"
        assert settings.text_model == "gemini-2.5-flash"
        assert settings.api_key == "K"
        assert settings.research_model is None
        assert settings.extra_headers is None

    def test_research_model_from_gemini_research_block(self):
        cfg = {
            "gemini": {
                "api_key": "K",
                "text_model": "gemini-2.5-flash",
                "research": {"model": "gemini-2.5-pro"},
            }
        }
        settings = resolve_llm_settings(cfg)
        assert settings.research_model == "gemini-2.5-pro"

    def test_research_model_from_top_level_research_block(self):
        cfg = {
            "gemini": {"api_key": "K", "text_model": "gemini-2.5-flash"},
            "research": {"model": "gemini-2.5-pro"},
        }
        settings = resolve_llm_settings(cfg)
        assert settings.research_model == "gemini-2.5-pro"

    def test_llm_research_model_wins_over_both_legacy_sources(self):
        cfg = {
            "llm": {"research_model": "llm-research-model"},
            "gemini": {
                "api_key": "K",
                "text_model": "gemini-2.5-flash",
                "research": {"model": "gemini-research-model"},
            },
            "research": {"model": "top-level-research-model"},
        }
        settings = resolve_llm_settings(cfg)
        assert settings.research_model == "llm-research-model"


class TestResolveLlmSettingsNewSection:
    def test_new_llm_section_wins_over_legacy(self):
        cfg = {
            "llm": {
                "provider": "openai",
                "text_model": "gpt-4o",
                "api_key": "OK",
                "api_base": "http://x",
            },
            "gemini": {"api_key": "GK", "text_model": "gemini-2.5-flash"},
        }
        settings = resolve_llm_settings(cfg)
        assert settings.provider == "openai"
        assert settings.text_model == "gpt-4o"
        assert settings.api_key == "OK"
        assert settings.api_base == "http://x"

    def test_provider_is_lowercased_and_stripped(self):
        cfg = {"llm": {"provider": "OpenAI "}}
        settings = resolve_llm_settings(cfg)
        assert settings.provider == "openai"


class TestResolveLlmSettingsExtraHeaders:
    def test_legacy_service_tier_becomes_extra_header_for_gemini(self):
        cfg = {
            "gemini": {
                "api_key": "K",
                "text_model": "gemini-2.5-flash",
                "service_tier": "flex",
            }
        }
        settings = resolve_llm_settings(cfg)
        assert settings.extra_headers == {"x-goog-api-service-tier": "flex"}

    def test_explicit_extra_headers_pass_through_verbatim(self):
        cfg = {
            "llm": {"extra_headers": {"x-custom": "value"}},
            "gemini": {"api_key": "K", "text_model": "gemini-2.5-flash"},
        }
        settings = resolve_llm_settings(cfg)
        assert settings.extra_headers == {"x-custom": "value"}

    def test_explicit_service_tier_header_not_overwritten_by_legacy(self):
        cfg = {
            "llm": {
                "extra_headers": {"x-goog-api-service-tier": "standard"},
            },
            "gemini": {
                "api_key": "K",
                "text_model": "gemini-2.5-flash",
                "service_tier": "flex",
            },
        }
        settings = resolve_llm_settings(cfg)
        assert settings.extra_headers == {"x-goog-api-service-tier": "standard"}

    def test_legacy_service_tier_does_not_leak_for_non_gemini_provider(self):
        cfg = {
            "llm": {"provider": "openai"},
            "gemini": {
                "api_key": "K",
                "text_model": "gemini-2.5-flash",
                "service_tier": "flex",
            },
        }
        settings = resolve_llm_settings(cfg)
        assert settings.extra_headers is None


class TestResolveTtsSettings:
    def test_no_tts_section_defaults_to_gemini(self):
        settings = resolve_tts_settings({})
        assert settings.backend == "gemini"
        assert settings.moss == {}

    def test_moss_backend_with_config(self):
        cfg = {"tts": {"backend": "moss", "moss": {"api_base": "http://m"}}}
        settings = resolve_tts_settings(cfg)
        assert settings.backend == "moss"
        assert settings.moss == {"api_base": "http://m"}

    def test_backend_is_lowercased_and_stripped(self):
        cfg = {"tts": {"backend": " MOSS "}}
        settings = resolve_tts_settings(cfg)
        assert settings.backend == "moss"
