# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Environment is managed by **uv** (Python 3.13+). Never `pip install` globally.

```bash
uv sync                                    # install/refresh deps
uv run tts-podcast run <URL> [<URL> ...]   # full pipeline (URLs)
uv run tts-podcast run -f doc.pdf          # local file input
uv run tts-podcast run -s "search topic"   # web-search query input
uv run tts-podcast config init             # write config to $XDG_CONFIG_HOME/tts-podcast/config.yaml
uv run pytest tests/ -q                    # tests (quiet)
uv run pytest tests/test_research.py::test_name -v   # single test
uv run ruff check src/ tests/              # lint
```

`ffmpeg` must be in `PATH` for audio export (pydub uses it). `--dry-run` / `--no-audio` skip the preflight check.

The text-generation API key is read at runtime from the env var named by `llm.api_key_env` (falling back to legacy `gemini.api_key_env`, default `GEMINI_API_KEY`). Loaded from `.env` automatically via `python-dotenv` at CLI startup. Gemini TTS always resolves its own key from `gemini.api_key_env` independently, since a run can generate its dialogue on one provider while still rendering audio through Gemini TTS.

## Architecture

The pipeline in `cli.py::run` is strictly linear; each stage produces dataclasses defined in or near its own module and the next stage consumes them. There is no mutable shared state besides `TokenTracker`. The text stage (dialogue, research, duo casting) is provider-agnostic via LiteLLM (`llm_client.py`); the TTS stage is pluggable via a `TtsBackend` protocol (`tts/`), defaulting to Gemini.

Three input kinds feed the same pipeline via the `Source.kind` field (`"url"` / `"file"` / `"search"`):
- `"url"` — fetched by `web_scraper.scrape_urls`; default when no `-f`/`-s` flag is used.
- `"file"` — read locally by `local_loader.load_local_files` (txt, md, html, pdf); no network call.
- `"search"` — a natural-language query materialised as a synthetic `Source`; research stage investigates it via Google Search grounding. Research is auto-bumped to 1 round when only search inputs are present.

```
Inputs (URLs / -f files / -s queries) ── cli.py ──► list[Source]  (kind="url"|"file"|"search")
         │
         ├─ URL  ─── web_scraper.scrape_urls
         ├─ file ─── local_loader.load_local_files
         └─ search ─ _make_search_source (synthetic, scraped_ok=True)
                                       │
                          (optional) research.conduct_research
                                       │
                                       ▼
                              ResearchReport.combined_notes (str)
                                       │
        llm_summarizer.generate_dialogue (Source + notes ──► llm_client.complete, provider from `llm:` config)
                                       │
                                       ▼
                              list[DialogueChunk]  (~3000 UTF-8 bytes each, split at speaker turns)
                                       │
        TtsBackend.synthesize (gemini|moss, resolved by `tts.backend`; parallel workers)
                                       │
                                       ▼
                              list[bytes]  (raw PCM, shape given by backend's AudioFormat)
                                       │
        audio_exporter.export_audio  ──►  mp3 / wav  (pydub → ffmpeg, format-aware)
                                       │
        report_generator.generate_report  ──► tts_<stem>/{overview,sources,script,research,summary}.md
```

### Key invariants & non-obvious behaviour

- **Chunk byte budget**: `_MAX_CHUNK_BYTES = 3000` in `llm_summarizer.py`. The TTS prompt prepends a personality/scene preamble of ~600–800 bytes; total must stay below Gemini TTS's ~4000-byte text limit. Splits **only at speaker-turn boundaries** (lines starting with `<SpeakerName>:`).
- **Audio cues vs. audio tags**: `llm_summarizer._audio_tags_enabled` auto-detects from `tts_model` (Gemini 3.x → English bracketed tags `[curiosity]`; older → parenthetical cues in target language). Override via `gemini.tts_style.audio_tags: on|off|auto`.
- **Research is iterative**: round 1 looks for complementary angles; round N≥2 receives all prior round notes via `_ROUND_N_PROMPT` and is told to drill into gaps. Each round is a separate `llm_client.complete` call with `enable_google_search=True`, which attaches the native `googleSearch` tool **only when the resolved model is a Gemini route** (`llm_client.is_gemini_model`) — billed with search overhead on Gemini, run without grounding (degraded, not broken) on any other provider.
- **LLM text layer is provider-agnostic via LiteLLM**: `llm_client.complete(...)` wraps `litellm.completion` and is the only entry point the three text call sites (`llm_summarizer.generate_dialogue`, `research.conduct_research`, `duo_generator.generate_duo`) use — none of them import `google-genai` anymore. It always returns a neutral `LlmResult(text, input_tokens, output_tokens, grounding)`; `build_model_string(provider, model)` builds the `"provider/model"` LiteLLM string (`gemini/`, `openai/`, `anthropic/`, `ollama_chat/`). `google-genai` is now used only by the Gemini TTS backend.
- **`llm:` config section, legacy fallback**: `settings.resolve_llm_settings(cfg)` reads the optional `llm:` block (`provider`, `text_model`, `research_model`, `api_key_env`, `api_base`, `temperature`, `extra_headers`) field-by-field, falling back to the legacy `gemini:` block for any field it leaves unset — a config with only `gemini:` keeps working unchanged, implicitly on `provider: gemini`. Grounding read-back (citations + issued queries) is Gemini-specific (LiteLLM's `_hidden_params["vertex_ai_grounding_metadata"]`) and only attempted on a Gemini route.
- **`extra_headers` replaces `service_tier`**: the old Gemini-only `gemini.service_tier` is generalized to provider-agnostic `llm.extra_headers` (arbitrary passthrough HTTP headers). `resolve_llm_settings` still honours the legacy `gemini.service_tier` key, folding it into `extra_headers["x-goog-api-service-tier"]` when the provider is Gemini and no explicit header already set it. **TTS calls never use a service tier** (Gemini TTS does not support it). Pricing supports both flat and tier-aware formats; `TokenTracker._resolve_pricing` picks the right rate.
- **`thinking_level`/`thinking_budget` map onto `reasoning_effort`**: `llm_summarizer._reasoning_effort(dialogue_cfg)` replaces the removed `_build_thinking_config` and maps `gemini.dialogue.thinking_level` 1:1 onto LiteLLM's provider-agnostic `reasoning_effort` (`"minimal"|"low"|"medium"|"high"`); `thinking_budget: 0` maps to `"minimal"`, any other value to `"low"`.
- **Retry policy**: `retry.gemini_retry` retries `google.genai.errors.ServerError` (5xx) for the direct Gemini TTS SDK calls; `retry.llm_retry` retries LiteLLM's transient exception types (`InternalServerError`, `ServiceUnavailableError`, `APIConnectionError`, `Timeout`) for every `llm_client.complete` call. Both share the same back-off schedule (5 attempts, 2 s → 60 s). Client errors (4xx) are not retried by either.
- **Scrape failures don't abort**: `scrape_urls` returns `Source(scraped_ok=False)` for failures; the run continues with whatever scraped successfully, and aborts only if **all** URLs failed.
- **Optional CloakBrowser fallback**: when `scraping.cloak_fallback: true`, a trafilatura scrape that yields no content (download `None` or empty extraction — the typical access-error signature: 403/429, Cloudflare, JS-only pages) is retried through `cloak_fetcher.fetch_html`, which drives the optional `cloakbrowser` stealth Chromium and feeds its rendered HTML back into the **same** `_extract_from_html` path. The dependency is an optional extra (`uv sync --extra cloak`); `cloak_fetcher` degrades to `None`/no-op when the package is absent or errors, so the flag is safe to leave on without it installed. Default is off. The fallback is never reached on a successful trafilatura scrape.
- **Output stem**: `_build_output_stem` = `<host>-<6-char-sha1-of-urls>-<ISO-date>`, stable for a given URL set within a day.
- **Token tracking is opt-in per call site**: every call must thread `token_tracker` through. The Gemini TTS SDK path calls `tracker.record_usage(model, response.usage_metadata)`; the LiteLLM text call sites call `tracker.record(bare_model_name, input_tokens, output_tokens)` on the `LlmResult` returned by `llm_client.complete`. Pricing keys stay the **bare** model name (e.g. `gemini-2.5-flash`, never `gemini/gemini-2.5-flash`) in both cases. Missing wire-ups silently undercount cost.
- **Pluggable TTS backend**: `tts.resolve_tts_backend(cfg, gemini_cfg)` picks a `TtsBackend` (protocol in `tts/base.py`, alongside the `AudioFormat` dataclass it advertises) from `tts.backend` (`gemini` default, or `moss`), so legacy configs without a `tts:` section render through Gemini unchanged. `tts/gemini_backend.py::GeminiTtsBackend` is a thin wrapper over the unchanged `tts_generator.generate_audio_chunks` (24 kHz / mono / 16-bit LE) — the "personality read verbatim, never mutated" invariant below is untouched by this layer. `tts/moss_backend.py::MossTtsBackend` is an HTTP client to a self-hosted MOSS-TTSD server (OpenMOSS 8B spoken-dialogue model) served by the project's own `OpenMOSS/sglang` fork. That fork exposes SGLang's native API, not an OpenAI-compatible one, so the client POSTs to `{api_base}/generate` (body: `text`, optional `audio_data` list of reference clips, optional nested `sampling_params`, plus `extra_body` passthrough) and gets back `{"text": "<base64 WAV>", "meta_info": {...}}`, which it base64-decodes and parses with the stdlib `wave` module, reading the real sample rate/channels/width off the WAV header rather than assuming them. It rewrites each chunk's `<name>:` turns into MOSS' native `[S1]`/`[S2]` inline tags, strips Gemini-style `(cue)`/`[tag]` delivery markers by default (they'd collide with `[Sn]`), and does zero-shot voice cloning from an optional per-speaker `ref_audio`/`ref_text` (the reference transcript has no structured field, so it is prepended into `text` behind its `[Sn]` tag, controlled by `ref_prefix_mode`). `audio_exporter.py` (`export_audio` / `encode_audio` / `_combine_pcm`) takes the backend's `AudioFormat` instead of hard-coded Gemini constants, defaulting to the old 24 kHz/mono/16-bit shape.
- **Style & angle injection points**: `--preset` / `--style` / `--angle` / `--speaker[12]-style` write into `gemini.style.*` and `gemini.speaker[12].style_overlay` (never into `personality`). `llm_summarizer._build_prompt` renders them inside the dialogue prompt: per-speaker overlays in a dedicated `Episode-specific adjustments:` block between `Host personalities:` and `Instructions:`; preset + free style as a `Stylistic guidance:` sub-section inside `Instructions:`; angle as an `- Episode angle:` bullet. The angle is also injected into `research._ROUND_1_PROMPT` (and nowhere else — round N≥2 only sees it indirectly via `previous_notes`, so gap-analysis stays neutral).
- **Voice duo resolution**: `duos.py` holds `BUILTIN_DUOS` (warm/contrast/explorer/journalist/debate), available out of the box. `cli.py::run` resolves the active duo *before* reading any speaker field, with precedence `--duo` > `gemini.default_duo` > legacy `gemini.speakerN` blocks > built-in `contrast`, then writes the result into `gemini_cfg["speaker1"/"speaker2"]`. This is the single injection point — every downstream consumer (TTS preamble, dialogue prompt, `--speakerN-style` overlays) reads `gemini.speakerN` unchanged, and a config defining only legacy `speaker1`/`speaker2` keeps working untouched. A user `gemini.duos` mapping is merged over the built-ins (same slug overrides; new slugs extend). `tts-podcast duos` lists them (reads the *raw* config, so it needs no API key).
- **`--duo auto` — content-aware duo generation**: when `--duo` is the literal string `"auto"`, duo resolution is *deferred*. After scraping and research, `duo_generator.generate_duo` calls `llm_client.complete` with structured output (voice names validated against a `GEMINI_VOICES` Literal enum) to generate names, voices, and personalities suited to the content. The result is injected at the same single injection point in `gemini_cfg["speaker1"/"speaker2"]`; speaker names are also recalculated at that point. Hard invariants: (1) `personality` strings from the generated duo reach downstream consumers verbatim — never mutated; (2) the non-auto code path is behaviorally unchanged; (3) `generate_duo` is never called without `--duo auto`.
- **Hard invariant — TTS preamble untouched**: `tts_generator._build_tts_prompt` reads `gemini_cfg["speakerN"]["personality"]` verbatim. The new `style_overlay` key is for the dialogue prompt only and MUST NEVER be read by the TTS path. `personality` is never mutated, in memory or on disk, by any code path. Regression test: `tests/test_tts_generator.py::test_tts_preamble_unaffected_by_speaker_overlay`.
- **Snapshot fixture for the dialogue prompt**: `tests/fixtures/dialogue_prompt_no_overlay.txt` is the byte-identical baseline used by `test_no_flags_byte_identical`. When `_SYSTEM_PROMPT_TEMPLATE` is intentionally edited (typo, wording tweak): (1) edit the template, (2) `uv run python -m tests.fixtures.regen_dialogue_prompt`, (3) review the diff, (4) commit the fixture alongside the template change. The `tests/conftest.py` `collect_ignore_glob = ["fixtures/*"]` line guarantees pytest never auto-collects anything under `tests/fixtures/`.

### Configuration loader

`config.load_config` resolves any YAML key ending in `_env` by looking up the named environment variable, then drops the `_env` suffix in the returned dict. So `api_key_env: GEMINI_API_KEY` in YAML becomes `cfg["gemini"]["api_key"] = os.environ["GEMINI_API_KEY"]`. Missing env vars raise `ConfigError` at load time (fail-fast).

`settings.resolve_llm_settings(cfg)` and `settings.resolve_tts_settings(cfg)` turn the loaded config into typed `LlmSettings` / `TtsSettings`. Both new sections are optional: `llm:` (provider selection for text calls, see the LLM invariants above) falls back field-by-field to the legacy `gemini:` block, and `tts:` (backend selection) defaults to `backend: gemini` — so a config carrying only the legacy `gemini:` block keeps behaving exactly as before.

CLI flags override config: `-R/--research`, `-d/--duration`, `-o/--output-dir`. The duration override mutates `gemini_cfg["dialogue"]["target_duration_minutes"]` in memory; min/max default to 70 % / 150 % of target unless set explicitly in config.

### Module map

| Module | Role |
|---|---|
| `cli.py` | Click entry point, pipeline orchestration, `config init/show` wizard, `duos` command |
| `config.py` | YAML loader + `_env` resolution |
| `settings.py` | `LlmSettings`/`resolve_llm_settings` + `TtsSettings`/`resolve_tts_settings` — resolve the `llm:`/`tts:` config sections with legacy `gemini:` fallback |
| `llm_client.py` | Provider-agnostic text layer: `complete(...)` wraps `litellm.completion`, returns neutral `LlmResult`; `build_model_string` / `is_gemini_model` helpers. The only text call sites; `google-genai` is no longer used here |
| `duos.py` | Named voice duos: `BUILTIN_DUOS` registry + `resolve_duo` / `describe_duos`; `GEMINI_VOICES` set (single source of truth for all 30 prebuilt voices) |
| `duo_generator.py` | Structured-output call (via `llm_client.complete`) that auto-generates a content-aware duo; voice names validated against `GEMINI_VOICES`; called only when `--duo auto` is passed |
| `models.py` | `Source` dataclass with `kind` field (`"url"` / `"file"` / `"search"`) |
| `web_scraper.py` | trafilatura-based scraping, parallel (≤10 workers), optional CloakBrowser fallback |
| `cloak_fetcher.py` | Optional `cloakbrowser` stealth-Chromium fetch (graceful no-op when absent) |
| `local_loader.py` | Local file reader (txt, md, html via trafilatura, pdf via pypdf) |
| `research.py` | Iterative research rounds via `llm_client.complete`; Google Search grounding only on a Gemini route |
| `llm_summarizer.py` | Dialogue generation (via `llm_client.complete`) + byte-bounded chunking |
| `tts/base.py` | `AudioFormat` dataclass + `TtsBackend` protocol — the neutral speech-backend contract |
| `tts/__init__.py` | `resolve_tts_backend(cfg, gemini_cfg)` factory (`gemini` \| `moss`) |
| `tts/gemini_backend.py` | `GeminiTtsBackend` — thin `TtsBackend` wrapper over `tts_generator.generate_audio_chunks` |
| `tts/moss_backend.py` | `MossTtsBackend` — HTTP client to a self-hosted MOSS-TTSD server |
| `tts_generator.py` | Gemini multi-speaker TTS implementation (parallel, ≤5 workers); wrapped by `GeminiTtsBackend`, otherwise unchanged |
| `audio_exporter.py` | PCM → mp3/wav via pydub + ffmpeg, format-aware (`AudioFormat` from the active backend) |
| `report_generator.py` | Markdown report folder rendering |
| `link_extractor.py` | URL categorisation (repo / model / paper / source / other) |
| `token_tracker.py` | Token accounting + tier-aware cost estimation |
| `retry.py` | `gemini_retry` (direct Gemini TTS SDK, 5xx only) + `llm_retry` (LiteLLM text calls, transient errors) decorators |
| `user_agent.py` | Shared browser-UA string |

## Conventions

- **Version bump before every feature merge**: bump `version` in `pyproject.toml` (single source of truth, SemVer) before merging any feature (feature branch or PR). No feature lands without a version increment.
- Full NumPy-style docstrings on every public class and function (existing code is the reference).
- `coloredlogs` is configured by the CLI; modules just call `logging.getLogger(__name__)`.
- Use `from __future__ import annotations` in every module.
- Heavy imports gated behind `TYPE_CHECKING` to keep CLI startup fast (see `web_scraper.py`, `tts_generator.py`).
- Tests live in `tests/`, mirror module names (`test_<module>.py`), and mock `llm_client.complete` / the Gemini SDK rather than hitting the network.
