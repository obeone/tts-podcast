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

The Gemini API key is read at runtime from the env var named by `gemini.api_key_env` (default `GEMINI_API_KEY`). Loaded from `.env` automatically via `python-dotenv` at CLI startup.

## Architecture

The pipeline in `cli.py::run` is strictly linear; each stage produces dataclasses defined in or near its own module and the next stage consumes them. There is no mutable shared state besides `TokenTracker`.

Three input kinds feed the same pipeline via the `Source.kind` field (`"url"` / `"file"` / `"search"`):
- `"url"` — fetched by `web_scraper.scrape_urls`; default when no `-f`/`-s` flag is used.
- `"file"` — read locally by `local_loader.load_local_file` (txt, md, html, pdf); no network call.
- `"search"` — a natural-language query materialised as a synthetic `Source`; research stage investigates it via Google Search grounding. Research is auto-bumped to 1 round when only search inputs are present.

```
Inputs (URLs / -f files / -s queries) ── cli.py ──► list[Source]  (kind="url"|"file"|"search")
         │
         ├─ URL  ─── web_scraper.scrape_urls
         ├─ file ─── local_loader.load_local_files
         └─ search ─ _make_search_source (synthetic, scraped_ok=True)
                                       │
URLs ── web_scraper.scrape_urls ──► list[Source]
                                       │
                          (optional) research.conduct_research
                                       │
                                       ▼
                              ResearchReport.combined_notes (str)
                                       │
        llm_summarizer.generate_dialogue (Source + notes ──► Gemini text model)
                                       │
                                       ▼
                              list[DialogueChunk]  (~3000 UTF-8 bytes each, split at speaker turns)
                                       │
        tts_generator.generate_audio_chunks (parallel, ThreadPool ≤5)
                                       │
                                       ▼
                              list[bytes]  (raw PCM 24 kHz / mono / 16-bit LE)
                                       │
        audio_exporter.export_audio  ──►  mp3 / wav  (pydub → ffmpeg)
                                       │
        report_generator.generate_report  ──► tts_<stem>/{overview,sources,script,research,summary}.md
```

### Key invariants & non-obvious behaviour

- **Chunk byte budget**: `_MAX_CHUNK_BYTES = 3000` in `llm_summarizer.py`. The TTS prompt prepends a personality/scene preamble of ~600–800 bytes; total must stay below Gemini TTS's ~4000-byte text limit. Splits **only at speaker-turn boundaries** (lines starting with `<SpeakerName>:`).
- **Audio cues vs. audio tags**: `llm_summarizer._audio_tags_enabled` auto-detects from `tts_model` (Gemini 3.x → English bracketed tags `[curiosity]`; older → parenthetical cues in target language). Override via `gemini.tts_style.audio_tags: on|off|auto`.
- **Research is iterative**: round 1 looks for complementary angles; round N≥2 receives all prior round notes via `_ROUND_N_PROMPT` and is told to drill into gaps. Each round is a separate Gemini call with the `google_search` grounding tool — billed with search overhead.
- **Service tiers** (`gemini.service_tier`): when set, passed as `x-goog-api-service-tier` HTTP header on text/research calls. **TTS calls never use a service tier** (Gemini TTS does not support it). Pricing supports both flat and tier-aware formats; `TokenTracker._resolve_pricing` picks the right rate.
- **Retry policy**: `retry.gemini_retry` only retries `google.genai.errors.ServerError` (5xx) — exponential back-off, 5 attempts, 2 s → 60 s. Client errors (4xx) are not retried.
- **Scrape failures don't abort**: `scrape_urls` returns `Source(scraped_ok=False)` for failures; the run continues with whatever scraped successfully, and aborts only if **all** URLs failed.
- **Output stem**: `_build_output_stem` = `<host>-<6-char-sha1-of-urls>-<ISO-date>`, stable for a given URL set within a day.
- **Token tracking is opt-in per call site**: every Gemini call must thread `token_tracker` through and call `tracker.record_usage(model, response.usage_metadata)`. Missing wire-ups silently undercount cost.

### Configuration loader

`config.load_config` resolves any YAML key ending in `_env` by looking up the named environment variable, then drops the `_env` suffix in the returned dict. So `api_key_env: GEMINI_API_KEY` in YAML becomes `cfg["gemini"]["api_key"] = os.environ["GEMINI_API_KEY"]`. Missing env vars raise `ConfigError` at load time (fail-fast).

CLI flags override config: `-R/--research`, `-d/--duration`, `-o/--output-dir`. The duration override mutates `gemini_cfg["dialogue"]["target_duration_minutes"]` in memory; min/max default to 70 % / 150 % of target unless set explicitly in config.

### Module map

| Module | Role |
|---|---|
| `cli.py` | Click entry point, pipeline orchestration, `config init/show` wizard |
| `config.py` | YAML loader + `_env` resolution |
| `models.py` | `Source` dataclass with `kind` field (`"url"` / `"file"` / `"search"`) |
| `web_scraper.py` | trafilatura-based scraping, parallel (≤10 workers) |
| `local_loader.py` | Local file reader (txt, md, html via trafilatura, pdf via pypdf) |
| `research.py` | Iterative Gemini + Google Search grounding rounds |
| `llm_summarizer.py` | Dialogue generation + byte-bounded chunking |
| `tts_generator.py` | Gemini multi-speaker TTS, parallel (≤5 workers) |
| `audio_exporter.py` | PCM → mp3/wav via pydub + ffmpeg |
| `report_generator.py` | Markdown report folder rendering |
| `link_extractor.py` | URL categorisation (repo / model / paper / source / other) |
| `token_tracker.py` | Token accounting + tier-aware cost estimation |
| `retry.py` | `@gemini_retry` decorator (5xx only) |
| `user_agent.py` | Shared browser-UA string |

## Conventions

- Full NumPy-style docstrings on every public class and function (existing code is the reference).
- `coloredlogs` is configured by the CLI; modules just call `logging.getLogger(__name__)`.
- Use `from __future__ import annotations` in every module.
- Heavy imports gated behind `TYPE_CHECKING` to keep CLI startup fast (see `web_scraper.py`, `tts_generator.py`).
- Tests live in `tests/`, mirror module names (`test_<module>.py`), and mock the Gemini SDK rather than hitting the network.
