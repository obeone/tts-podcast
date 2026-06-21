# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Environment is managed by **uv** (Python 3.13+). Never `pip install` globally.

```bash
uv sync                                    # install/refresh deps
uv run tts-podcast run <URL> [<URL> ...]   # full pipeline (URLs)
uv run tts-podcast run -f doc.pdf          # local file input
uv run tts-podcast run -s "search topic"   # web-search query input
uv run tts-podcast run -L <URL>            # follow interesting links found inside inputs (--follow-depth N)
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
- **Link following (`--follow-links [--follow-depth N]`)**: opt-in stage in `link_follower.follow_links`, run in `cli.py` **before** research and link extraction (stage "2a-bis") so the kept pages feed BOTH research and the dialogue. Two-stage selection: (1) `link_extractor.is_followable_link` is a cheap pre-fetch URL heuristic (drops non-http schemes, anchors, asset extensions, ad/social/tracker hosts, auth/commerce paths; keeps `.pdf`/`.html` and all real-content categories); (2) one batched Gemini call (`_judge_sources`, structured JSON output, model from `gemini.follow.model` → `text_model` → `gemini-2.5-flash`) classifies each fetched page's *content* into `core` / `supporting` / `irrelevant`. The judge uses a **stable ARRAY response schema** (`[{url, label}, …]`, not URL-keyed properties — a dynamic per-call schema the API may reject/ignore, silently turning the judge into a no-op); the array is flattened back to `{url: label}`, omitted/unknown labels coerce to `supporting`. Kept pages (core+supporting) become full `Source` objects with their verdict stored on the `Source.relevance` field. That verdict is **surfaced downstream** — dialogue prompt (`llm_summarizer._build_prompt`), research prompt (`research._format_articles`), and the report (`report_generator._render_sources`) all annotate followed pages via the shared `link_extractor.relevance_label` helper. Every annotation is **gated on `relevance is not None`**, so seed/primary inputs (`relevance=None`) render byte-identically — this keeps the dialogue-prompt snapshot (`test_no_flags_byte_identical`) green with no fixture regeneration. Traversal is a level-by-level BFS up to `depth` (default 1), capped at `follow.max_links_per_level` (default 5) links per level AND at `follow.max_links_total` (default 20) links total across all levels (a global cost cap; per-level candidate lists are truncated to the remaining budget, logging `global follow budget N reached`); a `seen` set of normalized URLs (lowercased host, fragment + trailing slash stripped, seeded with the seed URLs) gives cycle/dedup safety. The whole stage is best-effort — every failure logs and degrades, and the judge **fails open** (defaults to `supporting`) so content is kept rather than silently dropped. `--follow-depth` without `--follow-links` is a no-op (warns). **Link capture at scrape time** (`Source.links`): because trafilatura's plain-text extraction drops all hyperlinks, `_extract_from_html` (web scraper) and `_read_html` (local loader) run a second trafilatura pass with `include_links=True, output_format="markdown"` to capture article-body hrefs (nav/footer links are excluded by trafilatura's body detector), then parse markdown `[text](url)` targets and bare URLs, deduplicate preserving order, and store the result in `Source.links`. `_gather_candidates` reads `Source.links` ∪ `extract_links_from_text(full_text)` so HTML sources work (their hrefs live in `Source.links`) AND plain-text/markdown/PDF sources with bare URLs in `full_text` keep working; `full_text` is never polluted with markdown syntax.
- **Optional CloakBrowser fallback**: when `scraping.cloak_fallback: true`, a trafilatura scrape that yields no content (download `None` or empty extraction — the typical access-error signature: 403/429, Cloudflare, JS-only pages) is retried through `cloak_fetcher.fetch_html`, which drives the optional `cloakbrowser` stealth Chromium and feeds its rendered HTML back into the **same** `_extract_from_html` path. The dependency is an optional extra (`uv sync --extra cloak`); `cloak_fetcher` degrades to `None`/no-op when the package is absent or errors, so the flag is safe to leave on without it installed. Default is off. The fallback is never reached on a successful trafilatura scrape.
- **Output stem**: `_build_output_stem` = `<host>-<6-char-sha1-of-urls>-<ISO-date>`, stable for a given URL set within a day.
- **Token tracking is opt-in per call site**: every Gemini call must thread `token_tracker` through and call `tracker.record_usage(model, response.usage_metadata)`. Missing wire-ups silently undercount cost.
- **Style & angle injection points**: `--preset` / `--style` / `--angle` / `--speaker[12]-style` write into `gemini.style.*` and `gemini.speaker[12].style_overlay` (never into `personality`). `llm_summarizer._build_prompt` renders them inside the dialogue prompt: per-speaker overlays in a dedicated `Episode-specific adjustments:` block between `Host personalities:` and `Instructions:`; preset + free style as a `Stylistic guidance:` sub-section inside `Instructions:`; angle as an `- Episode angle:` bullet. The angle is also injected into `research._ROUND_1_PROMPT` (and nowhere else — round N≥2 only sees it indirectly via `previous_notes`, so gap-analysis stays neutral).
- **Voice duo resolution**: `duos.py` holds `BUILTIN_DUOS` (warm/contrast/explorer/journalist/debate), available out of the box. `cli.py::run` resolves the active duo *before* reading any speaker field, with precedence `--duo` > `gemini.default_duo` > legacy `gemini.speakerN` blocks > built-in `contrast`, then writes the result into `gemini_cfg["speaker1"/"speaker2"]`. This is the single injection point — every downstream consumer (TTS preamble, dialogue prompt, `--speakerN-style` overlays) reads `gemini.speakerN` unchanged, and a config defining only legacy `speaker1`/`speaker2` keeps working untouched. A user `gemini.duos` mapping is merged over the built-ins (same slug overrides; new slugs extend). `tts-podcast duos` lists them (reads the *raw* config, so it needs no API key).
- **Hard invariant — TTS preamble untouched**: `tts_generator._build_tts_prompt` reads `gemini_cfg["speakerN"]["personality"]` verbatim. The new `style_overlay` key is for the dialogue prompt only and MUST NEVER be read by the TTS path. `personality` is never mutated, in memory or on disk, by any code path. Regression test: `tests/test_tts_generator.py::test_tts_preamble_unaffected_by_speaker_overlay`.
- **Snapshot fixture for the dialogue prompt**: `tests/fixtures/dialogue_prompt_no_overlay.txt` is the byte-identical baseline used by `test_no_flags_byte_identical`. When `_SYSTEM_PROMPT_TEMPLATE` is intentionally edited (typo, wording tweak): (1) edit the template, (2) `uv run python -m tests.fixtures.regen_dialogue_prompt`, (3) review the diff, (4) commit the fixture alongside the template change. The `tests/conftest.py` `collect_ignore_glob = ["fixtures/*"]` line guarantees pytest never auto-collects anything under `tests/fixtures/`.

### Configuration loader

`config.load_config` resolves any YAML key ending in `_env` by looking up the named environment variable, then drops the `_env` suffix in the returned dict. So `api_key_env: GEMINI_API_KEY` in YAML becomes `cfg["gemini"]["api_key"] = os.environ["GEMINI_API_KEY"]`. Missing env vars raise `ConfigError` at load time (fail-fast).

CLI flags override config: `-R/--research`, `-d/--duration`, `-o/--output-dir`. The duration override mutates `gemini_cfg["dialogue"]["target_duration_minutes"]` in memory; min/max default to 70 % / 150 % of target unless set explicitly in config.

### Module map

| Module | Role |
|---|---|
| `cli.py` | Click entry point, pipeline orchestration, `config init/show` wizard, `duos` command |
| `config.py` | YAML loader + `_env` resolution |
| `duos.py` | Named voice duos: `BUILTIN_DUOS` registry + `resolve_duo` / `describe_duos` |
| `models.py` | `Source` dataclass with `kind` field (`"url"` / `"file"` / `"search"`) |
| `web_scraper.py` | trafilatura-based scraping, parallel (≤10 workers), optional CloakBrowser fallback |
| `cloak_fetcher.py` | Optional `cloakbrowser` stealth-Chromium fetch (graceful no-op when absent) |
| `local_loader.py` | Local file reader (txt, md, html via trafilatura, pdf via pypdf) |
| `research.py` | Iterative Gemini + Google Search grounding rounds |
| `llm_summarizer.py` | Dialogue generation + byte-bounded chunking |
| `tts_generator.py` | Gemini multi-speaker TTS, parallel (≤5 workers) |
| `audio_exporter.py` | PCM → mp3/wav via pydub + ffmpeg |
| `report_generator.py` | Markdown report folder rendering |
| `link_extractor.py` | URL categorisation (repo / model / paper / source / other) + `is_followable_link` heuristic |
| `link_follower.py` | Optional two-stage BFS link following (heuristic pre-filter + LLM content-relevance judge) |
| `token_tracker.py` | Token accounting + tier-aware cost estimation |
| `retry.py` | `@gemini_retry` decorator (5xx only) |
| `user_agent.py` | Shared browser-UA string |

## Conventions

- Full NumPy-style docstrings on every public class and function (existing code is the reference).
- `coloredlogs` is configured by the CLI; modules just call `logging.getLogger(__name__)`.
- Use `from __future__ import annotations` in every module.
- Heavy imports gated behind `TYPE_CHECKING` to keep CLI startup fast (see `web_scraper.py`, `tts_generator.py`).
- Tests live in `tests/`, mirror module names (`test_<module>.py`), and mock the Gemini SDK rather than hitting the network.
