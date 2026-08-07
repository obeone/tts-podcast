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
                              list[DialogueChunk]  (byte budget resolved per run from the rendered
                                                    TTS preamble, 1500-3000, split at speaker turns)
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

- **Chunk byte budget**: resolved per run from the active config, not pinned to a static constant. `llm_summarizer._resolve_chunk_budget` renders the TTS preamble for an empty chunk through `tts_generator._build_tts_prompt`, measures it in UTF-8 bytes, and returns `_TTS_TEXT_LIMIT - _TTS_PREAMBLE_HEADROOM - preamble` clamped to `[_MIN_CHUNK_BYTES, _MAX_CHUNK_BYTES]`, i.e. `[1500, 3000]`. **The only byte figure worth remembering is derived, not measured**: a preamble of `4000 - 200 - 3000 = 800` bytes or less resolves to the full `_MAX_CHUNK_BYTES`, so any config whose preamble stays under that chunks byte-identically to the pre-`voice_direction` behaviour (same request count, same audio splice points); above 800 the chunk shrinks byte for byte. Per-config byte counts are deliberately **not** transcribed here or in the README: they go stale the moment the preamble wording changes, and nothing enforces them. What is enforced is the property, by `tests/test_tts_generator.py::TestPreambleByteBudget`: for any config, its own resolved budget plus its own rendered preamble stays inside the cap, checked over every built-in duo in two languages, a legacy config, synthetic envelopes and a scene-length sweep. Generate a duo-by-duo table from `BUILTIN_DUOS` if one is ever needed; do not write one down. The `_TTS_TEXT_LIMIT = 4000` figure is inherited from the comment that predates this code (Gemini TTS's approximate request text cap), not something measured in this repo, which is why `_TTS_PREAMBLE_HEADROOM = 200` sits under it instead of budgeting right up to the edge; the other reason is that the API's counting unit is unknown (characters or tokens, not necessarily UTF-8 bytes). That headroom does **not** cover an oversized single speaker turn: `_split_dialogue_into_chunks` never splits mid-turn, so a turn larger than `max_bytes` is emitted alone and over budget, and logs a warning naming it. Its docstring says so; don't "fix" it back to promising every chunk fits. Only a genuinely oversized preamble reaches `_MIN_CHUNK_BYTES`; when it does, `_resolve_chunk_budget` warns with the measured preamble size and names the three fields to shorten (`gemini.tts_style.scene`, the speakers' `voice_direction`, their `personality`). Without that warning the symptom is a mid-episode 4xx from the TTS API, which `retry.gemini_retry` deliberately does not retry. **Where the budget is resolved matters**: the computation reads only `speaker1`/`speaker2`/`language`/`tts_style` and never the generated text, so `cli.py::run` resolves it right after the duo `tts_style` fill (before research and before the dialogue call) and passes it as `generate_dialogue(..., max_bytes=…)`. That is what puts the over-budget warning *before* the billing rather than next to the request it condemns. `max_bytes` defaults to `None`, in which case `generate_dialogue` resolves it itself, so direct library callers keep the measured behaviour. Two implementation details worth preserving: `_build_tts_prompt` is imported **inside** `_resolve_chunk_budget` (`tts_generator` imports this module only under `TYPE_CHECKING`, so keeping the import local prevents that edge from ever becoming a real cycle, and library callers that import `llm_summarizer` alone do not pull in the TTS deps; it buys nothing for CLI startup, since `cli.py` imports `tts_generator` at module scope anyway), and the measurement shallow-copies the config to fill a placeholder `name` on any missing `speakerN` block, because `generate_dialogue` treats those blocks as optional (names arrive as arguments) while `_build_tts_prompt` indexes them directly. Growing the preamble envelope no longer requires editing a constant: it just shrinks the resolved budget for the configs that actually grow it. Splits **only at speaker-turn boundaries** (lines starting with `<SpeakerName>:`).
- **Audio cues vs. audio tags**: `llm_summarizer._audio_tags_enabled` auto-detects from `tts_model` (Gemini 3.x → English bracketed tags `[curiosity]`; older → parenthetical cues in target language). Override via `gemini.tts_style.audio_tags: on|off|auto`.
- **Research is iterative**: round 1 looks for complementary angles; round N≥2 receives all prior round notes via `_ROUND_N_PROMPT` and is told to drill into gaps. Each round is a separate Gemini call with the `google_search` grounding tool — billed with search overhead.
- **Service tiers** (`gemini.service_tier`): when set, passed as `x-goog-api-service-tier` HTTP header on text/research calls. **TTS calls never use a service tier** (Gemini TTS does not support it). Pricing supports both flat and tier-aware formats; `TokenTracker._resolve_pricing` picks the right rate.
- **Retry policy**: `retry.gemini_retry` only retries `google.genai.errors.ServerError` (5xx) — exponential back-off, 5 attempts, 2 s → 60 s. Client errors (4xx) are not retried.
- **Scrape failures don't abort**: `scrape_urls` returns `Source(scraped_ok=False)` for failures; the run continues with whatever scraped successfully, and aborts only if **all** URLs failed.
- **Optional CloakBrowser fallback**: when `scraping.cloak_fallback: true`, a trafilatura scrape that yields no content (download `None` or empty extraction — the typical access-error signature: 403/429, Cloudflare, JS-only pages) is retried through `cloak_fetcher.fetch_html`, which drives the optional `cloakbrowser` stealth Chromium and feeds its rendered HTML back into the **same** `_extract_from_html` path. The dependency is an optional extra (`uv sync --extra cloak`); `cloak_fetcher` degrades to `None`/no-op when the package is absent or errors, so the flag is safe to leave on without it installed. Default is off. The fallback is never reached on a successful trafilatura scrape.
- **Output stem**: `_build_output_stem` = `<host>-<6-char-sha1-of-urls>-<ISO-date>`, stable for a given URL set within a day.
- **Token tracking is opt-in per call site**: every Gemini call must thread `token_tracker` through and call `tracker.record_usage(model, response.usage_metadata)`. Missing wire-ups silently undercount cost.
- **Style & angle injection points**: `--preset` / `--style` / `--angle` / `--speaker[12]-style` write into `gemini.style.*` and `gemini.speaker[12].style_overlay` (never into `personality`). `llm_summarizer._build_prompt` renders them inside the dialogue prompt: per-speaker overlays in a dedicated `Episode-specific adjustments:` block between `Host personalities:` and `Instructions:`; preset + free style as a `Stylistic guidance:` sub-section inside `Instructions:`; angle as an `- Episode angle:` bullet. The angle is also injected into `research._ROUND_1_PROMPT` (and nowhere else — round N≥2 only sees it indirectly via `previous_notes`, so gap-analysis stays neutral).
- **Voice duo resolution**: `duos.py` holds `BUILTIN_DUOS` (warm/contrast/explorer/journalist/debate), available out of the box. `cli.py::run` resolves the active duo *before* reading any speaker field, with precedence `--duo` > `gemini.default_duo` > legacy `gemini.speakerN` blocks > built-in `contrast`, then writes the result into `gemini_cfg["speaker1"/"speaker2"]`. This is the single injection point — every downstream consumer (TTS preamble, dialogue prompt, `--speakerN-style` overlays) reads `gemini.speakerN` unchanged, and a config defining only legacy `speaker1`/`speaker2` keeps working untouched. A user `gemini.duos` mapping is merged over the built-ins (same slug overrides; new slugs extend). `tts-podcast duos` lists them (reads the *raw* config, so it needs no API key).
- **Duo content contract**: the five built-in slugs are stable (renaming or dropping one breaks user configs and CLI invocations), but their voices, names, personalities and descriptions are not frozen. Each duo must stay a *distinct conversational dynamic* (transmission / tempo flip / co-discovery / role asymmetry / open conflict): the reported bug was that four of five were the same "excited host plus calm analyst" pairing renamed. Two rules keep them apart, both documented in the `duos.py` module docstring: no two duos share a timbre family (ten distinct voices, ten distinct official descriptors), and within a duo the two `voice_direction` notes are opposed on register, tempo, articulation *and* breathing, each describing a steady trait rather than a trajectory (a ramping note fights the anti-crescendo instruction in the TTS director's notes). A third rule covers wording: `personality` has two consumers with two sentence frames, the TTS preamble's `"{name} is {personality}."` and the dialogue prompt's `"- {name}: {personality}"`, so every personality is a noun phrase opening with a determiner. A bare job title reads fine as a bullet but ships `"Nora is desk anchor;"` to the TTS model. `tests/test_duos.py::TestPersonalityGrammar` enforces it.
- **Duo-supplied `tts_style` defaults**: a duo may declare top-level `scene` and `pace`. `resolve_duo` surfaces them only when non-blank, so the caller can tell "the duo is silent" from "the duo set an empty value". `cli.py::run` fills `gemini_cfg["tts_style"][scene|pace]` from the duo **only when the user's own value is falsy**: user config always wins, and the fill happens on the already-copied `gemini_cfg`, so `cfg["gemini"]["tts_style"]` is never mutated. Consequence for `config.example.yaml`: `tts_style.scene` and `tts_style.pace` ship **commented out** there. Shipping them active pins one scene and one pace across all five duos, which reproduces the "every duo sounds the same" symptom in every config derived from the example.
- **Hard invariant — TTS preamble untouched**: `tts_generator._build_tts_prompt` reads `gemini_cfg["speakerN"]["personality"]` verbatim. The `style_overlay` key is for the dialogue prompt only and MUST NEVER be read by the TTS path. `personality` is never mutated, in memory or on disk, by any code path. Regression test: `tests/test_tts_generator.py::test_tts_preamble_unaffected_by_speaker_overlay`.
- **Hard invariant, `voice_direction` is TTS-only**: the exact mirror image of `style_overlay`. `gemini.speakerN.voice_direction` (also settable per duo speaker) is an optional Director's Note covering register, tempo, articulation and breathing. It is read **only** by `tts_generator._build_tts_prompt`, which renders it as `Voice direction for <name>: <direction>` right after that host's personality line, and it MUST NEVER reach the dialogue prompt built by `llm_summarizer._build_prompt`. When absent or blank the rendered preamble is byte-identical to the pre-`voice_direction` output, so legacy configs sound unchanged. `duos.validate_speaker` rejects a non-string value; `name` and `voice` remain the only required speaker fields. It runs on duo speakers via `resolve_duo` **and** on legacy `gemini.speaker1` / `speaker2` blocks from `cli.py::run` (that path bypasses `resolve_duo`, so without the explicit call a YAML `voice_direction: 42` or list only surfaced as an `AttributeError` inside the TTS thread pool, after the dialogue had been billed). `_build_tts_prompt` additionally coerces with `str()` so a config assembled in Python cannot crash a worker thread.
- **Snapshot fixture for the dialogue prompt**: `tests/fixtures/dialogue_prompt_no_overlay.txt` is the byte-identical baseline used by `test_no_flags_byte_identical`. When `_SYSTEM_PROMPT_TEMPLATE` is intentionally edited (typo, wording tweak): (1) edit the template, (2) `uv run python -m tests.fixtures.regen_dialogue_prompt`, (3) review the diff, (4) commit the fixture alongside the template change. The `tests/conftest.py` `collect_ignore_glob = ["fixtures/*"]` line guarantees pytest never auto-collects anything under `tests/fixtures/`.

### Configuration loader

`config.load_config` resolves any YAML key ending in `_env` by looking up the named environment variable, then drops the `_env` suffix in the returned dict. So `api_key_env: GEMINI_API_KEY` in YAML becomes `cfg["gemini"]["api_key"] = os.environ["GEMINI_API_KEY"]`. Missing env vars raise `ConfigError` at load time (fail-fast).

CLI flags override config: `-R/--research`, `-d/--duration`, `-o/--output-dir`. The duration override mutates `gemini_cfg["dialogue"]["target_duration_minutes"]` in memory; min/max default to 70 % / 150 % of target unless set explicitly in config.

### Module map

| Module | Role |
|---|---|
| `cli.py` | Click entry point, pipeline orchestration, `config init/show` wizard, `duos` command |
| `config.py` | YAML loader + `_env` resolution |
| `duos.py` | Named voice duos: `BUILTIN_DUOS` registry (per-speaker `voice_direction`, duo-level `scene` / `pace`) + `resolve_duo` / `describe_duos` |
| `models.py` | `Source` dataclass with `kind` field (`"url"` / `"file"` / `"search"`) |
| `web_scraper.py` | trafilatura-based scraping, parallel (≤10 workers), optional CloakBrowser fallback |
| `cloak_fetcher.py` | Optional `cloakbrowser` stealth-Chromium fetch (graceful no-op when absent) |
| `local_loader.py` | Local file reader (txt, md, html via trafilatura, pdf via pypdf) |
| `research.py` | Iterative Gemini + Google Search grounding rounds |
| `llm_summarizer.py` | Dialogue generation + byte-bounded chunking (budget measured per run from the TTS preamble) |
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
