# Plan — Local document & web-search inputs

**Status:** pending approval
**Branch:** `feat/local-and-search-inputs` (already created off `main`)
**Date:** 2026-05-23

## Requirements summary

Today, `tts-podcast run` only accepts URLs as positional arguments. Add two new input kinds, processed through the same pipeline:

1. **Local documents** — txt, md, html, pdf. Read locally, no network. Treated as `Source` objects with full text populated.
2. **Web-search queries** — natural-language topics that drive the research stage (no scraped article behind them). Materialised as synthetic `Source` objects so the rest of the pipeline is unchanged.

### Design choices (confirmed via interview)

| Decision | Choice | Rationale |
|---|---|---|
| File formats | `.txt`, `.md`, `.markdown`, `.html`, `.htm`, `.pdf` | Covers the common cases; `pypdf` is small and pure-Python. |
| CLI shape | Positionals auto-detect URL vs. file; new `-s/--search` flag (repeatable) | Backward compatible, ergonomic, no need to type `-f` for the common case. |
| Search-only behaviour | Auto-bump `--research` to 1 when search-only inputs and `-R == 0` | Sans research, le dialogue n'aurait rien à discuter. |
| `Source` differentiation | Add `kind: str = "url" \| "file" \| "search"` field with default `"url"` | Explicit, forward-compatible, default keeps existing call sites working. |

## Acceptance criteria

All criteria are concrete and testable.

1. `uv run tts-podcast run -n some_doc.pdf` produces a dialogue using only local PDF text — no `trafilatura.fetch_url` is called for that input.
2. `uv run tts-podcast run -n -s "agentic AI memory"` runs at least 1 research round (auto-bumped from 0), then emits a dialogue, with no scraped source.
3. `uv run tts-podcast run -n https://example.com/post README.md -s "follow-up topic"` accepts mixed inputs and produces one combined dialogue. The URL is scraped, the markdown is read locally, the search query is investigated via research.
4. `uv run tts-podcast run` (no inputs) exits with code 1 and a clear error: "No inputs provided. Pass URL(s), -f FILE, or -s 'search query'".
5. A positional that is neither a URL nor an existing file exits with code 1 and a clear error naming the offending value.
6. Local files: `.txt` / `.md` / `.markdown` decoded as UTF-8 (errors=replace); `.html` / `.htm` passed through `trafilatura.extract`; `.pdf` parsed with `pypdf.PdfReader` joining page texts with `\n\n`.
7. Empty-extraction local files (e.g., a scanned PDF with no text layer) yield `Source(scraped_ok=False)` and surface a warning, but do not abort the run if other ok sources exist.
8. The "all inputs failed" abort path still fires when no `Source` has `scraped_ok=True` (e.g., all local files empty, all URLs 404, no search queries).
9. `pypdf>=4.0` is added to `[project] dependencies` in `pyproject.toml`.
10. Output stem builder accepts the unified identifier list (URL, file path, search query) and derives a sensible label from the first item: hostname for URL, filename stem for file, slugified first-N-chars for search.
11. Report folder (`overview.md`, `sources.md`) renders **search** sources without a clickable URL (label only); file/url sources keep current rendering.
12. `link_extractor.extract_links` only adds **http(s)** sources to the `report.sources` bucket — `file://` and `search://` are excluded from categorised link tables.
13. `uv run pytest tests/ -q` passes (existing suite + new `test_local_loader.py`).
14. `uv run ruff check src/ tests/` passes.
15. The dialogue prompt for `kind="search"` Sources replaces the empty body with "Topic to investigate via web research: <query>" (so the LLM understands).
16. The research prompt for `kind="search"` Sources renders as "Topic to investigate: <query>" instead of "URL: <url>\n<empty body>".

## Implementation steps

Listed in build order. Each step is small enough to verify in isolation.

### Step 1 — Dependency

- `pyproject.toml`: add `pypdf>=4.0` to `[project].dependencies` (keeps the install lean; no extras group).
- Run `uv sync` to update `uv.lock`.

### Step 2 — `Source` gets a `kind` field

- `src/tts_podcast/models.py`: add `kind: str = field(default="url")` to `Source`. Document allowed values (`"url"`, `"file"`, `"search"`) in the docstring.
- Backward compatibility: existing positional constructions (5 args) are unaffected because `kind` is the last field with a default.

### Step 3 — New module `src/tts_podcast/local_loader.py`

- Public API:
  - `load_local_file(path: Path) -> Source` — dispatches on extension, returns a populated `Source(kind="file")`. Errors → `scraped_ok=False`, no exception bubble.
  - `load_local_files(paths: list[Path], *, progress=None, task_id=None) -> list[Source]` — sequential (local I/O is fast), advances progress per file.
- Internal readers:
  - `_read_text(path)` — UTF-8, `errors="replace"`.
  - `_read_html(path)` — read raw, pass through `trafilatura.extract`.
  - `_read_pdf(path)` — `pypdf.PdfReader`, join non-empty page texts with `\n\n`. Per-page extraction wrapped in try/except → log warning, continue.
- `url` field on file Sources: `file://{abs_path}` so reports render as a working local link.
- Summary: first 500 chars of full text (matches `web_scraper._SUMMARY_CHARS`).

### Step 4 — Search-source factory

- Helper in `src/tts_podcast/cli.py` (small enough to stay inline):
  ```python
  def _make_search_source(query: str) -> Source:
      return Source(
          url=f"search://{query}",
          title=f"Web search: {query}",
          summary=f"Topic to investigate via web research: {query}",
          full_text=f"Topic to investigate via web research: {query}",
          scraped_ok=True,
          kind="search",
      )
  ```
- `scraped_ok=True` because the source is conceptually valid (research will provide the content).

### Step 5 — Prompt rendering aware of `kind`

- `src/tts_podcast/research.py::_format_articles`: branch on `src.kind == "search"` → emit `[i] Topic to investigate: {title}` (no URL/body lines). Otherwise unchanged.
- `src/tts_podcast/llm_summarizer.py::_build_prompt`: same branch — search sources render as `[i] Topic of investigation: {title}\n(See research notes below for findings.)`.

### Step 6 — Link extractor & report tweaks

- `src/tts_podcast/link_extractor.py::extract_links`: skip adding to `report.sources` when `source.url` doesn't start with `http://` or `https://`. Body-text URL scanning is unchanged (regex already matches only `https?://`).
- `src/tts_podcast/report_generator.py`:
  - `_render_overview` — Sources list: for `kind == "search"`, render `{idx}. *Web search:* \`{title}\``. For `kind == "file"`, render normally (file:// link works). Rename "scrape failed" marker to "input failed" (generic across URL/file).
  - `_render_sources` — for `kind == "search"`, omit `**URL:**` line, write `*Topic researched via Google Search grounding — see research notes.*`. File/URL sources unchanged.

### Step 7 — CLI surface

- `src/tts_podcast/cli.py`:
  - Positional argument renamed `inputs` (still `nargs=-1`, `required=False`), metavar `URL_OR_FILE...`.
  - New options:
    - `-f, --file files multiple=True type=click.Path(exists=True, dir_okay=False, path_type=Path)`
    - `-s, --search search_queries multiple=True`
  - Input parsing block (after `_setup_logging`):
    - For each positional: if starts with `http://` / `https://` → URL; else if `Path(arg).is_file()` → file; else → error.
    - Validate at least one input exists overall.
  - Build the unified `all_sources`: scrape URLs → load files (with progress task) → append search sources.
  - Auto-bump research:
    ```python
    if research_rounds == 0 and search_list and not (url_list or file_paths):
        # search-only fallback handled by the chosen design (see step 0)
        logger.info("Search-only run with no research rounds — bumping to 1.")
        research_rounds = 1
    ```
    Effectively: if any search query is present **and** there are no scraped/file sources contributing content, force at least one round.
- Generalise `_build_output_stem(identifiers: list[str]) -> str`:
  - URL → host (strip `www.`).
  - `file://` URI → `Path(path).stem`.
  - `search://` URI → slugified first ~40 chars.
  - Hash combined list (sha1[:6]) + ISO date.
  - Add helper `_slugify(text) -> str`.
- Call site: `stem = _build_output_stem(identifiers)` where `identifiers = url_list + [f"file://{p.resolve()}" for p in file_paths] + [f"search://{q}" for q in search_list]`.

### Step 8 — Tests

- `tests/test_local_loader.py`:
  - Plain text round-trip.
  - Markdown round-trip.
  - HTML through trafilatura (mock `trafilatura.extract` to assert it's called with raw bytes).
  - PDF: stub `pypdf.PdfReader` via `monkeypatch` to return fake pages → assert join.
  - Missing file → `FileNotFoundError`.
  - Empty file → `scraped_ok=False`.
  - Unknown extension → reads as text + warning.
- Optionally add a smoke test in `tests/` covering `_make_search_source` and the auto-detection branch (kept light to avoid CLI brittleness).

### Step 9 — Docs

- Update `README.md` Usage section with `-f` and `-s` examples.
- Update `CLAUDE.md` Architecture section to mention the three input kinds and the `kind` field on `Source`.
- Update `config.example.yaml` comments only if a new config key is added (none planned).

## File-by-file impact

| File | Change |
|---|---|
| `pyproject.toml` | + `pypdf>=4.0` dependency |
| `uv.lock` | regenerated by `uv sync` |
| `src/tts_podcast/models.py` | + `kind` field on `Source` |
| `src/tts_podcast/local_loader.py` | **new** |
| `src/tts_podcast/research.py` | `_format_articles` → kind-aware |
| `src/tts_podcast/llm_summarizer.py` | `_build_prompt` → kind-aware |
| `src/tts_podcast/link_extractor.py` | skip non-http urls in `report.sources` |
| `src/tts_podcast/report_generator.py` | render search/file kinds in overview & sources |
| `src/tts_podcast/cli.py` | new flags, positional auto-detect, generalised stem, auto-bump research |
| `tests/test_local_loader.py` | **new** |
| `README.md` | + usage examples |
| `CLAUDE.md` | + architecture note |

Total: 8 files modified, 2 created.

## Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `pypdf` adds ~1 MB to install | High | Low | Acceptable for a CLI that's already shipping ffmpeg requirement. Keep in core deps, not extras. |
| PDF extraction quality varies (scanned/image PDFs return empty) | Medium | Medium | Empty-text PDFs return `scraped_ok=False` and log a warning; pipeline continues if other sources are OK. Don't try OCR. |
| `search://` URL scheme renders weirdly in markdown viewers | High | Low | Report explicitly handles `kind == "search"` (no link). |
| Auto-detection ambiguity (filename starts with `http://`) | Vanishing | Low | Document the rule: starts with `http://`/`https://` wins. Users can use `-f` to force file mode if needed. |
| Search-only auto-bump surprises a user who really wanted 0 rounds | Low | Low | Logged at INFO level. User can pass `-R 0` explicitly with URLs/files to bypass. |
| `pypdf` import error if the dep isn't installed (e.g., dev environment) | Low | Medium | Wrap import in a function-local try/except with a clear error message pointing to `uv add pypdf`. |
| Behaviour change: `Source` constructor signature now has 6 fields | Low | Low | New field has a default; positional callers untouched. Tests using `Source(url=..., title=..., ...)` keyword form continue to work. |
| Existing tests rely on `Source` shape | Low | Low | All current `Source` constructions in tests use kwargs or 1-2 positional args; verified during exploration. |

## Verification plan

In order, with concrete commands:

1. **Lint**: `uv run ruff check src/ tests/` — no errors.
2. **Tests**: `uv run pytest tests/ -q` — all green, including new `test_local_loader.py`.
3. **Smoke (file)**: `uv run tts-podcast run -n README.md` — dry-run prints a dialogue derived from the README content.
4. **Smoke (search)**: `uv run tts-podcast run -n -s "Claude Code release notes 2026"` — logs an auto-bump to 1 round, runs research, prints dialogue.
5. **Smoke (mixed)**: `uv run tts-podcast run -n https://example.com README.md -s "edge cases"` — all three input kinds funnelled into one dialogue (rely on `httpbin` or any reachable URL; ok to fail-soft on URL).
6. **Error path**: `uv run tts-podcast run` — exits 1 with the expected message.
7. **Error path**: `uv run tts-podcast run not-a-file-or-url` — exits 1, names the offending value.
8. **Report check**: pick the run from step 3 (with `--report`) and grep the produced `overview.md` for the kind-aware rendering of the file source.

## Out of scope (not in this plan)

- `.docx` support — declined during interview.
- OCR fallback for image-only PDFs.
- A `--config` knob for which file types are enabled.
- Streaming/very-large-PDF handling.
- Caching of local file reads.
- URL scheme other than `http(s)://` (e.g., `ftp://`, `gemini://`).
