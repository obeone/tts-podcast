# Plan — Style & angle CLI controls

**Status:** pending approval
**Branch:** `feat/style-and-angle-cli` (to be created off `main`)
**Date:** 2026-05-23
**Mode:** Consensus (RALPLAN-DR short)
**Revision:** 2 (incorporates Architect + Critic feedback, iteration 1)

## RALPLAN-DR summary

### Principles

1. **Backward compatibility is mandatory.** A run with no new flags must produce a byte-identical prompt to today's behaviour. The feature is opt-in.
2. **CLI overrides config.** New options follow the existing precedence pattern (`--duration` → `gemini_cfg["dialogue"]["target_duration_minutes"]`). One way to do it.
3. **Style is a dialogue-content concern, not a voice-acting concern.** The new options touch the LLM dialogue prompt (and the research prompt for the angle). The existing `gemini.tts_style.scene/pace` keys keep their semantics — they control *how the speakers sound*, not *what they discuss*. `tts_generator._build_tts_prompt` (`tts_generator.py:41–87`) is not modified, and `speaker[N].personality` is **never mutated** so the TTS preamble continues to read the configured personality verbatim.
4. **Free-text injections must not dominate the prompt.** Cap each free-text field at 500 chars to prevent accidental megaprompts that drown out the article content or shift the LLM's attention away from its core task. (The TTS chunk byte budget at `llm_summarizer.py:33` is a separate constraint that bounds *output chunks*, not the dialogue-generation prompt.)
5. **Presets are a stable contract.** They are a versioned, finite list living next to the prompt template; renaming or removing one is a breaking change. The preset *name* must be unambiguous — `--preset` accepts a preset key only, errors on unknown name. Free text always goes through `--style`. The two surfaces never share a parser.

### Decision drivers (top 3)

1. **Author ergonomics** — episode-to-episode style tuning without editing YAML. The flag must be the obvious answer when the user thinks "I want this one to be more academic".
2. **Steering-signal coherence** — bad style text should not corrupt the dialogue, and contradictory steering across presets/overlays/angle should not be silently introduced. Each surface owns one slot.
3. **Implementation surface / blast radius** — must stay scoped to dialogue + research prompts. No schema migration, no API contract changes, **no TTS-side changes** (criterion #15 below is a hard invariant, not a soft preference).

### Viable options

**Option A — Split flags: `--preset NAME` + `--style TEXT` (chosen)**
*Pros:* zero parser ambiguity (preset is a key, style is free text); user's "presets + texte libre" preference is still served (the two flags compose); preset list maintainability stays low (no resolution helper, no escape hatches); each flag does exactly one thing.
*Cons:* three flags instead of two for the style dimension; users discover the combo via `--help`.

**Option B — Free-text only, no presets**
*Pros:* minimum code; no preset list to maintain; no naming bikeshed.
*Cons:* user must hand-craft prompt fragments every time; no safety net for poorly worded style requests; loses repeatability across episodes; contradicts the user's confirmed preference at interview ("presets + texte libre").
*Invalidation:* user explicitly chose "presets + free text" at interview; Option B drops the preset half outright. Re-listed because the Critic correctly flagged that the maintenance-cost angle deserves acknowledgement: preset list drift is a real long-term cost (mitigated by keeping the initial set small and curated — 5 presets).

**Option C — Structured per-speaker tone YAML schema (new declarative block)**
*Pros:* fully declarative; per-attribute tuning (pace_hint, formality, humor_level…); reproducible.
*Cons:* large schema-design effort; YAML churn; over-engineered for a *quick CLI nudge* use case; would still need a CLI surface anyway, so it's additive to Option A rather than a replacement.
*Invalidation:* out of scope for the requested CLI ergonomics; can be revisited later if the additive overlay proves insufficient.

**Option D — Reuse `gemini.tts_style.scene` as the unified style surface**
*Pros:* one less config block; user's "studio scene" already steers TTS voice acting, so extending it to dialogue content has a surface logic; fewer keys to discover.
*Cons:* conflates two distinct concerns. `tts_style.scene` is voice-acting context (where the speakers are; how they sound) consumed by `tts_generator._build_tts_prompt`. Dialogue style/angle is content shape (what they discuss, how rigorously, from what angle). Today's `scene = "Two friends co-hosting a casual French tech podcast in a cozy studio"` is appropriate for voice direction but useless for steering the dialogue tone toward academic vs casual. Repurposing the field would force one of two bad outcomes: (a) lose backwards compatibility for users who already set `scene` for TTS only, or (b) make the field a polysemic mess that means different things depending on which stage reads it. The Architect/Critic exchange confirmed this is a non-trivial design call.
*Invalidation:* yes — keeping `scene` as voice-acting context and introducing a separate `style` block preserves the existing semantics, avoids backward-compat hazards, and matches the "Style is a dialogue concern, not a TTS concern" principle. Documented here so the choice isn't silent.

## Requirements summary

Add three CLI surfaces that influence dialogue (and, for the angle, the first research round) without touching TTS or the rest of the pipeline:

1. **Overall podcast style** — `--preset NAME` (resolves against a curated dict of style fragments, errors on unknown name) and/or `--style TEXT` (free text). The two flags **compose**: both may be set in the same invocation. Their effects render in distinct prompt sub-sections.
2. **Per-speaker style overlay** — `--speaker1-style TEXT` and `--speaker2-style TEXT`. **Strictly additive**: the existing `gemini.speaker[1|2].personality` stays in YAML and reaches the TTS preamble unchanged (so voice acting is not affected). The overlay is read from a new `style_overlay` key and rendered in its own dedicated `Episode-specific adjustments:` prompt block — never inlined into the personality string.
3. **Angle** — `--angle TEXT`. Injected into the dialogue prompt (`_SYSTEM_PROMPT_TEMPLATE`) and into the **first research round only** (`_ROUND_1_PROMPT`). Round N≥2 inherits the angle via `{previous_notes}` only — no re-injection — to avoid the contradictory steering identified by the Architect (the gap-analysis prompt would conflict with re-affirmed focus).

### Design choices (resolved during consensus review)

| Decision | Choice | Rationale |
|---|---|---|
| Style flag shape | Split: `--preset NAME` + `--style TEXT` | Two flags, two parsers, zero ambiguity. Kills the resolution-rule footgun that the Architect flagged. |
| Per-speaker mutation policy | **`speaker[N].personality` is never mutated**, even in memory. Overlay lives in `speaker[N].style_overlay`. | Prevents the TTS-preamble leak the Critic identified at `tts_generator.py:72–77`. Hard invariant. |
| Per-speaker rendering | Dedicated `Episode-specific adjustments:` block in `_SYSTEM_PROMPT_TEMPLATE`, only present when at least one overlay is set | Structural visibility — the LLM treats it as a directive, not as a tail to the personality string. |
| Angle injection points | Dialogue prompt **and** round-1 research prompt only. Not re-injected in round N≥2. | Survives `--research 0`; focuses round 1; lets round N≥2 do gap-analysis as designed. |
| Schema location | New `gemini.style` top-level block (`preset`, `text`, `angle`) sibling to `dialogue`, `tts_style`; per-speaker overlays as new `speaker[N].style_overlay` keys | Keeps the qualitative-prompt-steering fields out of `gemini.dialogue` (which holds quantitative knobs: `target_duration_minutes`, `words_per_minute`). |
| Free-text length cap | 500 chars per free-text field; truncate with `logger.warning` | Prompt steering coherence (not byte budget). Free text exceeding the cap is silently truncated to 500 chars after a warning. |
| Preset list | Initial 5: `casual`, `academic`, `humorous`, `debate`, `vulgarized` | Small, curated, reconciled (was 7 in earlier draft). Adding more later is opt-in; removing or renaming is a breaking change. |
| Disable-default semantics | `--style ""` or `--angle ""` (empty string) explicitly forces no value, overriding any config default. `--preset ""` likewise treated as no preset. | Click distinguishes empty string from `None` (flag not passed). Documented in `--help`. |

## Acceptance criteria

All criteria are concrete and testable. Numbering kept stable where possible; revisions marked `[rev]`.

### CLI surface

1. **[rev]** `uv run tts-podcast run --help` lists five new options, all long-form only (no short flags to avoid `-y`/`-a` collisions and case-sensitivity gotchas with `-A`): `--preset NAME`, `--style TEXT`, `--speaker1-style TEXT`, `--speaker2-style TEXT`, `--angle TEXT`.
2. **[rev]** `uv run tts-podcast run -n URL --preset academic` injects the `academic` preset fragment into the dialogue prompt under a `Stylistic guidance:` header.
3. `uv run tts-podcast run -n URL --style "lively but rigorous, French academic feel"` injects the free text into the dialogue prompt under the same `Stylistic guidance:` header (concatenated below the preset fragment if both are set).
4. **[rev]** `uv run tts-podcast run -n URL --preset academic --style "but extra dry"` shows both the preset fragment and the free text under `Stylistic guidance:`, in that order, separated by a blank line. (Replaces the old first-token-matching rule entirely.)
5. **[rev]** `--speaker1-style "more skeptical than usual"` causes the dialogue prompt to contain a dedicated `Episode-specific adjustments:` block listing `- Alex: more skeptical than usual`. **No code path mutates `gemini_cfg["speaker1"]["personality"]`**, in memory or on disk. The TTS preamble built by `_build_tts_prompt` reads the unchanged `personality` field. Verified by a test that asserts the configured personality string appears verbatim in the TTS preamble even when the overlay is set.
6. **[rev]** `--angle "the economic implications"` injects the angle into (a) `_ROUND_1_PROMPT` under a new `Angle to emphasize:` line, and (b) `_SYSTEM_PROMPT_TEMPLATE` under an `Episode angle:` instruction line. The angle is **not** re-injected into `_ROUND_N_PROMPT` (criterion #17 reflects this).
7. Free text in any of the four free-text fields (`--style`, `--speaker1-style`, `--speaker2-style`, `--angle`) exceeding 500 chars is truncated to 500 chars; `logger.warning` names the field; the truncated value is what reaches the prompt.
8. **[rev]** `uv run tts-podcast run --preset nosuchpreset URL` exits with code 2 and prints the list of valid preset names. `uv run tts-podcast run --style "nosuchpreset"` is treated as free text (no error). The two flags never share a parser, so there is no ambiguity.
9. `uv run tts-podcast run` (no new flags) behaves byte-identically to today. Verified by `test_no_flags_byte_identical` in tests/test_llm_summarizer.py which snapshots the prompt against a fixture.
10. **[rev / new]** `--style ""` (empty string) explicitly overrides a configured default to "no style", same for `--angle ""`, `--speaker1-style ""`, `--speaker2-style ""`. Documented in `--help` text. Click's empty-string vs `None` distinction is exploited deliberately for these four **free-text** flags.
    **Note on `--preset`:** because `--preset` uses Click's `Choice`, passing `--preset ""` would fail with exit code 2 ("'' is not one of …"). To disable a configured `gemini.style.preset` for one run, omit the flag (config default applies) or use the explicit sentinel `--preset none` (added as a recognized choice that maps to `None` in `validate_preset`). Documented in the flag's help text.

### Config surface

11. **[rev]** `config.example.yaml` documents five new keys, all optional, all defaulting to empty string:
    - `gemini.style.preset` — preset name (must match a key in `STYLE_PRESETS`, validated at runtime).
    - `gemini.style.text` — free-text style guidance.
    - `gemini.style.angle` — free-text episode angle.
    - `gemini.speaker1.style_overlay` — free-text overlay for speaker 1.
    - `gemini.speaker2.style_overlay` — free-text overlay for speaker 2.
    Each appears as a commented-out example with a one-line docstring explaining purpose and precedence ("CLI overrides config").
12. CLI flags override config (same pattern as `--duration` mutating `gemini_cfg["dialogue"]["target_duration_minutes"]` at `cli.py:447–457`). For per-speaker overlays specifically, the CLI writes to `gemini_cfg["speaker1"]["style_overlay"]` (a *separate* key), never to `gemini_cfg["speaker1"]["personality"]`.
13. **[rev]** `uv run tts-podcast config init` wizard prompts for (a) preset name (with the list of valid presets shown as flavour text; blank to skip), (b) free-text style (blank to skip), (c) episode angle (blank to skip). It does NOT prompt for per-speaker overlays — those appear as commented-out keys in the generated YAML with a one-line docstring so users can discover and uncomment them. Anchor in the wizard: directly after the existing `speaker2.personality` prompt (the wizard does not currently have a duration prompt — the original plan was wrong about that).

### Prompt invariants (revised)

14. **[rev]** When all five free-text/preset fields are populated to their 500-char caps and a preset is selected, the rendered dialogue prompt:
    - (a) contains every injected fragment verbatim (preset fragment, free-text style, both speaker overlays, angle),
    - (b) places them in this fixed order in the prompt: `Host personalities:` → `Episode-specific adjustments:` → `Instructions:` block, with `Stylistic guidance:` as a sub-section inside it (immediately after the existing tone bullet) followed by the `- Episode angle: …` bullet → `Articles:`,
    - (c) does not double-emit any block header.
    Rationale (per Critic iter-2 Major #2): keeping all qualitative steering (`Stylistic guidance:` + `Episode angle:`) inside the `Instructions:` block produces one coherent directive list rather than a free-floating section above plus a bullet below. The baseline personality precedes the per-episode adjustment so the LLM reads the override as a *delta* against an established baseline, not as a competing definition.
    Verified by `test_prompt_section_order_when_all_options_set` modelled on the existing `test_research_notes_appear_in_prompt` (`tests/test_llm_summarizer.py:243`) using `prompt.index(...)` ordering assertions.
    (This replaces the previous byte-budget criterion — `_MAX_CHUNK_BYTES = 3000` is the TTS chunk budget, not a constraint on the text-generation prompt.)
15. Audio-tag detection (`_audio_tags_enabled`) is unchanged — style/angle text never contains literal `[tag]` placeholders inserted by us.
16. **[rev — hard invariant]** The TTS preamble (`tts_generator._build_tts_prompt` at `tts_generator.py:41–87`) is **not** modified. `speaker[N].personality` is never mutated, so the preamble at `tts_generator.py:75–79` reads the configured value verbatim. Verified by `test_tts_preamble_unaffected_by_speaker_overlay`: set `--speaker1-style "extremely angry"`, build the TTS prompt, assert the preamble's `f"{name} is {personality}.\n"` line contains the *original* personality and does *not* contain `"extremely angry"`.

### Tests

17. **[rev]** `tests/test_llm_summarizer.py`:
    - `test_preset_injected` — passing `preset="academic"` puts the preset fragment in the prompt under `Stylistic guidance:`.
    - `test_style_free_text_injected` — passing free text injects it under `Stylistic guidance:`.
    - `test_preset_plus_style_compose` — both `preset` and `style` set ⇒ both appear, in documented order.
    - `test_speaker_overlay_in_dedicated_block` — overlay appears under `Episode-specific adjustments:`, NOT inlined into `Host personalities:` block.
    - `test_speaker_overlay_does_not_mutate_personality` — after `generate_dialogue(..., speaker1_overlay="X")`, `gemini_cfg["speaker1"]["personality"]` equals its original value.
    - `test_angle_in_dialogue_prompt` — angle appears under `Episode angle:` in the dialogue prompt.
    - `test_angle_in_dialogue_prompt_without_research` — angle still reaches the dialogue prompt when `research_rounds=0`.
    - `test_truncation_warning_per_field` — 600-char input in any of `style`, `speaker1-style`, `speaker2-style`, `angle` is truncated to 500 with the field name in the warning (parametrized over the four fields).
    - `test_prompt_section_order_when_all_options_set` — see criterion #14.
    - `test_no_flags_byte_identical` — `_build_prompt(...)` with all new params at default (`None` / `""`) produces a string byte-identical to a fixture committed at `tests/fixtures/dialogue_prompt_no_overlay.txt`. Fixture is regenerated once during initial implementation by running the current (pre-refactor) code path on the same inputs.
18. **[rev]** `tests/test_research.py`:
    - `test_angle_in_round1_prompt` — angle appears in `_ROUND_1_PROMPT`.
    - `test_angle_header_NOT_re_injected_in_round_n_prompt` — the literal header line `Angle to emphasize:` does NOT appear in the round-N rendered prompt. (The angle *text* itself may legitimately appear inside `{previous_notes}` carried from round 1 — that's expected and desirable; this test forbids only the re-injected header, replacing the previous round-N injection requirement.)
    - `test_round1_no_angle` — no angle ⇒ no `Angle to emphasize:` line. Byte-identical round-1 prompt to pre-refactor.
    - `test_angle_plus_search_input` — when input is `-s "AI economy"` and `--angle "regulatory impact"` is set, the round-1 prompt contains both the search topic and the angle.
19. **[rev]** `tests/test_tts_generator.py` (extend existing, or add if absent): `test_tts_preamble_unaffected_by_speaker_overlay` — see criterion #16.
20. **[rev]** `tests/test_cli.py` (new): use `CliRunner` to invoke `tts-podcast run -n --preset academic --angle "economy" --speaker1-style "X" URL`. Assert that:
    - `generate_dialogue` receives `style`, `angle`, `speaker1_overlay`, `speaker2_overlay` as separate kwargs (not as mutations of `personality`).
    - `conduct_research` receives `angle="economy"` as a kwarg.
    - `gemini_cfg["speaker1"]["personality"]` is unchanged after the run.
    Mock `scrape_urls`, `generate_dialogue`, `conduct_research`, and `generate_audio_chunks` at the module boundary.
21. `uv run pytest tests/ -q` passes.
22. `uv run ruff check src/ tests/` passes.

## Implementation steps

Build order, each step independently verifiable. Note: this revision splits Step 1 from the original (no more `resolve_style` helper since the flags no longer share a parser).

### Step 1 — Preset dictionary + truncation helper

- New module `src/tts_podcast/style_presets.py`:
  - Module-level `STYLE_PRESETS: dict[str, str]` with **exactly 5** curated presets (reconciled count): `casual`, `academic`, `humorous`, `debate`, `vulgarized`. Keys are lowercase short names; values are one-paragraph **English** prompt fragments (English meta-instructions are robust across `{language}` dialogue targets — empirically Gemini handles language-mixed meta-instructions well).
  - Function `validate_preset(name: str | None) -> str | None`: returns the preset's prompt fragment if `name` is a valid key, returns `None` if `name` is `None`, empty string, or the literal sentinel `"none"`, raises `click.BadParameter` if `name` is non-empty but unknown. Lists valid keys in the error message. The `"none"` sentinel is the only way to disable a configured `gemini.style.preset` from the CLI (since Click's `Choice` rejects empty strings).
  - Function `truncate_with_warning(text: str | None, field: str, *, cap: int = 500) -> str | None` for criterion #7.
- Tests: `tests/test_style_presets.py` covers preset present, preset unknown (raises with key list), preset None/empty (returns None), cap.

### Step 2 — `llm_summarizer.py`: dialogue prompt structure

- Extend `_SYSTEM_PROMPT_TEMPLATE` (`llm_summarizer.py:87`) with three new placeholders. Each renders to an empty string when the corresponding inputs are unset, preserving byte-identical backward compatibility (criterion #9). **All `\n` in the rendered-fragment strings below are real newline characters in the output (interpreted, not literal backslash-n).**
  - After the `Host personalities:` block: `{speaker_adjustments_block}`. Renders empty, OR a literal `"\nEpisode-specific adjustments:\n- {speaker1_name}: {overlay1}\n- {speaker2_name}: {overlay2}\n"` (only the populated overlays appear; if only one overlay is set, only one bullet renders).
  - **Inside** the `Instructions:` block, immediately after the existing tone bullet (`"Keep the tone informative but lively …"` at `llm_summarizer.py:107`): `{style_block}`. Renders empty, OR a literal `"\nStylistic guidance:\n{preset_fragment}\n\n{style_free_text}\n"` (with single fragments if only one of preset/style is set, no blank line separator in that case). Moved inside `Instructions:` per Critic iter-2 Major #2 so all qualitative steering lives in one directive block.
  - Inside the `Instructions:` block, adjacent to the tone bullet (after `{style_block}` if both are populated): `{angle_line}`. Renders empty, OR `"- Episode angle: {angle}. Weave this through the conversation; don't just mention it once.\n"`.
- Update `_build_prompt()` (`llm_summarizer.py:167`) to accept new keyword-only params: `preset: str | None = None`, `style_text: str | None = None`, `speaker1_overlay: str | None = None`, `speaker2_overlay: str | None = None`, `angle: str | None = None`. Resolve `preset` via `validate_preset` — this is **not** redundant with Click's `Choice` validation: Click only validates CLI args, but the preset can also arrive via `gemini.style.preset` from a YAML config file or from the `config init` wizard, neither of which goes through Click's `Choice` parser. `validate_preset` is therefore the single point of truth that catches typos regardless of source. Truncate `style_text`, overlays, and `angle` via `truncate_with_warning`. Render the three new slots.
- **Whitespace contract for placeholders**: each new placeholder renders with its own leading `\n` when populated, and to the empty string `""` when not. The slot for `{speaker_adjustments_block}` sits immediately after the speaker2 personality bullet (no trailing blank line in the empty case so the existing blank line between `Host personalities:` and `Instructions:` is preserved byte-for-byte). The `test_no_flags_byte_identical` fixture catches any off-by-one whitespace.
- Update `generate_dialogue()` (`llm_summarizer.py:383`):
  - Read new values via `gemini_cfg.get("style", {}).get("preset")`, `gemini_cfg.get("style", {}).get("text")`, `gemini_cfg.get("style", {}).get("angle")`, `gemini_cfg.get("speaker1", {}).get("style_overlay")`, `gemini_cfg.get("speaker2", {}).get("style_overlay")`.
  - Pass them to `_build_prompt` as kwargs.
  - **Do not** mutate `gemini_cfg["speaker1"]["personality"]` or `gemini_cfg["speaker2"]["personality"]`.

### Step 3 — `research.py`: angle injection in round 1 only

- Add an `{angle_block}` placeholder to `_ROUND_1_PROMPT` only (`research.py:38`). Renders empty, OR `"\nAngle to emphasize: {angle}. Prioritise sources and notes that illuminate this angle.\n"`.
- Do NOT add the placeholder to `_ROUND_N_PROMPT` (`research.py:50`). Round N receives the angle implicitly through `{previous_notes}` content.
- Update `conduct_research()` to accept a keyword-only `angle: str | None = None`. Truncate via `truncate_with_warning`. Thread it through the round-1 prompt formatting; ignore it in round N≥2.

### Step 4 — `cli.py`: new options + wiring

- Add Click options on the `run` command (`cli.py` around lines 265–362). All long-form only:
  - `--preset NAME` — Click `type=click.Choice([*STYLE_PRESETS.keys(), "none"])` with `case_sensitive=False`. Explicit `.keys()` makes intent unambiguous (so a future refactor of `STYLE_PRESETS` from `dict` to `list[tuple[str, str]]` won't silently break the call). Click normalizes the input to its matching choice's case *before* invoking the callback, so `validate_preset` always receives the canonical lowercase preset name (or the sentinel `"none"`) — no `.lower()` defensive call needed inside `validate_preset`. The literal sentinel `none` is the CLI-side way to disable a configured `gemini.style.preset`; `validate_preset("none")` returns `None`. Click handles the unknown-name error automatically with the available choices listed in the message.
  - `--style TEXT` — free text.
  - `--speaker1-style TEXT` and `--speaker2-style TEXT`.
  - `--angle TEXT`.
- After loading config, mutate the in-memory `gemini_cfg` to override with CLI values when present (same pattern as the duration override at `cli.py:447–457`):
  - `gemini_cfg.setdefault("style", {})["preset"] = preset` (and similar for `text`, `angle`).
  - `gemini_cfg.setdefault("speaker1", {})["style_overlay"] = speaker1_style` (and similar for `speaker2`).
  - **Never** touch `gemini_cfg["speaker1"]["personality"]` or `gemini_cfg["speaker2"]["personality"]`.
- Thread `angle` into the `conduct_research(...)` call site (approximately `cli.py:557`) as a keyword argument.
- Update the `run --help` epilog so new options appear as a clearly labelled group ("Style & angle (optional)").
- Empty-string handling: explicit `--style ""` from the CLI sets `gemini_cfg["style"]["text"] = ""`, which `_build_prompt` treats as "no style text" (same as `None`). Documented in the option's help text. Same for `--angle ""`, `--preset ""`, `--speaker1-style ""`, `--speaker2-style ""`.

### Step 5 — Config wizard + example file

- `cli.py::config init` wizard (`cli.py:683–797` per Critic's read; verify exact lines during implementation):
  - **Anchor**: insert the new prompts **after the existing `speaker2.personality` prompt** (the wizard does not currently have a duration prompt — fixing the earlier plan's mis-anchor).
  - Prompt order: (a) "Default preset (one of: casual, academic, humorous, debate, vulgarized; blank to skip):", (b) "Default style guidance (free text, blank to skip):", (c) "Default episode angle (blank to skip):".
  - Write only non-empty values, all under the new `gemini.style` block.
- `config.example.yaml`: add the new `gemini.style` block as a top-level sibling of `dialogue` / `tts_style` with all three keys commented out and a one-line docstring per key. Also add `style_overlay: ""` (commented out, with a one-line docstring) under each of `gemini.speaker1` and `gemini.speaker2`. Add a brief comment near the top of the `gemini.style` block explaining precedence: "CLI flags override these values for one run; preset name must match a key in tts_podcast.style_presets.STYLE_PRESETS; unknown names error at parse time."

### Step 6 — Tests

- Implement criteria #17–#20. Keep tests isolated: never hit the network; mock `Gemini.models.generate_content`. Use `caplog` for truncation-warning assertions.
- **Generate the fixture for `test_no_flags_byte_identical`**: run the **current** (pre-refactor) `_build_prompt` once with a fixed input, write the output to `tests/fixtures/dialogue_prompt_no_overlay.txt`, commit. The post-refactor code with all new params at default must produce the same bytes.
- **Fixture regeneration workflow** (for legitimate future edits to `_SYSTEM_PROMPT_TEMPLATE`): commit a small helper script at `tests/fixtures/regen_dialogue_prompt.py` exposing a `regen()` function that calls `_build_prompt` with the same fixed input used to seed the fixture, and overwrites `tests/fixtures/dialogue_prompt_no_overlay.txt`. Invoked via `uv run python -m tests.fixtures.regen_dialogue_prompt`. When `_SYSTEM_PROMPT_TEMPLATE` is intentionally changed (typo fix, wording tweak), the workflow is: (1) edit the template, (2) run the regen script, (3) `git diff tests/fixtures/dialogue_prompt_no_overlay.txt` and review, (4) commit the fixture alongside the template change. Documented in a one-paragraph note in `CLAUDE.md` under "Conventions" so the fixture isn't a "haunted" file for future contributors.
- **Pytest collection guards** (per Critic iter-2 Major #1):
  - Create an empty `tests/fixtures/__init__.py` so `python -m tests.fixtures.regen_dialogue_prompt` resolves. (Without this, the `-m` invocation fails with `No module named 'tests.fixtures'`.)
  - Create (or extend) `tests/conftest.py` with `collect_ignore_glob = ["fixtures/*"]` so pytest never collects anything under `tests/fixtures/` regardless of filename. This guards against a future contributor accidentally naming a helper `test_regen.py` (which pytest would collect and execute, overwriting the snapshot fixture on every test run and silently neutralizing the byte-identical guarantee).
  - The helper script name **must not** start with `test_` — kept as `regen_dialogue_prompt.py` so it doesn't match pytest's default `python_files = ["test_*.py", "*_test.py"]` pattern even in the absence of the `collect_ignore_glob` line; the conftest line is belt-and-braces.
- For `test_cli.py`, use Click's `CliRunner` and `unittest.mock.patch` at the module boundary (`patch("tts_podcast.cli.generate_dialogue")`, `patch("tts_podcast.cli.conduct_research")`, `patch("tts_podcast.cli.scrape_urls")`, `patch("tts_podcast.cli.generate_audio_chunks")`).

### Step 7 — Docs

- `CLAUDE.md` (project): add a paragraph under "Architecture" → "Key invariants & non-obvious behaviour" explaining: (a) style/angle injection points, (b) the hard invariant that `speaker[N].personality` is never mutated so TTS voice acting is unaffected, (c) the angle's round-1-only injection rule.
- `README.md`: add the five flags to the usage examples block, with at least one example combining `--preset` and `--style`.

## Risks and mitigations (revised)

| Risk | Severity | Mitigation |
|---|---|---|
| Per-speaker overlay leaks into TTS preamble (the Critic's critical #1). | HIGH | Hard invariant (criterion #16) + test `test_tts_preamble_unaffected_by_speaker_overlay`. Implementation rule: CLI writes overlay to `style_overlay` key only; `_build_tts_prompt` continues to read `personality`. Zero shared state between the two paths. |
| Backward compatibility regression — existing prompt-substring tests break. | MEDIUM | Empty-string defaults on all new template slots + `test_no_flags_byte_identical` snapshot test (criterion #9, #17) against a frozen fixture. |
| Free-text steering signal drowns the article content. | MEDIUM | 500-char cap per free-text field + `truncate_with_warning`. Aggregate ceiling (~2000 chars across four fields) is reasonable for a multi-thousand-token prompt window. |
| Preset list drift over time. | MEDIUM | Small initial set (5); enumeration test in `test_style_presets.py` locks the keys; deprecation requires bumping the major preset name and updating tests in the same commit. |
| Round-1 angle injection biases research too narrowly. | LOW | Phrasing is "Prioritise … illuminate this angle", not "exclude everything else". Keep the existing "complementary angles" guidance intact. Round N≥2 explicitly does NOT re-inject, so gap analysis stays neutral (criterion #6 + #18). |
| User confusion between `--preset` and `--style`. | LOW | Help text spells out the two surfaces clearly; preset accepts a known name only; style is always free text. README example shows both combined. |
| `gemini.style.preset = "academic"` conflicts with `tts_style.scene = "casual studio"`. | LOW | These operate at different stages (dialogue content vs voice acting context) and are not contradictory at the model level. Document independence in `config.example.yaml`. No code-level conflict resolution needed. |
| User wants to discover per-speaker overlay keys but they're not in the wizard. | LOW | Keys appear in `config.example.yaml` as commented-out lines with one-line docstrings (criterion #11, #13). |
| `--angle` with a `-s` search input — unclear interaction. | LOW | Tested directly via `test_angle_plus_search_input` (criterion #18). Both appear in the round-1 prompt; the angle steers the research, the search query defines the topic. |
| Migration burden from old `gemini.dialogue.style` (proposed in iteration 1) to `gemini.style.{preset,text,angle}`. | NONE | First release of all five new keys — no existing user has them set, no migration code needed. Noted here so future Critic/Architect rounds don't re-raise it. |

## Verification steps (revised)

1. `uv run pytest tests/ -q` — full suite green (existing + new).
2. `uv run pytest tests/test_style_presets.py tests/test_llm_summarizer.py tests/test_research.py tests/test_cli.py tests/test_tts_generator.py -v` — new and updated tests pass.
3. `uv run ruff check src/ tests/` — clean.
4. **[rev]** Manual smoke (deterministic prompt inspection): add a permanent `logger.debug("Dialogue prompt (%d chars):\n%s", len(prompt), prompt)` line in `_build_prompt` immediately before the return statement. **Dump format**: the entire prompt as a single log record, prefixed with its char count for quick eyeballing. The dialogue prompt is typically a few KB — well within log handler defaults — so a full dump is acceptable and most useful for debugging. Run `uv run tts-podcast run -n -A --verbose --preset academic --angle "regulatory impact" --speaker1-style "more cautious than usual" https://example.com/some-article 2>&1 | grep -B1 -A60 "Dialogue prompt"`. Confirm the captured prompt contains all four injection points in the documented order. Leave the debug log in place permanently (helpful for future debugging; emitted only when log level ≤ DEBUG, so it doesn't pollute normal runs).
5. Run `uv run tts-podcast run --help` and verify the new flags appear in the "Style & angle (optional)" group with the documented help text.
6. **[rev]** Backward-compat check: `uv run pytest tests/test_llm_summarizer.py::test_no_flags_byte_identical -v` — the post-refactor prompt with all new params at default matches the pre-refactor fixture byte-for-byte.
7. **[rev]** TTS invariant check: `uv run pytest tests/test_tts_generator.py::test_tts_preamble_unaffected_by_speaker_overlay -v` — TTS preamble is unaffected by `--speaker1-style`.

## Files touched (summary)

- **New**: `src/tts_podcast/style_presets.py`, `tests/test_style_presets.py`, `tests/test_cli.py`, `tests/fixtures/__init__.py` (empty), `tests/fixtures/dialogue_prompt_no_overlay.txt`, `tests/fixtures/regen_dialogue_prompt.py`, `tests/conftest.py` (or extend existing) with `collect_ignore_glob = ["fixtures/*"]`.
- **Modified**: `src/tts_podcast/cli.py`, `src/tts_podcast/llm_summarizer.py`, `src/tts_podcast/research.py`, `config.example.yaml`, `tests/test_llm_summarizer.py`, `tests/test_research.py`, `tests/test_tts_generator.py` (or extend existing — check for presence), `CLAUDE.md`, `README.md`.
- **Unchanged**: `src/tts_podcast/tts_generator.py` (hard invariant, criterion #16), `src/tts_podcast/audio_exporter.py`, `src/tts_podcast/models.py`, `src/tts_podcast/web_scraper.py`, `src/tts_podcast/local_loader.py`, `src/tts_podcast/token_tracker.py`.

## ADR

**Decision:** Add `--preset NAME`, `--style TEXT`, `--speaker1-style TEXT`, `--speaker2-style TEXT`, and `--angle TEXT` as long-form CLI options (with config-file mirrors under a new `gemini.style` block and per-speaker `style_overlay` keys). Preset is a Click `Choice` from a 5-key dict; style/angle/overlays are free text capped at 500 chars. Angle injects into the dialogue prompt and the round-1 research prompt only. Per-speaker overlays render in a dedicated `Episode-specific adjustments:` prompt block; `speaker[N].personality` is never mutated.

**Drivers:** (1) Author ergonomics for episode-level style nudges. (2) Steering-signal coherence — each prompt slot owns one purpose. (3) Hard scope ceiling — no TTS-side changes.

**Alternatives considered:** Option B (free text only — dropped because the user wants preset-driven repeatability), Option C (declarative per-attribute YAML — dropped as over-engineering), Option D (reuse `tts_style.scene` — dropped because it would conflate voice-acting context with dialogue content, breaking backward compat for existing users).

**Why chosen:** Option A's split-flag design eliminates the parser ambiguity that the first iteration carried (first-token preset matching), makes the `--preset` validation a one-line `Choice`, and keeps each surface single-purpose. The per-speaker overlay path was redesigned to write to a dedicated `style_overlay` key (never touching `personality`), which closes the TTS-leak hole the Critic identified. Angle is constrained to round 1 to avoid the tunnel-vision loop the Architect flagged.

**Consequences:**
- The 5-preset list becomes a stable contract; adding presets requires a code change and a test update.
- Users who want preset-driven repeatability commit a snippet to YAML once; users who want one-shot nudges use the CLI.
- The dialogue prompt grows by up to ~2000 chars when all free-text fields are maxed (well within Gemini's text-gen context window).
- The TTS preamble stays exactly as it is today — voice acting is decoupled from content steering, by design and by test (`test_tts_preamble_unaffected_by_speaker_overlay`).
- A new fixture file `tests/fixtures/dialogue_prompt_no_overlay.txt` lands in the repo as a byte-identical baseline; the prompt template can only change going forward by regenerating the fixture deliberately.

**Follow-ups (deferred, out of scope for this plan):**
- A future Option C-style declarative tone schema could augment the overlay if free-text proves insufficient.
- Telemetry: if/when token usage spikes due to maxed-out free-text fields, consider lowering the cap or adding a config-level aggregate cap.
- If users frequently set the same preset+style combination in CLI, surface a "named profile" feature (e.g. `--profile my-investigative-style`) — but only after observed demand.

## Changelog (revisions applied during consensus)

**Iteration 2 final sub-revision** — Applied after Critic re-review of iteration 2 (`APPROVE_WITH_IMPROVEMENTS`):

- **(Major #1) Pytest collection guards added**: `tests/fixtures/__init__.py` (empty package marker) so the regen script's `-m` invocation resolves; `tests/conftest.py` (or extension) with `collect_ignore_glob = ["fixtures/*"]` to guarantee pytest never collects anything under `tests/fixtures/` regardless of filename — guards against a future contributor silently neutralizing the snapshot fixture by naming a helper `test_*.py`.
- **(Major #2) `Stylistic guidance:` block moved inside `Instructions:`**. The original placement had it outside as a free-floating section between `Episode-specific adjustments:` and `Instructions:`, while `Episode angle:` sat inside `Instructions:` — asymmetric. New layout: all qualitative steering (`Stylistic guidance:` + `Episode angle:`) lives inside the `Instructions:` block as a coherent directive list. Criterion #14, Step 2, and the order test updated accordingly. Rationale added: baseline personality precedes the per-episode adjustment so the LLM reads the override as a delta.
- **(Minor #1) Click `Choice` spelling**: `[*STYLE_PRESETS.keys(), "none"]` (explicit `.keys()`) so a future `STYLE_PRESETS` refactor to `list[tuple[str, str]]` cannot silently break the choice list.
- **(Minor #2) Click case-normalization documented**: Click lowercases inputs against the choice list before invoking the callback, so `validate_preset` doesn't need a defensive `.lower()`. Documented in Step 4 to head off paranoia-driven double-normalization.
- **(Minor #3) Verification step 4 dump format pinned**: full prompt as one log record, prefixed with char count for eyeballing. Documented as a permanent DEBUG-level log line.
- **(Ambiguity)** Step 2 now states explicitly that `\n` sequences in the rendered-fragment strings are real newline characters (interpreted), not literal `\\n` byte pairs.

**Iteration 2 sub-revision** — Applied after Architect re-review of iteration 2:

- **Criterion #10 fix:** `--preset ""` is unimplementable with `click.Choice` (rejects empty strings). Added sentinel `--preset none` (mapped to `None` by `validate_preset`) as the CLI-side way to disable a configured preset. Empty-string semantics retained for the four free-text flags. Step 4 updated to register `none` as a choice in the Click `Choice`.
- **Step 2 rephrase:** `validate_preset` is not defence-in-depth — it's the only validation for the YAML config and `config init` wizard paths (Click only validates CLI args). Rephrased so future implementers don't delete it as dead code.
- **Step 6 fixture regen workflow:** Added a `tests/fixtures/regen_dialogue_prompt.py` helper script and documented the regeneration procedure in `CLAUDE.md`. The fixture is no longer a "haunted" file when `_SYSTEM_PROMPT_TEMPLATE` is intentionally edited.
- **Polish:** Renamed `test_angle_NOT_in_round_n_prompt` → `test_angle_header_NOT_re_injected_in_round_n_prompt` to make explicit that the test forbids the re-injected header line, not the angle text appearing via `{previous_notes}`. Added a Step 2 whitespace-contract note for the new placeholders. Added a "NONE" migration-burden row to the risks table.

**Iteration 1 → Iteration 2** — Applied after Architect and Critic review:

From Architect:
1. Split `--style PRESET_OR_TEXT` into `--preset NAME` (Click `Choice`) + `--style TEXT` (free text). Removed the first-token-matching resolution rule entirely (was criterion #4).
2. Angle injection restricted to `_ROUND_1_PROMPT`. `_ROUND_N_PROMPT` no longer receives the angle (was test #17 contradiction).
3. Per-speaker overlay rendered in a dedicated `Episode-specific adjustments:` block, not inlined into `Host personalities:` (was criterion #5).
4. Criterion #14 (was #13) reformulated as a placement/order test, not a byte-budget test.
5. Dropped `-y` short form for `--style` and `-a` for `--angle` — all new flags are long-form only to avoid `-y`/`-a`/`-A` collisions.
6. `config.example.yaml` and the wizard pinned to concrete behaviour (per-speaker overlays appear as commented-out keys; wizard anchors after `speaker2.personality`, not after a non-existent duration prompt).

From Critic:
1. **(Critical)** Per-speaker overlay path redesigned: writes to a new `style_overlay` key, never to `personality`. Criterion #16 codifies this as a hard invariant; `test_tts_preamble_unaffected_by_speaker_overlay` enforces it.
2. **(Critical)** Option D (reuse `gemini.tts_style.scene`) added to the alternatives list and explicitly invalidated with a defensible rationale.
3. Principle #4 reworded from a (wrong) byte-budget framing to a (correct) "free-text injections must not dominate the prompt" framing.
4. Verification step 4's "or" eliminated — now specifies a single concrete debug-log path with grep.
5. Added `test_no_flags_byte_identical` snapshot test (criterion #9, #17).
6. Added `test_angle_in_dialogue_prompt_without_research` and `test_angle_plus_search_input` (criteria #17, #18).
7. Schema relocated from `gemini.dialogue.{style,angle}` to a new `gemini.style.{preset,text,angle}` block to keep qualitative steering separate from `gemini.dialogue`'s quantitative tuning knobs.
8. Preset count reconciled to **5** (was inconsistent: 5 in the summary, 7 in Step 1).
9. Empty-string semantics for all five CLI flags explicitly defined (criterion #10).
10. Wizard anchor corrected — there is no duration prompt in `config init`; anchor is after `speaker2.personality`.
