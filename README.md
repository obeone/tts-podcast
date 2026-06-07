# tts-podcast

Turn any article URL into a two-voice podcast via Google Gemini TTS, with
optional iterative Google-Search-grounded research enrichment.

`tts-podcast` is a CLI tool that takes one or more article URLs, local
documents, or web-search queries, optionally enriches them with iterative
web research, generates a conversational dialogue between two hosts using
Gemini, and synthesises an MP3 (or WAV) using Gemini's multi-speaker TTS.

## Features

- **Any URL → podcast** — feed one or several article URLs; the tool
  handles scraping, dialogue generation, and audio export.
- **Local documents** (`-f FILE`) — include `.txt`, `.md`, `.html`, or
  `.pdf` files directly; no network request is made for them.
- **Web-search queries** (`-s QUERY`) — pass a natural-language topic;
  the research stage investigates it via Google Search grounding.
- **Iterative research** (`--research N`) — runs *N* sequential Gemini
  rounds using the [`google_search`][grounding] grounding tool. Each round
  builds on the previous round's findings, drilling into gaps and
  unanswered questions.
- **Multi-voice TTS** — two distinct Gemini voices with configurable
  personalities, scene, and delivery cues.
- **Named voice duos** — five built-in pairings (`contrast` by default,
  plus `warm`, `explorer`, `journalist`, `debate`) selectable per run with
  `--duo`, or define your own; pick from all 30 prebuilt Gemini voices.
  See [Voice duos](#voice-duos).
- **Report folder** — generates `overview.md`, `sources.md`, `script.md`,
  `research.md`, and `summary.md` alongside the audio file.
- **Token & cost tracking** — accumulates token usage per model and
  estimates cost based on configurable per-model pricing.

[grounding]: https://ai.google.dev/gemini-api/docs/google-search

## Install

```bash
uv sync
```

`ffmpeg` is required for audio export (skip if you only use `--no-audio` /
`--dry-run`):

```bash
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Debian/Ubuntu
```

## Configure

```bash
uv run tts-podcast config init
```

Then export your Gemini API key:

```bash
export GEMINI_API_KEY=<your key>
```

The default config path is `$XDG_CONFIG_HOME/tts-podcast/config.yaml`
(typically `~/.config/tts-podcast/config.yaml`). See
[`config.example.yaml`](config.example.yaml) for the full schema.

## Usage

```bash
# Single URL, no research
uv run tts-podcast run https://blog.example.com/article

# Multiple URLs with two rounds of complementary research
uv run tts-podcast run -R 2 \
    https://blog.example.com/a \
    https://blog.example.com/b

# Local document (PDF, txt, md, html) — no network request
uv run tts-podcast run -n -f paper.pdf

# Web-search query — research auto-bumped to 1 if no other inputs
uv run tts-podcast run -n -s "agentic AI memory systems"

# Mixed: URL + local file + search query in one episode
uv run tts-podcast run -n https://blog.example.com/article \
    -f notes.md -s "related follow-up topic"

# Preview the dialogue without calling TTS
uv run tts-podcast run -n https://blog.example.com/article

# Generate the script + report but skip audio synthesis
uv run tts-podcast run -A https://blog.example.com/article

# Style & angle: nudge tone via a preset + free text, focus on one angle
uv run tts-podcast run -R 1 \
    --preset academic \
    --style "extra rigorous, French academic feel" \
    --angle "the regulatory implications" \
    https://blog.example.com/article

# Per-episode speaker overlay (TTS voice acting stays unchanged)
uv run tts-podcast run --speaker1-style "more skeptical than usual" \
    --speaker2-style "extra warm and forgiving" \
    https://blog.example.com/article

# List the available voice duos, then run one (debate pairs well with --preset debate)
uv run tts-podcast duos
uv run tts-podcast run --duo debate --preset debate https://blog.example.com/article
```

### Key flags

| Flag | Description |
|---|---|
| `-f, --file FILE` | Local document to include (repeatable). Supports `.txt`, `.md`, `.html`, `.pdf`. |
| `-s, --search QUERY` | Web-search query to seed the podcast (repeatable). Research auto-bumped to 1 if search-only. |
| `-R, --research N` | Number of Google-Search-grounded research rounds (default 0). |
| `--duo NAME` | Named voice duo for both speakers (`warm`, `contrast`, `explorer`, `journalist`, `debate`). Overrides `gemini.default_duo` and legacy `speakerN` blocks. See `tts-podcast duos`. |
| `-n, --dry-run` | Print dialogue to stdout, no TTS. |
| `-A, --no-audio` | Generate script + report only. |
| `-o, --output-dir DIR` | Output directory (overrides config). |
| `--no-report` | Skip the report folder. |
| `-v, --verbose` | Enable DEBUG logging. |
| `--preset NAME` | Named style preset for the dialogue (`casual`, `academic`, `humorous`, `debate`, `vulgarized`, or `none` to disable a configured default). |
| `--style TEXT` | Free-text style guidance (capped at 500 chars). Composes with `--preset`. |
| `--speaker1-style TEXT` / `--speaker2-style TEXT` | Per-episode style overlay for the given speaker. The baseline personality (and TTS voice acting) stays unchanged. |
| `--angle TEXT` | Episode angle. Steers the dialogue prompt and the first research round only. |

Run `uv run tts-podcast run --help` for the full list.

## Voice duos

A *duo* bundles both speakers — name, prebuilt Gemini voice, and baseline
personality — under a single slug, so you switch the whole pairing at once
instead of editing `speaker1` / `speaker2` by hand. List them anytime:

```bash
uv run tts-podcast duos
```

### Built-in duos

| Slug | Speaker 1 | Speaker 2 | Vibe |
|---|---|---|---|
| `contrast` *(default)* | Puck (Upbeat) | Kore (Firm) | High timbre contrast — Google's own multi-speaker pairing |
| `warm` | Sulafat (Warm) | Achird (Friendly) | Accessible, mainstream feel |
| `explorer` | Fenrir (Excitable) | Sadaltager (Knowledgeable) | Excited explorer + calm expert; vulgarisation-friendly |
| `journalist` | Zephyr (Bright) | Algieba (Smooth) | Fast-paced tech-journalism feel |
| `debate` | Laomedeia (Upbeat) | Algenib (Gravelly) | Opposing viewpoints — techno-optimist vs skeptic (pair with `--preset debate`) |

> Gemini doesn't officially document voice gender; the pairings are curated
> from each voice's [official descriptor][voices] plus community reports.
> Audition voices in [Google AI Studio][voices] before committing.

[voices]: https://ai.google.dev/gemini-api/docs/speech-generation

### Selecting a duo

```bash
# Pick a duo for one run (overrides config)
uv run tts-podcast run --duo journalist https://blog.example.com/article

# Opposing viewpoints, structured as a debate
uv run tts-podcast run --duo debate --preset debate https://blog.example.com/article
```

Set a persistent default in `config.yaml`:

```yaml
gemini:
  default_duo: contrast
```

### Custom duos

Define your own under `gemini.duos`; they're merged over the built-ins
(same slug overrides one, a new slug adds one). Match each voice's
descriptor to the personality for the best result:

```yaml
gemini:
  default_duo: my_duo
  duos:
    my_duo:
      description: "my custom pairing"
      speaker1:
        name: Robin
        voice: Laomedeia   # Upbeat
        personality: "techno-optimist; champions the upside"
      speaker2:
        name: Sasha
        voice: Algenib     # Gravelly
        personality: "hard-nosed skeptic; probes risks and costs"
```

**Resolution precedence:** `--duo` > `gemini.default_duo` >
legacy `gemini.speaker1` / `speaker2` blocks > built-in `contrast`. A config
that defines only the legacy `speakerN` blocks keeps working unchanged.

## Output layout

```
<output_dir>/
├── <stem>.mp3
└── tts_<stem>/
    ├── overview.md       # metadata, link breakdown, token/cost summary
    ├── sources.md        # per-source content (title, URL, summary, full text)
    ├── script.md         # full two-host dialogue
    ├── research.md       # only when --research >= 1
    └── summary.md        # synthetic reference sheet with categorised links
```

The stem combines the first URL's hostname, a 6-char digest of the URL
list, and today's date, e.g. `arxiv.org-a1b2c3-2026-05-23.mp3`.

## Research cost note

Each `--research` round is a separate Gemini call with Google Search
grounding enabled, which adds search overhead to the standard input
token cost. The tool logs the cumulative cost after each round so you
can watch the bill while iterating.

## Testing

```bash
uv run pytest tests/ -v
```

## License

MIT
