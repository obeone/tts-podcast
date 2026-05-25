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
```

### Key flags

| Flag | Description |
|---|---|
| `-f, --file FILE` | Local document to include (repeatable). Supports `.txt`, `.md`, `.html`, `.pdf`. |
| `-s, --search QUERY` | Web-search query to seed the podcast (repeatable). Research auto-bumped to 1 if search-only. |
| `-R, --research N` | Number of Google-Search-grounded research rounds (default 0). |
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
