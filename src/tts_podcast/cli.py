"""
URL → two-voice podcast CLI.

Commands
--------
    tts-podcast --version                          Print the installed version and exit.
    tts-podcast run URL [URL ...] [OPTIONS]        Run the full pipeline.
    tts-podcast config init                        Interactive configuration wizard.
    tts-podcast config show [--resolve]            Display the current configuration file.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import click
import coloredlogs
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.syntax import Syntax

from tts_podcast.audio_exporter import encode_audio, export_audio
from tts_podcast.config import ConfigError, load_config
from tts_podcast.duos import BUILTIN_DUOS, DEFAULT_DUO, describe_duos, resolve_duo
from tts_podcast import link_follower
from tts_podcast.link_extractor import extract_links
from tts_podcast.llm_summarizer import generate_dialogue
from tts_podcast.local_loader import load_local_files
from tts_podcast.models import Source
from tts_podcast.report_generator import generate_report
from tts_podcast.research import conduct_research
from tts_podcast.style_presets import STYLE_PRESETS
from tts_podcast.token_tracker import TokenTracker
from tts_podcast.tts_generator import generate_audio_chunks
from tts_podcast.user_agent import BROWSER_USER_AGENT
from tts_podcast.web_scraper import scrape_urls

logger = logging.getLogger(__name__)

console = Console(stderr=False)


def _xdg_config_home() -> Path:
    """Return ``$XDG_CONFIG_HOME`` or its default (``~/.config``)."""
    value = os.environ.get("XDG_CONFIG_HOME")
    return Path(value) if value else Path.home() / ".config"


_DEFAULT_CONFIG = _xdg_config_home() / "tts-podcast" / "config.yaml"

# The 30 prebuilt Gemini TTS voices, with their official one-word descriptor.
# See https://ai.google.dev/gemini-api/docs/speech-generation for previews.
# Google does not document gender; audition in AI Studio before committing.
_GEMINI_VOICES = [
    "Zephyr",       # Bright
    "Puck",         # Upbeat
    "Charon",       # Informative
    "Kore",         # Firm
    "Fenrir",       # Excitable
    "Leda",         # Youthful
    "Orus",         # Firm
    "Aoede",        # Breezy
    "Callirrhoe",   # Easy-going
    "Autonoe",      # Bright
    "Enceladus",    # Breathy
    "Iapetus",      # Clear
    "Umbriel",      # Easy-going
    "Algieba",      # Smooth
    "Despina",      # Smooth
    "Erinome",      # Clear
    "Algenib",      # Gravelly
    "Rasalgethi",   # Informative
    "Laomedeia",    # Upbeat
    "Achernar",     # Soft
    "Alnilam",      # Firm
    "Schedar",      # Even
    "Gacrux",       # Mature
    "Pulcherrima",  # Forward
    "Achird",       # Friendly
    "Zubenelgenubi",# Casual
    "Vindemiatrix", # Gentle
    "Sadachbia",    # Lively
    "Sadaltager",   # Knowledgeable
    "Sulafat",      # Warm
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _slugify(text: str, limit: int = 40) -> str:
    """
    Convert arbitrary text to a URL-safe slug.

    Parameters
    ----------
    text : str
        Input string.
    limit : int, optional
        Maximum output length, by default 40.

    Returns
    -------
    str
        Lowercase alphanumeric slug, dashes replacing non-alphanumeric runs.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:limit] or "search"


def _make_search_source(query: str) -> Source:
    """
    Return a synthetic :class:`Source` representing a web-search topic.

    Parameters
    ----------
    query : str
        Natural-language search query.

    Returns
    -------
    Source
        Source with ``kind="search"`` and ``scraped_ok=True``.
    """
    return Source(
        url=f"search://{query}",
        title=f"Web search: {query}",
        summary=f"Topic to investigate via web research: {query}",
        full_text=f"Topic to investigate via web research: {query}",
        scraped_ok=True,
        kind="search",
    )


def _check_ffmpeg() -> None:
    """
    Verify that ffmpeg is available in PATH and abort with a helpful message if not.

    Raises
    ------
    SystemExit
        Exits with code 1 if ffmpeg cannot be found.
    """
    if shutil.which("ffmpeg") is None:
        click.echo(
            "[ERROR] ffmpeg not found in PATH. Install it to enable audio export.\n"
            "  macOS:         brew install ffmpeg\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  Windows:       https://ffmpeg.org/download.html",
            err=True,
        )
        sys.exit(1)


def _setup_logging(verbose: bool) -> None:
    """Configure coloredlogs for the root logger."""
    level = "DEBUG" if verbose else "INFO"
    coloredlogs.install(
        level=level,
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def _make_progress(disable: bool) -> Progress:
    """
    Build a rich Progress instance with a consistent column layout.

    Parameters
    ----------
    disable : bool
        When ``True``, the progress bar renders nothing (all output is
        suppressed).

    Returns
    -------
    rich.progress.Progress
        Configured progress instance.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        disable=disable,
    )


def _resolve_config_path(config_path: str | None) -> str:
    """
    Return the effective config file path, falling back to the XDG default.

    Parameters
    ----------
    config_path : str | None
        Path supplied via ``--config``, or ``None`` to use the default.

    Returns
    -------
    str
        Resolved path to an existing config file.

    Raises
    ------
    SystemExit
        Exits with code 1 if no config file can be found.
    """
    if config_path is not None:
        return config_path
    if _DEFAULT_CONFIG.exists():
        return str(_DEFAULT_CONFIG)
    click.echo(
        f"[ERROR] No --config provided and no default config found at {_DEFAULT_CONFIG}.\n"
        f"  Run `tts-podcast config init` to create one.",
        err=True,
    )
    sys.exit(1)


def _build_output_stem(identifiers: list[str]) -> str:
    """
    Build the filename stem for an episode from a mixed list of identifiers.

    Derives a human-readable label from the first identifier (hostname for
    URLs, filename stem for ``file://`` URIs, slugified query for
    ``search://`` URIs), appends a 6-character SHA-1 digest of all
    identifiers for collision resistance, and suffixes today's ISO date.

    Parameters
    ----------
    identifiers : list[str]
        URLs, ``file://`` URIs, or ``search://`` URIs (any mix).

    Returns
    -------
    str
        Filename stem without extension, e.g. ``"arxiv.org-a1b2c3-2026-05-23"``.
    """
    first = identifiers[0]
    if first.startswith(("http://", "https://")):
        parsed = urlparse(first)
        host = parsed.netloc or "podcast"
        if host.startswith("www."):
            host = host[4:]
        label = host
    elif first.startswith("file://"):
        label = Path(first[len("file://"):]).stem or "file"
    elif first.startswith("search://"):
        label = _slugify(first[len("search://"):])
    else:
        label = Path(first).stem or "podcast"
    digest = hashlib.sha1("\n".join(identifiers).encode("utf-8")).hexdigest()[:6]
    return f"{label}-{digest}-{date.today().isoformat()}"


def _resolve_output_path(
    output_file: str | None,
    output_dir: str,
    stem: str,
    default_fmt: str,
) -> tuple[Path, str]:
    """
    Resolve the audio destination path and format from ``--output``.

    When *output_file* is ``None`` the auto-generated ``<stem>.<fmt>`` name is
    placed inside *output_dir*.  An explicit *output_file* with a directory
    component (or an absolute path) is honoured verbatim; a bare filename
    lands inside *output_dir*.  The format is taken from the file extension
    when present, otherwise *default_fmt*.

    Parameters
    ----------
    output_file : str | None
        Value of ``--output`` (never ``"-"``; stdout is handled by the caller).
    output_dir : str
        Directory used for the auto-generated name and for bare filenames.
    stem : str
        Filename stem for the auto-generated name.
    default_fmt : str
        Fallback format when the path carries no extension.

    Returns
    -------
    tuple[Path, str]
        The resolved output path and the format string.
    """
    if not output_file:
        return Path(output_dir) / f"{stem}.{default_fmt}", default_fmt

    candidate = Path(output_file)
    if candidate.is_absolute() or candidate.parent != Path("."):
        out_path = candidate
    else:
        out_path = Path(output_dir) / candidate
    fmt = candidate.suffix.lstrip(".").lower() or default_fmt
    return out_path, fmt


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(
    None,
    "--version",
    package_name="tts-podcast",
    prog_name="tts-podcast",
)
def cli() -> None:
    """URL → two-voice podcast: turn any article into a Gemini-TTS podcast MP3."""


# ---------------------------------------------------------------------------
# `run` command
# ---------------------------------------------------------------------------

@cli.command("run")
@click.argument("inputs", nargs=-1, required=False, metavar="URL_OR_FILE...")
@click.option(
    "-f", "--file",
    "files",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Local document(s) to include (txt, md, html, pdf). Repeatable.",
)
@click.option(
    "-s", "--search",
    "search_queries",
    multiple=True,
    help="Web search query/queries to seed the podcast. Repeatable.",
)
@click.option(
    "-c", "--config",
    "config_path",
    required=False,
    default=None,
    type=click.Path(dir_okay=False),
    help=f"Path to the YAML configuration file. Defaults to {_DEFAULT_CONFIG}.",
)
@click.option(
    "-R", "--research",
    "research_rounds",
    type=int,
    default=None,
    help=(
        "Number of Google-Search-grounded research rounds to run before "
        "generating the dialogue.  0 disables research entirely.  "
        "Defaults to research.rounds_default in config (or 0 if absent)."
    ),
)
@click.option(
    "-d", "--duration",
    "target_duration",
    type=float,
    default=None,
    help=(
        "Target episode duration in minutes.  Overrides "
        "gemini.dialogue.target_duration_minutes from the config.  "
        "Implicit min/max are 70%% and 150%% of this value; pass them "
        "explicitly in the config to override."
    ),
)
@click.option(
    "-o", "--output-dir",
    "output_dir_override",
    default=None,
    type=click.Path(file_okay=False),
    help="Directory where the podcast file is written. Overrides config output.dir.",
)
@click.option(
    "-O", "--output",
    "output_file",
    default=None,
    help=(
        "Output file path (or bare name) for the audio. Use '-' to stream it "
        "to stdout. A bare name lands in --output-dir; a path with a directory "
        "component is used as-is. Format is inferred from the extension, "
        "otherwise output.format."
    ),
)
@click.option(
    "-n", "--dry-run",
    is_flag=True,
    default=False,
    help="Print generated dialogue to stdout instead of synthesising audio.",
)
@click.option(
    "-A", "--no-audio",
    "no_audio",
    is_flag=True,
    default=False,
    help="Skip TTS synthesis and audio export (add --report for the report folder).",
)
@click.option(
    "--no-progress",
    "no_progress",
    is_flag=True,
    default=False,
    help="Disable the rich progress bar.",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    default=False,
    help="Enable DEBUG-level logging.",
)
@click.option(
    "-r/-N", "--report/--no-report",
    default=False,
    help="Generate a report folder (sources, script, research, links, overview) "
         "alongside the podcast. Off by default; pass --report to enable it.",
)
@click.option(
    "--duo",
    "duo",
    default=None,
    help=(
        "Named voice duo to use for both speakers (e.g. warm, contrast, "
        "explorer, journalist, debate). Overrides gemini.default_duo and any "
        "legacy gemini.speakerN blocks. Run `tts-podcast duos` to list them."
    ),
)
@click.option(
    "--preset",
    "preset",
    type=click.Choice([*STYLE_PRESETS.keys(), "none"], case_sensitive=False),
    default=None,
    help=(
        "Style & angle (optional): named style preset for the dialogue. "
        "Use 'none' to disable a configured gemini.style.preset for one run."
    ),
)
@click.option(
    "--style",
    "style_text",
    default=None,
    help=(
        "Style & angle (optional): free-text style guidance for the dialogue "
        "(capped at 500 chars). Composes with --preset when both are set. "
        "Pass an empty string to override a configured gemini.style.text."
    ),
)
@click.option(
    "--speaker1-style",
    "speaker1_style",
    default=None,
    help=(
        "Style & angle (optional): per-episode overlay for speaker 1 "
        "(capped at 500 chars). Renders in a dedicated 'Episode-specific "
        "adjustments:' block; the baseline personality (and TTS voice "
        "acting) stays unchanged."
    ),
)
@click.option(
    "--speaker2-style",
    "speaker2_style",
    default=None,
    help=(
        "Style & angle (optional): per-episode overlay for speaker 2 "
        "(capped at 500 chars). Same semantics as --speaker1-style."
    ),
)
@click.option(
    "--angle",
    "angle",
    default=None,
    help=(
        "Style & angle (optional): episode angle (capped at 500 chars). "
        "Steers the dialogue prompt and the first research round only."
    ),
)
@click.option(
    "-L", "--follow-links",
    "follow_links",
    is_flag=True,
    default=False,
    help=(
        "After scraping inputs, discover and follow interesting links found "
        "inside them (heuristic pre-filter + LLM content-relevance judgement). "
        "Fetched pages feed both research and dialogue."
    ),
)
@click.option(
    "--follow-depth",
    "follow_depth",
    type=int,
    default=None,
    help=(
        "How many link-following hops to perform when --follow-links is set "
        "(default 1). Each hop re-extracts links from the pages kept in the "
        "previous hop."
    ),
)
def run(
    inputs: tuple[str, ...],
    files: tuple[Path, ...],
    search_queries: tuple[str, ...],
    config_path: str | None,
    research_rounds: int | None,
    target_duration: float | None,
    output_dir_override: str | None,
    output_file: str | None,
    dry_run: bool,
    no_audio: bool,
    no_progress: bool,
    verbose: bool,
    report: bool,
    duo: str | None,
    preset: str | None,
    style_text: str | None,
    speaker1_style: str | None,
    speaker2_style: str | None,
    angle: str | None,
    follow_links: bool,
    follow_depth: int | None,
) -> None:
    """Fetch URLs, local files, or web search queries and generate a two-voice podcast MP3."""
    load_dotenv()
    _setup_logging(verbose)

    url_list: list[str] = []
    file_paths: list[Path] = list(files)
    for arg in inputs:
        if arg.startswith(("http://", "https://")):
            url_list.append(arg)
        elif Path(arg).is_file():
            file_paths.append(Path(arg))
        else:
            click.echo(
                f"[ERROR] Argument is neither a URL nor an existing file: {arg!r}",
                err=True,
            )
            sys.exit(1)
    search_list = list(search_queries)

    if not (url_list or file_paths or search_list):
        click.echo(
            "[ERROR] No inputs provided. Pass URL(s), -f FILE, or -s 'search query'.",
            err=True,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 0. Preflight checks
    # ------------------------------------------------------------------
    if not dry_run and not no_audio:
        _check_ffmpeg()

    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    config_path = _resolve_config_path(config_path)
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        click.echo(f"[ERROR] Configuration error: {exc}", err=True)
        sys.exit(1)

    web_cfg = cfg.get("web", {})
    gemini_cfg = cfg.get("gemini", {})
    scraping_cfg = cfg.get("scraping", {})
    output_cfg = cfg.get("output", {})
    research_cfg = cfg.get("research", {}) or {}
    pricing_cfg: dict = cfg.get("pricing", {})

    # Resolve the active voice duo before reading any speaker field.
    # Precedence: CLI --duo > gemini.default_duo > legacy gemini.speakerN
    # blocks > built-in default. A resolved duo populates
    # gemini_cfg["speaker1"/"speaker2"], so every downstream consumer (TTS
    # preamble, dialogue prompt, --speakerN-style overlays) is unchanged and
    # configs that only define legacy speaker1/speaker2 keep working as-is.
    config_duos = gemini_cfg.get("duos")
    has_legacy_speakers = bool(gemini_cfg.get("speaker1")) and bool(gemini_cfg.get("speaker2"))
    duo_name = duo or gemini_cfg.get("default_duo")
    if duo_name is None and not has_legacy_speakers:
        duo_name = DEFAULT_DUO
    try:
        resolved_duo = resolve_duo(duo_name, config_duos)
    except click.BadParameter as exc:
        click.echo(f"[ERROR] {exc.format_message()}", err=True)
        sys.exit(1)
    if resolved_duo is not None:
        gemini_cfg = {
            **gemini_cfg,
            "speaker1": resolved_duo["speaker1"],
            "speaker2": resolved_duo["speaker2"],
        }
        logger.info("Using voice duo %r", duo_name)

    scrape_timeout: int = scraping_cfg.get("timeout_seconds", 10)
    cloak_fallback: bool = bool(scraping_cfg.get("cloak_fallback", False))

    output_dir: str = (
        output_dir_override
        or output_cfg.get("dir")
        or output_cfg.get("directory")
        or "."
    )
    output_fmt: str = output_cfg.get("format", "mp3")

    # When streaming audio to stdout, every status line, progress bar, and
    # summary must go to stderr so they never corrupt the binary blob.
    to_stdout: bool = output_file == "-"

    def _status(message: str) -> None:
        """Emit a status line — on stderr when audio is streamed to stdout."""
        click.echo(message, err=to_stdout)

    speaker1_name: str = gemini_cfg.get("speaker1", {}).get("name", "Alex")
    speaker2_name: str = gemini_cfg.get("speaker2", {}).get("name", "Jordan")

    web_user_agent: str = web_cfg.get("user_agent", BROWSER_USER_AGENT)

    service_tier: str | None = gemini_cfg.get("service_tier") or None
    if service_tier:
        logger.info("Gemini service tier: %s", service_tier)

    # Resolve research rounds: CLI > config > 0.
    if research_rounds is None:
        research_rounds = int(research_cfg.get("rounds_default", 0))
    if research_rounds < 0:
        click.echo(
            f"[ERROR] --research must be a non-negative integer (got {research_rounds}).",
            err=True,
        )
        sys.exit(1)

    # Make the rounds value visible to the research module too.
    gemini_research_cfg = dict(gemini_cfg.get("research", {}) or {})
    if "model" in research_cfg and "model" not in gemini_research_cfg:
        gemini_research_cfg["model"] = research_cfg["model"]
    gemini_cfg = {**gemini_cfg, "research": gemini_research_cfg}

    # Resolve link-following: validate the depth, then thread the follow model
    # through gemini_cfg so link_follower._judge_sources can read it. A bare
    # --follow-depth without --follow-links is a no-op (warn, don't error).
    if follow_depth is not None and follow_depth < 1:
        click.echo(
            f"[ERROR] --follow-depth must be a positive integer (got {follow_depth}).",
            err=True,
        )
        sys.exit(1)
    follow_depth_resolved = follow_depth if follow_depth is not None else 1
    if follow_depth is not None and not follow_links:
        logger.warning("--follow-depth is ignored without --follow-links.")

    follow_cfg = cfg.get("follow", {}) or {}
    max_links_per_level = int(follow_cfg.get("max_links_per_level", 5))
    max_links_total = int(follow_cfg.get("max_links_total", 20))
    if follow_cfg.get("model"):
        gemini_cfg = {**gemini_cfg, "follow": {"model": follow_cfg.get("model")}}

    # CLI --duration overrides gemini.dialogue.target_duration_minutes.
    if target_duration is not None:
        if target_duration <= 0:
            click.echo(
                f"[ERROR] --duration must be positive (got {target_duration}).",
                err=True,
            )
            sys.exit(1)
        dialogue_cfg = dict(gemini_cfg.get("dialogue", {}) or {})
        dialogue_cfg["target_duration_minutes"] = target_duration
        gemini_cfg = {**gemini_cfg, "dialogue": dialogue_cfg}

    # CLI style/angle flags override gemini.style.* and gemini.speakerN.style_overlay.
    # Writes to dedicated keys only — gemini.speakerN.personality is NEVER mutated
    # so the TTS preamble keeps reading the baseline personality verbatim.
    if preset is not None or style_text is not None or angle is not None:
        style_cfg = dict(gemini_cfg.get("style", {}) or {})
        if preset is not None:
            style_cfg["preset"] = preset
        if style_text is not None:
            style_cfg["text"] = style_text
        if angle is not None:
            style_cfg["angle"] = angle
        gemini_cfg = {**gemini_cfg, "style": style_cfg}
    if speaker1_style is not None:
        speaker1_cfg = dict(gemini_cfg.get("speaker1", {}) or {})
        speaker1_cfg["style_overlay"] = speaker1_style
        gemini_cfg = {**gemini_cfg, "speaker1": speaker1_cfg}
    if speaker2_style is not None:
        speaker2_cfg = dict(gemini_cfg.get("speaker2", {}) or {})
        speaker2_cfg["style_overlay"] = speaker2_style
        gemini_cfg = {**gemini_cfg, "speaker2": speaker2_cfg}

    # Auto-bump research when the only inputs are search queries.
    if (
        research_rounds == 0
        and search_list
        and not (url_list or file_paths)
    ):
        logger.info(
            "Search-only run with no research rounds — bumping to 1 to materialise content."
        )
        research_rounds = 1

    tracker = TokenTracker(pricing=pricing_cfg, service_tier=service_tier)

    total_inputs = len(url_list) + len(file_paths) + len(search_list)
    logger.info(
        "Processing %d input(s) (%d URL(s), %d file(s), %d search query/queries) | research rounds: %d",
        total_inputs, len(url_list), len(file_paths), len(search_list), research_rounds,
    )

    # ------------------------------------------------------------------
    # 2. Scrape → Research → Dialogue → TTS
    # ------------------------------------------------------------------
    with _make_progress(disable=no_progress or to_stdout) as progress:

        # 2a. Collect all sources (URLs, local files, search queries)
        scraped_sources: list[Source] = []
        if url_list:
            scrape_task = progress.add_task(
                f"[cyan]Scraping[/cyan] {len(url_list)} URL(s)…",
                total=len(url_list),
            )
            scraped_sources = scrape_urls(
                url_list,
                timeout=scrape_timeout,
                user_agent=web_user_agent,
                progress=progress,
                task_id=scrape_task,
                use_cloak_fallback=cloak_fallback,
            )

        file_sources: list[Source] = []
        if file_paths:
            load_task = progress.add_task(
                f"[cyan]Loading[/cyan] {len(file_paths)} local file(s)…",
                total=len(file_paths),
            )
            file_sources = load_local_files(file_paths, progress=progress, task_id=load_task)

        search_sources: list[Source] = [_make_search_source(q) for q in search_list]

        all_sources = scraped_sources + file_sources + search_sources

        ok_sources = [s for s in all_sources if s.scraped_ok]
        if not ok_sources:
            progress.stop()
            failed_inputs = ", ".join(s.url for s in all_sources)
            click.echo(
                f"[ERROR] Could not extract content from any input: {failed_inputs}",
                err=True,
            )
            sys.exit(1)

        if len(ok_sources) < len(all_sources):
            failed = [s.url for s in all_sources if not s.scraped_ok]
            logger.warning(
                "Input failed for %d/%d source(s); continuing with the rest. Failed: %s",
                len(failed), len(all_sources), ", ".join(failed),
            )

        # 2a-bis. Optional link following (runs before research and link
        # extraction so the kept pages flow into BOTH the research stage and
        # the dialogue). Two-stage: heuristic URL pre-filter, then an LLM
        # content-relevance judgement on the fetched pages.
        if follow_links:
            follow_task = progress.add_task(
                f"[cyan]Following links[/cyan] (depth {follow_depth_resolved})…",
                total=follow_depth_resolved,
            )
            followed = link_follower.follow_links(
                ok_sources,
                depth=follow_depth_resolved,
                gemini_cfg=gemini_cfg,
                scrape_timeout=scrape_timeout,
                user_agent=web_user_agent,
                cloak_fallback=cloak_fallback,
                token_tracker=tracker,
                max_links_per_level=max_links_per_level,
                max_links_total=max_links_total,
                progress=progress,
                task_id=follow_task,
            )
            if followed:
                ok_sources = ok_sources + followed
                all_sources = all_sources + followed
                logger.info(
                    "Followed %d additional source(s) via --follow-links.",
                    len(followed),
                )

        # 2b. Optional iterative research
        research_report = None
        if research_rounds > 0:
            research_task = progress.add_task(
                f"[cyan]Research[/cyan] ({research_rounds} round(s))…",
                total=research_rounds,
            )
            research_report = conduct_research(
                ok_sources,
                rounds=research_rounds,
                gemini_cfg=gemini_cfg,
                token_tracker=tracker,
                progress=progress,
                task_id=research_task,
                angle=gemini_cfg.get("style", {}).get("angle"),
            )
            progress.update(
                research_task,
                description=(
                    f"[cyan]Research[/cyan] done — {len(research_report.rounds)} round(s)"
                    f" · {tracker.live_line()}"
                ),
            )

        # 2c. Link extraction (for report and dry-run display)
        link_report = None
        if report:
            link_report = extract_links(ok_sources)

        # 2d. Dialogue generation
        llm_task = progress.add_task("[cyan]Generating dialogue…[/cyan]", total=1)
        research_notes = research_report.combined_notes if research_report else ""
        chunks = generate_dialogue(
            ok_sources,
            gemini_cfg,
            speaker1_name,
            speaker2_name,
            token_tracker=tracker,
            progress=progress,
            task_id=llm_task,
            research_notes=research_notes,
        )
        progress.update(
            llm_task,
            description=f"[cyan]Dialogue[/cyan]: {len(chunks)} chunk(s) — {tracker.live_line()}",
        )

        if dry_run:
            progress.stop()
            click.echo("\n=== Dialogue Preview ===\n")
            for chunk in chunks:
                click.echo(chunk.text)
                click.echo()
            if research_report is not None:
                click.echo("=== Research Notes ===\n")
                click.echo(research_report.combined_notes or "(no notes)")
                click.echo()
            sys.exit(0)

        # 2e. TTS synthesis (skipped when --no-audio)
        pcm_chunks: list[bytes] = []
        if not no_audio:
            tts_task = progress.add_task(
                f"[cyan]TTS synthesis[/cyan] ({len(chunks)} chunk(s))…",
                total=len(chunks),
            )
            pcm_chunks = generate_audio_chunks(
                chunks,
                gemini_cfg,
                token_tracker=tracker,
                progress=progress,
                task_id=tts_task,
            )
            if to_stdout:
                logger.info("TTS done — %s", tracker.live_line())
            else:
                progress.console.print(
                    f"  [dim]TTS done — {tracker.live_line()}[/dim]"
                )

    # ------------------------------------------------------------------
    # 3. Audio export (skipped when --no-audio)
    # ------------------------------------------------------------------
    identifiers: list[str] = (
        url_list
        + [f"file://{p.resolve()}" for p in file_paths]
        + [f"search://{q}" for q in search_list]
    )
    stem = _build_output_stem(identifiers)
    saved: Path | None = None
    if not no_audio:
        if to_stdout:
            data = encode_audio(pcm_chunks, fmt=output_fmt)
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
            logger.info(
                "Wrote %d byte(s) of %s audio to stdout", len(data), output_fmt.upper()
            )
        else:
            out_path, out_fmt = _resolve_output_path(
                output_file, output_dir, stem, output_fmt
            )
            logger.info("Exporting audio to %s…", out_path)
            saved = export_audio(pcm_chunks, out_path, fmt=out_fmt)
            _status(f"Podcast saved to: {saved}")
    else:
        if output_file:
            logger.warning("--output is ignored together with --no-audio.")
        _status("Skipping TTS synthesis and audio export (--no-audio).")

    # ------------------------------------------------------------------
    # 4. Report folder
    # ------------------------------------------------------------------
    if link_report is not None:
        report_dir = generate_report(
            sources=ok_sources,
            chunks=chunks,
            link_report=link_report,
            output_dir=output_dir,
            stem=stem,
            research=research_report,
            audio_path=saved,
            token_summary=tracker.summary(),
        )
        _status(f"Report folder saved to: {report_dir}")

    # ------------------------------------------------------------------
    # 5. Token / cost summary
    # ------------------------------------------------------------------
    _status("")
    _status(tracker.summary())


# ---------------------------------------------------------------------------
# `duos` command
# ---------------------------------------------------------------------------

@cli.command("duos")
def duos_cmd() -> None:
    """List the available voice duos (built-in + config-defined)."""
    raw = _load_raw_config()
    gemini_raw = raw.get("gemini", {}) if isinstance(raw.get("gemini"), dict) else {}
    config_duos = gemini_raw.get("duos")
    effective_default = gemini_raw.get("default_duo") or DEFAULT_DUO

    try:
        rows = describe_duos(config_duos)
    except click.BadParameter as exc:
        click.echo(f"[ERROR] {exc.format_message()}", err=True)
        sys.exit(1)

    click.echo("Available voice duos:\n")
    for slug, desc, sp1, sp2 in rows:
        marker = " [default]" if slug == effective_default else ""
        click.echo(f"  {click.style(slug, bold=True)}{marker}")
        click.echo(f"      {sp1}  ·  {sp2}")
        if desc:
            click.echo(f"      {desc}")
        click.echo()
    click.echo(
        "Select one with `--duo NAME` on `run`, or set gemini.default_duo in your config."
    )


# ---------------------------------------------------------------------------
# `config` subgroup
# ---------------------------------------------------------------------------

@cli.group("config", context_settings=CONTEXT_SETTINGS)
def config_group() -> None:
    """Manage the tts-podcast configuration file."""


def _load_raw_config() -> dict:
    """
    Load the raw YAML config without env-var resolution, or return an empty dict.

    Returns
    -------
    dict
        Parsed YAML content, or ``{}`` if the file does not exist.
    """
    if _DEFAULT_CONFIG.exists():
        raw = yaml.safe_load(_DEFAULT_CONFIG.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    return {}


def _prompt(label: str, default: str, **kwargs) -> str:
    """Thin wrapper around ``click.prompt`` that shows the default inline."""
    return click.prompt(label, default=default, **kwargs)


@config_group.command("init")
@click.option(
    "--output",
    "output_path",
    default=str(_DEFAULT_CONFIG),
    show_default=True,
    help="Where to write the config file.",
)
def config_init(output_path: str) -> None:
    """Interactively create or update the configuration file."""
    existing = _load_raw_config()

    def _get(section: str, key: str, fallback: str = "") -> str:
        return str(existing.get(section, {}).get(key, fallback))

    click.echo(f"\nConfiguring tts-podcast → {output_path}\n")
    click.echo("Press Enter to keep the current value shown in brackets.\n")

    click.echo("── Web fetching ──────────────────────────────────────────────")
    web_user_agent = _prompt(
        "User-Agent header",
        _get("web", "user_agent", BROWSER_USER_AGENT),
    )
    web_timeout = _prompt(
        "HTTP timeout (seconds)",
        _get("web", "timeout_seconds", "15"),
    )
    cloak_fallback = click.confirm(
        "Enable CloakBrowser stealth fallback for blocked pages? "
        "(optional; needs `uv sync --extra cloak`)",
        default=str(existing.get("scraping", {}).get("cloak_fallback", False)).lower() == "true",
    )

    click.echo("\n── Gemini ────────────────────────────────────────────────────")
    gemini_key_env  = _prompt(
        "Env var for Gemini API key",
        _get("gemini", "api_key_env", "GEMINI_API_KEY"),
    )
    gemini_text_model = _prompt("Text model",  _get("gemini", "text_model", "gemini-3.5-flash"))
    gemini_tts_model  = _prompt("TTS model",   _get("gemini", "tts_model", "gemini-3.1-flash-tts-preview"))
    gemini_language   = _prompt("Podcast language", _get("gemini", "language", "French"))
    gemini_tier       = _prompt(
        "Service tier (standard/flex/priority, leave empty for default)",
        _get("gemini", "service_tier", ""),
    )

    click.echo("\n── Voices ────────────────────────────────────────────────────")
    click.echo(f"Built-in duos: {', '.join(BUILTIN_DUOS)}")
    click.echo("Pick one, or leave blank to configure the two speakers manually.")
    default_duo_choice = _prompt(
        "Default duo (blank = manual speakers)",
        str(existing.get("gemini", {}).get("default_duo", "") or ""),
        show_default=True,
    ).strip()

    speaker_blocks: dict = {}
    if default_duo_choice:
        try:
            resolve_duo(default_duo_choice)
        except click.BadParameter as exc:
            click.echo(f"[ERROR] {exc.format_message()}", err=True)
            sys.exit(1)
    else:
        click.echo("\n── Speaker 1 ─────────────────────────────────────────────────")
        sp1 = existing.get("gemini", {}).get("speaker1", {}) if isinstance(existing.get("gemini", {}).get("speaker1"), dict) else {}
        sp1_name  = _prompt("Name",        sp1.get("name", "Alex"))
        sp1_voice = _prompt(
            f"Voice (one of {len(_GEMINI_VOICES)}: {', '.join(_GEMINI_VOICES)})",
            sp1.get("voice", "Sulafat"),
        )
        sp1_personality = _prompt(
            "Personality",
            sp1.get("personality", "warm, welcoming, makes complex tech feel human and inviting"),
        )

        click.echo("\n── Speaker 2 ─────────────────────────────────────────────────")
        sp2 = existing.get("gemini", {}).get("speaker2", {}) if isinstance(existing.get("gemini", {}).get("speaker2"), dict) else {}
        sp2_name  = _prompt("Name",        sp2.get("name", "Jordan"))
        sp2_voice = _prompt(
            f"Voice (one of {len(_GEMINI_VOICES)}: {', '.join(_GEMINI_VOICES)})",
            sp2.get("voice", "Achird"),
        )
        sp2_personality = _prompt(
            "Personality",
            sp2.get("personality", "friendly, witty, asks the questions the listener is thinking"),
        )
        speaker_blocks = {
            "speaker1": {
                "name": sp1_name,
                "voice": sp1_voice,
                "personality": sp1_personality,
            },
            "speaker2": {
                "name": sp2_name,
                "voice": sp2_voice,
                "personality": sp2_personality,
            },
        }

    click.echo("\n── Style & angle (optional) ────────────────────────────────")
    style_existing = existing.get("gemini", {}).get("style", {}) if isinstance(existing.get("gemini", {}).get("style"), dict) else {}
    style_preset = _prompt(
        "Default preset (casual, academic, humorous, debate, vulgarized; blank to skip)",
        str(style_existing.get("preset", "")),
        show_default=True,
    )
    style_text = _prompt(
        "Default style guidance (free text, blank to skip)",
        str(style_existing.get("text", "")),
        show_default=True,
    )
    style_angle = _prompt(
        "Default episode angle (blank to skip)",
        str(style_existing.get("angle", "")),
        show_default=True,
    )

    click.echo("\n── Dialogue thinking (optional) ─────────────────────────────")
    dialogue_existing = (
        existing.get("gemini", {}).get("dialogue", {})
        if isinstance(existing.get("gemini", {}).get("dialogue"), dict)
        else {}
    )
    dialogue_thinking_level = _prompt(
        "Dialogue thinking level for Gemini 3.x models "
        "(minimal|low|medium|high; blank = API default)",
        str(dialogue_existing.get("thinking_level", "")),
        show_default=True,
    )

    click.echo("\n── Research ──────────────────────────────────────────────────")
    research_rounds_default = _prompt(
        "Default research rounds (0 disables; override per run with -R)",
        str(existing.get("research", {}).get("rounds_default", 0)),
    )

    click.echo("\n── Output ────────────────────────────────────────────────────")
    output_dir = _prompt(
        "Output directory",
        existing.get("output", {}).get("dir") or existing.get("output", {}).get("directory") or ".",
    )
    output_fmt = _prompt("Format (mp3/wav)",  _get("output", "format", "mp3"))

    cfg: dict = {
        "web": {
            "user_agent": web_user_agent,
            "timeout_seconds": int(web_timeout),
        },
        "gemini": {
            "api_key_env": gemini_key_env,
            "text_model": gemini_text_model,
            "tts_model": gemini_tts_model,
            "language": gemini_language,
        },
        "research": {
            "rounds_default": int(research_rounds_default),
        },
        "scraping": {
            "timeout_seconds": int(existing.get("scraping", {}).get("timeout_seconds", 10)),
            "cloak_fallback": cloak_fallback,
        },
        "output": {
            "dir": output_dir,
            "format": output_fmt,
        },
        "pricing": existing.get("pricing", {}),
    }
    # Voices: a chosen duo writes gemini.default_duo; otherwise the manual
    # speaker1/speaker2 blocks collected above are written verbatim.
    if default_duo_choice:
        cfg["gemini"]["default_duo"] = default_duo_choice
    else:
        cfg["gemini"].update(speaker_blocks)
    if gemini_tier.strip():
        cfg["gemini"]["service_tier"] = gemini_tier.strip()

    # Dialogue section: write only non-empty / non-default values.
    dialogue_block: dict = {}
    if dialogue_thinking_level.strip():
        dialogue_block["thinking_level"] = dialogue_thinking_level.strip()
    if dialogue_block:
        cfg["gemini"]["dialogue"] = dialogue_block

    # Style & angle: write only the non-empty values under gemini.style.
    style_block: dict[str, str] = {}
    if style_preset.strip():
        style_block["preset"] = style_preset.strip()
    if style_text.strip():
        style_block["text"] = style_text.strip()
    if style_angle.strip():
        style_block["angle"] = style_angle.strip()
    if style_block:
        cfg["gemini"]["style"] = style_block

    if not cfg["pricing"]:
        example = Path(__file__).parent.parent.parent / "config.example.yaml"
        if example.exists():
            example_raw = yaml.safe_load(example.read_text(encoding="utf-8"))
            cfg["pricing"] = example_raw.get("pricing", {})

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        yaml.dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    click.echo(f"\nConfiguration written to {dest}")
    click.echo(
        f"\nMake sure this environment variable is set before running:\n"
        f"  export {gemini_key_env}=<your Gemini API key>"
    )


@config_group.command("show")
@click.option(
    "--resolve",
    is_flag=True,
    default=False,
    help="Show resolved values (secrets masked).",
)
def config_show(resolve: bool) -> None:
    """Display the current configuration file."""
    if not _DEFAULT_CONFIG.exists():
        click.echo(
            f"No config file found at {_DEFAULT_CONFIG}.\n"
            "Run `tts-podcast config init` to create one.",
            err=True,
        )
        sys.exit(1)

    if resolve:
        try:
            cfg = load_config(_DEFAULT_CONFIG)
        except ConfigError as exc:
            click.echo(f"[ERROR] {exc}", err=True)
            sys.exit(1)

        def _mask(data):
            if isinstance(data, dict):
                return {k: ("***" if any(s in k for s in ("key", "password", "token", "secret")) else _mask(v)) for k, v in data.items()}
            if isinstance(data, list):
                return [_mask(i) for i in data]
            return data

        masked = _mask(cfg)
        text = yaml.dump(masked, allow_unicode=True, sort_keys=False, default_flow_style=False)
        console.print(Syntax(text, "yaml", theme="monokai"))
    else:
        text = _DEFAULT_CONFIG.read_text(encoding="utf-8")
        console.print(Syntax(text, "yaml", theme="monokai"))
        click.echo(f"\n{_DEFAULT_CONFIG}")


if __name__ == "__main__":
    cli()
