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

from tts_podcast.audio_exporter import export_audio
from tts_podcast.config import ConfigError, load_config
from tts_podcast.link_extractor import extract_links
from tts_podcast.llm_summarizer import generate_dialogue
from tts_podcast.report_generator import generate_report
from tts_podcast.research import conduct_research
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

_GEMINI_VOICES = [
    "Puck", "Charon", "Kore", "Fenrir", "Aoede",
    "Leda", "Orus", "Zephyr",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


def _build_output_stem(urls: list[str]) -> str:
    """
    Build the filename stem for an episode based on its input URLs.

    Combines the first URL's hostname, a 6-character SHA-1 digest of all
    URLs joined by ``"\\n"``, and today's ISO date to keep stems short
    while remaining collision-resistant when the same domain is run
    multiple times in a day.

    Parameters
    ----------
    urls : list[str]
        URLs supplied on the command line.

    Returns
    -------
    str
        Filename stem without extension, e.g. ``"arxiv.org-a1b2c3-2026-05-23"``.
    """
    parsed = urlparse(urls[0])
    host = parsed.netloc or "podcast"
    # Trim leading "www."
    if host.startswith("www."):
        host = host[4:]
    digest = hashlib.sha1("\n".join(urls).encode("utf-8")).hexdigest()[:6]
    return f"{host}-{digest}-{date.today().isoformat()}"


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
@click.argument("urls", nargs=-1, required=True)
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
    "-o", "--output-dir",
    "output_dir_override",
    default=None,
    type=click.Path(file_okay=False),
    help="Directory where the podcast file is written. Overrides config output.dir.",
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
    help="Generate the dialogue script and report, but skip TTS synthesis and audio export.",
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
    default=True,
    help="Generate a report folder (sources, script, research, links, overview) alongside the podcast.",
)
def run(
    urls: tuple[str, ...],
    config_path: str | None,
    research_rounds: int | None,
    output_dir_override: str | None,
    dry_run: bool,
    no_audio: bool,
    no_progress: bool,
    verbose: bool,
    report: bool,
) -> None:
    """Fetch one or more URLs and generate a two-voice podcast MP3."""
    load_dotenv()
    _setup_logging(verbose)

    url_list = list(urls)

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

    scrape_timeout: int = scraping_cfg.get("timeout_seconds", 10)

    output_dir: str = (
        output_dir_override
        or output_cfg.get("dir")
        or output_cfg.get("directory")
        or "."
    )
    output_fmt: str = output_cfg.get("format", "mp3")

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

    tracker = TokenTracker(pricing=pricing_cfg, service_tier=service_tier)

    logger.info("Processing %d URL(s) | research rounds: %d", len(url_list), research_rounds)

    # ------------------------------------------------------------------
    # 2. Scrape → Research → Dialogue → TTS
    # ------------------------------------------------------------------
    with _make_progress(disable=no_progress) as progress:

        # 2a. Scraping
        scrape_task = progress.add_task(
            f"[cyan]Scraping[/cyan] {len(url_list)} URL(s)…",
            total=len(url_list),
        )
        sources = scrape_urls(
            url_list,
            timeout=scrape_timeout,
            user_agent=web_user_agent,
            progress=progress,
            task_id=scrape_task,
        )

        ok_sources = [s for s in sources if s.scraped_ok]
        if not ok_sources:
            progress.stop()
            failed_urls = ", ".join(s.url for s in sources)
            click.echo(
                f"[ERROR] Could not extract content from any URL: {failed_urls}",
                err=True,
            )
            sys.exit(1)

        if len(ok_sources) < len(sources):
            failed = [s.url for s in sources if not s.scraped_ok]
            logger.warning(
                "Scraping failed for %d/%d URL(s); continuing with the rest. Failed: %s",
                len(failed), len(sources), ", ".join(failed),
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
            progress.console.print(
                f"  [dim]TTS done — {tracker.live_line()}[/dim]"
            )

    # ------------------------------------------------------------------
    # 3. Audio export (skipped when --no-audio)
    # ------------------------------------------------------------------
    stem = _build_output_stem(url_list)
    saved: Path | None = None
    if not no_audio:
        filename = f"{stem}.{output_fmt}"
        out_path = Path(output_dir) / filename

        logger.info("Exporting audio to %s…", out_path)
        saved = export_audio(pcm_chunks, out_path, fmt=output_fmt)
        click.echo(f"Podcast saved to: {saved}")
    else:
        click.echo("Skipping TTS synthesis and audio export (--no-audio).")

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
        click.echo(f"Report folder saved to: {report_dir}")

    # ------------------------------------------------------------------
    # 5. Token / cost summary
    # ------------------------------------------------------------------
    click.echo()
    click.echo(tracker.summary())


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

    click.echo("\n── Speaker 1 ─────────────────────────────────────────────────")
    sp1 = existing.get("gemini", {}).get("speaker1", {}) if isinstance(existing.get("gemini", {}).get("speaker1"), dict) else {}
    sp1_name  = _prompt("Name",        sp1.get("name", "Alex"))
    sp1_voice = _prompt(f"Voice ({', '.join(_GEMINI_VOICES)})", sp1.get("voice", "Puck"))
    sp1_personality = _prompt(
        "Personality",
        sp1.get("personality", "enthusiastic, curious, quick to get excited about tech innovations"),
    )

    click.echo("\n── Speaker 2 ─────────────────────────────────────────────────")
    sp2 = existing.get("gemini", {}).get("speaker2", {}) if isinstance(existing.get("gemini", {}).get("speaker2"), dict) else {}
    sp2_name  = _prompt("Name",        sp2.get("name", "Jordan"))
    sp2_voice = _prompt(f"Voice ({', '.join(_GEMINI_VOICES)})", sp2.get("voice", "Charon"))
    sp2_personality = _prompt(
        "Personality",
        sp2.get("personality", "analytical, mildly skeptical, adds nuance and historical context"),
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
        },
        "research": {
            "rounds_default": int(research_rounds_default),
        },
        "scraping": existing.get("scraping", {"timeout_seconds": 10}),
        "output": {
            "dir": output_dir,
            "format": output_fmt,
        },
        "pricing": existing.get("pricing", {}),
    }
    if gemini_tier.strip():
        cfg["gemini"]["service_tier"] = gemini_tier.strip()

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
