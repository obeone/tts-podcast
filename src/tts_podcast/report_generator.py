"""
Report generator for tts-podcast runs.

Creates an output folder for each generation containing:

- ``overview.md`` — generation metadata: source URLs, article count,
  research rounds, audio path, and token/cost summary.
- ``sources.md`` — scraped sources with title, URL, summary, and full text
  when available.
- ``script.md`` — the full two-host dialogue script.
- ``research.md`` — per-round research notes and citations (only when
  research was performed).
- ``summary.md`` — synthetic reference sheet with categorised links to all
  sources, repositories, models, and papers mentioned in the articles.

The heavy lifting for link categorisation is delegated to
:mod:`tts_podcast.link_extractor`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tts_podcast.link_extractor import CategorisedLink, LinkReport
    from tts_podcast.llm_summarizer import DialogueChunk
    from tts_podcast.models import Source
    from tts_podcast.research import ResearchReport

logger = logging.getLogger(__name__)


def _render_overview(
    sources: list[Source],
    chunks: list[DialogueChunk],
    link_report: LinkReport,
    research: ResearchReport | None = None,
    audio_path: Path | str | None = None,
    token_summary: str | None = None,
    duo: dict | None = None,
) -> str:
    """
    Render a generation overview as Markdown.

    Parameters
    ----------
    sources : list[Source]
        Scraped sources included in the podcast.
    chunks : list[DialogueChunk]
        Dialogue chunks produced by the LLM.
    link_report : LinkReport
        Categorised links extracted from the sources.
    research : ResearchReport or None
        Research output (when research was performed).  ``None`` or zero
        rounds collapses the "Research rounds" row to ``0``.
    audio_path : Path or str or None
        Path to the generated audio file.
    token_summary : str or None
        Human-readable token/cost summary from
        :class:`~tts_podcast.token_tracker.TokenTracker`.
    duo : dict or None
        Generated voice duo dict (from ``duo_generator.generate_duo``),
        present only when ``--duo auto`` was used.  When provided, a
        "Voice Duo" section is appended to the overview.

    Returns
    -------
    str
        Markdown content for ``overview.md``.
    """
    lines: list[str] = ["# Podcast Generation Overview\n"]

    research_rounds = len(research.rounds) if research is not None else 0

    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Sources | {len(sources)} |")
    lines.append(f"| Research rounds | {research_rounds} |")
    lines.append(f"| Dialogue chunks | {len(chunks)} |")
    lines.append(f"| Links extracted | {link_report.total} |")
    if audio_path:
        lines.append(f"| Audio file | `{audio_path}` |")
    lines.append("")

    if duo is not None:
        # Auto-generated voice duo — surface who the hosts are and why.
        s1 = duo.get("speaker1", {})
        s2 = duo.get("speaker2", {})
        lines.append("## Voice Duo (auto-generated)\n")
        if duo.get("description"):
            lines.append(f"*{duo['description']}*\n")
        lines.append("| Speaker | Voice | Personality |")
        lines.append("|---|---|---|")
        lines.append(
            f"| **{s1.get('name', '?')}** | `{s1.get('voice', '?')}` | {s1.get('personality', '')} |"
        )
        lines.append(
            f"| **{s2.get('name', '?')}** | `{s2.get('voice', '?')}` | {s2.get('personality', '')} |"
        )
        lines.append("")

    if sources:
        lines.append("## Sources\n")
        for idx, source in enumerate(sources, start=1):
            title = source.title or source.url
            if source.kind == "search":
                lines.append(f"{idx}. *Web search:* `{title}`")
            else:
                ok_marker = "" if source.scraped_ok else " *(input failed)*"
                lines.append(f"{idx}. [{title}]({source.url}){ok_marker}")
        lines.append("")

    if link_report.total > 0:
        lines.append("## Link Breakdown\n")
        lines.append("| Category | Count |")
        lines.append("|---|---|")
        for label, items in (
            ("Source articles", link_report.sources),
            ("Repositories", link_report.repos),
            ("Models", link_report.models),
            ("Papers", link_report.papers),
            ("Other", link_report.other),
        ):
            if items:
                lines.append(f"| {label} | {len(items)} |")
        lines.append("")

    if token_summary:
        lines.append("## Token Usage & Cost\n")
        lines.append("```")
        lines.append(token_summary)
        lines.append("```\n")

    return "\n".join(lines) + "\n"


def _render_sources(sources: list[Source]) -> str:
    """
    Render the list of scraped sources as Markdown.

    Parameters
    ----------
    sources : list[Source]
        Scraped sources included in the podcast.

    Returns
    -------
    str
        Markdown content for ``sources.md``.
    """
    lines: list[str] = ["# Sources\n"]

    for source in sources:
        title = source.title or source.url
        if source.kind == "search":
            lines.append(f"## Web search: {title}\n")
            lines.append("*Topic researched via Google Search grounding — see research notes.*\n")
            continue
        lines.append(f"## {title}\n")
        lines.append(f"**URL:** <{source.url}>\n")
        if not source.scraped_ok:
            lines.append("*Input failed — no content extracted.*\n")
            continue
        if source.summary:
            lines.append(f"**Summary:** {source.summary}\n")
        if source.full_text and source.full_text != source.summary:
            lines.append("<details>\n<summary>Full text</summary>\n")
            lines.append(f"{source.full_text}\n")
            lines.append("</details>\n")

    return "\n".join(lines) + "\n"


def _render_script(chunks: list[DialogueChunk]) -> str:
    """
    Render the dialogue script as Markdown.

    Parameters
    ----------
    chunks : list[DialogueChunk]
        Dialogue chunks produced by the LLM summariser.

    Returns
    -------
    str
        Markdown content for ``script.md``.
    """
    lines: list[str] = ["# Podcast Script\n"]
    for chunk in chunks:
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines) + "\n"


def _render_research(research: ResearchReport) -> str:
    """
    Render the per-round research notes as Markdown.

    Parameters
    ----------
    research : ResearchReport
        Research output, including any citations and queries issued by
        Gemini for each round.

    Returns
    -------
    str
        Markdown content for ``research.md``.
    """
    lines: list[str] = ["# Complementary Research\n"]

    if not research.rounds:
        lines.append("*No research rounds were executed.*\n")
        return "\n".join(lines) + "\n"

    for r in research.rounds:
        lines.append(f"## Round {r.index + 1} — {r.query_hint}\n")
        notes = r.notes.strip() or "*(no notes returned)*"
        lines.append(notes)
        lines.append("")

        if r.raw_search_queries:
            lines.append("**Search queries issued:**\n")
            for q in r.raw_search_queries:
                lines.append(f"- `{q}`")
            lines.append("")

        if r.citations:
            lines.append("**Citations:**\n")
            for c in r.citations:
                label = c.title or c.uri
                lines.append(f"- [{label}]({c.uri})")
            lines.append("")

    return "\n".join(lines) + "\n"


def _render_summary(
    link_report: LinkReport,
    sources: list[Source] | None = None,
) -> str:
    """
    Render the synthetic reference sheet as Markdown.

    When *sources* is provided, the summary includes per-source detail
    blocks with the summary text and all links extracted from that source
    grouped by category.  Without *sources*, falls back to flat categorised
    link lists.

    Parameters
    ----------
    link_report : LinkReport
        Categorised links extracted from the articles.
    sources : list[Source] or None, optional
        Scraped sources used to enrich the summary with context.

    Returns
    -------
    str
        Markdown content for ``summary.md``.
    """
    lines: list[str] = ["# Synthetic Summary — Sources & References\n"]

    if link_report.total == 0:
        lines.append("*No external links were found in the articles.*\n")
        return "\n".join(lines) + "\n"

    source_map: dict[str, Source] = {}
    if sources:
        for s in sources:
            source_map[s.title] = s

    links_by_article: dict[str, list[CategorisedLink]] = {}
    for bucket in (link_report.repos, link_report.models, link_report.papers, link_report.other):
        for link in bucket:
            links_by_article.setdefault(link.label, []).append(link)

    if sources:
        lines.append("## Sources\n")
        for source_link in link_report.sources:
            source = source_map.get(source_link.label)
            lines.append(f"### [{source_link.label}]({source_link.url})\n")

            if source and source.summary:
                lines.append(f"{source.summary}\n")

            secondary = links_by_article.get(source_link.label, [])
            if secondary:
                category_labels = {
                    "repo": "Repository",
                    "model": "Model",
                    "paper": "Paper",
                    "other": "Link",
                }
                for link in secondary:
                    cat_label = category_labels.get(link.category, "Link")
                    lines.append(f"- {cat_label}: [{_url_short_label(link.url)}]({link.url})")
                lines.append("")

    flat_sections = [
        ("GitHub / GitLab Repositories", link_report.repos),
        ("Hugging Face Models", link_report.models),
        ("Academic Papers", link_report.papers),
        ("Other Links", link_report.other),
    ]

    has_flat = any(items for _, items in flat_sections)
    if has_flat:
        lines.append("\n---\n")
        lines.append("## All Links by Category\n")

        for heading, items in flat_sections:
            if not items:
                continue
            lines.append(f"### {heading}\n")
            for link in items:
                lines.append(f"- [{link.label}]({link.url})")
            lines.append("")

    return "\n".join(lines) + "\n"


def _url_short_label(url: str) -> str:
    """
    Extract a short display label from a URL.

    Returns the domain + path stripped of protocol and trailing slashes.

    Parameters
    ----------
    url : str
        Full URL.

    Returns
    -------
    str
        Shortened label, e.g. ``"github.com/org/repo"``.
    """
    label = url.split("://", 1)[-1].rstrip("/")
    if len(label) > 80:
        label = label[:77] + "..."
    return label


def generate_report(
    sources: list[Source],
    chunks: list[DialogueChunk],
    link_report: LinkReport,
    output_dir: str | Path,
    stem: str,
    research: ResearchReport | None = None,
    audio_path: Path | str | None = None,
    token_summary: str | None = None,
    duo: dict | None = None,
) -> Path:
    """
    Generate a complete report folder for one podcast generation.

    Creates ``<output_dir>/tts_<stem>/`` containing four Markdown files
    (``overview.md``, ``sources.md``, ``script.md``, ``summary.md``) plus
    ``research.md`` when *research* contains at least one round.

    Parameters
    ----------
    sources : list[Source]
        The scraped sources used for this podcast episode.
    chunks : list[DialogueChunk]
        The dialogue chunks produced by the LLM.
    link_report : LinkReport
        Categorised links extracted from the sources.
    output_dir : str | Path
        Parent directory for output (e.g. ``"output"``).
    stem : str
        Filename stem used to name the report folder (e.g.
        ``"arxiv.org-a1b2c3-2026-05-23"``).
    research : ResearchReport or None, optional
        Research output to write to ``research.md``.  When ``None`` or
        when ``rounds`` is empty, the file is omitted.
    audio_path : Path or str or None
        Path to the generated audio file (shown in the overview).
    token_summary : str or None
        Human-readable token/cost summary from the token tracker.
    duo : dict or None, optional
        Auto-generated voice duo dict (from ``duo_generator.generate_duo``).
        When present, the overview includes a "Voice Duo" section with the
        host names, Gemini voices, and personality descriptions.

    Returns
    -------
    Path
        Path to the created report folder.

    Raises
    ------
    OSError
        If the report directory cannot be created or files cannot be written.

    Examples
    --------
    >>> from pathlib import Path
    >>> report_dir = generate_report(sources, chunks, links, "output", "example.com-abc123")
    >>> (report_dir / "overview.md").exists()
    True
    """
    report_dir = Path(output_dir) / f"tts_{stem}"
    report_dir.mkdir(parents=True, exist_ok=True)

    overview_path = report_dir / "overview.md"
    overview_path.write_text(
        _render_overview(
            sources, chunks, link_report,
            research=research,
            audio_path=audio_path,
            token_summary=token_summary,
            duo=duo,
        ),
        encoding="utf-8",
    )
    logger.info("Written %s", overview_path)

    sources_path = report_dir / "sources.md"
    sources_path.write_text(_render_sources(sources), encoding="utf-8")
    logger.info("Written %s", sources_path)

    script_path = report_dir / "script.md"
    script_path.write_text(_render_script(chunks), encoding="utf-8")
    logger.info("Written %s", script_path)

    if research is not None and research.rounds:
        research_path = report_dir / "research.md"
        research_path.write_text(_render_research(research), encoding="utf-8")
        logger.info("Written %s", research_path)

    summary_path = report_dir / "summary.md"
    summary_path.write_text(_render_summary(link_report, sources=sources), encoding="utf-8")
    logger.info("Written %s", summary_path)

    logger.info("Report folder created: %s", report_dir)
    return report_dir
