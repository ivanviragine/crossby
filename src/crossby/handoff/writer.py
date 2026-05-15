"""Render a HandoffDocument or RawHandoff to a markdown file under .crossby/handoffs/."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from crossby.handoff.models import HandoffDocument, RawHandoff


class HandoffWriter:
    """Serializes a handoff document to a markdown file.

    Accepts either a structured :class:`HandoffDocument` or an unparsed
    :class:`RawHandoff` (produced when a custom summarization prompt is used)
    and dispatches to the right renderer.
    """

    def __init__(self, project_root: Path) -> None:
        self.output_dir = project_root / ".crossby" / "handoffs"

    def write(
        self,
        doc: HandoffDocument | RawHandoff,
        output_path: Path | None = None,
    ) -> Path:
        """Render ``doc`` to markdown and return the written path."""
        path = output_path or self._default_path(doc.created_at)
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered = render_markdown_raw(doc) if isinstance(doc, RawHandoff) else render_markdown(doc)
        path.write_text(rendered, encoding="utf-8")
        return path

    def _default_path(self, created_at: datetime) -> Path:
        stamp = created_at.strftime("%Y%m%dT%H%M%S%f")
        return self.output_dir / f"HANDOFF-{stamp}.md"


def render_markdown(doc: HandoffDocument) -> str:
    """Render a HandoffDocument as the structured markdown we write to disk."""
    lines: list[str] = [
        f"# Handoff: {doc.source_tool} → {doc.target_tool}",
        "",
        f"- **Source session**: `{doc.session_ref.session_id}`",
        f"- **Source path**: `{doc.session_ref.path}`",
        f"- **Session started**: {doc.session_ref.started_at.isoformat()}",
        f"- **Handoff created**: {doc.created_at.isoformat()}",
        "",
        "## Current Task",
        "",
        doc.current_task.strip() or "_(not captured)_",
        "",
        "## Key Decisions",
        "",
        *_bulleted(doc.key_decisions),
        "",
        "## Modified Files",
        "",
        *_bulleted([str(p) for p in doc.modified_files]),
        "",
        "## Blockers",
        "",
        *_bulleted(doc.blockers),
        "",
        "## Next Steps",
        "",
        *_bulleted(doc.next_steps),
        "",
        "## Critical Context",
        "",
        doc.critical_context.strip() or "_(none)_",
        "",
    ]
    return "\n".join(lines)


def render_markdown_raw(doc: RawHandoff) -> str:
    """Render a RawHandoff: metadata header plus the summarizer's raw body."""
    lines: list[str] = [
        f"# Handoff: {doc.source_tool} → {doc.target_tool} (raw)",
        "",
        f"- **Source session**: `{doc.session_ref.session_id}`",
        f"- **Source path**: `{doc.session_ref.path}`",
        f"- **Session started**: {doc.session_ref.started_at.isoformat()}",
        f"- **Handoff created**: {doc.created_at.isoformat()}",
        f"- **Prompt**: {doc.prompt_source}",
        "",
        doc.body,
        "",
    ]
    return "\n".join(lines)


def _bulleted(items: list[str]) -> list[str]:
    if not items:
        return ["_(none)_"]
    return [f"- {item}" for item in items]
