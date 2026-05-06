"""Persistent sync reports + portable markdown-table format.

Crossby's CLI prints a Rich table at the end of every sync run; that
output disappears as soon as the terminal scrolls. This module owns two
complementary outputs that survive past the run:

- A persistent ``.crossby/sync-report.md`` written after every real run
  so the user can re-open it (or paste it into a PR description).
- A portable ``| Status | Item | Notes |`` markdown table that the CLI
  can render in place of the Rich table when the user passes
  ``--report-format markdown-table``.

The ``Status`` column uses three controlled values: ``Added``,
``Check before using``, ``Not Added``. ``classify_status`` maps each
:class:`SyncResult` into that vocabulary by inspecting action +
message.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from crossby.sync.base import SyncConcern, SyncResult


REPORT_PATH = Path(".crossby") / "sync-report.md"


# Concern → singular display label used in the Item column.
_CONCERN_LABELS: dict[SyncConcern, str] = {
    SyncConcern.PERMISSIONS: "Permission",
    SyncConcern.RULES: "Rule",
    SyncConcern.MCP: "MCP",
    SyncConcern.AGENTS: "Agent",
    SyncConcern.SKILLS: "Skill",
    SyncConcern.HOOKS: "Hook",
}


# Hints we use to detect "Check before using" — i.e. an artifact was
# written but the writer flagged a lossy translation.
_CHECK_HINTS = (
    "foreign markers",
    "translated",
    "manual_fix",
    "manual-fix",
    "copy (symlink failed)",
)


def classify_status(result: SyncResult) -> str:
    """Map a :class:`SyncResult` to one of the three controlled statuses.

    The classification rule for ``skipped`` rows uses ``file_path``: when a
    writer skipped *and* never identified a target artifact (``file_path is
    None``) the right reading is "nothing was synced for this concern" →
    ``Not Added``. When ``file_path`` is set the skip means "already in
    place from a prior run" → ``Added``. This is more robust than
    substring-matching the message, which mis-labelled e.g. ``"no hooks
    config"`` as ``Added``.
    """
    message = (result.message or "").lower()
    if result.action in {"created", "updated"}:
        if any(hint in message for hint in _CHECK_HINTS):
            return "Check before using"
        return "Added"
    if result.action == "skipped":
        if result.file_path is None:
            return "Not Added"
        return "Added"
    # error
    return "Not Added"


def _item_label(result: SyncResult, project_root: Path | None = None) -> str:
    """Build the ``Item`` cell for a sync result.

    Paths are rendered relative to ``project_root`` when supplied; this
    keeps the markdown report portable across machines instead of leaking
    absolute paths into PR descriptions.
    """
    type_label = _CONCERN_LABELS.get(result.concern, result.concern.value)
    if result.file_path is not None:
        path = result.file_path
        if project_root is not None and path.is_absolute():
            try:
                path = path.relative_to(project_root)
            except ValueError:
                pass
        name = path.as_posix()
    elif result.tool_id is not None:
        name = str(result.tool_id)
    else:
        name = ""
    return f"`{type_label}` {name}".rstrip()


def render_markdown_table(
    results: Sequence[SyncResult],
    *,
    project_root: Path | None = None,
) -> str:
    """Render a portable ``| Status | Item | Notes |`` table.

    Returns the empty string when there are no rows so callers can omit
    the section entirely instead of producing a header-only table. Pass
    ``project_root`` to relativize file paths in the ``Item`` cell.
    """
    if not results:
        return ""
    lines = [
        "| Status | Item | Notes |",
        "| --- | --- | --- |",
    ]
    for result in results:
        status = classify_status(result)
        item = _item_label(result, project_root)
        notes = (result.message or "").strip() or "—"
        # Pipes in any of the cells would break the table; escape them.
        lines.append(
            "| `"
            + status
            + "` | "
            + item.replace("|", r"\|")
            + " | "
            + notes.replace("|", r"\|")
            + " |"
        )
    return "\n".join(lines)


def render_persistent_report(
    results: Sequence[SyncResult],
    *,
    project_name: str,
    timestamp: datetime | None = None,
    project_root: Path | None = None,
) -> str:
    """Build the markdown body that gets written to ``.crossby/sync-report.md``.

    The header carries a UTC timestamp + project name; the body is the
    same markdown table as :func:`render_markdown_table`. When there are
    no rows, the body explains that nothing was synced.
    """
    when = (timestamp or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = f"# crossby sync report\n\n_{when} · **{project_name}**_\n"
    if not results:
        return header + "\n_No sync rows were produced._\n"
    return (
        header
        + "\n"
        + render_markdown_table(results, project_root=project_root)
        + "\n"
    )


def write_persistent_report(
    results: Sequence[SyncResult],
    project_root: Path,
) -> Path:
    """Write the report to ``.crossby/sync-report.md`` and return the path."""
    body = render_persistent_report(
        results,
        project_name=project_root.name or "(unnamed project)",
        project_root=project_root,
    )
    path = project_root / REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


__all__ = [
    "REPORT_PATH",
    "classify_status",
    "render_markdown_table",
    "render_persistent_report",
    "write_persistent_report",
]
