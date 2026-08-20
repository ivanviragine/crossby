"""``crossby tools`` — manage the installed AI tools themselves.

Currently exposes::

    crossby tools update [--tool/-t <id> ...] [--yes/-y] [--dry-run]

Each adapter declares a static ``update_command`` in its capabilities; this
command runs the updaters for the installed, updatable tools and renders a
report. All subprocess/version logic lives in
``crossby.services.tool_update`` — the CLI only orchestrates selection, the
report, and the exit code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from crossby.models.ai import AIToolID
from crossby.ui.console import console

if TYPE_CHECKING:
    from crossby.services.tool_update import UpdateResult

tools_app = typer.Typer(
    name="tools",
    help="Manage installed AI tools.",
)


@tools_app.command("update")
def update(
    tool: list[str] | None = typer.Option(
        None,
        "--tool",
        "-t",
        help="Update only these tool ids (repeatable). Omit to pick interactively.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip crossby's confirmation prompt (does not inject tool-specific flags).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the resolved update command per tool and exit without running anything.",
    ),
) -> None:
    """Update installed AI tools by running each tool's own updater.

    Lists the installed, updatable tools, lets you select which to update
    (default all), runs each tool's static update command sequentially, and
    prints a report. GUI tools (the Antigravity IDE, VS Code) self-update
    through their IDE and are never offered. This updates the *managed tools*,
    not the crossby CLI itself.
    """
    from crossby.ai_tools.base import AbstractAITool
    from crossby.services.tool_update import run_update, updatable_tools
    from crossby.ui import prompts

    # Compute the inventory once. Keeping ``installed`` lets --tool give a precise
    # per-value reason that ``updatable`` alone (only the survivors) can't.
    installed = set(AbstractAITool.detect_installed())
    updatable = updatable_tools()

    if tool:
        # Validate the explicit selection BEFORE any "nothing to update" early
        # return, so a bad --tool errors clearly even on an empty inventory.
        selected = _resolve_explicit(tool, installed, updatable)
    else:
        if not updatable:
            console.info("No updatable AI tools detected.")
            raise typer.Exit(0)
        selected = _select_interactive(updatable)
        if not selected:
            console.info("No tools selected.")
            raise typer.Exit(0)

    if dry_run:
        console.info("Dry run — no updaters will run.")
        for tool_id in selected:
            display_name, command = _tool_command(tool_id)
            console.step(f"{display_name}: {' '.join(command)}")
        raise typer.Exit(0)

    # Confirm unless suppressed. Non-TTY auto-confirms (prompts.confirm returns
    # its default there); --yes skips only crossby's own prompt.
    if not yes and prompts.is_tty():
        count = len(selected)
        noun = "tool" if count == 1 else "tools"
        if not prompts.confirm(f"Update {count} {noun}?", default=True):
            console.info("Aborted.")
            raise typer.Exit(0)

    results: list[UpdateResult] = []
    for tool_id in selected:
        display_name, _command = _tool_command(tool_id)
        with console.status(f"Updating {display_name}…"):
            results.append(run_update(tool_id))

    _render_report(results)

    if any(not r.success for r in results):
        raise typer.Exit(1)


def _resolve_explicit(
    values: list[str],
    installed: set[AIToolID],
    updatable: list[AIToolID],
) -> list[AIToolID]:
    """Resolve/validate ``--tool`` values, de-duplicated in first-seen order.

    Rejection precedence per value: unknown id → not installed → installed but
    no update path. Any rejection exits 1 with a friendly reason.
    """
    updatable_set = set(updatable)
    seen: set[AIToolID] = set()
    resolved: list[AIToolID] = []
    for raw in values:
        try:
            tool_id = AIToolID(raw)
        except ValueError:
            valid = ", ".join(t.value for t in AIToolID)
            console.error(f"Unknown tool: {raw!r}. Valid ids: {valid}")
            raise typer.Exit(1) from None
        if tool_id in seen:
            continue
        seen.add(tool_id)
        if tool_id not in installed:
            console.error(f"{tool_id.value} is not installed.")
            raise typer.Exit(1)
        if tool_id not in updatable_set:
            console.error(f"{tool_id.value} has no update command.")
            raise typer.Exit(1)
        resolved.append(tool_id)
    return resolved


def _select_interactive(updatable: list[AIToolID]) -> list[AIToolID]:
    """Pick tools via multi-select over ``"<name> — <command>"`` labels.

    ``multi_select`` has no hints parameter, so the command is baked into the
    label. Non-TTY returns all updatable tools (the existing convention).
    """
    from crossby.ui import prompts

    labels: list[str] = []
    for tool_id in updatable:
        display_name, command = _tool_command(tool_id)
        labels.append(f"{display_name} — {' '.join(command)}")
    indices = prompts.multi_select("Select tools to update", labels, select_all=True)
    return [updatable[i] for i in indices]


def _tool_command(tool_id: AIToolID) -> tuple[str, tuple[str, ...]]:
    """Return ``(display_name, update_command)`` for an updatable tool."""
    from crossby.ai_tools.base import AbstractAITool

    caps = AbstractAITool.get(tool_id).capabilities()
    # Guaranteed non-None: callers only pass tools that passed updatable_tools().
    assert caps.update_command is not None
    return caps.display_name, caps.update_command


def _render_report(results: list[UpdateResult]) -> None:
    """Print the 3-column report table plus per-failure detail and warnings."""
    from rich.table import Table

    table = Table(title="Update report", title_style="header.title", header_style="header.title")
    table.add_column("Tool")
    table.add_column("Version")
    table.add_column("Status", justify="center")
    for r in results:
        table.add_row(r.display_name, _version_cell(r), _status_cell(r))
    console.empty()
    console.out.print(table)

    # Failure detail under the table (error is always set on failure).
    for r in results:
        if not r.success:
            console.error(f"{r.display_name} update failed: {r.error}")
            if r.output_tail:
                console.detail(r.output_tail)

    # Version-unchanged warnings — success that didn't move the version.
    for r in results:
        if r.unchanged:
            console.warn(
                f"{r.display_name} reported success but version did not change "
                f"({r.before_version})."
            )


def _status_cell(result: UpdateResult) -> str:
    """Format the Status column: updated, version unchanged, unknown-version success, or failed."""
    if not result.success:
        return f"[error]{console.ERR}[/]"
    if result.updated:
        return f"[success]{console.OK} updated[/]"
    if result.unchanged:
        return f"[dim]{console.OK} version unchanged[/]"
    return f"[success]{console.OK}[/]"


def _version_cell(result: UpdateResult) -> str:
    """Format the Version column: ``before → after``, a single known version, or ``—``."""
    if result.before_version and result.after_version:
        return f"{result.before_version} {console.ARROW} {result.after_version}"
    known = result.before_version or result.after_version
    return known or "—"
