"""CROSSBY CLI — main entry point."""

from __future__ import annotations

from pathlib import Path

import typer

import crossby

app = typer.Typer(
    name="crossby",
    help="CROSSBY — Cross-platform Bridge for Your AI agents.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"crossby {crossby.__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool | None = typer.Option(
        None, "--version", "-V", callback=_version_callback, is_eager=True, help="Show version."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """CROSSBY — Cross-platform Bridge for Your AI agents."""
    from crossby.logging import configure

    configure(verbose=verbose)

    if ctx.invoked_subcommand is not None:
        return

    from crossby.ui import prompts

    if prompts.is_tty():
        _interactive_main_menu(ctx)
        return

    # Non-TTY: preserve the old "no args prints help" contract.
    typer.echo(ctx.get_help())
    raise typer.Exit(0)


# Register subcommands
from crossby.cli.agents import agents_app  # noqa: E402
from crossby.cli.convert import convert  # noqa: E402
from crossby.cli.handoff import handoff  # noqa: E402
from crossby.cli.init import init  # noqa: E402
from crossby.cli.launch import launch  # noqa: E402
from crossby.cli.scene import scene_app  # noqa: E402
from crossby.cli.stats import stats  # noqa: E402
from crossby.cli.sync import sync  # noqa: E402

app.command()(launch)
app.command()(sync)
app.command()(convert)
app.command()(stats)
app.command()(handoff)
app.command()(init)
app.add_typer(agents_app, name="agents")
app.add_typer(scene_app, name="scene")


def _interactive_main_menu(ctx: typer.Context) -> None:
    """Top-level wizard menu shown when ``crossby`` is invoked with no args.

    Each entry dispatches by calling the subcommand's top-level Python function
    directly (not ``ctx.invoke``). Direct calls avoid the ``from`` Python-keyword
    trap: ``sync`` and ``handoff`` have a ``from_tool`` parameter whose Click
    option name is ``--from``, which would force ``ctx.invoke`` call sites into
    awkward ``**{"from": None}`` unpacking.
    """
    from crossby.config.loader import find_config_file
    from crossby.ui import prompts

    # Omit Init when .crossby.yml already exists. find_config_file() walks
    # upward from the given path, so Init is hidden whenever any ancestor
    # directory has .crossby.yml — intentional first-cut behaviour.
    has_config = find_config_file(Path.cwd()) is not None

    entries: list[tuple[str, str, str]] = [
        ("Launch", "Launch an AI tool", "crossby launch"),
        ("Sync", "Sync configs between tools", "crossby sync"),
        ("Handoff", "Hand off a session between tools", "crossby handoff"),
        ("Convert", "Translate allowlist patterns", "crossby convert"),
        ("Stats", "Parse session transcripts", "crossby stats"),
        ("Scene", "Activate a scene", "crossby scene"),
    ]
    if not has_config:
        entries.append(("Init", "Initialize .crossby.yml", "crossby init"))

    labels = [label for label, _desc, _hint in entries]
    hints = [hint for _label, _desc, hint in entries]

    idx = prompts.menu(
        "What would you like to do?",
        labels,
        hints=hints,
        version=f"crossby {crossby.__version__}",
    )
    label = entries[idx][0]

    if label == "Launch":
        launch(
            path=Path("."),
            tool=None,
            model=None,
            effort=None,
            yolo=None,
            plan=False,
            accept_edits=None,
            auto=None,
            command=None,
            prompt=None,
            complexity=None,
            transcript=None,
            profile=None,
            scene=None,
            resume=None,
            trusted_dirs=None,
        )
    elif label == "Sync":
        sync(
            concern=None,
            from_tool=None,
            to_tool=None,
            dry_run=False,
            force=False,
            path=Path("."),
        )
    elif label == "Handoff":
        handoff(
            from_tool=None,
            to_tool=None,
            session_id=None,
            output=None,
            no_launch=False,
            summarizer_tool=None,
            token_budget=32_000,
            prompt=None,
            prompt_preset="default",
            path=Path("."),
        )
    elif label == "Convert":
        pattern, from_tool, to_tool = _prompt_convert_args()
        convert(pattern=pattern, from_tool=from_tool, to_tool=to_tool)
    elif label == "Stats":
        transcript_path, tool = _prompt_stats_args()
        stats(transcript_path=transcript_path, tool=tool)
    elif label == "Scene":
        _run_scene_menu()
    elif label == "Init":
        init(path=Path("."), force=False, non_interactive=False)


def _run_scene_menu() -> None:
    """Dispatch a scene action from the interactive menu.

    Each arm calls a ``crossby.cli.scene`` subcommand function directly with
    every parameter spelled out, mirroring how the other menu entries dispatch.
    """
    from crossby.cli.scene import (
        clear_active,
        list_scenes,
        scene_status,
        show_scene,
        use_scene,
    )
    from crossby.ui import prompts

    actions = ["Use a scene", "Show a scene", "List scenes", "Status", "Clear active scene"]
    idx = prompts.select("Scene action", actions)
    if idx == 2:
        list_scenes(tool=None, path=Path("."))
        return
    if idx == 3:
        scene_status(tool=None, path=Path("."))
        return
    if idx == 4:
        clear_active(tool=None, plan=False, force=False, path=Path("."))
        return

    name = _prompt_scene_name()
    if name is None:
        return
    if idx == 0:
        use_scene(name=name, tool=None, plan=False, force=False, path=Path("."))
    else:
        show_scene(name=name, tool=None, path=Path("."))


def _prompt_scene_name() -> str | None:
    """Pick a scene name from ``.crossby.yml``; ``None`` when none are defined."""
    from crossby.config.loader import ConfigError, load_config
    from crossby.ui import prompts

    try:
        config = load_config(Path("."))
    except ConfigError:
        config = None
    names = sorted(config.scenes) if config is not None else []
    if not names:
        from crossby.ui.console import console

        console.info("No scenes defined in .crossby.yml.")
        return None
    idx = prompts.select("Scene", names)
    return names[idx]


def _prompt_convert_args() -> tuple[str, str, str]:
    """Collect the three required args for ``crossby convert`` inline."""
    from crossby.ui import prompts

    pattern = prompts.input_prompt("Pattern")
    tool_choices = ["canonical", "claude", "copilot", "cursor"]
    from_idx = prompts.select("Source tool", tool_choices)
    to_idx = prompts.select("Target tool", tool_choices)
    return pattern, tool_choices[from_idx], tool_choices[to_idx]


def _prompt_stats_args() -> tuple[Path, str | None]:
    """Collect transcript path + optional tool hint for ``crossby stats``."""
    from crossby.ai_tools.base import AbstractAITool
    from crossby.ui import prompts

    installed = [str(t) for t in AbstractAITool.detect_installed()]
    tool_choices = ["(auto-detect)", *installed]
    tool_idx = prompts.select("Tool", tool_choices)
    tool = None if tool_idx == 0 else tool_choices[tool_idx]
    transcript_str = prompts.input_prompt("Session path")
    return Path(transcript_str), tool


def cli_main() -> None:
    """Entry point for the ``crossby`` console script."""
    app()
