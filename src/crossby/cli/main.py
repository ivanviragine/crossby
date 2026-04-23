"""CROSSBY CLI — main entry point."""

from __future__ import annotations

import typer

import crossby

app = typer.Typer(
    name="crossby",
    help="CROSSBY — Cross-platform Bridge for Your AI agents.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"crossby {crossby.__version__}")
        raise typer.Exit()


@app.callback()
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


# Register subcommands
from crossby.cli.convert import convert  # noqa: E402
from crossby.cli.handoff import handoff  # noqa: E402
from crossby.cli.launch import launch  # noqa: E402
from crossby.cli.stats import stats  # noqa: E402
from crossby.cli.sync import sync  # noqa: E402

app.command()(launch)
app.command()(sync)
app.command()(convert)
app.command()(stats)
app.command()(handoff)


def cli_main() -> None:
    """Entry point for the ``crossby`` console script."""
    app()
