"""``crossby agents`` — subcommands for subagent management.

Currently exposes::

    crossby agents convert --from <tool> --to <tool> <input>

Translates a single subagent definition between any pair of supported tools.
With no ``--output`` the rendered payload goes to stdout; with ``--output``
it's written to disk.  Codex is the asymmetric case: it emits both an agent
``.toml`` and a config-fragment string for ``~/.codex/config.toml``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from crossby.subagents import (
    SUPPORTED_TOOLS,
    CodexEmission,
    ConversionWarning,
    WarningSeverity,
)
from crossby.subagents import convert as _convert
from crossby.ui.console import console

agents_app = typer.Typer(
    name="agents",
    help="Subagent utilities — convert definitions between tools.",
)


def _validate_tool(value: str, label: str) -> str:
    if value not in SUPPORTED_TOOLS:
        supported = ", ".join(SUPPORTED_TOOLS)
        console.error(f"Unknown {label} tool: {value!r}. Supported: {supported}")
        raise typer.Exit(1)
    return value


@agents_app.command("convert")
def convert_cmd(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Subagent file to translate.",
    ),
    from_tool: str = typer.Option(..., "--from", help="Source tool format."),
    to_tool: str = typer.Option(..., "--to", help="Target tool format."),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write to this path (file or directory). Stdout if omitted.",
    ),
) -> None:
    """Translate a single subagent file between tool formats.

    Examples::

        crossby agents convert --from claude --to gemini .claude/agents/researcher.md
        crossby agents convert --from codex --to copilot ~/.codex/agents/worker.toml
        crossby agents convert --from claude --to codex agent.md --output ./out/
    """
    _validate_tool(from_tool, "source")
    _validate_tool(to_tool, "target")

    content = input_path.read_text(encoding="utf-8")
    result = _convert(from_tool, to_tool, content, source_path=input_path)

    _emit_warnings(result.warnings)

    if isinstance(result.payload, CodexEmission):
        _write_codex(result.payload, output)
    else:
        _write_markdown(result.payload, result.ir.name, to_tool, output)


def _write_markdown(body: str, agent_name: str, to_tool: str, output: Path | None) -> None:
    if output is None:
        console.raw(body)
        return
    extension = ".agent.md" if to_tool == "copilot" else ".md"
    target = _resolve_target(output, agent_name, extension)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    console.success(f"Wrote {target}")


def _write_codex(emission: CodexEmission, output: Path | None) -> None:
    if output is None:
        # Render both artifacts to stdout, separated for clarity.
        console.raw(emission.agent_toml)
        console.raw("")
        console.raw("# --- config.toml fragment (append to ~/.codex/config.toml) ---")
        console.raw(emission.config_fragment)
        return

    if output.is_dir() or output.suffix == "":
        agent_path = output / emission.suggested_filename
        stem = emission.suggested_filename.removesuffix(".toml")
        fragment_path = output / f"{stem}.config-fragment.toml"
    else:
        agent_path = output
        fragment_path = output.with_suffix(".config-fragment.toml")

    agent_path.parent.mkdir(parents=True, exist_ok=True)
    agent_path.write_text(emission.agent_toml, encoding="utf-8")
    fragment_path.write_text(emission.config_fragment, encoding="utf-8")
    console.success(f"Wrote {agent_path}")
    console.success(f"Wrote {fragment_path} (merge into ~/.codex/config.toml)")


def _resolve_target(output: Path, agent_name: str, extension: str) -> Path:
    """If ``output`` is a directory, append ``<name><extension>``; else use it as-is."""
    if output.exists() and output.is_dir():
        return output / f"{agent_name}{extension}"
    if not output.exists() and output.suffix == "":
        # Treat extensionless non-existent paths as directories.
        return output / f"{agent_name}{extension}"
    return output


def _emit_warnings(warnings: list[ConversionWarning]) -> None:
    if not warnings:
        return
    for w in warnings:
        if w.severity is WarningSeverity.DROPPED:
            console.warn(str(w))
        else:
            console.info(str(w))
