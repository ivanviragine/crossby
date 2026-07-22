"""Public façade for subagent translation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crossby.subagents.emitters import emit as _emit
from crossby.subagents.ir import ConversionWarning, SubagentIR
from crossby.subagents.parsers import parse as _parse

SUPPORTED_TOOLS: tuple[str, ...] = ("claude", "cursor", "copilot", "codex")


@dataclass
class ConversionResult:
    """Result of a one-shot ``convert(from_tool, to_tool, content)`` call.

    ``payload`` is a string for markdown emitters and a
    :class:`crossby.subagents.emitters.CodexEmission` for Codex.  Inspect
    ``ir.source_tool`` if you need to know what the input was.
    """

    ir: SubagentIR
    payload: Any
    warnings: list[ConversionWarning]
    target_tool: str


def parse(
    tool: str, content: str, source_path: Path | None = None
) -> tuple[SubagentIR, list[ConversionWarning]]:
    """Parse a tool-specific subagent file into the canonical IR."""
    return _parse(tool, content, source_path)


def emit(tool: str, ir: SubagentIR) -> tuple[Any, list[ConversionWarning]]:
    """Emit ``ir`` in the target tool's format."""
    return _emit(tool, ir)


def convert(
    from_tool: str,
    to_tool: str,
    content: str,
    source_path: Path | None = None,
) -> ConversionResult:
    """Translate ``content`` from one tool's format to another's.

    Warnings from both stages are concatenated. Same-tool conversion is a
    valid (and useful) round-trip — call sites can treat it as a normalize.
    """
    ir, parse_warnings = parse(from_tool, content, source_path)
    payload, emit_warnings = emit(to_tool, ir)
    return ConversionResult(
        ir=ir,
        payload=payload,
        warnings=[*parse_warnings, *emit_warnings],
        target_tool=to_tool,
    )
