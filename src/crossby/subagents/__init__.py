"""Subagent format translation — canonical IR + parsers/emitters per tool.

Public API::

    from crossby.subagents import parse, emit, convert, SUPPORTED_TOOLS

    ir, warnings = parse("claude", path.read_text(), source_path=path)
    out, warnings = emit("codex", ir)        # out is a CodexEmission for codex
    result = convert("claude", "cursor", path.read_text())
"""

from __future__ import annotations

from crossby.subagents.api import (
    SUPPORTED_TOOLS,
    ConversionResult,
    convert,
    emit,
    parse,
)
from crossby.subagents.emitters import CodexEmission
from crossby.subagents.ir import (
    ConversionWarning,
    SubagentIR,
    WarningSeverity,
)

__all__ = [
    "SUPPORTED_TOOLS",
    "CodexEmission",
    "ConversionResult",
    "ConversionWarning",
    "SubagentIR",
    "WarningSeverity",
    "convert",
    "emit",
    "parse",
]
