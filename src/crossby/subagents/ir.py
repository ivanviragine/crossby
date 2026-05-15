"""Canonical intermediate representation for subagents.

The IR holds the union of fields observed across Claude, Cursor, Gemini,
Copilot, and Codex.  Each parser populates the subset its source format
exposes; each emitter reads the subset its target format understands and
emits structured warnings for fields it cannot represent.

Tool-specific fields that don't generalize (e.g. Copilot's ``target``,
Codex's ``nickname_candidates``) live in :attr:`SubagentIR.extras` so they
survive a round trip when the source and target are the same tool.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WarningSeverity(StrEnum):
    """Severity of a translation warning surfaced to the user."""

    INFO = "info"  # field carried verbatim, no semantic loss
    LOSSY = "lossy"  # field translated with reduced fidelity
    DROPPED = "dropped"  # field could not be represented at all


class ConversionWarning(BaseModel):
    """Structured note about something the conversion did or could not do."""

    model_config = ConfigDict(frozen=True)

    field: str
    severity: WarningSeverity
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.field}: {self.message}"


class SubagentIR(BaseModel):
    """Tool-agnostic subagent definition.

    All fields are optional except ``name`` and ``body`` — the bare minimum a
    target emitter can use to produce a working agent.  ``tools`` is a list of
    canonical lowercase snake_case names (see ``tool_map.CANONICAL_TOOLS``);
    parsers translate inbound names through the per-tool table.
    """

    model_config = ConfigDict(extra="forbid")

    # Required for any usable agent
    name: str
    body: str = ""  # system prompt / developer_instructions

    # Common metadata
    description: str | None = None
    model: str | None = None  # tool-specific id (e.g. "sonnet", "gpt-5.4")

    # Tool / permission allowlist (canonical names)
    tools: list[str] | None = None
    disallowed_tools: list[str] | None = None

    # MCP servers — opaque dict per server name; emitters serialize per-tool
    mcp_servers: dict[str, Any] | None = None

    # Behavioural knobs
    temperature: float | None = None  # Gemini
    max_turns: int | None = None  # Gemini, Claude
    timeout_mins: int | None = None  # Gemini
    effort: str | None = None  # Claude / Codex (low|medium|high|max)

    # Permission posture (cross-tool)
    sandbox_mode: str | None = None  # Codex: read-only|workspace-write
    readonly: bool | None = None  # Cursor
    permission_mode: str | None = None  # Claude: default|acceptEdits|plan|...

    # Tool-specific behaviour preserved when possible
    target: str | None = None  # Copilot: vscode|github-copilot
    user_invocable: bool | None = None  # Copilot
    disable_model_invocation: bool | None = None  # Copilot
    is_background: bool | None = None  # Cursor
    background: bool | None = None  # Claude
    kind: str | None = None  # Gemini: local|remote
    color: str | None = None  # Claude
    isolation: str | None = None  # Claude: worktree
    memory: str | None = None  # Claude: user|project|local
    skills: list[Any] | None = None  # Claude / Codex
    hooks: dict[str, Any] | None = None  # Claude
    initial_prompt: str | None = None  # Claude
    nickname_candidates: list[str] | None = None  # Codex
    metadata: dict[str, Any] | None = None  # Copilot

    # Provenance
    source_tool: str | None = None
    source_path: str | None = None

    # Escape hatch — fields the parser couldn't classify but wants to preserve.
    # Emitters only round-trip these into the same source tool.
    extras: dict[str, Any] = Field(default_factory=dict)
