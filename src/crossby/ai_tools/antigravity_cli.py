"""Antigravity CLI (agy) adapter — terminal agent, distinct from the Antigravity IDE."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from crossby.ai_tools.base import AbstractAITool
from crossby.models.ai import (
    AIToolCapabilities,
    AIToolID,
    AIToolType,
    EffortLevel,
    HookOutputDialect,
    HookStopDialect,
    TokenUsage,
)

# agy only exposes three effort tiers; xhigh/max both collapse to high.
_ANTIGRAVITY_CLI_EFFORT_MAP: dict[EffortLevel, str] = {
    EffortLevel.LOW: "low",
    EffortLevel.MEDIUM: "medium",
    EffortLevel.HIGH: "high",
    EffortLevel.XHIGH: "high",
    EffortLevel.MAX: "high",
}


class AntigravityCLIAdapter(AbstractAITool):
    """Adapter for Antigravity CLI (``agy``), the terminal surface of Google
    Antigravity 2.0. Not to be confused with ``AntigravityAdapter``, which
    launches the Antigravity IDE."""

    TOOL_ID: ClassVar[AIToolID] = AIToolID.ANTIGRAVITY_CLI

    def capabilities(self) -> AIToolCapabilities:
        return AIToolCapabilities(
            tool_id=AIToolID.ANTIGRAVITY_CLI,
            display_name="Antigravity CLI",
            binary="agy",
            tool_type=AIToolType.TERMINAL,
            supports_model_flag=True,
            # -p/--print/--prompt run a single prompt non-interactively and exit.
            headless_flag="--print",
            supports_headless=True,
            supports_effort=True,
            supports_yolo=True,
            supports_resume=True,
            supports_trusted_dirs=True,
            supports_plan_mode=True,
            supports_accept_edits=True,
            # agy exposes a Claude-style hook system (PreToolUse/PostToolUse/
            # Pre/PostInvocation/Stop). It reads decisions as a top-level
            # {"decision": …} object (the DECISION dialect) and fails *closed*
            # on a PreToolUse hook that errors — a non-zero exit denies the tool
            # call (observed in agy integrations, e.g. cmux issue #4768 where a
            # failing PreToolUse hook blocks every tool call) — so
            # hook_fail_open_default stays False.
            supports_stop_hook=True,
            hook_output_dialect=HookOutputDialect.DECISION,
            # Inverted polarity vs every other tool: agy blocks a Stop by
            # telling the agent to *continue*. Its PreToolUse vocabulary is
            # allow/deny/ask/force_ask — "continue" is a Stop-only word there,
            # and "block" is not a word agy knows at all (it errors with
            # `unknown pre-tool hook decision "block"`).
            hook_stop_dialect=HookStopDialect.CONTINUE_DECISION,
            hook_fail_open_default=False,
            # sandboxes_writes stays False deliberately: agy's write sandbox is an
            # opt-in flag (``--sandbox``, passed only in yolo_args), not a verified
            # default, so we do NOT tell wade an out-of-worktree write is already
            # confined — wade keeps its own containment guard rather than trusting
            # an unconfirmed native sandbox. agy's own bundled plugin registers no
            # PreToolUse hook, so that guard is best-effort there; Stop is the
            # reliable enforcement surface.
            sandboxes_writes=False,
        )

    def initial_message_args(self, prompt: str) -> list[str]:
        """``--prompt-interactive`` runs an initial prompt interactively and
        continues the session — the interactive-launch equivalent of an
        initial message."""
        return ["--prompt-interactive", prompt]

    def plan_mode_args(self) -> list[str]:
        """agy's ``--mode`` flag accepts ``accept-edits`` or ``plan``."""
        return ["--mode", "plan"]

    def accept_edits_args(self) -> list[str]:
        """agy's ``--mode accept-edits`` auto-applies edits on the execution-mode
        axis; shell stays gated by the separate permissions axis."""
        return ["--mode", "accept-edits"]

    def plan_dir_args(self, plan_dir: str) -> list[str]:
        """agy uses --add-dir (repeatable) to grant workspace access."""
        return ["--add-dir", plan_dir]

    def yolo_args(self) -> list[str]:
        """Skip permission prompts while keeping the terminal sandbox active —
        mirrors Codex's philosophy of skipping approvals without removing the
        safety sandbox."""
        return ["--dangerously-skip-permissions", "--sandbox"]

    def build_resume_command(self, session_id: str) -> list[str] | None:
        """Resume a specific Antigravity CLI conversation by ID."""
        return ["agy", "--conversation", session_id]

    def effort_args(self, effort: EffortLevel) -> list[str]:
        """agy's ``--effort`` only accepts low|medium|high."""
        mapped = _ANTIGRAVITY_CLI_EFFORT_MAP.get(effort, effort.value)
        return ["--effort", mapped]

    def parse_transcript(self, transcript_path: Path) -> TokenUsage:
        """Antigravity CLI persists conversations as opaque per-conversation
        SQLite databases (protobuf-like blob columns) under
        ``~/.gemini/antigravity-cli/conversations/`` — verified locally via
        ``agy --print`` + inspecting that directory, this is the real
        install path (Antigravity CLI is a Gemini-family product, hence the
        ``~/.gemini/`` prefix), not a leftover Gemini-CLI reference. Not
        parseable text, so this mirrors the known Gemini-CLI
        transcript-persistence limitation for a different underlying reason."""
        return TokenUsage()
