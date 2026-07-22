"""AI tool domain models — AIToolID, AIModel, ModelTier, TokenUsage."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel


class EffortLevel(StrEnum):
    """Reasoning effort / thinking depth level for AI tools."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class AIToolID(StrEnum):
    """Canonical identifiers for all supported AI tools."""

    CLAUDE = "claude"
    COPILOT = "copilot"
    CODEX = "codex"
    ANTIGRAVITY = "antigravity"
    ANTIGRAVITY_CLI = "antigravity-cli"
    VSCODE = "vscode"
    OPENCODE = "opencode"
    CURSOR = "cursor"


class AIToolType(StrEnum):
    """How the AI tool runs."""

    TERMINAL = "terminal"
    GUI = "gui"


class ModelTier(StrEnum):
    """Capability tier — maps to complexity levels for auto-selection."""

    FAST = "fast"
    BALANCED = "balanced"
    POWERFUL = "powerful"


class HookOutputDialect(StrEnum):
    """How a tool expects a hook to signal an allow/deny/context decision.

    Grouped by output *shape*, not by tool — several tools share one shape:

    - ``HOOK_SPECIFIC_OUTPUT`` — a ``{"hookSpecificOutput": {...}}`` object on
      stdout carrying ``permissionDecision`` / ``additionalContext`` (Claude,
      Codex).
    - ``PERMISSION`` — a ``{"permission": "allow"|"deny", ...}`` object on
      stdout (Cursor).
    - ``EXIT_CODE`` — no structured stdout contract; the exit code is the only
      block signal, with a human message on stderr (Copilot).

    A deny always also exits non-zero (2) so the block is honored even by tools
    that ignore stdout — the dialect only governs the stdout payload shape.
    """

    HOOK_SPECIFIC_OUTPUT = "hook_specific_output"
    PERMISSION = "permission"
    EXIT_CODE = "exit_code"


class AIModel(BaseModel, frozen=True):
    """A concrete model available through an AI tool.

    Models come from a bundled static registry (``data/models.json``).
    The model ID format matches what each tool's CLI accepts.
    """

    id: str
    display_name: str | None = None
    tier: ModelTier | None = None
    is_alias: bool = False

    def __str__(self) -> str:
        return self.id


class AIToolCapabilities(BaseModel, frozen=True):
    """What an AI tool can do — declared by each adapter."""

    tool_id: AIToolID
    display_name: str
    binary: str
    tool_type: AIToolType
    supports_model_flag: bool = True
    model_flag: str = "--model"
    headless_flag: str | None = None
    supports_headless: bool = False
    supports_initial_message: bool = True
    blocks_until_exit: bool = True
    supports_effort: bool = False
    supports_yolo: bool = False
    supports_resume: bool = False
    supports_trusted_dirs: bool = False
    supports_plan_mode: bool = False

    # --- Hook lifecycle & runtime I/O (consumed by crossby.hooks.runtime) ---
    supports_stop_hook: bool = False
    """Tool fires a Stop / agent-turn-complete hook that can block completion."""
    supports_session_start_hook: bool = False
    """Tool fires a SessionStart hook that can inject additional context."""
    supports_user_prompt_submit_hook: bool = False
    """Tool fires a prompt-submit hook that can inject context (Claude/Codex
    ``UserPromptSubmit``, Cursor ``beforeSubmitPrompt``)."""
    sandboxes_writes: bool = False
    """Tool hard-confines file writes to its trusted/workspace dirs (e.g. Codex
    ``--sandbox workspace-write``). When True, an out-of-worktree write is
    already blocked by the runtime, so a wade worktree-containment guard hook is
    redundant. Distinct from ``supports_trusted_dirs`` (which only means the tool
    accepts a trusted-dir flag; Claude adds dirs but still prompts rather than
    hard-blocks)."""
    hook_output_dialect: HookOutputDialect = HookOutputDialect.HOOK_SPECIFIC_OUTPUT
    """Which stdout shape this tool reads a hook decision from."""
    hook_fail_open_default: bool = False
    """Tool treats a hook that errors/crashes as *allow* (fail-open) unless the
    hook config opts into fail-closed. True for Cursor — callers writing a
    security guard must set the tool's fail-closed flag when this is True."""
    supports_usage_reporting: bool = False
    """Tool emits structured token usage in headless output (``--output-format
    json`` / ``codex exec --json``), so usage need not be scraped from a
    transcript log. False for Cursor (no usage fields in CLI output)."""


class TokenUsage(BaseModel):
    """Token usage metrics from an AI session."""

    total_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    premium_requests: int | None = None
    model_breakdown: list[ModelBreakdown] = []
    raw_transcript_path: Path | None = None
    session_id: str | None = None  # full resume command or session ID as printed by the tool


class ModelBreakdown(BaseModel):
    """Per-model token usage within a session."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    premium_requests: int = 0
