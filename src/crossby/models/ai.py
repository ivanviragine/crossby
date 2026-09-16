"""AI tool domain models — AIToolID, AIModel, ModelTier, TokenUsage."""

from __future__ import annotations

import json
import math
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

_HEADLESS_EVENT_LIMIT = 4096
_HEADLESS_EVENT_MESSAGE_LIMIT = 64 * 1024
_HEADLESS_EVENT_PAYLOAD_LIMIT = 256 * 1024
_HEADLESS_FINAL_PAYLOAD_LIMIT = 8 * 1024 * 1024
_HEADLESS_DIAGNOSTIC_LIMIT = 16 * 1024


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


class PlanModeActivation(StrEnum):
    """How an adapter activates a harness's native planning mode."""

    CLI_ARGUMENT = "cli_argument"
    TERMINAL_INPUT = "terminal_input"
    CODEX_APP_SERVER = "codex_app_server"
    OPENCODE_SERVER = "opencode_server"
    ACP = "acp"
    UNSUPPORTED = "unsupported"


class PlanLaunchApprovalMode(StrEnum):
    """Approval options that can coexist with native interactive planning."""

    YOLO = "yolo"
    AUTO = "auto"
    ACCEPT_EDITS = "accept_edits"


class InteractiveLaunchEventKind(StrEnum):
    """Startup milestones, independent of a tool's terminal rendering."""

    PLAN_READY = "plan_ready"
    MESSAGE_SUBMITTED = "message_submitted"


class InteractiveLaunchEvent(BaseModel):
    """An observed startup milestone; not a model turn or plan artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: InteractiveLaunchEventKind
    tool_id: AIToolID


class PlanArtifactLocation(StrEnum):
    """Where a harness can persist artifacts created by native plan mode."""

    REQUESTED_PATH = "requested_path"
    WORKSPACE_MANAGED = "workspace_managed"
    PRIVATE = "private"
    SESSION = "session"
    UNAVAILABLE = "unavailable"


class PlanArtifactSource(StrEnum):
    """Authoritative source from which Crossby collected a native plan."""

    REQUESTED_PATH = "requested_path"
    STRUCTURED_OUTPUT = "structured_output"
    SESSION_EXPORT = "session_export"
    PROTOCOL_EVENT = "protocol_event"


class PlanSessionBinding(StrEnum):
    """Evidence that binds an artifact to the run Crossby launched."""

    ISOLATED_RUN_PATH = "isolated_run_path"
    SESSION_ID = "session_id"
    CONVERSATION_ID = "conversation_id"
    THREAD_TURN_IDS = "thread_turn_ids"


class PlanSessionTransport(StrEnum):
    """Transport used for a complete, collected planning session."""

    INTERACTIVE_CLI = "interactive_cli"
    HEADLESS_CLI = "headless_cli"
    OPENCODE_SERVER = "opencode_server"
    CODEX_APP_SERVER = "codex_app_server"
    ACP = "acp"
    UNAVAILABLE = "unavailable"


class PlanInteractionSupport(StrEnum):
    """How native planning questions are surfaced to a caller."""

    TERMINAL = "terminal"
    CALLBACK = "callback"
    RESUMABLE_CALLBACK = "resumable_callback"
    NONE = "none"


class PlanRequestBehavior(StrEnum):
    """How a collector treats a sandbox or approval request dimension."""

    PRESERVED = "preserved"
    TOOL_MANAGED = "tool_managed"
    UNSUPPORTED = "unsupported"


class PlanCommandPolicySupport(StrEnum):
    """How a collector preserves session-scoped command preauthorization."""

    NATIVE = "native"
    CALLBACK = "callback"
    UNSUPPORTED = "unsupported"


class PlanApprovalPolicy(StrEnum):
    """Portable approval posture for collected planning sessions."""

    ON_REQUEST = "on-request"
    UNTRUSTED = "untrusted"
    NEVER = "never"


class PlanInteractionKind(StrEnum):
    """Kind of native elicitation surfaced during a plan session."""

    QUESTION = "question"
    PERMISSION = "permission"
    PLAN_APPROVAL = "plan_approval"


class PlanInteractionOutcome(StrEnum):
    """Caller-selected outcome for a native elicitation."""

    ANSWERED = "answered"
    APPROVED = "approved"
    DENIED = "denied"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class PlanOperationKind(StrEnum):
    """Authoritative native operation category, when a provider supplies one."""

    COMMAND = "command"
    FILE_CHANGE = "file_change"
    NETWORK = "network"
    MCP = "mcp"
    WRITE_STDIN = "write_stdin"
    OTHER = "other"


class PlanPermissionTargetKind(StrEnum):
    """Kind of resource covered by a native permission request."""

    COMMAND_PATTERN = "command_pattern"
    FILESYSTEM_READ = "filesystem_read"
    FILESYSTEM_WRITE = "filesystem_write"
    FILESYSTEM_DENY = "filesystem_deny"
    NETWORK_HOST = "network_host"
    RESOURCE = "resource"


class HeadlessInteractionMode(StrEnum):
    """How an ordinary managed session resolves native elicitation."""

    UNATTENDED = "unattended"
    BROKERED = "brokered"


class HeadlessNativeTransport(StrEnum):
    """Native transport owned by an adapter's managed-session hook."""

    SUBPROCESS = "subprocess"
    HEADLESS_CLI = "headless_cli"
    JSON_RPC = "json_rpc"
    SERVER = "server"
    HTTP_SERVER = "http_server"
    APP_SERVER = "app_server"
    ACP = "acp"
    SDK = "sdk"
    UNAVAILABLE = "unavailable"


class HeadlessNativeOutput(StrEnum):
    """Native output stream selected independently from response validation."""

    TEXT = "text"
    JSON = "json"
    JSONL = "jsonl"


class HeadlessPromptTransport(StrEnum):
    """How a prompt is delivered without transferring process ownership."""

    STDIN = "stdin"
    PROTOCOL = "protocol"
    API = "api"
    ARGUMENT = "argument"


class HeadlessEventKind(StrEnum):
    """Small normalized event vocabulary exposed to session consumers."""

    STARTED = "started"
    PROGRESS = "progress"
    OUTPUT = "output"
    INTERACTION = "interaction"
    WARNING = "warning"
    ERROR = "error"
    TERMINAL = "terminal"


class HeadlessTerminalStatus(StrEnum):
    """Normalized terminal outcomes for managed headless sessions."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    INVALID_OUTPUT = "invalid_output"


class HeadlessPreflightCheck(StrEnum):
    """Request facts established before a managed transport starts."""

    SESSION_CAPABILITY = "session_capability"
    REQUEST_COMPATIBILITY = "request_compatibility"
    RESPONSE_SCHEMA = "response_schema"
    CLI_VERSION = "cli_version"


class HeadlessPreflightDeferredCheck(StrEnum):
    """Facts that must be repeated or established by the live transport."""

    FILESYSTEM = "filesystem"
    AUTHENTICATION = "authentication"
    MODEL_AVAILABILITY = "model_availability"
    TRANSPORT_STARTUP = "transport_startup"
    PROTOCOL_NEGOTIATION = "protocol_negotiation"
    OUTPUT_COLLECTION = "output_collection"


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
    - ``PERMISSION_DECISION`` — a **flat, top-level**
      ``{"permissionDecision": "allow"|"deny"|"ask", "permissionDecisionReason":
      …}`` object on stdout (Copilot). Same field *names* as the payload nested
      inside ``HOOK_SPECIFIC_OUTPUT``, but never nested:
      ``hookSpecificOutput`` appears nowhere in GitHub's hooks docs — it is a
      Claude/VS Code construct. ``permissionDecisionReason`` is required on a
      deny. Copilot strips ``{"type":"progress"}`` lines and then runs a single
      ``JSON.parse``, so a hook must emit exactly *one* JSON object.
    - ``EXIT_CODE`` — no structured stdout contract; the exit code is the only
      block signal, with a human message on stderr. No tool crossby models uses
      this today (Copilot moved to ``PERMISSION_DECISION`` once its documented
      stdout schema was confirmed); kept for tools that genuinely have no stdout
      channel.
    - ``DECISION`` — a ``{"decision": "deny"|"allow"|"ask", "reason": …}`` object
      on stdout, with a Stop hook blocking via ``{"decision": "continue"}``
      (Antigravity CLI / ``agy``). Field names are top-level, and the shape is
      **per-event**: on **PreToolUse** ``decision`` is **required** — a payload
      with none (a bare ``{}``) is read as a *deny*, so allow/context emit an
      explicit ``{"decision": "allow"}`` — but **PostToolUse** expects a bare
      ``{}`` (no decision field), since the call already ran and cannot be gated.

    A deny exits non-zero (2) on every dialect **except** ``DECISION``, so the
    block is honored even by tools that ignore stdout and a security guard stays
    fail-*closed*. ``DECISION`` (agy) is the exception: agy reads a non-zero exit
    as a hook *crash* (raw stderr surfaced, stdout discarded), so its deny is
    **exit 0** and fail-closed is carried by the structured
    ``{"decision": "deny"}`` on stdout, per agy's contract. Otherwise the dialect
    only governs the stdout payload shape.

    This covers the *tool-call* channel only. A tool's Stop channel is
    independent and is declared separately as :class:`HookStopDialect` — Copilot
    is the proof that one enum cannot serve both, since its PreToolUse is the
    flat permission shape while its stop is ``{"decision": "block", …}``.
    """

    HOOK_SPECIFIC_OUTPUT = "hook_specific_output"
    PERMISSION = "permission"
    PERMISSION_DECISION = "permission_decision"
    EXIT_CODE = "exit_code"
    DECISION = "decision"


class HookStopDialect(StrEnum):
    """How a tool expects a hook to block (or allow) the *end of a turn*.

    Deliberately separate from :class:`HookOutputDialect`: a tool's Stop channel
    does not follow from its tool-call channel. Copilot reads a flat
    ``permissionDecision`` for PreToolUse but a ``{"decision": "block"}`` for
    ``agentStop``, so threading one enum through both would have to special-case
    the tool anyway.

    - ``BLOCK_DECISION`` — ``{"decision": "block", "reason": …}`` (Claude, Codex,
      Copilot).
    - ``FOLLOWUP_MESSAGE`` — ``{"followup_message": …}``, auto-submitted back to
      the agent (Cursor). Re-fires are bounded by the tool's own ``loop_limit``.
    - ``CONTINUE_DECISION`` — ``{"decision": "continue", "reason": …}`` (agy).
      Note the inverted polarity: agy blocks a stop by telling the agent to
      *continue*, where the other dialects use a top-level ``continue`` boolean
      as the no-op.
    - ``NONE`` — the tool has no Stop-block channel; a Stop hook cannot block it.

    Unlike a PreToolUse deny, a Stop decision never exits 2 — the Stop channel
    stays fail-*open* on purpose, so a broken guard can never trap the agent in
    a turn it cannot end.
    """

    BLOCK_DECISION = "block_decision"
    FOLLOWUP_MESSAGE = "followup_message"
    CONTINUE_DECISION = "continue_decision"
    NONE = "none"


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


class PlanQuestionOption(BaseModel):
    """One native option presented by a planning harness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    option_id: str
    label: str
    description: str | None = None

    @field_validator("option_id", "label")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("plan question options must have non-blank identifiers and labels")
        return value


class PlanPermissionTarget(BaseModel):
    """One authoritative resource named by a native permission request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: PlanPermissionTargetKind
    value: str

    @field_validator("value")
    @classmethod
    def _value_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("permission targets must be non-blank")
        return value


class PlanNativeBindingID(BaseModel):
    """One provider-defined identifier binding operation evidence to a session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    value: str

    @field_validator("name", "value")
    @classmethod
    def _binding_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("native binding names and values must be non-blank")
        return value


class PlanOperation(BaseModel):
    """Structured native evidence for an operation awaiting permission.

    ``argv`` and ``shell_expression`` are deliberately distinct. Providers that
    only supply display prose leave both unset; callers must never reconstruct
    executable input from :attr:`PlanInteraction.prompt`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: PlanOperationKind
    argv: tuple[str, ...] | None = None
    shell_expression: str | None = None
    execution_dir: Path | None = None
    permission_targets: tuple[PlanPermissionTarget, ...] = ()
    native_binding_ids: tuple[PlanNativeBindingID, ...] = ()

    @model_validator(mode="after")
    def _command_representation_is_unambiguous(self) -> Self:
        if self.argv is not None and self.shell_expression is not None:
            raise ValueError("an operation cannot contain both argv and a shell expression")
        if self.argv is not None and (not self.argv or not self.argv[0].strip()):
            raise ValueError("authoritative argv requires a non-blank executable")
        if self.shell_expression is not None and not self.shell_expression.strip():
            raise ValueError("authoritative shell expressions must be non-blank")
        return self


class PlanInteraction(BaseModel):
    """A session-bound native question or approval request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: PlanInteractionKind
    question_id: str
    prompt: str
    options: tuple[PlanQuestionOption, ...] = ()
    allow_multiple: bool = False
    allow_other: bool = False
    session_id: str
    thread_id: str | None = None
    turn_id: str | None = None
    conversation_id: str | None = None
    artifact_id: str | None = None
    operation: PlanOperation | None = None

    @field_validator("question_id", "prompt", "session_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("plan interactions require non-blank identifiers and prompt text")
        return value


class PlanInteractionResponse(BaseModel):
    """A caller's explicit response to one native interaction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: PlanInteractionOutcome
    answer: str | None = None
    option_id: str | None = None
    option_ids: tuple[str, ...] = ()

    @field_validator("option_ids")
    @classmethod
    def _option_ids_are_not_blank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not option_id.strip() for option_id in value):
            raise ValueError("native option IDs must be non-blank")
        return value

    @model_validator(mode="after")
    def _answer_matches_outcome(self) -> Self:
        if self.outcome is PlanInteractionOutcome.ANSWERED and not (
            (self.answer and self.answer.strip())
            or (self.option_id and self.option_id.strip())
            or self.option_ids
        ):
            raise ValueError("an answered interaction requires answer text or a native option ID")
        return self


# The interaction contract originated in collected Plan mode.  These neutral
# names are now canonical for ordinary managed sessions while the Plan-prefixed
# names above remain source-compatible aliases for existing consumers.
SessionQuestionOption = PlanQuestionOption
SessionOption = PlanQuestionOption
SessionPermissionTarget = PlanPermissionTarget
SessionTarget = PlanPermissionTarget
SessionNativeBindingID = PlanNativeBindingID
SessionBinding = PlanNativeBindingID
SessionOperation = PlanOperation
SessionInteraction = PlanInteraction
SessionInteractionResponse = PlanInteractionResponse
SessionInteractionKind = PlanInteractionKind
SessionInteractionOutcome = PlanInteractionOutcome
SessionOperationKind = PlanOperationKind
SessionPermissionTargetKind = PlanPermissionTargetKind


class HeadlessSessionRequest(BaseModel):
    """Portable inputs for one complete, managed ordinary agent session.

    ``native_output`` selects the adapter's wire/output dialect.  It does not
    imply structured response validation; ``response_schema`` is an independent
    caller requirement and may be supported over any declared native output.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    prompt: str
    working_dir: Path
    model: str | None = None
    effort: EffortLevel | None = None
    trusted_dirs: tuple[Path, ...] = ()
    sandbox: bool = True
    network_access: bool = False
    approval_policy: PlanApprovalPolicy = PlanApprovalPolicy.ON_REQUEST
    command_policy: PlanCommandPolicy | None = None
    interaction_mode: HeadlessInteractionMode = HeadlessInteractionMode.UNATTENDED
    native_output: HeadlessNativeOutput = Field(
        default=HeadlessNativeOutput.TEXT,
        validation_alias=AliasChoices(
            "native_output",
            "native_output_mode",
            "output_mode",
            "output",
        ),
    )
    response_schema: dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("response_schema", "json_schema"),
    )
    resume_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("resume_id", "resume_session_id"),
    )
    timeout_seconds: float = 600.0
    idle_timeout_seconds: float | None = None
    interaction_timeout_seconds: float = 60.0

    @field_validator("prompt")
    @classmethod
    def _prompt_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a headless session requires a non-blank prompt")
        return value

    @field_validator("model", "resume_id")
    @classmethod
    def _optional_text_is_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("provided headless-session identifiers must be non-blank")
        return value

    @field_validator("timeout_seconds", "interaction_timeout_seconds")
    @classmethod
    def _required_timeout_is_positive_and_finite(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("headless-session timeouts must be positive and finite")
        return value

    @field_validator("idle_timeout_seconds")
    @classmethod
    def _optional_timeout_is_positive_and_finite(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("headless-session idle timeout must be positive and finite")
        return value

    @property
    def native_output_mode(self) -> HeadlessNativeOutput:
        """Compatibility spelling for callers that include ``mode``."""
        return self.native_output

    @property
    def output_mode(self) -> HeadlessNativeOutput:
        """Concise compatibility spelling for ``native_output``."""
        return self.native_output

    @property
    def json_schema(self) -> dict[str, Any] | None:
        return self.response_schema

    @property
    def resume_session_id(self) -> str | None:
        return self.resume_id


class HeadlessEvent(BaseModel):
    """One bounded, normalized event emitted in deterministic sequence order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1)
    kind: HeadlessEventKind
    message: str | None = None
    payload: dict[str, Any] | None = None
    terminal_status: HeadlessTerminalStatus | None = None
    elapsed_seconds: float = Field(default=0.0, ge=0)
    session_id: str | None = None
    thread_id: str | None = None
    turn_id: str | None = None
    conversation_id: str | None = None

    @field_validator("message", "session_id", "thread_id", "turn_id", "conversation_id")
    @classmethod
    def _provided_text_is_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("provided headless-event text must be non-blank")
        if value is not None and len(value.encode("utf-8")) > _HEADLESS_EVENT_MESSAGE_LIMIT:
            raise ValueError("headless event text exceeded the fixed size limit")
        return value

    @field_validator("elapsed_seconds")
    @classmethod
    def _elapsed_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("headless event elapsed time must be finite")
        return value

    @model_validator(mode="after")
    def _terminal_status_matches_kind(self) -> Self:
        if (self.kind is HeadlessEventKind.TERMINAL) != (self.terminal_status is not None):
            raise ValueError("only terminal events carry terminal_status")
        if self.payload is not None:
            try:
                encoded = json.dumps(self.payload, allow_nan=False).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ValueError("headless event payloads must be finite JSON objects") from exc
            if len(encoded) > _HEADLESS_EVENT_PAYLOAD_LIMIT:
                raise ValueError("headless event payload exceeded the fixed size limit")
        return self


class PlanSessionRequest(BaseModel):
    """Portable inputs for one complete native planning lifecycle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str
    working_dir: Path
    model: str | None = None
    effort: EffortLevel | None = None
    trusted_dirs: tuple[Path, ...] = ()
    plan_output_dir: Path | None = None
    sandbox: bool = True
    network_access: bool = False
    approval_policy: PlanApprovalPolicy = PlanApprovalPolicy.ON_REQUEST
    command_policy: PlanCommandPolicy | None = None
    timeout_seconds: float = Field(default=600.0, gt=0, le=3600.0)

    @field_validator("prompt")
    @classmethod
    def _prompt_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a collected plan session requires a non-blank prompt")
        return value


class PlanCommandPolicy(BaseModel):
    """Canonical commands preauthorized for one collected planning session.

    Patterns use Crossby's existing command grammar. This contract accepts
    exact commands and a single trailing ``:*`` or `` *`` wildcard. Compound
    shell syntax and other wildcard placements are rejected because they do not
    have portable, safe semantics across the supported collectors.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_commands: tuple[str, ...] = Field(min_length=1)

    @field_validator("allowed_commands")
    @classmethod
    def _commands_are_portable(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        from crossby.ai_tools.plan_policy import validate_plan_command_pattern

        if len(set(value)) != len(value):
            raise ValueError("command policy patterns must be unique")
        for pattern in value:
            validate_plan_command_pattern(pattern)
        return value


SessionApprovalPolicy = PlanApprovalPolicy
SessionCommandPolicy = PlanCommandPolicy
SessionCommandPolicySupport = PlanCommandPolicySupport


class PlanPreflightCheck(StrEnum):
    """Statically validated part of a collected-session request."""

    SESSION_CAPABILITY = "session_capability"
    REQUEST_COMPATIBILITY = "request_compatibility"
    COMMAND_POLICY = "command_policy"
    CLI_VERSION = "cli_version"


class PlanPreflightDeferredCheck(StrEnum):
    """Runtime fact that successful static preflight does not promise."""

    FILESYSTEM = "filesystem"
    AUTHENTICATION = "authentication"
    MODEL_AVAILABILITY = "model_availability"
    PROTOCOL_NEGOTIATION = "protocol_negotiation"
    ARTIFACT_COLLECTION = "artifact_collection"


class PlanSessionPreflight(BaseModel, frozen=True):
    """Evidence returned after bounded, non-mutating session preflight."""

    tool: AIToolID
    detected_version: str
    normalized_version: tuple[int, int, int]
    capability: PlanModeCapability
    checked: tuple[PlanPreflightCheck, ...]
    deferred: tuple[PlanPreflightDeferredCheck, ...]


class PlanSessionResult(BaseModel, frozen=True):
    """Normalized Markdown plan plus exact native provenance."""

    tool: AIToolID
    version: str
    plan: str
    session_id: str
    native_mode: str
    artifact_source: PlanArtifactSource
    binding: PlanSessionBinding
    exit_code: int
    thread_id: str | None = None
    turn_id: str | None = None
    artifact_id: str | None = None
    artifact_path: Path | None = None

    @field_validator("version", "plan", "session_id", "native_mode")
    @classmethod
    def _non_blank_result_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("successful plan-session text and provenance must be non-blank")
        return value

    @field_validator("thread_id", "turn_id", "artifact_id")
    @classmethod
    def _optional_provenance_is_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("provided plan-session provenance IDs must be non-blank")
        return value

    @model_validator(mode="after")
    def _provenance_is_consistent(self) -> Self:
        if self.exit_code != 0:
            raise ValueError("a successful plan-session result requires exit_code=0")
        if self.binding is PlanSessionBinding.THREAD_TURN_IDS and not (
            self.thread_id and self.turn_id
        ):
            raise ValueError("thread/turn binding requires both thread_id and turn_id")
        if self.binding is PlanSessionBinding.ISOLATED_RUN_PATH and self.artifact_path is None:
            raise ValueError("isolated-path binding requires artifact_path")
        if self.artifact_source is PlanArtifactSource.REQUESTED_PATH and self.artifact_path is None:
            raise ValueError("requested-path artifact provenance requires artifact_path")
        if self.artifact_source is PlanArtifactSource.PROTOCOL_EVENT and self.artifact_id is None:
            raise ValueError("protocol-event provenance requires artifact_id")
        expected_bindings = {
            PlanArtifactSource.REQUESTED_PATH: {PlanSessionBinding.ISOLATED_RUN_PATH},
            PlanArtifactSource.STRUCTURED_OUTPUT: {PlanSessionBinding.CONVERSATION_ID},
            PlanArtifactSource.SESSION_EXPORT: {PlanSessionBinding.SESSION_ID},
            PlanArtifactSource.PROTOCOL_EVENT: {
                PlanSessionBinding.SESSION_ID,
                PlanSessionBinding.THREAD_TURN_IDS,
            },
        }
        if self.binding not in expected_bindings[self.artifact_source]:
            raise ValueError(
                f"{self.artifact_source.value} is inconsistent with {self.binding.value} binding"
            )
        return self


class PlanModeCapability(BaseModel, frozen=True):
    """Truthful contract for activation and complete plan collection.

    ``version_requirement`` deliberately accepts a human-readable selector
    requirement instead of pretending every upstream publishes a reliable
    numeric introduction version. ``verified_version`` records the oldest
    concrete CLI build Crossby verified and is the conservative runtime floor.
    ``collector_verified_version`` overrides that floor for a separate collector
    transport when its verified version differs from interactive startup.
    Activation-only launches remain available through :attr:`supported`; a
    complete collected session additionally requires ``artifact_source`` and an
    exact ``binding``.
    """

    activation: PlanModeActivation
    activation_detail: str
    version_requirement: str
    verified_version: str | None = None
    collector_verified_version: str | None = None
    initial_prompt_after_activation: bool
    artifact_location: PlanArtifactLocation
    artifact_location_detail: str
    artifact_path_template: str | None = None
    export_command: tuple[str, ...] | None = None
    import_command: tuple[str, ...] | None = None
    remediation: str | None = None
    supported_launch_approval_modes: tuple[PlanLaunchApprovalMode, ...] = ()
    supports_ready_event: bool = False
    collector_activation: PlanModeActivation | None = None
    transport: PlanSessionTransport = PlanSessionTransport.UNAVAILABLE
    artifact_source: PlanArtifactSource | None = None
    binding: PlanSessionBinding | None = None
    interaction: PlanInteractionSupport = PlanInteractionSupport.NONE
    sandbox_behavior: PlanRequestBehavior = PlanRequestBehavior.UNSUPPORTED
    approval_behavior: PlanRequestBehavior = PlanRequestBehavior.UNSUPPORTED
    supported_approval_policies: tuple[PlanApprovalPolicy, ...] = (PlanApprovalPolicy.ON_REQUEST,)
    command_policy_support: PlanCommandPolicySupport = PlanCommandPolicySupport.UNSUPPORTED
    command_policy_detail: str = (
        "No verified session-scoped command preauthorization mechanism is declared."
    )

    @property
    def activation_supported(self) -> bool:
        """Whether Crossby can select native mode before the first turn."""
        return self.activation is not PlanModeActivation.UNSUPPORTED

    @property
    def supported(self) -> bool:
        """Compatibility alias for activation-only native plan support."""
        return self.activation_supported

    @property
    def session_activation(self) -> PlanModeActivation:
        """Activation used by collection, which may differ from ``launch``."""
        return self.collector_activation or self.activation

    @property
    def session_supported(self) -> bool:
        """Whether Crossby can return an artifact bound to the launched run."""
        return (
            self.session_activation is not PlanModeActivation.UNSUPPORTED
            and self.transport is not PlanSessionTransport.UNAVAILABLE
            and self.artifact_source is not None
            and self.binding is not None
        )


_UNSUPPORTED_PLAN_MODE = PlanModeCapability(
    activation=PlanModeActivation.UNSUPPORTED,
    activation_detail="No native plan-mode activation strategy is declared.",
    version_requirement="No supported version is declared.",
    initial_prompt_after_activation=False,
    artifact_location=PlanArtifactLocation.UNAVAILABLE,
    artifact_location_detail="Native plan artifacts are unavailable.",
    remediation="Use an adapter that declares native plan-mode support.",
)


class HeadlessCapability(BaseModel):
    """Truthful declaration for the managed ordinary-session boundary.

    A declaration is unavailable unless it names a transport, prompt channel,
    and at least one native output.  Legacy ``supports_headless`` and
    ``headless_flag`` remain independent until each concrete adapter has a
    verified managed transport.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    transport: HeadlessNativeTransport = HeadlessNativeTransport.UNAVAILABLE
    prompt_transport: HeadlessPromptTransport | None = None
    native_outputs: tuple[HeadlessNativeOutput, ...] = Field(
        default=(),
        validation_alias=AliasChoices("native_outputs", "output_modes"),
    )
    interaction_modes: tuple[HeadlessInteractionMode, ...] = Field(
        default=(HeadlessInteractionMode.UNATTENDED,),
        validation_alias=AliasChoices("interaction_modes", "supported_interaction_modes"),
    )
    supports_response_schema: bool = False
    supports_resume: bool = False
    supports_native_abort: bool = False
    terminal_event_required: bool = False
    terminal_event_authoritative: bool = False
    successful_native_statuses: tuple[str, ...] = ()
    sandbox_behavior: PlanRequestBehavior = PlanRequestBehavior.PRESERVED
    approval_behavior: PlanRequestBehavior = PlanRequestBehavior.PRESERVED
    supported_approval_policies: tuple[PlanApprovalPolicy, ...] = (PlanApprovalPolicy.ON_REQUEST,)
    command_policy_support: PlanCommandPolicySupport = PlanCommandPolicySupport.UNSUPPORTED
    version_requirement: str = "No managed headless-session version is declared."
    verified_version: str | None = None
    remediation: str | None = None

    @field_validator("native_outputs", "interaction_modes")
    @classmethod
    def _declarations_are_unique(cls, value: tuple[Any, ...]) -> tuple[Any, ...]:
        if len(set(value)) != len(value):
            raise ValueError("headless capability declarations must not contain duplicates")
        return value

    @field_validator("successful_native_statuses")
    @classmethod
    def _native_statuses_are_safe(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or any(not status.strip() for status in value):
            raise ValueError("successful native statuses must be unique non-blank strings")
        return value

    @model_validator(mode="after")
    def _declaration_is_coherent(self) -> Self:
        if self.transport is HeadlessNativeTransport.UNAVAILABLE:
            if (
                self.prompt_transport is not None
                or self.native_outputs
                or self.supports_response_schema
                or self.supports_resume
                or self.supports_native_abort
                or self.terminal_event_required
                or self.terminal_event_authoritative
                or self.successful_native_statuses
            ):
                raise ValueError("an unavailable headless transport cannot declare I/O support")
            return self
        if self.prompt_transport is None or not self.native_outputs:
            raise ValueError("a managed headless transport requires prompt and output declarations")
        if not self.interaction_modes:
            raise ValueError("a managed headless transport requires an interaction mode")
        if self.terminal_event_authoritative and not self.terminal_event_required:
            raise ValueError("authoritative terminal events must also be required")
        return self

    @property
    def managed_supported(self) -> bool:
        """Whether this declaration is complete enough to start a session."""
        return (
            self.transport is not HeadlessNativeTransport.UNAVAILABLE
            and self.prompt_transport is not None
            and bool(self.native_outputs)
        )

    @property
    def supported(self) -> bool:
        """Compatibility spelling for ``managed_supported``."""
        return self.managed_supported

    @property
    def output_modes(self) -> tuple[HeadlessNativeOutput, ...]:
        """Compatibility spelling for ``native_outputs``."""
        return self.native_outputs


_UNSUPPORTED_HEADLESS = HeadlessCapability()


class HeadlessSessionPreflight(BaseModel):
    """Evidence from bounded, non-mutating managed-session preflight."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: AIToolID
    detected_version: str
    normalized_version: tuple[int, int, int]
    capability: HeadlessCapability
    working_dir: Path
    trusted_dirs: tuple[Path, ...] = ()
    checked: tuple[HeadlessPreflightCheck, ...]
    deferred: tuple[HeadlessPreflightDeferredCheck, ...]

    @field_validator("detected_version")
    @classmethod
    def _version_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("headless preflight version must be non-blank")
        return value


class HeadlessSessionResult(BaseModel):
    """Bounded public result for one managed ordinary headless session."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    tool: AIToolID
    version: str
    status: HeadlessTerminalStatus
    events: tuple[HeadlessEvent, ...] = ()
    final_text: str | None = None
    final_json: Any | None = None
    native_status: str | None = None
    exit_code: int | None = None
    session_id: str | None = None
    thread_id: str | None = None
    turn_id: str | None = None
    conversation_id: str | None = None
    usage: TokenUsage | None = None
    duration_seconds: float = Field(ge=0)
    denials: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    is_partial: bool = False

    @field_validator(
        "version",
        "final_text",
        "native_status",
        "session_id",
        "thread_id",
        "turn_id",
        "conversation_id",
    )
    @classmethod
    def _result_text_is_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("provided headless-session result text must be non-blank")
        return value

    @field_validator("denials", "warnings")
    @classmethod
    def _diagnostics_are_non_blank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("headless-session diagnostics must be non-blank")
        if any(len(item.encode("utf-8")) > _HEADLESS_DIAGNOSTIC_LIMIT for item in value):
            raise ValueError("headless-session diagnostics exceeded the fixed size limit")
        return value

    @field_validator("duration_seconds")
    @classmethod
    def _duration_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("headless-session duration must be finite")
        return value

    @model_validator(mode="after")
    def _result_is_consistent(self) -> Self:
        if len(self.events) > _HEADLESS_EVENT_LIMIT:
            raise ValueError("headless result exceeded the fixed event count limit")
        if (
            self.final_text is not None
            and len(self.final_text.encode("utf-8")) > _HEADLESS_FINAL_PAYLOAD_LIMIT
        ):
            raise ValueError("headless result text exceeded the fixed size limit")
        if self.final_json is not None:
            try:
                encoded = json.dumps(self.final_json, allow_nan=False).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ValueError("headless result JSON must be a finite JSON value") from exc
            if len(encoded) > _HEADLESS_FINAL_PAYLOAD_LIMIT:
                raise ValueError("headless result JSON exceeded the fixed size limit")
        if self.final_text is not None and self.final_json is not None:
            raise ValueError("a headless result cannot contain both text and JSON output")
        expected = 1
        terminal: HeadlessTerminalStatus | None = None
        for event in self.events:
            if event.sequence != expected:
                raise ValueError("headless result events must have contiguous sequence numbers")
            expected += 1
            for name in ("session_id", "thread_id", "turn_id", "conversation_id"):
                event_value = getattr(event, name)
                if event_value is not None and event_value != getattr(self, name):
                    raise ValueError("headless event provenance must match result provenance")
            if event.kind is HeadlessEventKind.TERMINAL:
                if terminal is not None:
                    raise ValueError("headless results cannot contain multiple terminal events")
                terminal = event.terminal_status
        if terminal is not None and self.events[-1].kind is not HeadlessEventKind.TERMINAL:
            raise ValueError("the terminal event must be the final normalized event")
        if terminal is not None and terminal is not self.status:
            raise ValueError("the terminal event and result status must agree")
        if self.is_partial:
            if terminal is not None or self.final_text is not None or self.final_json is not None:
                raise ValueError("safe partial results cannot contain terminal or final output")
            if self.status is not HeadlessTerminalStatus.FAILED:
                raise ValueError("safe partial results use failed status until reconciled")
        elif terminal is None:
            raise ValueError("a terminal headless result requires exactly one terminal event")
        if self.status is HeadlessTerminalStatus.SUCCEEDED:
            if self.exit_code not in (None, 0):
                raise ValueError("a successful headless result cannot have a failing exit code")
            if self.final_text is None and self.final_json is None:
                raise ValueError("a successful headless result requires final output")
        return self

    @property
    def output_text(self) -> str | None:
        return self.final_text

    @property
    def structured_output(self) -> Any | None:
        return self.final_json

    @property
    def validated_json(self) -> Any | None:
        return self.final_json

    @property
    def terminal_status(self) -> HeadlessTerminalStatus:
        return self.status


# Alternate public spellings kept deliberately small and explicit.  They make
# the enum roles discoverable without coupling consumers to one naming style.
HeadlessSessionTransport = HeadlessNativeTransport
HeadlessTransport = HeadlessNativeTransport
HeadlessOutputMode = HeadlessNativeOutput
HeadlessNativeOutputMode = HeadlessNativeOutput
HeadlessOutput = HeadlessNativeOutput
HeadlessOutputFormat = HeadlessNativeOutput
HeadlessSessionStatus = HeadlessTerminalStatus
HeadlessTerminalState = HeadlessTerminalStatus


class AIToolCapabilities(BaseModel):
    """What an AI tool can do — declared by each adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_id: AIToolID
    display_name: str
    binary: str
    tool_type: AIToolType
    update_command: tuple[str, ...] | None = None
    """Static command that updates this tool in place (e.g. ``("claude",
    "update")``). ``None`` means the tool has no crossby-driven update path (GUI
    tools that self-update, or a tool whose update subcommand crossby can't
    assume) and is excluded from ``crossby tools update``. This is a STATIC
    declaration only — crossby does not introspect the install method (npm vs
    brew vs standalone). A tuple (not a list) keeps this frozen model hashable,
    matching the ``supported_efforts`` convention."""
    supports_model_flag: bool = True
    model_flag: str = "--model"
    headless_flag: str | None = None
    supports_headless: bool = False
    supports_initial_message: bool = True
    blocks_until_exit: bool = True
    supports_effort: bool = False
    supported_efforts: tuple[EffortLevel, ...] = tuple(EffortLevel)
    """Effort levels this tool can actually honor — the set a picker should offer.
    Defaults to every EffortLevel; tools whose CLI accepts a subset narrow it
    (antigravity-cli → low/medium/high). Independent of supports_effort, which
    only says the tool has an effort concept crossby drives (Cursor/antigravity-cli
    bake effort into the model ID rather than emitting a flag)."""
    supports_yolo: bool = False
    supports_resume: bool = False
    supports_trusted_dirs: bool = False
    headless: HeadlessCapability = _UNSUPPORTED_HEADLESS
    """Managed ordinary-session contract.  The default is unavailable even
    when the legacy command builder advertises ``supports_headless``."""
    plan_mode: PlanModeCapability = _UNSUPPORTED_PLAN_MODE
    """Typed native plan-mode contract. Adapters must explicitly declare this
    for supported activation; the default fails closed."""
    supports_accept_edits: bool = False
    """Tool can auto-approve file edits at launch while still prompting for
    shell/commands (the accept-edits autonomy tier)."""
    supports_auto: bool = False
    """Tool exposes a classifier-mediated ``auto`` mode at launch (a separate
    model reviews tool calls). Supported by Claude and Cursor; ordinary
    launches downgrade elsewhere, while Plan launches require explicit support."""

    @property
    def supports_plan_mode(self) -> bool:
        """Compatibility activation-only view over the typed capability."""
        return self.plan_mode.supported

    @property
    def supports_plan_session(self) -> bool:
        """Whether the adapter implements exact-session plan collection."""
        return self.plan_mode.session_supported

    @property
    def supports_managed_headless_session(self) -> bool:
        """Whether a protected adapter hook may own an ordinary session."""
        return self.headless.managed_supported

    @property
    def supports_headless_session(self) -> bool:
        """Compatibility view over managed headless support."""
        return self.supports_managed_headless_session

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
    supports_sandbox_toggle: bool = False
    """Tool can map the programmatic launch-time ``sandbox`` input to an
    explicit enabled/disabled sandbox selection. This is independent of
    ``sandboxes_writes`` (the adapter's normal confinement) and
    ``supports_network_access`` (networking inside a sandbox)."""
    supports_network_access: bool = False
    """Tool exposes a launch-time opt-in to allow network access from inside its
    sandbox (``crossby launch --network``). Codex-only: it pins
    ``sandbox_workspace_write.network_access`` whenever crossby forces
    workspace-write. Narrower and clearer than reusing ``sandboxes_writes``:
    every path (launch, resume, GUI) warns and ignores ``--network`` when this is
    False, so a non-Codex tool never receives a network flag it cannot honor."""
    hook_output_dialect: HookOutputDialect = HookOutputDialect.HOOK_SPECIFIC_OUTPUT
    """Which stdout shape this tool reads a *tool-call* hook decision from."""
    hook_stop_dialect: HookStopDialect = HookStopDialect.NONE
    """Which stdout shape this tool reads a *Stop* hook decision from. Declared
    separately from ``hook_output_dialect`` because the two channels are
    independent per tool (see :class:`HookStopDialect`).

    Defaults to ``NONE`` to stay consistent with ``supports_stop_hook``, which
    defaults to False: an adapter that never opts into a Stop hook should not
    imply it speaks one. Every adapter crossby ships declares this explicitly."""
    hook_fail_open_default: bool = False
    """Tool treats a hook that errors/crashes as *allow* (fail-open) unless the
    hook config opts into fail-closed. True for Cursor — callers writing a
    security guard must set the tool's fail-closed flag when this is True."""
    supports_usage_reporting: bool = False
    """Tool emits structured token usage in headless output (``--output-format
    json`` / ``codex exec --json``), so usage need not be scraped from a
    transcript log. False for Cursor (no usage fields in CLI output)."""

    # --- Session-scoped scenes (crossby launch --scene) ---------------------
    # Declared per adapter following the ``supports_*`` convention. These say
    # *how* a tool can take a whole scene on the command line for one session
    # without mutating tracked project files; the per-adapter
    # ``scene_launch_args`` renders the artefacts and emits the flags. A tool
    # that leaves ``supports_scene_launch`` False has no session-scoped lever, so
    # ``crossby launch --scene`` falls back to persistent activation for it.
    supports_scene_launch: bool = False
    """Tool has at least one session-scoped scene lever (a settings/mcp-config
    file flag, a named profile, or a config-dir env var). When False,
    ``crossby launch --scene`` falls back to persistent ``scene use`` activation
    for this tool rather than emitting launch flags."""
    scene_settings_flag: str | None = None
    """CLI flag that loads a session-scoped settings file (Claude ``--settings``);
    ``None`` when the tool has no such flag."""
    scene_mcp_config_flag: str | None = None
    """CLI flag that loads a session-scoped MCP config file (Claude
    ``--mcp-config``); ``None`` when unsupported."""
    scene_mcp_strict_flag: str | None = None
    """CLI flag that makes the session MCP config authoritative — the tool loads
    *only* it and ignores other MCP sources (Claude ``--strict-mcp-config``);
    ``None`` when the tool has no strict mode."""
    scene_config_dir_env: str | None = None
    """Environment variable pointing the tool at a scene-materialised config
    dir/file; ``None`` when the tool exposes no usable config-dir override.
    Currently ``None`` for every adapter: Cursor's ``CURSOR_CONFIG_DIR``
    relocates the whole config base (auth included) and OpenCode's
    ``OPENCODE_CONFIG`` loads between the global and project layers (a project
    config can override it), so neither is a sound session-scoped lever."""
    scene_profile_flag: str | None = None
    """CLI flag selecting a named profile layered over the base config (Codex
    ``--profile``); ``None`` when unsupported."""
    scene_tool_denylist_flag: str | None = None
    """CLI flag excluding named tools for the session (Copilot
    ``--excluded-tools``); ``None`` when unsupported."""


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
