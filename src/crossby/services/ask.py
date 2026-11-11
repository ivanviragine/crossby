"""One read-only question to one agent: the engine behind ``crossby.api.ask``.

An ask is a fresh, unattended, read-only headless session: nothing is resumed
or replayed, and the agent answers from the folder it runs in without changing
it. crossby adds nothing to the question except, when the caller supplies
instructions, the documented envelope that carries them (see
:func:`render_envelope`).

Everything a caller needs to judge the answer comes back with it: the status,
the output, usage as the tool reported it, which CLI build ran, and every
channel's restriction state for the profile that build actually ran with.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    field_validator,
    model_validator,
)

from crossby.models.ai import (
    AIToolID,
    EffortLevel,
    HeadlessFailureReason,
    HeadlessInteractionMode,
    HeadlessNativeTransport,
    HeadlessSessionPreflight,
    HeadlessSessionRequest,
    HeadlessSessionResult,
    HeadlessTerminalStatus,
    Restriction,
    RestrictionChannel,
    RestrictionState,
    TokenUsage,
)
from crossby.services.workspaces import OwnedWorkspace

if TYPE_CHECKING:
    from crossby.ai_tools.base import AbstractAITool

SCHEMA_VERSION = 1
"""Version of the serialized :class:`AskResult` and :class:`PreflightReport`."""

ENVELOPE_VERSION = 1
"""Version of the instructions envelope (see :func:`render_envelope`)."""

DEFAULT_TIMEOUT_SECONDS = 600.0
PREFLIGHT_TIMEOUT_SECONDS = 20.0


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class CliBackend(_Frozen):
    """Which agent CLI answers, and with which model and effort."""

    tool: AIToolID
    model: str | None = None
    effort: EffortLevel | None = None

    @field_validator("model")
    @classmethod
    def _model_is_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("model must be non-blank when given")
        return value


class ProjectWorkspace(_Frozen):
    """An existing folder the agent answers from; crossby never writes to it.

    See :class:`~crossby.services.workspaces.OwnedWorkspace` for a persona
    folder crossby provisioned with one tool's own instruction and skill files.

    The agent sees the folder as it is, with its own instruction files,
    skills and configuration. ``instructions`` is literal text (never a path)
    delivered in the first message's envelope, so it reaches the agent the
    same way on every tool.
    """

    kind: Literal["project"] = "project"
    path: Path
    instructions: str | None = None

    @field_validator("instructions")
    @classmethod
    def _instructions_are_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("instructions must be non-blank when given")
        return value


def _workspace_kind(value: Any) -> str:
    """The workspace variant; a mapping without ``kind`` is a project folder."""
    if isinstance(value, dict):
        return str(value.get("kind", "project"))
    return str(getattr(value, "kind", "project"))


# Untagged mappings ({"path": ...}) stay project folders, as before persona
# folders existed.
Workspace = Annotated[
    Annotated[ProjectWorkspace, Tag("project")] | Annotated[OwnedWorkspace, Tag("owned")],
    Discriminator(_workspace_kind),
]


class AskRequest(_Frozen):
    """One question for one agent.

    ``timeout_seconds`` bounds the whole call, from validation and the CLI's
    version probe to collecting the answer. ``require_blocked`` names
    channels that must be verified blocked on the installed build; an unmet
    requirement is refused before the agent starts. ``response_schema`` asks
    for structured output that the runtime validates against the schema.
    """

    backend: CliBackend
    prompt: str
    workspace: Workspace
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    require_blocked: frozenset[RestrictionChannel] = frozenset()
    response_schema: dict[str, Any] | None = None
    # Run even though the folder has hooks the tool would run (a hook can
    # change files); otherwise such an ask is refused with ProjectHooksError.
    allow_project_hooks: bool = False

    @field_validator("prompt")
    @classmethod
    def _prompt_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("the question is empty")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def _timeout_is_positive(cls, value: float) -> float:
        if not value > 0 or value == float("inf"):
            raise ValueError("timeout_seconds must be positive and finite")
        return value

    @model_validator(mode="after")
    def _owned_folder_matches_its_tool(self) -> AskRequest:
        workspace = self.workspace
        if isinstance(workspace, OwnedWorkspace) and workspace.tool is not self.backend.tool:
            raise ValueError(
                f"the persona folder was provisioned for {workspace.tool.value}, not "
                f"{self.backend.tool.value}"
            )
        return self


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class AskStatus(StrEnum):
    """How an ask ended once its session started."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    INVALID_OUTPUT = "invalid_output"


class TextOutput(_Frozen):
    kind: Literal["text"] = "text"
    text: str


class JsonOutput(_Frozen):
    """Structured output; ``value`` may be JSON ``null`` and still be present."""

    kind: Literal["json"] = "json"
    value: Any


Output = Annotated[TextOutput | JsonOutput, Field(discriminator="kind")]


class UsageReport(_Frozen):
    """Tokens as the tool reported them for this one invocation.

    A field is ``None`` when the tool did not report it, which is never the
    same as zero. ``complete`` is ``None`` because no tool states whether its
    totals cover everything it spent.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    total_tokens: int | None = None
    premium_requests: int | None = None
    # Cost in US dollars exactly as the tool reported it (Claude Code does);
    # crossby keeps no prices and never estimates one.
    reported_cost_usd: float | None = None
    scope: Literal["invocation"] = "invocation"
    complete: bool | None = None
    source: str


class CliProvenance(_Frozen):
    """Which build ran, how, and the native IDs it reported."""

    tool: AIToolID
    installed_version: str | None = None
    # The oldest builds crossby verified the unattended session and its
    # read-only profile on.
    verified_version: str | None = None
    read_only_verified_version: str | None = None
    transport: HeadlessNativeTransport | None = None
    requested_model: str | None = None
    # The model the tool reported using, when it reported exactly one.
    observed_model: str | None = None
    session_id: str | None = None
    thread_id: str | None = None
    turn_id: str | None = None
    conversation_id: str | None = None


class AskResult(_Frozen):
    """The outcome of one ask, including what a failed run left behind."""

    schema_version: int = SCHEMA_VERSION
    status: AskStatus
    output: Output | None = None
    error: str | None = None
    duration_seconds: float = Field(ge=0)
    usage: UsageReport | None = None
    restrictions: tuple[Restriction, ...] = ()
    provenance: CliProvenance
    denials: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    # Why a failed, invalid or timed-out ask failed, when the tool said so in
    # a recognized way; ``None`` means unrecognized, not "no reason".
    failure_reason: HeadlessFailureReason | None = None
    # "prompt": the question was sent exactly as given; "prompt_envelope":
    # it was sent inside the instructions envelope of ``envelope_version``;
    # "native_files": as given, to a persona folder whose instructions and
    # skills are the tool's own files.
    delivery: Literal["prompt", "prompt_envelope", "native_files"]
    envelope_version: int | None = None
    # For a persona folder, whether it still held exactly its manifest after
    # the ask; ``None`` for a project folder, which crossby does not inventory.
    workspace_unchanged: bool | None = None

    @model_validator(mode="after")
    def _delivery_matches_envelope(self) -> AskResult:
        if (self.delivery == "prompt_envelope") != (self.envelope_version is not None):
            raise ValueError("only an envelope delivery carries an envelope version")
        return self

    @property
    def succeeded(self) -> bool:
        return self.status is AskStatus.SUCCEEDED and self.error is None

    @property
    def text(self) -> str | None:
        """The answer's text, for a text answer."""
        return self.output.text if isinstance(self.output, TextOutput) else None

    def restriction(self, channel: RestrictionChannel) -> Restriction:
        return _find(self.restrictions, channel)


class PreflightReport(_Frozen):
    """What an ask would run with, established without starting the agent."""

    schema_version: int = SCHEMA_VERSION
    provenance: CliProvenance
    restrictions: tuple[Restriction, ...]
    # Required channels that are not enforced on this build; an ask with the
    # same request would be refused.
    unmet: tuple[Restriction, ...] = ()
    checked: tuple[str, ...] = ()
    # Checks only a real run can make, such as authentication.
    deferred: tuple[str, ...] = ()

    def restriction(self, channel: RestrictionChannel) -> Restriction:
        return _find(self.restrictions, channel)


def _find(restrictions: tuple[Restriction, ...], channel: RestrictionChannel) -> Restriction:
    for item in restrictions:
        if item.channel is channel:
            return item
    return Restriction(channel=channel, state=RestrictionState.UNKNOWN)


# ---------------------------------------------------------------------------
# Errors raised before any agent starts
# ---------------------------------------------------------------------------


class AskError(Exception):
    """An ask was refused before any agent started."""


class AskRequestError(AskError):
    """The request is invalid, e.g. its folder is missing or its schema is malformed."""


class AskUnsupportedError(AskError):
    """The tool or its installed build cannot run this ask (or is not installed)."""


class RequirementNotMetError(AskUnsupportedError):
    """A channel in ``require_blocked`` is not enforced on the installed build."""

    def __init__(
        self, message: str, *, unmet: tuple[Restriction, ...], restrictions: tuple[Restriction, ...]
    ) -> None:
        super().__init__(message)
        self.unmet = unmet
        self.restrictions = restrictions


class ProjectHooksError(AskUnsupportedError):
    """The folder has hooks the tool would run during the read-only session.

    ``sources`` names them, relative to the folder. Remove them, ask in a
    persona folder, use a tool whose read-only profile disables hooks, or set
    ``allow_project_hooks``.
    """

    def __init__(self, message: str, *, sources: tuple[str, ...]) -> None:
        super().__init__(message)
        self.sources = sources


class NestedAskError(AskError):
    """Called from inside an agent that crossby itself started."""


class WorkspaceChangedError(AskRequestError):
    """The persona folder no longer holds exactly what was provisioned into it."""

    def __init__(self, message: str, *, changed: tuple[str, ...]) -> None:
        super().__init__(message)
        self.changed = changed


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------

_ENVELOPE = "<instructions>\n{instructions}\n</instructions>\n\n{prompt}"


def render_envelope(prompt: str, instructions: str | None) -> str:
    """The first message for ``prompt``: as is, or inside envelope v1.

    Envelope v1 is exactly::

        <instructions>
        {instructions}
        </instructions>

        {prompt}

    Both parts are inserted verbatim; nothing else is added.
    """
    if instructions is None:
        return prompt
    return _ENVELOPE.format(instructions=instructions, prompt=prompt)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SessionRunner(Protocol):
    """The headless boundary an ask runs through; injectable for tests."""

    def preflight(
        self, tool: AIToolID, request: HeadlessSessionRequest, timeout_seconds: float
    ) -> HeadlessSessionPreflight: ...

    def run(
        self, tool: AIToolID, request: HeadlessSessionRequest, cancel: threading.Event | None
    ) -> HeadlessSessionResult: ...

    def restrictions(
        self, tool: AIToolID, request: HeadlessSessionRequest, version: str | None
    ) -> tuple[Restriction, ...]: ...


class AdapterRunner:
    """Runs asks through each tool's managed headless session."""

    def preflight(
        self, tool: AIToolID, request: HeadlessSessionRequest, timeout_seconds: float
    ) -> HeadlessSessionPreflight:
        return _adapter(tool).preflight_headless_session(request, timeout_seconds=timeout_seconds)

    def run(
        self, tool: AIToolID, request: HeadlessSessionRequest, cancel: threading.Event | None
    ) -> HeadlessSessionResult:
        return run_cancellable(tool, request, cancel)

    def restrictions(
        self, tool: AIToolID, request: HeadlessSessionRequest, version: str | None
    ) -> tuple[Restriction, ...]:
        return _adapter(tool).headless_restrictions(request, version)


def run_cancellable(
    tool: AIToolID, request: HeadlessSessionRequest, cancel: threading.Event | None = None
) -> HeadlessSessionResult:
    """Run one managed session in a worker so Ctrl-C tears it down cleanly.

    ``KeyboardInterrupt`` only reaches the main thread. Waiting here and
    setting the session's cancel event on interrupt lets the managed runtime
    stop the native process group before the interrupt propagates.
    """
    adapter = _adapter(tool)
    stop = cancel if cancel is not None else threading.Event()
    box: queue.Queue[HeadlessSessionResult | BaseException] = queue.Queue(1)

    def work() -> None:
        try:
            box.put(adapter.run_headless_session(request, cancel_event=stop))
        except BaseException as exc:
            box.put(exc)

    worker = threading.Thread(target=work, daemon=True, name=f"crossby-session-{tool}")
    worker.start()
    try:
        while worker.is_alive():
            worker.join(0.1)
    except KeyboardInterrupt:
        stop.set()
        worker.join(30)
        raise
    value = box.get_nowait()
    if isinstance(value, BaseException):
        raise value
    return value


def _adapter(tool: AIToolID) -> AbstractAITool:
    import crossby.ai_tools  # noqa: F401  (registers every adapter)
    from crossby.ai_tools.base import AbstractAITool

    return AbstractAITool.get(tool)


def session_request(
    request: AskRequest, *, timeout_seconds: float, environment: Mapping[str, str]
) -> HeadlessSessionRequest:
    """The headless session an ask runs: fresh, unattended and read-only."""
    workspace = request.workspace
    if isinstance(workspace, OwnedWorkspace):
        # The instructions are the tool's own files; the question goes as is.
        prompt = request.prompt
        # A persona folder's layout was verified on its own build.
        minimum_version: str | None = workspace.layout.verified_version
    else:
        prompt = render_envelope(request.prompt, workspace.instructions)
        minimum_version = None
    try:
        return HeadlessSessionRequest(
            prompt=prompt,
            working_dir=workspace.path,
            model=request.backend.model,
            effort=request.backend.effort,
            interaction_mode=HeadlessInteractionMode.UNATTENDED,
            response_schema=request.response_schema,
            read_only=True,
            require_blocked=request.require_blocked,
            allow_project_hooks=request.allow_project_hooks,
            minimum_version=minimum_version,
            environment=dict(environment),
            timeout_seconds=timeout_seconds,
        )
    except ValueError as exc:
        raise AskRequestError(str(exc)) from None


def _verified_versions(tool: AIToolID) -> tuple[str | None, str | None, HeadlessNativeTransport]:
    adapter = _adapter(tool)
    capability = adapter.capabilities().headless
    if not capability.managed_supported:
        return None, None, capability.transport
    profile = capability.profile_for(HeadlessInteractionMode.UNATTENDED)
    read_only = adapter.HEADLESS_READ_ONLY_VERIFIED_VERSION or profile.verified_version
    return profile.verified_version, read_only, profile.transport


def _provenance(
    request: AskRequest, result: HeadlessSessionResult | None, *, version: str | None = None
) -> CliProvenance:
    tool = request.backend.tool
    verified, read_only, transport = _verified_versions(tool)
    observed: str | None = None
    if result is not None and result.usage is not None:
        models = {item.model for item in result.usage.model_breakdown if item.model}
        observed = next(iter(models)) if len(models) == 1 else None
    return CliProvenance(
        tool=tool,
        installed_version=result.version if result is not None else version,
        verified_version=verified,
        read_only_verified_version=read_only,
        transport=transport,
        requested_model=request.backend.model,
        observed_model=observed,
        session_id=result.session_id if result is not None else None,
        thread_id=result.thread_id if result is not None else None,
        turn_id=result.turn_id if result is not None else None,
        conversation_id=result.conversation_id if result is not None else None,
    )


def _usage(tool: AIToolID, usage: TokenUsage | None) -> UsageReport | None:
    if usage is None:
        return None
    fields = (
        usage.input_tokens,
        usage.output_tokens,
        usage.cached_tokens,
        usage.total_tokens,
        usage.premium_requests,
        usage.reported_cost_usd,
    )
    if all(value is None for value in fields):
        return None
    return UsageReport(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_tokens=usage.cached_tokens,
        total_tokens=usage.total_tokens,
        premium_requests=usage.premium_requests,
        reported_cost_usd=usage.reported_cost_usd,
        source=f"{tool.value} result",
    )


def _output(result: HeadlessSessionResult) -> TextOutput | JsonOutput | None:
    if result.final_json_present:
        return JsonOutput(value=result.final_json)
    if result.final_text is not None:
        return TextOutput(text=result.final_text)
    return None


FAILURE_PHRASES: dict[HeadlessFailureReason, str] = {
    HeadlessFailureReason.NOT_SIGNED_IN: "not signed in",
    HeadlessFailureReason.MODEL_UNAVAILABLE: "the model is not available",
    HeadlessFailureReason.RATE_LIMITED: "a usage or rate limit was reached",
    HeadlessFailureReason.ACTION_DENIED: "it ended the turn after denying an action",
    HeadlessFailureReason.UNTRUSTED_FOLDER: "it refused to run in this folder",
}


_SIGN_IN: dict[AIToolID, str] = {
    AIToolID.CLAUDE: "Sign in with `claude` and /login, or set a valid ANTHROPIC_API_KEY.",
    AIToolID.CODEX: "Sign in with `codex login`.",
    AIToolID.CURSOR: "Sign in with `agent login`, or set a valid CURSOR_API_KEY.",
    AIToolID.COPILOT: "Sign in with `copilot` and /login, or set a valid COPILOT_GITHUB_TOKEN.",
    AIToolID.OPENCODE: "Sign in with `opencode auth login`.",
    AIToolID.ANTIGRAVITY_CLI: "Sign in by running `agy` once.",
}


def failure_hint(tool: AIToolID, reason: HeadlessFailureReason | None) -> str | None:
    """What a person can do about a recognized failure, for the CLI to show."""
    if reason is HeadlessFailureReason.NOT_SIGNED_IN:
        return _SIGN_IN.get(tool)
    if reason is HeadlessFailureReason.MODEL_UNAVAILABLE:
        return "Pick a model this account can use with --model, or omit it for the default."
    if reason is HeadlessFailureReason.RATE_LIMITED:
        return "Wait for the limit to reset, or ask another tool."
    if reason is HeadlessFailureReason.ACTION_DENIED:
        return (
            "The agent tried something its read-only profile denies (see the denials) and "
            "gave no answer; ask for one it can give by reading."
        )
    if reason is HeadlessFailureReason.UNTRUSTED_FOLDER:
        return "Run it in a git repository or a folder the tool trusts."
    return None


def describe_session_status(result: HeadlessSessionResult) -> str:
    """``failed``, plus the recognized reason and the tool's own status.

    For example ``failed: not signed in (native: api_error)``. The reason is a
    fixed phrase and the native status a short validated token, never native
    output, so both are safe to show.
    """
    reason = (
        f": {FAILURE_PHRASES[result.failure_reason]}" if result.failure_reason is not None else ""
    )
    native = f" (native: {result.native_status})" if result.native_status else ""
    return f"{result.status.value}{reason}{native}"


def _status_text(result: HeadlessSessionResult) -> str:
    return describe_session_status(result)


def _delivery(request: AskRequest) -> dict[str, Any]:
    workspace = request.workspace
    if isinstance(workspace, OwnedWorkspace):
        return {"delivery": "native_files", "envelope_version": None}
    if workspace.instructions is None:
        return {"delivery": "prompt", "envelope_version": None}
    return {"delivery": "prompt_envelope", "envelope_version": ENVELOPE_VERSION}


def _result_from_session(
    request: AskRequest,
    session: HeadlessSessionRequest,
    result: HeadlessSessionResult,
    *,
    duration: float,
    runner: SessionRunner,
    error: str | None = None,
) -> AskResult:
    tool = request.backend.tool
    status = AskStatus(result.status.value)
    output = _output(result)
    if error is None and result.status is not HeadlessTerminalStatus.SUCCEEDED:
        denied = f" ({'; '.join(result.denials)})" if result.denials else ""
        error = f"{tool.value} session ended with status {_status_text(result)}{denied}"
    unchanged: bool | None = None
    if isinstance(request.workspace, OwnedWorkspace):
        from crossby.services.workspaces import workspace_unchanged

        unchanged = workspace_unchanged(request.workspace)
    return AskResult(
        status=status,
        output=output,
        error=error,
        duration_seconds=duration,
        workspace_unchanged=unchanged,
        usage=_usage(tool, result.usage),
        restrictions=runner.restrictions(tool, session, result.version),
        provenance=_provenance(request, result),
        denials=result.denials,
        warnings=result.warnings,
        failure_reason=result.failure_reason,
        **_delivery(request),
    )


def _translate_preflight(exc: BaseException) -> AskError:
    from crossby.ai_tools.headless import (
        HeadlessProjectHooksError,
        HeadlessRequestError,
        HeadlessRequirementError,
        HeadlessUnsupportedError,
    )

    if isinstance(exc, HeadlessProjectHooksError):
        return ProjectHooksError(str(exc), sources=exc.sources)
    if isinstance(exc, HeadlessRequirementError):
        return RequirementNotMetError(str(exc), unmet=exc.unmet, restrictions=exc.restrictions)
    if isinstance(exc, HeadlessRequestError):
        return AskRequestError(str(exc))
    if isinstance(exc, HeadlessUnsupportedError):
        return AskUnsupportedError(str(exc))
    return AskUnsupportedError(str(exc))


def run_ask(
    request: AskRequest,
    *,
    cancel: threading.Event | None = None,
    runner: SessionRunner | None = None,
    clock: Callable[[], float] = monotonic,
) -> AskResult:
    """Run one ask; see :func:`crossby.api.ask`."""
    from crossby.ai_tools.headless import (
        HeadlessAdapterContractError,
        HeadlessPreflightError,
        HeadlessTransportError,
    )
    from crossby.services.converse import ConverseDepthError, check_depth, child_environment

    try:
        check_depth()
    except ConverseDepthError as exc:
        raise NestedAskError(str(exc)) from None
    active: SessionRunner = runner or AdapterRunner()
    started = clock()
    workspace = request.workspace
    if isinstance(workspace, OwnedWorkspace):
        from crossby.services.workspaces import WorkspaceDriftError, verify

        try:
            verify(workspace)
        except WorkspaceDriftError as exc:
            raise WorkspaceChangedError(str(exc), changed=exc.changed) from None
    session = session_request(
        request, timeout_seconds=request.timeout_seconds, environment=child_environment()
    )
    if cancel is not None and cancel.is_set():
        return AskResult(
            status=AskStatus.CANCELLED,
            error="cancelled before the session started",
            duration_seconds=0.0,
            provenance=_provenance(request, None),
            **_delivery(request),
        )
    try:
        result = active.run(request.backend.tool, session, cancel)
    except HeadlessAdapterContractError as exc:
        return AskResult(
            status=AskStatus.FAILED,
            error=str(exc),
            duration_seconds=clock() - started,
            provenance=_provenance(request, None),
            **_delivery(request),
        )
    except HeadlessPreflightError as exc:
        if cancel is not None and cancel.is_set():
            return AskResult(
                status=AskStatus.CANCELLED,
                error="cancelled before the session started",
                duration_seconds=clock() - started,
                provenance=_provenance(request, None),
                **_delivery(request),
            )
        raise _translate_preflight(exc) from None
    except HeadlessTransportError as exc:
        partial = exc.partial_result
        return _result_from_session(
            request,
            session,
            partial,
            duration=clock() - started,
            runner=active,
            error=f"{request.backend.tool.value}: {exc}",
        )
    return _result_from_session(request, session, result, duration=clock() - started, runner=active)


def run_preflight(
    request: AskRequest,
    *,
    runner: SessionRunner | None = None,
) -> PreflightReport:
    """Check an ask without starting the agent; see :func:`crossby.api.preflight`."""
    from crossby.ai_tools.headless import HeadlessPreflightError, HeadlessRequirementError
    from crossby.services.converse import child_environment

    active: SessionRunner = runner or AdapterRunner()
    session = session_request(
        request,
        timeout_seconds=request.timeout_seconds,
        environment=child_environment(),
    )
    # Resolve every channel even when some required one is unmet: the report
    # says what would block the ask instead of raising.
    probe = session.model_copy(update={"require_blocked": frozenset()})
    timeout = min(PREFLIGHT_TIMEOUT_SECONDS, request.timeout_seconds)
    try:
        report = active.preflight(request.backend.tool, probe, timeout)
    except HeadlessRequirementError as exc:  # pragma: no cover - requirements were cleared
        raise _translate_preflight(exc) from None
    except HeadlessPreflightError as exc:
        raise _translate_preflight(exc) from None
    unmet = tuple(
        item
        for item in report.restrictions
        if item.channel in request.require_blocked and not item.enforced
    )
    return PreflightReport(
        provenance=_provenance(request, None, version=report.detected_version),
        restrictions=report.restrictions,
        unmet=unmet,
        checked=tuple(check.value for check in report.checked),
        deferred=tuple(check.value for check in report.deferred),
    )
