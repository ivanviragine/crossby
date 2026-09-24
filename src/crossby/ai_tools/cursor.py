"""Cursor CLI adapter."""

from __future__ import annotations

import shutil
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import structlog

from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.plan_mode import (
    PlanInteractionHandler,
    parse_plan_question_options,
    validate_plan_option_selection,
)
from crossby.data import get_models_for_tool
from crossby.handoff.models import ConversationTranscript, SessionRef
from crossby.handoff.readers import cursor as cursor_reader
from crossby.models.ai import (
    AIToolCapabilities,
    AIToolID,
    AIToolType,
    EffortLevel,
    HeadlessCapability,
    HeadlessInteractionMode,
    HeadlessNativeOutput,
    HeadlessNativeTransport,
    HeadlessPromptTransport,
    HeadlessSessionRequest,
    HeadlessSessionResult,
    HeadlessTerminalStatus,
    HookOutputDialect,
    HookStopDialect,
    PlanApprovalPolicy,
    PlanArtifactLocation,
    PlanArtifactSource,
    PlanCommandPolicySupport,
    PlanInteraction,
    PlanInteractionKind,
    PlanInteractionOutcome,
    PlanInteractionSupport,
    PlanLaunchApprovalMode,
    PlanModeActivation,
    PlanModeCapability,
    PlanNativeBindingID,
    PlanOperation,
    PlanOperationKind,
    PlanPermissionTarget,
    PlanPermissionTargetKind,
    PlanQuestionOption,
    PlanRequestBehavior,
    PlanSessionBinding,
    PlanSessionRequest,
    PlanSessionResult,
    PlanSessionTransport,
)

if TYPE_CHECKING:
    from crossby.ai_tools.headless import HeadlessRuntimeContext

logger = structlog.get_logger()

# Cursor encodes reasoning effort in the model ID rather than a flag:
# ``<family>[-thinking]-<effort>[-fast]`` (``claude-opus-5-5-high-fast``,
# ``claude-opus-5-thinking-xhigh``) or, for older families,
# ``<family>-<effort>-thinking`` (``claude-4.6-sonnet-medium-thinking``). GPT-5.5
# spells xhigh as ``extra-high``, and some bare IDs (``gpt-5.3-codex``) are the
# family's default effort. ``none``/``minimal`` exist in the catalog but are not
# Crossby effort levels, so they are recognized only as a position, never chosen.
_EFFORT_WORDS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_EFFORT_SPELLINGS: dict[EffortLevel, tuple[str, ...]] = {
    EffortLevel.LOW: ("low",),
    EffortLevel.MEDIUM: ("medium",),
    EffortLevel.HIGH: ("high",),
    EffortLevel.XHIGH: ("xhigh", "extra-high"),
    EffortLevel.MAX: ("max",),
}
_EFFORT_ORDER: tuple[EffortLevel, ...] = tuple(_EFFORT_SPELLINGS)
# Tokens that may follow the effort word without being part of the family name.
_TRAILING_VARIANT_TOKENS = frozenset({"fast", "thinking"})
_EFFORT_SLOT = "\0"

# Models that have no "-thinking" variant — appending the suffix produces an invalid ID.
_NO_THINKING_MODELS: frozenset[str] = frozenset({"auto"})

_THINKING_EFFORTS = frozenset({EffortLevel.HIGH, EffortLevel.XHIGH, EffortLevel.MAX})


def _effort_template(model: str) -> tuple[str, EffortLevel | None, bool]:
    """Split a Cursor model ID into an effort template and its effort.

    Returns ``(template, effort, explicit)``: ``template`` has the effort token
    replaced by ``_EFFORT_SLOT``; ``explicit`` is False for a bare ID (no effort
    token), whose template gains a slot before any trailing ``-fast``. Only an
    effort word followed solely by ``-fast``/``-thinking`` counts, so an effort
    word inside a family name is never mistaken for the effort.
    """
    tokens = model.split("-")
    end = len(tokens)
    while end > 0 and tokens[end - 1] in _TRAILING_VARIANT_TOKENS:
        end -= 1
    if end > 0 and tokens[end - 1] in _EFFORT_WORDS:
        start = end - 1
        if tokens[start] == "high" and start > 0 and tokens[start - 1] == "extra":
            start -= 1
        word = "-".join(tokens[start:end])
        effort = next((lvl for lvl, names in _EFFORT_SPELLINGS.items() if word in names), None)
        template = "-".join([*tokens[:start], _EFFORT_SLOT, *tokens[end:]])
        return template, effort, True
    base, fast = model.removesuffix("-fast"), model.endswith("-fast")
    return f"{base}-{_EFFORT_SLOT}" + ("-fast" if fast else ""), None, False


def _nearest_effort(requested: EffortLevel, offered: set[EffortLevel]) -> EffortLevel:
    """Pick ``requested`` if offered, else the closest tier (ties go higher)."""
    rank = _EFFORT_ORDER.index(requested)
    return min(
        offered,
        key=lambda lvl: (abs(_EFFORT_ORDER.index(lvl) - rank), -_EFFORT_ORDER.index(lvl)),
    )


def _encoded_effort(model: str) -> EffortLevel | None:
    """Return an explicit effort tier encoded near the end of a Cursor model ID."""
    parts = model.split("[", 1)[0].removesuffix("-fast").split("-")
    if len(parts) > 1 and parts[-2:] == ["extra", "high"]:
        return EffortLevel.XHIGH
    candidate = parts[-2] if parts[-1] == "thinking" and len(parts) > 1 else parts[-1]
    try:
        return EffortLevel(candidate)
    except ValueError:
        return None


def _parameterized_effort(model: str) -> EffortLevel | None:
    """Return an effort value from Cursor's documented bracket overrides."""
    if "[" not in model and "]" not in model:
        return None
    if not model.endswith("]") or model.count("[") != 1 or model.count("]") != 1:
        raise ValueError("malformed bracket overrides")
    parameters = model.rsplit("[", 1)[1][:-1]
    efforts: list[EffortLevel] = []
    for parameter in parameters.split(","):
        key, separator, value = parameter.partition("=")
        key = key.strip()
        if not separator or key != "effort":
            raise ValueError(f"unsupported bracket override {key or parameter.strip()!r}")
        efforts.append(EffortLevel(value.strip()))
    if len(efforts) > 1:
        raise ValueError("multiple effort overrides")
    return efforts[0] if efforts else None


def _cursor_model_base(model: str) -> str:
    """Translate a CLI variant ID to the base ID exposed by ACP's model picker."""
    base = model.split("[", 1)[0].removesuffix("-fast")
    effort_suffixes = (
        "-extra-high",
        "-xhigh",
        "-medium",
        "-high",
        "-low",
        "-max",
    )
    for _pass in range(2):
        if base.endswith("-thinking"):
            base = base.removesuffix("-thinking")
        for suffix in effort_suffixes:
            if base.endswith(suffix):
                base = base.removesuffix(suffix)
                break
    return base


def _cursor_config_option(
    result: dict[str, Any],
    *,
    option_id: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """Return one validated Cursor ACP configuration option from a response."""
    options = result.get("configOptions")
    if not isinstance(options, list) or any(not isinstance(item, dict) for item in options):
        raise ValueError("Cursor ACP omitted valid session configuration options")
    matches = [
        item
        for item in options
        if (option_id is None or item.get("id") == option_id)
        and (category is None or item.get("category") == category)
    ]
    if len(matches) != 1:
        target = option_id or category or "requested"
        raise ValueError(f"Cursor ACP did not advertise exactly one {target} configuration option")
    option = matches[0]
    if not isinstance(option.get("id"), str) or not isinstance(option.get("currentValue"), str):
        raise ValueError("Cursor ACP returned malformed session configuration state")
    values = option.get("options")
    if not isinstance(values, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("value"), str) for item in values
    ):
        raise ValueError("Cursor ACP returned malformed session configuration choices")
    return cast(dict[str, Any], option)


def _cursor_optional_config_option(
    result: dict[str, Any], *, option_id: str
) -> dict[str, Any] | None:
    """Return one validated option when the selected model exposes it."""
    options = result.get("configOptions")
    if not isinstance(options, list) or any(not isinstance(item, dict) for item in options):
        raise ValueError("Cursor ACP omitted valid session configuration options")
    matches = [item for item in options if item.get("id") == option_id]
    if not matches:
        return None
    return _cursor_config_option(result, option_id=option_id)


def _cursor_effort_value(option: dict[str, Any], effort: EffortLevel) -> str | None:
    """Map Crossby's effort vocabulary to one exact ACP option value."""
    values = {item["value"] for item in option["options"]}
    candidates = ("xhigh", "extra-high") if effort is EffortLevel.XHIGH else (effort.value,)
    return next((candidate for candidate in candidates if candidate in values), None)


def _cursor_effort_option(
    result: dict[str, Any], effort: EffortLevel
) -> tuple[dict[str, Any], str] | None:
    """Select the one thought-level option whose choices encode ``effort``."""
    options = result.get("configOptions")
    if not isinstance(options, list) or any(not isinstance(item, dict) for item in options):
        raise ValueError("Cursor ACP omitted valid session configuration options")
    matches: list[tuple[dict[str, Any], str]] = []
    for item in options:
        if item.get("category") != "thought_level":
            continue
        option_id = item.get("id")
        if not isinstance(option_id, str) or not option_id:
            raise ValueError("Cursor ACP returned malformed thought-level configuration")
        option = _cursor_config_option(result, option_id=option_id)
        value = _cursor_effort_value(option, effort)
        if value is not None:
            matches.append((option, value))
    if len(matches) > 1:
        raise ValueError("Cursor ACP advertised ambiguous reasoning-effort configuration")
    return matches[0] if matches else None


class CursorAdapter(AbstractAITool):
    """Adapter for Cursor CLI (``agent`` binary).

    Cursor is an AI-powered IDE with a terminal CLI that supports plan mode,
    model selection, headless execution, and skill discovery.

    Cursor uses its own model ID namespace — e.g. ``claude-sonnet-4-6``,
    ``claude-opus-4-6``, ``gpt-5.3-codex`` — so no format normalization is
    needed.

    Cursor has no effort flag: effort is part of the model ID
    (``claude-opus-5-5-high``), so ``resolve_effort_model`` swaps in the catalog
    sibling that carries the requested effort.
    """

    TOOL_ID: ClassVar[AIToolID] = AIToolID.CURSOR

    def capabilities(self) -> AIToolCapabilities:
        return AIToolCapabilities(
            tool_id=AIToolID.CURSOR,
            display_name="Cursor",
            binary="agent",
            tool_type=AIToolType.TERMINAL,
            # `agent update` — updates Cursor Agent to the latest version. The
            # `agent` and `cursor-agent` binaries are the same tool; `agent`
            # matches the launch/probe binary, so the update tuple reuses it
            # rather than the `cursor-agent` alias.
            update_command=("agent", "update"),
            supports_model_flag=True,
            headless_flag="--print",
            supports_headless=True,
            supports_effort=True,
            supports_yolo=True,
            headless=HeadlessCapability(
                transport=HeadlessNativeTransport.HEADLESS_CLI,
                prompt_transport=HeadlessPromptTransport.ARGUMENT,
                # Cursor's print mode has no verified stdin contract, and its
                # plain-text output carries no terminal status, so only the two
                # execution-output formats are offered.
                native_outputs=(HeadlessNativeOutput.JSON, HeadlessNativeOutput.JSONL),
                interaction_modes=(HeadlessInteractionMode.UNATTENDED,),
                supports_response_schema=False,
                successful_native_statuses=("success",),
                sandbox_behavior=PlanRequestBehavior.PRESERVED,
                approval_behavior=PlanRequestBehavior.PRESERVED,
                command_policy_support=PlanCommandPolicySupport.UNSUPPORTED,
                version_requirement=(
                    "Cursor Agent exposing --print with --output-format json|stream-json "
                    "and --trust."
                ),
                verified_version="2026.09.10-fd3934a",
                remediation=(
                    "Upgrade Cursor Agent to the 2026.09.10 build or newer. Cursor exposes no "
                    "final-response JSON Schema flag, so schema-constrained sessions must use "
                    "Claude Code, Codex CLI, or Antigravity CLI."
                ),
            ),
            plan_mode=PlanModeCapability(
                supported_launch_approval_modes=(
                    PlanLaunchApprovalMode.YOLO,
                    PlanLaunchApprovalMode.AUTO,
                ),
                activation=PlanModeActivation.CLI_ARGUMENT,
                activation_detail="Passes --mode plan before the first user turn.",
                version_requirement="Cursor Agent exposing --mode plan.",
                verified_version="2026.09.02-c22c1a3",
                initial_prompt_after_activation=True,
                artifact_location=PlanArtifactLocation.SESSION,
                artifact_location_detail=(
                    "Cursor returns its plan in the interactive session and exposes no launch "
                    "option that guarantees an on-disk plan file at a requested path."
                ),
                remediation=(
                    "Use run_plan_session() for ACP collection, or use Claude when a specific "
                    "filesystem output directory is required."
                ),
                collector_activation=PlanModeActivation.ACP,
                transport=PlanSessionTransport.ACP,
                artifact_source=PlanArtifactSource.PROTOCOL_EVENT,
                binding=PlanSessionBinding.SESSION_ID,
                interaction=PlanInteractionSupport.CALLBACK,
                sandbox_behavior=PlanRequestBehavior.PRESERVED,
                approval_behavior=PlanRequestBehavior.PRESERVED,
                supported_approval_policies=(
                    PlanApprovalPolicy.ON_REQUEST,
                    PlanApprovalPolicy.NEVER,
                ),
                command_policy_support=PlanCommandPolicySupport.UNSUPPORTED,
                command_policy_detail=(
                    "The verified Cursor ACP permission request omits authoritative raw command "
                    "input and exposes only display text."
                ),
            ),
            supports_accept_edits=True,
            supports_auto=True,
            supports_sandbox_toggle=True,
            supports_stop_hook=True,
            supports_user_prompt_submit_hook=True,
            # Cursor does fire `sessionStart`, and its `additional_context` does
            # reach the model — both verified against cursor-agent
            # 2026.04.17. Caveats worth knowing before relying on it:
            #   * it is fire-and-forget — the agent loop does not wait for or
            #     enforce a blocking response, so it can inject but not gate;
            #   * it is not supported in cloud agents;
            #   * which events `cursor-agent` fires has varied across versions,
            #     so treat CLI coverage as version-dependent rather than
            #     guaranteed by the docs.
            supports_session_start_hook=True,
            hook_output_dialect=HookOutputDialect.PERMISSION,
            hook_stop_dialect=HookStopDialect.FOLLOWUP_MESSAGE,
            # Cursor is the one tool that defaults hooks to fail-*open*, so a
            # security guard must set HookEntry.fail_closed to be worth
            # anything. `failClosed` is a generic per-script option: Cursor's
            # published docs name it only for beforeShellExecution /
            # beforeMCPExecution / beforeReadFile, but the shipped
            # implementation also honours it on preToolUse — verified by
            # reading cursor-agent's bundled hook runtime, where preToolUse is
            # in the fail-closed event set and a failed hook there is converted
            # into {"permission": "deny"}. Documented as under-documented
            # rather than unsupported.
            hook_fail_open_default=True,
            # No session-scoped scene lever: CURSOR_CONFIG_DIR is Cursor's only
            # per-session config knob, but it relocates the *entire* config base
            # (auth + cli-config.json, not just mcp.json), so pointing it at a
            # scene-only dir would launch Cursor unauthenticated. crossby falls
            # back to persistent ``scene use`` activation for Cursor instead.
        )

    def sandbox_config_args(
        self,
        *,
        autonomy_args: list[str],
        trusted_dirs: list[str] | None,
        working_dir: Path | None,
        network_access: bool,
        sandbox: bool = True,
    ) -> list[str]:
        """Select Cursor's sandbox explicitly for every adapter launch."""
        if not self.capabilities().supports_sandbox_toggle:
            return super().sandbox_config_args(
                autonomy_args=autonomy_args,
                trusted_dirs=trusted_dirs,
                working_dir=working_dir,
                network_access=network_access,
            )
        return ["--sandbox", "enabled" if sandbox else "disabled"]

    def initial_message_args(self, prompt: str) -> list[str]:
        """Cursor accepts the initial message as a positional argument."""
        return [prompt]

    def locate_sessions(self, project_path: Path) -> list[SessionRef]:
        return cursor_reader.locate_sessions(project_path)

    def read_session(self, ref: SessionRef) -> ConversationTranscript:
        return cursor_reader.read_session(ref)

    def is_model_compatible(self, model: str) -> bool:
        """Cursor accepts all model IDs."""
        return True

    def plan_mode_args(self) -> list[str]:
        """Cursor supports ``--mode plan``."""
        return ["--mode", "plan"]

    def _validate_headless_requirements(self, request: HeadlessSessionRequest) -> None:
        """Reject effort requests that Cursor cannot encode without a model."""
        if request.effort is not None and not request.model:
            from crossby.ai_tools.headless import HeadlessRequestError

            raise HeadlessRequestError(
                "Cursor requires an explicit model when effort is requested.",
                tool_id=self.TOOL_ID,
                capability=self.capabilities().headless,
            )

    def _headless_argv_for_validation(
        self, request: HeadlessSessionRequest, **_kwargs: object
    ) -> list[str]:
        """Return Cursor's complete command, including its argument-delivered prompt."""
        return self._headless_command(request)

    def _headless_command(self, request: HeadlessSessionRequest) -> list[str]:
        """Build the exact unattended ``agent --print`` invocation."""
        command = [
            "agent",
            "--print",
            "--output-format",
            "json" if request.native_output is HeadlessNativeOutput.JSON else "stream-json",
            # Without this Cursor stops on its interactive workspace-trust
            # gate, which an unattended session can never answer. Approval of
            # individual tool calls stays separate: --force/--yolo is never
            # emitted here.
            "--trust",
            "--sandbox",
            "enabled" if request.sandbox else "disabled",
        ]
        model = self.resolve_effort_model(request.model, request.effort)
        if model:
            command.extend(("--model", model))
        command.append(request.prompt)
        return command

    def _run_headless_session(
        self,
        request: HeadlessSessionRequest,
        version: str,
        context: HeadlessRuntimeContext,
    ) -> HeadlessSessionResult:
        """Run one unattended Cursor print turn and normalize its result envelope."""
        from crossby.ai_tools.headless_cli import (
            capture_failure_warnings,
            complete_session,
            frame_streamer,
            non_blank_text,
            parse_json_lines,
            parse_json_object,
            run_managed_command,
            usage_from,
        )
        from crossby.ai_tools.plan_process import child_environment

        streaming = request.native_output is HeadlessNativeOutput.JSONL
        output = run_managed_command(
            context,
            argv=self._headless_command(request),
            cwd=request.working_dir,
            env=child_environment({"NO_COLOR": "1"}),
            on_stdout_lines=(
                frame_streamer(
                    context,
                    label="cursor",
                    kind_of=lambda frame: non_blank_text(frame.get("type")),
                    provenance_of=lambda frame: {
                        "session_id": non_blank_text(frame.get("session_id"))
                    },
                )
                if streaming
                else None
            ),
        )
        warnings = capture_failure_warnings(output)
        # Print mode was not granted force/yolo, so Cursor decides each tool call
        # under its own default policy: file changes can remain proposals rather
        # than applied edits, and nothing escalates to a terminal prompt.
        warnings = (
            *warnings,
            "Cursor print mode ran without --force/--yolo, so tool calls needing explicit "
            "approval are not auto-approved and file changes may remain proposals.",
        )
        if output.overflowed or output.undecodable:
            return context.complete(
                HeadlessTerminalStatus.INVALID_OUTPUT,
                exit_code=output.returncode,
                warnings=warnings,
            )

        session_id: str | None = None
        envelope: dict[str, Any] | None = None
        if request.native_output is HeadlessNativeOutput.JSON:
            envelope = parse_json_object(output.stdout)
            if envelope is not None and envelope.get("type") != "result":
                envelope = None
        else:
            frames = parse_json_lines(output.stdout)
            if frames is None:
                return context.complete(
                    HeadlessTerminalStatus.INVALID_OUTPUT,
                    exit_code=output.returncode,
                    warnings=(*warnings, "Cursor emitted malformed streaming JSON."),
                )
            # Progress and provenance were already emitted live by the streamer,
            # which never copies Cursor's prompt-echoing ``user`` frame content.
            for frame in frames:
                session_id = session_id or non_blank_text(frame.get("session_id"))
                if non_blank_text(frame.get("type")) == "result":
                    envelope = frame
        if envelope is None:
            return context.complete(
                HeadlessTerminalStatus.INVALID_OUTPUT,
                exit_code=output.returncode,
                session_id=session_id,
                warnings=(*warnings, "Cursor did not emit its final result envelope."),
            )

        session_id = session_id or non_blank_text(envelope.get("session_id"))
        response_text = non_blank_text(envelope.get("result"))
        native_error = bool(envelope.get("is_error"))
        if native_error:
            warnings = (
                *warnings,
                "Cursor reported a native error result.",
            )
        return complete_session(
            context,
            request,
            status=(
                HeadlessTerminalStatus.FAILED
                if native_error or output.returncode != 0
                else HeadlessTerminalStatus.SUCCEEDED
            ),
            exit_code=output.returncode,
            response_text=response_text,
            native_object=envelope,
            native_status=non_blank_text(envelope.get("subtype")),
            session_id=session_id,
            usage=usage_from(
                envelope.get("usage"),
                input_tokens="inputTokens",
                output_tokens="outputTokens",
                cached="cacheReadTokens",
                session_id=session_id,
            ),
            warnings=warnings,
        )

    def _validate_collected_plan_requirements(self, request: PlanSessionRequest) -> None:
        """Validate model/effort encodings before prospective workspace creation."""
        from crossby.ai_tools.plan_mode import PlanSessionUnsupportedError

        capability = self.capabilities().plan_mode
        if request.model is None or not request.model.strip() or request.effort is None:
            raise PlanSessionUnsupportedError(
                "Cursor requires an explicit model and effort for collected plan sessions.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        try:
            parameterized_effort = _parameterized_effort(request.model)
        except ValueError as exc:
            raise PlanSessionUnsupportedError(
                f"Cursor model {request.model!r} has invalid effort overrides: {exc}.",
                tool_id=self.TOOL_ID,
                capability=capability,
            ) from exc
        encoded_effort = _encoded_effort(request.model)
        if (
            parameterized_effort is not None
            and encoded_effort is not None
            and parameterized_effort is not encoded_effort
        ):
            raise PlanSessionUnsupportedError(
                f"Cursor model {request.model!r} contains conflicting effort encodings.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        declared_effort = parameterized_effort or encoded_effort
        if declared_effort is not None and declared_effort is not request.effort:
            raise PlanSessionUnsupportedError(
                f"Cursor model {request.model!r} encodes effort={declared_effort.value!r}, "
                f"which conflicts with requested effort={request.effort.value!r} for a "
                "collected plan session.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        if request.model.split("[", 1)[0] == "auto":
            raise PlanSessionUnsupportedError(
                f"Cursor cannot preserve effort={request.effort.value!r} with model='auto' "
                "because the selected model is not known before launch.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )

    def _run_plan_session(
        self,
        request: PlanSessionRequest,
        version: str,
        interaction_handler: PlanInteractionHandler | None,
    ) -> PlanSessionResult:
        """Collect the blocking ``cursor/create_plan`` request from one ACP session."""
        from crossby.ai_tools.plan_mode import (
            PlanArtifactAmbiguousError,
            PlanArtifactMalformedError,
            PlanArtifactMissingError,
            PlanInteractionRequiredError,
            PlanSessionError,
            PlanSessionUnsupportedError,
            PlanTransportError,
        )
        from crossby.ai_tools.plan_process import JsonRpcProcess

        capability = self.capabilities().plan_mode
        assert request.model is not None and request.effort is not None
        command = ["agent", "--sandbox", "enabled" if request.sandbox else "disabled"]
        effective_model = request.model
        model_base = _cursor_model_base(effective_model)
        thinking_value = "true" if "-thinking" in effective_model.split("[", 1)[0] else "false"
        fast_value = "true" if effective_model.split("[", 1)[0].endswith("-fast") else "false"
        command.append("acp")
        deadline = time.monotonic() + request.timeout_seconds
        rpc: JsonRpcProcess | None = None
        rpc_closed = False

        def remaining() -> float:
            wait = deadline - time.monotonic()
            if wait <= 0:
                raise TimeoutError("Cursor ACP plan session exceeded its timeout")
            return wait

        def wait_response(request_id: int) -> dict[str, Any]:
            assert rpc is not None
            while True:
                message = rpc.read(timeout=remaining())
                if message.get("id") != request_id:
                    if "method" in message and "id" in message:
                        raise PlanTransportError(
                            f"Cursor ACP requested {message.get('method')!r} before session setup "
                            "completed.",
                            tool_id=self.TOOL_ID,
                            capability=capability,
                        )
                    continue
                if "error" in message:
                    raise PlanTransportError(
                        f"Cursor ACP rejected request {request_id}.",
                        tool_id=self.TOOL_ID,
                        capability=capability,
                        stderr=rpc.stderr,
                    )
                result = message.get("result")
                if not isinstance(result, dict):
                    raise PlanTransportError(
                        "Cursor ACP returned a malformed response.",
                        tool_id=self.TOOL_ID,
                        capability=capability,
                    )
                return result

        try:
            rpc = JsonRpcProcess(command, cwd=request.working_dir, timeout=remaining())
            rpc.request(
                1,
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {"_meta": {"parameterizedModelPicker": True}},
                    "clientInfo": {"name": "crossby", "version": "plan-session-v1"},
                },
            )
            initialized = wait_response(1)
            auth_methods = initialized.get("authMethods")
            if isinstance(auth_methods, list) and auth_methods:
                first = auth_methods[0]
                method_id = first.get("id") if isinstance(first, dict) else first
                if not isinstance(method_id, str) or not method_id:
                    raise PlanTransportError(
                        "Cursor ACP advertised a malformed authentication method.",
                        tool_id=self.TOOL_ID,
                        capability=capability,
                    )
                rpc.request(2, "authenticate", {"methodId": method_id})
                wait_response(2)

            rpc.request(
                3,
                "session/new",
                {"cwd": str(request.working_dir.resolve()), "mcpServers": []},
            )
            session_result = wait_response(3)
            session_id = session_result.get("sessionId")
            if not isinstance(session_id, str) or not session_id.strip():
                raise PlanTransportError(
                    "Cursor ACP session/new response omitted sessionId.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                )
            modes = session_result.get("modes")
            available = modes.get("availableModes") if isinstance(modes, dict) else None
            if not isinstance(available, list) or not any(
                (mode.get("id") if isinstance(mode, dict) else mode) == "plan" for mode in available
            ):
                raise PlanSessionUnsupportedError(
                    "Cursor ACP did not advertise the native plan session mode.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=session_id,
                )
            model_option = _cursor_config_option(session_result, option_id="model")
            model_values = {item["value"] for item in model_option["options"]}
            if model_base not in model_values:
                raise PlanSessionUnsupportedError(
                    f"Cursor ACP did not advertise requested model {request.model!r}.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=session_id,
                )
            rpc.request(
                4,
                "session/set_config_option",
                {"sessionId": session_id, "configId": "model", "value": model_base},
            )
            configured_model = wait_response(4)
            model_option = _cursor_config_option(configured_model, option_id="model")
            if model_option["currentValue"] != model_base:
                raise PlanTransportError(
                    "Cursor ACP did not preserve the requested model.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=session_id,
                )
            effort_config = _cursor_effort_option(configured_model, request.effort)
            if effort_config is None:
                raise PlanSessionUnsupportedError(
                    f"Cursor ACP model {request.model!r} cannot preserve "
                    f"effort={request.effort.value!r}.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=session_id,
                )
            effort_option, effort_value = effort_config
            rpc.request(
                5,
                "session/set_config_option",
                {
                    "sessionId": session_id,
                    "configId": effort_option["id"],
                    "value": effort_value,
                },
            )
            configuration_state = wait_response(5)
            confirmed_effort = _cursor_effort_option(configuration_state, request.effort)
            if (
                confirmed_effort is None
                or confirmed_effort[0]["id"] != effort_option["id"]
                or confirmed_effort[0]["currentValue"] != effort_value
            ):
                raise PlanTransportError(
                    "Cursor ACP did not preserve the requested reasoning effort.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=session_id,
                )
            next_request_id = 6
            configured_variants: dict[str, str] = {}
            for variant_id, variant_value in (
                ("thinking", thinking_value),
                ("fast", fast_value),
            ):
                variant_option = _cursor_optional_config_option(
                    configuration_state,
                    option_id=variant_id,
                )
                if variant_option is None:
                    if variant_value == "true":
                        raise PlanSessionUnsupportedError(
                            f"Cursor ACP model {request.model!r} cannot preserve its "
                            f"{variant_id} setting.",
                            tool_id=self.TOOL_ID,
                            capability=capability,
                            session_id=session_id,
                        )
                    continue
                if variant_value not in {item["value"] for item in variant_option["options"]}:
                    raise PlanSessionUnsupportedError(
                        f"Cursor ACP model {request.model!r} cannot preserve its "
                        f"{variant_id} setting.",
                        tool_id=self.TOOL_ID,
                        capability=capability,
                        session_id=session_id,
                    )
                rpc.request(
                    next_request_id,
                    "session/set_config_option",
                    {
                        "sessionId": session_id,
                        "configId": variant_option["id"],
                        "value": variant_value,
                    },
                )
                configuration_state = wait_response(next_request_id)
                confirmed_variant = _cursor_config_option(
                    configuration_state,
                    option_id=variant_id,
                )
                if confirmed_variant["currentValue"] != variant_value:
                    raise PlanTransportError(
                        f"Cursor ACP did not preserve the requested {variant_id} model setting.",
                        tool_id=self.TOOL_ID,
                        capability=capability,
                        session_id=session_id,
                    )
                configured_variants[variant_id] = variant_value
                next_request_id += 1
            final_model = _cursor_config_option(configuration_state, option_id="model")
            final_effort = _cursor_effort_option(configuration_state, request.effort)
            if final_model["currentValue"] != model_base:
                raise PlanTransportError(
                    "Cursor ACP reset the requested model while configuring the session.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=session_id,
                )
            if (
                final_effort is None
                or final_effort[0]["id"] != effort_option["id"]
                or final_effort[0]["currentValue"] != effort_value
            ):
                raise PlanTransportError(
                    "Cursor ACP reset the requested reasoning effort while configuring the "
                    "session.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=session_id,
                )
            for variant_id, variant_value in configured_variants.items():
                final_variant = _cursor_config_option(
                    configuration_state,
                    option_id=variant_id,
                )
                if final_variant["currentValue"] != variant_value:
                    raise PlanTransportError(
                        f"Cursor ACP reset the requested {variant_id} model setting while "
                        "configuring the session.",
                        tool_id=self.TOOL_ID,
                        capability=capability,
                        session_id=session_id,
                    )
            rpc.request(
                next_request_id,
                "session/set_mode",
                {"sessionId": session_id, "modeId": "plan"},
            )
            wait_response(next_request_id)
            prompt_request_id = next_request_id + 1
            rpc.request(
                prompt_request_id,
                "session/prompt",
                {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": request.prompt}],
                },
            )

            plans: list[tuple[str, str]] = []
            prompt_completed = False
            while not prompt_completed:
                message = rpc.read(timeout=remaining())
                if message.get("id") == prompt_request_id and "method" not in message:
                    if "error" in message:
                        raise PlanTransportError(
                            "Cursor ACP session/prompt failed.",
                            tool_id=self.TOOL_ID,
                            capability=capability,
                            session_id=session_id,
                            stderr=rpc.stderr,
                        )
                    result = message.get("result")
                    if not isinstance(result, dict):
                        raise PlanTransportError(
                            "Cursor ACP returned a malformed session/prompt response.",
                            tool_id=self.TOOL_ID,
                            capability=capability,
                            session_id=session_id,
                        )
                    stop_reason = result.get("stopReason")
                    if not isinstance(stop_reason, str) or not stop_reason.strip():
                        raise PlanTransportError(
                            "Cursor ACP returned a malformed session/prompt stopReason.",
                            tool_id=self.TOOL_ID,
                            capability=capability,
                            session_id=session_id,
                        )
                    if stop_reason != "end_turn":
                        raise PlanTransportError(
                            f"Cursor ACP session/prompt ended with stop reason {stop_reason!r}.",
                            tool_id=self.TOOL_ID,
                            capability=capability,
                            session_id=session_id,
                        )
                    prompt_completed = True
                    continue
                method = message.get("method")
                if method == "cursor/create_plan" and "id" in message:
                    params = _cursor_bound_params(
                        message,
                        session_id=session_id,
                        tool_id=self.TOOL_ID,
                        capability=capability,
                    )
                    plan = params.get("plan") or params.get("markdown") or params.get("content")
                    if not isinstance(plan, str):
                        raise PlanArtifactMalformedError(
                            "Cursor create_plan request omitted Markdown content.",
                            tool_id=self.TOOL_ID,
                            capability=capability,
                            session_id=session_id,
                            artifact_id=str(message["id"]),
                        )
                    tool_call_id = params.get("toolCallId")
                    artifact_id = (
                        tool_call_id
                        if isinstance(tool_call_id, str) and tool_call_id.strip()
                        else str(message["id"])
                    )
                    plans.append((artifact_id, plan))
                    interaction = PlanInteraction(
                        kind=PlanInteractionKind.PLAN_APPROVAL,
                        question_id=artifact_id,
                        prompt=str(
                            params.get("overview")
                            or params.get("name")
                            or "Cursor produced the plan. Choose a non-executing outcome."
                        ),
                        options=(
                            PlanQuestionOption(option_id="rejected", label="Keep plan only"),
                            PlanQuestionOption(option_id="cancelled", label="Cancel"),
                        ),
                        session_id=session_id,
                        artifact_id=artifact_id,
                    )
                    if interaction_handler is None:
                        raise PlanInteractionRequiredError(
                            "Cursor requires a final plan outcome; Crossby will not approve "
                            "implementation automatically.",
                            interaction=interaction,
                            tool_id=self.TOOL_ID,
                            capability=capability,
                        )
                    outcome = interaction_handler(interaction)
                    if outcome.outcome is PlanInteractionOutcome.APPROVED:
                        raise PlanInteractionRequiredError(
                            "Approving Cursor's plan would authorize implementation; choose a "
                            "non-executing outcome.",
                            interaction=interaction,
                            tool_id=self.TOOL_ID,
                            capability=capability,
                        )
                    if outcome.outcome is PlanInteractionOutcome.CANCELLED:
                        native_outcome: dict[str, str] = {"outcome": "cancelled"}
                    elif outcome.outcome in {
                        PlanInteractionOutcome.DENIED,
                        PlanInteractionOutcome.SKIPPED,
                    }:
                        native_outcome = {
                            "outcome": "rejected",
                            "reason": outcome.answer or "Plan collected without implementation.",
                        }
                    else:
                        try:
                            selected = validate_plan_option_selection(interaction, outcome)
                            if len(selected) != 1:
                                raise ValueError("final plan outcome requires one native option")
                        except ValueError as exc:
                            raise PlanInteractionRequiredError(
                                "Cursor final plan outcomes require one valid native option ID.",
                                interaction=interaction,
                                tool_id=self.TOOL_ID,
                                capability=capability,
                            ) from exc
                        native_outcome = {"outcome": selected[0]}
                        if selected[0] == "rejected":
                            native_outcome["reason"] = (
                                outcome.answer or "Plan collected without implementation."
                            )
                    rpc.respond(message["id"], {"outcome": native_outcome})
                    continue
                if method == "cursor/ask_question" and "id" in message:
                    _answer_cursor_question(
                        rpc,
                        message,
                        session_id=session_id,
                        handler=interaction_handler,
                        tool_id=self.TOOL_ID,
                        capability=capability,
                    )
                    continue
                if method in {"session/request_permission", "cursor/request_permission"} and (
                    "id" in message
                ):
                    _answer_cursor_permission(
                        rpc,
                        message,
                        session_id=session_id,
                        handler=interaction_handler,
                        tool_id=self.TOOL_ID,
                        capability=capability,
                        deny_automatically=request.approval_policy is PlanApprovalPolicy.NEVER,
                    )
                    continue
                if isinstance(method, str) and "id" in message:
                    raise PlanTransportError(
                        f"Cursor ACP requested unsupported method {method!r}.",
                        tool_id=self.TOOL_ID,
                        capability=capability,
                        session_id=session_id,
                    )

            if not plans:
                raise PlanArtifactMissingError(
                    "Cursor ACP session completed without a blocking cursor/create_plan request.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=session_id,
                )
            unique_plans = list(dict.fromkeys(plan for _, plan in plans))
            if len(plans) != 1 or len(unique_plans) != 1:
                raise PlanArtifactAmbiguousError(
                    "Cursor ACP emitted multiple plan artifacts for one session.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=session_id,
                    artifact_id=",".join(item_id for item_id, _ in plans),
                )
            artifact_id, plan = plans[0]
            if not plan.strip():
                raise PlanArtifactMalformedError(
                    "Cursor ACP emitted a blank plan artifact.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=session_id,
                    artifact_id=artifact_id,
                )
            exit_code = rpc.close()
            rpc_closed = True
            if exit_code != 0:
                raise PlanTransportError(
                    f"Cursor ACP exited with status {exit_code} after completing the session.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    exit_code=exit_code,
                    session_id=session_id,
                    stderr=rpc.stderr,
                )
            return PlanSessionResult(
                tool=self.TOOL_ID,
                version=version,
                plan=plan,
                session_id=session_id,
                native_mode="ACP session/set_mode plan",
                artifact_source=PlanArtifactSource.PROTOCOL_EVENT,
                binding=PlanSessionBinding.SESSION_ID,
                exit_code=exit_code,
                artifact_id=artifact_id,
            )
        except PlanSessionError:
            raise
        except (OSError, ValueError, EOFError, TimeoutError) as exc:
            raise PlanTransportError(
                f"Cursor ACP plan session failed: {exc}",
                tool_id=self.TOOL_ID,
                capability=capability,
                stderr=rpc.stderr if rpc is not None else None,
            ) from exc
        finally:
            if rpc is not None and not rpc_closed:
                rpc.close()

    def yolo_args(self) -> list[str]:
        """Cursor uses ``--force`` (``--yolo`` is an alias)."""
        return ["--force"]

    def auto_args(self) -> list[str]:
        """Use Cursor CLI's native classifier-mediated Smart Auto run mode."""
        return ["--auto-review"]

    def accept_edits_args(self) -> list[str]:
        """No flag needed — the Cursor CLI's default Agent mode *is* accept-edits
        (auto-applies edits, prompts for shell). Declaring
        ``supports_accept_edits=True`` while returning ``[]`` means ``--accept-edits``
        is honored with no warning. (This is the inverse of the Cursor *IDE*
        default, which confirms edits and auto-runs allowlisted commands — the
        CLI and IDE differ.)"""
        return []

    def resolve_effort_model(self, model: str | None, effort: EffortLevel | None) -> str | None:
        """Swap ``model`` for the Cursor catalog ID that encodes ``effort``.

        Cursor bakes effort into the model ID, so the requested effort selects a
        sibling of ``model`` from the bundled Cursor registry:
        ``claude-opus-5-5-medium`` + HIGH -> ``claude-opus-5-5-high``. The
        requested effort overrides one already in the ID; the ``-thinking`` and
        ``-fast`` choices are kept. A bare ID whose family also ships explicit
        efforts (``gpt-5.3-codex``) stands for medium.

        When the family lacks the requested tier, the nearest offered tier is
        used (ties go higher) with a warning. A result is always a registry ID:
        unknown models, families without effort variants (``gemini-3.1-pro``),
        ``auto``, and IDs carrying Cursor's bracket overrides
        (``claude-opus-4-8[effort=high]``, which only some models accept) are
        returned unchanged. As a last resort, a bare ID with a registered
        ``<model>-thinking`` twin still upgrades to it for high/xhigh/max.
        """
        if not model or effort is None or model in _NO_THINKING_MODELS or "[" in model:
            return model

        registry = set(get_models_for_tool(AIToolID.CURSOR))
        template, current, explicit = _effort_template(model)
        offered: dict[EffortLevel, str] = {}
        for level, spellings in _EFFORT_SPELLINGS.items():
            for spelling in spellings:
                candidate = template.replace(_EFFORT_SLOT, spelling)
                if candidate in registry:
                    offered[level] = candidate
                    break
        bare = template.replace(f"-{_EFFORT_SLOT}", "")
        if offered and EffortLevel.MEDIUM not in offered and bare in registry:
            # The family's bare ID (gpt-5.3-codex) is its default, medium tier.
            offered[EffortLevel.MEDIUM] = bare
        if model == offered.get(EffortLevel.MEDIUM):
            current = EffortLevel.MEDIUM

        if not offered:
            thinking = f"{model}-thinking"
            if effort in _THINKING_EFFORTS and not explicit and thinking in registry:
                return thinking
            return model

        chosen = _nearest_effort(effort, set(offered))
        if chosen is not effort:
            kept = " (keeping the model as given)" if chosen is current else ""
            warnings.warn(
                f"Cursor offers no {effort.value!r} effort for {model!r}; "
                f"using {chosen.value!r} ({offered[chosen]!r}) instead{kept}.",
                UserWarning,
                stacklevel=2,
            )
        return offered[chosen]

    def preserve_session_data(self, working_dir: Path, main_checkout_path: Path) -> bool:
        """Copy Cursor session data from source directory to target's project dir.

        Cursor stores sessions in ``~/.cursor/projects/<encoded-path>/``.
        The path encoding strips the leading ``/`` then replaces remaining
        ``/`` with ``-``, so ``/Users/foo/bar`` becomes ``Users-foo-bar``
        (note: no leading dash, unlike Claude Code).

        Files are copied without overwriting any that already exist in the
        target's session directory, so existing data is preserved.
        """
        cursor_projects_dir = Path.home() / ".cursor" / "projects"

        wt_encoded = str(working_dir).lstrip("/").replace("/", "-")
        main_encoded = str(main_checkout_path).lstrip("/").replace("/", "-")

        wt_session_dir = cursor_projects_dir / wt_encoded
        main_session_dir = cursor_projects_dir / main_encoded

        if not wt_session_dir.exists():
            logger.debug(
                "cursor.preserve_session_data.no_source",
                working_dir=str(working_dir),
            )
            return True

        main_session_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for item in wt_session_dir.iterdir():
            dest = main_session_dir / item.name
            if dest.exists():
                continue
            if item.is_file():
                shutil.copy2(item, dest)
                copied += 1
            elif item.is_dir():
                shutil.copytree(item, dest)
                copied += 1

        logger.info(
            "cursor.preserve_session_data.copied",
            working_dir=str(working_dir),
            main=str(main_checkout_path),
            items=copied,
        )
        return True

    def session_data_dirs(self) -> list[str]:
        return [".cursor"]


def _cursor_bound_params(
    message: dict[str, Any],
    *,
    session_id: str,
    tool_id: AIToolID,
    capability: PlanModeCapability,
    require_session_id: bool = False,
) -> dict[str, Any]:
    from crossby.ai_tools.plan_mode import PlanBindingMismatchError, PlanTransportError

    params = message.get("params")
    if not isinstance(params, dict):
        raise PlanTransportError(
            "Cursor ACP request had malformed params.",
            tool_id=tool_id,
            capability=capability,
            session_id=session_id,
        )
    # Cursor's extension methods (cursor/ask_question and cursor/create_plan)
    # omit sessionId in the verified ACP shape. This client creates exactly one
    # session in a dedicated child process, so request correlation plus that
    # process boundary is the binding; standard ACP permission requests do carry
    # sessionId and require it below.
    emitted_session_id = params.get("sessionId")
    if (require_session_id and emitted_session_id != session_id) or (
        emitted_session_id is not None and emitted_session_id != session_id
    ):
        raise PlanBindingMismatchError(
            "Cursor ACP request belonged to a different session.",
            tool_id=tool_id,
            capability=capability,
            session_id=session_id,
            artifact_id=str(message.get("id") or "") or None,
        )
    return params


def _answer_cursor_question(
    rpc: Any,
    message: dict[str, Any],
    *,
    session_id: str,
    handler: PlanInteractionHandler | None,
    tool_id: AIToolID,
    capability: PlanModeCapability,
) -> None:
    from crossby.ai_tools.plan_mode import PlanInteractionRequiredError, PlanTransportError

    params = _cursor_bound_params(
        message,
        session_id=session_id,
        tool_id=tool_id,
        capability=capability,
    )
    raw_questions = params.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise PlanTransportError(
            "Cursor ACP ask_question request contained no native questions.",
            tool_id=tool_id,
            capability=capability,
            session_id=session_id,
        )
    tool_call_id = str(params.get("toolCallId") or message.get("id"))
    interactions: list[PlanInteraction] = []
    question_ids: set[str] = set()
    for raw_question in raw_questions:
        if not isinstance(raw_question, dict):
            raise PlanTransportError(
                "Cursor ACP emitted a malformed planning question.",
                tool_id=tool_id,
                capability=capability,
                session_id=session_id,
            )
        question_id = raw_question.get("id")
        prompt = raw_question.get("prompt")
        if not isinstance(question_id, str) or not isinstance(prompt, str):
            raise PlanTransportError(
                "Cursor ACP planning question omitted its ID or prompt.",
                tool_id=tool_id,
                capability=capability,
                session_id=session_id,
            )
        if not question_id.strip() or not prompt.strip():
            raise PlanTransportError(
                "Cursor ACP planning question contained a blank ID or prompt.",
                tool_id=tool_id,
                capability=capability,
                session_id=session_id,
            )
        if question_id in question_ids:
            raise PlanTransportError(
                "Cursor ACP planning questions contained a duplicate native ID.",
                tool_id=tool_id,
                capability=capability,
                session_id=session_id,
            )
        question_ids.add(question_id)
        allow_multiple = raw_question.get("allowMultiple", False)
        if not isinstance(allow_multiple, bool):
            raise PlanTransportError(
                "Cursor ACP planning question had a non-boolean multi-select field.",
                tool_id=tool_id,
                capability=capability,
                session_id=session_id,
            )
        options = parse_plan_question_options(
            raw_question.get("options"),
            require_id=True,
        )
        interactions.append(
            PlanInteraction(
                kind=PlanInteractionKind.QUESTION,
                question_id=question_id,
                prompt=prompt,
                options=options,
                allow_multiple=allow_multiple,
                session_id=session_id,
                artifact_id=tool_call_id,
            )
        )

    native_answers: list[dict[str, Any]] = []
    for interaction in interactions:
        if handler is None:
            raise PlanInteractionRequiredError(
                "Cursor requires an answer to continue the planning session.",
                interaction=interaction,
                tool_id=tool_id,
                capability=capability,
            )
        response = handler(interaction)
        if response.outcome in {
            PlanInteractionOutcome.DENIED,
            PlanInteractionOutcome.CANCELLED,
            PlanInteractionOutcome.SKIPPED,
        }:
            rpc.respond(
                message["id"],
                {
                    "outcome": {
                        "outcome": (
                            "cancelled"
                            if response.outcome is PlanInteractionOutcome.CANCELLED
                            else "skipped"
                        ),
                        "reason": response.answer or "Question was not answered.",
                    }
                },
            )
            return
        selected_ids = (
            *((response.option_id,) if response.option_id is not None else ()),
            *response.option_ids,
        )
        if response.answer and selected_ids:
            raise PlanInteractionRequiredError(
                "Cursor planning questions cannot combine answer text with native option IDs.",
                interaction=interaction,
                tool_id=tool_id,
                capability=capability,
            )
        if response.answer and not selected_ids:
            textual_matches = tuple(
                option.option_id
                for option in interaction.options
                if response.answer in {option.option_id, option.label}
            )
            if len(textual_matches) == 1:
                selected_ids = textual_matches
        valid_ids = {option.option_id for option in interaction.options}
        if (
            not selected_ids
            or len(set(selected_ids)) != len(selected_ids)
            or any(option_id not in valid_ids for option_id in selected_ids)
            or (not interaction.allow_multiple and len(selected_ids) != 1)
        ):
            raise PlanInteractionRequiredError(
                "Cursor planning questions require valid native option IDs.",
                interaction=interaction,
                tool_id=tool_id,
                capability=capability,
            )
        native_answers.append(
            {
                "questionId": interaction.question_id,
                "selectedOptionIds": list(selected_ids),
            }
        )
    rpc.respond(
        message["id"],
        {"outcome": {"outcome": "answered", "answers": native_answers}},
    )


def _answer_cursor_permission(
    rpc: Any,
    message: dict[str, Any],
    *,
    session_id: str,
    handler: PlanInteractionHandler | None,
    tool_id: AIToolID,
    capability: PlanModeCapability,
    deny_automatically: bool,
) -> None:
    from crossby.ai_tools.plan_mode import PlanInteractionRequiredError, PlanTransportError

    params = _cursor_bound_params(
        message,
        session_id=session_id,
        tool_id=tool_id,
        capability=capability,
        require_session_id=True,
    )
    options = parse_plan_question_options(
        params.get("options"),
        id_fields=("optionId",),
        label_fields=("name", "label"),
        require_id=True,
    )
    tool_call = params.get("toolCall")
    tool_call_id = tool_call.get("toolCallId") if isinstance(tool_call, dict) else None
    title = tool_call.get("title") if isinstance(tool_call, dict) else None
    native_kind = tool_call.get("kind") if isinstance(tool_call, dict) else None
    if not isinstance(native_kind, str):
        native_kind = ""
    operation_kind = {
        "execute": PlanOperationKind.COMMAND,
        "edit": PlanOperationKind.FILE_CHANGE,
        "delete": PlanOperationKind.FILE_CHANGE,
        "move": PlanOperationKind.FILE_CHANGE,
        "fetch": PlanOperationKind.NETWORK,
    }.get(native_kind, PlanOperationKind.OTHER)
    raw_input = tool_call.get("rawInput") if isinstance(tool_call, dict) else None
    argv: tuple[str, ...] | None = None
    shell_expression: str | None = None
    execution_dir: Path | None = None
    if isinstance(raw_input, dict):
        has_argv_field = "argv" in raw_input
        has_command_field = "command" in raw_input
        raw_argv = raw_input.get("argv")
        raw_command = raw_input.get("command")
        parsed_argv: tuple[str, ...] | None = None
        if (
            isinstance(raw_argv, list)
            and raw_argv
            and all(isinstance(value, str) for value in raw_argv)
        ):
            parsed_argv = tuple(raw_argv)
        has_command = isinstance(raw_command, str) and bool(raw_command.strip())
        if has_argv_field and has_command_field:
            raise PlanTransportError(
                "Cursor ACP permission request contained conflicting command representations.",
                tool_id=tool_id,
                capability=capability,
                session_id=session_id,
            )
        if parsed_argv is not None:
            argv = parsed_argv
        elif has_command:
            shell_expression = raw_command
        raw_cwd = raw_input.get("cwd")
        raw_working_directory = raw_input.get("workingDirectory")
        cwd = raw_cwd if isinstance(raw_cwd, str) and raw_cwd.strip() else None
        working_directory = (
            raw_working_directory
            if isinstance(raw_working_directory, str) and raw_working_directory.strip()
            else None
        )
        if cwd is not None and working_directory is not None and cwd != working_directory:
            raise PlanTransportError(
                "Cursor ACP permission request contained conflicting working directories.",
                tool_id=tool_id,
                capability=capability,
                session_id=session_id,
            )
        selected_directory = cwd if cwd is not None else working_directory
        if selected_directory is not None:
            execution_dir = Path(selected_directory)
    locations = tool_call.get("locations") if isinstance(tool_call, dict) else None
    permission_targets: list[PlanPermissionTarget] = []
    if locations is not None:
        if not isinstance(locations, list):
            raise PlanTransportError(
                "Cursor ACP permission request contained malformed locations.",
                tool_id=tool_id,
                capability=capability,
                session_id=session_id,
            )
        for location in locations:
            path = location.get("path") if isinstance(location, dict) else None
            if not isinstance(path, str) or not path.strip():
                raise PlanTransportError(
                    "Cursor ACP permission request contained malformed locations.",
                    tool_id=tool_id,
                    capability=capability,
                    session_id=session_id,
                )
            permission_targets.append(
                PlanPermissionTarget(
                    kind=(
                        PlanPermissionTargetKind.FILESYSTEM_WRITE
                        if operation_kind is PlanOperationKind.FILE_CHANGE
                        else PlanPermissionTargetKind.RESOURCE
                    ),
                    value=path,
                )
            )
    request_id = str(message.get("id"))
    binding_value = (
        tool_call_id if isinstance(tool_call_id, str) and tool_call_id.strip() else request_id
    )
    native_binding_ids = [PlanNativeBindingID(name="request_id", value=request_id)]
    if isinstance(tool_call_id, str) and tool_call_id.strip():
        native_binding_ids.append(PlanNativeBindingID(name="tool_call_id", value=tool_call_id))
    interaction = PlanInteraction(
        kind=PlanInteractionKind.PERMISSION,
        question_id=request_id,
        prompt=str(title or params.get("reason") or "Cursor requests permission during planning."),
        options=options,
        session_id=session_id,
        artifact_id=binding_value,
        operation=PlanOperation(
            kind=operation_kind,
            argv=argv,
            shell_expression=shell_expression,
            execution_dir=execution_dir,
            permission_targets=tuple(permission_targets),
            native_binding_ids=tuple(native_binding_ids),
        ),
    )
    if deny_automatically:
        selected = next(
            (
                option.option_id
                for option in options
                if any(word in option.label.lower() for word in ("deny", "reject", "cancel"))
            ),
            None,
        )
    elif handler is None:
        raise PlanInteractionRequiredError(
            "Cursor requires a permission decision to continue planning.",
            interaction=interaction,
            tool_id=tool_id,
            capability=capability,
        )
    else:
        response = handler(interaction)
        if response.outcome is PlanInteractionOutcome.CANCELLED:
            selected = None
        elif response.outcome in {
            PlanInteractionOutcome.DENIED,
            PlanInteractionOutcome.SKIPPED,
        }:
            selected = next(
                (
                    option.option_id
                    for option in options
                    if any(
                        word in option.label.lower()
                        for word in ("deny", "reject", "cancel", "decline")
                    )
                ),
                None,
            )
        else:
            if response.answer is not None:
                raise PlanInteractionRequiredError(
                    "Cursor permission decisions do not accept free text.",
                    interaction=interaction,
                    tool_id=tool_id,
                    capability=capability,
                )
            selected_ids = (
                *((response.option_id,) if response.option_id is not None else ()),
                *response.option_ids,
            )
            if len(selected_ids) > 1:
                raise PlanInteractionRequiredError(
                    "Cursor permissions require one valid native option ID.",
                    interaction=interaction,
                    tool_id=tool_id,
                    capability=capability,
                )
            selected = next(iter(selected_ids), None)
            if selected is None and response.outcome is PlanInteractionOutcome.APPROVED:
                selected = next(
                    (
                        option.option_id
                        for option in options
                        if "allow" in option.label.lower() or "approve" in option.label.lower()
                    ),
                    None,
                )
            valid_ids = {option.option_id for option in options}
            if selected is None or selected not in valid_ids:
                raise PlanInteractionRequiredError(
                    "Cursor permissions require one valid native option ID.",
                    interaction=interaction,
                    tool_id=tool_id,
                    capability=capability,
                )
    if selected is None:
        rpc.respond(message["id"], {"outcome": {"outcome": "cancelled"}})
    else:
        rpc.respond(
            message["id"],
            {"outcome": {"outcome": "selected", "optionId": selected}},
        )
