"""Cursor CLI adapter."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, ClassVar

import structlog

from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.plan_mode import PlanInteractionHandler
from crossby.data import get_models_for_tool
from crossby.handoff.models import ConversationTranscript, SessionRef
from crossby.handoff.readers import cursor as cursor_reader
from crossby.models.ai import (
    AIToolCapabilities,
    AIToolID,
    AIToolType,
    EffortLevel,
    HookOutputDialect,
    HookStopDialect,
    PlanApprovalPolicy,
    PlanArtifactLocation,
    PlanArtifactSource,
    PlanInteraction,
    PlanInteractionKind,
    PlanInteractionOutcome,
    PlanInteractionSupport,
    PlanModeActivation,
    PlanModeCapability,
    PlanQuestionOption,
    PlanRequestBehavior,
    PlanSessionBinding,
    PlanSessionRequest,
    PlanSessionResult,
    PlanSessionTransport,
)

logger = structlog.get_logger()

# Cursor model IDs that already encode an effort level in their name — e.g.
# "claude-opus-4-7-high", "claude-opus-4-7-thinking-xhigh". Appending
# "-thinking" to these would produce invalid IDs.
_EFFORT_LEVEL_SUFFIXES = frozenset({"-low", "-medium", "-high", "-xhigh", "-max"})

# Models that have no "-thinking" variant — appending the suffix produces an invalid ID.
_NO_THINKING_MODELS: frozenset[str] = frozenset({"auto"})

_THINKING_EFFORTS = frozenset({EffortLevel.HIGH, EffortLevel.XHIGH, EffortLevel.MAX})


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
        if separator and key.strip() == "effort":
            efforts.append(EffortLevel(value.strip()))
    if len(efforts) > 1:
        raise ValueError("multiple effort overrides")
    return efforts[0] if efforts else None


def _with_parameterized_effort(model: str, effort: EffortLevel) -> str:
    """Add an exact per-run effort override without changing the model identity."""
    if model.endswith("]") and "[" in model:
        prefix, parameters = model.rsplit("[", 1)
        existing = parameters[:-1].strip()
        separator = "," if existing else ""
        return f"{prefix}[{existing}{separator}effort={effort.value}]"
    return f"{model}[effort={effort.value}]"


class CursorAdapter(AbstractAITool):
    """Adapter for Cursor CLI (``agent`` binary).

    Cursor is an AI-powered IDE with a terminal CLI that supports plan mode,
    model selection, headless execution, and skill discovery.

    Cursor uses its own model ID namespace — e.g. ``sonnet-4.6``, ``opus-4.6``,
    ``gpt-5.3-codex`` — so no format normalization is needed.

    For high/max effort, Cursor uses thinking model variants (e.g.,
    ``sonnet-4.6-thinking``) rather than a separate effort flag.
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
            plan_mode=PlanModeCapability(
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
            ),
            supports_accept_edits=True,
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
        if request.model is None or not request.model.strip() or request.effort is None:
            raise PlanSessionUnsupportedError(
                "Cursor requires an explicit model and effort for collected plan sessions.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        command = ["agent", "--sandbox", "enabled" if request.sandbox else "disabled"]
        effective_model = request.model
        if request.effort is not None:
            try:
                parameterized_effort = _parameterized_effort(effective_model)
            except ValueError as exc:
                raise PlanSessionUnsupportedError(
                    f"Cursor model {effective_model!r} has invalid effort overrides: {exc}.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                ) from exc
            encoded_effort = _encoded_effort(effective_model)
            if (
                parameterized_effort is not None
                and encoded_effort is not None
                and parameterized_effort is not encoded_effort
            ):
                raise PlanSessionUnsupportedError(
                    f"Cursor model {effective_model!r} contains conflicting effort encodings.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                )
            declared_effort = parameterized_effort or encoded_effort
            if declared_effort is not None and declared_effort is not request.effort:
                raise PlanSessionUnsupportedError(
                    f"Cursor model {effective_model!r} encodes effort={declared_effort.value!r}, "
                    f"which conflicts with requested effort={request.effort.value!r} for a "
                    "collected plan session.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                )
            if effective_model.split("[", 1)[0] == "auto":
                raise PlanSessionUnsupportedError(
                    f"Cursor cannot preserve effort={request.effort.value!r} with model='auto' "
                    "because the selected model is not known before launch.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                )
            if declared_effort is None and effective_model.split("[", 1)[
                0
            ] not in get_models_for_tool(AIToolID.CURSOR):
                raise PlanSessionUnsupportedError(
                    f"Cursor cannot verify parameterized effort support for unknown model "
                    f"{request.model!r} in a collected plan session.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                )
            if declared_effort is None:
                effective_model = _with_parameterized_effort(effective_model, request.effort)
        if effective_model:
            command.extend(("--model", effective_model))
        command.append("acp")
        deadline = time.monotonic() + request.timeout_seconds
        rpc: JsonRpcProcess | None = None

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
            rpc = JsonRpcProcess(command, cwd=request.working_dir)
            rpc.request(
                1,
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {},
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
            rpc.request(4, "session/set_mode", {"sessionId": session_id, "modeId": "plan"})
            wait_response(4)
            rpc.request(
                5,
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
                if message.get("id") == 5 and "method" not in message:
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
                    if (
                        outcome.outcome is PlanInteractionOutcome.CANCELLED
                        or outcome.option_id == "cancelled"
                    ):
                        native_outcome: dict[str, str] = {"outcome": "cancelled"}
                    else:
                        native_outcome = {
                            "outcome": "rejected",
                            "reason": outcome.answer or "Plan collected without implementation.",
                        }
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
            return PlanSessionResult(
                tool=self.TOOL_ID,
                version=version,
                plan=plan,
                session_id=session_id,
                native_mode="ACP session/set_mode plan",
                artifact_source=PlanArtifactSource.PROTOCOL_EVENT,
                binding=PlanSessionBinding.SESSION_ID,
                exit_code=0,
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
            if rpc is not None:
                rpc.close()

    def yolo_args(self) -> list[str]:
        """Cursor uses ``--force`` (``--yolo`` is an alias)."""
        return ["--force"]

    def accept_edits_args(self) -> list[str]:
        """No flag needed — the Cursor CLI's default Agent mode *is* accept-edits
        (auto-applies edits, prompts for shell). Declaring
        ``supports_accept_edits=True`` while returning ``[]`` means ``--accept-edits``
        is honored with no warning. (This is the inverse of the Cursor *IDE*
        default, which confirms edits and auto-runs allowlisted commands — the
        CLI and IDE differ.)"""
        return []

    def resolve_effort_model(self, model: str | None, effort: EffortLevel | None) -> str | None:
        """For high/xhigh/max effort, append ``-thinking`` to the model ID.

        Models that already encode effort (e.g. ``-high``, ``-xhigh``) or
        thinking mode (``-thinking``) in their name are returned unchanged.

        The constructed ``<model>-thinking`` ID is validated against the
        bundled Cursor model registry. If the registry has no matching
        entry (e.g. the model has no thinking variant, or is unknown to
        crossby), the original ``model`` is returned unchanged so the
        Cursor CLI receives a valid ID.
        """
        if (
            effort in _THINKING_EFFORTS
            and model
            and model not in _NO_THINKING_MODELS
            and not model.endswith("-thinking")
            and not model.endswith(tuple(_EFFORT_LEVEL_SUFFIXES))
        ):
            candidate = f"{model}-thinking"
            if candidate in get_models_for_tool(AIToolID.CURSOR):
                return candidate
        return model

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
    native_answers: list[dict[str, Any]] = []
    tool_call_id = str(params.get("toolCallId") or message.get("id"))
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
        options = tuple(
            PlanQuestionOption(
                option_id=str(option.get("id")),
                label=str(option.get("label")),
            )
            for option in raw_question.get("options") or []
            if isinstance(option, dict) and option.get("id") and option.get("label")
        )
        interaction = PlanInteraction(
            kind=PlanInteractionKind.QUESTION,
            question_id=question_id,
            prompt=prompt,
            options=options,
            allow_multiple=bool(raw_question.get("allowMultiple")),
            session_id=session_id,
            artifact_id=tool_call_id,
        )
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
        selected_ids = response.option_ids or (
            (response.option_id,) if response.option_id is not None else ()
        )
        if response.answer and not selected_ids:
            selected_ids = tuple(
                option.option_id
                for option in options
                if response.answer in {option.option_id, option.label}
            )
        valid_ids = {option.option_id for option in options}
        if (
            not selected_ids
            or any(option_id not in valid_ids for option_id in selected_ids)
            or (not interaction.allow_multiple and len(selected_ids) != 1)
        ):
            raise PlanInteractionRequiredError(
                "Cursor planning questions require valid native option IDs.",
                interaction=interaction,
                tool_id=tool_id,
                capability=capability,
            )
        native_answers.append({"questionId": question_id, "selectedOptionIds": list(selected_ids)})
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
    from crossby.ai_tools.plan_mode import PlanInteractionRequiredError

    params = _cursor_bound_params(
        message,
        session_id=session_id,
        tool_id=tool_id,
        capability=capability,
        require_session_id=True,
    )
    raw_options = params.get("options") or []
    options = tuple(
        PlanQuestionOption(
            option_id=str(option.get("optionId") or option.get("id") or option.get("name")),
            label=str(option.get("name") or option.get("label") or option.get("optionId")),
        )
        for option in raw_options
        if isinstance(option, dict)
        and (option.get("optionId") or option.get("id") or option.get("name"))
    )
    tool_call = params.get("toolCall")
    tool_call_id = tool_call.get("toolCallId") if isinstance(tool_call, dict) else None
    title = tool_call.get("title") if isinstance(tool_call, dict) else None
    interaction = PlanInteraction(
        kind=PlanInteractionKind.PERMISSION,
        question_id=str(message.get("id")),
        prompt=str(title or params.get("reason") or "Cursor requests permission during planning."),
        options=options,
        session_id=session_id,
        artifact_id=str(tool_call_id or message.get("id")),
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
            selected_ids = response.option_ids or (
                (response.option_id,) if response.option_id is not None else ()
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
