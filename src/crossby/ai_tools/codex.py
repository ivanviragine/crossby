"""OpenAI Codex CLI adapter."""

from __future__ import annotations

import os
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.plan_mode import (
    PlanInteractionHandler,
    PlanTransportError,
    parse_plan_question_options,
    validate_plan_option_selection,
)
from crossby.ai_tools.plan_policy import operation_matches_command_policy
from crossby.handoff.models import ConversationTranscript, SessionRef
from crossby.handoff.readers import codex as codex_reader
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
    PlanCommandPolicy,
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
from crossby.utils.git_worktree import outside_root_git_metadata_dirs

if TYPE_CHECKING:
    from crossby.ai_tools.headless import HeadlessRuntimeContext
    from crossby.ai_tools.interactive import InteractiveLaunchHandler
    from crossby.scenes.launch import SceneLaunchArgs, SceneLaunchContext

# Codex uses "xhigh" for both our XHIGH and MAX levels
_CODEX_EFFORT_MAP: dict[EffortLevel, str] = {
    EffortLevel.LOW: "low",
    EffortLevel.MEDIUM: "medium",
    EffortLevel.HIGH: "high",
    EffortLevel.XHIGH: "xhigh",
    EffortLevel.MAX: "xhigh",
}


def _codex_frame_kind(frame: dict[str, Any]) -> str | None:
    """Describe one ``codex exec --json`` frame without exposing its content."""
    from crossby.ai_tools.headless_cli import non_blank_text

    kind = non_blank_text(frame.get("type"))
    if kind != "item.completed":
        return kind
    item = frame.get("item")
    item_kind = non_blank_text(item.get("type")) if isinstance(item, dict) else None
    return f"item.completed/{item_kind or 'item'}"


class CodexAdapter(AbstractAITool):
    """Adapter for OpenAI Codex CLI."""

    TOOL_ID: ClassVar[AIToolID] = AIToolID.CODEX

    def capabilities(self) -> AIToolCapabilities:
        return AIToolCapabilities(
            tool_id=AIToolID.CODEX,
            display_name="Codex CLI",
            binary="codex",
            tool_type=AIToolType.TERMINAL,
            # `codex update` — updates Codex to the latest version. Like Claude,
            # can update a different install than the one on PATH under a
            # package-manager setup; the version-unchanged warning signals it.
            update_command=("codex", "update"),
            supports_model_flag=True,
            headless_flag="exec",
            supports_headless=True,
            supports_effort=True,
            supports_yolo=True,
            supports_resume=True,
            supports_trusted_dirs=True,
            headless=HeadlessCapability(
                transport=HeadlessNativeTransport.HEADLESS_CLI,
                prompt_transport=HeadlessPromptTransport.STDIN,
                # ``codex exec`` without --json prints a human transcript rather
                # than an isolated final response, so every managed run uses the
                # JSONL event wire and derives TEXT from its agent messages.
                native_outputs=(HeadlessNativeOutput.TEXT, HeadlessNativeOutput.JSONL),
                interaction_modes=(HeadlessInteractionMode.UNATTENDED,),
                supports_response_schema=True,
                successful_native_statuses=("turn.completed",),
                sandbox_behavior=PlanRequestBehavior.PRESERVED,
                approval_behavior=PlanRequestBehavior.PRESERVED,
                command_policy_support=PlanCommandPolicySupport.UNSUPPORTED,
                version_requirement=(
                    "Codex CLI exposing codex exec --json turn events and --output-schema."
                ),
                verified_version="0.154.0",
                remediation=(
                    "Upgrade Codex CLI to 0.154.0 or newer so codex exec emits thread/turn "
                    "JSONL events and accepts a final-response --output-schema file."
                ),
            ),
            plan_mode=PlanModeCapability(
                activation=PlanModeActivation.TERMINAL_INPUT,
                activation_detail=(
                    "Crossby launches the native TUI, submits /plan through its composer, "
                    "waits for the Plan indicator, then emits PLAN_READY before any task input. "
                    "This temporary terminal adapter uses observed UI state, not a native flag."
                ),
                version_requirement=(
                    "Codex CLI 0.154.x on a POSIX interactive terminal. Other versions fail "
                    "closed until their startup UI or a native selector is verified."
                ),
                verified_version="0.154.0",
                collector_verified_version="0.153.4",
                initial_prompt_after_activation=True,
                supports_ready_event=True,
                supported_launch_approval_modes=(
                    PlanLaunchApprovalMode.YOLO,
                    PlanLaunchApprovalMode.ACCEPT_EDITS,
                    PlanLaunchApprovalMode.AUTO,
                ),
                artifact_location=PlanArtifactLocation.SESSION,
                artifact_location_detail=(
                    "Interactive plans remain in the native session; no custom file destination "
                    "is promised. Collected sessions retain exact app-server thread/turn binding."
                ),
                remediation=(
                    "Use an interactive terminal with Codex 0.154.x, or use run_plan_session() "
                    "for structured collection without the terminal startup adapter."
                ),
                collector_activation=PlanModeActivation.CODEX_APP_SERVER,
                transport=PlanSessionTransport.CODEX_APP_SERVER,
                artifact_source=PlanArtifactSource.PROTOCOL_EVENT,
                binding=PlanSessionBinding.THREAD_TURN_IDS,
                interaction=PlanInteractionSupport.CALLBACK,
                sandbox_behavior=PlanRequestBehavior.PRESERVED,
                approval_behavior=PlanRequestBehavior.PRESERVED,
                supported_approval_policies=(
                    PlanApprovalPolicy.ON_REQUEST,
                    PlanApprovalPolicy.NEVER,
                ),
                command_policy_support=PlanCommandPolicySupport.CALLBACK,
                command_policy_detail=(
                    "Matches only authoritative simple-command app-server approval payloads and "
                    "approves each matched operation once."
                ),
            ),
            supports_accept_edits=True,
            supports_stop_hook=True,
            supports_session_start_hook=True,
            supports_user_prompt_submit_hook=True,
            # Both dialects stated explicitly rather than left to the model
            # default, so the capability matrix reads the same in every adapter.
            hook_output_dialect=HookOutputDialect.HOOK_SPECIFIC_OUTPUT,
            hook_stop_dialect=HookStopDialect.BLOCK_DECISION,
            sandboxes_writes=True,
            supports_sandbox_toggle=True,
            supports_network_access=True,
            supports_usage_reporting=True,
            # Session-scoped scenes: Codex takes a named profile that layers a
            # generated ``$CODEX_HOME/<name>.config.toml`` over the base config
            # (requires codex >= 0.134.0 — gated in scene_launch_ready()).
            supports_scene_launch=True,
            scene_profile_flag="--profile",
        )

    def _validate_terminal_plan_request(
        self, initial_message: str | None, *, interactive: bool
    ) -> None:
        from crossby.ai_tools.plan_mode import PlanModeUnsupportedError

        if os.name != "posix" or not interactive:
            raise PlanModeUnsupportedError(
                "Codex terminal Plan startup requires an interactive POSIX terminal; "
                "use run_plan_session() for collected/headless planning.",
                tool_id=self.TOOL_ID,
                capability=self.capabilities().plan_mode,
            )
        if initial_message is not None:
            from crossby.ai_tools.codex_terminal import validate_message

            validate_message(initial_message)

    def _validate_terminal_plan_version(self, version: tuple[int, int, int]) -> None:
        from crossby.ai_tools.plan_mode import PlanModeUnsupportedError

        if version[:2] != (0, 154):
            raise PlanModeUnsupportedError.for_installed_version(
                tool_id=self.TOOL_ID,
                display_name=self.capabilities().display_name,
                capability=self.capabilities().plan_mode,
                installed_version=version,
            )

    def plan_mode_args(self) -> list[str]:
        """The selector is terminal input; keep its observed screen in inline mode."""
        return ["--no-alt-screen"]

    def plan_approval_args(self, mode: PlanLaunchApprovalMode) -> list[str]:
        if mode is PlanLaunchApprovalMode.AUTO:
            # The native --approve-for-me preset also selects workspace-write.
            # Set its approval component explicitly so sandbox remains independent.
            return ["-a", "on-request", "-c", 'approvals_reviewer="auto_review"']
        return super().plan_approval_args(mode)

    def _wrap_terminal_plan_command(self, cmd: list[str], initial_message: str | None) -> list[str]:
        # A bare command from build_launch_command() must activate Plan too.
        wrapper = [sys.executable, "-m", "crossby.ai_tools.codex_terminal"]
        if initial_message is not None:
            wrapper.append("--initial-message=" + initial_message)
        # Native Plan effort has a separate setting from Default-mode effort.
        for argument in tuple(cmd):
            if argument.startswith("model_reasoning_effort="):
                cmd.extend(
                    [
                        "-c",
                        argument.replace(
                            "model_reasoning_effort=", "plan_mode_reasoning_effort=", 1
                        ),
                    ]
                )
        return [*wrapper, "--", *cmd]

    def _run_terminal_plan_command(
        self,
        cmd: list[str],
        working_dir: Path,
        transcript_path: Path | None,
        env: dict[str, str] | None,
        on_event: InteractiveLaunchHandler | None,
    ) -> int:
        from crossby.ai_tools.codex_terminal import (
            TerminalStartupError,
            parse_wrapper_args,
            run_terminal_plan,
        )
        from crossby.ai_tools.plan_mode import PlanModeLaunchError

        prompt, native = parse_wrapper_args(cmd[3:])
        try:
            return run_terminal_plan(
                native,
                working_dir,
                prompt=prompt,
                transcript_path=transcript_path,
                env=env,
                on_event=on_event,
            )
        except TerminalStartupError as exc:
            raise PlanModeLaunchError(
                str(exc),
                tool_id=self.TOOL_ID,
                capability=self.capabilities().plan_mode,
            ) from exc

    def build_resume_command(
        self,
        session_id: str,
        *,
        working_dir: Path | None = None,
        network_access: bool = False,
        sandbox: bool = True,
    ) -> list[str] | None:
        """Resume a Codex session: ``codex resume <session_id>``.

        In a linked worktree, append the sandbox config so git writes work —
        ``--sandbox workspace-write`` + the git-metadata ``writable_roots`` +
        the network pin — but **approval-neutral**: resume deliberately skips
        autonomy resolution, so no ``-a`` flag is injected and the session's
        existing approval policy is preserved (forcing ``-a never`` would disable
        approval prompts for a user who never requested YOLO). A non-worktree
        resume with no ``--network`` composes nothing extra, so it stays
        byte-identical to ``["codex", "resume", <id>]``.
        """
        return [
            "codex",
            "resume",
            session_id,
            *self.sandbox_config_args(
                autonomy_args=[],
                trusted_dirs=None,
                working_dir=working_dir,
                network_access=network_access,
                sandbox=sandbox,
            ),
        ]

    def locate_sessions(self, project_path: Path) -> list[SessionRef]:
        return codex_reader.locate_sessions(project_path)

    def read_session(self, ref: SessionRef) -> ConversationTranscript:
        return codex_reader.read_session(ref)

    def initial_message_args(self, prompt: str) -> list[str]:
        """Codex accepts the initial message as a positional argument."""
        return [prompt]

    def headless_prompt_stdin_args(self) -> list[str] | None:
        """``codex exec`` reads instructions from stdin when no positional prompt
        is passed (a piped stdin is otherwise appended as a ``<stdin>`` block)."""
        return ["exec"]

    def _headless_command(
        self,
        request: HeadlessSessionRequest,
        *,
        schema_path: Path | None,
    ) -> list[str]:
        """Build the exact unattended ``codex exec`` invocation.

        ``codex exec`` has no interactive approval channel, so the requested
        sandbox posture is the whole policy: a command the sandbox refuses fails
        the turn natively instead of waiting for an approval nobody can give.
        """
        command = [
            "codex",
            "exec",
            "--json",
            "--color",
            "never",
            "--sandbox",
            "workspace-write" if request.sandbox else "danger-full-access",
        ]
        for path in request.trusted_dirs:
            command.extend(self.plan_dir_args(str(path)))
        for meta_dir in outside_root_git_metadata_dirs(request.working_dir):
            command.extend(self.plan_dir_args(str(meta_dir)))
        if request.sandbox:
            # Pin the flag both ways so an ambient config value can never
            # silently widen networking inside a crossby-managed sandbox.
            enabled = "true" if request.network_access else "false"
            command.extend(("-c", f"sandbox_workspace_write.network_access={enabled}"))
        if request.model:
            command.extend(("-m", request.model))
        if request.effort is not None:
            command.extend(self.effort_args(request.effort))
        if schema_path is not None:
            command.extend(("--output-schema", str(schema_path)))
        # A bare ``-`` keeps the prompt on stdin instead of in argv.
        command.append("-")
        return command

    def _run_headless_session(
        self,
        request: HeadlessSessionRequest,
        version: str,
        context: HeadlessRuntimeContext,
    ) -> HeadlessSessionResult:
        """Run one unattended ``codex exec`` turn and normalize its JSONL events."""
        import json
        import shutil
        import tempfile

        from crossby.ai_tools.headless_cli import (
            MISSING,
            capture_failure_warnings,
            complete_session,
            frame_streamer,
            non_blank_text,
            parse_json_lines,
            run_managed_command,
            usage_from,
        )
        from crossby.ai_tools.plan_process import child_environment

        schema_root: str | None = None
        schema_path: Path | None = None
        if request.response_schema is not None:
            # --output-schema takes a file path, never inline JSON. Keep it out
            # of the workspace so a managed run never writes tracked files.
            schema_root = tempfile.mkdtemp(prefix="crossby-codex-schema-")
            schema_path = Path(schema_root) / "response-schema.json"
            schema_path.write_text(
                json.dumps(request.response_schema, separators=(",", ":")),
                encoding="utf-8",
            )
        try:
            output = run_managed_command(
                context,
                argv=self._headless_command(request, schema_path=schema_path),
                cwd=request.working_dir,
                env=child_environment({"NO_COLOR": "1"}),
                stdin_text=request.prompt,
                on_stdout_lines=frame_streamer(
                    context,
                    label="codex",
                    kind_of=_codex_frame_kind,
                    provenance_of=lambda frame: {
                        "thread_id": non_blank_text(frame.get("thread_id"))
                    },
                ),
            )
        finally:
            if schema_root is not None:
                shutil.rmtree(schema_root, ignore_errors=True)

        warnings = capture_failure_warnings(output)
        if output.overflowed or output.undecodable:
            return context.complete(
                HeadlessTerminalStatus.INVALID_OUTPUT,
                exit_code=output.returncode,
                warnings=warnings,
            )
        frames = parse_json_lines(output.stdout)
        if frames is None:
            return context.complete(
                HeadlessTerminalStatus.INVALID_OUTPUT,
                exit_code=output.returncode,
                warnings=(*warnings, "Codex CLI emitted malformed JSONL events."),
            )

        # Progress and thread provenance were already emitted live by the streamer.
        thread_id: str | None = None
        agent_items: list[dict[str, Any]] = []
        terminal: str | None = None
        failure: str | None = None
        native_errors = 0
        last_error: str | None = None
        usage_values: Any = None
        for frame in frames:
            kind = non_blank_text(frame.get("type"))
            if kind is None:
                continue
            if kind == "thread.started":
                thread_id = thread_id or non_blank_text(frame.get("thread_id"))
            elif kind == "item.completed":
                item = frame.get("item")
                if isinstance(item, dict) and non_blank_text(item.get("type")) == "agent_message":
                    agent_items.append(item)
            elif kind == "turn.completed":
                terminal = kind
                usage_values = frame.get("usage")
            elif kind == "turn.failed":
                terminal = kind
                error = frame.get("error")
                failure = non_blank_text(error.get("message")) if isinstance(error, dict) else None
            elif kind == "error":
                native_errors += 1
                last_error = non_blank_text(frame.get("message")) or last_error

        if native_errors:
            # Codex retries transport errors, so keep one bounded summary rather
            # than one warning per retry frame.
            warnings = (
                *warnings,
                f"Codex CLI reported {native_errors} native error event(s); "
                f"last: {last_error or 'no detail'}",
            )
        if terminal is None:
            return context.complete(
                HeadlessTerminalStatus.INVALID_OUTPUT,
                exit_code=output.returncode,
                thread_id=thread_id,
                warnings=(*warnings, "Codex CLI ended without a terminal turn event."),
            )
        if terminal == "turn.failed":
            warnings = (*warnings, f"Codex CLI turn failed: {failure or 'no detail'}")

        texts = [
            text
            for item in agent_items
            if (text := non_blank_text(item.get("text")) or non_blank_text(item.get("message")))
        ]
        if len(agent_items) > 1 and request.native_output is HeadlessNativeOutput.JSONL:
            warnings = (
                *warnings,
                "Only the final native agent message is returned; request text output for the "
                "complete response.",
            )
        structured: Any = MISSING
        if request.response_schema is not None and texts:
            try:
                structured = json.loads(texts[-1])
            except json.JSONDecodeError:
                warnings = (
                    *warnings,
                    "Codex CLI returned a final message that was not valid schema JSON.",
                )
        return complete_session(
            context,
            request,
            status=(
                HeadlessTerminalStatus.SUCCEEDED
                if terminal == "turn.completed" and output.returncode == 0
                else HeadlessTerminalStatus.FAILED
            ),
            exit_code=output.returncode,
            response_text="\n".join(texts) or None,
            native_object=agent_items[-1] if agent_items else MISSING,
            structured_output=structured,
            native_status=terminal,
            thread_id=thread_id,
            usage=usage_from(
                usage_values,
                input_tokens="input_tokens",
                output_tokens="output_tokens",
                cached="cached_input_tokens",
            ),
            warnings=warnings,
        )

    def _run_plan_session(
        self,
        request: PlanSessionRequest,
        version: str,
        interaction_handler: PlanInteractionHandler | None,
    ) -> PlanSessionResult:
        """Collect the authoritative completed plan item from Codex app-server."""
        from crossby.ai_tools.plan_mode import (
            PlanArtifactAmbiguousError,
            PlanArtifactMalformedError,
            PlanArtifactMissingError,
            PlanBindingMismatchError,
            PlanSessionError,
            PlanSessionUnsupportedError,
            PlanTransportError,
        )
        from crossby.ai_tools.plan_process import HeaderlessJsonRpcProcess

        capability = self.capabilities().plan_mode
        deadline = time.monotonic() + request.timeout_seconds
        rpc: HeaderlessJsonRpcProcess | None = None
        rpc_closed = False

        def remaining() -> float:
            wait = deadline - time.monotonic()
            if wait <= 0:
                raise TimeoutError("Codex plan session exceeded its timeout")
            return wait

        def wait_response(
            request_id: int,
            *,
            notification_handler: Callable[[dict[str, Any]], None] | None = None,
        ) -> dict[str, Any]:
            assert rpc is not None
            while True:
                message = rpc.read(timeout=remaining())
                if message.get("id") != request_id:
                    if "method" in message and "id" in message:
                        raise PlanTransportError(
                            f"Codex requested {message.get('method')!r} before the plan turn "
                            "was established.",
                            tool_id=self.TOOL_ID,
                            capability=capability,
                        )
                    if "id" not in message and notification_handler is not None:
                        notification_handler(message)
                    continue
                if "error" in message:
                    raise PlanTransportError(
                        f"Codex app-server rejected {_rpc_method_name(request_id)}.",
                        tool_id=self.TOOL_ID,
                        capability=capability,
                        stderr=rpc.stderr,
                    )
                result = message.get("result")
                if not isinstance(result, dict):
                    raise PlanTransportError(
                        "Codex app-server returned a malformed response.",
                        tool_id=self.TOOL_ID,
                        capability=capability,
                    )
                return result

        try:
            rpc = HeaderlessJsonRpcProcess(
                ["codex", "app-server", "--stdio"],
                cwd=request.working_dir,
                timeout=remaining(),
            )
            rpc.request(
                1,
                "initialize",
                {
                    "clientInfo": {
                        "name": "crossby",
                        "title": "Crossby",
                        "version": "plan-session-v1",
                    },
                    "capabilities": {"experimentalApi": True},
                },
            )
            wait_response(1)
            rpc.notify("initialized", {})

            rpc.request(2, "collaborationMode/list", {})
            modes = wait_response(2).get("data")
            if not isinstance(modes, list) or not any(
                isinstance(mode, dict) and mode.get("mode") == "plan" for mode in modes
            ):
                raise PlanSessionUnsupportedError(
                    "Codex app-server did not advertise the native plan collaboration mode.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                )

            config: dict[str, Any] = {}
            if request.sandbox:
                sandbox_config: dict[str, Any] = {
                    # Pin both values so ambient config cannot silently widen a
                    # Crossby-managed planning thread.
                    "network_access": request.network_access,
                    "writable_roots": [str(path) for path in request.trusted_dirs],
                }
                config["sandbox_workspace_write"] = sandbox_config
            thread_params: dict[str, Any] = {
                "cwd": str(request.working_dir.resolve()),
                "model": request.model,
                "sandbox": "workspace-write" if request.sandbox else "danger-full-access",
                "approvalPolicy": request.approval_policy.value,
                "ephemeral": True,
            }
            if config:
                thread_params["config"] = config
            rpc.request(3, "thread/start", thread_params)
            thread_result = wait_response(3)
            thread = thread_result.get("thread")
            thread_id = thread.get("id") if isinstance(thread, dict) else None
            if not isinstance(thread_id, str) or not thread_id.strip():
                raise PlanTransportError(
                    "Codex thread/start response omitted the thread ID.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                )
            effective_model = thread_result.get("model") or request.model or ""
            rpc.request(
                4,
                "turn/start",
                {
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": request.prompt}],
                    "collaborationMode": {
                        "mode": "plan",
                        "settings": {
                            "model": effective_model,
                            "reasoning_effort": (
                                _CODEX_EFFORT_MAP.get(request.effort, request.effort.value)
                                if request.effort is not None
                                else None
                            ),
                            "developer_instructions": None,
                        },
                    },
                },
            )
            turn_result = wait_response(4)
            turn = turn_result.get("turn")
            turn_id = turn.get("id") if isinstance(turn, dict) else None
            if not isinstance(turn_id, str) or not turn_id.strip():
                raise PlanTransportError(
                    "Codex turn/start response omitted the turn ID.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=thread_id,
                    thread_id=thread_id,
                )

            completed_plan: tuple[str, str] | None = None

            def capture_completed_plan(message: dict[str, Any]) -> None:
                nonlocal completed_plan
                method = message.get("method")
                params = message.get("params")
                if method == "item/completed" and isinstance(params, dict):
                    item = params.get("item")
                    if not isinstance(item, dict) or item.get("type") != "plan":
                        return
                    if params.get("threadId") != thread_id or params.get("turnId") != turn_id:
                        raise PlanBindingMismatchError(
                            "Codex emitted a plan item for a different thread or turn.",
                            tool_id=self.TOOL_ID,
                            capability=capability,
                            session_id=thread_id,
                            thread_id=str(params.get("threadId") or "") or None,
                            turn_id=str(params.get("turnId") or "") or None,
                            artifact_id=str(item.get("id") or "") or None,
                        )
                    artifact_id = item.get("id")
                    text = item.get("text")
                    if (
                        not isinstance(artifact_id, str)
                        or not artifact_id.strip()
                        or not isinstance(text, str)
                    ):
                        raise PlanArtifactMalformedError(
                            "Codex completed plan item omitted its ID or text.",
                            tool_id=self.TOOL_ID,
                            capability=capability,
                            session_id=thread_id,
                            thread_id=thread_id,
                            turn_id=turn_id,
                        )
                    if completed_plan is not None:
                        raise PlanArtifactAmbiguousError(
                            "Codex emitted multiple authoritative completed plan items for the "
                            "bound turn.",
                            tool_id=self.TOOL_ID,
                            capability=capability,
                            session_id=thread_id,
                            thread_id=thread_id,
                            turn_id=turn_id,
                            artifact_id=artifact_id,
                        )
                    completed_plan = (artifact_id, text)

            while True:
                message = rpc.read(timeout=remaining())
                if message.get("method") == "item/completed":
                    capture_completed_plan(message)
                    continue
                method = message.get("method")
                params = message.get("params")
                if method == "turn/completed" and isinstance(params, dict):
                    completed_thread = params.get("threadId")
                    completed_turn = params.get("turn")
                    if not isinstance(completed_turn, dict):
                        raise PlanArtifactMalformedError(
                            "Codex turn/completed omitted the completed turn object.",
                            tool_id=self.TOOL_ID,
                            capability=capability,
                            session_id=thread_id,
                            thread_id=thread_id,
                            turn_id=turn_id,
                        )
                    completed_turn_id = completed_turn.get("id")
                    if completed_thread != thread_id or completed_turn_id != turn_id:
                        raise PlanBindingMismatchError(
                            "Codex completed a different thread or turn than the launched one.",
                            tool_id=self.TOOL_ID,
                            capability=capability,
                            session_id=thread_id,
                            thread_id=str(completed_thread or "") or None,
                            turn_id=str(completed_turn_id or "") or None,
                        )
                    status = completed_turn.get("status")
                    if status != "completed":
                        raise PlanTransportError(
                            f"Codex plan turn ended with status {status!r}.",
                            tool_id=self.TOOL_ID,
                            capability=capability,
                            session_id=thread_id,
                            thread_id=thread_id,
                            turn_id=turn_id,
                            stderr=rpc.stderr,
                        )
                    break
                if method == "item/tool/requestUserInput" and "id" in message:
                    _answer_codex_questions(
                        rpc,
                        message,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        handler=interaction_handler,
                        tool_id=self.TOOL_ID,
                        capability=capability,
                    )
                    continue
                if (
                    method
                    in {
                        "item/commandExecution/requestApproval",
                        "item/fileChange/requestApproval",
                    }
                    and "id" in message
                ):
                    _answer_codex_approval(
                        rpc,
                        message,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        handler=interaction_handler,
                        tool_id=self.TOOL_ID,
                        capability=capability,
                        deny_automatically=request.approval_policy.value == "never",
                        command_policy=request.command_policy,
                        allowed_execution_roots=(
                            request.working_dir,
                            *request.trusted_dirs,
                        ),
                    )
                    continue
                if isinstance(method, str) and "id" in message:
                    raise PlanTransportError(
                        f"Codex app-server requested unsupported method {method!r}.",
                        tool_id=self.TOOL_ID,
                        capability=capability,
                        session_id=thread_id,
                        thread_id=thread_id,
                        turn_id=turn_id,
                    )

            rpc.request(
                5,
                "thread/backgroundTerminals/clean",
                {"threadId": thread_id},
            )
            wait_response(5, notification_handler=capture_completed_plan)
            if completed_plan is None:
                raise PlanArtifactMissingError(
                    "Codex turn completed without an authoritative completed plan item.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=thread_id,
                    thread_id=thread_id,
                    turn_id=turn_id,
                )
            artifact_id, plan = completed_plan
            if not plan.strip():
                raise PlanArtifactMalformedError(
                    "Codex completed plan item was blank.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=thread_id,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    artifact_id=artifact_id,
                )
            exit_code = rpc.close()
            rpc_closed = True
            if exit_code != 0:
                raise PlanTransportError(
                    f"Codex app-server exited with status {exit_code} after completing the turn.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    exit_code=exit_code,
                    session_id=thread_id,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    stderr=rpc.stderr,
                )
            return PlanSessionResult(
                tool=self.TOOL_ID,
                version=version,
                plan=plan,
                session_id=thread_id,
                native_mode="collaborationMode.mode=plan",
                artifact_source=PlanArtifactSource.PROTOCOL_EVENT,
                binding=PlanSessionBinding.THREAD_TURN_IDS,
                exit_code=exit_code,
                thread_id=thread_id,
                turn_id=turn_id,
                artifact_id=artifact_id,
            )
        except PlanSessionError:
            raise
        except (OSError, ValueError, EOFError, TimeoutError) as exc:
            raise PlanTransportError(
                f"Codex app-server plan session failed: {exc}",
                tool_id=self.TOOL_ID,
                capability=capability,
                stderr=rpc.stderr if rpc is not None else None,
            ) from exc
        finally:
            if rpc is not None and not rpc_closed:
                rpc.close()

    def plan_dir_args(self, plan_dir: str) -> list[str]:
        """Codex uses --add-dir for plan directory access."""
        return ["--add-dir", plan_dir]

    def sandbox_config_args(
        self,
        *,
        autonomy_args: list[str],
        trusted_dirs: list[str] | None,
        working_dir: Path | None,
        network_access: bool,
        sandbox: bool = True,
    ) -> list[str]:
        """Single owner of Codex's sandbox / writable-root / network argv.

        Reached from :meth:`build_launch_command` (the launch hook) and, with
        ``autonomy_args=[]``, from :meth:`build_resume_command`.

        Emits, **in order**, a single ``--sandbox workspace-write`` before any
        ``--add-dir``, then one ``--add-dir`` per trusted dir and per
        linked-worktree git-metadata dir (so sandboxed git writes reach the
        external gitdir), and an explicit ``network_access`` pin — but only when
        crossby actually **forces** workspace-write. When nothing forces it,
        returns ``[]`` so the launch/resume stays byte-identical to an unmanaged
        Codex run.

        The metadata dirs go through ``--add-dir`` (which *adds* to the sandbox's
        writable roots) rather than ``-c sandbox_workspace_write.writable_roots``
        (which would *replace* any roots the user configured). The network pin
        is deliberately the replacing form — see below.

        workspace-write is forced by any of: accept-edits/auto (detected as
        ``-a on-request`` in ``autonomy_args``), one or more ``trusted_dirs``,
        out-of-root worktree metadata, or ``network_access``. YOLO alone
        (``-a never``) does **not** force it. This never emits an approval flag
        itself: on launch the approval flag already sits in ``autonomy_args``;
        on resume it is intentionally absent (approval-neutral). Treating
        ``--sandbox``/``-s`` as one setting and owning the ordering here is what
        guarantees the mode is emitted exactly once.

        The network pin defends against an ambient ``network_access=true`` in the
        user's config: whenever crossby forces workspace-write it explicitly sets
        the flag (``true`` only with ``--network``, else ``false``), so ambient
        config can never silently enable networking in a crossby-managed sandbox.

        When ``sandbox`` is false, this returns only
        ``--sandbox danger-full-access`` before inspecting worktree metadata or
        other sandbox context. Approval policy remains owned by the autonomy
        composer and is therefore unchanged.
        """
        if not sandbox:
            return ["--sandbox", "danger-full-access"]

        trusted = list(trusted_dirs or [])
        metadata = outside_root_git_metadata_dirs(working_dir) if working_dir is not None else []
        accept_edits = "on-request" in autonomy_args

        if not (accept_edits or trusted or metadata or network_access):
            return []

        args: list[str] = ["--sandbox", "workspace-write"]
        for d in trusted:
            args.extend(self.plan_dir_args(d))
        for meta_dir in metadata:
            args.extend(self.plan_dir_args(str(meta_dir)))
        args.extend(
            [
                "-c",
                f"sandbox_workspace_write.network_access={'true' if network_access else 'false'}",
            ]
        )
        return args

    def is_model_compatible(self, model: str) -> bool:
        """Codex accepts codex-*, gpt-*, and o<digit>* model IDs."""
        lower = model.lower()
        if lower.startswith("codex-") or lower.startswith("gpt-"):
            return True
        # o1, o3, o4-mini etc.
        return bool(re.match(r"^o\d", lower))

    def effort_args(self, effort: EffortLevel) -> list[str]:
        """Codex uses ``-c model_reasoning_effort="<mapped>"``."""
        mapped = _CODEX_EFFORT_MAP.get(effort, effort.value)
        return ["-c", f'model_reasoning_effort="{mapped}"']

    def accept_edits_args(self) -> list[str]:
        """Codex accept-edits: the workspace-write + ``-a on-request`` posture —
        the approval half only (``-a on-request``).

        Codex CLI 0.152 removed ``untrusted`` from ``--ask-for-approval`` (it
        now accepts only ``on-request``/``never``), so the old ``-a untrusted``
        fails argument parsing before a headless agent can run. ``on-request``
        is the closest surviving policy: it is Codex's native "Auto" posture,
        where the model runs edits *and* commands freely inside the
        workspace-write sandbox and escalates to a human only before anything
        that would escape it (network access, writes outside the workspace).

        This is deliberately **not** a byte-for-byte match of the old
        ``untrusted`` policy, which prompted for essentially every command:
        0.152 removed per-command prompting entirely, so no surviving policy
        reproduces it. The workspace-write sandbox (owned by
        :meth:`sandbox_config_args`) — not a per-command prompt — is therefore
        the enforced safety boundary; see the README's "Codex sandbox" section.
        ``--approve-for-me`` is deliberately avoided as the more-autonomous
        option: it routes even those out-of-sandbox escalations through
        automatic review with no human prompt at all. ``on-request`` also
        predates 0.152, so this path needs no version gate.

        The workspace-write sandbox that accept-edits needs is emitted by
        :meth:`sandbox_config_args`, which owns sandbox-mode selection so the
        ``--sandbox workspace-write`` flag is passed exactly once (before any
        ``--add-dir``) even when trusted dirs or worktree metadata are also
        present. The old ``--approval-mode auto-edit`` flag was removed in the
        Rust CLI (v0.14x) and must not be used.
        """
        return ["-a", "on-request"]

    def yolo_args(self) -> list[str]:
        """Codex skips approval prompts with ``-a never`` while keeping its
        sandbox intact.

        ``--yolo`` (an alias for ``--dangerously-bypass-approvals-and-sandbox``)
        is deliberately avoided: it would also disable the OS sandbox
        (Seatbelt/Landlock), making Codex's yolo mode far more permissive than
        the approval-only yolo of every other adapter. Yolo here means "skip
        approval prompts", not "remove the sandbox".
        """
        return ["-a", "never"]

    def scene_launch_ready(self) -> bool:
        """Codex ``--profile`` scenes need ``codex >= 0.134.0``.

        The legacy in-config ``[profiles.<name>]`` tables were removed in that
        release; only from it on does ``--profile <name>`` layer
        ``$CODEX_HOME/<name>.config.toml`` over the base config. On an older or
        unknown build this returns False so the launch path falls back to
        persistent activation rather than emitting a ``--profile`` the CLI would
        ignore.
        """
        if not self.capabilities().supports_scene_launch:
            return False
        from crossby.scenes import versioning
        from crossby.scenes.launch import CODEX_PROFILE_MIN

        version = versioning.detect_tool_version(AIToolID.CODEX)
        return versioning.at_least(version, CODEX_PROFILE_MIN)

    def scene_launch_concerns(self) -> set[str]:
        """Codex scopes only MCP at launch (via the layered profile)."""
        return {"mcp"}

    def scene_launch_args(self, scene: SceneLaunchContext) -> SceneLaunchArgs:
        """Compile the scene into a namespaced ``$CODEX_HOME`` profile.

        The scene's deselected MCP servers become ``[mcp_servers.<id>] enabled =
        false`` in ``$CODEX_HOME/crossby-<project-slug>-<scene>.config.toml``,
        and ``--profile crossby-<project-slug>-<scene>`` layers it over the base
        config for the session. Emitted only when the scene narrows MCP (the one
        Codex session lever); skills/agents/hooks/permissions have no Codex
        launch flag and are left to persistent ``scene use``.

        The profile lives under ``$CODEX_HOME`` — the documented exception to the
        "everything under ``.crossby/scene/``" rule — because ``--profile`` reads
        nowhere else. It is namespaced by a project-root hash and carries a
        generated-by header so pruning never deletes a hand-written profile.

        If that namespaced path is already occupied by a hand-written profile,
        signal the top-level launch orchestrator before any child starts.  It
        preserves the profile, runs the shared recoverable persistent lifecycle,
        and retries the launch once without ``--profile``.
        """
        from crossby.scenes.launch import (
            SceneLaunchArgs,
            SceneLaunchFallbackError,
            codex_profile_name,
            write_codex_profile,
        )

        if not scene.narrows_mcp():
            return SceneLaunchArgs()

        try:
            write_codex_profile(scene.project_root, scene.name, scene.deselected_mcp())
        except FileExistsError as exc:
            raise SceneLaunchFallbackError(str(exc)) from exc

        flag = self.capabilities().scene_profile_flag or "--profile"
        return SceneLaunchArgs(args=(flag, codex_profile_name(scene.project_root, scene.name)))


def _rpc_method_name(request_id: int) -> str:
    return {
        1: "initialize",
        2: "collaborationMode/list",
        3: "thread/start",
        4: "turn/start",
        5: "thread/backgroundTerminals/clean",
    }.get(request_id, f"request {request_id}")


def _require_codex_binding(
    params: dict[str, Any],
    *,
    thread_id: str,
    turn_id: str,
    tool_id: AIToolID,
    capability: PlanModeCapability,
) -> None:
    from crossby.ai_tools.plan_mode import PlanBindingMismatchError

    if params.get("threadId") != thread_id or params.get("turnId") != turn_id:
        raise PlanBindingMismatchError(
            "Codex interaction belonged to a different thread or turn.",
            tool_id=tool_id,
            capability=capability,
            session_id=thread_id,
            thread_id=str(params.get("threadId") or "") or None,
            turn_id=str(params.get("turnId") or "") or None,
        )


def _answer_codex_questions(
    rpc: Any,
    message: dict[str, Any],
    *,
    thread_id: str,
    turn_id: str,
    handler: PlanInteractionHandler | None,
    tool_id: AIToolID,
    capability: PlanModeCapability,
) -> None:
    from crossby.ai_tools.plan_mode import PlanInteractionRequiredError, PlanTransportError

    params = message.get("params")
    if not isinstance(params, dict):
        raise PlanTransportError(
            "Codex request-user-input message had malformed params.",
            tool_id=tool_id,
            capability=capability,
        )
    _require_codex_binding(
        params,
        thread_id=thread_id,
        turn_id=turn_id,
        tool_id=tool_id,
        capability=capability,
    )
    questions = params.get("questions")
    if not isinstance(questions, list) or not questions:
        raise PlanTransportError(
            "Codex request-user-input message contained no questions.",
            tool_id=tool_id,
            capability=capability,
            session_id=thread_id,
            thread_id=thread_id,
            turn_id=turn_id,
        )
    interactions: list[PlanInteraction] = []
    question_ids: set[str] = set()
    for raw_question in questions:
        if not isinstance(raw_question, dict):
            raise PlanTransportError(
                "Codex emitted a malformed planning question.",
                tool_id=tool_id,
                capability=capability,
                session_id=thread_id,
                thread_id=thread_id,
                turn_id=turn_id,
            )
        question_id = raw_question.get("id")
        prompt = raw_question.get("question")
        if not isinstance(question_id, str) or not isinstance(prompt, str):
            raise PlanTransportError(
                "Codex planning question omitted its ID or prompt.",
                tool_id=tool_id,
                capability=capability,
                session_id=thread_id,
                thread_id=thread_id,
                turn_id=turn_id,
            )
        if not question_id.strip() or not prompt.strip():
            raise PlanTransportError(
                "Codex planning question contained a blank ID or prompt.",
                tool_id=tool_id,
                capability=capability,
                session_id=thread_id,
                thread_id=thread_id,
                turn_id=turn_id,
            )
        if question_id in question_ids:
            raise PlanTransportError(
                "Codex planning questions contained a duplicate native ID.",
                tool_id=tool_id,
                capability=capability,
                session_id=thread_id,
                thread_id=thread_id,
                turn_id=turn_id,
            )
        question_ids.add(question_id)
        multiple_values: list[bool] = []
        for field in ("allowMultiple", "isMultiple"):
            if field not in raw_question:
                continue
            value = raw_question[field]
            if not isinstance(value, bool):
                raise PlanTransportError(
                    "Codex planning question had a non-boolean multi-select field.",
                    tool_id=tool_id,
                    capability=capability,
                    session_id=thread_id,
                    thread_id=thread_id,
                    turn_id=turn_id,
                )
            multiple_values.append(value)
        if len(set(multiple_values)) > 1:
            raise PlanTransportError(
                "Codex planning question had conflicting multi-select fields.",
                tool_id=tool_id,
                capability=capability,
                session_id=thread_id,
                thread_id=thread_id,
                turn_id=turn_id,
            )
        allow_other = raw_question.get("isOther", False)
        if not isinstance(allow_other, bool):
            raise PlanTransportError(
                "Codex planning question had a non-boolean free-form field.",
                tool_id=tool_id,
                capability=capability,
                session_id=thread_id,
                thread_id=thread_id,
                turn_id=turn_id,
            )
        options = parse_plan_question_options(raw_question.get("options"))
        interactions.append(
            PlanInteraction(
                kind=PlanInteractionKind.QUESTION,
                question_id=question_id,
                prompt=prompt,
                options=options,
                allow_multiple=multiple_values[0] if multiple_values else False,
                allow_other=allow_other,
                session_id=thread_id,
                thread_id=thread_id,
                turn_id=turn_id,
                artifact_id=str(params.get("itemId") or "") or None,
            )
        )

    answers: dict[str, dict[str, list[str]]] = {}
    for interaction in interactions:
        if handler is None:
            raise PlanInteractionRequiredError(
                "Codex requires an answer to continue the native planning turn.",
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
            raise PlanInteractionRequiredError(
                "Codex planning question was left unanswered.",
                interaction=interaction,
                tool_id=tool_id,
                capability=capability,
            )
        if response.answer and not interaction.allow_other:
            raise PlanInteractionRequiredError(
                "Codex planning question does not allow a free-form answer.",
                interaction=interaction,
                tool_id=tool_id,
                capability=capability,
            )
        try:
            selected_ids = validate_plan_option_selection(interaction, response)
        except ValueError as exc:
            raise PlanInteractionRequiredError(
                "Codex planning questions require valid native option IDs.",
                interaction=interaction,
                tool_id=tool_id,
                capability=capability,
            ) from exc
        if response.answer and selected_ids and not interaction.allow_multiple:
            raise PlanInteractionRequiredError(
                "Codex single-select planning questions require exactly one answer.",
                interaction=interaction,
                tool_id=tool_id,
                capability=capability,
            )
        answer_values = [response.answer] if response.answer else []
        for option_id in selected_ids:
            selected = next(
                option for option in interaction.options if option.option_id == option_id
            )
            answer_values.append(selected.label)
        if not answer_values:
            raise PlanInteractionRequiredError(
                "Codex planning question was left unanswered.",
                interaction=interaction,
                tool_id=tool_id,
                capability=capability,
            )
        answers[interaction.question_id] = {"answers": answer_values}
    rpc.respond(message["id"], {"answers": answers})


def _codex_permission_targets(params: dict[str, Any]) -> tuple[PlanPermissionTarget, ...]:
    """Extract only documented additional resources from an approval payload."""
    targets: list[PlanPermissionTarget] = []

    def unknown(value: str) -> None:
        targets.append(PlanPermissionTarget(kind=PlanPermissionTargetKind.RESOURCE, value=value))

    network_context = params.get("networkApprovalContext")
    if isinstance(network_context, dict):
        host = network_context.get("host")
        if isinstance(host, str) and host.strip():
            targets.append(
                PlanPermissionTarget(kind=PlanPermissionTargetKind.NETWORK_HOST, value=host)
            )
        else:
            unknown("unrecognized native network approval target")
    elif network_context is not None:
        unknown("unrecognized native network approval context")

    additional = params.get("additionalPermissions")
    if additional is not None and not isinstance(additional, dict):
        unknown("unrecognized native additional permissions")
        return tuple(targets)
    if isinstance(additional, dict) and (
        set(additional) - {"fileSystem", "network"}
        or not {
            "fileSystem",
            "network",
        }.issubset(additional)
    ):
        unknown("unrecognized native additional permission fields")
    filesystem = additional.get("fileSystem") if isinstance(additional, dict) else None
    if isinstance(filesystem, dict):
        if set(filesystem) - {"read", "write", "entries", "globScanMaxDepth"}:
            unknown("unrecognized native filesystem permission fields")
        for field, kind in (
            ("read", PlanPermissionTargetKind.FILESYSTEM_READ),
            ("write", PlanPermissionTargetKind.FILESYSTEM_WRITE),
        ):
            values = filesystem.get(field)
            if isinstance(values, list):
                for path_value in values:
                    if isinstance(path_value, str) and path_value.strip():
                        targets.append(PlanPermissionTarget(kind=kind, value=path_value))
                    else:
                        unknown("unrecognized native filesystem permission target")
            elif values is not None:
                unknown("unrecognized native filesystem permission list")
        entries = filesystem.get("entries")
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    unknown("unrecognized native filesystem permission entry")
                    continue
                access = entry.get("access")
                resource = entry.get("path")
                target_value: str | None = None
                if isinstance(resource, dict):
                    target_value = next(
                        (
                            resource[field]
                            for field in ("path", "pattern")
                            if isinstance(resource.get(field), str) and resource[field].strip()
                        ),
                        None,
                    )
                    special = resource.get("value")
                    if target_value is None and isinstance(special, dict):
                        special_kind = special.get("kind")
                        target_value = (
                            f"special:{special_kind}"
                            if isinstance(special_kind, str) and special_kind.strip()
                            else "unrecognized native special filesystem target"
                        )
                if target_value is None:
                    target_value = "unrecognized native filesystem target"
                target_kind = {
                    "read": PlanPermissionTargetKind.FILESYSTEM_READ,
                    "write": PlanPermissionTargetKind.FILESYSTEM_WRITE,
                    "deny": PlanPermissionTargetKind.FILESYSTEM_DENY,
                }.get(
                    access if isinstance(access, str) else "",
                    PlanPermissionTargetKind.RESOURCE,
                )
                targets.append(PlanPermissionTarget(kind=target_kind, value=target_value))
        elif entries is not None:
            unknown("unrecognized native filesystem permission entries")
    elif filesystem is not None:
        unknown("unrecognized native filesystem permissions")
    network = additional.get("network") if isinstance(additional, dict) else None
    if isinstance(network, dict) and network.get("enabled") is True:
        targets.append(
            PlanPermissionTarget(
                kind=PlanPermissionTargetKind.NETWORK_HOST,
                value="network access",
            )
        )
    elif network is not None:
        if not isinstance(network, dict):
            unknown("unrecognized native additional network permissions")
        else:
            enabled = network.get("enabled")
            if enabled is not True and enabled is not False and enabled is not None:
                unknown("unrecognized native additional network permissions")
            if set(network) - {"enabled"}:
                unknown("unrecognized native additional network permission fields")
    return tuple(targets)


def _codex_permission_operation(message: dict[str, Any], params: dict[str, Any]) -> PlanOperation:
    """Build operation evidence from documented app-server approval fields."""
    method = message.get("method")
    bindings: list[PlanNativeBindingID] = []
    for field, name in (
        ("itemId", "item_id"),
        ("approvalId", "approval_id"),
        ("environmentId", "environment_id"),
    ):
        value = params.get(field)
        if isinstance(value, str) and value.strip():
            bindings.append(PlanNativeBindingID(name=name, value=value))

    targets = list(_codex_permission_targets(params))
    if method == "item/fileChange/requestApproval":
        grant_root = params.get("grantRoot")
        if isinstance(grant_root, str) and grant_root.strip():
            targets.append(
                PlanPermissionTarget(
                    kind=PlanPermissionTargetKind.FILESYSTEM_WRITE,
                    value=grant_root,
                )
            )
        return PlanOperation(
            kind=PlanOperationKind.FILE_CHANGE,
            permission_targets=tuple(targets),
            native_binding_ids=tuple(bindings),
        )

    native_kind = params.get("kind", "command")
    operation_kind = {
        "command": PlanOperationKind.COMMAND,
        "writeStdin": PlanOperationKind.WRITE_STDIN,
    }.get(native_kind, PlanOperationKind.OTHER)
    command = params.get("command")
    cwd = params.get("cwd")
    return PlanOperation(
        kind=operation_kind,
        shell_expression=(command if isinstance(command, str) and command.strip() else None),
        execution_dir=Path(cwd) if isinstance(cwd, str) and cwd.strip() else None,
        permission_targets=tuple(targets),
        native_binding_ids=tuple(bindings),
    )


def _codex_approval_options(
    params: dict[str, Any],
    *,
    tool_id: AIToolID,
    capability: PlanModeCapability,
    thread_id: str,
    turn_id: str,
) -> tuple[PlanQuestionOption, ...]:
    """Preserve native decision IDs when app-server advertises them."""
    available = params.get("availableDecisions")
    decision_ids: tuple[str, ...]
    if available is None:
        decision_ids = ("accept", "decline", "cancel")
    elif not isinstance(available, list) or not available:
        raise PlanTransportError(
            "Codex returned malformed native approval decisions.",
            tool_id=tool_id,
            capability=capability,
            session_id=thread_id,
            thread_id=thread_id,
            turn_id=turn_id,
        )
    else:
        decision_ids = tuple(value for value in available if isinstance(value, str))
        structured = tuple(value for value in available if isinstance(value, dict))
        if (
            any(not value.strip() for value in decision_ids)
            or len(set(decision_ids)) != len(decision_ids)
            or structured
            or len(decision_ids) + len(structured) != len(available)
            or not decision_ids
        ):
            raise PlanTransportError(
                "Codex returned malformed or unrepresentable native approval decisions.",
                tool_id=tool_id,
                capability=capability,
                session_id=thread_id,
                thread_id=thread_id,
                turn_id=turn_id,
            )
    labels = {
        "accept": "Approve once",
        "acceptForSession": "Approve for session",
        "decline": "Deny",
        "cancel": "Cancel turn",
    }
    if any(decision not in labels for decision in decision_ids):
        raise PlanTransportError(
            "Codex returned an unknown native approval decision.",
            tool_id=tool_id,
            capability=capability,
            session_id=thread_id,
            thread_id=thread_id,
            turn_id=turn_id,
        )
    return tuple(
        PlanQuestionOption(option_id=decision, label=labels[decision]) for decision in decision_ids
    )


def _answer_codex_approval(
    rpc: Any,
    message: dict[str, Any],
    *,
    thread_id: str,
    turn_id: str,
    handler: PlanInteractionHandler | None,
    tool_id: AIToolID,
    capability: PlanModeCapability,
    deny_automatically: bool,
    command_policy: PlanCommandPolicy | None = None,
    allowed_execution_roots: tuple[Path, ...] = (),
) -> None:
    from crossby.ai_tools.plan_mode import (
        PlanCommandPolicyUnsupportedError,
        PlanInteractionRequiredError,
    )

    params = message.get("params")
    if not isinstance(params, dict):
        raise PlanTransportError(
            "Codex approval request had malformed params.",
            tool_id=tool_id,
            capability=capability,
        )
    _require_codex_binding(
        params,
        thread_id=thread_id,
        turn_id=turn_id,
        tool_id=tool_id,
        capability=capability,
    )
    question_id = str(params.get("approvalId") or params.get("itemId") or "approval")
    operation = _codex_permission_operation(message, params)
    options = _codex_approval_options(
        params,
        tool_id=tool_id,
        capability=capability,
        thread_id=thread_id,
        turn_id=turn_id,
    )
    option_ids = {option.option_id for option in options}
    interaction = PlanInteraction(
        kind=PlanInteractionKind.PERMISSION,
        question_id=question_id,
        prompt=str(params.get("reason") or "Codex requests permission during planning."),
        options=options,
        session_id=thread_id,
        thread_id=thread_id,
        turn_id=turn_id,
        artifact_id=str(params.get("itemId") or "") or None,
        operation=operation,
    )
    policy_match = command_policy is not None and operation_matches_command_policy(
        command_policy,
        operation,
        allowed_execution_roots=allowed_execution_roots,
    )
    if policy_match:
        if "accept" not in option_ids:
            raise PlanCommandPolicyUnsupportedError(
                "Codex cannot preserve command_policy for a matched operation because the "
                "native approval request does not offer one-operation acceptance.",
                tool_id=tool_id,
                capability=capability,
                session_id=thread_id,
                thread_id=thread_id,
                turn_id=turn_id,
                artifact_id=interaction.artifact_id,
            )
        decision = "accept"
    elif deny_automatically:
        if "decline" not in option_ids:
            raise PlanInteractionRequiredError(
                "Codex cannot preserve approval_policy='never' because the native request "
                "does not offer denial.",
                interaction=interaction,
                tool_id=tool_id,
                capability=capability,
            )
        decision = "decline"
    elif handler is None:
        raise PlanInteractionRequiredError(
            "Codex requires a permission decision to continue the planning turn.",
            interaction=interaction,
            tool_id=tool_id,
            capability=capability,
        )
    else:
        response = handler(interaction)
        if response.outcome is PlanInteractionOutcome.CANCELLED:
            if "cancel" not in option_ids:
                raise PlanInteractionRequiredError(
                    "Codex cannot preserve cancellation because the native request does not "
                    "offer it.",
                    interaction=interaction,
                    tool_id=tool_id,
                    capability=capability,
                )
            decision = "cancel"
        elif response.outcome in {
            PlanInteractionOutcome.DENIED,
            PlanInteractionOutcome.SKIPPED,
        }:
            if "decline" not in option_ids:
                raise PlanInteractionRequiredError(
                    "Codex cannot preserve denial because the native request does not offer it.",
                    interaction=interaction,
                    tool_id=tool_id,
                    capability=capability,
                )
            decision = "decline"
        elif response.answer is not None:
            raise PlanInteractionRequiredError(
                "Codex permission decisions do not accept free text.",
                interaction=interaction,
                tool_id=tool_id,
                capability=capability,
            )
        elif response.outcome is PlanInteractionOutcome.APPROVED:
            if "accept" not in option_ids:
                raise PlanInteractionRequiredError(
                    "Codex cannot preserve one-operation approval because the native request "
                    "does not offer it.",
                    interaction=interaction,
                    tool_id=tool_id,
                    capability=capability,
                )
            decision = "accept"
        else:
            try:
                selected_options = validate_plan_option_selection(interaction, response)
                if len(selected_options) != 1:
                    raise ValueError("permission response must select exactly one option")
            except ValueError as exc:
                raise PlanInteractionRequiredError(
                    "Codex permission decisions require exactly one valid native option ID.",
                    interaction=interaction,
                    tool_id=tool_id,
                    capability=capability,
                ) from exc
            decision = selected_options[0]
    rpc.respond(message["id"], {"decision": decision})
