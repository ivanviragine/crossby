"""GitHub Copilot CLI adapter."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from crossby.ai_tools.base import AbstractAITool
from crossby.handoff.models import ConversationTranscript, SessionRef
from crossby.handoff.readers import copilot as copilot_reader
from crossby.models.ai import (
    AIToolCapabilities,
    AIToolID,
    AIToolType,
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
    ModelBreakdown,
    PlanArtifactLocation,
    PlanCommandPolicySupport,
    PlanLaunchApprovalMode,
    PlanModeActivation,
    PlanModeCapability,
    PlanRequestBehavior,
    RestrictionChannel,
    RestrictionState,
    TokenUsage,
)

if TYPE_CHECKING:
    from crossby.ai_tools.headless import HeadlessRuntimeContext
    from crossby.scenes.launch import SceneLaunchArgs, SceneLaunchContext

_PLACEHOLDER_SESSION_ID = "00000000-0000-4000-8000-000000000000"


def _is_uuid(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value.lower()
    except ValueError:
        return False


class CopilotAdapter(AbstractAITool):
    """Adapter for GitHub Copilot CLI."""

    TOOL_ID: ClassVar[AIToolID] = AIToolID.COPILOT
    # Verified live on the read-only floor; see
    # docs/unattended-headless-verification.md#what-read-only-blocks.
    HEADLESS_READ_ONLY_RESTRICTIONS: ClassVar[
        Mapping[RestrictionChannel, tuple[RestrictionState, str]]
    ] = {
        RestrictionChannel.FILE_EDITS: (RestrictionState.ENFORCED, "--deny-tool write"),
        RestrictionChannel.SHELL_COMMANDS: (RestrictionState.ENFORCED, "--deny-tool shell"),
        RestrictionChannel.SHELL_WRITES: (RestrictionState.ENFORCED, "--deny-tool shell"),
        RestrictionChannel.HOOKS: (
            RestrictionState.NOT_ENFORCED,
            "the read-only profile leaves hooks enabled",
        ),
        RestrictionChannel.MCP_TOOLS: (
            RestrictionState.NOT_ENFORCED,
            "the built-in GitHub MCP server stays connected",
        ),
    }

    def capabilities(self) -> AIToolCapabilities:
        from crossby.ai_tools.copilot_sdk import brokered_profile

        brokered = brokered_profile()
        return AIToolCapabilities(
            tool_id=AIToolID.COPILOT,
            display_name="GitHub Copilot",
            binary="copilot",
            tool_type=AIToolType.TERMINAL,
            # `copilot update [channel]` — downloads the latest version.
            update_command=("copilot", "update"),
            supports_model_flag=True,
            headless_flag="--prompt",
            supports_headless=True,
            supports_yolo=True,
            supports_resume=True,
            supports_trusted_dirs=True,
            headless=HeadlessCapability(
                transport=HeadlessNativeTransport.HEADLESS_CLI,
                # Prompt mode reads a piped prompt when --prompt is omitted.
                prompt_transport=HeadlessPromptTransport.STDIN,
                # The run reads Copilot's JSONL stream (--output-format json,
                # verified on 1.0.105 and 1.0.109) for its final answer, session,
                # usage and denials; callers get the final answer as text.
                native_outputs=(HeadlessNativeOutput.TEXT,),
                interaction_modes=(
                    (HeadlessInteractionMode.UNATTENDED, HeadlessInteractionMode.BROKERED)
                    if brokered is not None
                    else (HeadlessInteractionMode.UNATTENDED,)
                ),
                brokered=brokered,
                supports_response_schema=False,
                # Every new session gets an explicit ``--session-id``, and the
                # same flag resumes it, so the ID is always one crossby chose.
                supports_resume=True,
                # ``--deny-tool write`` and ``--deny-tool shell``.
                supports_read_only=True,
                sandbox_behavior=PlanRequestBehavior.TOOL_MANAGED,
                approval_behavior=PlanRequestBehavior.PRESERVED,
                # Per-invocation --allow-tool shell(...) entries; no other tool
                # is auto-approved because --allow-all-tools is never emitted.
                command_policy_support=PlanCommandPolicySupport.NATIVE,
                version_requirement=(
                    "GitHub Copilot CLI reading a piped prompt with --silent, --no-ask-user, "
                    "--session-id, and --deny-tool."
                ),
                # A piped prompt, --session-id resume, and the read-only deny
                # rules were all verified live on this build.
                verified_version="1.0.105",
                remediation=(
                    "Upgrade GitHub Copilot CLI to 1.0.105 or newer. Copilot exposes no "
                    "final-response JSON Schema flag, so schema-constrained sessions must use "
                    "Claude Code, Codex CLI, or Antigravity CLI."
                ),
            ),
            plan_mode=PlanModeCapability(
                supported_launch_approval_modes=(
                    PlanLaunchApprovalMode.YOLO,
                    PlanLaunchApprovalMode.ACCEPT_EDITS,
                ),
                activation=PlanModeActivation.CLI_ARGUMENT,
                activation_detail="Passes --plan before the first interactive prompt.",
                version_requirement="GitHub Copilot CLI exposing --plan.",
                verified_version="1.0.83",
                initial_prompt_after_activation=True,
                artifact_location=PlanArtifactLocation.SESSION,
                artifact_location_detail=(
                    "Interactive plans remain in Copilot-managed session storage. "
                    "No complete collected-session transport is verified."
                ),
                remediation=(
                    "Use interactive launch(..., plan_mode=True), or another adapter for "
                    "collected sessions. Copilot 1.0.83 --prompt omits ask_user; its result and "
                    "share formats do not establish a native question/continuation contract."
                ),
            ),
            supports_accept_edits=True,
            supports_session_start_hook=True,
            supports_stop_hook=True,
            # Copilot's preToolUse has a documented structured stdout schema —
            # flat and top-level, never nested under `hookSpecificOutput` (which
            # appears nowhere in GitHub's hooks docs; it is a Claude/VS Code
            # construct). Modelling this as EXIT_CODE, as crossby did through
            # 0.12.x, threw away the reason string and the `ask` decision.
            hook_output_dialect=HookOutputDialect.PERMISSION_DECISION,
            # ...but its stop channel is a different vocabulary again
            # (`agentStop` reads {"decision": "block", "reason": …}), which is
            # exactly why the two dialects are declared independently.
            hook_stop_dialect=HookStopDialect.BLOCK_DECISION,
            # Non-zero exits other than 2 are fail-closed on preToolUse, but a
            # hook *timeout* is fail-open on every Copilot event, so there is no
            # per-hook fail-closed switch to opt into. Default timeout is 30s
            # (`timeoutSec`).
            hook_fail_open_default=False,
            # Session-scoped scenes: deselected MCP servers become repeated
            # --disable-mcp-server flags. Profile-native approvals travel on an
            # independent launch channel and are filtered against the scene here.
            supports_scene_launch=True,
            scene_tool_denylist_flag="--excluded-tools",
        )

    def build_resume_command(
        self,
        session_id: str,
        *,
        working_dir: Path | None = None,
        network_access: bool = False,
        sandbox: bool = True,
    ) -> list[str] | None:
        """Resume a Copilot session: ``copilot --resume=<session_id>``.

        Accepts and ignores the sandbox context (Copilot does not hard-confine
        writes); the keyword-only params keep polymorphic dispatch TypeError-free.
        """
        return ["copilot", f"--resume={session_id}"]

    def locate_sessions(self, project_path: Path) -> list[SessionRef]:
        return copilot_reader.locate_sessions(project_path)

    def read_session(self, ref: SessionRef) -> ConversationTranscript:
        return copilot_reader.read_session(ref)

    def initial_message_args(self, prompt: str) -> list[str]:
        """Copilot uses -i for the initial message."""
        return ["-i", prompt]

    def parse_transcript(self, transcript_path: Path) -> TokenUsage:
        from crossby.ai_tools.transcript import parse_copilot_transcript

        return parse_copilot_transcript(transcript_path)

    def is_model_compatible(self, model: str) -> bool:
        """Copilot accepts all model IDs."""
        return True

    def plan_mode_args(self) -> list[str]:
        """Copilot supports ``--plan`` (GA'd Jan 2026)."""
        return ["--plan"]

    def plan_dir_args(self, plan_dir: str) -> list[str]:
        """Copilot uses --add-dir for plan directory access."""
        return ["--add-dir", plan_dir]

    def allowed_commands_args(self, commands: list[str]) -> list[str]:
        """Translate canonical patterns to Copilot --allow-tool flags.

        Canonical ``"cmd:args"`` becomes ``--allow-tool "shell(cmd:args)"``.
        """
        result: list[str] = []
        for cmd in commands:
            parts = cmd.split(":", 1)
            binary = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            pattern = f"shell({binary}:{args})" if args else f"shell({binary})"
            result.extend(["--allow-tool", pattern])
        return result

    def accept_edits_args(self) -> list[str]:
        """Copilot auto-approves file writes with ``--allow-tool write`` while
        ``shell`` stays gated."""
        return ["--allow-tool", "write"]

    def yolo_args(self) -> list[str]:
        """Copilot uses ``--yolo`` (alias for ``--allow-all``)."""
        return ["--yolo"]

    def normalize_model_format(self, model_id: str) -> str:
        """Copilot uses dotted format for Claude models."""
        if model_id.startswith("claude-"):
            import re

            # Convert claude-haiku-4-5 -> claude-haiku-4.5
            # Only convert version number separators (digit-digit)
            return re.sub(r"(\d)-(\d)", r"\1.\2", model_id)
        return model_id

    def _headless_argv_for_validation(
        self, request: HeadlessSessionRequest, **_kwargs: object
    ) -> list[str]:
        """Return Copilot's complete command; the prompt itself is piped on stdin."""
        if request.interaction_mode is HeadlessInteractionMode.BROKERED:
            from crossby.ai_tools.copilot_sdk import command

            return command()
        # The real run adds ``--session-id <uuid>``; size the argv the same way.
        return self._headless_command(request, session_id=_PLACEHOLDER_SESSION_ID)

    def _validate_headless_request(
        self, request: HeadlessSessionRequest, **kwargs: Any
    ) -> HeadlessSessionRequest:
        from crossby.ai_tools.copilot_sdk import SDK_REMEDIATION, load_sdk
        from crossby.ai_tools.headless import (
            HeadlessRequestError,
            HeadlessUnsupportedError,
            _is_safe_provenance_id,
        )

        if request.interaction_mode is HeadlessInteractionMode.UNATTENDED:
            if request.resume_id is not None and not _is_uuid(request.resume_id):
                raise HeadlessRequestError(
                    "Copilot session IDs are UUIDs; the requested resume ID is not one.",
                    tool_id=self.TOOL_ID,
                    capability=self.capabilities().headless,
                )
            if request.read_only and request.command_policy is not None:
                raise HeadlessRequestError(
                    "A read-only Copilot session denies the shell, so it cannot take a "
                    "command policy.",
                    tool_id=self.TOOL_ID,
                    capability=self.capabilities().headless,
                )
        if request.interaction_mode is HeadlessInteractionMode.BROKERED:
            if request.resume_id is not None and not _is_safe_provenance_id(request.resume_id):
                raise HeadlessUnsupportedError(
                    "Copilot SDK resume requires an exact bounded native session ID.",
                    tool_id=self.TOOL_ID,
                    capability=self.capabilities().headless,
                )
            try:
                load_sdk()
            except Exception:
                raise HeadlessUnsupportedError(
                    "Copilot brokered sessions require the compatible optional SDK. "
                    + SDK_REMEDIATION,
                    tool_id=self.TOOL_ID,
                    capability=self.capabilities().headless,
                ) from None
        return super()._validate_headless_request(request, **kwargs)

    def _validate_headless_launch_context(self, request: HeadlessSessionRequest) -> None:
        if request.interaction_mode is not HeadlessInteractionMode.BROKERED:
            return super()._validate_headless_launch_context(request)
        from crossby.ai_tools.copilot_sdk import approval_rules
        from crossby.ai_tools.headless import HeadlessUnsupportedError

        try:
            approval_rules(request)
        except ValueError:
            raise HeadlessUnsupportedError(
                "Unsupported Copilot native rule; use kind or kind(argument).",
                tool_id=self.TOOL_ID,
                capability=self.capabilities().headless,
            ) from None

    def _validate_headless_mode_version(
        self, request: HeadlessSessionRequest, detected: Any
    ) -> None:
        super()._validate_headless_mode_version(request, detected)
        if request.interaction_mode is HeadlessInteractionMode.BROKERED:
            from crossby.ai_tools.copilot_sdk import NATIVE_VERSION, SDK_REMEDIATION
            from crossby.ai_tools.headless import HeadlessUnsupportedError

            if detected.normalized != (1, 0, 97):
                raise HeadlessUnsupportedError(
                    f"Copilot SDK sessions require runtime {NATIVE_VERSION}. {SDK_REMEDIATION}",
                    tool_id=self.TOOL_ID,
                    capability=self.capabilities().headless,
                )

    def _detect_headless_version(
        self,
        *,
        timeout_seconds: float,
        deadline: float | None = None,
        mode: HeadlessInteractionMode = HeadlessInteractionMode.UNATTENDED,
    ) -> Any:
        if mode is not HeadlessInteractionMode.BROKERED:
            return super()._detect_headless_version(
                timeout_seconds=timeout_seconds, deadline=deadline, mode=mode
            )
        from crossby.ai_tools.copilot_sdk import SDK_REMEDIATION, command
        from crossby.ai_tools.headless import HeadlessPreflightError
        from crossby.ai_tools.session_process import run_captured
        from crossby.utils.versioning import BinaryVersion, parse_semver

        try:
            probe = run_captured(
                [command()[0], "--version", "--no-auto-update"],
                cwd=Path.cwd(),
                timeout=min(5.0, timeout_seconds),
                deadline=deadline,
            )
            if probe.returncode == 0:
                for line in (probe.stdout + "\n" + probe.stderr).splitlines():
                    parsed = parse_semver(line)
                    if parsed is not None:
                        return BinaryVersion(normalized=parsed, text=line.strip())
        except (OSError, ValueError, TimeoutError):
            pass
        raise HeadlessPreflightError(
            "Cannot verify pinned Copilot runtime. " + SDK_REMEDIATION,
            tool_id=self.TOOL_ID,
            capability=self.capabilities().headless,
        )

    def _headless_command(
        self, request: HeadlessSessionRequest, *, session_id: str | None = None
    ) -> list[str]:
        """Build the exact unattended ``copilot`` prompt-mode invocation.

        The prompt is not an argument: piped on stdin, it runs in prompt mode
        just as ``--prompt <text>`` does, with no command-line size limit.
        ``--session-id`` resumes ``request.resume_id``, or else pins the new
        session to ``session_id``: text output carries no session ID, so the
        only ID crossby can resume is one it chose.
        """
        command = [
            "copilot",
            # -s keeps stdout to the agent's output alone, and --no-ask-user
            # removes the ask_user tool so the agent cannot stop on a question
            # nobody can answer.
            "-s",
            "--no-ask-user",
            "--no-color",
            "--log-level",
            "none",
            # JSONL events: the final answer, the session, premium requests,
            # the model used and every denied tool call (1.0.105 and 1.0.109).
            "--output-format",
            "json",
        ]
        pinned = request.resume_id or session_id
        if pinned is not None:
            command.extend(("--session-id", pinned))
        if request.model:
            command.extend(("--model", self.normalize_model_format(request.model)))
        for path in request.trusted_dirs:
            command.extend(self.plan_dir_args(str(path)))
        if request.read_only:
            # Verified on 1.0.105: forced shell and apply_patch writes were both
            # refused ("Permission to run this tool was denied"). Deny rules win
            # over every allow rule, including ambient ones.
            command.extend(("--deny-tool", "write", "--deny-tool", "shell"))
        if request.command_policy is not None:
            command.extend(
                self.allowed_commands_args(list(request.command_policy.allowed_commands))
            )
        return command

    def _run_headless_session(
        self,
        request: HeadlessSessionRequest,
        version: str,
        context: HeadlessRuntimeContext,
    ) -> HeadlessSessionResult:
        """Run one unattended Copilot prompt turn and normalize its JSONL events."""
        if request.interaction_mode is HeadlessInteractionMode.BROKERED:
            from crossby.ai_tools.copilot_sdk import run_session

            return run_session(request, context)
        from crossby.ai_tools.headless_cli import (
            capture_failure_warnings,
            complete_session,
            frame_streamer,
            parse_json_lines,
            recognized_frame_kind,
            run_managed_command,
        )

        session_id = request.resume_id or _new_session_id()
        output = run_managed_command(
            context,
            argv=self._headless_command(request, session_id=session_id),
            cwd=request.working_dir,
            extra_env={"NO_COLOR": "1"},
            stdin_text=request.prompt,
            on_stdout_lines=frame_streamer(
                context,
                kind_of=lambda frame: recognized_frame_kind(
                    frame, field="type", recognized=_COPILOT_STREAM_EVENT_KINDS
                ),
                provenance_of=lambda frame: {"session_id": _copilot_session(frame)},
                failure_text_of=_copilot_failure_text,
            ),
        )
        warnings = capture_failure_warnings(output)
        # --allow-all-tools is deliberately never emitted, so any tool call
        # outside the session's explicit --allow-tool entries is refused by
        # Copilot rather than escalated to a prompt.
        warnings = (
            *warnings,
            "Copilot ran without --allow-all-tools, so tool calls outside the session command "
            "policy are denied natively.",
        )
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
                warnings=(*warnings, "GitHub Copilot CLI emitted malformed JSONL events."),
            )
        results = [frame for frame in frames if frame.get("type") == "result"]
        if not results:
            # A refusal before the session starts (an org policy, a bad login)
            # is reported on stderr with a failing exit and no events.
            return context.complete(
                HeadlessTerminalStatus.FAILED
                if output.returncode != 0
                else HeadlessTerminalStatus.INVALID_OUTPUT,
                exit_code=output.returncode,
                warnings=(*warnings, "GitHub Copilot CLI ended without its result event."),
            )
        if len(results) > 1 or frames[-1] is not results[0]:
            return context.complete(
                HeadlessTerminalStatus.INVALID_OUTPUT,
                exit_code=output.returncode,
                warnings=(*warnings, "GitHub Copilot CLI emitted events after its result."),
            )
        result = results[0]
        reported = _copilot_session(result)
        if reported != session_id:
            return context.complete(
                HeadlessTerminalStatus.INVALID_OUTPUT,
                exit_code=output.returncode,
                warnings=(
                    *warnings,
                    "GitHub Copilot CLI reported a different session than the one crossby pinned.",
                ),
            )
        native_exit = result.get("exitCode")
        if not isinstance(native_exit, int) or isinstance(native_exit, bool):
            return context.complete(
                HeadlessTerminalStatus.INVALID_OUTPUT,
                exit_code=output.returncode,
                session_id=session_id,
                warnings=(*warnings, "GitHub Copilot CLI's result omitted its exit code."),
            )
        denied = sum(1 for frame in frames if _copilot_denied(frame))
        if denied:
            # Native entries name the tool and its arguments; keep the count.
            context.add_denial(f"GitHub Copilot CLI denied {denied} tool call(s) under its rules")
        answer, models = _copilot_answer(frames)
        return complete_session(
            context,
            request,
            status=(
                HeadlessTerminalStatus.SUCCEEDED
                if native_exit == 0 and output.returncode == 0
                else HeadlessTerminalStatus.FAILED
            ),
            exit_code=output.returncode,
            response_text=answer,
            native_status=None if native_exit == 0 else f"exit_{native_exit}",
            session_id=session_id,
            usage=_copilot_usage(result, models, session_id),
            warnings=warnings,
        )

    def scene_launch_concerns(self) -> set[str]:
        """Copilot scopes only MCP at launch (per-server disable + allow filter)."""
        return {"mcp"}

    def allow_tools_args(
        self,
        allow_tools: list[str],
        scene: SceneLaunchContext | None,
    ) -> list[str]:
        """Render Copilot-native profile approvals without canonical translation.

        Profile values already use Copilot's ``--allow-tool`` grammar, such as
        ``shell(git:*)`` or ``github(create_issue)``. A scene contributes only
        the MCP servers it excludes; entries naming those servers are removed
        while surviving values retain their source ordering and spelling.
        """
        excluded = scene.deselected_mcp() if scene is not None else set()
        args: list[str] = []
        for entry in allow_tools:
            if not _allow_entry_excluded(entry, excluded):
                args += ["--allow-tool", entry]
        return args

    def scene_launch_args(self, scene: SceneLaunchContext) -> SceneLaunchArgs:
        """Render Copilot scene visibility as server-disable flags only.

        Native profile approvals are independent of scene rendering and are
        emitted by :meth:`allow_tools_args`, where the same excluded-server set
        filters them before subprocess dispatch. Copilot's scene is entirely
        flag-driven and writes no artefact files.
        """
        from crossby.scenes.launch import SceneLaunchArgs

        excluded = scene.deselected_mcp()
        args: list[str] = []
        for name in sorted(excluded):
            args += ["--disable-mcp-server", name]
        return SceneLaunchArgs(args=tuple(args))


def _allow_entry_excluded(entry: str, excluded_servers: set[str]) -> bool:
    """True when an ``--allow-tool`` *entry* names a scene-excluded MCP server.

    Matches the bare server name and both per-tool spellings Copilot has used —
    the documented ``<server>(<tool>)`` form and the ``<server>__<tool>``
    namespacing — so ``github``, ``github(create_issue)`` and
    ``github__create_issue`` are all dropped when ``github`` is excluded, while
    an unrelated ``shell(git:*)`` survives.
    """
    return any(
        entry == server or entry.startswith(f"{server}__") or entry.startswith(f"{server}(")
        for server in excluded_servers
    )


# Copilot JSONL event types recorded live on 1.0.105 and 1.0.109. Only these
# refresh the idle deadline; their content never leaves the adapter (the
# ``user.message`` event echoes the prompt).
_COPILOT_STREAM_EVENT_KINDS = frozenset(
    {
        "session.start",
        "session.model_change",
        "session.auto_mode_resolved",
        "user.message",
        "assistant.turn_start",
        "assistant.message_start",
        "assistant.message_delta",
        "assistant.reasoning",
        "assistant.reasoning_delta",
        "assistant.tool_call_delta",
        "assistant.message",
        "assistant.turn_end",
        "assistant.idle",
        "model.call_start",
        "model.call_finished",
        "model.call_final_result",
        "tool.execution_start",
        "tool.execution_complete",
        "session.usage_checkpoint",
        "session.error",
        "result",
    }
)


def _new_session_id() -> str:
    return str(uuid.uuid4())


def _copilot_session(frame: dict[str, Any]) -> str | None:
    """The session a frame names: top-level on ``result``, nested on ``session.start``."""
    value = frame.get("sessionId")
    if value is None and frame.get("type") == "session.start":
        data = frame.get("data")
        value = data.get("sessionId") if isinstance(data, dict) else None
    return value if isinstance(value, str) and value.strip() else None


def _copilot_failure_text(frame: dict[str, Any]) -> str | None:
    """A ``session.error`` event's wording, as failure evidence."""
    if frame.get("type") != "session.error":
        return None
    data = frame.get("data")
    message = data.get("message") if isinstance(data, dict) else None
    return message if isinstance(message, str) else None


def _copilot_denied(frame: dict[str, Any]) -> bool:
    """A tool call Copilot refused under its rules (``error.code == "denied"``)."""
    if frame.get("type") != "tool.execution_complete":
        return False
    data = frame.get("data")
    if not isinstance(data, dict) or data.get("success") is not False:
        return False
    error = data.get("error")
    return isinstance(error, dict) and error.get("code") == "denied"


def _copilot_answer(frames: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    """The final answer's text, and the models that wrote any assistant message.

    Copilot marks the answer ``phase: "final_answer"``; messages that only
    request tools have empty content. Without the marker, the last message
    with content and no tool requests is the answer.
    """
    answer: str | None = None
    fallback: str | None = None
    models: list[str] = []
    for frame in frames:
        if frame.get("type") != "assistant.message":
            continue
        data = frame.get("data")
        if not isinstance(data, dict):
            continue
        model = data.get("model")
        if isinstance(model, str) and model.strip() and model not in models:
            models.append(model)
        content = data.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if data.get("phase") == "final_answer":
            answer = content.strip()
        elif not data.get("toolRequests"):
            fallback = content.strip()
    return answer or fallback, models


def _copilot_usage(result: dict[str, Any], models: list[str], session_id: str) -> TokenUsage | None:
    """Premium requests from the result event, and the models used.

    ``premiumRequests`` is the session's running total (a resumed second turn
    reported 2 on 1.0.105 and 1.0.109), so it is one invocation's usage only
    for a fresh session. Copilot reports no token counts in its JSONL stream.
    """
    usage = result.get("usage")
    raw = usage.get("premiumRequests") if isinstance(usage, dict) else None
    premium = raw if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0 else None
    if premium is None and not models:
        return None
    return TokenUsage(
        premium_requests=premium,
        model_breakdown=[ModelBreakdown(model=name) for name in models],
        session_id=session_id,
    )
