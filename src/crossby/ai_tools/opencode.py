"""OpenCode adapter — terminal AI coding agent with multi-provider model support."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.model_utils import classify_tier_universal, has_date_suffix
from crossby.ai_tools.plan_mode import PlanInteractionHandler
from crossby.models.ai import (
    AIModel,
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
    PlanArtifactLocation,
    PlanArtifactSource,
    PlanCommandPolicySupport,
    PlanInteractionSupport,
    PlanLaunchApprovalMode,
    PlanModeActivation,
    PlanModeCapability,
    PlanRequestBehavior,
    PlanSessionBinding,
    PlanSessionRequest,
    PlanSessionResult,
    PlanSessionTransport,
    TokenUsage,
)

if TYPE_CHECKING:
    from crossby.ai_tools.headless import HeadlessRuntimeContext


class OpenCodeAdapter(AbstractAITool):
    """Adapter for OpenCode CLI.

    OpenCode is a terminal-based AI coding agent that supports 75+ LLM
    providers via the AI SDK. Models are specified as ``provider_id/model_id``
    (e.g., ``anthropic/claude-sonnet-4``).

    Headless mode: ``opencode run "text"`` runs a single
    prompt non-interactively without launching the TUI.
    """

    TOOL_ID: ClassVar[AIToolID] = AIToolID.OPENCODE

    def capabilities(self) -> AIToolCapabilities:
        return AIToolCapabilities(
            tool_id=AIToolID.OPENCODE,
            display_name="OpenCode",
            binary="opencode",
            tool_type=AIToolType.TERMINAL,
            # `opencode upgrade` (not `update`) — the tool's own self-updater
            # handles its install method (npm/brew/pnpm/bun) internally, so
            # crossby needs no Homebrew assumption.
            update_command=("opencode", "upgrade"),
            supports_model_flag=True,
            headless_flag="run",
            supports_headless=True,
            supports_effort=True,
            supported_efforts=(EffortLevel.LOW, EffortLevel.MEDIUM, EffortLevel.HIGH),
            supports_resume=True,
            supports_yolo=True,
            headless=HeadlessCapability(
                transport=HeadlessNativeTransport.HEADLESS_CLI,
                prompt_transport=HeadlessPromptTransport.ARGUMENT,
                # Every managed run uses the --format json event wire; TEXT
                # returns the joined assistant text parts from those events.
                native_outputs=(HeadlessNativeOutput.TEXT, HeadlessNativeOutput.JSONL),
                interaction_modes=(HeadlessInteractionMode.UNATTENDED,),
                supports_response_schema=False,
                successful_native_statuses=("stop",),
                sandbox_behavior=PlanRequestBehavior.TOOL_MANAGED,
                approval_behavior=PlanRequestBehavior.TOOL_MANAGED,
                command_policy_support=PlanCommandPolicySupport.UNSUPPORTED,
                version_requirement="OpenCode exposing opencode run --format json raw events.",
                verified_version="1.18.31",
                remediation=(
                    "Upgrade OpenCode to 1.18.31 or newer. OpenCode exposes no final-response "
                    "JSON Schema flag, so schema-constrained sessions must use Claude Code, "
                    "Codex CLI, or Antigravity CLI."
                ),
            ),
            plan_mode=PlanModeCapability(
                supported_launch_approval_modes=(PlanLaunchApprovalMode.YOLO,),
                activation=PlanModeActivation.CLI_ARGUMENT,
                activation_detail="Selects OpenCode's built-in plan agent with --agent plan.",
                version_requirement="OpenCode exposing the built-in plan agent and --agent.",
                verified_version="1.18.29",
                initial_prompt_after_activation=True,
                artifact_location=PlanArtifactLocation.SESSION,
                artifact_location_detail=(
                    "Crossby captures the session ID emitted by this invocation and exports only "
                    "that exact OpenCode session."
                ),
                export_command=("opencode", "export", "<session-id>"),
                remediation=(
                    "Use run_plan_session() to normalize the exact session export, or use Claude "
                    "when a requested filesystem output directory is required."
                ),
                collector_activation=PlanModeActivation.OPENCODE_SERVER,
                transport=PlanSessionTransport.OPENCODE_SERVER,
                artifact_source=PlanArtifactSource.SESSION_EXPORT,
                binding=PlanSessionBinding.SESSION_ID,
                interaction=PlanInteractionSupport.CALLBACK,
                sandbox_behavior=PlanRequestBehavior.TOOL_MANAGED,
                approval_behavior=PlanRequestBehavior.TOOL_MANAGED,
                command_policy_support=PlanCommandPolicySupport.NATIVE,
                command_policy_detail=(
                    "Pins a session permission catch-all to ask, then appends scoped native shell "
                    "allow rules without changing project or global configuration."
                ),
            ),
            # No session-scoped scene lever. OpenCode loads the OPENCODE_CONFIG
            # file *between* its global and project config layers, so a project
            # opencode.json can re-enable an MCP server the scene deselected —
            # the isolation a scene promises can't be guaranteed. Mirroring the
            # Cursor decision (#106), OpenCode drops the launch-time MCP lever
            # and falls back to persistent `scene use` activation instead.
        )

    def get_models(self) -> list[AIModel]:
        """Return known OpenCode models from the static registry."""
        from crossby.data import get_models_for_tool

        return [
            AIModel(
                id=mid,
                tier=classify_tier_universal(mid.split("/")[-1] if "/" in mid else mid),
                is_alias=not has_date_suffix(mid.split("/")[-1] if "/" in mid else mid),
            )
            for mid in get_models_for_tool(str(self.TOOL_ID))
        ]

    def build_resume_command(
        self,
        session_id: str,
        *,
        working_dir: Path | None = None,
        network_access: bool = False,
        sandbox: bool = True,
    ) -> list[str] | None:
        """Resume an OpenCode session: ``opencode -s <session_id>``.

        Accepts and ignores the sandbox context (OpenCode does not hard-confine
        writes); the keyword-only params keep polymorphic dispatch TypeError-free.
        """
        return ["opencode", "-s", session_id]

    def initial_message_args(self, prompt: str) -> list[str]:
        """OpenCode uses --prompt for the initial message."""
        return ["--prompt", prompt]

    def plan_mode_args(self) -> list[str]:
        """Select OpenCode's built-in, read-only ``plan`` agent."""
        return ["--agent", "plan"]

    def yolo_args(self) -> list[str]:
        """OpenCode's --auto skips approvals, retaining explicit native denials.

        This is not Crossby's classifier-mediated --auto tier.
        """
        return ["--auto"]

    def _headless_command(self, request: HeadlessSessionRequest) -> list[str]:
        """Build the exact unattended ``opencode run`` invocation.

        ``--auto`` is deliberately never emitted: OpenCode's own noninteractive
        permission behavior stays in force, so a permission it cannot resolve is
        refused natively instead of being auto-approved.
        """
        command = ["opencode", "run", "--format", "json", "--log-level", "ERROR"]
        if request.model:
            command.extend(("--model", request.model))
        if request.effort is not None:
            command.extend(self.effort_args(request.effort))
        command.extend(("--", request.prompt))
        return command

    def _run_headless_session(
        self,
        request: HeadlessSessionRequest,
        version: str,
        context: HeadlessRuntimeContext,
    ) -> HeadlessSessionResult:
        """Run one unattended ``opencode run`` turn and normalize its raw events."""
        from crossby.ai_tools.headless_cli import (
            MISSING,
            capture_failure_warnings,
            complete_session,
            frame_streamer,
            non_blank_text,
            parse_json_lines,
            run_managed_command,
        )
        from crossby.ai_tools.plan_process import child_environment

        output = run_managed_command(
            context,
            argv=self._headless_command(request),
            cwd=request.working_dir,
            env=child_environment({"NO_COLOR": "1"}),
            on_stdout_lines=frame_streamer(
                context,
                label="opencode",
                kind_of=lambda frame: non_blank_text(frame.get("type")),
                provenance_of=lambda frame: {"session_id": non_blank_text(frame.get("sessionID"))},
            ),
        )
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
                warnings=(*warnings, "OpenCode emitted malformed raw JSON events."),
            )

        # Progress and provenance were already emitted live by the streamer.
        session_id: str | None = None
        text_parts: list[dict[str, Any]] = []
        finish_reason: str | None = None
        usage: TokenUsage | None = None
        for frame in frames:
            session_id = session_id or non_blank_text(frame.get("sessionID"))
            kind = non_blank_text(frame.get("type"))
            part = frame.get("part")
            if kind == "text" and isinstance(part, dict):
                text_parts.append(part)
            elif kind == "step_finish" and isinstance(part, dict):
                finish_reason = non_blank_text(part.get("reason")) or finish_reason
                usage = _opencode_usage(part.get("tokens"), session_id) or usage

        if finish_reason is None:
            return context.complete(
                HeadlessTerminalStatus.INVALID_OUTPUT,
                exit_code=output.returncode,
                session_id=session_id,
                warnings=(*warnings, "OpenCode ended without a terminal step_finish event."),
            )
        texts = [text for part in text_parts if (text := non_blank_text(part.get("text")))]
        if len(text_parts) > 1 and request.native_output is HeadlessNativeOutput.JSONL:
            warnings = (
                *warnings,
                "Only the final native text part is returned; request text output for the "
                "complete response.",
            )
        if finish_reason != "stop":
            warnings = (*warnings, f"OpenCode finished with native reason {finish_reason!r}.")
        return complete_session(
            context,
            request,
            status=(
                HeadlessTerminalStatus.SUCCEEDED
                if finish_reason == "stop" and output.returncode == 0
                else HeadlessTerminalStatus.FAILED
            ),
            exit_code=output.returncode,
            response_text="\n".join(texts) or None,
            native_object=text_parts[-1] if text_parts else MISSING,
            native_status=finish_reason,
            session_id=session_id,
            usage=usage,
            warnings=warnings,
        )

    def _validate_collected_plan_requirements(self, request: PlanSessionRequest) -> None:
        """Validate a caller-supplied public model identifier without server I/O."""
        if request.model is None:
            return
        provider, separator, model = request.model.partition("/")
        if not separator or not provider.strip() or not model.strip():
            from crossby.ai_tools.plan_mode import PlanSessionUnsupportedError

            raise PlanSessionUnsupportedError(
                "OpenCode requires a provider/model identifier for collected plan sessions.",
                tool_id=self.TOOL_ID,
                capability=self.capabilities().plan_mode,
            )

    def _run_plan_session(
        self,
        request: PlanSessionRequest,
        version: str,
        interaction_handler: PlanInteractionHandler | None,
    ) -> PlanSessionResult:
        """Run the native plan agent with live questions and export its exact session."""
        from crossby.ai_tools.opencode_server import run_native_plan
        from crossby.ai_tools.plan_mode import (
            PlanArtifactAmbiguousError,
            PlanArtifactMalformedError,
            PlanArtifactMissingError,
            PlanBindingMismatchError,
            PlanTransportError,
        )
        from crossby.ai_tools.plan_process import run_captured

        capability = self.capabilities().plan_mode
        working_dir = request.working_dir.resolve()
        deadline = time.monotonic() + request.timeout_seconds
        session_id, terminal_message_id = run_native_plan(
            request, interaction_handler, capability, deadline=deadline
        )

        def remaining_timeout(*, session_id: str) -> float:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PlanTransportError(
                    "OpenCode plan session exceeded its timeout.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=session_id,
                )
            return remaining

        try:
            exported = run_captured(
                ["opencode", "export", session_id],
                cwd=working_dir,
                timeout=remaining_timeout(session_id=session_id),
            )
        except subprocess.TimeoutExpired as exc:
            raise PlanTransportError(
                f"OpenCode export timed out after {exc.timeout} seconds.",
                tool_id=self.TOOL_ID,
                capability=capability,
                session_id=session_id,
            ) from None
        except (OSError, subprocess.SubprocessError) as exc:
            raise PlanTransportError(
                f"OpenCode export failed for session {session_id}: {exc}",
                tool_id=self.TOOL_ID,
                capability=capability,
                session_id=session_id,
            ) from exc
        if exported.returncode != 0:
            raise PlanTransportError(
                f"OpenCode export exited with status {exported.returncode}.",
                tool_id=self.TOOL_ID,
                capability=capability,
                exit_code=exported.returncode,
                session_id=session_id,
                stderr=exported.stderr,
            )
        try:
            payload = json.loads(exported.stdout)
        except json.JSONDecodeError as exc:
            raise PlanArtifactMalformedError(
                f"OpenCode export was not valid JSON: {exc.msg}",
                tool_id=self.TOOL_ID,
                capability=capability,
                exit_code=exported.returncode,
                session_id=session_id,
            ) from exc
        if not isinstance(payload, dict):
            raise PlanArtifactMalformedError(
                "OpenCode export root was not an object.",
                tool_id=self.TOOL_ID,
                capability=capability,
                exit_code=exported.returncode,
                session_id=session_id,
            )
        exported_ids = set(_export_session_ids(payload))
        export_info = payload.get("info")
        if not exported_ids or exported_ids != {session_id}:
            raise PlanBindingMismatchError(
                "OpenCode export session IDs did not match the launched session.",
                tool_id=self.TOOL_ID,
                capability=capability,
                exit_code=exported.returncode,
                session_id=session_id,
            )
        exported_directory = export_info.get("directory") if isinstance(export_info, dict) else None
        if not isinstance(exported_directory, str) or not exported_directory.strip():
            raise PlanArtifactMalformedError(
                "OpenCode export omitted its working directory.",
                tool_id=self.TOOL_ID,
                capability=capability,
                exit_code=exported.returncode,
                session_id=session_id,
            )
        exported_directories = [exported_directory]
        for message in payload.get("messages") or []:
            message_info = message.get("info") if isinstance(message, dict) else None
            path_info = message_info.get("path") if isinstance(message_info, dict) else None
            if not isinstance(path_info, dict):
                continue
            message_cwd = path_info.get("cwd")
            if isinstance(message_cwd, str) and message_cwd.strip():
                exported_directories.append(message_cwd)
        for directory in exported_directories:
            exported_path = Path(directory)
            if not exported_path.is_absolute():
                raise PlanArtifactMalformedError(
                    "OpenCode export contained a relative working directory.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    exit_code=exported.returncode,
                    session_id=session_id,
                )
            if exported_path.resolve() != working_dir:
                raise PlanBindingMismatchError(
                    "OpenCode export working directory did not match the requested workspace.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    exit_code=exported.returncode,
                    session_id=session_id,
                )
        messages = payload.get("messages")
        matching = (
            [
                message
                for message in messages
                if isinstance(message, dict)
                and isinstance(message.get("info"), dict)
                and message["info"].get("id") == terminal_message_id
            ]
            if isinstance(messages, list)
            else []
        )
        if not matching:
            raise PlanBindingMismatchError(
                "OpenCode export did not contain the completed planning message.",
                tool_id=self.TOOL_ID,
                capability=capability,
                session_id=session_id,
                artifact_id=terminal_message_id,
            )
        if len(matching) != 1:
            raise PlanArtifactAmbiguousError(
                "OpenCode export contained multiple records for the completed planning message.",
                tool_id=self.TOOL_ID,
                capability=capability,
                session_id=session_id,
                artifact_id=terminal_message_id,
            )
        # Progress and question/tool-call messages belong to the session, but
        # only the exact terminal message observed through the API is a plan.
        plans = _opencode_plans({"info": export_info, "messages": matching})
        if not plans:
            raise PlanArtifactMissingError(
                "OpenCode's exact session export contained no assistant plan.",
                tool_id=self.TOOL_ID,
                capability=capability,
                exit_code=exported.returncode,
                session_id=session_id,
            )
        plan, artifact_id = plans[0]
        if not plan.strip():
            raise PlanArtifactMalformedError(
                "OpenCode's exact session export contained a blank plan.",
                tool_id=self.TOOL_ID,
                capability=capability,
                exit_code=exported.returncode,
                session_id=session_id,
                artifact_id=artifact_id,
            )
        return PlanSessionResult(
            tool=self.TOOL_ID,
            version=version,
            plan=plan,
            session_id=session_id,
            native_mode="OpenCode native API agent=plan",
            artifact_source=PlanArtifactSource.SESSION_EXPORT,
            binding=PlanSessionBinding.SESSION_ID,
            exit_code=0,
            artifact_id=artifact_id,
        )

    def parse_transcript(self, transcript_path: Path) -> TokenUsage:
        return TokenUsage()

    def standardize_model_id(self, raw_model_id: str) -> str:
        """Convert OpenCode's dashed format back to the internal dotted format."""
        if "claude-" in raw_model_id:
            import re

            # Convert anthropic/claude-haiku-4-5 -> anthropic/claude-haiku-4.5
            return re.sub(r"(\d)-(\d)", r"\1.\2", raw_model_id)
        return raw_model_id

    def effort_args(self, effort: EffortLevel) -> list[str]:
        """OpenCode uses ``--variant <level>`` (xhigh/max map to high for launches)."""
        mapped = "high" if effort in (EffortLevel.XHIGH, EffortLevel.MAX) else effort.value
        return ["--variant", mapped]


def _opencode_usage(tokens: Any, session_id: str | None) -> TokenUsage | None:
    """Normalize one OpenCode ``step_finish`` token object, cache included."""
    from crossby.ai_tools.headless_cli import optional_int

    if not isinstance(tokens, Mapping):
        return None
    cache = tokens.get("cache")
    usage = TokenUsage(
        total_tokens=optional_int(tokens.get("total")),
        input_tokens=optional_int(tokens.get("input")),
        output_tokens=optional_int(tokens.get("output")),
        cached_tokens=optional_int(cache.get("read")) if isinstance(cache, Mapping) else None,
        session_id=session_id,
    )
    if all(
        value is None for value in (usage.total_tokens, usage.input_tokens, usage.output_tokens)
    ):
        return None
    return usage


def _export_session_ids(payload: dict[str, Any]) -> list[str]:
    """Collect IDs from exact documented OpenCode export envelope fields."""
    found: list[str] = []
    info = payload.get("info")
    if isinstance(info, dict) and isinstance(info.get("id"), str):
        found.append(info["id"])
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return found
    for message in messages:
        if not isinstance(message, dict):
            continue
        message_info = message.get("info")
        if isinstance(message_info, dict) and isinstance(message_info.get("sessionID"), str):
            found.append(message_info["sessionID"])
        parts = message.get("parts")
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("sessionID"), str):
                    found.append(part["sessionID"])
    return found


def _opencode_plans(payload: dict[str, Any]) -> list[tuple[str, str | None]]:
    """Return all nonblank assistant responses explicitly bound to plan mode."""
    info = payload.get("info")
    messages = payload.get("messages")
    if not isinstance(info, dict) or not isinstance(messages, list):
        return []
    plans: list[tuple[str, str | None]] = []
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        message_info = message.get("info")
        if (
            not isinstance(message_info, dict)
            or message_info.get("role") != "assistant"
            or (message_info.get("mode") != "plan" and message_info.get("agent") != "plan")
        ):
            continue
        parts = message.get("parts")
        if not isinstance(parts, list):
            continue
        plan_parts = [
            str(part.get("text"))
            for part in parts
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
            and str(part.get("text")).strip()
        ]
        artifact_id = message_info.get("id")
        candidate_id = (str(artifact_id).strip() or None) if artifact_id is not None else None
        if plan_parts:
            plans.append(("\n".join(plan_parts), candidate_id))
    return plans
