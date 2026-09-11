"""OpenCode adapter — terminal AI coding agent with multi-provider model support."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, ClassVar

from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.model_utils import classify_tier_universal, has_date_suffix
from crossby.ai_tools.plan_mode import (
    PlanInteractionHandler,
    parse_plan_question_options,
    validate_plan_option_selection,
)
from crossby.models.ai import (
    AIModel,
    AIToolCapabilities,
    AIToolID,
    AIToolType,
    EffortLevel,
    PlanArtifactLocation,
    PlanArtifactSource,
    PlanInteraction,
    PlanInteractionKind,
    PlanInteractionOutcome,
    PlanInteractionSupport,
    PlanModeActivation,
    PlanModeCapability,
    PlanRequestBehavior,
    PlanSessionBinding,
    PlanSessionRequest,
    PlanSessionResult,
    PlanSessionTransport,
    TokenUsage,
)


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
            plan_mode=PlanModeCapability(
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
                transport=PlanSessionTransport.HEADLESS_CLI,
                artifact_source=PlanArtifactSource.SESSION_EXPORT,
                binding=PlanSessionBinding.SESSION_ID,
                interaction=PlanInteractionSupport.RESUMABLE_CALLBACK,
                sandbox_behavior=PlanRequestBehavior.TOOL_MANAGED,
                approval_behavior=PlanRequestBehavior.TOOL_MANAGED,
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

    def _run_plan_session(
        self,
        request: PlanSessionRequest,
        version: str,
        interaction_handler: PlanInteractionHandler | None,
    ) -> PlanSessionResult:
        """Run OpenCode's plan agent and export only its emitted session ID."""
        from crossby.ai_tools.plan_mode import (
            PlanArtifactAmbiguousError,
            PlanArtifactMalformedError,
            PlanArtifactMissingError,
            PlanBindingMismatchError,
            PlanInteractionRequiredError,
            PlanTransportError,
        )
        from crossby.ai_tools.plan_process import parse_jsonl, run_captured

        capability = self.capabilities().plan_mode
        working_dir = request.working_dir.resolve()
        command = [
            "opencode",
            "run",
            "--dir",
            str(working_dir),
            "--format",
            "json",
            "--agent",
            "plan",
        ]
        execution_args: list[str] = []
        if request.model:
            execution_args.extend(("--model", request.model))
        if request.effort is not None:
            execution_args.extend(self.effort_args(request.effort))
        command.extend(execution_args)
        command.extend(("--", request.prompt))
        deadline = time.monotonic() + request.timeout_seconds

        def remaining_timeout(*, session_id: str | None = None) -> float:
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
            run = run_captured(
                command,
                cwd=working_dir,
                timeout=remaining_timeout(),
            )
        except subprocess.TimeoutExpired as exc:
            raise PlanTransportError(
                f"OpenCode plan process timed out after {exc.timeout} seconds.",
                tool_id=self.TOOL_ID,
                capability=capability,
            ) from None
        except (OSError, subprocess.SubprocessError) as exc:
            raise PlanTransportError(
                f"OpenCode plan process failed: {exc}",
                tool_id=self.TOOL_ID,
                capability=capability,
            ) from exc
        if run.returncode != 0:
            raise PlanTransportError(
                f"OpenCode plan process exited with status {run.returncode}.",
                tool_id=self.TOOL_ID,
                capability=capability,
                exit_code=run.returncode,
                stderr=run.stderr,
            )
        try:
            events = parse_jsonl(run.stdout)
        except ValueError as exc:
            raise PlanArtifactMalformedError(
                f"OpenCode emitted malformed JSON events: {exc}",
                tool_id=self.TOOL_ID,
                capability=capability,
                exit_code=run.returncode,
            ) from exc
        session_ids = {
            value for event in events for value in _event_session_ids(event) if value.strip()
        }
        if not session_ids:
            raise PlanArtifactMissingError(
                "OpenCode completed without emitting the launched session ID.",
                tool_id=self.TOOL_ID,
                capability=capability,
                exit_code=run.returncode,
            )
        if len(session_ids) != 1:
            raise PlanArtifactAmbiguousError(
                "OpenCode emitted events for multiple session IDs.",
                tool_id=self.TOOL_ID,
                capability=capability,
                exit_code=run.returncode,
                session_id=",".join(sorted(session_ids)),
            )
        session_id = next(iter(session_ids))
        seen_questions: set[str] = set()
        max_continuations = 8
        for continuation_count in range(max_continuations + 1):
            try:
                questions = _opencode_questions(events, session_id)
            except ValueError as exc:
                raise PlanArtifactMalformedError(
                    f"OpenCode emitted malformed interaction data: {exc}",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    exit_code=0,
                    session_id=session_id,
                ) from exc
            pending = [
                question for question in questions if question.question_id not in seen_questions
            ]
            if not pending:
                break
            if continuation_count == max_continuations:
                raise PlanTransportError(
                    "OpenCode exceeded the bounded planning-question continuation limit.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=session_id,
                )
            if interaction_handler is None:
                raise PlanInteractionRequiredError(
                    "OpenCode requires an answer to continue the exact planning session.",
                    interaction=pending[0],
                    tool_id=self.TOOL_ID,
                    capability=capability,
                )
            answers: list[str] = []
            for interaction in pending:
                response = interaction_handler(interaction)
                if response.outcome in {
                    PlanInteractionOutcome.DENIED,
                    PlanInteractionOutcome.CANCELLED,
                    PlanInteractionOutcome.SKIPPED,
                }:
                    raise PlanInteractionRequiredError(
                        "OpenCode planning question was left unanswered.",
                        interaction=interaction,
                        tool_id=self.TOOL_ID,
                        capability=capability,
                    )
                try:
                    selected_ids = validate_plan_option_selection(interaction, response)
                except ValueError as exc:
                    raise PlanInteractionRequiredError(
                        "OpenCode planning question requires valid native option IDs.",
                        interaction=interaction,
                        tool_id=self.TOOL_ID,
                        capability=capability,
                    ) from exc
                answer = response.answer or ", ".join(selected_ids) or None
                if not answer:
                    raise PlanInteractionRequiredError(
                        "OpenCode planning question was left unanswered.",
                        interaction=interaction,
                        tool_id=self.TOOL_ID,
                        capability=capability,
                    )
                answers.append(answer)
                seen_questions.add(interaction.question_id)
            continuation_command = [
                "opencode",
                "run",
                "--dir",
                str(working_dir),
                "--session",
                session_id,
                "--format",
                "json",
                "--agent",
                "plan",
                *execution_args,
                "--",
                *answers,
            ]
            try:
                continued = run_captured(
                    continuation_command,
                    cwd=working_dir,
                    timeout=remaining_timeout(session_id=session_id),
                )
            except subprocess.TimeoutExpired as exc:
                raise PlanTransportError(
                    f"OpenCode continuation timed out after {exc.timeout} seconds.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=session_id,
                ) from None
            except (OSError, subprocess.SubprocessError) as exc:
                raise PlanTransportError(
                    f"OpenCode continuation failed for session {session_id}: {exc}",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=session_id,
                ) from exc
            if continued.returncode != 0:
                raise PlanTransportError(
                    f"OpenCode continuation exited with status {continued.returncode}.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    exit_code=continued.returncode,
                    session_id=session_id,
                    stderr=continued.stderr,
                )
            try:
                continued_events = parse_jsonl(continued.stdout)
            except ValueError as exc:
                raise PlanArtifactMalformedError(
                    f"OpenCode continuation emitted malformed JSON events: {exc}",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=session_id,
                ) from exc
            continuation_ids = {
                value for event in continued_events for value in _event_session_ids(event)
            }
            if continuation_ids != {session_id}:
                raise PlanBindingMismatchError(
                    "OpenCode continuation changed or omitted the captured session ID.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=session_id,
                )
            events.extend(continued_events)
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
        plans = _opencode_plans(payload)
        if not plans:
            raise PlanArtifactMissingError(
                "OpenCode's exact session export contained no assistant plan.",
                tool_id=self.TOOL_ID,
                capability=capability,
                exit_code=exported.returncode,
                session_id=session_id,
            )
        if len(plans) != 1:
            raise PlanArtifactAmbiguousError(
                "OpenCode's exact session export contained multiple authoritative plans.",
                tool_id=self.TOOL_ID,
                capability=capability,
                exit_code=exported.returncode,
                session_id=session_id,
                artifact_id=",".join(item_id or "unknown" for _, item_id in plans),
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
            native_mode="--agent plan",
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


def _event_session_ids(event: dict[str, Any]) -> list[str]:
    """Collect IDs from exact documented event and ``part`` fields."""
    part = event.get("part")
    values = (
        event.get("sessionID"),
        part.get("sessionID") if isinstance(part, dict) else None,
    )
    return [value for value in values if isinstance(value, str)]


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
    """Return the terminal assistant response explicitly bound to plan mode."""
    info = payload.get("info")
    messages = payload.get("messages")
    if not isinstance(info, dict) or not isinstance(messages, list):
        return []
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
            return [("\n".join(plan_parts), candidate_id)]
    return []


def _opencode_questions(events: list[dict[str, Any]], session_id: str) -> list[PlanInteraction]:
    questions: list[PlanInteraction] = []
    for event in events:
        part = event.get("part")
        source = part if isinstance(part, dict) and part.get("type") == "question" else event
        if source.get("type") not in {"question", "ask_question"}:
            continue
        question_id = source.get("id") or source.get("questionID") or source.get("questionId")
        prompt = source.get("question") or source.get("prompt")
        if not isinstance(question_id, str) or not isinstance(prompt, str):
            raise ValueError("recognized question omitted a string question ID or prompt")
        if not question_id.strip() or not prompt.strip():
            raise ValueError("recognized question contained a blank question ID or prompt")
        options = parse_plan_question_options(source.get("options"))
        questions.append(
            PlanInteraction(
                kind=PlanInteractionKind.QUESTION,
                question_id=question_id,
                prompt=prompt,
                options=options,
                session_id=session_id,
                artifact_id=str(source.get("id") or "") or None,
            )
        )
    return questions
