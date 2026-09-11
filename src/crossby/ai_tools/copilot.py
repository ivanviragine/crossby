"""GitHub Copilot CLI adapter."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.plan_mode import PlanInteractionHandler
from crossby.handoff.models import ConversationTranscript, SessionRef
from crossby.handoff.readers import copilot as copilot_reader
from crossby.models.ai import (
    AIToolCapabilities,
    AIToolID,
    AIToolType,
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
    TokenUsage,
)

if TYPE_CHECKING:
    from crossby.scenes.launch import SceneLaunchArgs, SceneLaunchContext


class CopilotAdapter(AbstractAITool):
    """Adapter for GitHub Copilot CLI."""

    TOOL_ID: ClassVar[AIToolID] = AIToolID.COPILOT

    def capabilities(self) -> AIToolCapabilities:
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
            plan_mode=PlanModeCapability(
                activation=PlanModeActivation.CLI_ARGUMENT,
                activation_detail="Passes --plan before the first interactive prompt.",
                version_requirement="GitHub Copilot CLI exposing --plan.",
                verified_version="1.0.83",
                initial_prompt_after_activation=True,
                artifact_location=PlanArtifactLocation.SESSION,
                artifact_location_detail=(
                    "Crossby assigns a fresh UUID and parses only the unique local --share export "
                    "created for that session."
                ),
                export_command=("copilot", "--share=<run-owned-path>"),
                remediation=(
                    "Use run_plan_session() for the normalized local share export, or use Claude "
                    "when a persistent requested output directory is required."
                ),
                transport=PlanSessionTransport.HEADLESS_CLI,
                artifact_source=PlanArtifactSource.SESSION_EXPORT,
                binding=PlanSessionBinding.SESSION_ID,
                interaction=PlanInteractionSupport.RESUMABLE_CALLBACK,
                sandbox_behavior=PlanRequestBehavior.TOOL_MANAGED,
                approval_behavior=PlanRequestBehavior.PRESERVED,
                supported_approval_policies=(
                    PlanApprovalPolicy.ON_REQUEST,
                    PlanApprovalPolicy.NEVER,
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
            # --disable-mcp-server flags; the visibility (--excluded-tools) and
            # approval (--allow-tool) layers are independent, so a scene-excluded
            # tool is also filtered out of any profile-supplied allow entries.
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

    def _run_plan_session(
        self,
        request: PlanSessionRequest,
        version: str,
        interaction_handler: PlanInteractionHandler | None,
    ) -> PlanSessionResult:
        """Collect one UUID-bound local Copilot share without remote sharing."""
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
        session_id = str(uuid.uuid4())
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix="crossby-copilot-plan-"))
        except OSError as exc:
            raise PlanTransportError(
                f"GitHub Copilot could not create its temporary share directory: {exc}",
                tool_id=self.TOOL_ID,
                capability=capability,
                session_id=session_id,
            ) from exc
        export_path = temp_dir / f"{session_id}.md"
        base = [
            "copilot",
            "--plan",
            "--session-id",
            session_id,
            f"--share={export_path}",
            "--output-format",
            "json",
            "--no-remote",
            "--no-remote-export",
        ]
        if request.model:
            base.extend(("--model", request.model))
        for path in request.trusted_dirs:
            base.extend(("--add-dir", str(path)))
        if request.approval_policy is PlanApprovalPolicy.NEVER:
            base.append("--deny-tool=*")

        command = [*base, "--prompt", request.prompt]
        seen_questions: set[str] = set()
        deadline = time.monotonic() + request.timeout_seconds
        try:
            for _continuation in range(9):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PlanTransportError(
                        "GitHub Copilot plan session exceeded its timeout.",
                        tool_id=self.TOOL_ID,
                        capability=capability,
                        session_id=session_id,
                        paths=(export_path,),
                    )
                try:
                    run = run_captured(
                        command,
                        cwd=request.working_dir,
                        timeout=remaining,
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    raise PlanTransportError(
                        f"GitHub Copilot plan process failed: {exc}",
                        tool_id=self.TOOL_ID,
                        capability=capability,
                        session_id=session_id,
                        paths=(export_path,),
                    ) from exc
                if run.returncode != 0:
                    raise PlanTransportError(
                        f"GitHub Copilot exited with status {run.returncode}.",
                        tool_id=self.TOOL_ID,
                        capability=capability,
                        exit_code=run.returncode,
                        session_id=session_id,
                        paths=(export_path,),
                        stderr=run.stderr,
                    )
                try:
                    events = parse_jsonl(run.stdout)
                except ValueError as exc:
                    raise PlanArtifactMalformedError(
                        f"GitHub Copilot emitted malformed JSON events: {exc}",
                        tool_id=self.TOOL_ID,
                        capability=capability,
                        exit_code=run.returncode,
                        session_id=session_id,
                        paths=(export_path,),
                    ) from exc
                emitted_ids = {found for event in events for found in _copilot_session_ids(event)}
                if emitted_ids and emitted_ids != {session_id}:
                    raise PlanBindingMismatchError(
                        "GitHub Copilot emitted events for a different session UUID.",
                        tool_id=self.TOOL_ID,
                        capability=capability,
                        exit_code=run.returncode,
                        session_id=session_id,
                        paths=(export_path,),
                    )
                pending = [
                    item
                    for item in _copilot_interactions(events, session_id)
                    if item.question_id not in seen_questions
                ]
                if not pending:
                    break
                interaction = pending[0]
                if interaction_handler is None:
                    raise PlanInteractionRequiredError(
                        "GitHub Copilot requires an explicit response to continue planning.",
                        interaction=interaction,
                        tool_id=self.TOOL_ID,
                        capability=capability,
                    )
                response = interaction_handler(interaction)
                seen_questions.add(interaction.question_id)
                answer: str | None
                if interaction.kind is PlanInteractionKind.PLAN_APPROVAL:
                    # Never translate approval into implementation. The safe
                    # outcomes keep the exact session in planning and ask it to
                    # finish/export the draft.
                    if response.outcome is PlanInteractionOutcome.APPROVED:
                        raise PlanInteractionRequiredError(
                            "Approving Copilot's final plan would authorize implementation; "
                            "choose a non-executing outcome.",
                            interaction=interaction,
                            tool_id=self.TOOL_ID,
                            capability=capability,
                        )
                    answer = "Do not implement. Finish and export the plan."
                else:
                    answer = (
                        response.answer
                        or response.option_id
                        or ", ".join(response.option_ids)
                        or None
                    )
                    if (
                        response.outcome
                        in {
                            PlanInteractionOutcome.DENIED,
                            PlanInteractionOutcome.CANCELLED,
                            PlanInteractionOutcome.SKIPPED,
                        }
                        or not answer
                    ):
                        raise PlanInteractionRequiredError(
                            "GitHub Copilot planning question was left unanswered.",
                            interaction=interaction,
                            tool_id=self.TOOL_ID,
                            capability=capability,
                        )
                command = [*base, "--prompt", answer]
            else:
                raise PlanTransportError(
                    "GitHub Copilot exceeded the bounded planning-question continuation limit.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=session_id,
                    paths=(export_path,),
                )

            if not export_path.is_file() or export_path.is_symlink():
                raise PlanArtifactMissingError(
                    "GitHub Copilot completed without the unique local share export.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    exit_code=0,
                    session_id=session_id,
                    paths=(export_path,),
                )
            try:
                exported = export_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise PlanArtifactMalformedError(
                    f"GitHub Copilot share export could not be read as UTF-8: {exc}",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    exit_code=0,
                    session_id=session_id,
                    paths=(export_path,),
                ) from exc
            try:
                plan = _copilot_export_plan(exported, session_id)
            except ValueError as exc:
                raise PlanBindingMismatchError(
                    str(exc),
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    exit_code=0,
                    session_id=session_id,
                    paths=(export_path,),
                ) from exc
            except RuntimeError as exc:
                raise PlanArtifactAmbiguousError(
                    str(exc),
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    exit_code=0,
                    session_id=session_id,
                    paths=(export_path,),
                ) from exc
            if plan is None:
                raise PlanArtifactMissingError(
                    "GitHub Copilot share export contained no authoritative Plan section.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    exit_code=0,
                    session_id=session_id,
                    paths=(export_path,),
                )
            if not plan.strip():
                raise PlanArtifactMalformedError(
                    "GitHub Copilot share export contained a blank plan.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    exit_code=0,
                    session_id=session_id,
                    paths=(export_path,),
                )
            return PlanSessionResult(
                tool=self.TOOL_ID,
                version=version,
                plan=plan,
                session_id=session_id,
                native_mode="--plan",
                artifact_source=PlanArtifactSource.SESSION_EXPORT,
                binding=PlanSessionBinding.SESSION_ID,
                exit_code=0,
                artifact_id=export_path.name,
            )
        finally:
            # The local share is a run-owned transport artifact rather than
            # caller-owned output; artifact_id retains its provenance name.
            shutil.rmtree(temp_dir, ignore_errors=True)

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

    def scene_launch_concerns(self) -> set[str]:
        """Copilot scopes only MCP at launch (per-server disable + allow filter)."""
        return {"mcp"}

    def scene_launch_args(self, scene: SceneLaunchContext) -> SceneLaunchArgs:
        """Scope a scene for one session via Copilot's two independent layers.

        - **Visibility** — a repeated ``--disable-mcp-server <name>`` for each
          deselected MCP server. A server hidden here cannot be re-exposed by the
          approval layer, so this is authoritative.
        - **Approval** — any profile-supplied ``--allow-tool`` entry that names a
          scene-excluded server is dropped before the surviving entries are
          re-emitted, so a profile can never re-allow a tool the scene removed.
          crossby resolves this itself rather than relying on Copilot's internal
          precedence between the two layers.

        Writes no artefact files — Copilot's scene is entirely flag-driven.
        """
        from crossby.scenes.launch import SceneLaunchArgs

        excluded = scene.deselected_mcp()
        args: list[str] = []
        for name in sorted(excluded):
            args += ["--disable-mcp-server", name]
        for entry in scene.allow_tools:
            if not _allow_entry_excluded(entry, excluded):
                args += ["--allow-tool", entry]
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


def _copilot_session_ids(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"sessionId", "sessionID", "session_id"} and isinstance(nested, str):
                found.append(nested)
            else:
                found.extend(_copilot_session_ids(nested))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_copilot_session_ids(nested))
    return found


def _copilot_interactions(events: list[dict[str, Any]], session_id: str) -> list[PlanInteraction]:
    interactions: list[PlanInteraction] = []
    for event in events:
        event_type = str(event.get("type") or "")
        data = event.get("data")
        source = data if isinstance(data, dict) else event
        if event_type not in {
            "ask_user",
            "user_input_requested",
            "plan.approval",
            "plan_approval",
        }:
            continue
        kind = (
            PlanInteractionKind.PLAN_APPROVAL
            if "plan" in event_type and "approval" in event_type
            else PlanInteractionKind.QUESTION
        )
        question_id = source.get("id") or source.get("questionId") or event.get("id")
        prompt = source.get("question") or source.get("prompt") or source.get("message")
        if not isinstance(question_id, str) or not isinstance(prompt, str):
            continue
        options = tuple(
            PlanQuestionOption(
                option_id=str(option.get("id") or option.get("label")),
                label=str(option.get("label")),
                description=(
                    str(option["description"]) if option.get("description") is not None else None
                ),
            )
            for option in source.get("options") or []
            if isinstance(option, dict) and option.get("label")
        )
        interactions.append(
            PlanInteraction(
                kind=kind,
                question_id=question_id,
                prompt=prompt,
                options=options,
                session_id=session_id,
                artifact_id=str(event.get("id") or "") or None,
            )
        )
    return interactions


def _copilot_export_plan(exported: str, session_id: str) -> str | None:
    """Verify share metadata and extract one explicit Plan section."""
    stripped = exported.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        ids = set(_copilot_session_ids(payload))
        if ids != {session_id}:
            raise ValueError(
                "GitHub Copilot share metadata did not match the assigned session UUID."
            )
        plan = payload.get("plan")
        return plan if isinstance(plan, str) else None

    plan_markers = [
        match.start()
        for pattern in (
            r"(?is)<!--\s*plan:start\s*-->",
            r"(?im)^#{1,3}[ \t]+Plan[ \t]*$",
        )
        for match in re.finditer(pattern, exported)
    ]
    metadata = exported[: min(plan_markers, default=len(exported))]
    metadata_ids = set(
        re.findall(
            r"(?im)^(?:session(?:[_ -]?id)?)\s*:\s*[`\"']?([0-9a-f-]{36})",
            metadata,
        )
    )
    if metadata_ids != {session_id}:
        raise ValueError("GitHub Copilot share metadata did not match the assigned session UUID.")
    marked = re.findall(r"(?is)<!--\s*plan:start\s*-->(.*?)<!--\s*plan:end\s*-->", exported)
    headed: list[str] = []
    for heading in re.finditer(r"(?im)^(#{1,3})[ \t]+Plan[ \t]*$", exported):
        remainder = exported[heading.end() :]
        next_peer = re.search(
            rf"(?m)^#{{1,{len(heading.group(1))}}}(?:[ \t]+|$)",
            remainder,
        )
        end = next_peer.start() if next_peer is not None else len(remainder)
        headed.append(remainder[:end])
    candidates = [candidate.strip() for candidate in [*marked, *headed] if candidate.strip()]
    unique = list(dict.fromkeys(candidates))
    if len(unique) > 1:
        raise RuntimeError("GitHub Copilot share export contained conflicting Plan sections.")
    return unique[0] if unique else None
