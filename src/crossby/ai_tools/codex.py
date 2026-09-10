"""OpenAI Codex CLI adapter."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.plan_mode import PlanInteractionHandler
from crossby.handoff.models import ConversationTranscript, SessionRef
from crossby.handoff.readers import codex as codex_reader
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
from crossby.utils.git_worktree import outside_root_git_metadata_dirs

if TYPE_CHECKING:
    from crossby.scenes.launch import SceneLaunchArgs, SceneLaunchContext

# Codex uses "xhigh" for both our XHIGH and MAX levels
_CODEX_EFFORT_MAP: dict[EffortLevel, str] = {
    EffortLevel.LOW: "low",
    EffortLevel.MEDIUM: "medium",
    EffortLevel.HIGH: "high",
    EffortLevel.XHIGH: "xhigh",
    EffortLevel.MAX: "xhigh",
}


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
            plan_mode=PlanModeCapability(
                activation=PlanModeActivation.UNSUPPORTED,
                activation_detail=(
                    "Codex has a native Plan collaboration mode, but its interactive CLI has no "
                    "public launch-time selector that applies before a positional prompt. A "
                    "positional /plan prefix is ordinary prompt text, not mode activation."
                ),
                version_requirement=(
                    "A Codex CLI release with a public interactive launch-time collaboration-mode "
                    "selector, or a supported app-server-to-TUI activation path."
                ),
                verified_version="0.153.4",
                initial_prompt_after_activation=False,
                artifact_location=PlanArtifactLocation.SESSION,
                artifact_location_detail=(
                    "Collected sessions use the completed plan item from one exact app-server "
                    "thread and turn; ordinary interactive launches remain activation-unsupported."
                ),
                remediation=(
                    "Use run_plan_session() for deterministic collection. Interactive launch "
                    "still requires selecting Plan mode in Codex itself."
                ),
                collector_activation=PlanModeActivation.CODEX_APP_SERVER,
                transport=PlanSessionTransport.CODEX_APP_SERVER,
                artifact_source=PlanArtifactSource.PROTOCOL_EVENT,
                binding=PlanSessionBinding.THREAD_TURN_IDS,
                interaction=PlanInteractionSupport.CALLBACK,
                sandbox_behavior=PlanRequestBehavior.PRESERVED,
                approval_behavior=PlanRequestBehavior.PRESERVED,
                supported_approval_policies=tuple(PlanApprovalPolicy),
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
        from crossby.ai_tools.plan_process import JsonRpcProcess

        capability = self.capabilities().plan_mode
        deadline = time.monotonic() + request.timeout_seconds
        rpc: JsonRpcProcess | None = None

        def remaining() -> float:
            wait = deadline - time.monotonic()
            if wait <= 0:
                raise TimeoutError("Codex plan session exceeded its timeout")
            return wait

        def wait_response(request_id: int) -> dict[str, Any]:
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
            rpc = JsonRpcProcess(["codex", "app-server", "--stdio"], cwd=request.working_dir)
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
                }
                if request.trusted_dirs:
                    sandbox_config["writable_roots"] = [str(path) for path in request.trusted_dirs]
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
                                request.effort.value if request.effort is not None else None
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

            plans: list[tuple[str, str]] = []
            completed = False
            while not completed:
                message = rpc.read(timeout=remaining())
                method = message.get("method")
                params = message.get("params")
                if method == "item/completed" and isinstance(params, dict):
                    item = params.get("item")
                    if not isinstance(item, dict) or item.get("type") != "plan":
                        continue
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
                    plans.append((artifact_id, text))
                    continue
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
                    completed = True
                    continue
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

            if not plans:
                raise PlanArtifactMissingError(
                    "Codex turn completed without an authoritative completed plan item.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=thread_id,
                    thread_id=thread_id,
                    turn_id=turn_id,
                )
            if len(plans) != 1:
                raise PlanArtifactAmbiguousError(
                    "Codex emitted multiple completed plan items for one turn.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=thread_id,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    artifact_id=",".join(item_id for item_id, _ in plans),
                )
            artifact_id, plan = plans[0]
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
            return PlanSessionResult(
                tool=self.TOOL_ID,
                version=version,
                plan=plan,
                session_id=thread_id,
                native_mode="collaborationMode.mode=plan",
                artifact_source=PlanArtifactSource.PROTOCOL_EVENT,
                binding=PlanSessionBinding.THREAD_TURN_IDS,
                exit_code=0,
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
            if rpc is not None:
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
    answers: dict[str, dict[str, list[str]]] = {}
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
        options = tuple(
            PlanQuestionOption(
                option_id=str(option.get("id") or option.get("label")),
                label=str(option.get("label")),
                description=(
                    str(option["description"]) if option.get("description") is not None else None
                ),
            )
            for option in raw_question.get("options") or []
            if isinstance(option, dict) and option.get("label")
        )
        interaction = PlanInteraction(
            kind=PlanInteractionKind.QUESTION,
            question_id=question_id,
            prompt=prompt,
            options=options,
            allow_multiple=bool(
                raw_question.get("allowMultiple") or raw_question.get("isMultiple")
            ),
            session_id=thread_id,
            thread_id=thread_id,
            turn_id=turn_id,
            artifact_id=str(params.get("itemId") or "") or None,
        )
        if handler is None:
            raise PlanInteractionRequiredError(
                "Codex requires an answer to continue the native planning turn.",
                interaction=interaction,
                tool_id=tool_id,
                capability=capability,
            )
        response = handler(interaction)
        answer_values = [response.answer] if response.answer else []
        selected_ids = response.option_ids or (
            (response.option_id,) if response.option_id is not None else ()
        )
        for option_id in selected_ids:
            selected = next((option for option in options if option.option_id == option_id), None)
            answer_values.append(selected.label if selected is not None else option_id)
        if (
            response.outcome in {PlanInteractionOutcome.CANCELLED, PlanInteractionOutcome.SKIPPED}
            or not answer_values
        ):
            raise PlanInteractionRequiredError(
                "Codex planning question was left unanswered.",
                interaction=interaction,
                tool_id=tool_id,
                capability=capability,
            )
        answers[question_id] = {"answers": answer_values}
    rpc.respond(message["id"], {"answers": answers})


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
) -> None:
    from crossby.ai_tools.plan_mode import PlanInteractionRequiredError, PlanTransportError

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
    interaction = PlanInteraction(
        kind=PlanInteractionKind.PERMISSION,
        question_id=question_id,
        prompt=str(params.get("reason") or "Codex requests permission during planning."),
        options=(
            PlanQuestionOption(option_id="accept", label="Approve once"),
            PlanQuestionOption(option_id="decline", label="Deny"),
            PlanQuestionOption(option_id="cancel", label="Cancel turn"),
        ),
        session_id=thread_id,
        thread_id=thread_id,
        turn_id=turn_id,
        artifact_id=str(params.get("itemId") or "") or None,
    )
    if deny_automatically:
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
        selected_option = response.option_id or next(iter(response.option_ids), None)
        if response.outcome is PlanInteractionOutcome.APPROVED or selected_option == "accept":
            decision = "accept"
        elif response.outcome is PlanInteractionOutcome.CANCELLED or selected_option == "cancel":
            decision = "cancel"
        else:
            decision = "decline"
    rpc.respond(message["id"], {"decision": decision})
