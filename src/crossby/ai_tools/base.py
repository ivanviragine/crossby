"""Abstract base class for AI tool adapters with self-registration.

Adding a new AI tool = one file with one class. No other files to modify.
The `__init_subclass__` hook auto-registers each concrete adapter.
"""

from __future__ import annotations

import inspect
import math
import os
import queue
import shutil
import struct
import subprocess
import sys
import threading
import warnings
from abc import ABC, abstractmethod
from contextlib import suppress
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from crossby.ai_tools.headless import (
        HeadlessEventHandler,
        HeadlessRuntimeContext,
        SessionInteractionHandler,
    )
    from crossby.ai_tools.interactive import InteractiveLaunchHandler
    from crossby.handoff.models import ConversationTranscript, SessionRef
    from crossby.scenes.launch import SceneLaunchArgs, SceneLaunchContext
    from crossby.utils.versioning import BinaryVersion

import structlog

from crossby.ai_tools.plan_mode import (
    PlanArtifactLocationError,
    PlanInteractionHandler,
    PlanModeAdapterContractError,
)
from crossby.models.ai import (
    AIModel,
    AIToolCapabilities,
    AIToolID,
    EffortLevel,
    HeadlessInteractionMode,
    HeadlessPreflightCheck,
    HeadlessPreflightDeferredCheck,
    HeadlessPromptTransport,
    HeadlessSessionPreflight,
    HeadlessSessionRequest,
    HeadlessSessionResult,
    HeadlessTerminalStatus,
    ModelTier,
    PlanArtifactLocation,
    PlanArtifactSource,
    PlanCommandPolicySupport,
    PlanInteraction,
    PlanInteractionResponse,
    PlanInteractionSupport,
    PlanLaunchApprovalMode,
    PlanModeActivation,
    PlanPreflightCheck,
    PlanPreflightDeferredCheck,
    PlanRequestBehavior,
    PlanSessionPreflight,
    PlanSessionRequest,
    PlanSessionResult,
    TokenUsage,
)
from crossby.models.config import ComplexityModelMapping

logger = structlog.get_logger()

# Argument-delivered prompts and adapter-supplied argv values must fit the
# native process contract before an adapter begins version probing or spawns a
# child. POSIX limits one argument to 131,072 bytes on Linux; Windows limits
# the rendered *whole* command line to 32,767 UTF-16 code units, including its
# terminating NUL. These ceilings leave room for the adapter's fixed flags
# while keeping stdin and protocol transports unlimited. POSIX additionally
# caps the aggregate argv and environment passed to execve.
_MAX_HEADLESS_ARGUMENT_PROMPT = 30_000 if sys.platform.startswith("win") else 120_000
_MAX_HEADLESS_WINDOWS_COMMAND_LINE = 32_767
_HEADLESS_POSIX_EXEC_SAFETY_MARGIN = 8 * 1024


def _argument_prompt_length(prompt: str) -> int:
    """Return the native argv space used by one argument-delivered prompt."""
    if sys.platform.startswith("win"):
        rendered = subprocess.list2cmdline([prompt])
        return len(rendered.encode("utf-16-le")) // 2
    return len(os.fsencode(prompt))


def _headless_posix_exec_size(argv: list[str]) -> int:
    """Return a conservative byte count for the child argv and environment."""
    # Every managed native adapter inherits the current environment and pins
    # NO_COLOR for its child. Include NUL terminators and pointer slots so the
    # preflight remains below execve's aggregate allocation, not just below its
    # maximum single-argument limit.
    environment = {**os.environ, "NO_COLOR": "1"}
    strings = sum(len(os.fsencode(argument)) + 1 for argument in argv)
    strings += sum(
        len(os.fsencode(name)) + len(os.fsencode(value)) + 2 for name, value in environment.items()
    )
    pointer_bytes = (len(argv) + len(environment) + 2) * struct.calcsize("P")
    return strings + pointer_bytes


def _headless_posix_exec_limit() -> int:
    """Return the platform's aggregate execve budget with a safe fallback."""
    try:
        limit = os.sysconf("SC_ARG_MAX")
    except (AttributeError, OSError, ValueError):
        # This is below the per-argument ceiling and is the Linux minimum used
        # by the managed transport's existing argument guard.
        return 131_072
    return limit if isinstance(limit, int) and limit > 0 else 131_072


class AbstractAITool(ABC):
    """Base for all AI tool adapters.

    Concrete subclasses must set TOOL_ID as a class variable.
    Registration happens automatically via __init_subclass__.
    """

    TOOL_ID: ClassVar[AIToolID]
    _registry: ClassVar[dict[AIToolID, type[AbstractAITool]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "TOOL_ID") and not inspect.isabstract(cls):
            if cls.TOOL_ID in AbstractAITool._registry:
                existing_cls = AbstractAITool._registry[cls.TOOL_ID]
                warnings.warn(
                    f"TOOL_ID '{cls.TOOL_ID}' already registered by {existing_cls.__name__}; "
                    f"overwriting with {cls.__name__}",
                    UserWarning,
                    stacklevel=2,
                )
            AbstractAITool._registry[cls.TOOL_ID] = cls

    @classmethod
    def get(cls, tool_id: str | AIToolID) -> AbstractAITool:
        """Get an adapter instance by tool ID."""
        tid = AIToolID(tool_id) if not isinstance(tool_id, AIToolID) else tool_id
        if tid not in cls._registry:
            raise ValueError(f"Unknown AI tool: {tool_id}")
        return cls._registry[tid]()

    @classmethod
    def available_tools(cls) -> list[AIToolID]:
        """List all registered tool IDs."""
        return list(cls._registry.keys())

    @classmethod
    def detect_installed(cls) -> list[AIToolID]:
        """Detect which registered AI tools are installed on the system."""
        installed = []
        for tool_id, tool_cls in cls._registry.items():
            adapter = tool_cls()
            if shutil.which(adapter.capabilities().binary):
                installed.append(tool_id)
        return installed

    @abstractmethod
    def capabilities(self) -> AIToolCapabilities:
        """Declare what this tool can do."""
        raise NotImplementedError

    def get_models(self) -> list[AIModel]:
        """Return known models from the static registry.

        Uses universal tier classification. Override for tools with
        special model ID formats (e.g. OpenCode's provider/model).
        Returns an empty list if no models are registered for this tool.
        """
        from crossby.ai_tools.model_utils import classify_tier_universal, has_date_suffix
        from crossby.data import get_models_for_tool

        return [
            AIModel(
                id=mid,
                tier=classify_tier_universal(mid),
                is_alias=not has_date_suffix(mid),
            )
            for mid in get_models_for_tool(str(self.TOOL_ID))
        ]

    def get_default_model(self, tier: ModelTier) -> AIModel | None:
        """Get the best model for a given tier.

        Override in subclasses for tool-specific tier keywords.
        """
        models = self.get_models()
        tier_models = [m for m in models if m.tier == tier]
        if not tier_models:
            return None
        return pick_best_model(tier_models)

    def get_recommended_mapping(self) -> ComplexityModelMapping:
        """Get recommended model mapping for all complexity levels."""
        fast = self.get_default_model(ModelTier.FAST)
        balanced = self.get_default_model(ModelTier.BALANCED)
        powerful = self.get_default_model(ModelTier.POWERFUL)

        return ComplexityModelMapping(
            easy=fast.id if fast else None,
            medium=balanced.id if balanced else None,
            complex=balanced.id if balanced else None,
            very_complex=powerful.id if powerful else None,
        )

    def launch(
        self,
        working_dir: Path,
        model: str | None = None,
        prompt: str | None = None,
        detach: bool = False,
        transcript_path: Path | None = None,
        trusted_dirs: list[str] | None = None,
        effort: EffortLevel | None = None,
        allowed_commands: list[str] | None = None,
        yolo: bool = False,
        plan_mode: bool = False,
        accept_edits: bool = False,
        auto: bool = False,
        scene: SceneLaunchContext | None = None,
        network_access: bool = False,
        plan_output_dir: Path | None = None,
        allow_tools: list[str] | None = None,
        *,
        sandbox: bool = True,
        on_event: InteractiveLaunchHandler | None = None,
    ) -> int:
        """Launch the AI tool in the given directory.

        Default implementation builds a command via build_launch_command()
        and runs it with transcript capture. Override for tools with
        non-standard launch behavior (e.g. GUI tools).

        Args:
            working_dir: Directory to run in.
            model: Model ID to use (or None for tool default).
            prompt: Optional initial message passed to the tool on launch.
            detach: If True, launch in background (GUI tools).
            transcript_path: Optional path to write session transcript for
                token usage extraction.
            trusted_dirs: Optional list of directory paths to pre-authorize.
                Tools that support directory-trust flags (e.g. --add-dir) will
                pass these so the user is not prompted for confirmation.
            effort: Optional reasoning effort level for the AI tool.
            allowed_commands: Optional list of canonical command patterns to
                pre-authorize (e.g. ``["myapp:*", "./scripts/check.sh:*"]``).
            yolo: If True, skip all permission prompts (YOLO mode).
            plan_mode: If True, start in the tool's read-only plan/approval mode.
            accept_edits: If True, auto-approve file edits while still prompting
                for shell/commands (the accept-edits tier).
            auto: If True, request the tool's classifier-mediated auto mode
                (Claude and Cursor); downgrades to accept-edits, then default
                prompting, on tools that lack it outside Plan mode. Plan mode
                requires a declared native approval combination.
            scene: Optional session-scoped scene context. When set, this
                adapter's ``scene_launch_args`` renders the scene's artefacts and
                contributes extra argv (appended to the launch command) and env
                (merged over ``os.environ`` for the child) — touching nothing in
                the tracked project files.
            network_access: If True, request network access inside the tool's
                sandbox. Only Codex acts on it (it pins
                ``sandbox_workspace_write.network_access``); other tools warn and
                ignore it upstream via the ``supports_network_access`` capability.
            plan_output_dir: Optional filesystem directory in which native plan
                artifacts must be writable. Session-only, harness-managed, and
                private-artifact tools reject this requirement before launch.
            allow_tools: Optional list of tool-native approval patterns from a
                launch profile (e.g. Copilot's ``"shell(git:*)"``). Unlike
                ``allowed_commands``, these values are already in an adapter's
                native grammar and must never be translated as canonical command
                patterns.
            sandbox: Whether to launch with the tool's sandbox enabled. Only
                adapters declaring ``supports_sandbox_toggle`` translate this
                programmatic input into a sandbox-selection flag.
            on_event: Optional startup handler for adapters whose plan capability
                declares supports_ready_event. On PLAN_READY, call the supplied
                session.send_message() once. Cannot be combined with prompt.

        Returns:
            Exit code from the tool process (0 for detached).
        """
        from crossby.utils.process import run_with_transcript

        if on_event is not None and (
            not plan_mode
            or not self.capabilities().plan_mode.supports_ready_event
            or prompt is not None
        ):
            raise ValueError("Startup events require a supported Plan launch without prompt")

        if detach and plan_mode and self.capabilities().plan_mode.supports_ready_event:
            raise ValueError("Terminal Plan startup cannot run detached")

        self.validate_plan_mode_request(
            plan_mode=plan_mode,
            yolo=yolo,
            auto=auto,
            accept_edits=accept_edits,
            initial_message=prompt,
            plan_output_dir=plan_output_dir,
            working_dir=working_dir,
        )

        # Render the scene once, then split its result across the command builder
        # (argv) and the environment builder (env) so the artefacts are written a
        # single time per launch.
        scene_args = self.scene_launch_args(scene) if scene is not None else None
        command_kwargs: dict[str, Any] = {
            "model": model,
            "initial_message": prompt,
            "plan_mode": plan_mode,
            "trusted_dirs": trusted_dirs,
            "effort": effort,
            "allowed_commands": allowed_commands,
            "yolo": yolo,
            "accept_edits": accept_edits,
            "auto": auto,
            "scene": scene_args,
            "working_dir": working_dir,
            "network_access": network_access,
        }
        if plan_output_dir is not None:
            command_kwargs["plan_output_dir"] = plan_output_dir
        # ``build_launch_command`` is a public adapter hook. Preserve
        # pre-toggle overrides, which do not accept the new keyword.
        if self.capabilities().supports_sandbox_toggle:
            command_kwargs["sandbox"] = sandbox
        cmd = self.build_launch_command(**command_kwargs)
        # Native profile approvals intentionally compose *around* the public
        # command-builder hook. This preserves legacy/custom builders while
        # keeping native values out of allowed_commands_args(), which only
        # accepts crossby canonical command:arguments patterns.
        cmd.extend(self.allow_tools_args(allow_tools or [], scene))
        extra_env = self.build_launch_environment(scene=scene_args)
        child_env = {**os.environ, **extra_env} if extra_env else None
        logger.info("ai_tool.launch", tool=str(self.TOOL_ID), model=model, cwd=str(working_dir))
        if (
            plan_mode
            and self.capabilities().plan_mode.activation is PlanModeActivation.TERMINAL_INPUT
        ):
            return self._run_terminal_plan_command(
                cmd, working_dir, transcript_path, child_env, on_event
            )
        return run_with_transcript(cmd, transcript_path, cwd=working_dir, env=child_env)

    def _run_terminal_plan_command(
        self,
        cmd: list[str],
        working_dir: Path,
        transcript_path: Path | None,
        env: dict[str, str] | None,
        on_event: InteractiveLaunchHandler | None,
    ) -> int:
        """Adapters advertising TERMINAL_INPUT must implement the actual launcher."""
        raise NotImplementedError("Terminal Plan activation requires a launcher")

    def _wrap_terminal_plan_command(self, cmd: list[str], initial_message: str | None) -> list[str]:
        """Return an executable command that performs terminal activation, too."""
        raise NotImplementedError("Terminal Plan activation requires a command wrapper")

    def run_headless_session(
        self,
        request: HeadlessSessionRequest,
        interaction_handler: SessionInteractionHandler | None = None,
        event_handler: HeadlessEventHandler | None = None,
        *,
        cancel_event: threading.Event | None = None,
    ) -> HeadlessSessionResult:
        """Run one ordinary headless session under Crossby's managed boundary.

        The base layer owns the single absolute deadline, capability/version
        validation, callback isolation, terminal reconciliation, and cleanup.
        Adapters receive only :class:`HeadlessRuntimeContext`, not caller-owned
        subprocess responsibilities.
        """
        from crossby.ai_tools.headless import (
            HeadlessAdapterContractError,
            HeadlessPreflightError,
            HeadlessRuntimeContext,
            HeadlessTransportError,
            _HeadlessStopError,
        )

        started_at = monotonic()
        deadline = started_at + request.timeout_seconds
        if cancel_event is not None and cancel_event.is_set():
            capability = self.capabilities().headless
            raise HeadlessPreflightError(
                "The caller cancelled before managed headless preflight completed.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        normalized = self._validate_headless_request(
            request,
            interaction_handler=interaction_handler,
            require_existing_working_dir=True,
            deadline=deadline,
            cancel_event=cancel_event,
        )
        self._validate_headless_requirements(normalized)
        detected = self._detect_headless_version_bounded(
            deadline=deadline,
            cancel_event=cancel_event,
        )
        context = HeadlessRuntimeContext(
            tool_id=self.TOOL_ID,
            version=detected.text,
            capability=self.capabilities().headless,
            request=normalized,
            started_at=started_at,
            deadline=deadline,
            interaction_handler=interaction_handler,
            event_handler=event_handler,
            cancel_event=cancel_event,
        )

        def finalize_stop(stop: _HeadlessStopError) -> HeadlessSessionResult:
            """Publish the terminal outcome before closing the adapter context."""
            if not context._claim_runtime_stop(stop):
                # A result was published before the monitor claimed this stop,
                # so the natural terminal owner wins.
                result = context.result
                assert result is not None
                return result
            result = context.complete(
                stop.status,
                warnings=(stop.warning,),
                _terminal_finalization=True,
            )
            context.cleanup(abort=True)
            return result

        responses: queue.Queue[HeadlessSessionResult | None | BaseException] = queue.Queue(1)

        def invoke_adapter() -> None:
            try:
                value: HeadlessSessionResult | None | BaseException = self._run_headless_session(
                    normalized,
                    detected.text,
                    context,
                )
            except BaseException as exc:
                value = exc
            with suppress(queue.Full):
                responses.put_nowait(value)
                # A timeout/cancellation already closed the runtime.  A late
                # adapter result can no longer reach the caller.

        threading.Thread(
            target=invoke_adapter,
            daemon=True,
            name=f"crossby-headless-{self.TOOL_ID.value}",
        ).start()
        adapter_value: HeadlessSessionResult | None | BaseException
        try:
            while True:
                context.checkpoint()
                try:
                    adapter_value = responses.get(timeout=min(0.05, context.remaining_seconds()))
                except queue.Empty:
                    continue
                break
            context.checkpoint()
        except _HeadlessStopError as stop:
            return finalize_stop(stop)

        if isinstance(adapter_value, HeadlessTransportError):
            context.cleanup(abort=True)
            # Rebuild the snapshot after cleanup was requested so the caller
            # receives all safe progress observed before the failure.
            raise HeadlessTransportError(
                str(adapter_value),
                tool_id=self.TOOL_ID,
                capability=self.capabilities().headless,
                partial_result=context.partial_snapshot(),
            ) from None
        if isinstance(adapter_value, HeadlessAdapterContractError):
            context.cleanup(abort=True)
            raise adapter_value
        if isinstance(adapter_value, _HeadlessStopError):
            return finalize_stop(adapter_value)
        if isinstance(adapter_value, BaseException):
            context.cleanup(abort=True)
            raise context.transport_error(
                "The managed headless transport failed before producing a terminal result."
            ) from None

        result = context.result
        if result is None:
            context.cleanup(abort=True)
            raise context.adapter_contract_error(
                "The adapter returned without using the runtime's terminal result constructor."
            )
        if adapter_value is not None and adapter_value != result:
            context.cleanup(abort=True)
            raise context.adapter_contract_error(
                "The adapter returned a result different from the runtime's terminal result."
            )
        self._validate_headless_result(result, detected.text)
        context.cleanup(
            abort=result.status
            in {HeadlessTerminalStatus.CANCELLED, HeadlessTerminalStatus.TIMED_OUT}
        )
        return result

    def preflight_headless_session(
        self,
        request: HeadlessSessionRequest,
        interaction_handler: SessionInteractionHandler | None = None,
        *,
        timeout_seconds: float = 5.0,
        cancel_event: threading.Event | None = None,
    ) -> HeadlessSessionPreflight:
        """Boundedly validate a prospective managed session without starting it."""
        from crossby.ai_tools.headless import HeadlessPreflightError

        started_at = monotonic()
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("headless preflight timeout must be positive and finite")
        deadline = started_at + min(timeout_seconds, request.timeout_seconds)
        if cancel_event is not None and cancel_event.is_set():
            raise HeadlessPreflightError(
                "The caller cancelled managed headless preflight.",
                tool_id=self.TOOL_ID,
                capability=self.capabilities().headless,
            )
        normalized = self._validate_headless_request(
            request,
            interaction_handler=interaction_handler,
            require_existing_working_dir=False,
            deadline=deadline,
            cancel_event=cancel_event,
        )
        self._validate_headless_requirements(normalized)
        detected = self._detect_headless_version_bounded(
            deadline=deadline,
            cancel_event=cancel_event,
        )
        return HeadlessSessionPreflight(
            tool=self.TOOL_ID,
            detected_version=detected.text,
            normalized_version=detected.normalized,
            capability=self.capabilities().headless,
            working_dir=normalized.working_dir,
            trusted_dirs=normalized.trusted_dirs,
            checked=(
                HeadlessPreflightCheck.SESSION_CAPABILITY,
                HeadlessPreflightCheck.REQUEST_COMPATIBILITY,
                HeadlessPreflightCheck.RESPONSE_SCHEMA,
                HeadlessPreflightCheck.CLI_VERSION,
            ),
            deferred=(
                HeadlessPreflightDeferredCheck.FILESYSTEM,
                HeadlessPreflightDeferredCheck.AUTHENTICATION,
                HeadlessPreflightDeferredCheck.MODEL_AVAILABILITY,
                HeadlessPreflightDeferredCheck.TRANSPORT_STARTUP,
                HeadlessPreflightDeferredCheck.PROTOCOL_NEGOTIATION,
                HeadlessPreflightDeferredCheck.OUTPUT_COLLECTION,
            ),
        )

    def _validate_headless_request(
        self,
        request: HeadlessSessionRequest,
        *,
        interaction_handler: SessionInteractionHandler | None,
        require_existing_working_dir: bool,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> HeadlessSessionRequest:
        """Reject every statically knowable incompatibility before startup."""
        from crossby.ai_tools.headless import (
            HeadlessRequestError,
            HeadlessSchemaError,
            HeadlessUnsupportedError,
            validate_response_schema,
        )

        caps = self.capabilities()
        capability = caps.headless
        if not capability.managed_supported:
            raise HeadlessUnsupportedError.for_tool(
                tool_id=self.TOOL_ID,
                display_name=caps.display_name,
                capability=capability,
            )
        if capability.prompt_transport is HeadlessPromptTransport.ARGUMENT:
            prompt_length = _argument_prompt_length(request.prompt)
            if prompt_length > _MAX_HEADLESS_ARGUMENT_PROMPT:
                raise HeadlessRequestError(
                    f"{caps.display_name} cannot safely deliver a {prompt_length}-unit prompt "
                    f"through its native argv transport (limit: "
                    f"{_MAX_HEADLESS_ARGUMENT_PROMPT}). Use an adapter with stdin or protocol "
                    "prompt delivery, or shorten the prompt.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                )
        if request.interaction_mode not in capability.interaction_modes:
            raise HeadlessUnsupportedError(
                f"{caps.display_name} cannot preserve interaction_mode="
                f"{request.interaction_mode.value!r} for managed headless sessions.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        if request.interaction_mode is HeadlessInteractionMode.BROKERED:
            if interaction_handler is None:
                raise HeadlessRequestError(
                    "A brokered headless session requires an interaction handler.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                )
        elif interaction_handler is not None:
            raise HeadlessRequestError(
                "An unattended headless session cannot accept an interaction handler.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        if request.native_output not in capability.native_outputs:
            raise HeadlessUnsupportedError(
                f"{caps.display_name} does not support native_output="
                f"{request.native_output.value!r} for managed sessions.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        response_schema = request.response_schema
        if response_schema is not None:
            if not capability.supports_response_schema:
                raise HeadlessUnsupportedError(
                    f"{caps.display_name} does not support caller response schemas.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                )
            try:
                response_schema = validate_response_schema(response_schema)
            except ValueError as exc:
                raise HeadlessSchemaError(
                    str(exc),
                    tool_id=self.TOOL_ID,
                    capability=capability,
                ) from None
        if request.resume_id is not None and not capability.supports_resume:
            raise HeadlessUnsupportedError(
                f"{caps.display_name} does not support managed session resumption.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        working_dir = request.working_dir.resolve()
        if require_existing_working_dir and not working_dir.is_dir():
            raise HeadlessRequestError(
                f"{caps.display_name} requires an existing working directory: {working_dir}",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        trusted_dirs = tuple(path.resolve() for path in request.trusted_dirs)
        if trusted_dirs and not caps.supports_trusted_dirs:
            raise HeadlessUnsupportedError(
                f"{caps.display_name} cannot preserve trusted_dirs for managed sessions.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        if request.effort is not None and (
            not caps.supports_effort or request.effort not in caps.supported_efforts
        ):
            raise HeadlessUnsupportedError(
                f"{caps.display_name} cannot preserve the requested effort.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        if request.network_access and not caps.supports_network_access:
            raise HeadlessUnsupportedError(
                f"{caps.display_name} cannot preserve network_access=True.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        if not request.sandbox and capability.sandbox_behavior is not PlanRequestBehavior.PRESERVED:
            raise HeadlessUnsupportedError(
                f"{caps.display_name} cannot preserve sandbox=False.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        if request.approval_policy not in capability.supported_approval_policies:
            raise HeadlessUnsupportedError(
                f"{caps.display_name} cannot preserve approval_policy="
                f"{request.approval_policy.value!r}.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        if (
            request.command_policy is not None
            and capability.command_policy_support is PlanCommandPolicySupport.UNSUPPORTED
        ):
            raise HeadlessUnsupportedError(
                f"{caps.display_name} cannot preserve command_policy for managed sessions.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        normalized = request.model_copy(
            update={
                "working_dir": working_dir,
                "trusted_dirs": trusted_dirs,
                "response_schema": response_schema,
            }
        )
        self._validate_headless_argv(
            normalized,
            deadline=deadline,
            cancel_event=cancel_event,
        )
        return normalized

    def _detect_headless_version_bounded(
        self,
        *,
        deadline: float,
        cancel_event: threading.Event | None,
    ) -> BinaryVersion:
        """Probe within the existing absolute deadline and cancellation budget."""
        from crossby.ai_tools.headless import HeadlessPreflightError

        responses: queue.Queue[BinaryVersion | BaseException] = queue.Queue(1)

        def invoke() -> None:
            try:
                value: BinaryVersion | BaseException = self._detect_headless_version(
                    timeout_seconds=max(0.0, deadline - monotonic()),
                    deadline=deadline,
                )
            except BaseException as exc:
                value = exc
            with suppress(queue.Full):
                responses.put_nowait(value)

        threading.Thread(target=invoke, daemon=True, name="crossby-headless-version").start()
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise HeadlessPreflightError(
                    "The caller cancelled during managed headless version probing.",
                    tool_id=self.TOOL_ID,
                    capability=self.capabilities().headless,
                )
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise HeadlessPreflightError(
                    "Managed headless preflight timed out during version probing.",
                    tool_id=self.TOOL_ID,
                    capability=self.capabilities().headless,
                )
            try:
                value = responses.get(timeout=min(0.05, remaining))
            except queue.Empty:
                continue
            if isinstance(value, BaseException):
                raise value
            return value

    def _detect_headless_version(
        self,
        *,
        timeout_seconds: float,
        deadline: float | None = None,
    ) -> BinaryVersion:
        """Probe the CLI version against the managed capability declaration."""
        from crossby.ai_tools.headless import (
            HeadlessAdapterContractError,
            HeadlessPreflightError,
            HeadlessUnsupportedError,
        )
        from crossby.utils.versioning import detect_binary_version_info, parse_semver

        caps = self.capabilities()
        capability = caps.headless
        floor = parse_semver(capability.verified_version or "")
        if floor is None:
            raise HeadlessAdapterContractError(
                f"{caps.display_name} declares managed headless support without a parseable "
                "verified_version.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        detected = detect_binary_version_info(caps.binary, timeout_seconds=timeout_seconds)
        if deadline is not None and deadline - monotonic() <= 0:
            raise HeadlessPreflightError(
                "Managed headless preflight timed out during version probing.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        if detected is None or detected.normalized < floor:
            raise HeadlessUnsupportedError.for_installed_version(
                tool_id=self.TOOL_ID,
                display_name=caps.display_name,
                capability=capability,
                installed_version=detected.text if detected is not None else None,
            )
        return detected

    def _validate_headless_requirements(self, request: HeadlessSessionRequest) -> None:
        """Adapter hook for request constraints known without transport I/O."""
        return None

    def _headless_argv_for_validation(
        self,
        request: HeadlessSessionRequest,
        *,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[str] | None:
        """Return the native argv to validate before a version probe, if applicable."""
        return None

    def _validate_headless_argv(
        self,
        request: HeadlessSessionRequest,
        *,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """Reject an adapter's native argv when it exceeds process limits."""
        from crossby.ai_tools.headless import HeadlessRequestError

        argv = self._headless_argv_for_validation(
            request,
            deadline=deadline,
            cancel_event=cancel_event,
        )
        if argv is None:
            return

        caps = self.capabilities()
        capability = caps.headless
        if sys.platform.startswith("win"):
            rendered = subprocess.list2cmdline(argv)
            # CreateProcess counts the terminating NUL in its 32,767-unit
            # command-line limit; list2cmdline returns only the rendered text.
            command_length = len(rendered.encode("utf-16-le")) // 2 + 1
            if command_length > _MAX_HEADLESS_WINDOWS_COMMAND_LINE:
                raise HeadlessRequestError(
                    f"{caps.display_name} cannot safely deliver a {command_length}-unit "
                    "native command line "
                    f"(limit: {_MAX_HEADLESS_WINDOWS_COMMAND_LINE}). Shorten the prompt, "
                    "response schema, or other command arguments.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                )
            return

        for argument in argv:
            argument_length = len(os.fsencode(argument))
            if argument_length > _MAX_HEADLESS_ARGUMENT_PROMPT:
                raise HeadlessRequestError(
                    f"{caps.display_name} cannot safely deliver a {argument_length}-byte "
                    "native argv argument "
                    f"(limit: {_MAX_HEADLESS_ARGUMENT_PROMPT}). Shorten the prompt, "
                    "response schema, or other command arguments.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                )

        exec_size = _headless_posix_exec_size(argv)
        exec_limit = _headless_posix_exec_limit()
        if exec_size > exec_limit - _HEADLESS_POSIX_EXEC_SAFETY_MARGIN:
            raise HeadlessRequestError(
                f"{caps.display_name} cannot safely deliver a {exec_size}-byte native argv "
                f"and environment (limit: {exec_limit} bytes). Shorten the prompt, "
                "response schema, or other command arguments.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )

    def _run_headless_session(
        self,
        request: HeadlessSessionRequest,
        version: str,
        context: HeadlessRuntimeContext,
    ) -> HeadlessSessionResult | None:
        """Protected adapter hook for one complete managed native lifecycle."""
        from crossby.ai_tools.headless import HeadlessUnsupportedError

        caps = self.capabilities()
        raise HeadlessUnsupportedError.for_tool(
            tool_id=self.TOOL_ID,
            display_name=caps.display_name,
            capability=caps.headless,
        )

    def _validate_headless_result(self, result: HeadlessSessionResult, version: str) -> None:
        """Validate adapter identity/provenance not expressible in the model."""
        from crossby.ai_tools.headless import HeadlessAdapterContractError

        caps = self.capabilities()
        if result.tool is not self.TOOL_ID:
            raise HeadlessAdapterContractError(
                f"{caps.display_name} returned the wrong managed-session tool identifier.",
                tool_id=self.TOOL_ID,
                capability=caps.headless,
            )
        if result.version != version:
            raise HeadlessAdapterContractError(
                f"{caps.display_name} did not preserve the exact probed version text.",
                tool_id=self.TOOL_ID,
                capability=caps.headless,
            )
        terminals = [event for event in result.events if event.kind.value == "terminal"]
        if len(terminals) != 1 or result.events[-1] is not terminals[0]:
            raise HeadlessAdapterContractError(
                f"{caps.display_name} did not return exactly one final terminal event.",
                tool_id=self.TOOL_ID,
                capability=caps.headless,
            )

    def run_plan_session(
        self,
        request: PlanSessionRequest,
        interaction_handler: PlanInteractionHandler | None = None,
    ) -> PlanSessionResult:
        """Run and collect one native planning session without changing ``launch``.

        The base boundary owns capability/version/request validation and final
        provenance checks. Concrete adapters implement only their native wire or
        artifact lifecycle in :meth:`_run_plan_session`.
        """
        from crossby.ai_tools.plan_mode import PlanTransportError

        deadline = monotonic() + request.timeout_seconds
        request = self._validate_collected_plan_request(
            request,
            require_existing_working_dir=True,
            validate_adapter_requirements=False,
        )
        caps = self.capabilities()
        capability = caps.plan_mode
        probe_timeout = deadline - monotonic()
        if probe_timeout <= 0:
            raise PlanTransportError(
                f"{caps.display_name} plan session timed out before version probing.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        detected = self._detect_collected_plan_version(
            timeout_seconds=probe_timeout,
            deadline=deadline,
        )
        remaining_timeout = deadline - monotonic()
        if remaining_timeout <= 0:
            raise PlanTransportError(
                f"{caps.display_name} plan session timed out during version probing.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        request = request.model_copy(update={"timeout_seconds": remaining_timeout})
        # Preserve the established runtime failure order: exact version support
        # is established before collector-specific request requirements. Static
        # preflight performs the same requirements check before probing so a
        # consumer can reject an incompatible prospective request immediately.
        self._validate_collected_plan_requirements(request)
        if (
            interaction_handler is not None
            and capability.interaction is not PlanInteractionSupport.TERMINAL
        ):
            original_handler = interaction_handler

            def bounded_handler(interaction: PlanInteraction) -> PlanInteractionResponse:
                def timed_out() -> PlanTransportError:
                    return PlanTransportError(
                        f"{caps.display_name} plan session timed out waiting for an "
                        "interaction callback.",
                        tool_id=self.TOOL_ID,
                        capability=capability,
                        session_id=interaction.session_id,
                        thread_id=interaction.thread_id,
                        turn_id=interaction.turn_id,
                        artifact_id=interaction.artifact_id,
                    )

                if deadline - monotonic() <= 0:
                    raise timed_out()
                responses: queue.Queue[PlanInteractionResponse | BaseException] = queue.Queue(1)

                def invoke() -> None:
                    try:
                        responses.put(original_handler(interaction))
                    except BaseException as exc:
                        responses.put(exc)

                # Python cannot forcibly stop caller code. A daemon worker may
                # finish later, but only this waiting collector can send its
                # result to the native process, which is closed on timeout.
                threading.Thread(target=invoke, daemon=True, name="crossby-plan-callback").start()
                try:
                    response = responses.get(timeout=max(0.0, deadline - monotonic()))
                except queue.Empty:
                    raise timed_out() from None
                if deadline - monotonic() <= 0:
                    raise timed_out()
                if isinstance(response, BaseException):
                    raise response
                return response

            interaction_handler = bounded_handler
        result = self._run_plan_session(request, detected.text, interaction_handler)
        if result.tool is not self.TOOL_ID:
            raise PlanModeAdapterContractError(
                f"{caps.display_name} collector returned the wrong tool identifier.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        if result.version != detected.text:
            raise PlanModeAdapterContractError(
                f"{caps.display_name} collector did not preserve the exact probed version text.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        if (
            result.artifact_source is not capability.artifact_source
            or result.binding is not capability.binding
        ):
            raise PlanModeAdapterContractError(
                f"{caps.display_name} collector provenance disagrees with its capability.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        return result

    def preflight_plan_session(
        self,
        request: PlanSessionRequest,
        *,
        timeout_seconds: float = 5.0,
    ) -> PlanSessionPreflight:
        """Validate static collection requirements without creating any paths.

        The bounded version probe is the only subprocess. Runtime repeats every
        check and additionally validates filesystem state, authentication,
        model availability, native protocol negotiation, and artifact binding.
        """
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("plan-session preflight timeout must be positive and finite")
        self._validate_collected_plan_request(
            request,
            require_existing_working_dir=False,
        )
        deadline = monotonic() + timeout_seconds
        detected = self._detect_collected_plan_version(
            timeout_seconds=timeout_seconds,
            deadline=deadline,
        )
        capability = self.capabilities().plan_mode
        return PlanSessionPreflight(
            tool=self.TOOL_ID,
            detected_version=detected.text,
            normalized_version=detected.normalized,
            capability=capability,
            checked=(
                PlanPreflightCheck.SESSION_CAPABILITY,
                PlanPreflightCheck.REQUEST_COMPATIBILITY,
                PlanPreflightCheck.COMMAND_POLICY,
                PlanPreflightCheck.CLI_VERSION,
            ),
            deferred=(
                PlanPreflightDeferredCheck.FILESYSTEM,
                PlanPreflightDeferredCheck.AUTHENTICATION,
                PlanPreflightDeferredCheck.MODEL_AVAILABILITY,
                PlanPreflightDeferredCheck.PROTOCOL_NEGOTIATION,
                PlanPreflightDeferredCheck.ARTIFACT_COLLECTION,
            ),
        )

    def _validate_collected_plan_request(
        self,
        request: PlanSessionRequest,
        *,
        require_existing_working_dir: bool,
        validate_adapter_requirements: bool = True,
    ) -> PlanSessionRequest:
        """Normalize and validate requirements shared by preflight and runtime."""
        from crossby.ai_tools.plan_mode import (
            PlanCommandPolicyUnsupportedError,
            PlanSessionUnsupportedError,
        )

        caps = self.capabilities()
        capability = caps.plan_mode
        if not capability.session_supported:
            raise PlanSessionUnsupportedError.for_tool(
                tool_id=self.TOOL_ID,
                display_name=caps.display_name,
                capability=capability,
            )

        working_dir = request.working_dir.resolve()
        if require_existing_working_dir and not working_dir.is_dir():
            raise PlanSessionUnsupportedError(
                f"{caps.display_name} requires an existing working directory: {working_dir}",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        if request.plan_output_dir is not None and (
            capability.artifact_source is not PlanArtifactSource.REQUESTED_PATH
            or not request.plan_output_dir.resolve().is_relative_to(working_dir)
        ):
            raise PlanArtifactLocationError.for_tool(
                tool_id=self.TOOL_ID,
                display_name=caps.display_name,
                capability=capability,
                requested_dir=request.plan_output_dir,
            )
        if request.trusted_dirs and not caps.supports_trusted_dirs:
            raise PlanSessionUnsupportedError(
                f"{caps.display_name} cannot preserve trusted-directory choices for collected "
                "plan sessions. Remove trusted_dirs or use another adapter.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        if request.effort is not None and (
            not caps.supports_effort or request.effort not in caps.supported_efforts
        ):
            raise PlanSessionUnsupportedError(
                f"{caps.display_name} cannot preserve effort={request.effort.value!r} for "
                "collected plan sessions. Remove effort or use another adapter.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        if request.network_access and not caps.supports_network_access:
            raise PlanSessionUnsupportedError(
                f"{caps.display_name} cannot preserve network_access=True for collected plan "
                "sessions. Disable network access or use another adapter.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        if not request.sandbox and capability.sandbox_behavior is not PlanRequestBehavior.PRESERVED:
            raise PlanSessionUnsupportedError(
                f"{caps.display_name} cannot preserve sandbox=False for collected plan sessions.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        if request.approval_policy not in capability.supported_approval_policies:
            raise PlanSessionUnsupportedError(
                f"{caps.display_name} cannot preserve approval_policy="
                f"{request.approval_policy.value!r} for collected plan sessions.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        if (
            request.command_policy is not None
            and capability.command_policy_support is PlanCommandPolicySupport.UNSUPPORTED
        ):
            raise PlanCommandPolicyUnsupportedError.for_tool(
                tool_id=self.TOOL_ID,
                display_name=caps.display_name,
                capability=capability,
            )

        normalized = request.model_copy(
            update={
                "working_dir": working_dir,
                "trusted_dirs": tuple(path.resolve() for path in request.trusted_dirs),
                "plan_output_dir": (
                    request.plan_output_dir.resolve()
                    if request.plan_output_dir is not None
                    else None
                ),
            }
        )
        if validate_adapter_requirements:
            self._validate_collected_plan_requirements(normalized)
        return normalized

    def _validate_collected_plan_requirements(self, request: PlanSessionRequest) -> None:
        """Adapter hook for request facts knowable without filesystem or protocol I/O."""
        return None

    def _detect_collected_plan_version(
        self,
        *,
        timeout_seconds: float,
        deadline: float | None = None,
    ) -> BinaryVersion:
        """Probe and validate the exact CLI version against public capability metadata."""
        from crossby.ai_tools.plan_mode import (
            PlanModeAdapterContractError,
            PlanSessionUnsupportedError,
            PlanTransportError,
        )
        from crossby.utils.versioning import detect_binary_version_info, parse_semver

        caps = self.capabilities()
        capability = caps.plan_mode
        floor = parse_semver(
            capability.collector_verified_version or capability.verified_version or ""
        )
        if floor is None:
            raise PlanModeAdapterContractError(
                f"{caps.display_name} declares collected plan support without a parseable "
                "verified_version.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        detected = detect_binary_version_info(caps.binary, timeout_seconds=timeout_seconds)
        if deadline is not None and deadline - monotonic() <= 0:
            raise PlanTransportError(
                f"{caps.display_name} plan session timed out during version probing.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        if detected is None or detected.normalized < floor:
            raise PlanSessionUnsupportedError.for_installed_version(
                tool_id=self.TOOL_ID,
                display_name=caps.display_name,
                capability=capability,
                installed_version=detected.text if detected is not None else None,
            )
        return detected

    def _run_plan_session(
        self,
        request: PlanSessionRequest,
        version: str,
        interaction_handler: PlanInteractionHandler | None,
    ) -> PlanSessionResult:
        """Adapter hook for a complete native planning lifecycle."""
        from crossby.ai_tools.plan_mode import PlanSessionUnsupportedError

        caps = self.capabilities()
        raise PlanSessionUnsupportedError.for_tool(
            tool_id=self.TOOL_ID,
            display_name=caps.display_name,
            capability=caps.plan_mode,
        )

    def parse_transcript(self, transcript_path: Path) -> TokenUsage:
        """Parse a transcript file for token usage.

        Default implementation extracts token usage from transcript text.
        Override for tools with special parsing needs (e.g. premium requests).
        Returns TokenUsage with whatever fields could be parsed.
        """
        from crossby.ai_tools.transcript import parse_transcript_common

        return parse_transcript_common(transcript_path)

    def is_model_compatible(self, model: str) -> bool:
        """Check if a model ID is valid for this tool."""
        return True  # Default: allow all. Override per tool.

    # Session handoff — concrete readers own per-tool cwd matching.

    def locate_sessions(self, project_path: Path) -> list[SessionRef]:
        """Return all session refs this tool has recorded for ``project_path``.

        Default: no handoff support — concrete adapters override.
        """
        raise NotImplementedError(f"{self.TOOL_ID} does not support session handoff")

    def read_session(self, ref: SessionRef) -> ConversationTranscript:
        """Parse the session pointed to by ``ref`` into a ConversationTranscript.

        Default: no handoff support — concrete adapters override.
        """
        raise NotImplementedError(f"{self.TOOL_ID} does not support session handoff")

    def initial_message_args(self, prompt: str) -> list[str]:
        """Get CLI args to pass an initial message for an interactive session.

        Default: no support (returns empty list). Override per tool.
        Tools that support positional initial messages return [prompt].
        Tools that require a flag return [flag, prompt].
        """
        return []

    # ------------------------------------------------------------------
    # Extended API — used by build_launch_command() and library consumers
    # ------------------------------------------------------------------

    def plan_mode_args(self) -> list[str]:
        """Get extra CLI args for native plan/approval mode."""
        return []  # Default: no plan mode support

    def plan_output_args(self, plan_output_dir: Path, *, working_dir: Path) -> list[str]:
        """Route native plan artifacts to a requested workspace directory.

        Only adapters whose typed capability advertises
        :attr:`PlanArtifactLocation.REQUESTED_PATH` may override this hook.
        """
        return []

    def validate_plan_mode_request(
        self,
        *,
        plan_mode: bool,
        yolo: bool = False,
        auto: bool = False,
        accept_edits: bool = False,
        initial_message: str | None = None,
        plan_output_dir: Path | None = None,
        working_dir: Path | None = None,
        verify_version: bool = True,
        interactive: bool = True,
    ) -> None:
        """Reject any plan request the adapter cannot honor truthfully.

        This shared pre-launch gate is called by :meth:`launch`,
        :meth:`build_launch_command`, and ``crossby launch``. GUI adapters that
        override ``launch`` call it explicitly before constructing their
        workspace-open command.
        """
        from crossby.ai_tools.plan_mode import (
            PlanArtifactLocationError,
            PlanModeConflictError,
            PlanModeLaunchError,
            PlanModeUnsupportedError,
        )

        caps = self.capabilities()
        capability = caps.plan_mode

        if plan_output_dir is not None and not plan_mode:
            raise PlanModeLaunchError(
                f"{caps.display_name} received plan_output_dir={plan_output_dir} "
                "without plan_mode=True. Enable native plan mode or remove the output request.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        if not plan_mode:
            return

        requested = self._requested_plan_approval(yolo=yolo, auto=auto, accept_edits=accept_edits)
        if requested is not None and (
            not interactive or requested not in capability.supported_launch_approval_modes
        ):
            raise PlanModeConflictError.for_tool(
                tool_id=self.TOOL_ID,
                display_name=caps.display_name,
                capability=capability,
                conflicts=(requested.value,),
            )

        if not capability.supported:
            raise PlanModeUnsupportedError.for_tool(
                tool_id=self.TOOL_ID,
                display_name=caps.display_name,
                capability=capability,
            )

        if capability.activation is PlanModeActivation.TERMINAL_INPUT:
            self._validate_terminal_plan_request(initial_message, interactive=interactive)

        if initial_message and not capability.initial_prompt_after_activation:
            raise PlanModeUnsupportedError(
                f"{caps.display_name} cannot deliver an initial prompt after native plan-mode "
                "activation. Remediation: launch without an initial prompt, then submit the task "
                "after selecting plan mode manually.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )

        if plan_output_dir is not None:
            output_is_in_workspace = (
                working_dir is not None
                and plan_output_dir.resolve().is_relative_to(working_dir.resolve())
            )
            if (
                capability.artifact_location is not PlanArtifactLocation.REQUESTED_PATH
                or not output_is_in_workspace
            ):
                raise PlanArtifactLocationError.for_tool(
                    tool_id=self.TOOL_ID,
                    display_name=caps.display_name,
                    capability=capability,
                    requested_dir=plan_output_dir,
                )

        if capability.activation is PlanModeActivation.CLI_ARGUMENT and not self.plan_mode_args():
            raise PlanModeAdapterContractError(
                f"{caps.display_name} declares native CLI plan mode but its adapter emitted no "
                "activation arguments. Update Crossby before retrying.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )

        if not verify_version:
            return

        from crossby.utils.versioning import detect_binary_version, parse_semver

        verified_version = parse_semver(capability.verified_version or "")
        if verified_version is None:
            raise PlanModeAdapterContractError(
                f"{caps.display_name} declares native plan mode without a parseable "
                "verified_version. Update Crossby before retrying.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )
        installed_version = detect_binary_version(caps.binary)
        if installed_version is None or installed_version < verified_version:
            raise PlanModeUnsupportedError.for_installed_version(
                tool_id=self.TOOL_ID,
                display_name=caps.display_name,
                capability=capability,
                installed_version=installed_version,
            )
        if capability.activation is PlanModeActivation.TERMINAL_INPUT:
            self._validate_terminal_plan_version(installed_version)

    def _validate_terminal_plan_request(
        self, initial_message: str | None, *, interactive: bool
    ) -> None:
        raise NotImplementedError("Terminal Plan activation requires request validation")

    def _validate_terminal_plan_version(self, version: tuple[int, int, int]) -> None:
        raise NotImplementedError("Terminal Plan activation requires version validation")

    def plan_dir_args(self, plan_dir: str) -> list[str]:
        """Get extra CLI args to grant write access to a plan output directory."""
        return []  # Default: no plan dir support

    def trusted_dirs_args(
        self, dirs: list[str], *, autonomy_args: list[str] | None = None
    ) -> list[str]:
        """Get extra CLI args to grant access to a list of trusted directories.

        Default implementation delegates to plan_dir_args() per directory, so
        any adapter that overrides plan_dir_args() automatically supports this
        method. Adapters without directory-trust support return [].

        *autonomy_args* carries the already-resolved autonomy/permission-mode
        flags (from :meth:`_autonomy_launch_args`) so an adapter can avoid
        re-emitting a flag the autonomy tier already supplied (e.g. Codex's
        ``--sandbox workspace-write``). Ignored by the default implementation.
        """
        result: list[str] = []
        for d in dirs:
            result.extend(self.plan_dir_args(d))
        return result

    def sandbox_config_args(
        self,
        *,
        autonomy_args: list[str],
        trusted_dirs: list[str] | None,
        working_dir: Path | None,
        network_access: bool,
        sandbox: bool = True,
    ) -> list[str]:
        """Compose the sandbox / writable-root / network args for a launch.

        Default: emit only the trusted-directory flags (the pre-existing
        behavior), so ``working_dir``, ``network_access``, and ``sandbox`` are
        ignored. This keeps unsupported tools byte-identical to before.

        Capability-enabled adapters override this to translate explicit sandbox
        selection. Codex also owns writable-root and network composition here;
        Cursor maps the selection directly to its CLI flag. Called once by
        :meth:`build_launch_command` in place of the old direct
        ``trusted_dirs_args`` call.
        """
        if trusted_dirs:
            return self.trusted_dirs_args(trusted_dirs, autonomy_args=autonomy_args)
        return []

    def normalize_model_format(self, model_id: str) -> str:
        """Normalize a model ID to this tool's expected format.

        For example, Copilot uses dotted format (claude-haiku-4.5) while
        Claude uses dashed format (claude-haiku-4-5).

        Default: return as-is. Override per tool.
        """
        return model_id

    def standardize_model_id(self, raw_model_id: str) -> str:
        """Convert a tool-specific model ID to the internal standard format.

        Our internal registry uses dotted notation (e.g. claude-haiku-4.5).
        Tools that output dashed notation (claude-haiku-4-5) should override
        this to convert it back to dotted notation.

        Default: return as-is. Override per tool.
        """
        return raw_model_id

    def allowed_commands_args(self, commands: list[str]) -> list[str]:
        """Get CLI args to pre-authorize a list of command patterns.

        Canonical patterns use colon-separated syntax (e.g. ``"myapp:*"``,
        ``"./scripts/check.sh:*"``).  Each adapter translates them into
        tool-specific flags.

        Default: no support (returns empty list). Override per tool.
        """
        return []

    def allow_tools_args(
        self,
        allow_tools: list[str],
        scene: SceneLaunchContext | None,
    ) -> list[str]:
        """Render profile-native tool approvals for one launch.

        ``allow_tools`` entries are intentionally separate from
        :meth:`allowed_commands_args`: they already use an adapter's native
        syntax, rather than Crossby's canonical ``command:arguments`` grammar.
        The optional scene lets an adapter discard an approval whose tool the
        active scene excluded. Most adapters have no native approval channel.
        """
        return []

    def structured_output_args(self, json_schema: dict[str, Any]) -> list[str]:
        """Get extra CLI args to enforce structured JSON output according to a schema.

        Default: return empty list. Override per tool if they support it.
        """
        return []

    def unwrap_structured_output(self, raw: str) -> str:
        """Unwrap a tool-specific stdout envelope to yield the model payload.

        Default: return ``raw`` unchanged. Override per tool when the CLI wraps
        the model response in a metadata envelope so downstream parsers receive
        the model output directly.
        """
        return raw

    def headless_prompt_stdin_args(self) -> list[str] | None:
        """CLI args that make this tool read its headless prompt from stdin.

        Default: ``None`` — the tool has no confirmed stdin path, so callers
        must deliver the prompt through ``argv`` via
        :meth:`build_launch_command`'s ``prompt`` parameter.

        When non-``None``, the contract for the caller is:

        1. Build the command with ``prompt=None`` so the prompt is *not* placed
           on ``argv``.
        2. Append these args after the model / JSON-schema flags.
        3. Feed the prompt to the process via ``subprocess.run(..., input=<prompt>)``.

        This keeps the whole prompt out of ``argv``, so Linux's per-arg
        ``MAX_ARG_STRLEN`` limit (131,072 bytes) cannot be hit regardless of
        transcript size. Override only when a tool's stdin behaviour for its
        headless prompt is documented and stable across CLI releases.
        """
        return None

    def effort_args(self, effort: EffortLevel) -> list[str]:
        """Get extra CLI args to set reasoning effort level.

        Default: return empty list. Override per tool.
        """
        return []

    def yolo_args(self) -> list[str]:
        """Get extra CLI args to skip all permission prompts (YOLO mode).

        Default: return empty list. Override per tool.
        """
        return []

    def accept_edits_args(self) -> list[str]:
        """Get extra CLI args for the accept-edits tier (auto-apply file edits,
        still prompt for shell/commands).

        Default: return empty list. Override per tool. A tool may declare
        ``supports_accept_edits=True`` yet still return ``[]`` when its default
        launch mode already *is* accept-edits (e.g. Cursor CLI).
        """
        return []

    def auto_args(self) -> list[str]:
        """Get extra CLI args for the classifier-mediated auto tier.

        Default: return empty list. Override only on tools that expose a real
        launch-time classifier mode (Claude). Other tools leave
        ``supports_auto=False`` so ``auto`` downgrades to accept-edits.
        """
        return []

    def resolve_effort_model(self, model: str | None, effort: EffortLevel | None) -> str | None:
        """Resolve model variant based on effort level.

        Some tools use different model IDs for higher effort (e.g., thinking
        model variants). Called for every model at launch (``effort`` may be
        ``None``). Default: return model unchanged. Override per tool.
        """
        return model

    def preserve_session_data(self, source_dir: Path, target_dir: Path) -> bool:
        """Preserve AI tool session data from one directory to another.

        Called before removing a temporary working directory so that sessions
        can be resumed from the main project directory after cleanup.

        Default: no-op (return True). Override in tools that store path-bound
        session data.

        Args:
            source_dir: The directory being removed (e.g. a worktree).
            target_dir: The main directory to migrate session data into.

        Returns:
            True if preservation succeeded (or is not needed), False on failure.
        """
        return True

    def session_data_dirs(self) -> list[str]:
        """Return directory names that indicate this tool may have session data.

        Used as a fallback when there is no record of the tool used in a
        directory. If any of these directories exist, this adapter is
        selected for preservation.

        Default: empty list (no detection). Override per tool.
        """
        return []

    def build_resume_command(
        self,
        session_id: str,
        *,
        working_dir: Path | None = None,
        network_access: bool = False,
        sandbox: bool = True,
    ) -> list[str] | None:
        """Build a command to resume a previous session.

        Returns None if the tool does not support session resume.
        Override in adapters that support resume (and set supports_resume=True
        in capabilities).

        ``working_dir``, ``network_access``, and ``sandbox`` carry the same
        sandbox context as :meth:`build_launch_command`. They are keyword-only
        with defaults so every override can accept and ignore them without a
        ``TypeError`` on polymorphic dispatch; only Codex acts on them.
        """
        return None

    # ------------------------------------------------------------------
    # Session-scoped scenes (crossby launch --scene)
    # ------------------------------------------------------------------

    def scene_launch_args(self, scene: SceneLaunchContext) -> SceneLaunchArgs:
        """Render *scene* for one session and return extra argv + env.

        Default: no session-scoped lever — empty argv and env. Adapters that can
        take a scene on the command line (Claude, Codex, Copilot) override this
        to materialise the scene's artefacts under ``scene.launch_dir`` and point
        their CLI at them. Cursor and OpenCode have no launch lever (their only
        config-dir override relocates auth / can be overridden by a project
        config), so they inherit this no-op and fall back to persistent
        ``scene use`` activation. An override must render
        idempotently: :meth:`launch` calls it once per launch, but a library
        consumer may call it directly, and repeated calls must produce the same
        artefacts and the same argv/env.
        """
        from crossby.scenes.launch import SceneLaunchArgs

        return SceneLaunchArgs()

    def build_launch_environment(self, scene: SceneLaunchArgs | None = None) -> dict[str, str]:
        """Return the scene's environment additions for the child process.

        Sibling of :meth:`build_launch_command`: the command builder consumes a
        rendered :class:`SceneLaunchArgs`' argv, this one consumes its env.
        Defaults to ``{}`` (no additions); :meth:`launch` merges the result over
        ``os.environ`` for the child.
        """
        return dict(scene.env) if scene is not None else {}

    def scene_launch_ready(self) -> bool:
        """Whether this tool can apply a scene at launch *right now*.

        Combines the static capability (``supports_scene_launch``) with any
        runtime gate an adapter adds (e.g. Codex's minimum CLI version for
        ``--profile``). ``False`` tells ``cli/launch`` to fall back to persistent
        ``scene use`` activation for this tool rather than emitting launch flags.
        """
        return self.capabilities().supports_scene_launch

    def scene_launch_concerns(self) -> set[str]:
        """Scene concerns this tool can scope at launch (a subset of SCENE_CONCERNS).

        Used by ``cli/launch`` to warn when a scene narrows a concern this tool
        has no launch lever for (e.g. a Cursor launch of a scene that filters
        only agents) rather than silently applying nothing for it. Default:
        empty — adapters that support scene launch override with their real set
        (typically ``{"mcp"}``, plus ``skills``/``agents`` for Claude).
        """
        return set()

    @staticmethod
    def _requested_plan_approval(
        *, yolo: bool, auto: bool, accept_edits: bool
    ) -> PlanLaunchApprovalMode | None:
        """Use the same precedence as ordinary launches without changing Plan mode."""
        if yolo:
            return PlanLaunchApprovalMode.YOLO
        if auto:
            return PlanLaunchApprovalMode.AUTO
        if accept_edits:
            return PlanLaunchApprovalMode.ACCEPT_EDITS
        return None

    def plan_approval_args(self, mode: PlanLaunchApprovalMode) -> list[str]:
        """Compose only combinations declared by the native launch capability.

        Override when ordinary approval flags would replace the Plan selector.
        Collector approval policies remain an independent contract.
        """
        return {
            PlanLaunchApprovalMode.YOLO: self.yolo_args,
            PlanLaunchApprovalMode.AUTO: self.auto_args,
            PlanLaunchApprovalMode.ACCEPT_EDITS: self.accept_edits_args,
        }[mode]()

    def _autonomy_launch_args(
        self,
        caps: AIToolCapabilities,
        *,
        yolo: bool,
        auto: bool,
        accept_edits: bool,
        plan_mode: bool,
    ) -> list[str]:
        """Resolve the autonomy/permission-mode CLI args via one precedence chain.

        Ladder, most permissive first: ``yolo`` > ``auto`` > ``accept_edits`` >
        default prompting. Native Plan launches use their separately declared
        approval combinations, retaining the Plan selector and never downgrading.
        The highest requested autonomy tier that the tool supports wins. When
        the tool lacks a requested tier the request downgrades to the next lower
        autonomy tier it supports (never escalating), emitting a one-line
        warning. Downgrades stop at default prompting.

        The cascade collapses the requested flags to the single highest tier and
        walks *down* from there, so it relies on the capability invariant that a
        tool supporting a higher tier also supports every lower one (auto ⇒ yolo
        and accept_edits). All adapters honor this today; a violation would let a
        request land on a tier the user never asked for. ``TestCapabilityInvariants``
        in ``test_autonomy_modes.py`` guards it.
        """
        if plan_mode:
            requested_plan = self._requested_plan_approval(
                yolo=yolo, auto=auto, accept_edits=accept_edits
            )
            return self.plan_mode_args() + (
                self.plan_approval_args(requested_plan) if requested_plan is not None else []
            )

        _tier_labels = {"yolo": "YOLO", "auto": "classifier auto", "accept_edits": "accept-edits"}

        requested = "yolo" if yolo else "auto" if auto else "accept_edits" if accept_edits else None
        if requested is None:
            return []

        tiers = ("yolo", "auto", "accept_edits")
        supports = {
            "yolo": caps.supports_yolo,
            "auto": caps.supports_auto,
            "accept_edits": caps.supports_accept_edits,
        }
        args_fns = {
            "yolo": self.yolo_args,
            "auto": self.auto_args,
            "accept_edits": self.accept_edits_args,
        }

        for tier in tiers[tiers.index(requested) :]:
            if supports[tier]:
                if tier != requested:
                    warnings.warn(
                        f"{caps.display_name} does not support {_tier_labels[requested]} mode; "
                        f"downgrading to {_tier_labels[tier]}.",
                        UserWarning,
                        stacklevel=3,
                    )
                return args_fns[tier]()

        # No autonomy tier is supported. Degrade to default prompting.
        # Explicit Plan launches are handled above.
        warnings.warn(
            f"{caps.display_name} does not support {_tier_labels[requested]} mode; "
            "using default prompting.",
            UserWarning,
            stacklevel=3,
        )
        return []

    def build_launch_command(
        self,
        model: str | None = None,
        prompt: str | None = None,
        plan_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
        trusted_dirs: list[str] | None = None,
        initial_message: str | None = None,
        effort: EffortLevel | None = None,
        allowed_commands: list[str] | None = None,
        yolo: bool = False,
        accept_edits: bool = False,
        auto: bool = False,
        scene: SceneLaunchArgs | None = None,
        working_dir: Path | None = None,
        network_access: bool = False,
        plan_output_dir: Path | None = None,
        *,
        sandbox: bool = True,
        _skip_plan_version_check: bool = False,
    ) -> list[str]:
        """Build the command line for launching this tool.

        ``scene`` carries the argv an adapter's ``scene_launch_args`` rendered
        for a session-scoped scene; its ``args`` are appended last. When it is
        ``None`` (every non-scene launch) the command is byte-identical to
        before, so the existing argv-assertion tests need no changes.

        ``working_dir``, ``network_access``, and ``sandbox`` are sandbox context
        forwarded to :meth:`sandbox_config_args`. Only adapters declaring
        ``supports_sandbox_toggle`` may translate ``sandbox`` into a selection
        flag; unsupported adapters retain their existing trusted-directory
        composition, including legacy hook overrides.
        """
        self.validate_plan_mode_request(
            plan_mode=plan_mode,
            yolo=yolo,
            auto=auto,
            accept_edits=accept_edits,
            initial_message=initial_message or prompt,
            plan_output_dir=plan_output_dir,
            working_dir=working_dir,
            verify_version=not _skip_plan_version_check,
            interactive=prompt is None,
        )

        caps = self.capabilities()
        cmd = [caps.binary]

        # Initial message comes first so it is the first positional arg seen by
        # the tool's parser (before any flags that could interfere).
        terminal_plan = plan_mode and caps.plan_mode.activation is PlanModeActivation.TERMINAL_INPUT
        if initial_message and not terminal_plan:
            cmd.extend(self.initial_message_args(initial_message))

        # Resolve effort-based model variant before applying model flag. Called
        # for every model (effort may be None): tools that bake effort into the
        # ID (Cursor, antigravity-cli) need to translate even when no separate
        # effort is passed — e.g. a bare agy Gemini model requires a baked effort.
        effective_model = model
        if effective_model:
            effective_model = self.resolve_effort_model(effective_model, effort)

        # Cross-provider translation: if the user passed a model id that
        # belongs to another provider's family (Claude → Codex or Codex →
        # Claude), try to translate via the canonical family mappings rather
        # than handing the tool an id it will reject. Other tools that
        # accept arbitrary model ids (Cursor, Copilot, OpenCode) skip this
        # branch via is_model_compatible() returning True.
        if effective_model and not self.is_model_compatible(effective_model):
            translated = _maybe_translate_cross_provider(effective_model, self.TOOL_ID, effort)
            if translated is not None:
                translated_model, translated_effort = translated
                effort_note = (
                    f" (effort {effort.value} → {translated_effort.value})"
                    if effort and translated_effort and translated_effort != effort
                    else ""
                )
                warnings.warn(
                    f"Translating model {effective_model!r} → {translated_model!r} "
                    f"for {caps.display_name}{effort_note}; "
                    f"pass --model with a native id to skip this translation.",
                    UserWarning,
                    stacklevel=2,
                )
                effective_model = translated_model
                if translated_effort is not None:
                    effort = translated_effort

        if effective_model and caps.supports_model_flag:
            cmd.extend([caps.model_flag, self.normalize_model_format(effective_model)])

        if prompt and caps.supports_headless and caps.headless_flag:
            cmd.extend([caps.headless_flag, prompt])

        autonomy_args = self._autonomy_launch_args(
            caps,
            yolo=yolo,
            auto=auto,
            accept_edits=accept_edits,
            plan_mode=plan_mode,
        )
        cmd.extend(autonomy_args)

        if plan_output_dir is not None:
            if working_dir is None:
                raise PlanModeAdapterContractError(
                    f"{caps.display_name} requires working_dir to route native plan artifacts.",
                    tool_id=self.TOOL_ID,
                    capability=caps.plan_mode,
                )
            output_args = self.plan_output_args(plan_output_dir, working_dir=working_dir)
            if not output_args:
                raise PlanModeAdapterContractError(
                    f"{caps.display_name} advertises requested-path plan artifacts but emitted no "
                    "routing arguments. Update Crossby before retrying.",
                    tool_id=self.TOOL_ID,
                    capability=caps.plan_mode,
                )
            cmd.extend(output_args)

        effective_trusted_dirs = list(trusted_dirs or [])

        if json_schema:
            cmd.extend(self.structured_output_args(json_schema))

        # One hook owns sandbox mode + writable roots + trusted dirs + network,
        # so the mode is emitted once, before any --add-dir. Default reproduces
        # the old trusted-dir emission; capability-enabled adapters override it.
        if caps.supports_sandbox_toggle:
            sandbox_args = self.sandbox_config_args(
                autonomy_args=autonomy_args,
                trusted_dirs=effective_trusted_dirs or None,
                working_dir=working_dir,
                network_access=network_access,
                sandbox=sandbox,
            )
        else:
            # Preserve pre-toggle adapter overrides, whose signatures do not
            # accept ``sandbox``. Capability-enabled overrides gate selection
            # themselves when their capability is disabled dynamically.
            sandbox_args = self.sandbox_config_args(
                autonomy_args=autonomy_args,
                trusted_dirs=effective_trusted_dirs or None,
                working_dir=working_dir,
                network_access=network_access,
            )
        cmd.extend(sandbox_args)

        # Effort args (tool-specific flags like --settings, --variant, etc.)
        if effort and caps.supports_effort:
            cmd.extend(self.effort_args(effort))

        if allowed_commands:
            cmd.extend(self.allowed_commands_args(allowed_commands))

        # Scene argv is appended last so it never interferes with the parser's
        # reading of the positional initial message or the tool/model flags.
        if scene is not None:
            cmd.extend(scene.args)

        cmd = self._finalize_launch_command(cmd)
        if terminal_plan:
            return self._wrap_terminal_plan_command(cmd, initial_message)
        return cmd

    def _finalize_launch_command(self, cmd: list[str]) -> list[str]:
        """Let an adapter reconcile arguments that must form one logical source."""
        return cmd


def _maybe_translate_cross_provider(
    model: str,
    tool_id: AIToolID,
    effort: EffortLevel | None,
) -> tuple[str, EffortLevel | None] | None:
    """Apply Claude↔Codex family mapping when ``model`` isn't native to
    ``tool_id``.

    Returns ``(translated_model, translated_effort)`` when a translation is
    known, ``None`` otherwise. Effort is family-biased (Sonnet shifts up
    one tier on the way to Codex, reverse picks the lowest source tier).

    Imports the translation helpers lazily to avoid a circular dependency
    via ``crossby.sync`` package init.
    """
    from crossby.sync.translation import (
        find_claude_family,
        map_effort_claude_to_codex,
        map_effort_codex_to_claude,
        map_model_claude_to_codex,
        map_model_codex_to_claude,
    )

    # Claude model → Codex family
    if tool_id == AIToolID.CODEX and find_claude_family(model) is not None:
        translated_effort = map_effort_claude_to_codex(model, effort) if effort else None
        return map_model_claude_to_codex(model), translated_effort

    # Codex/GPT model → Claude family
    if tool_id == AIToolID.CLAUDE and model.startswith("gpt-"):
        target = map_model_codex_to_claude(model)
        translated_effort = map_effort_codex_to_claude(model, target, effort) if effort else None
        return target, translated_effort

    return None


def pick_best_model(models: list[AIModel]) -> AIModel | None:
    """Pick the best model from a list — prefer aliases (no date suffix)."""
    if not models:
        return None

    # Prefer models without date suffix (alias models)
    aliases = [m for m in models if m.is_alias]
    if aliases:
        return aliases[0]

    # Fallback: sort by ID and take the last (newest)
    return sorted(models, key=lambda m: m.id)[-1]
