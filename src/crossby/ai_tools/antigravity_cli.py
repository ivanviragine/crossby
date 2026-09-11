"""Antigravity CLI (agy) adapter — terminal agent, distinct from the Antigravity IDE."""

from __future__ import annotations

import json
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any, ClassVar

from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.plan_mode import PlanInteractionHandler
from crossby.models.ai import (
    AIToolCapabilities,
    AIToolID,
    AIToolType,
    EffortLevel,
    HookOutputDialect,
    HookStopDialect,
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

# agy bakes reasoning effort into the model ID and rejects a separate --effort on
# an already-suffixed model, while a bare Gemini base model *requires* an effort.
# Only these base families require/encode effort (verified via `agy models` + live
# probing); every other catalog model launches bare and ignores effort. Per-model
# tiers differ: the flash families accept low/medium/high, gemini-3.1-pro only low/high.
_ANTIGRAVITY_CLI_EFFORT_TIERS: dict[str, tuple[EffortLevel, ...]] = {
    "gemini-3.8-flash": (EffortLevel.LOW, EffortLevel.MEDIUM, EffortLevel.HIGH),
    "gemini-3.7-flash": (EffortLevel.LOW, EffortLevel.MEDIUM, EffortLevel.HIGH),
    "gemini-3.6-flash": (EffortLevel.LOW, EffortLevel.MEDIUM, EffortLevel.HIGH),
    "gemini-3.5-flash": (EffortLevel.LOW, EffortLevel.MEDIUM, EffortLevel.HIGH),
    "gemini-3.1-pro": (EffortLevel.LOW, EffortLevel.HIGH),
}

# These suffixes are part of the exact provider model ID, not a reasoning
# effort that Crossby may rewrite. Keep this separate from the Gemini effort
# table so a caller-supplied effort remains irrelevant for these fixed IDs.
_ANTIGRAVITY_CLI_FIXED_SUFFIX_MODELS = frozenset({"gpt-oss-120b-medium"})

# Effort suffixes that may appear on a stored model ID. agy only *emits* the
# low/medium/high tiers, but a hand-written config could carry an ``-xhigh``/
# ``-max`` suffix agy rejects; recognizing them lets resolve_effort_model
# normalize such an ID down to a valid tier instead of passing it through.
_EFFORT_SUFFIXES: dict[str, EffortLevel] = {
    "-low": EffortLevel.LOW,
    "-medium": EffortLevel.MEDIUM,
    "-high": EffortLevel.HIGH,
    "-xhigh": EffortLevel.XHIGH,
    "-max": EffortLevel.MAX,
}


def _split_effort_suffix(model: str) -> tuple[str, EffortLevel | None]:
    """Split a trailing effort suffix (``-low``/``-medium``/``-high``/``-xhigh``/
    ``-max``) off a model ID.

    Returns ``(base, effort)`` when the ID ends in a known effort suffix, else
    ``(model, None)``.
    """
    for suffix, level in _EFFORT_SUFFIXES.items():
        if model.endswith(suffix):
            return model[: -len(suffix)], level
    return model, None


def _nearest_tier(effort: EffortLevel, tiers: tuple[EffortLevel, ...]) -> EffortLevel:
    """Closest supported tier to ``effort`` by ``EffortLevel`` ordinal distance.

    Ties resolve toward the *higher* tier.
    """
    order = list(EffortLevel)
    target = order.index(effort)
    return min(tiers, key=lambda t: (abs(order.index(t) - target), -order.index(t)))


def _default_effort(tiers: tuple[EffortLevel, ...]) -> EffortLevel:
    """Deterministic effort when none is supplied: ``medium`` when the model
    supports it, otherwise the tier nearest to medium (so gemini-3.1-pro → high)."""
    if EffortLevel.MEDIUM in tiers:
        return EffortLevel.MEDIUM
    return _nearest_tier(EffortLevel.MEDIUM, tiers)


class AntigravityCLIAdapter(AbstractAITool):
    """Adapter for Antigravity CLI (``agy``), the terminal surface of Google
    Antigravity 2.0. Not to be confused with ``AntigravityAdapter``, which
    launches the Antigravity IDE."""

    TOOL_ID: ClassVar[AIToolID] = AIToolID.ANTIGRAVITY_CLI

    def capabilities(self) -> AIToolCapabilities:
        return AIToolCapabilities(
            tool_id=AIToolID.ANTIGRAVITY_CLI,
            display_name="Antigravity CLI",
            binary="agy",
            tool_type=AIToolType.TERMINAL,
            # `agy update` — "Update CLI" per the tool's own help.
            update_command=("agy", "update"),
            supports_model_flag=True,
            # -p/--print/--prompt run a single prompt non-interactively and exit.
            headless_flag="--print",
            supports_headless=True,
            supports_effort=True,
            # agy --effort accepts only low|medium|high; xhigh/max are rejected.
            supported_efforts=(EffortLevel.LOW, EffortLevel.MEDIUM, EffortLevel.HIGH),
            supports_yolo=True,
            supports_resume=True,
            supports_trusted_dirs=True,
            plan_mode=PlanModeCapability(
                activation=PlanModeActivation.CLI_ARGUMENT,
                activation_detail="Passes --mode plan before the first user turn.",
                version_requirement="Antigravity CLI exposing --mode plan.",
                verified_version="1.2.0",
                initial_prompt_after_activation=True,
                artifact_location=PlanArtifactLocation.SESSION,
                artifact_location_detail=(
                    "Crossby accepts only schema-validated structured_output bound to the "
                    "conversation ID emitted by this invocation."
                ),
                remediation=(
                    "Use run_plan_session() for structured collection, or choose Claude when a "
                    "specific filesystem output directory is required."
                ),
                transport=PlanSessionTransport.HEADLESS_CLI,
                artifact_source=PlanArtifactSource.STRUCTURED_OUTPUT,
                binding=PlanSessionBinding.CONVERSATION_ID,
                interaction=PlanInteractionSupport.RESUMABLE_CALLBACK,
                sandbox_behavior=PlanRequestBehavior.TOOL_MANAGED,
                approval_behavior=PlanRequestBehavior.TOOL_MANAGED,
            ),
            supports_accept_edits=True,
            # agy exposes a Claude-style hook system (PreToolUse/PostToolUse/
            # Pre/PostInvocation/Stop). It reads decisions as a top-level
            # {"decision": …} object (the DECISION dialect) and fails *closed*
            # on a PreToolUse hook that errors — a non-zero exit denies the tool
            # call (observed in agy integrations, e.g. cmux issue #4768 where a
            # failing PreToolUse hook blocks every tool call) — so
            # hook_fail_open_default stays False.
            supports_stop_hook=True,
            hook_output_dialect=HookOutputDialect.DECISION,
            # Inverted polarity vs every other tool: agy blocks a Stop by
            # telling the agent to *continue*. Its PreToolUse vocabulary is
            # allow/deny/ask/force_ask — "continue" is a Stop-only word there,
            # and "block" is not a word agy knows at all (it errors with
            # `unknown pre-tool hook decision "block"`).
            hook_stop_dialect=HookStopDialect.CONTINUE_DECISION,
            hook_fail_open_default=False,
            # sandboxes_writes stays False deliberately: agy exposes no verified
            # write-confinement mechanism (its ``--sandbox`` flag is a *terminal*
            # restriction, not a write jail, and crossby no longer emits it), so we
            # do NOT tell wade an out-of-worktree write is already confined — wade
            # keeps its own containment guard rather than trusting a native sandbox.
            # agy's own bundled plugin registers no PreToolUse hook, so that guard
            # is best-effort there; Stop is the reliable enforcement surface.
            sandboxes_writes=False,
        )

    def initial_message_args(self, prompt: str) -> list[str]:
        """``--prompt-interactive`` runs an initial prompt interactively and
        continues the session — the interactive-launch equivalent of an
        initial message."""
        return ["--prompt-interactive", prompt]

    def plan_mode_args(self) -> list[str]:
        """agy's ``--mode`` flag accepts ``accept-edits`` or ``plan``."""
        return ["--mode", "plan"]

    def _run_plan_session(
        self,
        request: PlanSessionRequest,
        version: str,
        interaction_handler: PlanInteractionHandler | None,
    ) -> PlanSessionResult:
        """Collect schema-validated output from one exact agy conversation."""
        from crossby.ai_tools.plan_mode import (
            PlanArtifactMalformedError,
            PlanBindingMismatchError,
            PlanInteractionRequiredError,
            PlanSessionUnsupportedError,
            PlanTransportError,
        )
        from crossby.ai_tools.plan_process import run_captured

        capability = self.capabilities().plan_mode
        if request.effort is not None and not _agy_model_encodes_effort(
            request.model, request.effort
        ):
            model = request.model or "no explicit model"
            raise PlanSessionUnsupportedError(
                f"Antigravity CLI cannot preserve effort={request.effort.value!r} with "
                f"{model!r}. Supply a compatible Gemini model/tier or omit effort.",
                tool_id=self.TOOL_ID,
                capability=capability,
            )

        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"plan": {"type": "string", "minLength": 1}},
            "required": ["plan"],
            "additionalProperties": False,
        }
        command = [
            "agy",
            "--print",
            request.prompt,
            "--mode",
            "plan",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema, separators=(",", ":")),
        ]
        effective_model = self.resolve_effort_model(request.model, request.effort)
        if effective_model:
            command.extend(("--model", effective_model))
        for path in request.trusted_dirs:
            command.extend(("--add-dir", str(path)))

        conversation_id: str | None = None
        response: dict[str, Any] | None = None
        deadline = time.monotonic() + request.timeout_seconds
        for _continuation in range(9):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PlanTransportError(
                    "Antigravity CLI plan session exceeded its timeout.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=conversation_id,
                )
            try:
                run = run_captured(
                    command,
                    cwd=request.working_dir,
                    timeout=remaining,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise PlanTransportError(
                    f"Antigravity CLI plan process failed: {exc}",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    session_id=conversation_id,
                ) from exc
            if run.returncode != 0:
                raise PlanTransportError(
                    f"Antigravity CLI exited with status {run.returncode}.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    exit_code=run.returncode,
                    session_id=conversation_id,
                    stderr=run.stderr,
                )
            try:
                loaded = json.loads(run.stdout)
            except json.JSONDecodeError as exc:
                raise PlanArtifactMalformedError(
                    f"Antigravity CLI output was not valid JSON: {exc.msg}",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    exit_code=run.returncode,
                    session_id=conversation_id,
                ) from exc
            if not isinstance(loaded, dict):
                raise PlanArtifactMalformedError(
                    "Antigravity CLI JSON output root was not an object.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    exit_code=run.returncode,
                    session_id=conversation_id,
                )
            response = loaded
            emitted_id = _agy_conversation_id(response)
            if not emitted_id:
                raise PlanArtifactMalformedError(
                    "Antigravity CLI output omitted conversation_id.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    exit_code=run.returncode,
                    session_id=conversation_id,
                )
            if conversation_id is not None and emitted_id != conversation_id:
                raise PlanBindingMismatchError(
                    "Antigravity CLI continuation changed the captured conversation ID.",
                    tool_id=self.TOOL_ID,
                    capability=capability,
                    exit_code=run.returncode,
                    session_id=conversation_id,
                )
            conversation_id = emitted_id
            if not _agy_waiting(response):
                break
            interaction = _agy_interaction(response, conversation_id)
            if interaction_handler is None:
                raise PlanInteractionRequiredError(
                    "Antigravity CLI requires an answer to continue the planning conversation.",
                    interaction=interaction,
                    tool_id=self.TOOL_ID,
                    capability=capability,
                )
            answer_response = interaction_handler(interaction)
            answer = (
                answer_response.answer
                or answer_response.option_id
                or ", ".join(answer_response.option_ids)
                or None
            )
            if (
                answer_response.outcome
                in {
                    PlanInteractionOutcome.DENIED,
                    PlanInteractionOutcome.CANCELLED,
                    PlanInteractionOutcome.SKIPPED,
                }
                or not answer
            ):
                raise PlanInteractionRequiredError(
                    "Antigravity CLI planning question was left unanswered.",
                    interaction=interaction,
                    tool_id=self.TOOL_ID,
                    capability=capability,
                )
            command = [
                "agy",
                "--conversation",
                conversation_id,
                "--print",
                answer,
                "--output-format",
                "json",
                "--json-schema",
                json.dumps(schema, separators=(",", ":")),
            ]
            if effective_model:
                command.extend(("--model", effective_model))
            for path in request.trusted_dirs:
                command.extend(("--add-dir", str(path)))
        else:
            raise PlanTransportError(
                "Antigravity CLI exceeded the bounded question-continuation limit.",
                tool_id=self.TOOL_ID,
                capability=capability,
                session_id=conversation_id,
            )

        assert response is not None and conversation_id is not None
        if not _agy_success(response):
            raise PlanTransportError(
                f"Antigravity CLI ended with terminal status {response.get('status')!r}.",
                tool_id=self.TOOL_ID,
                capability=capability,
                exit_code=0,
                session_id=conversation_id,
            )
        schema_echo = response.get("schema") or response.get("json_schema")
        if isinstance(schema_echo, str):
            try:
                schema_echo = json.loads(schema_echo)
            except json.JSONDecodeError:
                schema_echo = None
        if schema_echo != schema:
            raise PlanArtifactMalformedError(
                "Antigravity CLI did not echo the requested plan schema exactly.",
                tool_id=self.TOOL_ID,
                capability=capability,
                exit_code=0,
                session_id=conversation_id,
            )
        structured = response.get("structured_output")
        plan = structured.get("plan") if isinstance(structured, dict) else None
        if (
            not isinstance(structured, dict)
            or set(structured) != {"plan"}
            or not isinstance(plan, str)
            or not plan.strip()
        ):
            raise PlanArtifactMalformedError(
                "Antigravity CLI structured_output did not match the requested plan schema.",
                tool_id=self.TOOL_ID,
                capability=capability,
                exit_code=0,
                session_id=conversation_id,
            )
        raw_artifact_id = response.get("artifact_id")
        artifact_id = str(raw_artifact_id).strip() or None if raw_artifact_id is not None else None
        return PlanSessionResult(
            tool=self.TOOL_ID,
            version=version,
            plan=plan,
            session_id=conversation_id,
            native_mode="--mode plan",
            artifact_source=PlanArtifactSource.STRUCTURED_OUTPUT,
            binding=PlanSessionBinding.CONVERSATION_ID,
            exit_code=0,
            artifact_id=artifact_id,
        )

    def accept_edits_args(self) -> list[str]:
        """agy's ``--mode accept-edits`` auto-applies edits on the execution-mode
        axis; shell stays gated by the separate permissions axis."""
        return ["--mode", "accept-edits"]

    def plan_dir_args(self, plan_dir: str) -> list[str]:
        """agy uses --add-dir (repeatable) to grant workspace access."""
        return ["--add-dir", plan_dir]

    def yolo_args(self) -> list[str]:
        """Auto-approve all tool permission requests without prompting.

        Only ``--dangerously-skip-permissions`` — agy's ``--sandbox`` is a
        *terminal-restriction* flag ("run in a sandbox with terminal restrictions
        enabled"), NOT a write sandbox, and pairing it with skip-permissions blocks
        every shell command. wade does not rely on agy for write confinement
        (``sandboxes_writes=False``); it keeps its own worktree-containment guard.
        """
        return ["--dangerously-skip-permissions"]

    def build_resume_command(
        self,
        session_id: str,
        *,
        working_dir: Path | None = None,
        network_access: bool = False,
        sandbox: bool = True,
    ) -> list[str] | None:
        """Resume a specific Antigravity CLI conversation by ID.

        Accepts and ignores the sandbox context (agy has no writable-root
        mechanism to configure at resume — crossby emits no sandbox flag); the
        keyword-only params keep polymorphic dispatch TypeError-free.
        """
        return ["agy", "--conversation", session_id]

    def resolve_effort_model(self, model: str | None, effort: EffortLevel | None) -> str | None:
        """Bake reasoning effort into the model ID — the single form ``agy`` accepts.

        ``agy`` rejects a separate ``--effort`` on a suffixed model and *requires*
        an effort on a bare Gemini model, so effort is encoded in the ID (the
        Cursor ``-thinking`` pattern) and no ``--effort`` flag is ever emitted.

        Covers, in one pass:

        - **Fixed/bare models** (``claude-*``, ``gpt-oss-120b*``): returned
          unchanged — effort does not apply. ``gpt-oss-120b-medium`` is an
          exact provider ID whose suffix is part of its name, not an effort.
        - **Precedence**: an effort already baked into the ID wins over a
          separately supplied ``effort`` (agy would reject the two together).
        - **No effort anywhere**: a deterministic default is baked in so the
          command is valid (``gemini-3.8-flash`` → ``…-medium``, ``gemini-3.1-pro``
          → ``…-high``) rather than the rejected bare base model.
        - **xhigh/max**: normalized to ``high`` (agy rejects them), with a warning.
        - **Per-model gap / invalid stored suffix** (``gemini-3.1-pro-medium``):
          snapped to the nearest valid tier (ties → higher), with a warning.
        """
        if not model:
            # No --model to bake effort into (e.g. effort supplied with no model).
            return model

        if model in _ANTIGRAVITY_CLI_FIXED_SUFFIX_MODELS:
            return model

        base, suffix_effort = _split_effort_suffix(model)
        tiers = _ANTIGRAVITY_CLI_EFFORT_TIERS.get(base)
        if tiers is None:
            # Unknown non-Gemini suffixes are treated as invalid stored effort
            # values and repaired to their bare base. Exact fixed-suffix model
            # IDs returned above never enter this compatibility path.
            if suffix_effort is not None:
                dropped = f"-{suffix_effort.value}"
                warnings.warn(
                    f"Antigravity CLI model {base!r} does not accept a reasoning "
                    f"effort; dropping the {dropped!r} suffix and launching bare.",
                    UserWarning,
                    stacklevel=2,
                )
                return base
            # Bare model (claude-*, gpt-oss-120b): agy launches it with no effort.
            return model

        # A suffix on the model ID wins over a separately supplied effort.
        eff = suffix_effort if suffix_effort is not None else effort
        if eff is None:
            eff = _default_effort(tiers)

        if eff in (EffortLevel.XHIGH, EffortLevel.MAX):
            warnings.warn(
                f"Antigravity CLI accepts only low/medium/high effort; "
                f"normalizing {eff.value!r} to 'high'.",
                UserWarning,
                stacklevel=2,
            )
            eff = EffortLevel.HIGH

        if eff not in tiers:
            snapped = _nearest_tier(eff, tiers)
            available = ", ".join(t.value for t in tiers)
            warnings.warn(
                f"Antigravity CLI model {base!r} has no {eff.value!r} effort tier "
                f"(available: {available}); snapping to {snapped.value!r}.",
                UserWarning,
                stacklevel=2,
            )
            eff = snapped

        return f"{base}-{eff.value}"

    def parse_transcript(self, transcript_path: Path) -> TokenUsage:
        """Antigravity CLI persists conversations as opaque per-conversation
        SQLite databases (protobuf-like blob columns) under
        ``~/.gemini/antigravity-cli/conversations/`` — verified locally via
        ``agy --print`` + inspecting that directory, this is the real
        install path (Antigravity CLI is a Gemini-family product, hence the
        ``~/.gemini/`` prefix), not a leftover Gemini-CLI reference. Not
        parseable text, so this mirrors the known Gemini-CLI
        transcript-persistence limitation for a different underlying reason."""
        return TokenUsage()


def _agy_model_encodes_effort(model: str | None, effort: EffortLevel) -> bool:
    """Whether *model* can preserve an explicit collected-session effort."""
    if model is None or model in _ANTIGRAVITY_CLI_FIXED_SUFFIX_MODELS:
        return False
    base, suffix_effort = _split_effort_suffix(model)
    tiers = _ANTIGRAVITY_CLI_EFFORT_TIERS.get(base)
    return (
        tiers is not None and effort in tiers and (suffix_effort is None or suffix_effort is effort)
    )


def _agy_conversation_id(payload: dict[str, Any]) -> str | None:
    value = payload.get("conversation_id") or payload.get("conversationId")
    return value if isinstance(value, str) and value.strip() else None


def _agy_waiting(payload: dict[str, Any]) -> bool:
    return _agy_status(payload) in {"waiting", "waiting_for_input", "input_required"} or bool(
        payload.get("waiting_for_input")
    )


def _agy_success(payload: dict[str, Any]) -> bool:
    return _agy_status(payload) in {"success", "completed"} and not bool(payload.get("is_error"))


def _agy_status(payload: dict[str, Any]) -> str | None:
    value = payload.get("status")
    return value.strip().lower() if isinstance(value, str) else None


def _agy_interaction(payload: dict[str, Any], conversation_id: str) -> PlanInteraction:
    raw_question = payload.get("question")
    question = raw_question if isinstance(raw_question, dict) else payload
    prompt = question.get("prompt") or question.get("question") or question.get("message")
    question_id = question.get("id") or question.get("question_id") or "question"
    options = tuple(
        PlanQuestionOption(
            option_id=str(option.get("id") or option.get("label")),
            label=str(option.get("label")),
            description=(
                str(option["description"]) if option.get("description") is not None else None
            ),
        )
        for option in question.get("options") or []
        if isinstance(option, dict) and option.get("label")
    )
    return PlanInteraction(
        kind=PlanInteractionKind.QUESTION,
        question_id=str(question_id),
        prompt=str(prompt or "Antigravity CLI requires input to continue planning."),
        options=options,
        session_id=conversation_id,
    )
