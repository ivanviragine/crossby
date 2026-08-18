"""Antigravity CLI (agy) adapter — terminal agent, distinct from the Antigravity IDE."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import ClassVar

from crossby.ai_tools.base import AbstractAITool
from crossby.models.ai import (
    AIToolCapabilities,
    AIToolID,
    AIToolType,
    EffortLevel,
    HookOutputDialect,
    HookStopDialect,
    TokenUsage,
)

# agy bakes reasoning effort into the model ID and rejects a separate --effort on
# an already-suffixed model, while a bare Gemini base model *requires* an effort.
# Only these base families require/encode effort (verified via `agy models` + live
# probing); every other catalog model launches bare and ignores effort. Per-model
# tiers differ: the flash families accept low/medium/high, gemini-3.1-pro only low/high.
_ANTIGRAVITY_CLI_EFFORT_TIERS: dict[str, tuple[EffortLevel, ...]] = {
    "gemini-3.7-flash": (EffortLevel.LOW, EffortLevel.MEDIUM, EffortLevel.HIGH),
    "gemini-3.6-flash": (EffortLevel.LOW, EffortLevel.MEDIUM, EffortLevel.HIGH),
    "gemini-3.5-flash": (EffortLevel.LOW, EffortLevel.MEDIUM, EffortLevel.HIGH),
    "gemini-3.1-pro": (EffortLevel.LOW, EffortLevel.HIGH),
}

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
            supports_plan_mode=True,
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

        - **Bare models** (``claude-*``, ``gpt-oss-120b``): returned unchanged —
          effort does not apply and agy launches them bare. A stored effort
          suffix on such a base (the retired ``gpt-oss-120b-medium`` catalog ID)
          is spurious and dropped, with a warning, so agy is never handed a
          suffixed ID it rejects.
        - **Precedence**: an effort already baked into the ID wins over a
          separately supplied ``effort`` (agy would reject the two together).
        - **No effort anywhere**: a deterministic default is baked in so the
          command is valid (``gemini-3.7-flash`` → ``…-medium``, ``gemini-3.1-pro``
          → ``…-high``) rather than the rejected bare base model.
        - **xhigh/max**: normalized to ``high`` (agy rejects them), with a warning.
        - **Per-model gap / invalid stored suffix** (``gemini-3.1-pro-medium``):
          snapped to the nearest valid tier (ties → higher), with a warning.
        """
        if not model:
            # No --model to bake effort into (e.g. effort supplied with no model).
            return model

        base, suffix_effort = _split_effort_suffix(model)
        tiers = _ANTIGRAVITY_CLI_EFFORT_TIERS.get(base)
        if tiers is None:
            # Non-Gemini base: agy launches it bare and rejects a baked effort. A
            # stored effort suffix on such a base (e.g. the retired
            # ``gpt-oss-120b-medium`` catalog ID) is spurious — drop it so the
            # command stays valid instead of passing an ID agy would reject.
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
