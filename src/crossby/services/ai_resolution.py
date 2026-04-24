"""Shared AI tool and model resolution logic."""

from __future__ import annotations

import os
from typing import Any

import structlog

from crossby.ai_tools.base import AbstractAITool
from crossby.models.ai import AIToolID, EffortLevel
from crossby.models.config import CommandConfig, CrossbyConfig

logger = structlog.get_logger()

_CUSTOM_OPTION = "Custom…"


def resolve_ai_tool(
    ai_tool: str | None,
    config: CrossbyConfig,
    command: str = "plan",
    *,
    auto_detect: bool = True,
) -> str | None:
    """Resolve AI tool from args -> config -> detection.

    Fallback chain: explicit arg -> command-specific config -> global default
    -> auto-detect (when *auto_detect* is True).

    Set *auto_detect=False* when the caller handles multi-tool selection
    itself (e.g. TTY prompts in implement).
    """
    if ai_tool:
        return ai_tool

    config_tool = config.get_ai_tool(command)
    if config_tool:
        return config_tool

    if auto_detect:
        installed = AbstractAITool.detect_installed()
        if installed:
            return installed[0].value

    return None


def resolve_model(
    model: str | None,
    config: CrossbyConfig,
    command: str = "plan",
    *,
    tool: str | None = None,
    complexity: str | None = None,
    strict: bool = False,
) -> str | None:
    """Resolve model from args -> config -> complexity -> default.

    Fallback chain:
      1. Explicit *model* arg (e.g. ``--model`` CLI flag)
      2. Command-specific config (``ai.<command>.model``)
      3. Complexity-based mapping (``models.<tool>.<complexity>``)
      4. Global default (``ai.default_model``)

    When *tool* is provided, the resolved model is checked for compatibility
    with that tool.  Incompatible models are dropped (returns ``None``).

    When *strict* is True, incompatibility raises ``ValueError`` instead of
    returning ``None``.  Use strict mode for user-explicit CLI flags.
    """
    resolved: str | None = model

    # 2. Command-specific config
    if not resolved:
        cmd_config = config.ai.commands.get(command) if command else None
        if isinstance(cmd_config, CommandConfig) and cmd_config.model:
            resolved = cmd_config.model

    # 3. Complexity-based mapping
    if not resolved and tool and complexity:
        resolved = config.get_complexity_model(tool, complexity)

    # 4. Global default
    if not resolved:
        resolved = config.ai.default_model

    # Compatibility gate
    if resolved and tool:
        try:
            adapter = AbstractAITool.get(tool)
        except (ValueError, KeyError):
            if strict:
                raise
        else:
            if strict and not adapter.capabilities().supports_model_flag:
                raise ValueError(
                    f"Tool '{tool}' does not support explicit model selection"
                )
            if not adapter.is_model_compatible(resolved):
                if strict:
                    raise ValueError(f"Model '{resolved}' is not compatible with {tool}")
                logger.info(
                    "model.incompatible",
                    model=resolved,
                    tool=tool,
                )
                return None

    return resolved


def resolve_effort(
    effort: str | None,
    config: CrossbyConfig,
    command: str = "plan",
    *,
    tool: str | None = None,
    strict: bool = False,
) -> EffortLevel | None:
    """Resolve effort level from args -> env var -> config -> None.

    Fallback chain:
      1. Explicit *effort* arg (e.g. ``--effort`` CLI flag)
      2. ``CROSSBY_EFFORT`` environment variable
      3. Command-specific config (``ai.<command>.effort``)
      4. Global config (``ai.effort``)

    When *tool* is provided and the tool does not support effort, a warning
    is logged and ``None`` is returned.

    When *strict* is True, invalid levels or unsupported tools raise
    ``ValueError``.  Use strict mode for user-explicit CLI flags.
    """
    resolved: str | None = effort

    if not resolved:
        resolved = os.environ.get("CROSSBY_EFFORT")

    if not resolved:
        resolved = config.get_effort(command)

    if not resolved:
        return None

    # Validate
    try:
        level = EffortLevel(resolved)
    except ValueError as exc:
        if strict:
            raise ValueError(f"Invalid effort level: '{resolved}'") from exc
        logger.warning("effort.invalid_level", effort=resolved)
        return None

    # Check tool support
    if tool:
        try:
            adapter = AbstractAITool.get(tool)
        except (ValueError, KeyError):
            if strict:
                raise
        else:
            if not adapter.capabilities().supports_effort:
                if strict:
                    raise ValueError(f"{tool} does not support effort levels")
                logger.info("effort.unsupported_tool", tool=tool, effort=resolved)
                return None

    return level


def resolve_yolo(
    yolo: bool | None,
    config: CrossbyConfig,
    command: str = "plan",
    *,
    tool: str | None = None,
    strict: bool = False,
) -> bool:
    """Resolve YOLO mode from args -> config -> False.

    Fallback chain:
      1. Explicit *yolo* arg (e.g. ``--yolo`` CLI flag)
      2. Command-specific config (``ai.<command>.yolo``)
      3. Global config (``ai.yolo``)

    When *tool* is provided and the tool does not support YOLO, a warning
    is logged and ``False`` is returned.

    When *strict* is True, unsupported tools raise ``ValueError``.
    Use strict mode for user-explicit CLI flags.
    """
    resolved: bool | None = yolo

    if resolved is None:
        resolved = config.get_yolo(command)

    if not resolved:
        return False

    # Check tool support
    if tool:
        try:
            adapter = AbstractAITool.get(tool)
        except (ValueError, KeyError):
            if strict:
                raise
        else:
            if not adapter.capabilities().supports_yolo:
                if strict:
                    raise ValueError(f"{tool} does not support YOLO mode")
                logger.warning("yolo.unsupported_tool", tool=tool)
                return False

    return True


def confirm_ai_selection(
    resolved_tool: str | None,
    resolved_model: str | None,
    *,
    tool_explicit: bool,
    model_explicit: bool,
    resolved_effort: EffortLevel | None = None,
    effort_explicit: bool = False,
    resolved_yolo: bool = False,
    yolo_explicit: bool = True,
) -> tuple[str | None, str | None, EffortLevel | None, bool]:
    """Interactively confirm (and optionally change) the resolved AI tool/model/effort/yolo.

    Fires only when stdin is a TTY and at least one of the flags was not
    explicitly provided by the caller. When all flags are explicit, this
    is a no-op.

    Returns the (tool, model, effort, yolo) tuple after any user-driven changes.
    """
    from crossby.services.confirm import ConfirmField, confirm_defaults

    # No tool resolved → nothing to confirm; the caller handles the error.
    if resolved_tool is None:
        return resolved_tool, resolved_model, resolved_effort, resolved_yolo

    # Preserve non-interactive and all-explicit fast paths before any adapter
    # detection, since detect_installed() probes every registered tool.
    if not os.isatty(0):
        return resolved_tool, resolved_model, resolved_effort, resolved_yolo

    if tool_explicit and model_explicit and effort_explicit and yolo_explicit:
        return resolved_tool, resolved_model, resolved_effort, resolved_yolo

    installed = AbstractAITool.detect_installed()

    def _caps_for(tool_value: str | None) -> tuple[bool, bool]:
        """Return ``(supports_effort, supports_yolo)`` for *tool_value*."""
        if not tool_value:
            return False, False
        try:
            adapter = AbstractAITool.get(AIToolID(tool_value))
        except (ValueError, KeyError):
            return False, False
        caps = adapter.capabilities()
        return caps.supports_effort, caps.supports_yolo

    def _tool_change(current: Any, state: dict[str, Any]) -> dict[str, Any]:
        tool_names = [str(t) for t in installed]
        current_idx = tool_names.index(current) if current in tool_names else 0
        new_idx = _select_tool(tool_names, current_idx)
        new_tool = tool_names[new_idx]
        if new_tool == current:
            return {"tool": current}
        new_model = _prompt_model_selection(new_tool)
        supports_effort, supports_yolo = _caps_for(new_tool)
        updates: dict[str, Any] = {"tool": new_tool, "model": new_model}
        if state.get("effort") is not None and not supports_effort:
            updates["effort"] = None
        if state.get("yolo") and not supports_yolo:
            updates["yolo"] = False
        return updates

    def _model_change(_current: Any, state: dict[str, Any]) -> dict[str, Any]:
        return {"model": _prompt_model_selection(state["tool"])}

    def _effort_change(current: Any, _state: dict[str, Any]) -> dict[str, Any]:
        return {"effort": _prompt_effort_selection(current)}

    def _yolo_change(current: Any, _state: dict[str, Any]) -> dict[str, Any]:
        return {"yolo": not bool(current)}

    fields = [
        ConfirmField(
            name="tool",
            label="AI tool",
            current_value=resolved_tool,
            explicit=tool_explicit or len(installed) <= 1,
            change_fn=_tool_change,
            menu_label=lambda _v: "Change AI tool",
        ),
        ConfirmField(
            name="model",
            label="Model",
            current_value=resolved_model,
            explicit=model_explicit,
            change_fn=_model_change,
        ),
        ConfirmField(
            name="effort",
            label="Effort",
            current_value=resolved_effort,
            explicit=effort_explicit,
            change_fn=_effort_change,
            visible_when=lambda state: _caps_for(state.get("tool"))[0],
            render_value=lambda v: v.value if v else None,
        ),
        ConfirmField(
            name="yolo",
            label="YOLO mode",
            current_value=resolved_yolo,
            explicit=yolo_explicit,
            change_fn=_yolo_change,
            visible_when=lambda state: _caps_for(state.get("tool"))[1],
            render_value=lambda v: "on" if v else None,
            menu_label=lambda v: "Turn off YOLO mode" if v else "Turn on YOLO mode",
        ),
    ]

    result = confirm_defaults(fields, title="Confirm AI selection")
    return result["tool"], result["model"], result["effort"], result["yolo"]


def _select_tool(tool_names: list[str], current_idx: int) -> int:
    from crossby.ui import prompts

    return prompts.select("Select AI tool", tool_names, default=current_idx)


def _prompt_model_selection(tool: str) -> str | None:
    """Show a model picker for *tool* and return the chosen model (or None)."""
    from crossby.data import get_models_for_tool
    from crossby.ui import prompts

    models = get_models_for_tool(tool)
    choices = [*models, _CUSTOM_OPTION]
    idx = prompts.select(f"Select model for {tool}", choices)
    chosen = choices[idx]
    if chosen == _CUSTOM_OPTION:
        custom = prompts.input_prompt("Enter model name", allow_empty=True)
        return custom or None
    return chosen or None


def _prompt_effort_selection(current: EffortLevel | None) -> EffortLevel | None:
    """Show an effort level picker and return the chosen level (or None)."""
    from crossby.ui import prompts

    choices = ["(none — use tool default)", *[e.value for e in EffortLevel]]
    default_idx = 0
    if current:
        default_idx = [e.value for e in EffortLevel].index(current.value) + 1
    idx = prompts.select("Select effort level", choices, default=default_idx)
    if idx == 0:
        return None
    return EffortLevel(choices[idx])
