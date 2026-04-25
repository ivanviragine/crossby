"""crossby init — scaffold a ``.crossby.yml`` config file.

Interactive flow walks the user through AI launch defaults, sync defaults,
and handoff defaults. All answers funnel into a final ``confirm_defaults``
review step so any single value can be changed before the file is written.

The on-disk format is written by :func:`_render_init_yaml`, which dumps each
section separately and concatenates them with hand-written comment blocks —
PyYAML cannot preserve comments, and pulling in ``ruamel.yaml`` is out of
scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
import yaml

from crossby.ui.console import console

_CUSTOM = "Custom…"
_NONE = "(none)"


def init(
    path: Path = typer.Option(Path("."), "--path", help="Project root directory."),
    force: bool = typer.Option(
        False, "--force", help="Overwrite an existing .crossby.yml."
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Write a minimal default file with no prompting (CI-safe).",
    ),
) -> None:
    """Create a ``.crossby.yml`` in the project root.

    Examples::

        crossby init                     # walk the interactive wizard
        crossby init --non-interactive   # write a minimal default file
        crossby init --force             # overwrite an existing file
    """
    from crossby.ai_tools.base import AbstractAITool
    from crossby.config.loader import parse_config_file
    from crossby.ui import prompts

    project_root = path.resolve()
    target = project_root / ".crossby.yml"

    if target.exists() and not force:
        console.error(
            f"Refusing to overwrite existing config: {target}. Pass --force to replace it."
        )
        raise typer.Exit(1)

    installed = AbstractAITool.detect_installed()

    if non_interactive:
        answers = _non_interactive_defaults(installed)
    else:
        if not prompts.is_tty():
            console.error(
                "Interactive prompts require a TTY; use --non-interactive to "
                "write defaults without prompts."
            )
            raise typer.Exit(1)
        answers = _interactive_wizard(installed)

    rendered = _render_init_yaml(answers)
    target.write_text(rendered, encoding="utf-8")

    # Sanity check — the file must round-trip through the real loader.
    try:
        parse_config_file(target)
    except Exception as exc:  # pragma: no cover — keeps user out of a broken state
        console.error(f"Wrote invalid config ({exc}); removing and aborting.")
        target.unlink(missing_ok=True)
        raise typer.Exit(1) from exc

    console.success(f"Wrote {target}")


def _non_interactive_defaults(installed: list) -> dict[str, Any]:  # type: ignore[type-arg]
    """Produce the minimal default answer set used when --non-interactive is set."""
    default_tool = str(installed[0]) if installed else None
    return {
        "ai_tool": default_tool,
        "ai_model": None,
        "ai_effort": None,
        "ai_yolo": False,
        "sync_from": None,
        "sync_to": None,
        "handoff_from": None,
        "handoff_to": None,
        "handoff_preset": None,
        "handoff_budget": None,
    }


def _interactive_wizard(installed: list) -> dict[str, Any]:  # type: ignore[type-arg]
    """Run the initial Q&A, then let users review all answers before writing."""
    from crossby.services.confirm import confirm_defaults

    tool_names = [str(t) for t in installed]

    if tool_names:
        ai_tool = _pick_ai_tool(tool_names)
        ai_model, ai_effort, ai_yolo = _pick_ai_tool_details(ai_tool)
    else:
        console.warn("No AI tools detected in PATH — skipping AI launch defaults.")
        ai_tool = None
        ai_model = None
        ai_effort = None
        ai_yolo = False

    sync_from = _pick_tool_or_none("Sync default source", tool_names)
    sync_to = _pick_tool_or_none("Sync default target", tool_names)

    handoff_from = _pick_tool_or_none("Handoff default source", tool_names)
    handoff_to = _pick_tool_or_none("Handoff default target", tool_names)
    handoff_preset = _pick_preset_or_none()
    handoff_budget = _pick_token_budget()

    state: dict[str, Any] = {
        "ai_tool": ai_tool,
        "ai_model": ai_model,
        "ai_effort": ai_effort,
        "ai_yolo": ai_yolo,
        "sync_from": sync_from,
        "sync_to": sync_to,
        "handoff_from": handoff_from,
        "handoff_to": handoff_to,
        "handoff_preset": handoff_preset,
        "handoff_budget": handoff_budget,
    }

    fields = _build_review_fields(state, tool_names)
    result = confirm_defaults(fields, title="Confirm .crossby.yml values")
    return dict(result)


def _build_review_fields(
    state: dict[str, Any], tool_names: list[str]
) -> list:  # type: ignore[type-arg]
    """Build ConfirmFields for the final review step."""
    from crossby.services.confirm import ConfirmField

    tool_field_titles = {
        "sync_from": "Sync default source",
        "sync_to": "Sync default target",
        "handoff_from": "Handoff default source",
        "handoff_to": "Handoff default target",
    }

    def _tool_change_fn(field_name: str) -> Any:
        def _change(_current: Any, _state: dict[str, Any]) -> dict[str, Any]:
            prompt_title = tool_field_titles.get(field_name, field_name)
            return {field_name: _pick_tool_or_none(prompt_title, tool_names)}

        return _change

    def _ai_tool_change(_current: Any, _state: dict[str, Any]) -> dict[str, Any]:
        new_tool = _pick_ai_tool(tool_names) if tool_names else None
        if new_tool is None:
            return {"ai_tool": None, "ai_model": None, "ai_effort": None, "ai_yolo": False}
        new_model, new_effort, new_yolo = _pick_ai_tool_details(new_tool)
        return {
            "ai_tool": new_tool,
            "ai_model": new_model,
            "ai_effort": new_effort,
            "ai_yolo": new_yolo,
        }

    def _model_change(_current: Any, state_: dict[str, Any]) -> dict[str, Any]:
        tool = state_.get("ai_tool")
        if not tool:
            return {"ai_model": None}
        return {"ai_model": _pick_model_for_tool(tool)}

    def _effort_change(_current: Any, _state: dict[str, Any]) -> dict[str, Any]:
        return {"ai_effort": _pick_effort_or_none()}

    def _yolo_change(current: Any, _state: dict[str, Any]) -> dict[str, Any]:
        return {"ai_yolo": not bool(current)}

    def _preset_change(_current: Any, _state: dict[str, Any]) -> dict[str, Any]:
        return {"handoff_preset": _pick_preset_or_none()}

    def _budget_change(_current: Any, _state: dict[str, Any]) -> dict[str, Any]:
        return {"handoff_budget": _pick_token_budget()}

    fields: list[ConfirmField] = [
        ConfirmField(
            name="ai_tool",
            label="AI tool",
            current_value=state["ai_tool"],
            explicit=False,
            change_fn=_ai_tool_change,
            render_value=lambda v: str(v) if v else None,
        ),
        ConfirmField(
            name="ai_model",
            label="AI model",
            current_value=state["ai_model"],
            explicit=False,
            change_fn=_model_change,
            visible_when=lambda s: bool(s.get("ai_tool")),
        ),
        ConfirmField(
            name="ai_effort",
            label="AI effort",
            current_value=state["ai_effort"],
            explicit=False,
            change_fn=_effort_change,
            visible_when=lambda s: _tool_supports(s.get("ai_tool"), "supports_effort"),
        ),
        ConfirmField(
            name="ai_yolo",
            label="YOLO mode",
            current_value=state["ai_yolo"],
            explicit=False,
            change_fn=_yolo_change,
            visible_when=lambda s: _tool_supports(s.get("ai_tool"), "supports_yolo"),
            render_value=lambda v: "on" if v else None,
            menu_label=lambda v: "Turn off YOLO mode" if v else "Turn on YOLO mode",
        ),
        ConfirmField(
            name="sync_from",
            label="Sync default source",
            current_value=state["sync_from"],
            explicit=False,
            change_fn=_tool_change_fn("sync_from"),
        ),
        ConfirmField(
            name="sync_to",
            label="Sync default target",
            current_value=state["sync_to"],
            explicit=False,
            change_fn=_tool_change_fn("sync_to"),
        ),
        ConfirmField(
            name="handoff_from",
            label="Handoff default source",
            current_value=state["handoff_from"],
            explicit=False,
            change_fn=_tool_change_fn("handoff_from"),
        ),
        ConfirmField(
            name="handoff_to",
            label="Handoff default target",
            current_value=state["handoff_to"],
            explicit=False,
            change_fn=_tool_change_fn("handoff_to"),
        ),
        ConfirmField(
            name="handoff_preset",
            label="Handoff prompt preset",
            current_value=state["handoff_preset"],
            explicit=False,
            change_fn=_preset_change,
        ),
        ConfirmField(
            name="handoff_budget",
            label="Handoff token budget",
            current_value=state["handoff_budget"],
            explicit=False,
            change_fn=_budget_change,
            render_value=lambda v: str(v) if v else None,
        ),
    ]
    return fields


def _tool_supports(tool: str | None, cap: str) -> bool:
    if not tool:
        return False
    from crossby.ai_tools.base import AbstractAITool

    try:
        adapter = AbstractAITool.get(tool)
    except (ValueError, KeyError):
        return False
    return bool(getattr(adapter.capabilities(), cap, False))


def _pick_ai_tool(tool_names: list[str]) -> str:
    from crossby.ui import prompts

    idx = prompts.select("Default AI tool", tool_names)
    return tool_names[idx]


def _pick_ai_tool_details(tool: str) -> tuple[str | None, str | None, bool]:
    model = _pick_model_for_tool(tool)
    effort = _pick_effort_or_none() if _tool_supports(tool, "supports_effort") else None
    yolo = _pick_yolo() if _tool_supports(tool, "supports_yolo") else False
    return model, effort, yolo


def _pick_model_for_tool(tool: str) -> str | None:
    from crossby.data import get_models_for_tool
    from crossby.ui import prompts

    models = get_models_for_tool(tool)
    choices = [_NONE, *models, _CUSTOM]
    idx = prompts.select(f"Default model for {tool}", choices)
    choice = choices[idx]
    if choice == _NONE:
        return None
    if choice == _CUSTOM:
        return prompts.input_prompt("Enter model name", allow_empty=True) or None
    return choice


def _pick_effort_or_none() -> str | None:
    from crossby.models.ai import EffortLevel
    from crossby.ui import prompts

    choices = [_NONE, *[e.value for e in EffortLevel]]
    idx = prompts.select("Default effort level", choices)
    return None if idx == 0 else choices[idx]


def _pick_yolo() -> bool:
    from crossby.ui import prompts

    return prompts.confirm("Enable YOLO mode by default?", default=False)


def _pick_tool_or_none(title: str, tool_names: list[str]) -> str | None:
    if not tool_names:
        return None
    from crossby.ui import prompts

    choices = [_NONE, *tool_names]
    idx = prompts.select(title, choices)
    return None if idx == 0 else choices[idx]


def _pick_preset_or_none() -> str | None:
    from crossby.handoff.prompts import PRESETS
    from crossby.ui import prompts

    choices = [_NONE, *sorted(PRESETS)]
    idx = prompts.select("Default handoff prompt preset", choices)
    return None if idx == 0 else choices[idx]


def _pick_token_budget() -> int | None:
    from crossby.ui import prompts

    raw = prompts.input_prompt(
        "Default handoff token budget (blank = no default)", allow_empty=True
    )
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        console.warn(f"Not an integer ({raw!r}); leaving token budget unset.")
        return None


def _render_init_yaml(answers: dict[str, Any]) -> str:
    """Assemble the commented YAML document.

    PyYAML strips comments, so each section is dumped individually and
    concatenated with hand-written header blocks. ``exclude_none=True`` on the
    Pydantic dumps keeps unset fields out of the file — users see only what
    they explicitly chose.
    """
    from crossby.models.config import (
        AIConfig,
        HandoffDefaults,
        SyncDefaults,
    )

    ai = AIConfig(
        default_tool=answers["ai_tool"],
        default_model=answers["ai_model"],
        effort=answers["ai_effort"],
        yolo=answers["ai_yolo"] if answers["ai_yolo"] else None,
    )
    sync_defaults = SyncDefaults(
        from_tool=answers["sync_from"],
        to=answers["sync_to"],
    )
    handoff_defaults = HandoffDefaults(
        from_tool=answers["handoff_from"],
        to=answers["handoff_to"],
        prompt_preset=answers["handoff_preset"],
        token_budget=answers["handoff_budget"],
    )

    ai_payload = ai.model_dump(mode="json", exclude_none=True)
    sync_payload = sync_defaults.model_dump(
        by_alias=True, mode="json", exclude_none=True
    )
    handoff_payload = handoff_defaults.model_dump(
        by_alias=True, mode="json", exclude_none=True
    )

    parts: list[str] = [
        "# .crossby.yml — launch, sync, and handoff defaults.\n",
        "# See https://github.com/ivanviragine/crossby for the full schema.\n\n",
        "version: 1\n",
    ]

    def _section(header: str, key: str, payload: dict[str, Any]) -> None:
        parts.append(f"\n# {header}\n")
        if payload:
            parts.append(
                yaml.safe_dump({key: payload}, sort_keys=False, default_flow_style=False)
            )
        else:
            parts.append(f"{key}: {{}}\n")

    _section("AI launch defaults.", "ai", ai_payload)
    _section("Defaults for `crossby sync`.", "sync_defaults", sync_payload)
    _section("Defaults for `crossby handoff`.", "handoff_defaults", handoff_payload)

    return "".join(parts)
