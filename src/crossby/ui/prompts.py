"""Interactive prompts — confirm, input, select, menu.

TTY-aware: prompts are only displayed when stdin is a TTY.
When stdin is not a TTY, defaults are used silently.

Uses questionary for arrow-key navigation menus.
"""

from __future__ import annotations

import sys
from typing import Any

import questionary
import typer
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style
from questionary.prompts import common
from questionary.prompts.common import InquirerControl
from questionary.question import Question
from questionary.styles import merge_styles_default
from rich.console import Console

_console = Console(stderr=True)

# Sentinel value for the synthetic "Select all" checkbox choice.
_SELECT_ALL_VALUE = "__all__"

# Custom prompt_toolkit style matching the color palette
_style = Style(
    [
        ("qmark", "fg:#7c8aff bold"),  # ? marker
        ("question", "bold"),  # question text
        ("answer", "fg:#7c8aff bold"),  # submitted answer
        ("pointer", "fg:#7c8aff bold"),  # pointer character
        ("highlighted", "fg:#7c8aff bold"),  # currently highlighted choice
        ("selected", "fg:#7c8aff bold"),  # selected checkbox item
        ("instruction", "fg:#888888"),  # (Use arrow keys) hint
    ]
)


def is_tty() -> bool:
    """Check if stdin is connected to a terminal."""
    return sys.stdin.isatty()


def _handle_none(result: object) -> None:
    """Raise typer.Exit if questionary returns None (Ctrl+C)."""
    if result is None:
        raise typer.Exit(1)


def confirm(message: str, default: bool = False) -> bool:
    """Ask a yes/no confirmation question.

    Returns default when stdin is not a TTY.
    """
    if not is_tty():
        return default
    choices = ["Yes", "No"]
    default_choice = "Yes" if default else "No"

    result: str | None = questionary.select(
        message,
        choices=choices,
        default=default_choice,
        pointer="\u203a",
        style=_style,
        instruction="",
    ).ask()

    _handle_none(result)
    return result == "Yes"


def input_prompt(label: str, default: str = "", allow_empty: bool = False) -> str:
    """Ask for text input.

    Returns default when stdin is not a TTY.
    When allow_empty is True, pressing Enter without input returns "".
    """
    if not is_tty():
        return default
    instruction = "(Enter to skip)" if allow_empty and not default else None
    result: str | None = questionary.text(
        label,
        default=default,
        instruction=instruction,
        style=_style,
    ).ask()
    _handle_none(result)
    result_str = result or ""
    return result_str or default


def select(
    title: str,
    items: list[str],
    default: int = 0,
    hints: list[str] | None = None,
) -> int:
    """Arrow-key select picker — display items and let the user choose one.

    Returns the 0-based index of the selected item.
    Returns default when stdin is not a TTY.

    Args:
        title: The prompt title.
        items: List of item labels.
        default: Default 0-based index.
        hints: Optional right-aligned hints per item (e.g. command names).
    """
    if not is_tty():
        return default

    # Build display labels (with hints) — these are the plain string choices
    choices: list[str] = []
    for i, item in enumerate(items):
        if hints and i < len(hints) and hints[i]:
            choices.append(f"{item}  ({hints[i]})")
        else:
            choices.append(item)

    # Build the questionary choice list.
    q_choices: list[str] = list(choices)
    adjusted_default = default

    default_choice: str | questionary.Choice = (
        q_choices[adjusted_default] if 0 <= adjusted_default < len(q_choices) else q_choices[0]
    )
    result: object = questionary.select(
        title,
        choices=q_choices,
        default=default_choice,
        pointer="\u203a",
        style=_style,
        instruction="",
    ).ask()
    _handle_none(result)

    # result is a plain string — map it to a 0-based index into the original items
    if not isinstance(result, str):
        return default
    try:
        return choices.index(result)
    except ValueError:
        return default


def menu(
    title: str,
    items: list[str],
    default: int = 0,
    hints: list[str] | None = None,
    version: str | None = None,
) -> int:
    """Interactive menu with arrow-key navigation.

    Args:
        title: Menu heading.
        items: List of menu item labels.
        default: Default 0-based index.
        hints: Optional command hints per item.
        version: Optional version string to display above menu.
    """
    if not is_tty():
        return default

    # Show version header via Rich before the questionary prompt
    if version:
        _console.print(f"  [dim]{version}[/]")
        _console.print()

    return select(title, items, default=default, hints=hints)


def multi_select(
    title: str,
    items: list[str],
    select_all: bool = False,
) -> list[int]:
    """Checkbox multi-select — arrow keys + Space to toggle, Enter to confirm.

    Returns a list of 0-based indices.
    Returns all items when stdin is not a TTY.

    When select_all is True, a "Select all" choice is prepended, checked by
    default, and mutually exclusive with the individual items: toggling it
    on unchecks every individual item, and toggling any individual item on
    unchecks it.
    """
    if not is_tty():
        return list(range(len(items)))

    if select_all:
        result: list[str] | None = _select_all_checkbox(title, items).ask()
        _handle_none(result)
        return _resolve_select_all_indices(result or [], items)

    result = questionary.checkbox(
        title,
        choices=items,
        pointer="\u203a",
        style=_style,
        instruction="(Space to toggle, Enter to confirm)",
    ).ask()
    _handle_none(result)

    # Map selected labels back to indices
    selected = result or []
    return [items.index(s) for s in selected if s in items]


def _build_select_all_choices(items: list[str]) -> list[questionary.Choice]:
    """Build choices for select_all mode: "Select all" checked, items unchecked."""
    choices = [questionary.Choice("Select all", value=_SELECT_ALL_VALUE, checked=True)]
    choices.extend(questionary.Choice(item, value=item, checked=False) for item in items)
    return choices


def _toggle_select_all(selected: list[str], pointed_value: str) -> list[str]:
    """Apply the mutual-exclusion toggle for one checkbox interaction.

    Toggling "Select all" on clears every individual choice (and off leaves
    nothing selected). Toggling any individual choice clears "Select all".
    """
    if pointed_value == _SELECT_ALL_VALUE:
        if _SELECT_ALL_VALUE in selected:
            return []
        return [_SELECT_ALL_VALUE]

    new_selected = [v for v in selected if v != _SELECT_ALL_VALUE]
    if pointed_value in new_selected:
        new_selected.remove(pointed_value)
    else:
        new_selected.append(pointed_value)
    return new_selected


def _resolve_select_all_indices(selected: list[str], items: list[str]) -> list[int]:
    """Map selected checkbox values back to 0-based indices.

    "__all__" (or every individual item already checked) resolves to every
    index; otherwise only the selected items' indices are returned.
    """
    if _SELECT_ALL_VALUE in selected:
        return list(range(len(items)))
    return [items.index(v) for v in selected if v in items]


def _select_all_checkbox(title: str, items: list[str]) -> Question:
    """Build a checkbox Question with select_all's mutual-exclusion toggle.

    questionary's public checkbox() has no hook for customizing what a
    toggle does, so the Space/Enter key bindings are reimplemented here
    against InquirerControl directly, mirroring questionary's own
    checkbox.py minus the features (search filter, invert-all) this mode
    doesn't need.
    """
    merged_style = merge_styles_default([_style])
    ic = InquirerControl(_build_select_all_choices(items), None, pointer="\u203a")

    def get_prompt_tokens() -> list[tuple[str, str]]:
        tokens = [("class:qmark", "?"), ("class:question", f" {title} ")]
        if ic.is_answered:
            tokens.append(("class:answer", f"done ({len(ic.selected_options)} selections)"))
        else:
            tokens.append(("class:instruction", "(Space to toggle, Enter to confirm)"))
        return tokens

    bindings = KeyBindings()

    @bindings.add(Keys.ControlQ, eager=True)
    @bindings.add(Keys.ControlC, eager=True)
    def _abort(event: Any) -> None:
        event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

    @bindings.add(" ", eager=True)
    def _toggle(_event: Any) -> None:
        pointed_value = str(ic.get_pointed_at().value)
        ic.selected_options = _toggle_select_all(ic.selected_options, pointed_value)

    @bindings.add(Keys.Down, eager=True)
    def _down(_event: Any) -> None:
        ic.select_next()
        while not ic.is_selection_valid():
            ic.select_next()

    @bindings.add(Keys.Up, eager=True)
    def _up(_event: Any) -> None:
        ic.select_previous()
        while not ic.is_selection_valid():
            ic.select_previous()

    @bindings.add(Keys.ControlM, eager=True)
    def _submit(event: Any) -> None:
        ic.is_answered = True
        selected_values = [c.value for c in ic.get_selected_values()]
        event.app.exit(result=selected_values)

    @bindings.add(Keys.Any)
    def _other(_event: Any) -> None:
        """Disallow inserting other text."""

    layout = common.create_inquirer_layout(ic, get_prompt_tokens)

    return Question(
        Application(
            layout=layout,
            key_bindings=bindings,
            style=merged_style,
        )
    )
