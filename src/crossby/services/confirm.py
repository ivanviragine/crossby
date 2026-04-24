"""Generic "Proceed / Change X" confirmation loop.

Callers build a list of :class:`ConfirmField`s — one per value the user might
want to review — and pass them to :func:`confirm_defaults`. The helper prints
the current values, offers a menu with ``Proceed`` plus one entry per
non-explicit visible field, and loops until the user picks ``Proceed``.

There is no cancel path. Callers that need abort semantics follow the
helper with an explicit ``prompts.confirm()`` step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(kw_only=True)
class ConfirmField:
    """One value the user may review and optionally change.

    Attributes:
        name: Key used in the returned dict and in ``state`` snapshots.
        label: Display label for ``console.kv(label, value)``. Also used to
            derive the default menu entry (``f"Change {label.lower()}"``).
        current_value: Initial resolved value (may be ``None``).
        explicit: When ``True``, the field is shown but no "Change" entry is
            offered — either the user already gave it on the CLI, or there is
            nothing to choose from (e.g. only one installed tool).
        change_fn: Called when the user selects this field's menu entry.
            Receives ``(current_value, state)`` where ``state`` is a fresh
            snapshot of all field names → current values, so fields with
            option lists that depend on another field (e.g. model depending
            on tool) can re-query at call time. Returns a
            ``dict[str, Any]`` of updates to apply; the common case returns
            ``{name: new_value}`` but cascades may return multiple entries
            (e.g. picking a new tool returns ``{"tool": ..., "model": ...,
            "effort": ..., "yolo": ...}``). Any ``prompts.select()`` index
            inside the function must be converted to the underlying value
            before being put in the returned dict.
        visible_when: Optional predicate evaluated against the live state
            snapshot. Returns ``True`` when the field should be rendered and
            eligible for change (e.g. effort visible only when the current
            tool's capabilities report ``supports_effort``). ``None`` means
            always visible.
        render_value: Optional formatter turning the current value into the
            string shown via ``console.kv``. Returns ``None`` to skip the
            ``kv`` line entirely (useful for "off"/empty states). Default
            behaviour: ``str(value)`` when value is not ``None``, else skip.
        menu_label: Optional formatter turning the current value into the
            menu entry label. Default behaviour: ``f"Change {label.lower()}"``.
            Use a custom formatter when the menu text should toggle on state
            (e.g. ``"Turn on YOLO mode"`` vs ``"Turn off YOLO mode"``).
    """

    name: str
    label: str
    current_value: Any
    explicit: bool
    change_fn: Callable[[Any, dict[str, Any]], dict[str, Any]]
    visible_when: Callable[[dict[str, Any]], bool] | None = None
    render_value: Callable[[Any], str | None] | None = None
    menu_label: Callable[[Any], str] | None = None


def confirm_defaults(
    fields: list[ConfirmField],
    *,
    title: str = "Confirm selection",
) -> dict[str, Any]:
    """Review resolved defaults via a Proceed / Change loop.

    Fast paths:
      * Non-TTY stdin — returns current values unchanged.
      * Every visible field has ``explicit=True`` — returns current values
        unchanged.

    Otherwise loops: render each visible field's ``kv`` line (when its
    ``render_value`` returns non-``None``), build a menu with ``Proceed``
    plus one entry per non-explicit visible field, call the selected
    field's ``change_fn`` with a live state snapshot, merge the returned
    dict into the values, and repeat until the user picks ``Proceed``.

    Args:
        fields: Fields to display / allow changing, in render order.
        title: Title passed to ``prompts.select`` for the menu.

    Returns:
        Dict keyed by ``ConfirmField.name`` with the final values after
        any user-driven changes.
    """
    from crossby.ui import prompts
    from crossby.ui.console import console

    values: dict[str, Any] = {f.name: f.current_value for f in fields}

    def _is_visible(fld: ConfirmField, state: dict[str, Any]) -> bool:
        return fld.visible_when(state) if fld.visible_when else True

    def _render(fld: ConfirmField, state: dict[str, Any]) -> str | None:
        v = state[fld.name]
        if fld.render_value is not None:
            return fld.render_value(v)
        return str(v) if v is not None else None

    def _menu_label(fld: ConfirmField, state: dict[str, Any]) -> str:
        if fld.menu_label is not None:
            return fld.menu_label(state[fld.name])
        return f"Change {fld.label.lower()}"

    if not prompts.is_tty():
        return values

    initial_visible = [f for f in fields if _is_visible(f, values)]
    if all(f.explicit for f in initial_visible):
        return values

    while True:
        visible_fields = [f for f in fields if _is_visible(f, values)]

        for f in visible_fields:
            rendered = _render(f, values)
            if rendered is not None:
                console.kv(f.label, rendered)

        menu_items: list[str] = ["Proceed"]
        changeable: list[ConfirmField] = []
        for f in visible_fields:
            if not f.explicit:
                menu_items.append(_menu_label(f, values))
                changeable.append(f)

        if not changeable:
            break

        idx = prompts.select(title, menu_items)
        if idx == 0:
            break

        chosen = changeable[idx - 1]
        updates = chosen.change_fn(values[chosen.name], dict(values))
        values.update(updates)

    return values
