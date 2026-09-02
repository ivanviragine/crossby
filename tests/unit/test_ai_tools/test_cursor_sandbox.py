"""Cursor's explicit sandbox selection stays independent from autonomy."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from crossby.ai_tools.cursor import CursorAdapter


@pytest.mark.parametrize(
    ("sandbox", "value"),
    [(True, "enabled"), (False, "disabled")],
)
def test_default_sandbox_mapping(sandbox: bool, value: str) -> None:
    assert CursorAdapter().build_launch_command(sandbox=sandbox) == [
        "agent",
        "--sandbox",
        value,
    ]


@pytest.mark.parametrize("sandbox", [True, False])
def test_sandbox_is_orthogonal_to_autonomy(sandbox: bool) -> None:
    adapter = CursorAdapter()
    suffix = ["--sandbox", "enabled" if sandbox else "disabled"]

    assert adapter.build_launch_command(accept_edits=True, sandbox=sandbox) == ["agent", *suffix]
    assert adapter.build_launch_command(plan_mode=True, sandbox=sandbox) == [
        "agent",
        "--mode",
        "plan",
        *suffix,
    ]
    assert adapter.build_launch_command(yolo=True, sandbox=sandbox) == [
        "agent",
        "--force",
        *suffix,
    ]
    with pytest.warns(UserWarning, match="downgrading to accept-edits"):
        assert adapter.build_launch_command(auto=True, sandbox=sandbox) == ["agent", *suffix]


def test_capability_gate_blocks_accidental_translation() -> None:
    adapter = CursorAdapter()
    unsupported = adapter.capabilities().model_copy(update={"supports_sandbox_toggle": False})
    with patch.object(adapter, "capabilities", return_value=unsupported):
        assert adapter.build_launch_command(sandbox=True) == ["agent"]
        assert adapter.build_launch_command(sandbox=False) == ["agent"]
