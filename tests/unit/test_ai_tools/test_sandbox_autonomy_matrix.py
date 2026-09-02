"""Cross-adapter sandbox capability and autonomy independence matrix."""

from __future__ import annotations

import warnings

import pytest

from crossby.ai_tools.base import AbstractAITool


@pytest.mark.parametrize("tool", ["claude", "copilot", "antigravity-cli", "opencode"])
@pytest.mark.parametrize(
    "autonomy",
    [{}, {"accept_edits": True}, {"auto": True}, {"yolo": True}],
)
def test_unsupported_adapters_are_byte_identical(tool: str, autonomy: dict[str, bool]) -> None:
    adapter = AbstractAITool.get(tool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sandboxed = adapter.build_launch_command(**autonomy, sandbox=True)
        unsandboxed = adapter.build_launch_command(**autonomy, sandbox=False)
    assert sandboxed == unsandboxed
    assert "--sandbox" not in sandboxed


def test_unsupported_adapter_keeps_preexisting_trusted_dirs() -> None:
    adapter = AbstractAITool.get("claude")
    sandboxed = adapter.build_launch_command(trusted_dirs=["/tmp/plan"], sandbox=True)
    unsandboxed = adapter.build_launch_command(trusted_dirs=["/tmp/plan"], sandbox=False)
    assert sandboxed == unsandboxed == ["claude", "--add-dir", "/tmp/plan"]


@pytest.mark.parametrize(
    ("autonomy", "approval"),
    [
        ({}, []),
        ({"accept_edits": True}, ["-a", "on-request"]),
        ({"auto": True}, ["-a", "on-request"]),
        ({"yolo": True}, ["-a", "never"]),
    ],
)
def test_codex_approval_subsequence_is_sandbox_independent(
    autonomy: dict[str, bool], approval: list[str]
) -> None:
    adapter = AbstractAITool.get("codex")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        commands = [
            adapter.build_launch_command(**autonomy, sandbox=value) for value in (True, False)
        ]
    for command in commands:
        assert command[1 : 1 + len(approval)] == approval


@pytest.mark.parametrize(
    ("autonomy", "args"),
    [
        ({}, []),
        ({"accept_edits": True}, []),
        ({"auto": True}, []),
        ({"yolo": True}, ["--force"]),
    ],
)
def test_cursor_autonomy_subsequence_is_sandbox_independent(
    autonomy: dict[str, bool], args: list[str]
) -> None:
    adapter = AbstractAITool.get("cursor")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        commands = [
            adapter.build_launch_command(**autonomy, sandbox=value) for value in (True, False)
        ]
    for command in commands:
        assert command[1 : 1 + len(args)] == args
