"""Restriction reporting, ``require_blocked`` and per-session environments.

Each adapter declares what its unattended read-only profile does to each
channel, verified live on its read-only floor (see
``docs/unattended-headless-verification.md#what-read-only-blocks``). These
tests pin those declarations, how they resolve against the build and profile
a session actually runs with, and the runtime's refusal to start a session
whose required channels are not enforced.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.headless import (
    HeadlessRequirementError,
    HeadlessUnsupportedError,
)
from crossby.models.ai import (
    AIToolID,
    HeadlessInteractionMode,
    HeadlessSessionRequest,
    HeadlessTerminalStatus,
    RestrictionChannel,
    RestrictionState,
)
from crossby.utils.versioning import BinaryVersion, parse_semver
from tests.unit.test_ai_tools.test_headless_adapters import (
    AGY_RESULT,
    CLAUDE_RESULT,
    CODEX_EVENTS,
    CURSOR_RESULT,
    OPENCODE_EVENTS,
    _agy_result,
    _jsonl,
    _request,
)

MANAGED = (
    AIToolID.CLAUDE,
    AIToolID.CODEX,
    AIToolID.CURSOR,
    AIToolID.COPILOT,
    AIToolID.OPENCODE,
    AIToolID.ANTIGRAVITY_CLI,
)
E = RestrictionState.ENFORCED
N = RestrictionState.NOT_ENFORCED
U = RestrictionState.UNKNOWN
C = RestrictionChannel

# What each read-only profile was verified to do; every channel not listed is
# unknown. A change here must come with new live evidence in the docs.
VERIFIED: dict[AIToolID, dict[RestrictionChannel, RestrictionState]] = {
    AIToolID.CLAUDE: {
        C.FILE_EDITS: E,
        C.SHELL_COMMANDS: E,
        C.SHELL_WRITES: E,
        C.HOOKS: E,
        C.WEB: E,
    },
    AIToolID.CODEX: {
        C.FILE_EDITS: E,
        C.SHELL_COMMANDS: N,
        C.SHELL_WRITES: E,
        C.HOOKS: E,
        C.WEB: N,
        C.READS_OUTSIDE_WORKSPACE: N,
    },
    AIToolID.CURSOR: {
        C.FILE_EDITS: E,
        C.SHELL_COMMANDS: E,
        C.SHELL_WRITES: E,
        C.HOOKS: N,
        C.WEB: E,
        C.READS_OUTSIDE_WORKSPACE: N,
    },
    AIToolID.COPILOT: {
        C.FILE_EDITS: E,
        C.SHELL_COMMANDS: E,
        C.SHELL_WRITES: E,
        C.HOOKS: N,
        C.MCP_TOOLS: N,
    },
    AIToolID.OPENCODE: {
        C.FILE_EDITS: E,
        C.SHELL_COMMANDS: E,
        C.SHELL_WRITES: E,
        C.HOOKS: N,
        C.PLUGINS: N,
        C.WEB: N,
    },
    AIToolID.ANTIGRAVITY_CLI: {
        C.FILE_EDITS: E,
        C.SHELL_COMMANDS: E,
        C.SHELL_WRITES: E,
        C.HOOKS: N,
        C.WEB: N,
        C.READS_OUTSIDE_WORKSPACE: N,
    },
}

# A successful native payload per tool, for runs that must reach the CLI.
SUCCESS: dict[AIToolID, str] = {
    AIToolID.CLAUDE: json.dumps({**CLAUDE_RESULT, "structured_output": None, "result": "OK"}),
    AIToolID.CODEX: _jsonl(CODEX_EVENTS),
    AIToolID.CURSOR: json.dumps(CURSOR_RESULT),
    AIToolID.COPILOT: "OK\n",
    AIToolID.OPENCODE: _jsonl(OPENCODE_EVENTS),
    AIToolID.ANTIGRAVITY_CLI: _agy_result(
        {**AGY_RESULT, "response": "OK\n", "structured_output": None, "json_schema": None}
    ),
}


def _read_only_floor(adapter: AbstractAITool) -> str:
    floor = type(adapter).HEADLESS_READ_ONLY_VERIFIED_VERSION
    return floor or str(adapter.capabilities().headless.verified_version)


def _states(restrictions: tuple[Any, ...]) -> dict[RestrictionChannel, RestrictionState]:
    return {item.channel: item.state for item in restrictions}


def _recording_cli(record: Path, payload: str, names: tuple[str, ...]) -> list[str]:
    """A child that records some environment variables, then replays ``payload``."""
    program = "\n".join(
        (
            "import json,os,sys",
            "sys.stdin.read()",
            f"names={list(names)!r}",
            f"open({str(record)!r},'w').write(json.dumps({{n: os.environ.get(n) for n in names}}))",
            f"sys.stdout.write({payload!r})",
            "sys.stdout.flush()",
        )
    )
    return [sys.executable, "-c", program]


def _at_build(
    monkeypatch: pytest.MonkeyPatch, tool: AIToolID, argv: list[str], version: str
) -> AbstractAITool:
    adapter = AbstractAITool.get(tool)
    normalized = parse_semver(version)
    assert normalized is not None
    monkeypatch.setattr(
        type(adapter), "_headless_command", lambda self, request, **_k: list(argv), raising=False
    )
    monkeypatch.setattr(
        type(adapter),
        "_detect_headless_version",
        lambda self, **_kwargs: BinaryVersion(normalized, version),
        raising=True,
    )
    if tool is AIToolID.CLAUDE:
        from crossby.ai_tools import claude as claude_module

        monkeypatch.setattr(claude_module, "_new_session_id", lambda: CLAUDE_RESULT["session_id"])
    return adapter


# ---------------------------------------------------------------------------
# Declarations and resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool", MANAGED)
def test_read_only_sessions_report_the_verified_state_of_every_channel(
    tool: AIToolID, tmp_path: Path
) -> None:
    adapter = AbstractAITool.get(tool)
    floor = _read_only_floor(adapter)
    restrictions = adapter.headless_restrictions(_request(tmp_path, read_only=True), floor)
    assert [item.channel for item in restrictions] == list(RestrictionChannel)
    expected = {channel: VERIFIED[tool].get(channel, U) for channel in RestrictionChannel}
    assert _states(restrictions) == expected
    for item in restrictions:
        if item.state is U:
            assert item.mechanism is None and item.verified_on is None
        else:
            assert item.mechanism and item.verified_on == floor


@pytest.mark.parametrize("tool", MANAGED)
def test_every_read_only_profile_blocks_file_changes(tool: AIToolID, tmp_path: Path) -> None:
    # The responder invariant: no file change through file or shell tools.
    adapter = AbstractAITool.get(tool)
    restrictions = adapter.headless_restrictions(
        _request(tmp_path, read_only=True), _read_only_floor(adapter)
    )
    states = _states(restrictions)
    assert states[C.FILE_EDITS] is E
    assert states[C.SHELL_WRITES] is E


@pytest.mark.parametrize("tool", MANAGED)
def test_nothing_is_claimed_below_the_read_only_floor_or_without_read_only(
    tool: AIToolID, tmp_path: Path
) -> None:
    adapter = AbstractAITool.get(tool)
    floor = _read_only_floor(adapter)
    for request, version in (
        (_request(tmp_path, read_only=True), "0.0.1"),
        (_request(tmp_path, read_only=False), floor),
    ):
        restrictions = adapter.headless_restrictions(request, version)
        assert set(_states(restrictions).values()) == {U}


def test_a_brokered_session_claims_nothing(tmp_path: Path) -> None:
    # The verified table describes the unattended profile's flags only.
    adapter = AbstractAITool.get(AIToolID.CODEX)
    request = _request(tmp_path, read_only=True, interaction_mode=HeadlessInteractionMode.BROKERED)
    restrictions = adapter.headless_restrictions(request, "0.166.0")
    assert set(_states(restrictions).values()) == {U}


# ---------------------------------------------------------------------------
# require_blocked is enforced by the runtime, against the detected build
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool", MANAGED)
def test_an_unmet_requirement_is_refused_before_the_cli_starts(
    tool: AIToolID, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = AbstractAITool.get(tool)
    record = tmp_path / "ran.json"
    adapter = _at_build(
        monkeypatch,
        tool,
        _recording_cli(record, SUCCESS[tool], ("PATH",)),
        _read_only_floor(adapter),
    )
    # No tool's profile is verified to block MCP tool calls.
    request = _request(tmp_path, read_only=True, require_blocked=frozenset({C.MCP_TOOLS}))
    with pytest.raises(HeadlessRequirementError) as raised:
        adapter.run_headless_session(request)
    assert not record.exists()
    assert [item.channel for item in raised.value.unmet] == [C.MCP_TOOLS]
    state = VERIFIED[tool].get(C.MCP_TOOLS, U)
    assert raised.value.unmet[0].state is state
    assert f"mcp_tools ({state.value})" in str(raised.value)
    assert len(raised.value.restrictions) == len(RestrictionChannel)
    # The same refusal surfaces at preflight.
    with pytest.raises(HeadlessRequirementError):
        adapter.preflight_headless_session(request)


@pytest.mark.parametrize("tool", MANAGED)
def test_a_met_requirement_runs(
    tool: AIToolID, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = AbstractAITool.get(tool)
    record = tmp_path / "ran.json"
    adapter = _at_build(
        monkeypatch,
        tool,
        _recording_cli(record, SUCCESS[tool], ("PATH",)),
        _read_only_floor(adapter),
    )
    request = _request(
        tmp_path,
        read_only=True,
        require_blocked=frozenset({C.FILE_EDITS, C.SHELL_WRITES}),
    )
    result = adapter.run_headless_session(request)
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert record.exists()


def test_hooks_can_be_required_blocked_only_where_they_are_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path, read_only=True, require_blocked=frozenset({C.HOOKS}))
    for tool in MANAGED:
        adapter = AbstractAITool.get(tool)
        floor = _read_only_floor(adapter)
        adapter = _at_build(
            monkeypatch, tool, _recording_cli(tmp_path / f"{tool}.json", SUCCESS[tool], ()), floor
        )
        if tool in (AIToolID.CLAUDE, AIToolID.CODEX):
            assert adapter.run_headless_session(request).status is HeadlessTerminalStatus.SUCCEEDED
        else:
            with pytest.raises(HeadlessRequirementError, match=r"hooks \(not_enforced\)"):
                adapter.run_headless_session(request)


def test_a_requirement_is_resolved_against_the_build_that_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Claude's read-only profile postdates its unattended floor: on an older
    # build the session is refused as not read-only before any requirement.
    adapter = _at_build(
        monkeypatch,
        AIToolID.CLAUDE,
        _recording_cli(tmp_path / "ran.json", SUCCESS[AIToolID.CLAUDE], ()),
        "2.1.300",
    )
    request = _request(tmp_path, read_only=True, require_blocked=frozenset({C.FILE_EDITS}))
    with pytest.raises(HeadlessUnsupportedError, match="Read-only Claude Code sessions"):
        adapter.run_headless_session(request)
    # Without read-only, nothing is verified, so any requirement is unmet.
    plain = _request(tmp_path, require_blocked=frozenset({C.FILE_EDITS}))
    with pytest.raises(HeadlessRequirementError, match=r"file_edits \(unknown\)"):
        adapter.run_headless_session(plain)


@pytest.mark.parametrize("tool", MANAGED)
def test_preflight_reports_every_channel(
    tool: AIToolID, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = AbstractAITool.get(tool)
    floor = _read_only_floor(adapter)
    adapter = _at_build(monkeypatch, tool, ["unused"], floor)
    report = adapter.preflight_headless_session(_request(tmp_path, read_only=True))
    expected = {channel: VERIFIED[tool].get(channel, U) for channel in RestrictionChannel}
    assert _states(report.restrictions) == expected


# ---------------------------------------------------------------------------
# Per-session environment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool", MANAGED)
def test_session_environment_reaches_only_the_child(
    tool: AIToolID, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CROSSBY_TEST_MARK", raising=False)
    monkeypatch.setenv("CROSSBY_TEST_PARENT", "kept")
    record = tmp_path / "env.json"
    adapter = AbstractAITool.get(tool)
    adapter = _at_build(
        monkeypatch,
        tool,
        _recording_cli(
            record, SUCCESS[tool], ("CROSSBY_TEST_MARK", "CROSSBY_TEST_PARENT", "NO_COLOR", "PWD")
        ),
        _read_only_floor(adapter),
    )
    request = _request(
        tmp_path,
        read_only=True,
        # NO_COLOR is the adapter's own; the request cannot override it.
        environment={"CROSSBY_TEST_MARK": "child-only", "NO_COLOR": "0"},
    )
    result = adapter.run_headless_session(request)
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    seen = json.loads(record.read_text())
    assert seen["CROSSBY_TEST_MARK"] == "child-only"
    assert seen["CROSSBY_TEST_PARENT"] == "kept"
    assert seen["NO_COLOR"] == "1"
    assert seen["PWD"] == os.path.abspath(tmp_path)
    assert "CROSSBY_TEST_MARK" not in os.environ


@pytest.mark.parametrize(
    "environment",
    [
        {"PWD": "/elsewhere"},
        {"1BAD": "x"},
        {"A=B": "x"},
        {"NUL": "a\x00b"},
        {f"V{index}": "x" for index in range(65)},
        {"BIG": "x" * (33 * 1024)},
    ],
)
def test_session_environment_is_validated(environment: dict[str, str], tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        HeadlessSessionRequest(prompt="p", working_dir=tmp_path, environment=environment)


def test_a_transport_that_cannot_pass_an_environment_rejects_one(tmp_path: Path) -> None:
    # Codex's brokered profile runs over its app-server, not the CLI transport.
    adapter = AbstractAITool.get(AIToolID.CODEX)
    request = _request(
        tmp_path,
        interaction_mode=HeadlessInteractionMode.BROKERED,
        environment={"CROSSBY_TEST_MARK": "x"},
    )
    with pytest.raises(HeadlessUnsupportedError, match="cannot pass a session environment"):
        adapter.preflight_headless_session(request, interaction_handler=lambda _q: None)  # type: ignore[arg-type,return-value]
