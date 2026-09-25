"""Native Plan startup must precede exactly one caller-owned task submission."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from crossby.ai_tools.codex import CodexAdapter
from crossby.ai_tools.codex_terminal import (
    TerminalStartupError,
    _Phase,
    _Startup,
    _TerminalReplies,
    parse_wrapper_args,
    validate_message,
)
from crossby.ai_tools.interactive import InteractiveSession
from crossby.ai_tools.plan_mode import PlanModeUnsupportedError
from crossby.models.ai import (
    EffortLevel,
    InteractiveLaunchEvent,
    InteractiveLaunchEventKind,
    PlanModeActivation,
)

READY = "│ model: gpt-5.6-sol low │\n\u203a Ask Codex to do anything\ngpt-5.6-sol"
PLAN = READY + " Plan mode"
WIDE_PLAN = (
    "│ model: gpt-5.6-sol low │\n\u203a Ask Codex to do anything\n"
    "GPT-5.6-Sol high · /tmp/plan-worktree · Context 0% used · weekly 45% left · session "
    "Plan mode (shift+tab to cycle)    ⚠ 1 warning · f2 to view"
)


def activate(startup: _Startup) -> bytes:
    assert startup.advance(READY) == b"/plan"
    assert startup.advance("\u203a /plan  switch to Plan mode\n\u203a /plan") == b"\r"
    return startup.advance(PLAN)


def test_caller_receives_ready_before_input_and_submission_after_native_turn() -> None:
    received: list[InteractiveLaunchEventKind] = []
    prompt = "Plan this task.\n" * 1000

    def handler(event: InteractiveLaunchEvent, session: InteractiveSession) -> None:
        received.append(event.kind)
        if event.kind is InteractiveLaunchEventKind.PLAN_READY:
            session.send_message(prompt)
            with pytest.raises(ValueError, match="exactly one"):
                session.send_message("duplicate")
        else:
            with pytest.raises(ValueError, match="during PLAN_READY"):
                session.send_message("late")

    startup = _Startup(None, handler)
    assert activate(startup) == b"\x1b[200~" + prompt.encode() + b"\x1b[201~"
    assert received == [InteractiveLaunchEventKind.PLAN_READY]
    assert startup.advance(PLAN) == b""
    assert startup.advance(f"\u203a [Pasted Content {len(prompt)} chars]\ngpt Plan mode") == b"\r"
    assert startup.advance("Working (0s • esc to interrupt)\ngpt Plan mode") == b""
    assert startup.phase is _Phase.ACTIVE
    assert received == [
        InteractiveLaunchEventKind.PLAN_READY,
        InteractiveLaunchEventKind.MESSAGE_SUBMITTED,
    ]
    assert startup.advance(PLAN) == b""
    assert len(received) == 2


def test_wide_status_with_shortcut_and_warning_still_submits_task() -> None:
    received: list[InteractiveLaunchEventKind] = []

    def handler(event: InteractiveLaunchEvent, session: InteractiveSession) -> None:
        received.append(event.kind)
        if event.kind is InteractiveLaunchEventKind.PLAN_READY:
            session.send_message("Plan issue #188")

    startup = _Startup(None, handler)
    assert startup.advance(READY) == b"/plan"
    assert startup.advance("\u203a /plan  switch to Plan mode\n\u203a /plan") == b"\r"
    assert startup.advance(WIDE_PLAN) == b"\x1b[200~Plan issue #188\x1b[201~"
    assert received == [InteractiveLaunchEventKind.PLAN_READY]
    assert startup.advance("\u203a Plan issue #188\n" + WIDE_PLAN.splitlines()[-1]) == b"\r"
    assert startup.advance("Working (0s • esc to interrupt)\n" + WIDE_PLAN.splitlines()[-1]) == b""
    assert startup.phase is _Phase.ACTIVE
    assert received == [
        InteractiveLaunchEventKind.PLAN_READY,
        InteractiveLaunchEventKind.MESSAGE_SUBMITTED,
    ]


def test_loading_modal_and_plain_text_plan_do_not_trigger_activation() -> None:
    startup = _Startup("Task", None)
    assert startup.advance(READY.replace("gpt-5.6-sol low", "loading")) == b""
    assert startup.advance(READY + "\nDo you trust the contents of this directory?") == b""
    assert startup.user_screen
    assert startup.advance(READY) == b"/plan"
    assert startup.advance("\u203a /plan") == b"\r"
    assert startup.advance("An assistant mentioned Plan mode\n" + READY) == b""
    assert startup.phase is _Phase.PLAN


def test_plan_suggestion_alone_is_not_a_composer_confirmation() -> None:
    startup = _Startup(None, None)
    assert startup.advance(READY) == b"/plan"
    assert startup.advance("\u203a /plan  switch to Plan mode") == b""
    assert startup.phase is _Phase.COMMAND


def test_no_task_is_valid_for_interactive_plan_only_launch() -> None:
    startup = _Startup(None, None)
    assert activate(startup) == b""
    assert startup.phase is _Phase.ACTIVE


def test_callback_failure_closes_input_port() -> None:
    def fail(event: InteractiveLaunchEvent, session: InteractiveSession) -> None:
        raise RuntimeError("caller failed")

    startup = _Startup(None, fail)
    with pytest.raises(RuntimeError, match="caller failed"):
        activate(startup)
    with pytest.raises(ValueError):
        startup.port.send_message("late")


@pytest.mark.parametrize(
    "message",
    [
        "",
        "  ",
        "/exit",
        "  !rm anything",
        "text\x1b[201~",
        "x\x00",
        "x\x9b",
        "x\ud800",
        "a" * 100001,
    ],
)
def test_unsafe_or_unbounded_task_rejected(message: str) -> None:
    with pytest.raises(ValueError):
        validate_message(message)


def test_short_multiline_unicode_task_and_duplicate_ready() -> None:
    prompt = "Plan café\nKeep the name."
    startup = _Startup(prompt, None)
    activate(startup)
    assert startup.advance("\u203a Plan café\nKeep the name.\ngpt Plan mode") == b"\r"
    assert startup.advance("\u203a Plan café\nKeep the name.\ngpt Plan mode") == b""


def test_positional_slash_is_not_rewritten_outside_plan() -> None:
    command = CodexAdapter().build_launch_command(initial_message="/plan task")
    assert command == ["codex", "/plan task"]


def test_public_command_activates_plan_and_preserves_policy_effort() -> None:
    adapter = CodexAdapter()
    with patch("crossby.utils.versioning.detect_binary_version", return_value=(0, 154, 0)):
        command = adapter.build_launch_command(
            plan_mode=True,
            initial_message="--literal task",
            model="gpt-5.6-sol",
            effort=EffortLevel.HIGH,
            yolo=True,
        )
    assert command[:3] == [sys.executable, "-m", "crossby.ai_tools.codex_terminal"]
    prompt, native = parse_wrapper_args(command[3:])
    assert prompt == "--literal task"
    assert prompt not in native
    assert native[0] == "codex"
    assert "--no-alt-screen" in native
    assert native[native.index("-a") + 1] == "never"
    assert 'model_reasoning_effort="high"' in native
    assert 'plan_model_reasoning_effort="high"' not in native
    assert 'plan_mode_reasoning_effort="high"' in native
    assert adapter.capabilities().plan_mode.activation is PlanModeActivation.TERMINAL_INPUT
    assert adapter.capabilities().plan_mode.supports_ready_event


@pytest.mark.parametrize("version", [None, (0, 153, 4), (0, 155, 0), (1, 0, 0)])
def test_unverified_terminal_versions_rejected(version: tuple[int, int, int] | None) -> None:
    with (
        patch("crossby.utils.versioning.detect_binary_version", return_value=version),
        pytest.raises(PlanModeUnsupportedError),
    ):
        CodexAdapter().build_launch_command(plan_mode=True)


def test_codex_0157_terminal_startup_is_allowed() -> None:
    with patch("crossby.utils.versioning.detect_binary_version", return_value=(0, 157, 0)):
        command = CodexAdapter().build_launch_command(plan_mode=True)
    assert command[:3] == [sys.executable, "-m", "crossby.ai_tools.codex_terminal"]


def test_unverified_codex_version_names_supported_families() -> None:
    with (
        patch("crossby.utils.versioning.detect_binary_version", return_value=(0, 156, 0)),
        pytest.raises(PlanModeUnsupportedError) as error,
    ):
        CodexAdapter().build_launch_command(plan_mode=True)
    assert "0.154.x or 0.157.x" in str(error.value)
    assert "upgrade Codex CLI to 0.154.0 or newer" not in str(error.value)


def test_headless_and_non_posix_terminal_plan_rejected() -> None:
    with pytest.raises(PlanModeUnsupportedError, match="POSIX"):
        CodexAdapter().build_launch_command(plan_mode=True, prompt="Task")
    with patch("crossby.ai_tools.codex.os.name", "nt"), pytest.raises(PlanModeUnsupportedError):
        CodexAdapter().validate_plan_mode_request(plan_mode=True)


def test_launch_passes_callback_to_terminal_runner_without_task_argv(tmp_path: Path) -> None:
    def handler(event: InteractiveLaunchEvent, session: InteractiveSession) -> None:
        session.send_message("Task")

    with (
        patch("crossby.utils.versioning.detect_binary_version", return_value=(0, 154, 0)),
        patch("crossby.ai_tools.codex_terminal.run_terminal_plan", return_value=7) as run,
    ):
        assert CodexAdapter().launch(tmp_path, plan_mode=True, on_event=handler) == 7
    assert run.call_args.kwargs["on_event"] is handler
    assert run.call_args.kwargs["prompt"] is None
    assert run.call_args.args[0][0] == "codex"


def test_ready_handler_and_positional_prompt_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="without prompt"):
        CodexAdapter().launch(tmp_path, plan_mode=True, prompt="Task", on_event=lambda *_: None)


def test_partial_terminal_replies_and_keys_are_distinct() -> None:
    replies = _TerminalReplies()
    assert replies.feed(b"\x1b[12;") == b""
    assert replies.feed(b"3R\x1b]10;rgb:aaaa/bbbb/") == b"\x1b[12;3R"
    assert replies.feed(b"cccc\x1b\\\x1b[?1;2c") == b"\x1b]10;rgb:aaaa/bbbb/cccc\x1b\\\x1b[?1;2c"
    with pytest.raises(TerminalStartupError, match="interrupted"):
        replies.feed(b"\x1b[A")


@pytest.mark.parametrize("sandbox", [True, False])
@pytest.mark.parametrize("policy", ["auto", "yolo", "accept_edits"])
def test_plan_approval_composes_without_changing_sandbox(sandbox: bool, policy: str) -> None:
    with patch("crossby.utils.versioning.detect_binary_version", return_value=(0, 154, 0)):
        command = CodexAdapter().build_launch_command(
            plan_mode=True, sandbox=sandbox, trusted_dirs=["/tmp/reference"], **{policy: True}
        )
    _, native = parse_wrapper_args(command[3:])
    assert native[native.index("--sandbox") + 1] == (
        "workspace-write" if sandbox else "danger-full-access"
    )
    assert native[native.index("-a") + 1] == ("never" if policy == "yolo" else "on-request")
    assert ('approvals_reviewer="auto_review"' in native) is (policy == "auto")
    assert "--dangerously-bypass-approvals-and-sandbox" not in native
    assert "--no-alt-screen" in native


def test_detached_terminal_launch_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="detached"):
        CodexAdapter().launch(tmp_path, plan_mode=True, detach=True)


@pytest.mark.parametrize("response", [b"\x1b]10;rgb:aa/bb/cc\x1b\\", b"\x1b[12;3R", b"\x1b[?1;2c"])
def test_terminal_response_may_split_at_any_byte(response: bytes) -> None:
    replies = _TerminalReplies()
    result = b"".join(replies.feed(bytes([byte])) for byte in response)
    assert result == response
