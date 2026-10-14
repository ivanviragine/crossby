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
    _ScreenUpdates,
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
CURRENT_SCREENS = Path(__file__).parents[2] / "fixtures/plan_sessions/codex_0_162_1"


def current_screen(name: str) -> str:
    return (CURRENT_SCREENS / f"{name}.txt").read_text()


def test_native_notice_activates_without_transient_plan_footer() -> None:
    startup = _Startup("Task", None)
    assert startup.advance(current_screen("initial")) == b"/plan"
    assert startup.advance(current_screen("command")) == b"\r"
    assert startup.advance(current_screen("plan-notice-no-footer")) == b"\x1b[200~Task\x1b[201~"
    assert startup.phase is _Phase.PASTE


def test_native_paste_and_acknowledgement_do_not_require_plan_footer_again() -> None:
    events: list[InteractiveLaunchEventKind] = []
    startup = _Startup(
        None,
        lambda event, port: (
            events.append(event.kind),
            port.send_message("Task")
            if event.kind is InteractiveLaunchEventKind.PLAN_READY
            else None,
        ),
    )
    assert startup.advance(current_screen("initial")) == b"/plan"
    assert startup.advance(current_screen("command")) == b"\r"
    assert startup.advance(current_screen("plan")) == b"\x1b[200~Task\x1b[201~"
    footer = current_screen("plan-notice-no-footer").splitlines()[-1]
    assert startup.advance("\u203a Task\n" + footer) == b"\r"
    assert startup.advance(current_screen("working-no-plan-footer")) == b""
    assert startup.phase is _Phase.ACTIVE
    assert startup.advance(current_screen("working-no-plan-footer")) == b""
    assert events == [
        InteractiveLaunchEventKind.PLAN_READY,
        InteractiveLaunchEventKind.MESSAGE_SUBMITTED,
    ]


@pytest.mark.parametrize("boundary", ["ready", "command", "body"])
def test_stale_native_activation_notice_is_not_plan_proof(boundary: str) -> None:
    startup = _Startup("Task", None)
    notice = current_screen("plan-notice-no-footer")
    ready = notice if boundary == "ready" else current_screen("initial")
    if boundary == "body":
        ready = ready.replace("\u203a Ask", notice.splitlines()[0] + "\n\u203a Ask")
    assert startup.advance(ready) == b"/plan"
    command = notice.replace("\u203a Ask Codex to do anything", "\u203a /plan")
    assert startup.advance(command if boundary == "command" else current_screen("command")) == b"\r"
    assert startup.advance(notice) == b""
    assert startup.phase is _Phase.PLAN


@pytest.mark.parametrize(
    "mutation", ["prose", "quoted", "body", "model", "effort", "default", "modal"]
)
def test_notice_requires_exact_fresh_native_context(mutation: str) -> None:
    notice = current_screen("plan-notice-no-footer")
    if mutation == "prose":
        notice = notice.replace("• Model changed", "Assistant says Model changed")
    elif mutation == "quoted":
        notice = "Assistant quoted:\n" + notice
    elif mutation == "body":
        first, rest = notice.split("\n", 1)
        notice = rest.replace("\u203a Ask", first + "\n\u203a Ask")
    elif mutation == "model":
        notice = notice.replace("Model changed to gpt-6.1-sol", "Model changed to o3")
    elif mutation == "effort":
        notice = notice.replace("high for Plan", "low for Plan")
    elif mutation == "default":
        notice = notice.replace("for Plan mode.", "for Default mode.")
    else:
        notice += "\nUnknown modal\n\u203a 1. Continue"
    startup = _Startup("Task", None)
    assert startup.advance(current_screen("initial")) == b"/plan"
    assert startup.advance(current_screen("command")) == b"\r"
    assert startup.advance(notice) == b""
    assert startup.phase is _Phase.PLAN


@pytest.mark.parametrize(
    "body",
    [
        "The task mentions esc to interrupt",
        "• Working (0s • esc to interrupt)\nAssistant quoted this status",
        "\u203a Working (0s • esc to interrupt)",
        "• Working (0s • esc to interrupt)\n\u203a Task",
        "• Working (0s • esc to interrupt)\nUnknown modal\n\u203a Ask Codex to do anything",
    ],
)
def test_task_or_body_text_does_not_acknowledge_submission(body: str) -> None:
    startup = _Startup("Task", None)
    activate(startup)
    assert startup.advance("\u203a Task\ngpt Plan mode") == b"\r"
    footer = current_screen("initial-minimal").splitlines()[-1]
    assert startup.advance(body + "\n" + footer) == b""
    assert startup.phase is _Phase.START


def synchronized(text: str) -> bytes:
    return b"\x1b[?2026h\x1b[2J\x1b[H" + text.replace("\n", "\r\n").encode() + b"\x1b[?2026l"


@pytest.mark.parametrize("chunk", [1, 7, 4096, 65536])
def test_activation_survives_chunked_and_coalesced_native_redraws(chunk: int) -> None:
    startup = _Startup("Task", None)
    assert startup.advance(current_screen("initial")) == b"/plan"
    assert startup.advance(current_screen("command")) == b"\r"
    sent: list[bytes] = []

    def observe(screen: str) -> None:
        if not sent:
            output = startup.advance(screen)
            if output:
                sent.append(output)

    updates = _ScreenUpdates(180, 45, observe)
    # The actual Plan footer frame can disappear in the same os.read payload.
    data = synchronized(current_screen("plan")) + synchronized(current_screen("initial"))
    for offset in range(0, len(data), chunk):
        updates.feed(data[offset : offset + chunk])
    assert sent == [b"\x1b[200~Task\x1b[201~"]
    assert startup.phase is _Phase.PASTE


def test_incomplete_synchronized_redraw_does_not_submit() -> None:
    startup = _Startup("Task", None)
    assert startup.advance(current_screen("initial")) == b"/plan"
    assert startup.advance(current_screen("command")) == b"\r"
    observed: list[str] = []
    updates = _ScreenUpdates(180, 45, observed.append)
    frame = synchronized(current_screen("plan"))
    updates.feed(frame[:-1])
    assert observed == []
    updates.feed(frame[-1:])
    assert len(observed) == 1


@pytest.mark.parametrize(
    "name",
    [
        "initial-80",
        "initial-100",
        "initial-120",
        "initial",
        "initial-default",
        "initial-minimal",
        "initial-branch",
        "initial-root",
        "initial-home",
    ],
)
def test_current_ready_footer_can_omit_context_and_truncate_cwd(name: str) -> None:
    assert _Startup("Task", None).advance(current_screen(name)) == b"/plan"


@pytest.mark.parametrize("model", ["o3", "o4-mini", "codex-mini-latest", "gpt-4.1"])
def test_compatible_models_use_the_observed_native_model_effort_field(model: str) -> None:
    assert CodexAdapter().is_model_compatible(model)
    assert _Startup("Task", None).advance(current_screen(f"initial-{model}")) == b"/plan"


def test_loaded_model_token_is_not_a_brand_allowlist() -> None:
    screen = current_screen("initial").replace("GPT-6.1-Sol", "custom-provider/model:variant")
    assert _Startup("Task", None).advance(screen) == b"/plan"


@pytest.mark.parametrize("model", ["o3", "gpt-4.1"])
def test_unspecified_effort_uses_the_observed_native_default_field(model: str) -> None:
    assert _Startup("Task", None).advance(current_screen(f"initial-{model}-default")) == b"/plan"


@pytest.mark.parametrize("name", ["initial", "initial-minimal"])
@pytest.mark.parametrize("directory", ["/tmp/loading", "/tmp/lazy-loading"])
def test_directory_name_is_not_a_loading_status(name: str, directory: str) -> None:
    screen = current_screen(name).replace("/tmp/native-plan-fixture", directory)
    assert _Startup("Task", None).advance(screen) == b"/plan"


@pytest.mark.parametrize("directory", ["/", "~", "~/project", "/tmp/native plan with spaces"])
def test_current_ready_accepts_complete_root_home_and_spaced_directory_fields(
    directory: str,
) -> None:
    screen = current_screen("initial").replace("/tmp/native-plan-fixture", directory)
    assert _Startup("Task", None).advance(screen) == b"/plan"


@pytest.mark.parametrize("directory", ["relative/path", "~someone", "unknown"])
def test_current_ready_rejects_unrecognized_directory_fields(directory: str) -> None:
    screen = current_screen("initial").replace("/tmp/native-plan-fixture", directory)
    assert _Startup("Task", None).advance(screen) == b""


@pytest.mark.parametrize(
    "metadata", ["feat/loading", "loading-project", "loading thread title", "Loading model"]
)
def test_free_footer_metadata_does_not_override_loaded_model(metadata: str) -> None:
    screen = current_screen("initial-branch").replace("feat/loading", metadata)
    assert _Startup("Task", None).advance(screen) == b"/plan"


@pytest.mark.parametrize(
    "screen",
    [
        "Do you trust the contents of this directory?",
        current_screen("trust"),
        current_screen("trust").replace(
            "\u203a 1. Trust and continue\n  2. Quit", "  1. Trust and continue\n\u203a 2. Quit"
        ),
    ],
)
def test_trust_requires_caller_keyboard_before_automatic_startup(screen: str) -> None:
    startup = _Startup("Task", None)
    for _ in range(2):
        assert startup.advance(screen) == b""
        assert startup.user_screen
        assert startup.phase is _Phase.READY
        assert startup.port.message is None
    assert startup.advance(current_screen("initial")) == b"/plan"
    assert not startup.user_screen


@pytest.mark.parametrize(
    "screen",
    [
        current_screen("trust").replace("Folder access", "Unrecognized dialog"),
        current_screen("trust").replace("Trust this folder?", "Unknown question?"),
        current_screen("trust").replace("Your trust decision will be saved.", "Loading"),
        current_screen("trust").replace("1. Trust and continue", "1. Unknown action"),
        current_screen("trust").replace("2. Quit", ""),
        current_screen("trust").replace("enter continue · esc quit", ""),
        "Assistant quoted this dialog:\n" + current_screen("trust"),
        "Folder access\nLoading",
    ],
)
def test_partial_or_unknown_trust_dialog_never_allows_input_or_injection(screen: str) -> None:
    startup = _Startup("Task", None)
    assert startup.advance(screen) == b""
    assert not startup.user_screen
    assert startup.phase is _Phase.READY
    assert startup.port.message is None


def test_observed_current_native_screens_submit_exactly_once() -> None:
    received: list[InteractiveLaunchEventKind] = []

    def handler(event: InteractiveLaunchEvent, session: InteractiveSession) -> None:
        received.append(event.kind)
        if event.kind is InteractiveLaunchEventKind.PLAN_READY:
            session.send_message(
                "Plan changing greeting.py from Hello to a user-selected greeting."
            )

    startup = _Startup(None, handler)
    assert startup.advance(current_screen("initial")) == b"/plan"
    assert startup.advance(current_screen("command")) == b"\r"
    assert startup.advance(current_screen("plan")).startswith(b"\x1b[200~")
    assert startup.advance(current_screen("plan")) == b""
    assert startup.advance(current_screen("paste")) == b"\r"
    assert startup.advance(current_screen("paste")) == b""
    assert startup.advance(current_screen("working")) == b""
    assert startup.phase is _Phase.ACTIVE
    assert startup.advance(current_screen("question")) == b""
    assert startup.advance(current_screen("final")) == b""
    assert received == [
        InteractiveLaunchEventKind.PLAN_READY,
        InteractiveLaunchEventKind.MESSAGE_SUBMITTED,
    ]


@pytest.mark.parametrize(
    "screen",
    [
        "GPT-6.1-Sol high\n\u203a Ask Codex to do anything",
        "An assistant mentioned GPT-6.1-Sol high and Plan mode\n\u203a Ask Codex to do anything",
        current_screen("initial").replace(">_ OpenAI Codex (v0.162.1)", "Unknown application"),
        current_screen("initial").replace("GPT-6.1-Sol high", "loading"),
        current_screen("initial").replace("GPT-6.1-Sol high", "Unrecognized screen"),
        current_screen("initial").replace(
            "\u203a Ask Codex to do anything", "\u203a Unknown composer"
        ),
        current_screen("initial") + "Unknown modal\n\u203a 1. Continue",
        current_screen("initial") + "Sign in to continue",
        current_screen("initial") + "Do you trust the contents of this directory?",
        current_screen("question"),
        current_screen("final"),
        READY.replace(
            "\u203a Ask Codex to do anything", "\u203a Ask Codex to do anything\n\u203a /unknown"
        ),
    ],
)
def test_unrecognized_loading_and_blocked_ready_screens_never_inject(screen: str) -> None:
    startup = _Startup("Task", None)
    assert startup.advance(screen) == b""
    assert startup.phase is _Phase.READY
    assert startup.port.message is None


def test_current_plan_indicator_must_be_in_footer_and_composer_empty() -> None:
    startup = _Startup("Task", None)
    assert startup.advance(current_screen("initial")) == b"/plan"
    assert startup.advance(current_screen("command")) == b"\r"
    for screen in [
        "Assistant mentioned Plan mode\n" + current_screen("initial"),
        current_screen("plan") + "Unknown modal\n\u203a 1. Continue",
        current_screen("question"),
        current_screen("final"),
    ]:
        assert startup.advance(screen) == b""
        assert startup.phase is _Phase.PLAN
        assert startup.port.message is None


def test_task_about_sign_in_is_not_mistaken_for_startup_login() -> None:
    startup = _Startup("Plan sign in improvements", None)
    activate(startup)
    assert startup.advance("\u203a Plan sign in improvements\ngpt Plan mode") == b"\r"
    assert (
        startup.advance("Plan sign in improvements\nWorking • esc to interrupt\ngpt Plan mode")
        == b""
    )
    assert startup.phase is _Phase.ACTIVE


def test_paste_confirmation_without_plan_footer_never_submits() -> None:
    startup = _Startup("Task", None)
    activate(startup)
    assert startup.advance("\u203a Task\nUnknown modal") == b""
    assert startup.phase is _Phase.PASTE


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


@pytest.mark.parametrize("with_warning", [False, True])
def test_wide_status_with_shortcut_still_submits_task(with_warning: bool) -> None:
    received: list[InteractiveLaunchEventKind] = []
    plan_screen = WIDE_PLAN if with_warning else WIDE_PLAN.split("    ⚠", 1)[0]
    status_line = plan_screen.splitlines()[-1]

    def handler(event: InteractiveLaunchEvent, session: InteractiveSession) -> None:
        received.append(event.kind)
        if event.kind is InteractiveLaunchEventKind.PLAN_READY:
            session.send_message("Plan issue #188")

    startup = _Startup(None, handler)
    assert startup.advance(READY) == b"/plan"
    assert startup.advance("\u203a /plan  switch to Plan mode\n\u203a /plan") == b"\r"
    assert startup.advance(plan_screen) == b"\x1b[200~Plan issue #188\x1b[201~"
    assert received == [InteractiveLaunchEventKind.PLAN_READY]
    assert startup.advance("\u203a Plan issue #188\n" + status_line) == b"\r"
    assert (
        startup.advance(
            "Working (0s • esc to interrupt)\n\u203a Ask Codex to do anything\n" + status_line
        )
        == b""
    )
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


@pytest.mark.parametrize("version", [None, (0, 153, 4)])
def test_terminal_versions_below_floor_rejected(version: tuple[int, int, int] | None) -> None:
    with (
        patch("crossby.utils.versioning.detect_binary_version", return_value=version),
        pytest.raises(PlanModeUnsupportedError, match=r"0\.154\.0 or newer"),
    ):
        CodexAdapter().build_launch_command(plan_mode=True)


@pytest.mark.parametrize("version", [(0, 154, 0), (0, 155, 0), (0, 157, 0), (0, 162, 1), (1, 0, 0)])
def test_terminal_startup_allowed_at_or_above_floor(version: tuple[int, int, int]) -> None:
    with patch("crossby.utils.versioning.detect_binary_version", return_value=version):
        command = CodexAdapter().build_launch_command(plan_mode=True)
    assert command[:3] == [sys.executable, "-m", "crossby.ai_tools.codex_terminal"]


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
