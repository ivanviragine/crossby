"""Exercise actual nested PTYs, terminal restoration, and native input handoff."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Codex terminal adapter is POSIX")
pexpect = pytest.importorskip("pexpect")
FAKE = Path(__file__).parents[1] / "fixtures/plan_sessions/codex_terminal.py"
HARNESS = """
import json, sys, termios
from pathlib import Path
from crossby.ai_tools.codex_terminal import run_terminal_plan, TerminalStartupError, _Startup
root = Path(sys.argv[1])
saved = termios.tcgetattr(0)
advance = _Startup.advance
def observe_advance(startup, rendered):
    output = advance(startup, rendered)
    if "NATIVE_STALE_COMPOSER" in rendered and not (root / "stale-observed").exists():
        (root / "stale-observed").write_text(json.dumps({
            "phase": startup.phase.name, "output": output.hex(),
        }))
        print("CROSSBY_STALE_OBSERVED", flush=True)
    return output
_Startup.advance = observe_advance
def event(item, session):
    with (root / "events").open("a") as stream:
        stream.write(item.kind.value + "\\n")
    if item.kind.value == "message_submitted":
        print("CROSSBY_MESSAGE_SUBMITTED", flush=True)
    if item.kind.value == "plan_ready":
        session.send_message((root / "prompt").read_text())
try:
    code = run_terminal_plan(
        [sys.executable, sys.argv[2], str(root), sys.argv[3]], root,
        on_event=event, startup_timeout=2, transcript_path=root / "transcript",
    )
except TerminalStartupError as exc:
    (root / "error").write_text(str(exc))
    code = 91
finally:
    restored = termios.tcgetattr(0)
    # macOS sets the kernel PENDIN status bit on restoring canonical mode.
    restored[3] &= ~getattr(termios, "PENDIN", 0)
    saved[3] &= ~getattr(termios, "PENDIN", 0)
    (root / "restored").write_text(str(restored == saved))
raise SystemExit(code)
"""


def spawn(root: Path, scenario: str = "success", *, prompt: str | None = None):
    message = "Plan this café task.\n"
    if scenario == "current_body_heading":
        message += "Folder access\n"
    (root / "prompt").write_text(message * 1000 if prompt is None else prompt)
    return pexpect.spawn(
        sys.executable,
        ["-c", HARNESS, str(root), str(FAKE), scenario],
        encoding="utf-8",
        dimensions=(32, 120),
        timeout=5,
        echo=False,
    )


def failure_details(root: Path, child) -> str:
    error = (root / "error").read_text() if (root / "error").exists() else None
    events = (root / "events").read_text().splitlines() if (root / "events").exists() else []
    return f"terminal={child.before!r}\nstartup_error={error!r}\nevents={events!r}"


@pytest.mark.parametrize(
    "scenario",
    [
        "success",
        "current",
        "current_hidden",
        "current_hidden_no_plan",
        "current_pending",
        "current_notice",
        "current_coalesced",
        "current_chunked",
        "current_disappears",
        "current_ack_only",
        "current_body_heading",
        "trust",
        "current_trust",
        "reply",
        "exit_race",
    ],
)
def test_native_ui_prompt_once_resize_input_and_exit(tmp_path: Path, scenario: str) -> None:
    child = spawn(tmp_path, scenario)
    try:
        if scenario == "trust":
            child.expect_exact("Do you trust the contents of this directory?")
            child.send("y")
        if scenario == "current_trust":
            child.expect_exact("enter continue · esc quit")
            assert not (tmp_path / "events").exists()
            assert not (tmp_path / "message").exists()
            assert not (tmp_path / "trust-answer").exists()
            child.send("\r")
        if scenario == "reply":
            child.expect_exact("\x1b[6n")
            child.send("\x1b[12;")
            child.send("3R")
        child.expect_exact("NATIVE INTERACTION")
        if scenario == "current_pending":
            # A native banner can arrive before the Plan status is parsed.
            assert (tmp_path / "events").read_text().splitlines() == ["plan_ready"]
            (tmp_path / "show-status").touch()
        # Resize/input exercise ACTIVE handoff, so synchronize on Crossby's
        # acknowledgement rather than bytes printed by the native stand-in.
        child.expect_exact("CROSSBY_MESSAGE_SUBMITTED")
        child.setwinsize(40, 100)
        child.expect_exact("RESIZED")
        child.send("q")
        child.expect(pexpect.EOF)
        child.close()
        assert child.exitstatus == 7, failure_details(tmp_path, child)
        assert (tmp_path / "message").read_text() == (tmp_path / "prompt").read_text()
        assert (tmp_path / "events").read_text().splitlines() == ["plan_ready", "message_submitted"]
        assert (tmp_path / "answer").read_bytes() == b"q"
        if scenario == "current_trust":
            assert (tmp_path / "trust-answer").read_bytes() == b"\r"
        assert json.loads((tmp_path / "size").read_text()) == [100, 40]
        assert (tmp_path / "restored").read_text() == "True"
        assert "NATIVE INTERACTION" in (tmp_path / "transcript").read_text()
    except (pexpect.TIMEOUT, pexpect.EOF) as exc:
        pytest.fail(f"{exc}\n{failure_details(tmp_path, child)}", pytrace=False)
    finally:
        child.close(force=True)


@pytest.mark.parametrize("interrupt", [False, True])
def test_post_submission_still_requires_ack_before_keyboard_handoff(
    tmp_path: Path, interrupt: bool
) -> None:
    child = spawn(tmp_path, "current_ack_guard")
    try:
        child.expect_exact("NATIVE_PENDING")
        assert (tmp_path / "events").read_text().splitlines() == ["plan_ready"]
        if interrupt:
            child.send("q")
        child.expect(pexpect.EOF)
        child.close()
        assert child.exitstatus == 91, failure_details(tmp_path, child)
        assert (tmp_path / "message").read_text() == (tmp_path / "prompt").read_text()
        assert (tmp_path / "events").read_text().splitlines() == ["plan_ready"]
        assert not (tmp_path / "unexpected-input").exists()
        assert (tmp_path / "restored").read_text() == "True"
        error = (tmp_path / "error").read_text()
        assert ("Input interrupted" if interrupt else "timed out") in error
    finally:
        child.close(force=True)


@pytest.mark.parametrize(
    "message,ambiguous",
    [
        ("Ask", False),
        ("Ask Codex to do anything", True),
        ("Ask Codex to do anything\nDescribe this section.", True),
        ("Ask Codex to do anything about the project", False),
    ],
)
def test_stale_native_composer_cannot_submit_before_paste_receipt(
    tmp_path: Path, message: str, ambiguous: bool
) -> None:
    child = spawn(tmp_path, "current_stale_paste", prompt=message)
    try:
        child.expect_exact("CROSSBY_STALE_OBSERVED")
        # Parser barrier: output bytes alone do not prove the adapter observed
        # the stale frame. The wrapper only records results; it never changes them.
        assert json.loads((tmp_path / "stale-observed").read_text()) == {
            "phase": "PASTE",
            "output": "",
        }
        assert (tmp_path / "events").read_text().splitlines() == ["plan_ready"]
        assert not (tmp_path / "submission").exists()
        (tmp_path / "show-paste").touch()
        if ambiguous:
            child.expect(pexpect.EOF)
            child.close()
            assert child.exitstatus == 91, failure_details(tmp_path, child)
            assert (
                "timed out waiting for the initial task paste" in (tmp_path / "error").read_text()
            )
            assert not (tmp_path / "submission").exists()
            assert (tmp_path / "events").read_text().splitlines() == ["plan_ready"]
        else:
            child.expect_exact("CROSSBY_MESSAGE_SUBMITTED")
            child.send("q")
            child.expect(pexpect.EOF)
            child.close()
            assert child.exitstatus == 7, failure_details(tmp_path, child)
            assert (tmp_path / "submission").read_bytes() == b"\r"
            assert (tmp_path / "events").read_text().splitlines() == [
                "plan_ready",
                "message_submitted",
            ]
            assert (tmp_path / "answer").read_bytes() == b"q"
        assert (tmp_path / "message").read_text() == message
        assert not (tmp_path / "early-input").exists()
        assert (tmp_path / "restored").read_text() == "True"
    finally:
        child.close(force=True)


@pytest.mark.parametrize("scenario", ["unknown", "exit", "cancel"])
def test_startup_failure_does_not_submit_and_restores_terminal(
    tmp_path: Path, scenario: str
) -> None:
    child = spawn(tmp_path, "unknown" if scenario == "cancel" else scenario)
    try:
        if scenario == "cancel":
            child.expect_exact("UNRECOGNIZED STARTUP")
            child.sendcontrol("c")
        child.expect(pexpect.EOF)
        child.close()
        assert child.exitstatus == 91, child.before
        assert not (tmp_path / "message").exists()
        assert not (tmp_path / "events").exists()
        assert not (tmp_path / "unexpected-input").exists()
        assert (tmp_path / "restored").read_text() == "True"
        pid = int((tmp_path / "pid").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        child.close(force=True)
