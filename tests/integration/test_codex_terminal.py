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
from crossby.ai_tools.codex_terminal import run_terminal_plan, TerminalStartupError
root = Path(sys.argv[1])
saved = termios.tcgetattr(0)
def event(item, session):
    with (root / "events").open("a") as stream:
        stream.write(item.kind.value + "\\n")
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


def spawn(root: Path, scenario: str = "success"):
    (root / "prompt").write_text("Plan this café task.\n" * 1000)
    return pexpect.spawn(
        sys.executable,
        ["-c", HARNESS, str(root), str(FAKE), scenario],
        encoding="utf-8",
        dimensions=(32, 120),
        timeout=5,
        echo=False,
    )


@pytest.mark.parametrize("scenario", ["success", "trust", "reply"])
def test_native_ui_prompt_once_resize_input_and_exit(tmp_path: Path, scenario: str) -> None:
    child = spawn(tmp_path, scenario)
    try:
        if scenario == "trust":
            child.expect_exact("Do you trust the contents of this directory?")
            child.send("y")
        if scenario == "reply":
            child.expect_exact("\x1b[6n")
            child.send("\x1b[12;")
            child.send("3R")
        child.expect_exact("NATIVE INTERACTION")
        child.setwinsize(40, 100)
        child.expect_exact("RESIZED")
        child.send("q")
        child.expect(pexpect.EOF)
        child.close()
        assert child.exitstatus == 7, child.before
        assert (tmp_path / "message").read_text() == (tmp_path / "prompt").read_text()
        assert (tmp_path / "events").read_text().splitlines() == ["plan_ready", "message_submitted"]
        assert (tmp_path / "answer").read_bytes() == b"q"
        assert json.loads((tmp_path / "size").read_text()) == [100, 40]
        assert (tmp_path / "restored").read_text() == "True"
        assert "NATIVE INTERACTION" in (tmp_path / "transcript").read_text()
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
