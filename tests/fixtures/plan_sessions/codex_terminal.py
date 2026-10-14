"""Deterministic native terminal stand-in; no model, account, or network."""

import json
import os
import select
import signal
import sys
import time
import tty
from pathlib import Path

root = Path(sys.argv[1])
scenario = sys.argv[2]
current = scenario.startswith("current")
screens = Path(__file__).with_name("codex_0_162_1")
tty.setraw(sys.stdin.fileno())
(root / "pid").write_text(str(os.getpid()))


def display(text: str) -> None:
    sys.stdout.write("\x1b[2J\x1b[H" + text.replace("\n", "\r\n"))
    sys.stdout.flush()


def framed(text: str) -> bytes:
    return b"\x1b[?2026h\x1b[2J\x1b[H" + text.replace("\n", "\r\n").encode() + b"\x1b[?2026l"


def read_until(suffix: bytes) -> bytes:
    data = b""
    while not data.endswith(suffix):
        chunk = os.read(0, 1)
        if not chunk:
            raise SystemExit(2)
        data += chunk
    return data


if scenario == "exit":
    raise SystemExit(4)
if scenario == "unknown":
    display("UNRECOGNIZED STARTUP")
    (root / "unexpected-input").write_bytes(os.read(0, 100))
    raise SystemExit(3)
if scenario == "trust":
    display("Do you trust the contents of this directory?")
    assert os.read(0, 1) == b"y"
if scenario == "current_trust":
    display((screens / "trust.txt").read_text())
    answer = os.read(0, 1)
    (root / "trust-answer").write_bytes(answer)
    assert answer == b"\r"
if scenario == "reply":
    display("\x1b[6n")
    assert read_until(b"R") == b"\x1b[12;3R"
display(
    (screens / "initial.txt").read_text()
    if current
    else "model: test\n\u203a Ask Codex to do anything\ntest"
)
assert read_until(b"n") == b"/plan"
display(
    (screens / "command.txt").read_text() if current else "\u203a /plan\n/plan switch to Plan mode"
)
assert os.read(0, 1) == b"\r"

if scenario in {"current_notice", "current_coalesced", "current_chunked"}:
    notice_screen = (screens / "plan-notice-no-footer.txt").read_text().splitlines()
    notice_screen[-1] = (screens / "initial-minimal.txt").read_text().splitlines()[-1]
    data = framed("\n".join(notice_screen))
    if scenario != "current_notice":
        data = framed((screens / "plan.txt").read_text()) + data
    if scenario == "current_chunked":
        for byte in data:
            os.write(1, bytes([byte]))
    else:
        os.write(1, data)
else:
    display(
        (screens / "plan.txt").read_text()
        if current
        else "model: test\n\u203a Ask Codex to do anything\ntest Plan mode"
    )
paste = read_until(b"\x1b[201~")
assert paste.startswith(b"\x1b[200~")
message = paste[6:-6].decode()
(root / "message").write_text(message)
status = (screens / "plan.txt").read_text().splitlines()[-1] if current else "test Plan mode"
if scenario in {"current_notice", "current_coalesced", "current_chunked", "current_disappears"}:
    status = (screens / "initial-minimal.txt").read_text().splitlines()[-1]
paste_screen = f"\u203a [Pasted Content {len(message)} chars]\n{status}"
if scenario == "current_stale_paste":
    status = (screens / "initial-minimal.txt").read_text().splitlines()[-1]
    display("NATIVE_STALE_COMPOSER\n\u203a Ask Codex to do anything\n" + status)
    deadline = time.monotonic() + 2
    while not (root / "show-paste").exists():
        if select.select([0], [], [], 0.01)[0]:
            (root / "early-input").write_bytes(os.read(0, 100))
            raise SystemExit(3)
        if time.monotonic() >= deadline:
            raise RuntimeError("Parent did not release the stale composer")
    paste_screen = "\u203a " + message + "\n" + status
if scenario in {"current_notice", "current_coalesced", "current_chunked"}:
    os.write(1, framed(paste_screen))
else:
    display(paste_screen)
submission = os.read(0, 1)
if scenario == "current_stale_paste":
    (root / "submission").write_bytes(submission)
assert submission == b"\r"

if scenario == "current_ack_guard":
    status = (screens / "initial-minimal.txt").read_text().splitlines()[-1]
    display(
        "NATIVE_PENDING\nTask says esc to interrupt\n\u203a Ask Codex to do anything\n" + status
    )
    (root / "unexpected-input").write_bytes(os.read(0, 1))
    raise SystemExit(3)


def resized(signum: int, frame: object) -> None:
    (root / "size").write_text(json.dumps(list(os.get_terminal_size(0))))
    sys.stdout.write("RESIZED")
    sys.stdout.flush()


signal.signal(signal.SIGWINCH, resized)
if scenario in {"current_ack_only", "current_hidden_no_plan"}:
    status = (screens / "initial-minimal.txt").read_text().splitlines()[-1]
if scenario == "current_pending":
    display("Working (0s • esc to interrupt)\nNATIVE INTERACTION")
    # Explicit parent-controlled boundary: no complete working status yet.
    deadline = time.monotonic() + 2
    while not (root / "show-status").exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("Parent did not release the pending Plan status")
        time.sleep(0.01)
working_screen = "NATIVE INTERACTION\nWorking (0s • esc to interrupt)\n"
if scenario == "current_body_heading":
    working_screen = "\u203a Task echo\nFolder access\n" + working_screen
if current:
    if scenario not in {"current_hidden", "current_hidden_no_plan"}:
        working_screen += "\u203a Ask Codex to do anything\n"
else:
    # Preserve the original legacy stand-in's no-composer layout. This banner
    # is a test transport notification, not native acknowledgement evidence.
    working_screen = "Working (0s • esc to interrupt)\nNATIVE INTERACTION\n"
working_screen += status
if scenario in {"current_notice", "current_coalesced", "current_chunked"}:
    os.write(1, framed(working_screen))
else:
    display(working_screen)
answer = os.read(0, 1)
(root / "answer").write_bytes(answer)
if scenario == "exit_race":
    for fd in (0, 1, 2):
        os.close(fd)
    # Explicitly reproduce PTY EOF arriving before process exit.
    time.sleep(0.1)
raise SystemExit(7)
