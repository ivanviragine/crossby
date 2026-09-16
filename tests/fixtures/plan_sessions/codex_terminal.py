"""Deterministic native terminal stand-in; no model, account, or network."""

import json
import os
import signal
import sys
import time
import tty
from pathlib import Path

root = Path(sys.argv[1])
scenario = sys.argv[2]
tty.setraw(sys.stdin.fileno())
(root / "pid").write_text(str(os.getpid()))


def display(text: str) -> None:
    sys.stdout.write("\x1b[2J\x1b[H" + text.replace("\n", "\r\n"))
    sys.stdout.flush()


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
if scenario == "reply":
    display("\x1b[6n")
    assert read_until(b"R") == b"\x1b[12;3R"
display("model: test\n\u203a Ask Codex to do anything\ntest")
assert read_until(b"n") == b"/plan"
display("\u203a /plan\n/plan switch to Plan mode")
assert os.read(0, 1) == b"\r"
display("model: test\n\u203a Ask Codex to do anything\ntest Plan mode")
paste = read_until(b"\x1b[201~")
assert paste.startswith(b"\x1b[200~")
message = paste[6:-6].decode()
(root / "message").write_text(message)
display(f"\u203a [Pasted Content {len(message)} chars]\ntest Plan mode")
assert os.read(0, 1) == b"\r"


def resized(signum: int, frame: object) -> None:
    (root / "size").write_text(json.dumps(list(os.get_terminal_size(0))))
    sys.stdout.write("RESIZED")
    sys.stdout.flush()


signal.signal(signal.SIGWINCH, resized)
display("Working (0s • esc to interrupt)\nNATIVE INTERACTION\ntest Plan mode")
answer = os.read(0, 1)
(root / "answer").write_bytes(answer)
if scenario == "exit_race":
    for fd in (0, 1, 2):
        os.close(fd)
    # Explicitly reproduce PTY EOF arriving before process exit.
    time.sleep(0.1)
raise SystemExit(7)
