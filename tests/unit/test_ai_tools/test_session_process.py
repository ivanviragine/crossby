"""Shared process-path coverage beyond the collected-plan compatibility suite."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import crossby.ai_tools.plan_process as plan_process
import crossby.ai_tools.session_process as session_process
from crossby.ai_tools.session_process import SessionProcessCancelledError, run_captured


def test_session_process_is_the_same_live_module_as_plan_process() -> None:
    assert session_process is plan_process
    assert session_process.run_captured is plan_process.run_captured


def test_captured_child_cancellation_aborts_then_kills_and_reaps(tmp_path: Path) -> None:
    cancel = threading.Event()
    aborts: list[str] = []

    def cancel_soon() -> None:
        time.sleep(0.05)
        cancel.set()

    threading.Thread(target=cancel_soon, daemon=True).start()
    started = time.monotonic()
    with pytest.raises(SessionProcessCancelledError):
        run_captured(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            timeout=10,
            cancel_event=cancel,
            native_abort=lambda: aborts.append("abort"),
        )

    assert aborts == ["abort"]
    assert time.monotonic() - started < 2


def test_existing_absolute_deadline_shortens_child_timeout(tmp_path: Path) -> None:
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_captured(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            timeout=10,
            deadline=started + 0.05,
        )
    assert time.monotonic() - started < 2
