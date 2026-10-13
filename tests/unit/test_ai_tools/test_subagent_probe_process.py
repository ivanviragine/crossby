"""Real probe process ownership, without any installed AI-tool execution."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from crossby.ai_tools import (
    AbstractAITool,
    SubagentSpawnReason,
    SubagentSpawnState,
    codex_readiness,
)


@pytest.mark.skipif(
    os.name != "posix", reason="fixture uses POSIX executable scripts/process groups"
)
def test_read_probe_timeout_reaps_its_descendant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "codex"
    child_pid = tmp_path / "child.pid"
    methods = tmp_path / "methods.txt"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "if '--version' in sys.argv:\n"
        "    print('codex-cli 0.162.1')\n"
        "    sys.exit(0)\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "Path(os.environ['CHILD_PID']).write_text(str(child.pid))\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    with open(os.environ['METHODS'], 'a') as output:\n"
        "        output.write(request['method'] + '\\n')\n"
        "    if request['method'] == 'initialize':\n"
        "        print(json.dumps({'id': request['id'], 'result': {}}), flush=True)\n"
        "    elif request['method'] == 'config/read':\n"
        "        time.sleep(30)\n"
    )
    executable.chmod(0o755)
    monkeypatch.setattr(codex_readiness, "PROBE_SECONDS", 2.0)
    started = time.monotonic()
    status = AbstractAITool.get("codex").subagent_spawning_status(
        working_dir=tmp_path,
        environment={
            "PATH": str(tmp_path),
            "HOME": str(tmp_path),
            "CHILD_PID": str(child_pid),
            "METHODS": str(methods),
        },
    )
    assert status.state is SubagentSpawnState.UNKNOWN
    assert status.reason_code is SubagentSpawnReason.PROBE_TIMEOUT
    assert time.monotonic() - started < 4
    assert methods.read_text().splitlines() == ["initialize", "initialized", "config/read"]
    pid = int(child_pid.read_text())
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail("the timed-out readiness probe left its descendant alive")
