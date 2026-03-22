"""Shared helpers for deterministic CROSSBY subprocess contract tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_RUN_ENV_ALLOWLIST = (
    "HOME",
    "LANG",
    "LC_ALL",
    "LOGNAME",
    "PATH",
    "PATHEXT",
    "SHELL",
    "SYSTEMROOT",
    "TEMP",
    "TERM",
    "TMP",
    "TMPDIR",
    "USER",
)


@dataclass
class E2EContext:
    """Filesystem and environment for deterministic subprocess tests."""

    root: Path
    project: Path
    home: Path
    bin_dir: Path
    log_file: Path
    env: dict[str, str]


def make_e2e_context(tmp_path: Path) -> E2EContext:
    root = tmp_path / "e2e"
    project = root / "project"
    home = root / "home"
    bin_dir = root / "bin"
    log_file = root / "mock-cli.jsonl"

    project.mkdir(parents=True)
    home.mkdir(parents=True)
    bin_dir.mkdir(parents=True)

    env = {
        "HOME": str(home),
        "PATH": str(bin_dir),
        "CROSSBY_MOCK_LOG": str(log_file),
    }
    return E2EContext(
        root=root,
        project=project,
        home=home,
        bin_dir=bin_dir,
        log_file=log_file,
        env=env,
    )


def run_crossby(
    args: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    """Run CROSSBY as a subprocess with a deterministic environment."""
    run_env = {key: os.environ[key] for key in _RUN_ENV_ALLOWLIST if key in os.environ}
    run_env.setdefault("LANG", "C.UTF-8")
    run_env.setdefault("LC_ALL", run_env["LANG"])
    run_env.setdefault("NO_COLOR", "1")
    run_env.setdefault("PYTHONIOENCODING", "utf-8")
    run_env.setdefault("PYTHONUTF8", "1")
    run_env.setdefault("TERM", "dumb")

    pythonpath = str(REPO_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH") if env else run_env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath = os.pathsep.join([pythonpath, existing_pythonpath])
    run_env["PYTHONPATH"] = pythonpath

    if env:
        run_env.update(env)

    return subprocess.run(
        [sys.executable, "-c", "from crossby.cli.main import cli_main; cli_main()", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=run_env,
    )


def install_mock_binary(
    bin_dir: Path,
    name: str,
    *,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> Path:
    """Install a tiny fake CLI binary that logs argv/cwd to JSONL."""
    path = bin_dir / name
    script = f"""#!{sys.executable}
import json
import os
import sys

record = {{
    "binary": {name!r},
    "argv": sys.argv[1:],
    "cwd": os.getcwd(),
}}
log_path = os.environ.get("CROSSBY_MOCK_LOG")
if log_path:
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\\n")
if {stdout!r}:
    sys.stdout.write({stdout!r})
if {stderr!r}:
    sys.stderr.write({stderr!r})
raise SystemExit({exit_code})
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


def read_mock_log(log_file: Path) -> list[dict[str, Any]]:
    """Read the fake CLI JSONL log."""
    if not log_file.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in log_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def find_invocation(log_file: Path, binary: str) -> dict[str, Any]:
    """Return the first logged invocation for *binary*."""
    for row in read_mock_log(log_file):
        if row.get("binary") == binary:
            return row
    raise AssertionError(f"No invocation for {binary!r} found in {log_file}")


def assert_ordered_subsequence(actual: list[str], expected: list[str]) -> None:
    """Assert that *expected* appears in *actual* with preserved order."""
    if not expected:
        return
    index = 0
    for token in actual:
        if token == expected[index]:
            index += 1
            if index == len(expected):
                return
    raise AssertionError(f"Expected ordered subsequence {expected!r} in {actual!r}")
