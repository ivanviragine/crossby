"""Tests for ``run_with_transcript`` — env forwarding on every execution path."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from crossby.utils.process import run_with_transcript


def _ok(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(returncode=0)


class TestRunWithTranscriptEnv:
    def test_plain_path_forwards_env(self) -> None:
        """No transcript → plain subprocess.run receives the env verbatim."""
        env = {"OPENCODE_CONFIG": "/x/opencode.json", "PATH": "/bin"}
        with patch("crossby.utils.process.subprocess.run", side_effect=_ok) as run_mock:
            rc = run_with_transcript(["opencode"], None, env=env)
        assert rc == 0
        _, kwargs = run_mock.call_args
        assert kwargs["env"] == env

    def test_plain_path_default_env_is_none(self) -> None:
        """Default (no env) inherits the parent environment (env=None)."""
        with patch("crossby.utils.process.subprocess.run", side_effect=_ok) as run_mock:
            run_with_transcript(["opencode"], None)
        _, kwargs = run_mock.call_args
        assert kwargs["env"] is None

    def test_script_path_forwards_env(self, tmp_path: Any) -> None:
        """With a transcript, the ``script`` wrapper still receives the env.

        The version probe must NOT carry the env (it is a pure capability check);
        the actual exec must.
        """
        env = {"CURSOR_CONFIG_DIR": "/x/cursor", "PATH": "/bin"}
        calls: list[dict[str, Any]] = []

        def fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
            calls.append({"cmd": cmd, **kwargs})
            # GNU `script --version` returns 0 so the GNU branch is taken.
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with (
            patch("crossby.utils.process.shutil.which", return_value="/usr/bin/script"),
            patch("crossby.utils.process.subprocess.run", side_effect=fake_run),
        ):
            run_with_transcript(["cursor"], tmp_path / "t.txt", env=env)

        # Three subprocess calls at most: version probe, then the script exec.
        version_call = next(c for c in calls if c["cmd"][:2] == ["script", "--version"])
        exec_call = next(c for c in calls if c["cmd"][:1] == ["script"] and c is not version_call)
        assert "env" not in version_call or version_call.get("env") is None
        assert exec_call["env"] == env
