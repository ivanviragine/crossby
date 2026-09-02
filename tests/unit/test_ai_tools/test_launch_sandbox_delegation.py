"""The public launch contract forwards sandbox selection to command building."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from crossby.ai_tools.codex import CodexAdapter


def test_launch_forwards_unsandboxed_request(tmp_path: Path) -> None:
    adapter = CodexAdapter()
    with (
        patch.object(adapter, "build_launch_command", return_value=["codex"]) as build,
        patch("crossby.utils.process.run_with_transcript", return_value=0),
    ):
        assert adapter.launch(tmp_path, sandbox=False) == 0

    assert build.call_args.kwargs["sandbox"] is False
