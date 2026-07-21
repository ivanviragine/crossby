"""Antigravity CLI launch-command coverage: yolo, effort, initial message, build_launch_command."""

from __future__ import annotations

from pathlib import Path

from crossby.ai_tools.antigravity_cli import AntigravityCLIAdapter
from crossby.models.ai import EffortLevel, TokenUsage


class TestAntigravityCLIYolo:
    def test_yolo_skips_permissions_and_keeps_sandbox(self) -> None:
        args = AntigravityCLIAdapter().yolo_args()
        assert args == ["--dangerously-skip-permissions", "--sandbox"]

    def test_supports_yolo_capability(self) -> None:
        assert AntigravityCLIAdapter().capabilities().supports_yolo is True


class TestAntigravityCLIEffort:
    def test_low_medium_high_map_1to1(self) -> None:
        adapter = AntigravityCLIAdapter()
        assert adapter.effort_args(EffortLevel.LOW) == ["--effort", "low"]
        assert adapter.effort_args(EffortLevel.MEDIUM) == ["--effort", "medium"]
        assert adapter.effort_args(EffortLevel.HIGH) == ["--effort", "high"]

    def test_xhigh_and_max_collapse_to_high(self) -> None:
        adapter = AntigravityCLIAdapter()
        assert adapter.effort_args(EffortLevel.XHIGH) == ["--effort", "high"]
        assert adapter.effort_args(EffortLevel.MAX) == ["--effort", "high"]

    def test_supports_effort_capability(self) -> None:
        assert AntigravityCLIAdapter().capabilities().supports_effort is True


class TestAntigravityCLIInitialMessage:
    def test_initial_message_uses_prompt_interactive_flag(self) -> None:
        args = AntigravityCLIAdapter().initial_message_args("do the thing")
        assert args == ["--prompt-interactive", "do the thing"]

    def test_supports_initial_message_capability(self) -> None:
        assert AntigravityCLIAdapter().capabilities().supports_initial_message is True


class TestAntigravityCLIPlanMode:
    def test_plan_dir_args_uses_add_dir(self) -> None:
        args = AntigravityCLIAdapter().plan_dir_args("/tmp/plan")
        assert args == ["--add-dir", "/tmp/plan"]


class TestAntigravityCLIParseTranscript:
    def test_parse_transcript_always_empty(self, tmp_path: Path) -> None:
        transcript = tmp_path / "session.txt"
        transcript.write_text("anything at all, this is never parsed")
        usage = AntigravityCLIAdapter().parse_transcript(transcript)
        assert usage == TokenUsage()


class TestAntigravityCLIBuildLaunchCommand:
    def test_combines_model_effort_and_yolo(self) -> None:
        cmd = AntigravityCLIAdapter().build_launch_command(
            model="gemini-3.6-flash-high",
            effort=EffortLevel.HIGH,
            yolo=True,
        )
        assert cmd[0] == "agy"
        assert "--model" in cmd
        assert cmd[cmd.index("--model") + 1] == "gemini-3.6-flash-high"
        assert "--effort" in cmd
        assert cmd[cmd.index("--effort") + 1] == "high"
        assert "--dangerously-skip-permissions" in cmd
        assert "--sandbox" in cmd

    def test_initial_message_is_first_positional(self) -> None:
        cmd = AntigravityCLIAdapter().build_launch_command(initial_message="hello there")
        assert cmd == ["agy", "--prompt-interactive", "hello there"]
