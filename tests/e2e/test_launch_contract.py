"""Deterministic contract tests for `crossby launch`."""

from __future__ import annotations

import pytest
import yaml
from tests.e2e._support import (
    assert_ordered_subsequence,
    find_invocation,
    install_mock_binary,
    install_mock_script,
    read_mock_log,
    run_crossby,
)

pytestmark = [pytest.mark.contract, pytest.mark.e2e_deterministic]


def test_launch_uses_resolved_config_and_creates_transcript_parent(e2e_context) -> None:
    session_id = "11111111-1111-1111-1111-111111111111"
    install_mock_binary(
        e2e_context.bin_dir,
        "claude",
        stdout=f"Total tokens: 123\nclaude --resume {session_id}\n",
    )
    install_mock_script(e2e_context.bin_dir)
    transcript = e2e_context.project / "artifacts" / "sessions" / "launch.txt"
    config = {
        "version": 1,
        "ai": {
            "default_tool": "claude",
            "default_model": "claude-sonnet-4.6",
            "effort": "high",
        },
        "permissions": {
            "allowed_commands": ["git:*"],
        },
    }
    (e2e_context.project / ".crossby.yml").write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_crossby(
        [
            "launch",
            str(e2e_context.project),
            "--prompt",
            "hello from test",
            "--transcript",
            str(transcript),
        ],
        cwd=e2e_context.project,
        env=e2e_context.env,
    )

    assert result.returncode == 0, result.stderr
    assert transcript.exists()
    transcript_text = transcript.read_text(encoding="utf-8")
    assert "Total tokens: 123" in transcript_text
    assert session_id in transcript_text
    assert "Tokens" in result.stdout
    assert "123" in result.stdout
    assert session_id in result.stdout
    invocation = find_invocation(e2e_context.log_file, "claude")
    assert invocation["cwd"] == str(e2e_context.project)
    assert_ordered_subsequence(
        invocation["argv"],
        [
            "hello from test",
            "--model",
            "claude-sonnet-4-6",
            "--effort",
            "high",
            "--allowedTools",
            "Bash(git:*)",
        ],
    )


def test_launch_translates_canonical_allowlist_for_copilot(e2e_context) -> None:
    install_mock_binary(e2e_context.bin_dir, "copilot")
    config = {
        "version": 1,
        "ai": {"default_tool": "copilot"},
        "permissions": {"allowed_commands": ["crossby:*", "./scripts/check.sh:*"]},
    }
    (e2e_context.project / ".crossby.yml").write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_crossby(
        ["launch", str(e2e_context.project), "--prompt", "smoke"],
        cwd=e2e_context.project,
        env=e2e_context.env,
    )

    assert result.returncode == 0, result.stderr
    invocation = find_invocation(e2e_context.log_file, "copilot")
    assert_ordered_subsequence(
        invocation["argv"],
        [
            "-i",
            "smoke",
            "--allow-tool",
            "shell(crossby:*)",
            "--allow-tool",
            "shell(./scripts/check.sh:*)",
        ],
    )


def test_launch_fails_fast_for_explicit_incompatible_model(e2e_context) -> None:
    result = run_crossby(
        ["launch", str(e2e_context.project), "--tool", "claude", "--model", "gpt-4o"],
        cwd=e2e_context.project,
        env=e2e_context.env,
    )

    assert result.returncode == 1
    assert "Model 'gpt-4o' is not compatible with claude" in result.stderr
    assert read_mock_log(e2e_context.log_file) == []


def test_launch_fails_fast_for_explicit_unsupported_effort(e2e_context) -> None:
    result = run_crossby(
        ["launch", str(e2e_context.project), "--tool", "copilot", "--effort", "high"],
        cwd=e2e_context.project,
        env=e2e_context.env,
    )

    assert result.returncode == 1
    assert "copilot does not support effort levels" in result.stderr
    assert read_mock_log(e2e_context.log_file) == []


def test_launch_fails_fast_for_invalid_explicit_effort(e2e_context) -> None:
    result = run_crossby(
        ["launch", str(e2e_context.project), "--tool", "claude", "--effort", "ultra"],
        cwd=e2e_context.project,
        env=e2e_context.env,
    )

    assert result.returncode == 1
    assert "Invalid effort level: 'ultra'" in result.stderr
    assert read_mock_log(e2e_context.log_file) == []


def test_launch_fails_fast_for_explicit_unsupported_yolo(e2e_context) -> None:
    result = run_crossby(
        ["launch", str(e2e_context.project), "--tool", "opencode", "--yolo"],
        cwd=e2e_context.project,
        env=e2e_context.env,
    )

    assert result.returncode == 1
    assert "opencode does not support YOLO mode" in result.stderr
    assert read_mock_log(e2e_context.log_file) == []


def test_launch_fails_fast_for_explicit_model_on_tool_without_model_flag(e2e_context) -> None:
    result = run_crossby(
        ["launch", str(e2e_context.project), "--tool", "vscode", "--model", "gpt-5"],
        cwd=e2e_context.project,
        env=e2e_context.env,
    )

    assert result.returncode == 1
    assert "Tool 'vscode' does not support explicit model selection" in result.stderr
    assert read_mock_log(e2e_context.log_file) == []
