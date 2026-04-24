"""End-to-end test for `crossby handoff` — Claude source → Codex target.

Mocks only the external-process boundaries: `AbstractAITool.detect_installed`
(so both adapters pass the installed check) and `subprocess.run` in the
summarizer + CLI launch path. The Claude reader, the handoff writer, path
resolution, and the Typer CLI itself all run for real.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from crossby.ai_tools.base import AbstractAITool
from crossby.cli.main import app
from crossby.models.ai import AIToolID

_JSON_SUMMARY = (
    '{"current_task": "Refactor auth module", '
    '"key_decisions": ["drop shared cache"], '
    '"modified_files": ["auth.py"], '
    '"blockers": [], '
    '"next_steps": ["run migration"], '
    '"critical_context": "cache is load-bearing"}'
)


def _stage_claude_session(
    fixtures_dir: Path, fake_home: Path, project_root: Path
) -> None:
    """Install a real claude_happy.jsonl fixture into the fake home dir."""
    encoded = str(project_root).replace("/", "-").replace(".", "-")
    session_dir = fake_home / ".claude" / "projects" / encoded
    session_dir.mkdir(parents=True)
    shutil.copy(fixtures_dir / "claude_happy.jsonl", session_dir / "abc.jsonl")


def _make_unified_subprocess_run(launched: dict[str, object]):  # type: ignore[no-untyped-def]
    """Return a fake subprocess.run that dispatches by first-arg binary.

    ``subprocess`` is a shared module singleton — patching ``summarizer.subprocess.run``
    and ``cli.handoff.subprocess.run`` separately collides. A single unified fake
    that inspects the command is simpler and more correct.
    """

    def _run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        binary = cmd[0] if cmd else ""
        if binary == "claude":
            # Summarizer invocation — return canned JSON.
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=_JSON_SUMMARY, stderr=""
            )
        # Target launch invocation (codex, copilot, etc.) — capture & return 0.
        launched["cmd"] = list(cmd)
        launched["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    return _run


def test_handoff_claude_to_codex_writes_file_and_launches_under_arg_cap(
    fixtures_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "proj"
    project_root.mkdir()
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    _stage_claude_session(fixtures_dir, fake_home, project_root.resolve())

    launched: dict[str, object] = {}
    fake_run = _make_unified_subprocess_run(launched)

    runner = CliRunner()
    with patch.object(
        AbstractAITool,
        "detect_installed",
        return_value=[AIToolID.CLAUDE, AIToolID.CODEX],
    ):
        with patch("subprocess.run", side_effect=fake_run):
            result = runner.invoke(
                app,
                [
                    "handoff",
                    "--from",
                    "claude",
                    "--to",
                    "codex",
                    "--path",
                    str(project_root),
                ],
            )

    assert result.exit_code == 0, result.output

    # 1) Handoff file exists under .crossby/handoffs with expected content.
    handoffs = list((project_root / ".crossby" / "handoffs").glob("HANDOFF-*.md"))
    assert len(handoffs) == 1
    body = handoffs[0].read_text(encoding="utf-8")
    assert "Refactor auth module" in body
    assert "## Key Decisions" in body
    assert "drop shared cache" in body

    # 2) Launch command was assembled for Codex — codex binary + prompt arg.
    assert "cmd" in launched, "codex launch was not invoked"
    cmd = launched["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == "codex"
    # Codex takes the initial message as a positional arg — find it.
    prompt_args = [a for a in cmd if isinstance(a, str) and "HANDOFF-" in a]
    assert prompt_args, f"launch command missing handoff path arg: {cmd}"

    # 3) Command total byte size is well under the 4KB safety cap.
    total_bytes = sum(len(a.encode("utf-8")) for a in cmd)
    assert total_bytes < 4 * 1024, f"launch command too large: {total_bytes} bytes"

    # 4) Launch ran in the project root.
    assert launched["cwd"] == project_root.resolve()


def test_handoff_no_launch_skips_target(
    fixtures_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "proj"
    project_root.mkdir()
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    _stage_claude_session(fixtures_dir, fake_home, project_root.resolve())

    launched: dict[str, object] = {}
    fake_run = _make_unified_subprocess_run(launched)

    runner = CliRunner()
    with patch.object(
        AbstractAITool,
        "detect_installed",
        return_value=[AIToolID.CLAUDE, AIToolID.CODEX],
    ):
        with patch("subprocess.run", side_effect=fake_run):
            result = runner.invoke(
                app,
                [
                    "handoff",
                    "--from",
                    "claude",
                    "--to",
                    "codex",
                    "--path",
                    str(project_root),
                    "--no-launch",
                ],
            )

    assert result.exit_code == 0, result.output
    assert "cmd" not in launched
    handoffs = list((project_root / ".crossby" / "handoffs").glob("HANDOFF-*.md"))
    assert len(handoffs) == 1


def test_handoff_rejects_unsupported_source() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["handoff", "--from", "gemini", "--to", "claude"]
    )
    assert result.exit_code == 1
    assert "gemini" in result.output.lower() or "Gemini" in result.output


def test_handoff_errors_when_session_id_not_found(
    fixtures_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "proj"
    project_root.mkdir()
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    _stage_claude_session(fixtures_dir, fake_home, project_root.resolve())

    runner = CliRunner()
    with patch.object(
        AbstractAITool,
        "detect_installed",
        return_value=[AIToolID.CLAUDE, AIToolID.CODEX],
    ):
        result = runner.invoke(
            app,
            [
                "handoff",
                "--from",
                "claude",
                "--to",
                "codex",
                "--path",
                str(project_root),
                "--session-id",
                "does-not-exist",
            ],
        )
    assert result.exit_code == 1
    assert "does-not-exist" in result.output


def _make_raw_subprocess_run(launched: dict[str, object], summary: str):  # type: ignore[no-untyped-def]
    """Fake subprocess.run that returns free-form text for summarizer, captures launch."""

    def _run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        binary = cmd[0] if cmd else ""
        if binary == "claude":
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=summary, stderr=""
            )
        launched["cmd"] = list(cmd)
        launched["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    return _run


def test_handoff_with_cc_compact_preset_writes_raw_body(
    fixtures_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "proj"
    project_root.mkdir()
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    _stage_claude_session(fixtures_dir, fake_home, project_root.resolve())

    launched: dict[str, object] = {}
    raw_summary = "<analysis>trace</analysis>\n<summary>ship it</summary>"
    fake_run = _make_raw_subprocess_run(launched, raw_summary)

    runner = CliRunner()
    with patch.object(
        AbstractAITool,
        "detect_installed",
        return_value=[AIToolID.CLAUDE, AIToolID.CODEX],
    ):
        with patch("subprocess.run", side_effect=fake_run):
            result = runner.invoke(
                app,
                [
                    "handoff",
                    "--from", "claude",
                    "--to", "codex",
                    "--path", str(project_root),
                    "--no-launch",
                    "--prompt-preset", "cc-compact",
                ],
            )

    assert result.exit_code == 0, result.output
    handoffs = list((project_root / ".crossby" / "handoffs").glob("HANDOFF-*.md"))
    assert len(handoffs) == 1
    body = handoffs[0].read_text(encoding="utf-8")
    assert "(raw)" in body
    assert "**Prompt**: cc-compact" in body
    assert raw_summary in body
    # None of the structured headings should appear — raw mode skips schema.
    assert "## Current Task" not in body


def test_handoff_errors_when_prompt_and_preset_both_set(tmp_path: Path) -> None:
    custom = tmp_path / "custom.md"
    custom.write_text("my prompt", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "handoff",
            "--from", "claude",
            "--to", "codex",
            "--prompt", str(custom),
            "--prompt-preset", "cc-compact",
        ],
    )
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


def test_handoff_errors_when_custom_prompt_missing(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "handoff",
            "--from", "claude",
            "--to", "codex",
            "--prompt", str(tmp_path / "nonexistent.md"),
        ],
    )
    assert result.exit_code == 1
    assert "not found" in result.output.lower()
