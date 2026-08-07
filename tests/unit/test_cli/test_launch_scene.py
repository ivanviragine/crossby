"""Tests for ``crossby launch --scene`` — precedence, GUI/normalisation, fallback.

Adapter-level ``scene_launch_args`` behaviour lives in
``tests/unit/test_ai_tools/test_scene_launch.py``; this file exercises the CLI
wiring: scene validation, the explicit > scene > profile > defaults precedence,
GUI normalisation, and the two persistent-activation fallbacks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import yaml
from typer.testing import CliRunner

from crossby.cli.main import app
from crossby.models.ai import AIToolID, AIToolType
from crossby.sync.base import SyncConcern, SyncResult

runner = CliRunner()

_SCENE_CONFIG: dict[str, Any] = {
    "version": 1,
    "ai": {"default_tool": "claude"},
    "scenes": {
        "pr-review": {"mcp": {"include": ["github"]}},
        "with-profile": {"profile": "fast", "mcp": {"include": ["github"]}},
        "agents-only": {"agents": {"include": ["code-reviewer"]}},
    },
    "profiles": {
        "fast": {"tool": "cursor"},
        "slow": {"tool": "claude"},
    },
}


def _write_config(tmp_path: Path, config: dict[str, Any] | None = None) -> None:
    (tmp_path / ".crossby.yml").write_text(yaml.dump(config or _SCENE_CONFIG))


def _scene_adapter(
    *,
    tool_type: AIToolType = AIToolType.TERMINAL,
    supports_scene_launch: bool = True,
    scene_ready: bool = True,
    display_name: str = "Tool",
) -> MagicMock:
    adapter = MagicMock()
    adapter.launch.return_value = 0
    adapter.scene_launch_ready.return_value = scene_ready
    adapter.scene_launch_concerns.return_value = {"mcp"}
    adapter.capabilities.return_value = MagicMock(
        display_name=display_name,
        supports_initial_message=True,
        supports_trusted_dirs=False,
        supports_plan_mode=True,
        supports_accept_edits=True,
        supports_auto=True,
        supports_scene_launch=supports_scene_launch,
        tool_type=tool_type,
    )
    adapter.parse_transcript.return_value = MagicMock(total_tokens=None, session_id=None)
    return adapter


def _passthrough(tool: Any, model: Any, **kw: Any) -> tuple[Any, ...]:
    """Behave like confirm_ai_selection's non-TTY identity fast path."""
    return (
        tool,
        model,
        kw.get("resolved_effort"),
        kw.get("resolved_accept_edits", False),
        kw.get("resolved_auto", False),
        kw.get("resolved_yolo", False),
    )


class TestSceneValidation:
    def test_unknown_scene_exits_1(self, tmp_path: Path) -> None:
        _write_config(tmp_path)
        adapter = _scene_adapter()
        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=adapter),
            patch("crossby.ai_tools.base.AbstractAITool.detect_installed", return_value=[]),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "claude", "--scene", "nope"]
            )
        assert result.exit_code == 1
        assert "Unknown scene" in result.output

    def test_scene_with_resume_exits_1(self, tmp_path: Path) -> None:
        _write_config(tmp_path)
        adapter = _scene_adapter()
        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=adapter),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
        ):
            result = runner.invoke(
                app,
                [
                    "launch",
                    str(tmp_path),
                    "--tool",
                    "claude",
                    "--scene",
                    "pr-review",
                    "--resume",
                    "x",
                ],
            )
        assert result.exit_code == 1
        assert "cannot be combined with --resume" in result.output


class TestScenePassesContext:
    def test_terminal_tool_receives_scene_context(self, tmp_path: Path) -> None:
        _write_config(tmp_path)
        adapter = _scene_adapter(tool_type=AIToolType.TERMINAL)
        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=adapter),
            patch("crossby.ai_tools.base.AbstractAITool.detect_installed", return_value=[]),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "claude", "--scene", "pr-review"]
            )
        assert result.exit_code == 0, result.output
        _, kwargs = adapter.launch.call_args
        assert kwargs["scene"] is not None
        assert kwargs["scene"].name == "pr-review"


class TestGuiNormalisation:
    def test_gui_tool_warns_and_drops_scene(self, tmp_path: Path) -> None:
        _write_config(tmp_path)
        adapter = _scene_adapter(
            tool_type=AIToolType.GUI,
            supports_scene_launch=False,
            scene_ready=False,
            display_name="VS Code",
        )
        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=adapter),
            patch("crossby.ai_tools.base.AbstractAITool.detect_installed", return_value=[]),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "vscode", "--scene", "pr-review"]
            )
        assert result.exit_code == 0, result.output
        assert "GUI tool" in result.output
        _, kwargs = adapter.launch.call_args
        assert kwargs["scene"] is None


class TestPersistentFallback:
    def test_version_gate_falls_back_with_too_old_warning(self, tmp_path: Path) -> None:
        _write_config(tmp_path)
        # A tool that has a lever in principle but whose runtime gate failed.
        adapter = _scene_adapter(
            tool_type=AIToolType.TERMINAL,
            supports_scene_launch=True,
            scene_ready=False,
            display_name="Codex CLI",
        )
        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=adapter),
            patch("crossby.ai_tools.base.AbstractAITool.detect_installed", return_value=[]),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch("crossby.scenes.engine.apply_scene", return_value=[]) as apply_mock,
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "codex", "--scene", "pr-review"]
            )
        assert result.exit_code == 0, result.output
        assert "too old" in result.output
        apply_mock.assert_called_once()
        _, kwargs = adapter.launch.call_args
        assert kwargs["scene"] is None

    def test_no_lever_falls_back_with_warning(self, tmp_path: Path) -> None:
        _write_config(tmp_path)
        adapter = _scene_adapter(
            tool_type=AIToolType.TERMINAL,
            supports_scene_launch=False,
            scene_ready=False,
            display_name="Antigravity CLI",
        )
        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=adapter),
            patch("crossby.ai_tools.base.AbstractAITool.detect_installed", return_value=[]),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch("crossby.scenes.engine.apply_scene", return_value=[]) as apply_mock,
        ):
            result = runner.invoke(
                app,
                ["launch", str(tmp_path), "--tool", "antigravity-cli", "--scene", "pr-review"],
            )
        assert result.exit_code == 0, result.output
        assert "no session-scoped scene lever" in result.output
        apply_mock.assert_called_once()
        _, kwargs = adapter.launch.call_args
        assert kwargs["scene"] is None


class TestUnsupportedConcernWarning:
    def test_warns_when_scene_narrows_a_concern_the_tool_cannot_scope(self, tmp_path: Path) -> None:
        _write_config(tmp_path)
        # Cursor-like: a session lever, but only for MCP (scene_launch_concerns
        # returns {"mcp"} from _scene_adapter). The scene narrows agents.
        adapter = _scene_adapter(tool_type=AIToolType.TERMINAL, display_name="Cursor")
        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=adapter),
            patch("crossby.ai_tools.base.AbstractAITool.detect_installed", return_value=[]),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "cursor", "--scene", "agents-only"]
            )
        assert result.exit_code == 0, result.output
        assert "no session-scoped lever for agents" in result.output
        # It still applies the scene for the concerns it *can* scope.
        _, kwargs = adapter.launch.call_args
        assert kwargs["scene"] is not None


class TestScenePrecedence:
    def test_scene_supplies_default_profile(self, tmp_path: Path) -> None:
        """A scene's ``profile:`` selects the tool when no --profile is given."""
        _write_config(tmp_path)
        adapter = _scene_adapter(tool_type=AIToolType.TERMINAL)
        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=adapter) as get_mock,
            patch("crossby.ai_tools.base.AbstractAITool.detect_installed", return_value=[]),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
        ):
            result = runner.invoke(app, ["launch", str(tmp_path), "--scene", "with-profile"])
        assert result.exit_code == 0, result.output
        # scene's profile `fast` → tool cursor.
        assert get_mock.call_args.args[0] == "cursor"

    def test_explicit_profile_overrides_scene_profile(self, tmp_path: Path) -> None:
        """An explicit --profile wins over the scene-declared profile:."""
        _write_config(tmp_path)
        adapter = _scene_adapter(tool_type=AIToolType.TERMINAL)
        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=adapter) as get_mock,
            patch("crossby.ai_tools.base.AbstractAITool.detect_installed", return_value=[]),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--scene", "with-profile", "--profile", "slow"]
            )
        assert result.exit_code == 0, result.output
        # explicit profile `slow` → tool claude, overriding the scene's `fast`.
        assert get_mock.call_args.args[0] == "claude"

    def test_explicit_tool_overrides_scene_profile_tool(self, tmp_path: Path) -> None:
        _write_config(tmp_path)
        adapter = _scene_adapter(tool_type=AIToolType.TERMINAL)
        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=adapter) as get_mock,
            patch("crossby.ai_tools.base.AbstractAITool.detect_installed", return_value=[]),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--scene", "with-profile", "--tool", "codex"]
            )
        assert result.exit_code == 0, result.output
        assert get_mock.call_args.args[0] == "codex"


class TestFallbackReporting:
    """The persistent-activation fallback surfaces only genuinely-unsupported
    outcomes, and its message is result-dependent (no premature writes-config
    claim)."""

    def test_benign_skip_not_surfaced_and_message_not_premature(self, tmp_path: Path) -> None:
        _write_config(tmp_path)
        adapter = _scene_adapter(
            tool_type=AIToolType.TERMINAL,
            supports_scene_launch=False,
            scene_ready=False,
            display_name="Antigravity CLI",
        )
        benign = SyncResult(
            tool_id=AIToolID.ANTIGRAVITY_CLI,
            concern=SyncConcern.MCP,
            action="skipped",
            message="already applied; nothing to do",
        )
        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=adapter),
            patch("crossby.ai_tools.base.AbstractAITool.detect_installed", return_value=[]),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch("crossby.scenes.engine.apply_scene", return_value=[benign]),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "antigravity-cli", "--scene", "pr-review"]
            )
        assert result.exit_code == 0, result.output
        # A benign skip is not surfaced…
        assert "already applied" not in result.output
        # …and the fallback message never makes a premature writes-config claim.
        assert "this writes tool config files" not in result.output

    def test_unsupported_result_is_surfaced(self, tmp_path: Path) -> None:
        _write_config(tmp_path)
        adapter = _scene_adapter(
            tool_type=AIToolType.TERMINAL,
            supports_scene_launch=False,
            scene_ready=False,
            display_name="OpenCode",
        )
        unsupported = SyncResult(
            tool_id=AIToolID.OPENCODE,
            concern=SyncConcern.MCP,
            action="skipped",
            message="opencode has no per-server disable key; 1 deselected server(s) remain enabled",
            unsupported=True,
        )
        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=adapter),
            patch("crossby.ai_tools.base.AbstractAITool.detect_installed", return_value=[]),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch("crossby.scenes.engine.apply_scene", return_value=[unsupported]),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "opencode", "--scene", "pr-review"]
            )
        assert result.exit_code == 0, result.output
        # Collapse Rich's line-wrapping before matching the surfaced message.
        assert "remain enabled" in " ".join(result.output.split())


class TestOpenCodeRealAdapterFallback:
    """Integration test through the real OpenCodeAdapter (no session lever)."""

    def test_deselected_mcp_warns_and_no_env_leak(self, tmp_path: Path) -> None:
        config: dict[str, Any] = {
            "version": 1,
            "ai": {"default_tool": "opencode"},
            "scenes": {"pr-review": {"mcp": {"include": ["github"]}}},
        }
        _write_config(tmp_path, config)
        (tmp_path / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"github": {"command": "gh"}, "linear": {"command": "lin"}}})
        )

        captured: dict[str, Any] = {}

        def fake_run(cmd: list[str], transcript_path: Any, cwd: Any = None, env: Any = None) -> int:
            captured["cmd"] = cmd
            captured["env"] = env
            return 0

        with (
            patch(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                return_value=[AIToolID.OPENCODE],
            ),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch("crossby.utils.process.run_with_transcript", side_effect=fake_run),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "opencode", "--scene", "pr-review"]
            )
        assert result.exit_code == 0, result.output
        # OpenCode honestly reports it has no session lever and that the
        # deselected server stays enabled — no silent isolation claim. Collapse
        # Rich's line-wrapping before matching.
        normalized = " ".join(result.output.split())
        assert "no session-scoped scene lever" in normalized
        assert "remain enabled" in normalized
        # No env leak: OPENCODE_CONFIG was never set for the child process.
        assert captured["env"] is None or "OPENCODE_CONFIG" not in captured["env"]


class TestContainmentAbort:
    """A containment violation aborts the launch cleanly with no fallback and no
    partial artefact."""

    def test_symlinked_crossby_aborts_launch(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        config: dict[str, Any] = {
            "version": 1,
            "ai": {"default_tool": "claude"},
            "scenes": {"pr-review": {"mcp": {"include": ["github"]}}},
        }
        (project / ".crossby.yml").write_text(yaml.dump(config))
        (project / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"github": {"command": "gh"}, "linear": {"command": "lin"}}})
        )
        # Symlink the .crossby dir out of the project → any artefact write would
        # escape the project root.
        outside = tmp_path / "outside"
        outside.mkdir()
        (project / ".crossby").symlink_to(outside, target_is_directory=True)

        apply_mock = MagicMock(return_value=[])

        def fake_run(cmd: list[str], transcript_path: Any, cwd: Any = None, env: Any = None) -> int:
            return 0

        with (
            patch(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                return_value=[AIToolID.CLAUDE],
            ),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch("crossby.scenes.engine.apply_scene", apply_mock),
            patch("crossby.utils.process.run_with_transcript", side_effect=fake_run),
        ):
            result = runner.invoke(
                app, ["launch", str(project), "--tool", "claude", "--scene", "pr-review"]
            )
        assert result.exit_code == 1, result.output
        # No persistent-write fallback after a containment violation…
        apply_mock.assert_not_called()
        # …and no partial artefact escaped into the symlink target.
        assert list(outside.iterdir()) == []
