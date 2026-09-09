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

import pytest
import yaml
from typer.testing import CliRunner

from crossby.cli.main import app
from crossby.models.ai import AIToolID, AIToolType
from crossby.scenes.state import SCENE_STATE_PATH
from crossby.sync.base import SyncConcern, SyncResult
from tests.unit.test_scenes.conftest import populate_project, read_json

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

    def test_subdir_launch_scene_context_roots_at_config_root(self, tmp_path: Path) -> None:
        """The session-scoped ``SceneLaunchContext.project_root`` must be the
        config root, not the invocation subdirectory — adapters (e.g. Claude)
        read it back to build the disable set (``scene.project_root /
        SKILLS_DIR[...]``), and that read has to match the inventory
        ``resolve_scene`` used or the scene would silently narrow nothing.
        """
        _write_config(tmp_path)
        sub = tmp_path / "packages" / "app"
        sub.mkdir(parents=True)
        adapter = _scene_adapter(tool_type=AIToolType.TERMINAL)
        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=adapter),
            patch("crossby.ai_tools.base.AbstractAITool.detect_installed", return_value=[]),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
        ):
            result = runner.invoke(
                app, ["launch", str(sub), "--tool", "claude", "--scene", "pr-review"]
            )
        assert result.exit_code == 0, result.output
        _, kwargs = adapter.launch.call_args
        assert kwargs["scene"].project_root == tmp_path.resolve()
        assert not (sub / ".crossby").exists()
        # The subprocess itself still runs in the invocation directory.
        assert kwargs["working_dir"] == sub.resolve()


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

    def test_profile_collision_surfaces_partial_activation_before_launch(
        self, tmp_path: Path
    ) -> None:
        from crossby.scenes.launch import SceneLaunchFallbackError
        from crossby.services.scene_activation import SceneActivationOutcome

        _write_config(tmp_path)
        adapter = _scene_adapter(display_name="Codex CLI")
        adapter.launch.side_effect = [SceneLaunchFallbackError("profile collision"), 0]
        error = SyncResult(
            tool_id=AIToolID.CODEX,
            concern=SyncConcern.MCP,
            action="error",
            message="codex restriction failed",
        )
        outcome = SceneActivationOutcome(
            scope=(AIToolID.CODEX,),
            results=(error,),
            status="partial",
        )
        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=adapter),
            patch(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                return_value=[AIToolID.CODEX],
            ),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch("crossby.services.scene_activation.activate_scene", return_value=outcome),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "codex", "--scene", "pr-review"]
            )

        assert result.exit_code == 0, result.output
        assert "codex restriction failed" in " ".join(result.output.split())
        assert "activation is partial" in " ".join(result.output.lower().split())
        assert adapter.launch.call_count == 2
        assert adapter.launch.call_args_list[0].kwargs["scene"] is not None
        assert adapter.launch.call_args_list[1].kwargs["scene"] is None

    def test_subdir_launch_resolves_and_applies_against_config_root(self, tmp_path: Path) -> None:
        """Run from a subdirectory: resolve/apply must root at the config's dir.

        The persistent-fallback path (no session-scoped lever) is the one that
        writes state, so it is the clearest way to prove *where* it wrote —
        against the config root, never a shadow tree under the subdirectory,
        while the subprocess itself still runs in the invocation directory.
        """
        from crossby.sync.readers import scan_project as real_scan_project

        _write_config(tmp_path)
        sub = tmp_path / "packages" / "app"
        sub.mkdir(parents=True)
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
            patch("crossby.sync.readers.scan_project", wraps=real_scan_project) as scan_mock,
        ):
            result = runner.invoke(
                app,
                ["launch", str(sub), "--tool", "antigravity-cli", "--scene", "pr-review"],
            )
        assert result.exit_code == 0, result.output
        apply_mock.assert_called_once()
        applied_root = apply_mock.call_args[0][1]
        assert applied_root == tmp_path.resolve()
        scanned_root = scan_mock.call_args[0][0]
        assert scanned_root == tmp_path.resolve()
        # No shadow state/artifact tree written under the subdirectory.
        assert not (sub / ".crossby").exists()
        # The subprocess itself still runs in the invocation directory.
        _, kwargs = adapter.launch.call_args
        assert kwargs["working_dir"] == sub.resolve()


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

    def test_non_mcp_concern_is_warned_not_silently_ignored(self, tmp_path: Path) -> None:
        """A scene narrowing a concern OpenCode can scope neither way is named.

        OpenCode has no launch lever and no persistent mechanism for agents, so
        the restriction can't be honoured — warn instead of silently ignoring it.
        """
        config: dict[str, Any] = {
            "version": 1,
            "ai": {"default_tool": "opencode"},
            "scenes": {"agents-only": {"agents": {"include": ["code-reviewer"]}}},
        }
        _write_config(tmp_path, config)

        def fake_run(cmd: list[str], transcript_path: Any, cwd: Any = None, env: Any = None) -> int:
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
                app, ["launch", str(tmp_path), "--tool", "opencode", "--scene", "agents-only"]
            )
        assert result.exit_code == 0, result.output
        assert "cannot scope agents" in " ".join(result.output.split())


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


def _write_lifecycle_project(root: Path) -> None:
    populate_project(root)
    (root / ".crossby.yml").write_text(
        """\
version: 1
scenes:
  review:
    skills:
      include: ["review-*"]
    mcp:
      include: ["github"]
  deploy:
    skills:
      include: ["deploy-*"]
    mcp:
      include: ["linear"]
""",
        encoding="utf-8",
    )


class TestPersistentFallbackLifecycle:
    """Real-filesystem regressions for the recoverable launch lifecycle."""

    def test_cursor_fallback_status_then_clear_restores_all_skills(self, tmp_path: Path) -> None:
        _write_lifecycle_project(tmp_path)
        spawned: list[list[str]] = []

        def fake_run(cmd: list[str], *_args: Any, **_kwargs: Any) -> int:
            spawned.append(cmd)
            return 0

        with (
            patch(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                return_value=[AIToolID.CURSOR],
            ),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch("crossby.utils.process.run_with_transcript", side_effect=fake_run),
        ):
            launched = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "cursor", "--scene", "review"]
            )
            assert launched.exit_code == 0, launched.output
            assert len(spawned) == 1

            visible = {
                child.name
                for child in (tmp_path / ".cursor" / "skills").iterdir()
                if child.name != ".crossby-managed"
            }
            assert visible == {"review-skill"}

            status = runner.invoke(app, ["scene", "status", "--path", str(tmp_path)])
            assert status.exit_code == 0, status.output
            normalized = " ".join(status.output.split())
            assert "Active scene: review" in normalized
            assert "cursor skills project applied" in normalized

            cleared = runner.invoke(app, ["scene", "clear", "--path", str(tmp_path)])
            assert cleared.exit_code == 0, cleared.output

        restored = {
            child.name
            for child in (tmp_path / ".cursor" / "skills").iterdir()
            if child.name != ".crossby-managed"
        }
        assert restored == {"review-skill", "knowledge", "deploy-prod"}
        assert not (tmp_path / SCENE_STATE_PATH).exists()

    def test_launch_scope_expands_for_shared_skills_directory(self, tmp_path: Path) -> None:
        _write_lifecycle_project(tmp_path)

        with (
            patch(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                return_value=[AIToolID.CODEX, AIToolID.ANTIGRAVITY_CLI],
            ),
            patch("crossby.scenes.versioning.detect_tool_version", return_value=(0, 133, 0)),
            patch("crossby.scenes.trust.codex_trusts_project", return_value=True),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch("crossby.utils.process.run_with_transcript", return_value=0),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "codex", "--scene", "review"]
            )

        assert result.exit_code == 0, result.output
        state = read_json(tmp_path / SCENE_STATE_PATH)
        assert set(state["tools"]) == {"codex", "antigravity-cli"}
        assert "shared skills directory" in " ".join(result.output.split())

    def test_different_scene_scoped_fallback_refuses_to_strand_other_tools(
        self, tmp_path: Path
    ) -> None:
        _write_lifecycle_project(tmp_path)
        spawned: list[list[str]] = []

        with (
            patch(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                return_value=[AIToolID.CLAUDE, AIToolID.CURSOR],
            ),
            patch("crossby.scenes.versioning.detect_tool_version", return_value=(2, 1, 218)),
            patch("crossby.scenes.trust.codex_trusts_project", return_value=True),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch("crossby.ui.prompts.is_tty", return_value=False),
            patch(
                "crossby.utils.process.run_with_transcript",
                side_effect=lambda cmd, *_a, **_kw: spawned.append(cmd) or 0,
            ),
        ):
            first = runner.invoke(app, ["scene", "use", "review", "--path", str(tmp_path)])
            switched = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "cursor", "--scene", "deploy"]
            )

        assert first.exit_code == 0, first.output
        assert switched.exit_code == 1, switched.output
        assert "strand" in switched.output.lower()
        assert spawned == []
        assert read_json(tmp_path / SCENE_STATE_PATH)["scene"] == "review"

    def test_allowed_scoped_switch_replaces_single_tool_scene(self, tmp_path: Path) -> None:
        _write_lifecycle_project(tmp_path)
        spawned: list[list[str]] = []

        with (
            patch(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                return_value=[AIToolID.CURSOR],
            ),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch("crossby.ui.prompts.is_tty", return_value=False),
            patch(
                "crossby.utils.process.run_with_transcript",
                side_effect=lambda cmd, *_a, **_kw: spawned.append(cmd) or 0,
            ),
        ):
            first = runner.invoke(
                app,
                ["scene", "use", "review", "--tool", "cursor", "--path", str(tmp_path)],
            )
            switched = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "cursor", "--scene", "deploy"]
            )

        assert first.exit_code == 0, first.output
        assert switched.exit_code == 0, switched.output
        assert len(spawned) == 1
        state = read_json(tmp_path / SCENE_STATE_PATH)
        assert state["scene"] == "deploy"
        assert set(state["tools"]) == {"cursor"}

    def test_same_scene_scoped_fallback_merges_other_tool_records(self, tmp_path: Path) -> None:
        _write_lifecycle_project(tmp_path)

        with (
            patch(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                return_value=[AIToolID.CLAUDE, AIToolID.CURSOR],
            ),
            patch("crossby.scenes.versioning.detect_tool_version", return_value=(2, 1, 218)),
            patch("crossby.scenes.trust.codex_trusts_project", return_value=True),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch("crossby.ui.prompts.is_tty", return_value=False),
            patch("crossby.utils.process.run_with_transcript", return_value=0),
        ):
            first = runner.invoke(app, ["scene", "use", "review", "--path", str(tmp_path)])
            reapplied = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "cursor", "--scene", "review"]
            )

        assert first.exit_code == 0, first.output
        assert reapplied.exit_code == 0, reapplied.output
        assert set(read_json(tmp_path / SCENE_STATE_PATH)["tools"]) == {"claude", "cursor"}

    def test_fallback_refuses_outgoing_drift_without_starting_child(self, tmp_path: Path) -> None:
        _write_lifecycle_project(tmp_path)
        spawned: list[list[str]] = []

        with (
            patch(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                return_value=[AIToolID.CURSOR],
            ),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch("crossby.ui.prompts.is_tty", return_value=False),
            patch(
                "crossby.utils.process.run_with_transcript",
                side_effect=lambda cmd, *_a, **_kw: spawned.append(cmd) or 0,
            ),
        ):
            first = runner.invoke(
                app,
                ["scene", "use", "review", "--tool", "cursor", "--path", str(tmp_path)],
            )
            assert first.exit_code == 0, first.output
            (tmp_path / ".cursor" / "skills" / "manual-change").mkdir()
            refused = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "cursor", "--scene", "review"]
            )

        assert refused.exit_code == 1, refused.output
        assert "drift" in refused.output.lower()
        assert spawned == []

    @pytest.mark.parametrize(
        "ledger_body",
        [
            "{ not valid json",
            json.dumps(
                {
                    "version": 2,
                    "owned": {},
                    "scene": {"cursor": {"unknown_scene_key": ["x"]}},
                }
            ),
        ],
    )
    def test_corrupt_provenance_aborts_before_mutation_or_spawn(
        self, tmp_path: Path, ledger_body: str
    ) -> None:
        _write_lifecycle_project(tmp_path)
        ledger = tmp_path / ".crossby" / "owned.json"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(ledger_body, encoding="utf-8")
        spawned: list[list[str]] = []

        with (
            patch(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                return_value=[AIToolID.CURSOR],
            ),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch(
                "crossby.utils.process.run_with_transcript",
                side_effect=lambda cmd, *_a, **_kw: spawned.append(cmd) or 0,
            ),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "cursor", "--scene", "review"]
            )

        assert result.exit_code == 1, result.output
        assert "unreadable" in result.output.lower()
        assert ledger.read_text(encoding="utf-8") == ledger_body
        assert not (tmp_path / ".cursor" / "skills").exists()
        assert not (tmp_path / SCENE_STATE_PATH).exists()
        assert spawned == []

    def test_error_rows_record_partial_state_and_still_launch(self, tmp_path: Path) -> None:
        _write_lifecycle_project(tmp_path)
        error = SyncResult(
            tool_id=AIToolID.CURSOR,
            concern=SyncConcern.SKILLS,
            action="error",
            message="cursor projection failed",
        )
        spawned: list[list[str]] = []

        with (
            patch(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                return_value=[AIToolID.CURSOR],
            ),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch("crossby.scenes.engine.apply_scene", return_value=[error]),
            patch(
                "crossby.utils.process.run_with_transcript",
                side_effect=lambda cmd, *_a, **_kw: spawned.append(cmd) or 0,
            ),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "cursor", "--scene", "review"]
            )

        assert result.exit_code == 0, result.output
        assert len(spawned) == 1
        assert "partial" in result.output.lower()
        state = read_json(tmp_path / SCENE_STATE_PATH)
        assert state["status"] == "partial"
        assert state["tools"]["cursor"]["status"] == "failed"

    def test_apply_exception_records_recovery_and_aborts_child(self, tmp_path: Path) -> None:
        _write_lifecycle_project(tmp_path)
        from crossby.scenes import engine

        real_apply = engine.apply_scene
        spawned: list[list[str]] = []

        def apply_then_raise(*args: Any, **kwargs: Any) -> list[SyncResult]:
            real_apply(*args, **kwargs)
            raise RuntimeError("disk failed after provenance")

        with (
            patch(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                return_value=[AIToolID.CURSOR],
            ),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch("crossby.scenes.engine.apply_scene", side_effect=apply_then_raise),
            patch(
                "crossby.utils.process.run_with_transcript",
                side_effect=lambda cmd, *_a, **_kw: spawned.append(cmd) or 0,
            ),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "cursor", "--scene", "review"]
            )

        assert result.exit_code == 1, result.output
        assert spawned == []
        state = read_json(tmp_path / SCENE_STATE_PATH)
        assert state["status"] == "partial"

        with patch(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            return_value=[AIToolID.CURSOR],
        ):
            cleared = runner.invoke(app, ["scene", "clear", "--path", str(tmp_path)])
        assert cleared.exit_code == 0, cleared.output
        assert not (tmp_path / SCENE_STATE_PATH).exists()

    def test_state_write_failure_rolls_back_and_aborts_child(self, tmp_path: Path) -> None:
        _write_lifecycle_project(tmp_path)
        spawned: list[list[str]] = []

        with (
            patch(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                return_value=[AIToolID.CURSOR],
            ),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch(
                "crossby.services.scene_activation.save_scene_state",
                side_effect=OSError("read-only state path"),
            ),
            patch(
                "crossby.utils.process.run_with_transcript",
                side_effect=lambda cmd, *_a, **_kw: spawned.append(cmd) or 0,
            ),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "cursor", "--scene", "review"]
            )

        assert result.exit_code == 1, result.output
        assert spawned == []
        assert "rolled back" in result.output.lower()
        assert not (tmp_path / SCENE_STATE_PATH).exists()
        restored = {
            child.name
            for child in (tmp_path / ".cursor" / "skills").iterdir()
            if child.name != ".crossby-managed"
        }
        assert restored == {"review-skill", "knowledge", "deploy-prod"}

    def test_state_write_failure_after_record_write_removes_stale_state(
        self, tmp_path: Path
    ) -> None:
        from crossby.scenes.state import save_scene_state as real_save_scene_state

        _write_lifecycle_project(tmp_path)
        spawned: list[list[str]] = []

        def save_then_fail(project_root: Path, state: Any) -> None:
            real_save_scene_state(project_root, state)
            raise OSError("gitignore update failed")

        with (
            patch(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                return_value=[AIToolID.CURSOR],
            ),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch("crossby.services.scene_activation.save_scene_state", side_effect=save_then_fail),
            patch(
                "crossby.utils.process.run_with_transcript",
                side_effect=lambda cmd, *_a, **_kw: spawned.append(cmd) or 0,
            ),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "cursor", "--scene", "review"]
            )

        assert result.exit_code == 1, result.output
        assert spawned == []
        assert "rolled back" in result.output.lower()
        assert not (tmp_path / SCENE_STATE_PATH).exists()

    def test_scoped_reapply_state_failure_preserves_unaffected_tool_state(
        self, tmp_path: Path
    ) -> None:
        """Rollback must retain state that still represents untouched tools."""
        from crossby.scenes.state import save_scene_state as real_save_scene_state

        _write_lifecycle_project(tmp_path)
        common_patches = (
            patch(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                return_value=[AIToolID.CLAUDE, AIToolID.CURSOR],
            ),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch("crossby.ui.prompts.is_tty", return_value=False),
        )
        with common_patches[0], common_patches[1], common_patches[2]:
            initial = runner.invoke(app, ["scene", "use", "review", "--path", str(tmp_path)])
        assert initial.exit_code == 0, initial.output

        save_calls = 0
        spawned: list[list[str]] = []

        def save_then_fail_once(project_root: Path, state: Any) -> None:
            nonlocal save_calls
            save_calls += 1
            real_save_scene_state(project_root, state)
            if save_calls == 1:
                raise OSError("gitignore update failed")

        with (
            common_patches[0],
            common_patches[1],
            common_patches[2],
            patch(
                "crossby.services.scene_activation.save_scene_state",
                side_effect=save_then_fail_once,
            ),
            patch(
                "crossby.utils.process.run_with_transcript",
                side_effect=lambda cmd, *_a, **_kw: spawned.append(cmd) or 0,
            ),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "cursor", "--scene", "review"]
            )

        assert result.exit_code == 1, result.output
        assert spawned == []
        assert "rolled back" in result.output.lower()
        state = read_json(tmp_path / SCENE_STATE_PATH)
        assert state["scene"] == "review"
        assert set(state["tools"]) == {"claude"}

        with common_patches[0], common_patches[1], common_patches[2]:
            status = runner.invoke(app, ["scene", "status", "--path", str(tmp_path)])
            cleared = runner.invoke(app, ["scene", "clear", "--path", str(tmp_path)])

        assert status.exit_code == 0, status.output
        assert "Active scene: review" in " ".join(status.output.split())
        assert cleared.exit_code == 0, cleared.output
        assert not (tmp_path / SCENE_STATE_PATH).exists()

    def test_state_write_failure_reports_revocations_need_sync(self, tmp_path: Path) -> None:
        _write_lifecycle_project(tmp_path)
        removed = SyncResult(
            tool_id=AIToolID.CURSOR,
            concern=SyncConcern.HOOKS,
            action="updated",
            revoked=1,
        )
        spawned: list[list[str]] = []

        with (
            patch(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                return_value=[AIToolID.CURSOR],
            ),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch("crossby.scenes.engine.apply_scene", return_value=[removed]),
            patch("crossby.scenes.engine.clear_scene", return_value=[]),
            patch(
                "crossby.services.scene_activation.save_scene_state",
                side_effect=OSError("read-only state path"),
            ),
            patch(
                "crossby.utils.process.run_with_transcript",
                side_effect=lambda cmd, *_a, **_kw: spawned.append(cmd) or 0,
            ),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "cursor", "--scene", "review"]
            )

        normalized = " ".join(result.output.lower().split())
        assert result.exit_code == 1, result.output
        assert spawned == []
        assert "removed hooks remain narrowed" in normalized
        assert "crossby sync" in normalized
        assert "persistent changes were rolled back" not in normalized

    def test_codex_profile_collision_uses_recoverable_fallback_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_lifecycle_project(tmp_path)
        # Make Codex's project config the sole MCP source so discovery has no
        # duplicate-source warning (and the test does not cache a stdlib logger
        # into the later logging-isolation suite).
        (tmp_path / ".mcp.json").unlink()
        codex_home = tmp_path / "codex-home"
        codex_home.mkdir()
        monkeypatch.setenv("CODEX_HOME", str(codex_home))

        project_config = tmp_path / ".codex" / "config.toml"
        project_config.parent.mkdir()
        project_config.write_text(
            '[mcp_servers.github]\ncommand = "gh"\n\n[mcp_servers.linear]\ncommand = "linear"\n',
            encoding="utf-8",
        )
        from crossby.scenes.launch import codex_profile_path

        handwritten = codex_profile_path(tmp_path, "review")
        handwritten.write_text("model = 'gpt-5'\n", encoding="utf-8")
        original_profile = handwritten.read_bytes()
        spawned: list[list[str]] = []

        with (
            patch(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                return_value=[AIToolID.CODEX],
            ),
            patch("crossby.scenes.versioning.detect_tool_version", return_value=(0, 200, 0)),
            patch("crossby.scenes.trust.codex_trusts_project", return_value=True),
            patch("crossby.services.ai_resolution.confirm_ai_selection", side_effect=_passthrough),
            patch(
                "crossby.utils.process.run_with_transcript",
                side_effect=lambda cmd, *_a, **_kw: spawned.append(cmd) or 0,
            ),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "codex", "--scene", "review"]
            )

        assert result.exit_code == 0, result.output
        assert handwritten.read_bytes() == original_profile
        assert len(spawned) == 1
        assert "--profile" not in spawned[0]
        assert "hand-written profile was preserved" in " ".join(result.output.split())
        assert read_json(tmp_path / SCENE_STATE_PATH)["scene"] == "review"
        assert "enabled = false" in project_config.read_text(encoding="utf-8")

        with patch(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            return_value=[AIToolID.CODEX],
        ):
            status = runner.invoke(app, ["scene", "status", "--path", str(tmp_path)])
            cleared = runner.invoke(app, ["scene", "clear", "--path", str(tmp_path)])
        assert status.exit_code == 0, status.output
        assert "Active scene: review" in " ".join(status.output.split())
        assert cleared.exit_code == 0, cleared.output
        assert "enabled = false" not in project_config.read_text(encoding="utf-8")
        assert not (tmp_path / SCENE_STATE_PATH).exists()
