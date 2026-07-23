"""Tests for configure_worktree_hooks in all four config modules."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from crossby.config.claude_allowlist import (
    configure_worktree_hooks as claude_configure_worktree_hooks,
)
from crossby.config.copilot_hooks import (
    configure_worktree_hooks as copilot_configure_worktree_hooks,
)
from crossby.config.cursor_hooks import configure_worktree_hooks as cursor_configure_worktree_hooks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _guard(tmp_path: Path) -> Path:
    return tmp_path / "guard.sh"


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------


class TestClaudeConfigureWorktreeHooks:
    """configure_worktree_hooks writes to .claude/settings.json → hooks.PreToolUse."""

    def test_fresh_install(self, tmp_path: Path) -> None:
        guard = _guard(tmp_path)
        claude_configure_worktree_hooks(tmp_path, guard)

        settings = tmp_path / ".claude" / "settings.json"
        assert settings.is_file()
        data = json.loads(settings.read_text(encoding="utf-8"))
        pre_tool = data["hooks"]["PreToolUse"]
        assert isinstance(pre_tool, list)
        assert len(pre_tool) == 1
        entry = pre_tool[0]
        # Matcher covers Edit, Write, and NotebookEdit — .ipynb writes go
        # through NotebookEdit and must not bypass the worktree-isolation guard.
        assert entry["matcher"] == "Edit|Write|NotebookEdit"
        assert entry["hooks"] == [{"type": "command", "command": str(guard)}]

    def test_idempotent(self, tmp_path: Path) -> None:
        guard = _guard(tmp_path)
        claude_configure_worktree_hooks(tmp_path, guard)
        claude_configure_worktree_hooks(tmp_path, guard)

        data = json.loads((tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8"))
        commands = [
            inner["command"]
            for entry in data["hooks"]["PreToolUse"]
            for inner in entry.get("hooks", [])
            if isinstance(inner, dict)
        ]
        assert commands.count(str(guard)) == 1

    def test_coexists_with_existing_hooks(self, tmp_path: Path) -> None:
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        existing = {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Bash",
                        "hooks": [{"type": "command", "command": "/usr/local/bin/existing"}],
                    }
                ]
            },
            "theme": "dark",
        }
        settings_path.write_text(json.dumps(existing), encoding="utf-8")

        guard = _guard(tmp_path)
        claude_configure_worktree_hooks(tmp_path, guard)

        data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert data["theme"] == "dark"
        pre_tool = data["hooks"]["PreToolUse"]
        commands = [
            inner["command"]
            for entry in pre_tool
            for inner in entry.get("hooks", [])
            if isinstance(inner, dict)
        ]
        assert "/usr/local/bin/existing" in commands
        assert str(guard) in commands


# ---------------------------------------------------------------------------
# Cursor
# ---------------------------------------------------------------------------


class TestCursorConfigureWorktreeHooks:
    """configure_worktree_hooks writes to .cursor/hooks.json → preToolUse[]."""

    def test_fresh_install(self, tmp_path: Path) -> None:
        guard = _guard(tmp_path)
        cursor_configure_worktree_hooks(tmp_path, guard)

        hooks_file = tmp_path / ".cursor" / "hooks.json"
        assert hooks_file.is_file()
        data = json.loads(hooks_file.read_text(encoding="utf-8"))
        pre_tool = data["preToolUse"]
        assert isinstance(pre_tool, list)
        assert len(pre_tool) == 1
        entry = pre_tool[0]
        assert entry["command"] == str(guard)
        assert entry["event"] == "preToolUse"
        # Tools cover Edit, Write, and Delete — worktree isolation must
        # also block deletions via Cursor's Delete tool.
        assert "Edit" in entry["tools"]
        assert "Write" in entry["tools"]
        assert "Delete" in entry["tools"]

    def test_idempotent(self, tmp_path: Path) -> None:
        guard = _guard(tmp_path)
        cursor_configure_worktree_hooks(tmp_path, guard)
        cursor_configure_worktree_hooks(tmp_path, guard)

        data = json.loads((tmp_path / ".cursor" / "hooks.json").read_text(encoding="utf-8"))
        commands = [e["command"] for e in data["preToolUse"] if isinstance(e, dict)]
        assert commands.count(str(guard)) == 1

    def test_coexists_with_existing_hooks(self, tmp_path: Path) -> None:
        hooks_path = tmp_path / ".cursor" / "hooks.json"
        hooks_path.parent.mkdir(parents=True)
        existing = {
            "preToolUse": [
                {"event": "preToolUse", "command": "/usr/local/bin/existing", "tools": ["Bash"]},
            ]
        }
        hooks_path.write_text(json.dumps(existing), encoding="utf-8")

        guard = _guard(tmp_path)
        cursor_configure_worktree_hooks(tmp_path, guard)

        data = json.loads(hooks_path.read_text(encoding="utf-8"))
        commands = [e["command"] for e in data["preToolUse"] if isinstance(e, dict)]
        assert "/usr/local/bin/existing" in commands
        assert str(guard) in commands


# ---------------------------------------------------------------------------
# Copilot
# ---------------------------------------------------------------------------


class TestCopilotConfigureWorktreeHooks:
    """configure_worktree_hooks writes to .github/hooks/hooks.json → hooks.preToolUse[]."""

    def test_fresh_install(self, tmp_path: Path) -> None:
        guard = _guard(tmp_path)
        copilot_configure_worktree_hooks(tmp_path, guard)

        hooks_file = tmp_path / ".github" / "hooks" / "hooks.json"
        assert hooks_file.is_file()
        data = json.loads(hooks_file.read_text(encoding="utf-8"))
        assert data["version"] == 1
        pre_tool = data["hooks"]["preToolUse"]
        assert isinstance(pre_tool, list)
        assert len(pre_tool) == 1
        entry = pre_tool[0]
        assert entry["bash"] == str(guard)
        assert entry["type"] == "command"

    def test_idempotent(self, tmp_path: Path) -> None:
        guard = _guard(tmp_path)
        copilot_configure_worktree_hooks(tmp_path, guard)
        copilot_configure_worktree_hooks(tmp_path, guard)

        data = json.loads(
            (tmp_path / ".github" / "hooks" / "hooks.json").read_text(encoding="utf-8")
        )
        bashes = [e["bash"] for e in data["hooks"]["preToolUse"] if isinstance(e, dict)]
        assert bashes.count(str(guard)) == 1

    def test_coexists_with_existing_hooks(self, tmp_path: Path) -> None:
        hooks_path = tmp_path / ".github" / "hooks" / "hooks.json"
        hooks_path.parent.mkdir(parents=True)
        existing = {
            "version": 1,
            "hooks": {
                "preToolUse": [
                    {"type": "command", "bash": "/usr/local/bin/existing", "comment": ""},
                ]
            },
        }
        hooks_path.write_text(json.dumps(existing), encoding="utf-8")

        guard = _guard(tmp_path)
        copilot_configure_worktree_hooks(tmp_path, guard)

        data = json.loads(hooks_path.read_text(encoding="utf-8"))
        bashes = [e["bash"] for e in data["hooks"]["preToolUse"] if isinstance(e, dict)]
        assert "/usr/local/bin/existing" in bashes
        assert str(guard) in bashes

    def test_no_tool_filter_in_output(self, tmp_path: Path) -> None:
        """Copilot has no per-tool filter — guard fires on all tool calls."""
        guard = _guard(tmp_path)
        copilot_configure_worktree_hooks(tmp_path, guard)

        data = json.loads(
            (tmp_path / ".github" / "hooks" / "hooks.json").read_text(encoding="utf-8")
        )
        entry = data["hooks"]["preToolUse"][0]
        assert "tools" not in entry


# ---------------------------------------------------------------------------
# Error path: malformed JSON emits warnings.warn, does not raise
# ---------------------------------------------------------------------------


class TestMalformedJsonWarns:
    """Malformed config files surface as warnings, not exceptions."""

    def _write_bad_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{bad json!!", encoding="utf-8")

    def test_claude_warns_on_bad_json(self, tmp_path: Path) -> None:
        self._write_bad_json(tmp_path / ".claude" / "settings.json")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            claude_configure_worktree_hooks(tmp_path, _guard(tmp_path))
        assert any("invalid JSON" in str(warning.message) for warning in w)

    def test_cursor_warns_on_bad_json(self, tmp_path: Path) -> None:
        self._write_bad_json(tmp_path / ".cursor" / "hooks.json")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cursor_configure_worktree_hooks(tmp_path, _guard(tmp_path))
        assert any("invalid JSON" in str(warning.message) for warning in w)

    def test_copilot_warns_on_bad_json(self, tmp_path: Path) -> None:
        self._write_bad_json(tmp_path / ".github" / "hooks" / "hooks.json")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            copilot_configure_worktree_hooks(tmp_path, _guard(tmp_path))
        assert any("invalid JSON" in str(warning.message) for warning in w)
