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
    """configure_worktree_hooks writes to .cursor/hooks.json → hooks.preToolUse[]."""

    def test_fresh_install(self, tmp_path: Path) -> None:
        guard = _guard(tmp_path)
        cursor_configure_worktree_hooks(tmp_path, guard)

        hooks_file = tmp_path / ".cursor" / "hooks.json"
        assert hooks_file.is_file()
        data = json.loads(hooks_file.read_text(encoding="utf-8"))
        # The {version, hooks} wrapper is mandatory — Cursor rejects a config
        # without a top-level `hooks` object and loads nothing at all.
        assert data["version"] == 1
        pre_tool = data["hooks"]["preToolUse"]
        assert isinstance(pre_tool, list)
        assert len(pre_tool) == 1
        entry = pre_tool[0]
        assert entry["command"] == str(guard)
        assert entry["type"] == "command"
        # Scope is a single `matcher` regex; Cursor's schema has no `tools`.
        assert "tools" not in entry
        # Cursor has no Edit tool — Edit collapses into Write. Delete stays, so
        # worktree isolation still blocks deletions.
        assert entry["matcher"] == "Write|Delete"

    def test_idempotent(self, tmp_path: Path) -> None:
        guard = _guard(tmp_path)
        cursor_configure_worktree_hooks(tmp_path, guard)
        cursor_configure_worktree_hooks(tmp_path, guard)

        data = json.loads((tmp_path / ".cursor" / "hooks.json").read_text(encoding="utf-8"))
        commands = [e["command"] for e in data["hooks"]["preToolUse"] if isinstance(e, dict)]
        assert commands.count(str(guard)) == 1

    def test_coexists_with_existing_hooks(self, tmp_path: Path) -> None:
        hooks_path = tmp_path / ".cursor" / "hooks.json"
        hooks_path.parent.mkdir(parents=True)
        existing = {
            "version": 1,
            "hooks": {
                "preToolUse": [
                    {"type": "command", "command": "/usr/local/bin/existing", "matcher": "Shell"},
                ]
            },
        }
        hooks_path.write_text(json.dumps(existing), encoding="utf-8")

        guard = _guard(tmp_path)
        cursor_configure_worktree_hooks(tmp_path, guard)

        data = json.loads(hooks_path.read_text(encoding="utf-8"))
        commands = [e["command"] for e in data["hooks"]["preToolUse"] if isinstance(e, dict)]
        assert "/usr/local/bin/existing" in commands
        assert str(guard) in commands

    def test_migrates_legacy_flat_config(self, tmp_path: Path) -> None:
        """A pre-0.13 flat file is lifted into the wrapper instead of left broken.

        crossby <=0.12 wrote event arrays at the top level, which Cursor rejects
        outright — so every hook it wrote was inert. Re-syncing must repair the
        file, not append a second copy alongside the dead one.
        """
        hooks_path = tmp_path / ".cursor" / "hooks.json"
        hooks_path.parent.mkdir(parents=True)
        guard = _guard(tmp_path)
        legacy = {
            "preToolUse": [
                {
                    "event": "preToolUse",
                    "command": str(guard),
                    "tools": ["Edit", "Write", "Delete"],
                }
            ]
        }
        hooks_path.write_text(json.dumps(legacy), encoding="utf-8")

        cursor_configure_worktree_hooks(tmp_path, guard)

        data = json.loads(hooks_path.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert "preToolUse" not in data, "legacy top-level key must be lifted, not left behind"
        entries = data["hooks"]["preToolUse"]
        assert len(entries) == 1, "migration must not duplicate the guard"
        assert entries[0]["command"] == str(guard)
        assert entries[0]["matcher"] == "Write|Delete"
        assert "tools" not in entries[0]
        assert "event" not in entries[0]


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
        """crossby writes Copilot hooks unscoped — guard fires on all tool calls."""
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


class TestWorktreeGuardRevokesNothing:
    """The single-hook guard install must never remove another hook in the file."""

    def test_claude_preserves_existing_hooks(self, tmp_path: Path) -> None:
        settings = tmp_path / ".claude" / "settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [
                            {"matcher": "Bash", "hooks": [{"type": "command", "command": "other"}]}
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        claude_configure_worktree_hooks(tmp_path, _guard(tmp_path))
        commands = {
            inner["command"]
            for entry in json.loads(settings.read_text())["hooks"]["PreToolUse"]
            for inner in entry["hooks"]
        }
        assert "other" in commands
        assert str(_guard(tmp_path)) in commands

    def test_cursor_preserves_existing_hooks(self, tmp_path: Path) -> None:
        hooks = tmp_path / ".cursor" / "hooks.json"
        hooks.parent.mkdir(parents=True)
        hooks.write_text(
            json.dumps({"version": 1, "hooks": {"stop": [{"type": "command", "command": "keep"}]}}),
            encoding="utf-8",
        )
        cursor_configure_worktree_hooks(tmp_path, _guard(tmp_path))
        data = json.loads(hooks.read_text())
        assert data["hooks"]["stop"][0]["command"] == "keep"
        assert data["hooks"]["preToolUse"]

    def test_copilot_preserves_existing_hooks(self, tmp_path: Path) -> None:
        hooks = tmp_path / ".github" / "hooks" / "hooks.json"
        hooks.parent.mkdir(parents=True)
        hooks.write_text(
            json.dumps(
                {"version": 1, "hooks": {"agentStop": [{"type": "command", "bash": "keep"}]}}
            ),
            encoding="utf-8",
        )
        copilot_configure_worktree_hooks(tmp_path, _guard(tmp_path))
        data = json.loads(hooks.read_text())
        assert data["hooks"]["agentStop"][0]["bash"] == "keep"
        assert data["hooks"]["preToolUse"]
