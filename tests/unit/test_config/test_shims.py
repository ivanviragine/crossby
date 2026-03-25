"""Tests that backward-compat shims produce the same behavior as before.

These tests verify that ``config/claude_allowlist.py`` and
``config/cursor_allowlist.py`` still work identically to their original
implementations after being refactored as shims that delegate to the sync
writer classes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossby.config.claude_allowlist import (
    canonical_to_claude,
    configure_allowlist as claude_configure,
    is_allowlist_configured as claude_is_configured,
)
from crossby.config.cursor_allowlist import (
    canonical_to_cursor,
    configure_allowlist as cursor_configure,
    is_allowlist_configured as cursor_is_configured,
)


@pytest.fixture(autouse=True)
def _patch_cursor_global(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the Cursor global config path for cursor shim tests."""
    fake = tmp_path / ".cursor" / "cli-config.json"
    monkeypatch.setattr("crossby.sync.permissions._GLOBAL_CURSOR_CONFIG_PATH", fake)


def _cursor_global(tmp_path: Path) -> Path:
    return tmp_path / ".cursor" / "cli-config.json"


# ---------------------------------------------------------------------------
# Claude shim
# ---------------------------------------------------------------------------


class TestClaudeShim:
    def test_canonical_to_claude(self) -> None:
        assert canonical_to_claude("myapp:*") == "Bash(myapp:*)"

    def test_configure_creates_file(self, tmp_path: Path) -> None:
        claude_configure(tmp_path, ["myapp:*"])
        data = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        assert "Bash(myapp:*)" in data["permissions"]["allow"]

    def test_configure_idempotent(self, tmp_path: Path) -> None:
        claude_configure(tmp_path, ["myapp:*"])
        claude_configure(tmp_path, ["myapp:*"])
        data = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        assert data["permissions"]["allow"].count("Bash(myapp:*)") == 1

    def test_configure_non_destructive(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"theme": "dark", "permissions": {"allow": ["Bash(git *)"]}}))
        claude_configure(tmp_path, ["myapp:*"])
        data = json.loads(path.read_text())
        assert "Bash(myapp:*)" in data["permissions"]["allow"]
        assert data["theme"] == "dark"

    def test_is_configured_true(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"permissions": {"allow": ["Bash(myapp:*)"]}}))
        assert claude_is_configured(tmp_path, ["myapp:*"]) is True

    def test_is_configured_false_missing(self, tmp_path: Path) -> None:
        assert claude_is_configured(tmp_path, ["myapp:*"]) is False

    def test_is_configured_false_absent(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"permissions": {"allow": ["Bash(other:*)"]}}))
        assert claude_is_configured(tmp_path, ["myapp:*"]) is False


# ---------------------------------------------------------------------------
# Cursor shim
# ---------------------------------------------------------------------------


class TestCursorShim:
    def test_canonical_to_cursor(self) -> None:
        assert canonical_to_cursor("myapp:*") == "Shell(myapp:*)"

    def test_configure_per_project(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        cursor_configure(project, ["myapp:*"])
        data = json.loads((project / ".cursor" / "cli.json").read_text())
        assert "Shell(myapp:*)" in data["permissions"]["allow"]

    def test_configure_global(self, tmp_path: Path) -> None:
        cursor_configure(None, ["myapp:*"])
        data = json.loads(_cursor_global(tmp_path).read_text())
        assert "Shell(myapp:*)" in data["permissions"]["allow"]

    def test_configure_noop_no_patterns(self, tmp_path: Path) -> None:
        cursor_configure()
        assert not _cursor_global(tmp_path).exists()

    def test_configure_idempotent(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        cursor_configure(project, ["myapp:*"])
        cursor_configure(project, ["myapp:*"])
        data = json.loads((project / ".cursor" / "cli.json").read_text())
        assert data["permissions"]["allow"].count("Shell(myapp:*)") == 1

    def test_per_project_does_not_touch_global(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        cursor_configure(project, ["myapp:*"])
        assert not _cursor_global(tmp_path).exists()

    def test_is_configured_true_global(self, tmp_path: Path) -> None:
        path = _cursor_global(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"permissions": {"allow": ["Shell(myapp:*)"]}}))
        assert cursor_is_configured(None, ["myapp:*"]) is True

    def test_is_configured_false_missing(self) -> None:
        assert cursor_is_configured(None, ["myapp:*"]) is False

    def test_is_configured_per_project(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        path = project / ".cursor" / "cli.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"permissions": {"allow": ["Shell(myapp:*)"]}}))
        assert cursor_is_configured(project, ["myapp:*"]) is True

    def test_is_configured_vacuously_true_no_patterns(self) -> None:
        assert cursor_is_configured() is True
