"""Tests for allowlist reading (read_allowlist)."""

from __future__ import annotations

import json
from pathlib import Path

from crossby.config.claude_allowlist import read_allowlist as claude_read
from crossby.config.cursor_allowlist import read_allowlist as cursor_read


class TestClaudeReadAllowlist:
    def test_reads_bash_patterns(self, tmp_path: Path) -> None:
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings = {"permissions": {"allow": ["Bash(myapp:*)", "Bash(./scripts/check.sh:*)"]}}
        (claude_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")

        result = claude_read(tmp_path)
        assert result == ["myapp:*", "./scripts/check.sh:*"]

    def test_filters_non_bash_patterns(self, tmp_path: Path) -> None:
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings = {
            "permissions": {"allow": ["Bash(myapp:*)", "Read(**)", "Edit(***)", "Bash(npm:*)"]}
        }
        (claude_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")

        result = claude_read(tmp_path)
        assert result == ["myapp:*", "npm:*"]

    def test_returns_empty_when_missing(self, tmp_path: Path) -> None:
        assert claude_read(tmp_path) == []

    def test_returns_empty_for_corrupted_json(self, tmp_path: Path) -> None:
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text("{bad json!!", encoding="utf-8")
        assert claude_read(tmp_path) == []

    def test_returns_empty_for_empty_allowlist(self, tmp_path: Path) -> None:
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings = {"permissions": {"allow": []}}
        (claude_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
        assert claude_read(tmp_path) == []

    def test_returns_empty_for_non_dict_root(self, tmp_path: Path) -> None:
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text("[1,2,3]", encoding="utf-8")
        assert claude_read(tmp_path) == []


class TestCursorReadAllowlist:
    def test_reads_shell_patterns(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir()
        settings = {"permissions": {"allow": ["Shell(myapp:*)", "Shell(npm:*)"]}}
        (cursor_dir / "cli.json").write_text(json.dumps(settings), encoding="utf-8")

        result = cursor_read(tmp_path)
        assert result == ["myapp:*", "npm:*"]

    def test_filters_non_shell_patterns(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir()
        settings = {"permissions": {"allow": ["Shell(myapp:*)", "Read(**)", "Shell(npm:*)"]}}
        (cursor_dir / "cli.json").write_text(json.dumps(settings), encoding="utf-8")

        result = cursor_read(tmp_path)
        assert result == ["myapp:*", "npm:*"]

    def test_returns_empty_when_missing(self, tmp_path: Path) -> None:
        assert cursor_read(tmp_path) == []

    def test_returns_empty_for_corrupted_json(self, tmp_path: Path) -> None:
        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir()
        (cursor_dir / "cli.json").write_text("{bad json!!", encoding="utf-8")
        assert cursor_read(tmp_path) == []
