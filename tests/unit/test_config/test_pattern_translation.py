"""Tests for canonical pattern translation and configure_allowlist."""

from __future__ import annotations

import json
from pathlib import Path

from crossby.config.claude_allowlist import (
    canonical_to_claude,
    configure_allowlist,
)
from crossby.config.cursor_allowlist import (
    canonical_to_cursor,
)
from crossby.config.cursor_allowlist import (
    configure_allowlist as configure_cursor_allowlist,
)


class TestCanonicalToClaude:
    """Tests for the canonical_to_claude() helper."""

    def test_crossby_wildcard(self) -> None:
        assert canonical_to_claude("crossby:*") == "Bash(crossby:*)"

    def test_script_with_wildcard(self) -> None:
        assert canonical_to_claude("./scripts/check.sh:*") == "Bash(./scripts/check.sh:*)"

    def test_script_without_args(self) -> None:
        assert canonical_to_claude("./scripts/fmt.sh") == "Bash(./scripts/fmt.sh)"

    def test_command_with_multi_word_args(self) -> None:
        assert (
            canonical_to_claude("crossby:implementation-session done")
            == "Bash(crossby:implementation-session done)"
        )

    def test_bare_command(self) -> None:
        assert canonical_to_claude("git") == "Bash(git)"


class TestConfigureClaudeAllowlistWithPatterns:
    """Tests for configure_allowlist() with patterns parameter."""

    def test_adds_patterns(self, tmp_path: Path) -> None:
        """Patterns are translated and added to the allowlist."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        configure_allowlist(project_root, patterns=["crossby:*", "./scripts/check.sh:*"])

        settings_path = project_root / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        allow = data["permissions"]["allow"]
        assert "Bash(crossby:*)" in allow
        assert "Bash(./scripts/check.sh:*)" in allow

    def test_patterns_idempotent(self, tmp_path: Path) -> None:
        """Running twice with same patterns does not duplicate."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        configure_allowlist(project_root, patterns=["./scripts/check.sh:*"])
        configure_allowlist(project_root, patterns=["./scripts/check.sh:*"])

        settings_path = project_root / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        allow = data["permissions"]["allow"]
        assert allow.count("Bash(./scripts/check.sh:*)") == 1

    def test_multiple_patterns(self, tmp_path: Path) -> None:
        """Multiple patterns all get added."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        configure_allowlist(
            project_root,
            patterns=[
                "./scripts/check.sh:*",
                "./scripts/fmt.sh:*",
                "./scripts/test.sh:*",
            ],
        )

        settings_path = project_root / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        allow = data["permissions"]["allow"]
        assert len(allow) == 3
        assert "Bash(./scripts/check.sh:*)" in allow
        assert "Bash(./scripts/fmt.sh:*)" in allow
        assert "Bash(./scripts/test.sh:*)" in allow


class TestCanonicalToCursor:
    """Tests for the canonical_to_cursor() helper."""

    def test_crossby_wildcard(self) -> None:
        assert canonical_to_cursor("crossby:*") == "Shell(crossby:*)"

    def test_script_with_wildcard(self) -> None:
        assert canonical_to_cursor("./scripts/check.sh:*") == "Shell(./scripts/check.sh:*)"

    def test_script_without_args(self) -> None:
        assert canonical_to_cursor("./scripts/fmt.sh") == "Shell(./scripts/fmt.sh)"

    def test_command_with_multi_word_args(self) -> None:
        assert (
            canonical_to_cursor("crossby:implementation-session done")
            == "Shell(crossby:implementation-session done)"
        )

    def test_bare_command(self) -> None:
        assert canonical_to_cursor("git") == "Shell(git)"


class TestCursorConfigureAllowlistWithPatterns:
    """Tests for cursor configure_allowlist() with patterns parameter."""

    def test_adds_patterns(self, tmp_path: Path) -> None:
        """Patterns are translated and added to the cursor allowlist."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        configure_cursor_allowlist(project_root, patterns=["crossby:*", "./scripts/check.sh:*"])

        config_path = project_root / ".cursor" / "cli.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))
        allow = data["permissions"]["allow"]
        assert "Shell(crossby:*)" in allow
        assert "Shell(./scripts/check.sh:*)" in allow

    def test_patterns_idempotent(self, tmp_path: Path) -> None:
        """Running twice with same patterns does not duplicate."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        configure_cursor_allowlist(project_root, patterns=["./scripts/check.sh:*"])
        configure_cursor_allowlist(project_root, patterns=["./scripts/check.sh:*"])

        config_path = project_root / ".cursor" / "cli.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))
        allow = data["permissions"]["allow"]
        assert allow.count("Shell(./scripts/check.sh:*)") == 1

    def test_multiple_patterns(self, tmp_path: Path) -> None:
        """Multiple patterns all get added."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        configure_cursor_allowlist(
            project_root,
            patterns=[
                "./scripts/check.sh:*",
                "./scripts/fmt.sh:*",
                "./scripts/test.sh:*",
            ],
        )

        config_path = project_root / ".cursor" / "cli.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))
        allow = data["permissions"]["allow"]
        assert len(allow) == 3
        assert "Shell(./scripts/check.sh:*)" in allow
        assert "Shell(./scripts/fmt.sh:*)" in allow
        assert "Shell(./scripts/test.sh:*)" in allow
