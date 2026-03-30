"""Tests for Claude Code .claude/settings.json allowlist management."""

from __future__ import annotations

import json
from pathlib import Path

from crossby.config.claude_allowlist import (
    canonical_to_claude,
    configure_allowlist,
    is_allowlist_configured,
)


class TestCanonicalToClaude:
    """Tests for canonical_to_claude()."""

    def test_simple_pattern(self) -> None:
        assert canonical_to_claude("myapp:*") == "Bash(myapp:*)"

    def test_script_pattern(self) -> None:
        assert canonical_to_claude("./scripts/check.sh:*") == "Bash(./scripts/check.sh:*)"

    def test_bare_command(self) -> None:
        assert canonical_to_claude("./scripts/check.sh") == "Bash(./scripts/check.sh)"


class TestConfigureAllowlist:
    """Tests for configure_allowlist()."""

    def test_creates_settings_from_scratch(self, tmp_path: Path) -> None:
        """Creates .claude/settings.json when neither dir nor file exist."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        settings_path = project_root / ".claude" / "settings.json"

        configure_allowlist(project_root, patterns=["myapp:*"])

        assert settings_path.is_file()
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert data == {
            "permissions": {
                "allow": ["Bash(myapp:*)"],
            },
        }

    def test_adds_to_existing_settings(self, tmp_path: Path) -> None:
        """Adds pattern to existing settings.json that has other permissions."""
        project_root = tmp_path / "project"
        claude_dir = project_root / ".claude"
        claude_dir.mkdir(parents=True)
        settings_path = claude_dir / "settings.json"
        existing = {
            "permissions": {
                "allow": ["Bash(git *)"],
                "deny": ["Bash(rm -rf /)"],
            },
            "theme": "dark",
        }
        settings_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")

        configure_allowlist(project_root, patterns=["myapp:*"])

        data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert "Bash(myapp:*)" in data["permissions"]["allow"]
        assert "Bash(git *)" in data["permissions"]["allow"]
        assert data["permissions"]["deny"] == ["Bash(rm -rf /)"]
        assert data["theme"] == "dark"

    def test_idempotent_no_duplicate(self, tmp_path: Path) -> None:
        """Running twice does not duplicate the allow pattern."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        configure_allowlist(project_root, patterns=["myapp:*"])
        configure_allowlist(project_root, patterns=["myapp:*"])

        settings_path = project_root / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        allow_list = data["permissions"]["allow"]
        count = allow_list.count("Bash(myapp:*)")
        assert count == 1, f"Expected exactly 1 entry, got {count}"

    def test_handles_corrupted_json(self, tmp_path: Path) -> None:
        """Handles corrupted/invalid JSON gracefully by starting fresh."""
        project_root = tmp_path / "project"
        claude_dir = project_root / ".claude"
        claude_dir.mkdir(parents=True)
        settings_path = claude_dir / "settings.json"
        settings_path.write_text("{invalid json!!", encoding="utf-8")

        configure_allowlist(project_root, patterns=["myapp:*"])

        data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert "Bash(myapp:*)" in data["permissions"]["allow"]

    def test_preserves_other_settings_keys(self, tmp_path: Path) -> None:
        """Preserves all non-permissions keys in existing settings."""
        project_root = tmp_path / "project"
        claude_dir = project_root / ".claude"
        claude_dir.mkdir(parents=True)
        settings_path = claude_dir / "settings.json"
        existing = {
            "mcpServers": {
                "memory": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-memory"],
                }
            },
            "customInstructions": "Be concise.",
            "permissions": {
                "allow": ["Read(**)"],
            },
        }
        settings_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")

        configure_allowlist(project_root, patterns=["myapp:*"])

        data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert data["mcpServers"] == existing["mcpServers"]
        assert data["customInstructions"] == "Be concise."
        assert "Read(**)" in data["permissions"]["allow"]
        assert "Bash(myapp:*)" in data["permissions"]["allow"]

    def test_handles_non_dict_permissions(self, tmp_path: Path) -> None:
        """Handles permissions being a non-dict value (e.g. a string)."""
        project_root = tmp_path / "project"
        claude_dir = project_root / ".claude"
        claude_dir.mkdir(parents=True)
        settings_path = claude_dir / "settings.json"
        existing = {"permissions": "invalid", "other": "kept"}
        settings_path.write_text(json.dumps(existing) + "\n", encoding="utf-8")

        configure_allowlist(project_root, patterns=["myapp:*"])

        data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert "Bash(myapp:*)" in data["permissions"]["allow"]
        assert data["other"] == "kept"

    def test_handles_non_list_allow(self, tmp_path: Path) -> None:
        """Handles allow being a non-list value (e.g. a string)."""
        project_root = tmp_path / "project"
        claude_dir = project_root / ".claude"
        claude_dir.mkdir(parents=True)
        settings_path = claude_dir / "settings.json"
        existing = {"permissions": {"allow": "not-a-list"}}
        settings_path.write_text(json.dumps(existing) + "\n", encoding="utf-8")

        configure_allowlist(project_root, patterns=["myapp:*"])

        data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert isinstance(data["permissions"]["allow"], list)
        assert "Bash(myapp:*)" in data["permissions"]["allow"]

    def test_handles_non_dict_root(self, tmp_path: Path) -> None:
        """Handles settings.json containing a non-dict root (e.g. a list)."""
        project_root = tmp_path / "project"
        claude_dir = project_root / ".claude"
        claude_dir.mkdir(parents=True)
        settings_path = claude_dir / "settings.json"
        settings_path.write_text("[1, 2, 3]\n", encoding="utf-8")

        configure_allowlist(project_root, patterns=["myapp:*"])

        data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert "Bash(myapp:*)" in data["permissions"]["allow"]

    def test_multiple_patterns(self, tmp_path: Path) -> None:
        """Multiple patterns are all added."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        configure_allowlist(project_root, patterns=["myapp:*", "./scripts/check.sh:*"])

        settings_path = project_root / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert "Bash(myapp:*)" in data["permissions"]["allow"]
        assert "Bash(./scripts/check.sh:*)" in data["permissions"]["allow"]


class TestIsAllowlistConfigured:
    """Tests for is_allowlist_configured()."""

    def test_returns_true_when_pattern_present(self, tmp_path: Path) -> None:
        """Returns True when all requested patterns are in the allowlist."""
        project_root = tmp_path / "project"
        claude_dir = project_root / ".claude"
        claude_dir.mkdir(parents=True)
        settings_path = claude_dir / "settings.json"
        settings_path.write_text(
            '{"permissions": {"allow": ["Bash(myapp:*)", "Read(**)"]}}\n',
            encoding="utf-8",
        )

        assert is_allowlist_configured(project_root, patterns=["myapp:*"]) is True

    def test_returns_false_when_file_missing(self, tmp_path: Path) -> None:
        """Returns False when settings.json does not exist."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        assert is_allowlist_configured(project_root, patterns=["myapp:*"]) is False

    def test_returns_false_when_pattern_absent(self, tmp_path: Path) -> None:
        """Returns False when settings.json exists but pattern is not in allowlist."""
        project_root = tmp_path / "project"
        claude_dir = project_root / ".claude"
        claude_dir.mkdir(parents=True)
        settings_path = claude_dir / "settings.json"
        settings_path.write_text(
            '{"permissions": {"allow": ["Bash(git *)"]}}\n',
            encoding="utf-8",
        )

        assert is_allowlist_configured(project_root, patterns=["myapp:*"]) is False

    def test_returns_false_for_corrupted_json(self, tmp_path: Path) -> None:
        """Returns False when settings.json contains invalid JSON."""
        project_root = tmp_path / "project"
        claude_dir = project_root / ".claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / "settings.json").write_text("{invalid!!", encoding="utf-8")

        assert is_allowlist_configured(project_root, patterns=["myapp:*"]) is False

    def test_returns_false_when_partial_match(self, tmp_path: Path) -> None:
        """Returns False when only some of the requested patterns are present."""
        project_root = tmp_path / "project"
        claude_dir = project_root / ".claude"
        claude_dir.mkdir(parents=True)
        settings_path = claude_dir / "settings.json"
        settings_path.write_text(
            '{"permissions": {"allow": ["Bash(myapp:*)"]}}\n',
            encoding="utf-8",
        )

        assert is_allowlist_configured(project_root, patterns=["myapp:*", "other:*"]) is False


