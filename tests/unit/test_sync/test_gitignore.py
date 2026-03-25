"""Tests for rules .gitignore managed-block logic."""

from pathlib import Path

from crossby.models.config import CrossbyConfig, RulesConfig
from crossby.sync.rules import _BLOCK_END, _BLOCK_START, update_rules_gitignore


def _cfg() -> CrossbyConfig:
    return CrossbyConfig(rules=RulesConfig(enabled=True, gitignore=True))


class TestUpdateRulesGitignore:
    def test_add_block_to_empty_file(self, tmp_path: Path):
        result = update_rules_gitignore(_cfg(), tmp_path, synced_targets=["CLAUDE.md", ".cursorrules"])
        assert result is not None

        content = (tmp_path / ".gitignore").read_text()
        assert _BLOCK_START in content
        assert _BLOCK_END in content
        assert ".cursorrules" in content
        assert "CLAUDE.md" in content

    def test_add_block_to_existing_gitignore(self, tmp_path: Path):
        (tmp_path / ".gitignore").write_text("node_modules/\n.env\n")
        update_rules_gitignore(_cfg(), tmp_path, synced_targets=["CLAUDE.md"])

        content = (tmp_path / ".gitignore").read_text()
        assert content.startswith("node_modules/")
        assert "CLAUDE.md" in content

    def test_update_existing_block(self, tmp_path: Path):
        update_rules_gitignore(_cfg(), tmp_path, synced_targets=["CLAUDE.md"])
        update_rules_gitignore(_cfg(), tmp_path, synced_targets=["CLAUDE.md", "GEMINI.md"])

        content = (tmp_path / ".gitignore").read_text()
        assert "GEMINI.md" in content
        # Only one block
        assert content.count(_BLOCK_START) == 1

    def test_no_modification_when_up_to_date(self, tmp_path: Path):
        update_rules_gitignore(_cfg(), tmp_path, synced_targets=["CLAUDE.md"])
        result = update_rules_gitignore(_cfg(), tmp_path, synced_targets=["CLAUDE.md"])
        assert result is None

    def test_preserves_user_entries(self, tmp_path: Path):
        (tmp_path / ".gitignore").write_text("# my custom ignore\n*.pyc\n")
        update_rules_gitignore(_cfg(), tmp_path, synced_targets=["CLAUDE.md"])

        content = (tmp_path / ".gitignore").read_text()
        assert "# my custom ignore" in content
        assert "*.pyc" in content

    def test_entries_are_sorted(self, tmp_path: Path):
        update_rules_gitignore(
            _cfg(), tmp_path,
            synced_targets=["GEMINI.md", ".cursorrules", "CLAUDE.md"],
        )
        content = (tmp_path / ".gitignore").read_text()
        lines = content.splitlines()
        start = lines.index(_BLOCK_START)
        end = lines.index(_BLOCK_END)
        entries = lines[start + 1 : end]
        assert entries == sorted(entries)

    def test_creates_gitignore_if_missing(self, tmp_path: Path):
        assert not (tmp_path / ".gitignore").exists()
        update_rules_gitignore(_cfg(), tmp_path, synced_targets=["CLAUDE.md"])
        assert (tmp_path / ".gitignore").exists()

    def test_orphan_start_marker_does_not_duplicate(self, tmp_path: Path):
        """Orphan _BLOCK_START without _BLOCK_END should not create a second block."""
        (tmp_path / ".gitignore").write_text(
            f"node_modules/\n{_BLOCK_START}\nCLAUDE.md\n"
        )
        update_rules_gitignore(_cfg(), tmp_path, synced_targets=["GEMINI.md"])

        content = (tmp_path / ".gitignore").read_text()
        assert content.count(_BLOCK_START) == 1
        assert "GEMINI.md" in content

    def test_returns_none_when_disabled(self, tmp_path: Path):
        config = CrossbyConfig(rules=RulesConfig(enabled=True, gitignore=False))
        result = update_rules_gitignore(config, tmp_path, synced_targets=["CLAUDE.md"])
        assert result is None

    def test_returns_none_when_no_targets(self, tmp_path: Path):
        result = update_rules_gitignore(_cfg(), tmp_path, synced_targets=[])
        assert result is None
