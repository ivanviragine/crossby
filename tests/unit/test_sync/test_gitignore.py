"""Tests for .gitignore managed-block logic."""

from pathlib import Path

from crossby.sync.gitignore import _BLOCK_END, _BLOCK_START, update_gitignore_block


class TestUpdateGitignoreBlock:
    def test_add_block_to_empty_file(self, tmp_path: Path):
        modified = update_gitignore_block(tmp_path, ["CLAUDE.md", ".cursorrules"])
        assert modified is True

        content = (tmp_path / ".gitignore").read_text()
        assert _BLOCK_START in content
        assert _BLOCK_END in content
        assert ".cursorrules" in content
        assert "CLAUDE.md" in content

    def test_add_block_to_existing_gitignore(self, tmp_path: Path):
        (tmp_path / ".gitignore").write_text("node_modules/\n.env\n")
        update_gitignore_block(tmp_path, ["CLAUDE.md"])

        content = (tmp_path / ".gitignore").read_text()
        assert content.startswith("node_modules/")
        assert "CLAUDE.md" in content

    def test_update_existing_block(self, tmp_path: Path):
        update_gitignore_block(tmp_path, ["CLAUDE.md"])
        update_gitignore_block(tmp_path, ["CLAUDE.md", "GEMINI.md"])

        content = (tmp_path / ".gitignore").read_text()
        assert "GEMINI.md" in content
        # Only one block
        assert content.count(_BLOCK_START) == 1

    def test_remove_block_when_empty(self, tmp_path: Path):
        (tmp_path / ".gitignore").write_text(
            f"node_modules/\n\n{_BLOCK_START}\nCLAUDE.md\n{_BLOCK_END}\n"
        )
        update_gitignore_block(tmp_path, [])

        content = (tmp_path / ".gitignore").read_text()
        assert _BLOCK_START not in content
        assert "CLAUDE.md" not in content
        assert "node_modules/" in content

    def test_no_modification_when_up_to_date(self, tmp_path: Path):
        update_gitignore_block(tmp_path, ["CLAUDE.md"])
        modified = update_gitignore_block(tmp_path, ["CLAUDE.md"])
        assert modified is False

    def test_preserves_user_entries(self, tmp_path: Path):
        (tmp_path / ".gitignore").write_text("# my custom ignore\n*.pyc\n")
        update_gitignore_block(tmp_path, ["CLAUDE.md"])

        content = (tmp_path / ".gitignore").read_text()
        assert "# my custom ignore" in content
        assert "*.pyc" in content

    def test_entries_are_sorted(self, tmp_path: Path):
        update_gitignore_block(
            tmp_path,
            ["GEMINI.md", ".cursorrules", "CLAUDE.md"],
        )
        content = (tmp_path / ".gitignore").read_text()
        lines = content.splitlines()
        # Find the entries between markers
        start = lines.index(_BLOCK_START)
        end = lines.index(_BLOCK_END)
        entries = lines[start + 1 : end]
        assert entries == sorted(entries)

    def test_creates_gitignore_if_missing(self, tmp_path: Path):
        assert not (tmp_path / ".gitignore").exists()
        update_gitignore_block(tmp_path, ["CLAUDE.md"])
        assert (tmp_path / ".gitignore").exists()
