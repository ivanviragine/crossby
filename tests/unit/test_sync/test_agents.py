"""Tests for agent sync writers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from crossby.config.linker import create_symlink
from crossby.models.ai import AIToolID
from crossby.sync.agents import (
    CopilotAgentsWriter,
    ClaudeAgentsWriter,
    CodexAgentsWriter,
    CursorAgentsWriter,
    GeminiAgentsWriter,
    _GITIGNORE_BLOCK_ID,
    _parse_frontmatter,
    _render_frontmatter,
    _translate_tools,
    update_agents_gitignore,
)
from crossby.sync.base import SyncConcern, SyncData

# Derive the markers from the block ID, matching gitignore_utils conventions
_BLOCK_START = f"# >>> crossby {_GITIGNORE_BLOCK_ID} (generated — do not edit) >>>"
_BLOCK_END = f"# <<< crossby {_GITIGNORE_BLOCK_ID} <<<"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _data(
    source: str = ".crossby/agents",
    strategy: str = "symlink",
    gitignore: bool = True,
) -> SyncData:
    return SyncData(
        agents_source=source,
        agents_strategy=strategy,
        agents_gitignore=gitignore,
    )


def _make_source(tmp_path: Path, agents: list[str] | None = None) -> Path:
    """Create a .crossby/agents source dir with optional agent files."""
    source = tmp_path / ".crossby" / "agents"
    source.mkdir(parents=True)
    for name in agents or []:
        (source / name).write_text(
            f"---\nname: {name.removesuffix('.md')}\ndescription: test\n---\nBody.\n",
            encoding="utf-8",
        )
    return source


# ---------------------------------------------------------------------------
# create_symlink (shared helper, used by agents sync)
# ---------------------------------------------------------------------------


class TestCreateDirSymlink:
    def test_creates_relative_symlink(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        source.mkdir()
        link = tmp_path / "sub" / "link"
        created = create_symlink(source, link, force=True, dry_run=False)
        assert created is True
        assert link.is_symlink()
        # Symlink target must be relative
        assert not os.path.isabs(os.readlink(link))
        assert link.resolve() == source.resolve()

    def test_idempotent_correct_symlink(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        source.mkdir()
        link = tmp_path / "link"
        create_symlink(source, link, force=True, dry_run=False)
        created = create_symlink(source, link, force=True, dry_run=False)
        assert created is False  # already correct

    def test_dry_run_does_not_create(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        source.mkdir()
        link = tmp_path / "link"
        created = create_symlink(source, link, force=True, dry_run=True)
        assert created is True
        assert not link.exists()

    def test_wrong_target_updated(self, tmp_path: Path) -> None:
        source_a = tmp_path / "a"
        source_b = tmp_path / "b"
        source_a.mkdir()
        source_b.mkdir()
        link = tmp_path / "link"
        os.symlink(os.path.relpath(source_a, tmp_path), link)
        # Now change to source_b
        created = create_symlink(source_b, link, force=True, dry_run=False)
        assert created is True
        assert link.resolve() == source_b.resolve()

    def test_existing_real_path_skipped(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        source.mkdir()
        link = tmp_path / "link"
        link.mkdir()  # real dir
        created = create_symlink(source, link, force=True, dry_run=False)
        assert created is False  # skipped — linker refuses to remove real dirs


# ---------------------------------------------------------------------------
# Frontmatter helpers
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_parses_valid_frontmatter(self) -> None:
        content = "---\nname: foo\ndescription: bar\n---\nBody text.\n"
        fm, body = _parse_frontmatter(content)
        assert fm == {"name": "foo", "description": "bar"}
        assert body == "Body text.\n"

    def test_no_frontmatter(self) -> None:
        content = "Just some text."
        fm, body = _parse_frontmatter(content)
        assert fm is None
        assert body == "Just some text."

    def test_non_dict_yaml_frontmatter(self) -> None:
        """Non-dict YAML (list, scalar) in frontmatter returns None — copy verbatim."""
        content = "---\n- item1\n- item2\n---\nBody.\n"
        fm, body = _parse_frontmatter(content)
        assert fm is None
        assert body == content  # verbatim, not stripped

    def test_roundtrip(self) -> None:
        fm = {"name": "test", "tools": ["Read", "Bash"]}
        body = "Hello world.\n"
        rendered = _render_frontmatter(fm, body)
        fm2, body2 = _parse_frontmatter(rendered)
        assert fm2 == fm
        assert body2 == body


class TestTranslateTools:
    def test_copilot_translation(self) -> None:
        tools = ["Read", "Bash", "Grep", "Unknown"]
        result = _translate_tools(tools, "copilot")
        assert result == ["read", "shell", "search", "Unknown"]

    def test_cursor_translation(self) -> None:
        tools = ["Read", "Bash", "Grep"]
        result = _translate_tools(tools, "cursor")
        assert result == ["Read", "Shell", "Grep"]

    def test_claude_no_translation(self) -> None:
        tools = ["Read", "Bash", "Grep"]
        assert _translate_tools(tools, "claude") == tools

    def test_codex_no_translation(self) -> None:
        tools = ["WebSearch", "WebFetch"]
        assert _translate_tools(tools, "codex") == tools


# ---------------------------------------------------------------------------
# Directory symlink strategy (non-Copilot writers)
# ---------------------------------------------------------------------------


class TestClaudeAgentsWriter:
    def test_creates_symlink(self, tmp_path: Path) -> None:
        _make_source(tmp_path, ["reviewer.md"])
        w = ClaudeAgentsWriter()
        data = _data(source=".crossby/agents")
        result = w.sync(data, tmp_path)
        assert result.action == "created"
        link = tmp_path / ".claude" / "agents"
        assert link.is_symlink()
        assert link.resolve() == (tmp_path / ".crossby" / "agents").resolve()

    def test_idempotent(self, tmp_path: Path) -> None:
        _make_source(tmp_path, ["a.md"])
        w = ClaudeAgentsWriter()
        data = _data()
        w.sync(data, tmp_path)
        result = w.sync(data, tmp_path)
        assert result.action == "skipped"
        assert result.message == "already linked"

    def test_missing_source_is_error(self, tmp_path: Path) -> None:
        w = ClaudeAgentsWriter()
        data = _data(source=".crossby/agents")
        result = w.sync(data, tmp_path)
        assert result.action == "error"
        assert "source directory not found" in (result.message or "")

    def test_existing_real_dir_is_error_without_force(self, tmp_path: Path) -> None:
        _make_source(tmp_path, ["a.md"])
        target = tmp_path / ".claude" / "agents"
        target.mkdir(parents=True)
        (target / "unmanaged.txt").write_text("user content", encoding="utf-8")
        w = ClaudeAgentsWriter()
        data = _data()
        result = w.sync(data, tmp_path)
        assert result.action == "error"
        assert "--force" in (result.message or "")

    def test_symlinked_md_files_in_target_dir_are_unmanaged(self, tmp_path: Path) -> None:
        """A target dir containing .md symlinks is treated as unmanaged (not a safe fallback)."""
        _make_source(tmp_path, ["a.md"])
        target = tmp_path / ".claude" / "agents"
        target.mkdir(parents=True)
        # Symlinked .md file — should NOT be treated as a managed fallback
        other = tmp_path / "external.md"
        other.write_text("external", encoding="utf-8")
        os.symlink(os.path.relpath(other, target), target / "a.md")
        w = ClaudeAgentsWriter()
        data = _data()
        result = w.sync(data, tmp_path)
        assert result.action == "error"
        assert "--force" in (result.message or "")

    def test_empty_target_dir_is_managed_fallback(self, tmp_path: Path) -> None:
        """An empty target directory is treated as a managed fallback — proceed with copy."""
        _make_source(tmp_path, ["a.md"])
        target = tmp_path / ".claude" / "agents"
        target.mkdir(parents=True)
        w = ClaudeAgentsWriter()
        data = _data()
        result = w.sync(data, tmp_path)
        assert result.action == "created"
        assert (target / "a.md").is_file()

    def test_source_is_file_is_error(self, tmp_path: Path) -> None:
        """Source path that exists as a file (not a directory) returns an error."""
        source = tmp_path / ".crossby" / "agents"
        source.parent.mkdir(parents=True)
        source.write_text("not a directory", encoding="utf-8")
        w = ClaudeAgentsWriter()
        data = _data()
        result = w.sync(data, tmp_path)
        assert result.action == "error"
        assert "not a directory" in (result.message or "")

    def test_force_backs_up_and_replaces(self, tmp_path: Path) -> None:
        _make_source(tmp_path, ["a.md"])
        target = tmp_path / ".claude" / "agents"
        target.mkdir(parents=True)
        (target / "old.md").write_text("old", encoding="utf-8")
        w = ClaudeAgentsWriter()
        data = _data()
        result = w.sync(data, tmp_path, force=True)
        assert result.action == "created"
        assert target.is_symlink()
        backup = Path(str(target) + ".bak")
        assert backup.is_dir()
        assert (backup / "old.md").exists()

    def test_dry_run_no_files_written(self, tmp_path: Path) -> None:
        _make_source(tmp_path, ["a.md"])
        w = ClaudeAgentsWriter()
        data = _data()
        result = w.sync(data, tmp_path, dry_run=True)
        assert result.action == "created"
        assert not (tmp_path / ".claude" / "agents").exists()

    def test_force_dry_run_with_real_dir_reports_created(self, tmp_path: Path) -> None:
        """dry_run=True + force=True with existing real dir should report 'created'."""
        _make_source(tmp_path, ["a.md"])
        target = tmp_path / ".claude" / "agents"
        target.mkdir(parents=True)
        w = ClaudeAgentsWriter()
        data = _data()
        result = w.sync(data, tmp_path, dry_run=True, force=True)
        assert result.action == "created"
        # Nothing written
        assert target.is_dir() and not target.is_symlink()

    def test_force_twice_no_backup_collision(self, tmp_path: Path) -> None:
        """Running --force twice does not crash due to existing .bak directory."""
        _make_source(tmp_path, ["a.md"])
        target = tmp_path / ".claude" / "agents"
        target.mkdir(parents=True)
        (target / "old.md").write_text("old", encoding="utf-8")
        w = ClaudeAgentsWriter()
        data = _data()
        w.sync(data, tmp_path, force=True)
        # Replace symlink with a real dir again for second force run
        (tmp_path / ".claude" / "agents").unlink()
        target.mkdir(parents=True)
        (target / "newer.md").write_text("newer", encoding="utf-8")
        result = w.sync(data, tmp_path, force=True)
        assert result.action == "created"

    def test_regular_file_at_target_is_error(self, tmp_path: Path) -> None:
        """A regular file at the symlink target path returns an error (not 'already linked')."""
        _make_source(tmp_path, ["a.md"])
        target = tmp_path / ".claude" / "agents"
        target.parent.mkdir(parents=True)
        target.write_text("regular file", encoding="utf-8")
        w = ClaudeAgentsWriter()
        data = _data()
        result = w.sync(data, tmp_path)
        assert result.action == "error"
        assert "regular file or directory" in (result.message or "")
        assert "--force" in (result.message or "")

    def test_no_agents_source_skipped(self, tmp_path: Path) -> None:
        """Writers skip when agents_source is None (no agents config)."""
        _make_source(tmp_path, ["a.md"])
        w = ClaudeAgentsWriter()
        data = SyncData()  # agents_source=None by default
        result = w.sync(data, tmp_path)
        assert result.action == "skipped"


class TestRelativeSymlinkPaths:
    """Verify symlink targets are relative, not absolute."""

    @pytest.mark.parametrize(
        "writer_cls, target_rel",
        [
            (ClaudeAgentsWriter, ".claude/agents"),
            (CursorAgentsWriter, ".cursor/agents"),
            (GeminiAgentsWriter, ".gemini/agents"),
            (CodexAgentsWriter, ".agents"),
        ],
    )
    def test_symlink_is_relative(
        self, tmp_path: Path, writer_cls: type, target_rel: str
    ) -> None:
        _make_source(tmp_path, ["a.md"])
        w = writer_cls()
        data = _data()
        w.sync(data, tmp_path)
        link = tmp_path / target_rel
        assert link.is_symlink()
        assert not os.path.isabs(os.readlink(link)), "symlink target must be relative"


# ---------------------------------------------------------------------------
# Copy strategy
# ---------------------------------------------------------------------------


class TestCopyStrategy:
    def test_copy_creates_files(self, tmp_path: Path) -> None:
        _make_source(tmp_path, ["reviewer.md"])
        w = ClaudeAgentsWriter()
        data = _data(strategy="copy")
        result = w.sync(data, tmp_path)
        assert result.action == "created"
        target = tmp_path / ".claude" / "agents" / "reviewer.md"
        assert target.is_file()
        assert not target.is_symlink()

    def test_copy_translates_tool_names_copilot(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path)
        agent_content = (
            "---\nname: test\ndescription: d\ntools:\n  - Read\n  - Bash\n---\nBody.\n"
        )
        (source / "test.md").write_text(agent_content, encoding="utf-8")
        w = CopilotAgentsWriter()
        data = _data(strategy="copy")
        w.sync(data, tmp_path)
        dest = tmp_path / ".github" / "agents" / "test.agent.md"
        content = dest.read_text(encoding="utf-8")
        assert "read" in content
        assert "shell" in content
        assert "Read" not in content
        assert "Bash" not in content

    def test_copy_no_translation_for_claude(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path)
        agent_content = "---\nname: t\ndescription: d\ntools:\n  - Read\n  - Bash\n---\nBody.\n"
        (source / "t.md").write_text(agent_content, encoding="utf-8")
        w = ClaudeAgentsWriter()
        data = _data(strategy="copy")
        w.sync(data, tmp_path)
        dest = tmp_path / ".claude" / "agents" / "t.md"
        content = dest.read_text(encoding="utf-8")
        assert "Read" in content
        assert "Bash" in content

    def test_copy_errors_on_unmanaged_directory(self, tmp_path: Path) -> None:
        """Copy strategy also errors when target has unmanaged (non-.md) content."""
        _make_source(tmp_path, ["a.md"])
        target = tmp_path / ".claude" / "agents"
        target.mkdir(parents=True)
        (target / "unmanaged.txt").write_text("user content", encoding="utf-8")
        w = ClaudeAgentsWriter()
        data = _data(strategy="copy")
        result = w.sync(data, tmp_path)
        assert result.action == "error"
        assert "--force" in (result.message or "")

    def test_copy_handles_managed_directory(self, tmp_path: Path) -> None:
        """Copy strategy is re-entrant when target contains only .md files."""
        _make_source(tmp_path, ["a.md"])
        target = tmp_path / ".claude" / "agents"
        target.mkdir(parents=True)
        (target / "a.md").write_text("old content", encoding="utf-8")
        w = ClaudeAgentsWriter()
        data = _data(strategy="copy")
        result = w.sync(data, tmp_path)
        assert result.action == "created"

    def test_copy_errors_on_symlinked_target_without_force(self, tmp_path: Path) -> None:
        """Copy strategy errors when target path is a symlink (would write outside project)."""
        _make_source(tmp_path, ["a.md"])
        target = tmp_path / ".claude" / "agents"
        target.parent.mkdir(parents=True)
        other = tmp_path / "other"
        other.mkdir()
        os.symlink(os.path.relpath(other, target.parent), target)
        w = ClaudeAgentsWriter()
        data = _data(strategy="copy")
        result = w.sync(data, tmp_path)
        assert result.action == "error"
        assert "symlink" in (result.message or "")
        assert "--force" in (result.message or "")

    def test_copy_replaces_symlinked_target_with_force(self, tmp_path: Path) -> None:
        """With --force, copy strategy removes the symlink and copies into a real directory."""
        _make_source(tmp_path, ["a.md"])
        target = tmp_path / ".claude" / "agents"
        target.parent.mkdir(parents=True)
        other = tmp_path / "other"
        other.mkdir()
        os.symlink(os.path.relpath(other, target.parent), target)
        w = ClaudeAgentsWriter()
        data = _data(strategy="copy")
        result = w.sync(data, tmp_path, force=True)
        assert result.action == "created"
        assert not target.is_symlink()
        assert (target / "a.md").is_file()


# ---------------------------------------------------------------------------
# Copilot writer (file-level symlinks)
# ---------------------------------------------------------------------------


class TestCopilotAgentsWriter:
    def test_creates_agent_md_symlinks(self, tmp_path: Path) -> None:
        _make_source(tmp_path, ["reviewer.md", "tester.md"])
        w = CopilotAgentsWriter()
        data = _data()
        result = w.sync(data, tmp_path)
        assert result.action == "created"
        target_dir = tmp_path / ".github" / "agents"
        assert (target_dir / "reviewer.agent.md").is_symlink()
        assert (target_dir / "tester.agent.md").is_symlink()
        assert not (target_dir / "reviewer.md").exists()

    def test_symlinks_are_relative(self, tmp_path: Path) -> None:
        _make_source(tmp_path, ["a.md"])
        w = CopilotAgentsWriter()
        data = _data()
        w.sync(data, tmp_path)
        link = tmp_path / ".github" / "agents" / "a.agent.md"
        assert not os.path.isabs(os.readlink(link))

    def test_idempotent(self, tmp_path: Path) -> None:
        _make_source(tmp_path, ["a.md"])
        w = CopilotAgentsWriter()
        data = _data()
        w.sync(data, tmp_path)
        result = w.sync(data, tmp_path)
        assert result.action == "skipped"

    def test_stale_cleanup(self, tmp_path: Path) -> None:
        """Stale .agent.md symlinks are removed when source file is deleted."""
        source = _make_source(tmp_path, ["a.md", "old.md"])
        w = CopilotAgentsWriter()
        data = _data()
        w.sync(data, tmp_path)
        # Remove old.md from source
        (source / "old.md").unlink()
        # Re-sync
        w.sync(data, tmp_path)
        target_dir = tmp_path / ".github" / "agents"
        assert not (target_dir / "old.agent.md").exists()
        assert (target_dir / "a.agent.md").is_symlink()

    def test_missing_source_is_error(self, tmp_path: Path) -> None:
        w = CopilotAgentsWriter()
        data = _data()
        result = w.sync(data, tmp_path)
        assert result.action == "error"

    def test_copy_strategy_uses_agent_md_extension(self, tmp_path: Path) -> None:
        _make_source(tmp_path, ["foo.md"])
        w = CopilotAgentsWriter()
        data = _data(strategy="copy")
        w.sync(data, tmp_path)
        dest = tmp_path / ".github" / "agents" / "foo.agent.md"
        assert dest.is_file()
        assert not dest.is_symlink()

    def test_existing_real_dir_with_unmanaged_content_is_error(self, tmp_path: Path) -> None:
        """A pre-existing directory with non-.agent.md files blocks sync (needs --force)."""
        _make_source(tmp_path, ["a.md"])
        target = tmp_path / ".github" / "agents"
        target.mkdir(parents=True)
        (target / "existing.md").write_text("user content", encoding="utf-8")
        w = CopilotAgentsWriter()
        data = _data()
        result = w.sync(data, tmp_path)
        assert result.action == "error"
        assert "--force" in (result.message or "")

    def test_force_works_with_unmanaged_content(self, tmp_path: Path) -> None:
        _make_source(tmp_path, ["a.md"])
        target = tmp_path / ".github" / "agents"
        target.mkdir(parents=True)
        (target / "existing.md").write_text("user content", encoding="utf-8")
        w = CopilotAgentsWriter()
        data = _data()
        result = w.sync(data, tmp_path, force=True)
        assert result.action == "created"
        assert (target / "a.agent.md").is_symlink()

    def test_force_backs_up_unmanaged_directory(self, tmp_path: Path) -> None:
        """--force with unmanaged content backs up the directory before replacing."""
        _make_source(tmp_path, ["a.md"])
        target = tmp_path / ".github" / "agents"
        target.mkdir(parents=True)
        (target / "existing.md").write_text("user content", encoding="utf-8")
        w = CopilotAgentsWriter()
        data = _data()
        w.sync(data, tmp_path, force=True)
        backup = tmp_path / ".github" / "agents.bak"
        assert backup.is_dir()
        assert (backup / "existing.md").exists()

    def test_force_false_does_not_overwrite_wrong_symlink(self, tmp_path: Path) -> None:
        """Without --force, a wrong existing per-file symlink returns an error."""
        source = _make_source(tmp_path, ["a.md"])
        other = tmp_path / "other.md"
        other.write_text("other", encoding="utf-8")
        target_dir = tmp_path / ".github" / "agents"
        target_dir.mkdir(parents=True)
        link = target_dir / "a.agent.md"
        os.symlink(os.path.relpath(other, target_dir), link)
        w = CopilotAgentsWriter()
        data = _data()
        result = w.sync(data, tmp_path, force=False)
        assert result.action == "error"
        assert "--force" in (result.message or "")
        assert link.resolve() == other.resolve()  # symlink still points to "other"

    def test_source_is_file_is_error(self, tmp_path: Path) -> None:
        """Source path that exists as a file (not a directory) returns an error."""
        source = tmp_path / ".crossby" / "agents"
        source.parent.mkdir(parents=True)
        source.write_text("not a directory", encoding="utf-8")
        w = CopilotAgentsWriter()
        data = _data()
        result = w.sync(data, tmp_path)
        assert result.action == "error"
        assert "not a directory" in (result.message or "")

    def test_regular_file_at_link_path_is_updated(self, tmp_path: Path) -> None:
        """A regular .agent.md file (copy-fallback output) is updated via copy, not errored."""
        source = _make_source(tmp_path, ["a.md"])
        target_dir = tmp_path / ".github" / "agents"
        target_dir.mkdir(parents=True)
        (target_dir / "a.agent.md").write_text("stale content", encoding="utf-8")
        w = CopilotAgentsWriter()
        data = _data()
        result = w.sync(data, tmp_path)
        assert result.action == "created"
        # Content should now match what _copy_agent_file produces from source
        assert (target_dir / "a.agent.md").read_text(encoding="utf-8") != "stale content"

    def test_stale_regular_file_is_cleaned_up(self, tmp_path: Path) -> None:
        """A stale regular .agent.md file (copy-fallback) is removed when its source is gone."""
        _make_source(tmp_path, ["a.md"])
        target_dir = tmp_path / ".github" / "agents"
        target_dir.mkdir(parents=True)
        # Simulate a prior copy-fallback output for a source that no longer exists
        (target_dir / "old.agent.md").write_text("stale copy from prior run", encoding="utf-8")
        w = CopilotAgentsWriter()
        data = _data()
        w.sync(data, tmp_path)
        assert not (target_dir / "old.agent.md").exists()
        assert (target_dir / "a.agent.md").is_symlink()

    def test_target_symlink_is_error_without_force(self, tmp_path: Path) -> None:
        """A symlink at the target directory path errors without --force."""
        _make_source(tmp_path, ["a.md"])
        target_dir = tmp_path / ".github" / "agents"
        target_dir.parent.mkdir(parents=True)
        other = tmp_path / "other"
        other.mkdir()
        os.symlink(os.path.relpath(other, target_dir.parent), target_dir)
        w = CopilotAgentsWriter()
        data = _data()
        result = w.sync(data, tmp_path)
        assert result.action == "error"
        assert "symlink" in (result.message or "")

    def test_target_symlink_replaced_with_force(self, tmp_path: Path) -> None:
        """With --force, a symlink at the target path is replaced with a real directory."""
        _make_source(tmp_path, ["a.md"])
        target_dir = tmp_path / ".github" / "agents"
        target_dir.parent.mkdir(parents=True)
        other = tmp_path / "other"
        other.mkdir()
        os.symlink(os.path.relpath(other, target_dir.parent), target_dir)
        w = CopilotAgentsWriter()
        data = _data()
        result = w.sync(data, tmp_path, force=True)
        assert result.action == "created"
        assert not target_dir.is_symlink()
        assert (target_dir / "a.agent.md").exists()


# ---------------------------------------------------------------------------
# Gitignore managed block
# ---------------------------------------------------------------------------


class TestUpdateAgentsGitignore:
    def test_creates_block_when_missing(self, tmp_path: Path) -> None:
        data = _data()
        result = update_agents_gitignore(data, tmp_path)
        assert result is not None
        assert result.action in ("created", "updated")
        content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        assert _BLOCK_START in content
        assert ".claude/agents" in content
        assert _BLOCK_END in content

    def test_appends_to_existing_gitignore(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc\n__pycache__\n", encoding="utf-8")
        data = _data()
        update_agents_gitignore(data, tmp_path)
        content = gitignore.read_text(encoding="utf-8")
        assert "*.pyc" in content
        assert _BLOCK_START in content

    def test_malformed_block_missing_end_marker(self, tmp_path: Path) -> None:
        """A malformed block (start but no end marker) replaces from orphan to EOF."""
        gitignore = tmp_path / ".gitignore"
        malformed = f"*.pyc\n{_BLOCK_START}\n.old/agents\n# end marker is missing\n"
        gitignore.write_text(malformed, encoding="utf-8")
        data = _data()
        update_agents_gitignore(data, tmp_path)
        content = gitignore.read_text(encoding="utf-8")
        # Content before orphan preserved
        assert "*.pyc" in content
        # Fresh block replaces orphaned content
        assert content.count(_BLOCK_START) == 1
        assert _BLOCK_END in content
        assert ".claude/agents" in content

    def test_replaces_existing_block(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        old_block = f"{_BLOCK_START}\n.old/agents\n{_BLOCK_END}\n"
        gitignore.write_text(old_block, encoding="utf-8")
        data = _data()
        update_agents_gitignore(data, tmp_path)
        content = gitignore.read_text(encoding="utf-8")
        assert ".old/agents" not in content
        assert ".claude/agents" in content
        # Only one block
        assert content.count(_BLOCK_START) == 1

    def test_idempotent(self, tmp_path: Path) -> None:
        data = _data()
        update_agents_gitignore(data, tmp_path)
        content_before = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        result = update_agents_gitignore(data, tmp_path)
        assert result is None  # No change
        content_after = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        assert content_before == content_after

    def test_installed_tools_filter_entries(self, tmp_path: Path) -> None:
        data = _data()
        update_agents_gitignore(
            data, tmp_path, installed_tools=[AIToolID.CLAUDE, AIToolID.CURSOR]
        )
        content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        assert ".claude/agents" in content
        assert ".cursor/agents" in content
        assert ".github/agents" not in content
        assert ".gemini/agents" not in content

    def test_gitignore_false_does_nothing(self, tmp_path: Path) -> None:
        data = _data(gitignore=False)
        result = update_agents_gitignore(data, tmp_path)
        assert result is None
        assert not (tmp_path / ".gitignore").exists()

    def test_dry_run_no_write(self, tmp_path: Path) -> None:
        data = _data()
        result = update_agents_gitignore(data, tmp_path, dry_run=True)
        assert result is not None
        assert not (tmp_path / ".gitignore").exists()

    def test_tool_id_is_none(self, tmp_path: Path) -> None:
        """Gitignore result should have tool_id=None (cross-tool operation)."""
        data = _data()
        result = update_agents_gitignore(data, tmp_path)
        assert result is not None
        assert result.tool_id is None

    def test_action_created_when_file_absent(self, tmp_path: Path) -> None:
        """action='created' when .gitignore doesn't exist yet."""
        data = _data()
        result = update_agents_gitignore(data, tmp_path)
        assert result is not None
        assert result.action == "created"

    def test_action_updated_when_file_exists_but_empty(self, tmp_path: Path) -> None:
        """action='updated' for an existing but empty .gitignore (not 'created')."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("", encoding="utf-8")
        data = _data()
        result = update_agents_gitignore(data, tmp_path)
        assert result is not None
        assert result.action == "updated"

    def test_installed_tools_filters_entries_when_targets_empty(self, tmp_path: Path) -> None:
        """With no agents.targets, installed_tools restricts gitignore entries."""
        data = _data()  # targets={}
        result = update_agents_gitignore(
            data, tmp_path, installed_tools=[AIToolID.CLAUDE, AIToolID.CURSOR]
        )
        assert result is not None
        content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        assert ".claude/agents" in content
        assert ".cursor/agents" in content
        assert ".github/agents" not in content
        assert ".gemini/agents" not in content

    def test_installed_tools_none_includes_all(self, tmp_path: Path) -> None:
        """With installed_tools=None and no targets, all known paths are included."""
        data = _data()
        result = update_agents_gitignore(data, tmp_path, installed_tools=None)
        assert result is not None
        content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        assert ".claude/agents" in content
        assert ".github/agents" in content
        assert ".cursor/agents" in content


# ---------------------------------------------------------------------------
# Integration: run_sync with agents concern
# ---------------------------------------------------------------------------


class TestRunSyncAgents:
    def test_full_sync_all_tools(self, tmp_path: Path) -> None:
        from crossby.sync import run_sync
        from crossby.sync.base import SyncRegistry
        from crossby.sync.agents import (
            ClaudeAgentsWriter,
            CopilotAgentsWriter,
            CursorAgentsWriter,
        )

        _make_source(tmp_path, ["a.md"])
        reg = SyncRegistry()
        reg.register(ClaudeAgentsWriter())
        reg.register(CopilotAgentsWriter())
        reg.register(CursorAgentsWriter())

        data = _data()
        results = run_sync(
            data,
            tmp_path,
            tool_id=None,
            concern=SyncConcern.AGENTS,
            installed_tools=[AIToolID.CLAUDE, AIToolID.COPILOT, AIToolID.CURSOR],
            registry=reg,
        )

        tool_ids = [r.tool_id for r in results]
        assert AIToolID.CLAUDE in tool_ids
        assert AIToolID.COPILOT in tool_ids
        assert AIToolID.CURSOR in tool_ids

        # Gitignore should have been updated
        assert (tmp_path / ".gitignore").exists()
        content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        assert _BLOCK_START in content

    def test_dry_run_no_files_written(self, tmp_path: Path) -> None:
        from crossby.sync import run_sync
        from crossby.sync.base import SyncRegistry

        _make_source(tmp_path, ["a.md"])
        reg = SyncRegistry()
        reg.register(ClaudeAgentsWriter())

        data = _data()
        run_sync(
            data,
            tmp_path,
            concern=SyncConcern.AGENTS,
            installed_tools=[AIToolID.CLAUDE],
            dry_run=True,
            registry=reg,
        )

        assert not (tmp_path / ".claude" / "agents").exists()
        assert not (tmp_path / ".gitignore").exists()

    def test_force_forwarded_to_writers(self, tmp_path: Path) -> None:
        """force=True passed to run_sync is forwarded to each writer."""
        from crossby.sync import run_sync
        from crossby.sync.base import SyncRegistry

        _make_source(tmp_path, ["a.md"])
        # Create a real directory at the target — without force this would error
        target = tmp_path / ".claude" / "agents"
        target.mkdir(parents=True)
        (target / "unmanaged.txt").write_text("user content", encoding="utf-8")

        reg = SyncRegistry()
        reg.register(ClaudeAgentsWriter())

        data = _data()
        results = run_sync(
            data,
            tmp_path,
            concern=SyncConcern.AGENTS,
            installed_tools=[AIToolID.CLAUDE],
            force=True,
            registry=reg,
        )

        actions = [r.action for r in results]
        assert "created" in actions  # force allowed the backup+replace
