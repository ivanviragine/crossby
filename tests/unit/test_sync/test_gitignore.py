"""Tests for rules .gitignore managed-block logic."""

from pathlib import Path

from crossby.models.config import CrossbyConfig, RulesConfig, RulesTargetsConfig
from crossby.sync.rules import _BLOCK_END, _BLOCK_START, update_rules_gitignore


def _cfg(
    source: str = "AGENTS.md",
    targets: RulesTargetsConfig | None = None,
    gitignore: bool = True,
) -> CrossbyConfig:
    return CrossbyConfig(
        rules=RulesConfig(
            enabled=True,
            source=source,
            gitignore=gitignore,
            targets=targets or RulesTargetsConfig(),
        ),
    )


def _setup_source(tmp_path: Path, source: str = "AGENTS.md") -> None:
    """Create the source file so gitignore entries can be computed."""
    p = tmp_path / source
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# rules\n")


class TestUpdateRulesGitignore:
    def test_add_block_to_empty_file(self, tmp_path: Path):
        _setup_source(tmp_path)
        result = update_rules_gitignore(_cfg(), tmp_path)
        assert result is not None

        content = (tmp_path / ".gitignore").read_text()
        assert _BLOCK_START in content
        assert _BLOCK_END in content
        assert ".cursorrules" in content
        assert "CLAUDE.md" in content

    def test_add_block_to_existing_gitignore(self, tmp_path: Path):
        _setup_source(tmp_path)
        (tmp_path / ".gitignore").write_text("node_modules/\n.env\n")
        update_rules_gitignore(_cfg(), tmp_path)

        content = (tmp_path / ".gitignore").read_text()
        assert content.startswith("node_modules/")
        assert "CLAUDE.md" in content

    def test_update_existing_block(self, tmp_path: Path):
        _setup_source(tmp_path)
        # First: only claude
        update_rules_gitignore(
            _cfg(targets=RulesTargetsConfig(claude=True, cursor=False, copilot=False, gemini=False, codex=False)),
            tmp_path,
        )
        # Then: claude + gemini
        update_rules_gitignore(
            _cfg(targets=RulesTargetsConfig(claude=True, cursor=False, copilot=False, gemini=True, codex=False)),
            tmp_path,
        )

        content = (tmp_path / ".gitignore").read_text()
        assert "GEMINI.md" in content
        assert content.count(_BLOCK_START) == 1

    def test_no_modification_when_up_to_date(self, tmp_path: Path):
        _setup_source(tmp_path)
        update_rules_gitignore(_cfg(), tmp_path)
        result = update_rules_gitignore(_cfg(), tmp_path)
        assert result is None

    def test_preserves_user_entries(self, tmp_path: Path):
        _setup_source(tmp_path)
        (tmp_path / ".gitignore").write_text("# my custom ignore\n*.pyc\n")
        update_rules_gitignore(_cfg(), tmp_path)

        content = (tmp_path / ".gitignore").read_text()
        assert "# my custom ignore" in content
        assert "*.pyc" in content

    def test_entries_are_sorted(self, tmp_path: Path):
        _setup_source(tmp_path)
        update_rules_gitignore(_cfg(), tmp_path)
        content = (tmp_path / ".gitignore").read_text()
        lines = content.splitlines()
        start = lines.index(_BLOCK_START)
        end = lines.index(_BLOCK_END)
        entries = lines[start + 1 : end]
        assert entries == sorted(entries)

    def test_creates_gitignore_if_missing(self, tmp_path: Path):
        _setup_source(tmp_path)
        assert not (tmp_path / ".gitignore").exists()
        update_rules_gitignore(_cfg(), tmp_path)
        assert (tmp_path / ".gitignore").exists()

    def test_orphan_start_marker_does_not_duplicate(self, tmp_path: Path):
        """Orphan _BLOCK_START without _BLOCK_END should not create a second block."""
        _setup_source(tmp_path)
        (tmp_path / ".gitignore").write_text(
            f"node_modules/\n{_BLOCK_START}\nCLAUDE.md\n"
        )
        update_rules_gitignore(_cfg(), tmp_path)

        content = (tmp_path / ".gitignore").read_text()
        assert content.count(_BLOCK_START) == 1

    def test_returns_none_when_disabled(self, tmp_path: Path):
        _setup_source(tmp_path)
        result = update_rules_gitignore(_cfg(gitignore=False), tmp_path)
        assert result is None

    def test_returns_none_when_no_targets(self, tmp_path: Path):
        _setup_source(tmp_path)
        config = _cfg(targets=RulesTargetsConfig(
            claude=False, cursor=False, copilot=False, gemini=False, codex=False,
        ))
        result = update_rules_gitignore(config, tmp_path)
        assert result is None

    def test_skips_circular_target(self, tmp_path: Path):
        """AGENTS.md as source + codex target → codex excluded from gitignore."""
        _setup_source(tmp_path)
        config = _cfg(targets=RulesTargetsConfig(
            claude=True, cursor=False, copilot=False, gemini=False, codex=True,
        ))
        update_rules_gitignore(config, tmp_path)

        content = (tmp_path / ".gitignore").read_text()
        assert "CLAUDE.md" in content
        # AGENTS.md is both source and codex target — should NOT be in gitignore
        assert "AGENTS.md" not in content

    def test_all_enabled_targets_present_on_idempotent_run(self, tmp_path: Path):
        """Gitignore block includes ALL enabled targets, not just newly-synced ones."""
        _setup_source(tmp_path)
        config = _cfg()
        # First run creates the block
        update_rules_gitignore(config, tmp_path)
        content1 = (tmp_path / ".gitignore").read_text()
        assert "CLAUDE.md" in content1
        assert ".cursorrules" in content1

        # Second run (idempotent) — block should remain identical
        result = update_rules_gitignore(config, tmp_path)
        assert result is None  # No change
        content2 = (tmp_path / ".gitignore").read_text()
        assert content1 == content2
