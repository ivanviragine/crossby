"""Regressions for the four sites where crossby destroyed user content.

Each test here corresponds to a verified data-loss path:

1. ``sync --from X --to X --force`` backed up and ``rmtree``'d the *source*
   agents directory, then reported ``skipped`` with exit code 0.
2. The skills ``copy`` strategy ``rmtree``'d the whole target tree on every run.
3. ``.codex/config.toml`` was rewritten non-atomically through a
   ``tomllib``/``tomli_w`` round trip that discarded every comment.
4. ``crossby init --force`` overwrote ``.crossby.yml`` with no backup, dropping
   ``models:`` and ``profiles:`` entirely.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from crossby.models.ai import AIToolID
from crossby.models.config import HookEntry, MCPServerConfig
from crossby.sync import run_sync
from crossby.sync.agents import (
    ClaudeAgentsWriter,
    CodexAgentsWriter,
    CopilotAgentsWriter,
)
from crossby.sync.base import SyncData, SyncRegistry
from crossby.sync.file_utils import is_same_path
from crossby.sync.hooks import (
    ClaudeHooksWriter,
    _ensure_codex_hooks_feature_flag,
)
from crossby.sync.mcp import ClaudeMCPWriter, CodexMCPWriter
from crossby.sync.permissions import ClaudePermissionWriter
from crossby.sync.skills import ClaudeSkillsWriter

# ---------------------------------------------------------------------------
# 1. --from X --to X --force must not delete the source
# ---------------------------------------------------------------------------


def _agent_file(directory: Path, name: str = "a") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.md"
    path.write_text(f"---\nname: {name}\n---\nprecious\n", encoding="utf-8")
    return path


class TestAgentsSourceEqualsTarget:
    """``--from claude --to claude --force`` used to wipe ``.claude/agents``."""

    @pytest.mark.parametrize(
        ("writer", "shared_rel"),
        [
            (ClaudeAgentsWriter(), ".claude/agents"),
            (CodexAgentsWriter(), ".codex/agents"),
            (CopilotAgentsWriter(), ".github/agents"),
        ],
        ids=["claude", "codex", "copilot"],
    )
    @pytest.mark.parametrize("strategy", ["symlink", "copy", "translate"])
    def test_source_survives_force(
        self, writer, shared_rel: str, strategy: str, tmp_path: Path
    ) -> None:
        agent = _agent_file(tmp_path / shared_rel)

        result = writer.sync(
            SyncData(agents_source=shared_rel, agents_strategy=strategy),
            tmp_path,
            force=True,
        )

        assert result.action == "skipped"
        assert "same path" in (result.message or "")
        assert agent.is_file(), "the source agent file was deleted"
        assert agent.read_text(encoding="utf-8").endswith("precious\n")
        # No backup means nothing was cleared in the first place.
        assert not (tmp_path / f"{shared_rel}.bak").exists()

    def test_symlinked_source_into_target_is_detected(self, tmp_path: Path) -> None:
        """A source that is a symlink *into* the target is the same collision."""
        real = tmp_path / ".claude" / "agents"
        agent = _agent_file(real)
        source = tmp_path / ".crossby" / "agents"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.symlink_to(real, target_is_directory=True)

        result = ClaudeAgentsWriter().sync(
            SyncData(agents_source=".crossby/agents", agents_strategy="copy"),
            tmp_path,
            force=True,
        )

        assert result.action == "skipped"
        assert agent.is_file()

    def test_distinct_paths_still_sync(self, tmp_path: Path) -> None:
        """The guard must not fire for a normal source/target pair."""
        _agent_file(tmp_path / ".crossby" / "agents")
        result = ClaudeAgentsWriter().sync(
            SyncData(agents_source=".crossby/agents", agents_strategy="copy"),
            tmp_path,
        )
        assert result.action == "created"
        assert (tmp_path / ".claude" / "agents" / "a.md").is_file()


class TestIsSamePath:
    def test_identical_paths(self, tmp_path: Path) -> None:
        (tmp_path / "d").mkdir()
        assert is_same_path(tmp_path / "d", tmp_path / "d")

    def test_distinct_paths(self, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        assert not is_same_path(tmp_path / "a", tmp_path / "b")

    def test_symlinked_target_is_not_a_collision(self, tmp_path: Path) -> None:
        """An existing symlink target is the idempotent re-run, not a collision."""
        source = tmp_path / "source"
        source.mkdir()
        target = tmp_path / "target"
        target.symlink_to(source, target_is_directory=True)
        assert not is_same_path(source, target)

    def test_missing_paths_do_not_raise(self, tmp_path: Path) -> None:
        assert not is_same_path(tmp_path / "nope", tmp_path / "also-nope")


# ---------------------------------------------------------------------------
# 2. Skills copy must not wipe the target tree
# ---------------------------------------------------------------------------


def _make_skill(directory: Path, name: str, body: str = "body") -> None:
    skill = directory / name
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: d\n---\n{body}\n", encoding="utf-8"
    )


def _skills_data(strategy: str = "copy") -> SyncData:
    return SyncData(
        skills_source=".crossby/skills",
        skills_strategy=strategy,
        skills_gitignore=False,
    )


class TestSkillsCopyIsNotDestructive:
    def test_unrelated_target_file_survives_resync(self, tmp_path: Path) -> None:
        source = tmp_path / ".crossby" / "skills"
        _make_skill(source, "skill-a")
        writer = ClaudeSkillsWriter()
        target = tmp_path / ".claude" / "skills"

        assert writer.sync(_skills_data(), tmp_path).action == "created"
        keepsake = target / "NOTES.md"
        keepsake.write_text("hand-written notes\n", encoding="utf-8")

        writer.sync(_skills_data(), tmp_path)

        assert keepsake.is_file(), "an unrelated file in the target was deleted"
        assert keepsake.read_text(encoding="utf-8") == "hand-written notes\n"

    def test_unchanged_resync_writes_nothing(self, tmp_path: Path) -> None:
        source = tmp_path / ".crossby" / "skills"
        _make_skill(source, "skill-a")
        writer = ClaudeSkillsWriter()
        target = tmp_path / ".claude" / "skills"

        writer.sync(_skills_data(), tmp_path)
        synced = target / "skill-a" / "SKILL.md"
        before = synced.stat().st_mtime_ns

        result = writer.sync(_skills_data(), tmp_path)

        assert result.action == "skipped"
        assert synced.stat().st_mtime_ns == before, "an unchanged file was rewritten"

    def test_stale_skill_directory_is_removed(self, tmp_path: Path) -> None:
        source = tmp_path / ".crossby" / "skills"
        _make_skill(source, "skill-a")
        _make_skill(source, "skill-b")
        writer = ClaudeSkillsWriter()
        target = tmp_path / ".claude" / "skills"

        writer.sync(_skills_data(), tmp_path)
        assert (target / "skill-b").is_dir()

        import shutil

        shutil.rmtree(source / "skill-b")
        result = writer.sync(_skills_data(), tmp_path)

        assert result.action == "updated"
        assert not (target / "skill-b").exists()
        assert (target / "skill-a" / "SKILL.md").is_file()

    def test_edited_skill_is_refreshed(self, tmp_path: Path) -> None:
        source = tmp_path / ".crossby" / "skills"
        _make_skill(source, "skill-a", body="first")
        writer = ClaudeSkillsWriter()
        target = tmp_path / ".claude" / "skills"

        writer.sync(_skills_data(), tmp_path)
        _make_skill(source, "skill-a", body="second")
        result = writer.sync(_skills_data(), tmp_path)

        assert result.action == "updated"
        assert "second" in (target / "skill-a" / "SKILL.md").read_text(encoding="utf-8")

    def test_source_directory_over_a_target_file(self, tmp_path: Path) -> None:
        """A leftover target *file* must not block a same-named source skill.

        The stale-cleanup pass only removes directories, so such a file sticks
        around; mirroring a skill dir onto it used to raise FileExistsError and
        abort the whole sync with a traceback.
        """
        source = tmp_path / ".crossby" / "skills"
        _make_skill(source, "skill-a")
        target = tmp_path / ".claude" / "skills"
        target.mkdir(parents=True)
        (target / ".crossby-managed").write_text("", encoding="utf-8")
        (target / "skill-a").write_text("stale file in the way\n", encoding="utf-8")

        result = ClaudeSkillsWriter().sync(_skills_data(), tmp_path)

        assert result.action == "updated"
        assert (target / "skill-a" / "SKILL.md").is_file()

    def test_source_file_over_a_target_directory(self, tmp_path: Path) -> None:
        """The mirror image: a target directory where the source has a file."""
        source = tmp_path / ".crossby" / "skills"
        source.mkdir(parents=True)
        (source / "NOTES.md").write_text("notes\n", encoding="utf-8")
        target = tmp_path / ".claude" / "skills"
        (target / "NOTES.md").mkdir(parents=True)
        (target / ".crossby-managed").write_text("", encoding="utf-8")

        result = ClaudeSkillsWriter().sync(_skills_data(), tmp_path)

        assert result.action == "updated"
        assert (target / "NOTES.md").read_text(encoding="utf-8") == "notes\n"

    def test_stale_file_inside_a_skill_is_removed(self, tmp_path: Path) -> None:
        """Inside a skill directory crossby owns everything, so it mirrors exactly."""
        source = tmp_path / ".crossby" / "skills"
        _make_skill(source, "skill-a")
        (source / "skill-a" / "scripts").mkdir()
        (source / "skill-a" / "scripts" / "run.sh").write_text("echo hi\n", encoding="utf-8")
        writer = ClaudeSkillsWriter()
        target = tmp_path / ".claude" / "skills"

        writer.sync(_skills_data(), tmp_path)
        assert (target / "skill-a" / "scripts" / "run.sh").is_file()

        (source / "skill-a" / "scripts" / "run.sh").unlink()
        writer.sync(_skills_data(), tmp_path)

        assert not (target / "skill-a" / "scripts" / "run.sh").exists()


# ---------------------------------------------------------------------------
# 3. .codex/config.toml keeps its comments and is written atomically
# ---------------------------------------------------------------------------


_HAND_WRITTEN_CONFIG = """\
# My codex config — hand maintained.
model = "gpt-5.5"

# Sandbox policy: keep this strict.
[sandbox]
mode = "workspace-write"   # do not loosen

# Profiles I actually use.
[profiles.fast]
model = "gpt-5.4-mini"
"""


class TestCodexConfigPreservation:
    def test_mcp_sync_keeps_comments(self, tmp_path: Path) -> None:
        path = tmp_path / ".codex" / "config.toml"
        path.parent.mkdir(parents=True)
        path.write_text(_HAND_WRITTEN_CONFIG, encoding="utf-8")

        result = CodexMCPWriter().sync(
            SyncData(mcp_servers={"srv": MCPServerConfig(command="run-me")}), tmp_path
        )
        assert result.action == "updated"

        text = path.read_text(encoding="utf-8")
        for comment in (
            "# My codex config — hand maintained.",
            "# Sandbox policy: keep this strict.",
            "# do not loosen",
            "# Profiles I actually use.",
        ):
            assert comment in text, f"lost comment: {comment}"

        parsed = tomllib.loads(text)
        assert parsed["mcp_servers"]["srv"]["command"] == "run-me"
        assert parsed["model"] == "gpt-5.5"
        assert parsed["profiles"]["fast"]["model"] == "gpt-5.4-mini"

    def test_hooks_feature_flag_keeps_comments(self, tmp_path: Path) -> None:
        path = tmp_path / ".codex" / "config.toml"
        path.parent.mkdir(parents=True)
        path.write_text(_HAND_WRITTEN_CONFIG, encoding="utf-8")

        assert _ensure_codex_hooks_feature_flag(tmp_path, dry_run=False) is None

        text = path.read_text(encoding="utf-8")
        assert "# Sandbox policy: keep this strict." in text
        assert "# Profiles I actually use." in text
        assert tomllib.loads(text)["features"]["codex_hooks"] is True

    def test_existing_server_is_replaced_not_duplicated(self, tmp_path: Path) -> None:
        path = tmp_path / ".codex" / "config.toml"
        path.parent.mkdir(parents=True)
        path.write_text(
            '# keep\n[mcp_servers.srv]\ncommand = "old"\n\n[mcp_servers.srv.env]\nK = "v"\n',
            encoding="utf-8",
        )

        CodexMCPWriter().sync(
            SyncData(mcp_servers={"srv": MCPServerConfig(command="new")}), tmp_path
        )

        text = path.read_text(encoding="utf-8")
        assert text.count("[mcp_servers.srv]") == 1
        parsed = tomllib.loads(text)
        assert parsed["mcp_servers"]["srv"]["command"] == "new"
        assert "env" not in parsed["mcp_servers"]["srv"]
        assert "# keep" in text

    def test_malformed_config_still_falls_back_safely(self, tmp_path: Path) -> None:
        """An unparseable config is reported, never partially rewritten."""
        path = tmp_path / ".codex" / "config.toml"
        path.parent.mkdir(parents=True)
        path.write_text("this is [[not valid\n", encoding="utf-8")

        with pytest.warns(UserWarning):
            result = CodexMCPWriter().sync(
                SyncData(mcp_servers={"srv": MCPServerConfig(command="x")}), tmp_path
            )

        assert result.action == "error"
        assert path.read_text(encoding="utf-8") == "this is [[not valid\n"

    def test_no_temp_file_left_behind(self, tmp_path: Path) -> None:
        path = tmp_path / ".codex" / "config.toml"
        path.parent.mkdir(parents=True)
        path.write_text(_HAND_WRITTEN_CONFIG, encoding="utf-8")

        CodexMCPWriter().sync(
            SyncData(mcp_servers={"srv": MCPServerConfig(command="run-me")}), tmp_path
        )

        assert not list(path.parent.glob("*.tmp"))


# ---------------------------------------------------------------------------
# 4. init --force backs up and preserves unrendered sections
# ---------------------------------------------------------------------------


_HAND_WRITTEN_CROSSBY_YML = """\
version: 1

ai:
  default_tool: claude

models:
  claude:
    easy: claude-haiku-4.5
    very_complex: claude-opus-5

profiles:
  ccyolo:
    tool: claude
    model: claude-opus-5
    yolo: true
"""


class TestInitForcePreservesConfig:
    def _run_init(self, project_root: Path) -> None:
        from typer.testing import CliRunner

        from crossby.cli.main import app

        result = CliRunner().invoke(
            app, ["init", "--force", "--non-interactive", "--path", str(project_root)]
        )
        assert result.exit_code == 0, result.output

    def test_models_and_profiles_survive(self, tmp_path: Path) -> None:
        import yaml

        target = tmp_path / ".crossby.yml"
        target.write_text(_HAND_WRITTEN_CROSSBY_YML, encoding="utf-8")

        self._run_init(tmp_path)

        written = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert written["models"]["claude"]["very_complex"] == "claude-opus-5"
        assert written["profiles"]["ccyolo"]["yolo"] is True

    def test_backup_is_written(self, tmp_path: Path) -> None:
        target = tmp_path / ".crossby.yml"
        target.write_text(_HAND_WRITTEN_CROSSBY_YML, encoding="utf-8")

        self._run_init(tmp_path)

        backup = tmp_path / ".crossby.yml.bak"
        assert backup.is_file()
        assert backup.read_text(encoding="utf-8") == _HAND_WRITTEN_CROSSBY_YML

    def test_result_still_loads(self, tmp_path: Path) -> None:
        from crossby.config.loader import parse_config_file

        target = tmp_path / ".crossby.yml"
        target.write_text(_HAND_WRITTEN_CROSSBY_YML, encoding="utf-8")

        self._run_init(tmp_path)

        config = parse_config_file(target)
        assert config.models["claude"].very_complex == "claude-opus-5"
        assert config.profiles["ccyolo"].tool == AIToolID.CLAUDE

    def test_previous_config_is_restored_when_the_new_one_is_invalid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The safety net: a config we wrote but can't load must not replace theirs."""
        import crossby.config.loader as loader

        target = tmp_path / ".crossby.yml"
        target.write_text(_HAND_WRITTEN_CROSSBY_YML, encoding="utf-8")

        def _always_fails(_path: Path) -> None:
            raise ValueError("simulated loader rejection")

        monkeypatch.setattr(loader, "parse_config_file", _always_fails)

        from typer.testing import CliRunner

        from crossby.cli.main import app

        result = CliRunner().invoke(
            app, ["init", "--force", "--non-interactive", "--path", str(tmp_path)]
        )

        assert result.exit_code == 1
        assert target.read_text(encoding="utf-8") == _HAND_WRITTEN_CROSSBY_YML
        # The restored file is the config again, so the backup is cleaned up.
        assert not (tmp_path / ".crossby.yml.bak").exists()

    def test_unreadable_config_is_still_backed_up(self, tmp_path: Path) -> None:
        target = tmp_path / ".crossby.yml"
        target.write_text("{{ not: valid: yaml\n", encoding="utf-8")

        self._run_init(tmp_path)

        assert (tmp_path / ".crossby.yml.bak").read_text(
            encoding="utf-8"
        ) == "{{ not: valid: yaml\n"


# ---------------------------------------------------------------------------
# 5. Revocable sync never deletes an entry crossby didn't write
# ---------------------------------------------------------------------------


def _registry(*writers: object) -> SyncRegistry:
    reg = SyncRegistry()
    for w in writers:
        reg.register(w)  # type: ignore[arg-type]
    return reg


class TestNeverDeletesUnowned:
    """An entry absent from ``.crossby/owned.json`` survives a sync unconditionally.

    The ledger starts empty (nothing owned), so a first sync may only *add* — it
    can never revoke a hand-authored hook, permission, or MCP server, even one
    absent from the current ``SyncData``.
    """

    def test_hand_written_hook_survives(self, tmp_path: Path) -> None:
        settings = tmp_path / ".claude" / "settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [
                            {"matcher": "Edit", "hooks": [{"type": "command", "command": "human"}]}
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        run_sync(
            SyncData(hooks=[HookEntry(event="pre_tool_use", command="crossby", tools=["Write"])]),
            tmp_path,
            tool_id=AIToolID.CLAUDE,
            registry=_registry(ClaudeHooksWriter()),
        )
        commands = {
            inner["command"]
            for entry in json.loads(settings.read_text())["hooks"]["PreToolUse"]
            for inner in entry["hooks"]
        }
        assert "human" in commands

    def test_hand_written_permission_survives(self, tmp_path: Path) -> None:
        settings = tmp_path / ".claude" / "settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"permissions": {"allow": ["Bash(human:*)"]}}))
        run_sync(
            SyncData(allowed_commands=["crossby:*"]),
            tmp_path,
            tool_id=AIToolID.CLAUDE,
            registry=_registry(ClaudePermissionWriter()),
        )
        allow = json.loads(settings.read_text())["permissions"]["allow"]
        assert "Bash(human:*)" in allow

    def test_hand_written_mcp_server_survives_a_same_named_disable(self, tmp_path: Path) -> None:
        mcp = tmp_path / ".mcp.json"
        mcp.write_text(
            json.dumps({"mcpServers": {"shared": {"command": "hand-written"}}}), encoding="utf-8"
        )
        # Source disables a same-named server; crossby never wrote "shared", so
        # the ledger doesn't own it → it must survive.
        run_sync(
            SyncData(mcp_servers={"shared": MCPServerConfig(command="npx", enabled=False)}),
            tmp_path,
            tool_id=AIToolID.CLAUDE,
            registry=_registry(ClaudeMCPWriter()),
        )
        data = json.loads(mcp.read_text())
        assert data["mcpServers"]["shared"]["command"] == "hand-written"
