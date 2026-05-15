"""Tests for ClaudePermissionWriter, CursorPermissionWriter, and GeminiPermissionWriter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossby.models.ai import AIToolID
from crossby.sync import run_sync
from crossby.sync.base import SyncConcern, SyncData
from crossby.sync.permissions import (
    ClaudePermissionWriter,
    CursorPermissionWriter,
    GeminiPermissionWriter,
    canonical_to_claude,
    canonical_to_cursor,
)


def _make_data(patterns: list[str] | None = None) -> SyncData:
    """Build a SyncData with the given permission patterns."""
    return SyncData(allowed_commands=patterns or [])


@pytest.fixture(autouse=True)
def _patch_cursor_global(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the Cursor global config path to a temp directory."""
    fake = tmp_path / ".cursor" / "cli-config.json"
    monkeypatch.setattr("crossby.sync.permissions._GLOBAL_CURSOR_CONFIG_PATH", fake)


def _cursor_global(tmp_path: Path) -> Path:
    return tmp_path / ".cursor" / "cli-config.json"


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------


class TestClaudePermissionWriterCheck:
    def test_returns_false_when_file_missing(self, tmp_path: Path) -> None:
        assert ClaudePermissionWriter.check(tmp_path, ["myapp:*"]) is False

    def test_returns_true_when_present(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"permissions": {"allow": ["Bash(myapp:*)"]}}))
        assert ClaudePermissionWriter.check(tmp_path, ["myapp:*"]) is True

    def test_returns_false_when_pattern_absent(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"permissions": {"allow": ["Bash(other:*)"]}}))
        assert ClaudePermissionWriter.check(tmp_path, ["myapp:*"]) is False

    def test_returns_false_for_partial_match(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"permissions": {"allow": ["Bash(myapp:*)"]}}))
        assert ClaudePermissionWriter.check(tmp_path, ["myapp:*", "other:*"]) is False

    def test_returns_false_for_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text("{bad json!!")
        assert ClaudePermissionWriter.check(tmp_path, ["myapp:*"]) is False

    def test_returns_false_for_non_dict_root(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps(["not", "a", "dict"]))
        assert ClaudePermissionWriter.check(tmp_path, ["myapp:*"]) is False

    def test_returns_false_when_permissions_not_dict(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"permissions": "invalid"}))
        assert ClaudePermissionWriter.check(tmp_path, ["myapp:*"]) is False

    def test_returns_false_when_allow_not_list(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"permissions": {"allow": "not-a-list"}}))
        assert ClaudePermissionWriter.check(tmp_path, ["myapp:*"]) is False


class TestClaudePermissionWriterWrite:
    def test_creates_from_scratch(self, tmp_path: Path) -> None:
        ClaudePermissionWriter.write(tmp_path, ["myapp:*"])
        data = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        assert data == {"permissions": {"allow": ["Bash(myapp:*)"]}}

    def test_idempotent(self, tmp_path: Path) -> None:
        ClaudePermissionWriter.write(tmp_path, ["myapp:*"])
        ClaudePermissionWriter.write(tmp_path, ["myapp:*"])
        data = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        assert data["permissions"]["allow"].count("Bash(myapp:*)") == 1

    def test_non_destructive_merge(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"permissions": {"allow": ["Bash(git *)"]}, "theme": "dark"}))
        ClaudePermissionWriter.write(tmp_path, ["myapp:*"])
        data = json.loads(path.read_text())
        assert "Bash(myapp:*)" in data["permissions"]["allow"]
        assert "Bash(git *)" in data["permissions"]["allow"]
        assert data["theme"] == "dark"

    def test_multiple_patterns(self, tmp_path: Path) -> None:
        ClaudePermissionWriter.write(tmp_path, ["myapp:*", "./scripts/run.sh:*"])
        data = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        assert "Bash(myapp:*)" in data["permissions"]["allow"]
        assert "Bash(./scripts/run.sh:*)" in data["permissions"]["allow"]

    def test_write_refuses_to_overwrite_malformed_json(self, tmp_path: Path) -> None:
        """write() refuses to clobber a malformed file — returns an error tuple.

        Matches hooks/MCP behavior; the safer policy when the user's file is
        unparseable is to surface the error rather than start fresh.
        """
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        original = "{bad json!!"
        path.write_text(original)
        action, error = ClaudePermissionWriter.write(tmp_path, ["myapp:*"])
        assert action == "error"
        assert error is not None and "invalid JSON" in error
        # File is untouched.
        assert path.read_text() == original

    def test_write_refuses_non_dict_root(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps(["list", "value"]))
        original = path.read_text()
        action, error = ClaudePermissionWriter.write(tmp_path, ["myapp:*"])
        assert action == "error"
        assert error is not None
        assert path.read_text() == original

    def test_update_adds_new_pattern_preserving_old(self, tmp_path: Path) -> None:
        """Updating with a new pattern preserves old ones."""
        ClaudePermissionWriter.write(tmp_path, ["first:*"])
        ClaudePermissionWriter.write(tmp_path, ["second:*"])
        data = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        assert "Bash(first:*)" in data["permissions"]["allow"]
        assert "Bash(second:*)" in data["permissions"]["allow"]


class TestClaudePermissionWriterSync:
    def test_skips_when_no_patterns(self, tmp_path: Path) -> None:
        data = SyncData(allowed_commands=[])
        writer = ClaudePermissionWriter()
        result = writer.sync(data, tmp_path)
        assert result.action == "skipped"
        assert result.message == "no allowed_commands detected"

    def test_creates_file(self, tmp_path: Path) -> None:
        data = SyncData(allowed_commands=["myapp:*"])
        writer = ClaudePermissionWriter()
        result = writer.sync(data, tmp_path)
        assert result.action == "created"
        assert (tmp_path / ".claude" / "settings.json").exists()

    def test_updated_when_file_exists(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"permissions": {"allow": ["Bash(git *)"]}}))
        data = SyncData(allowed_commands=["myapp:*"])
        writer = ClaudePermissionWriter()
        result = writer.sync(data, tmp_path)
        assert result.action == "updated"

    def test_skips_when_already_configured(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"permissions": {"allow": ["Bash(myapp:*)"]}}))
        data = SyncData(allowed_commands=["myapp:*"])
        writer = ClaudePermissionWriter()
        result = writer.sync(data, tmp_path)
        assert result.action == "skipped"
        assert result.message == "already configured"

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        data = SyncData(allowed_commands=["myapp:*"])
        writer = ClaudePermissionWriter()
        result = writer.sync(data, tmp_path, dry_run=True)
        assert result.action == "created"
        assert not (tmp_path / ".claude" / "settings.json").exists()

    def test_malformed_json_returns_error_and_preserves_file(self, tmp_path: Path) -> None:
        """sync() surfaces a malformed file as an error row without writing.

        This is the same policy hooks/MCP use; permissions used to silently
        overwrite the broken file with a fresh ``{"permissions": {...}}``
        skeleton, which destroyed the user's content.
        """
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        original = "{this is broken!!"
        path.write_text(original, encoding="utf-8")
        data = SyncData(allowed_commands=["myapp:*"])
        writer = ClaudePermissionWriter()
        result = writer.sync(data, tmp_path)
        assert result.action == "error"
        assert "invalid JSON" in (result.message or "")
        # File untouched.
        assert path.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# Cursor
# ---------------------------------------------------------------------------


class TestCursorPermissionWriterCheck:
    def test_returns_true_vacuously_for_empty_patterns(self, tmp_path: Path) -> None:
        assert CursorPermissionWriter.check(tmp_path) is True

    def test_returns_false_when_file_missing(self, tmp_path: Path) -> None:
        assert CursorPermissionWriter.check(tmp_path, ["myapp:*"]) is False

    def test_per_project_check(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        path = project / ".cursor" / "cli.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"permissions": {"allow": ["Shell(myapp:*)"]}}))
        assert CursorPermissionWriter.check(project, ["myapp:*"]) is True

    def test_global_check(self, tmp_path: Path) -> None:
        path = _cursor_global(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"permissions": {"allow": ["Shell(myapp:*)"]}}))
        assert CursorPermissionWriter.check(None, ["myapp:*"]) is True


class TestCursorPermissionWriterWrite:
    def test_creates_per_project(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        CursorPermissionWriter.write(project, ["myapp:*"])
        data = json.loads((project / ".cursor" / "cli.json").read_text())
        assert data == {"permissions": {"allow": ["Shell(myapp:*)"]}}

    def test_creates_global(self, tmp_path: Path) -> None:
        CursorPermissionWriter.write(None, ["myapp:*"])
        data = json.loads(_cursor_global(tmp_path).read_text())
        assert data == {"permissions": {"allow": ["Shell(myapp:*)"]}}

    def test_noop_for_empty_patterns(self, tmp_path: Path) -> None:
        CursorPermissionWriter.write(tmp_path)
        assert not (tmp_path / ".cursor" / "cli.json").exists()

    def test_idempotent(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        CursorPermissionWriter.write(project, ["myapp:*"])
        CursorPermissionWriter.write(project, ["myapp:*"])
        data = json.loads((project / ".cursor" / "cli.json").read_text())
        assert data["permissions"]["allow"].count("Shell(myapp:*)") == 1

    def test_non_destructive_merge(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        path = project / ".cursor" / "cli.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"permissions": {"allow": ["Shell(ls)"]}, "version": 1}))
        CursorPermissionWriter.write(project, ["myapp:*"])
        data = json.loads(path.read_text())
        assert "Shell(myapp:*)" in data["permissions"]["allow"]
        assert "Shell(ls)" in data["permissions"]["allow"]
        assert data["version"] == 1

    def test_per_project_does_not_write_global(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        CursorPermissionWriter.write(project, ["myapp:*"])
        assert not _cursor_global(tmp_path).exists()


class TestCursorPermissionWriterSync:
    def test_skips_when_no_patterns(self, tmp_path: Path) -> None:
        data = SyncData(allowed_commands=[])
        writer = CursorPermissionWriter()
        result = writer.sync(data, tmp_path)
        assert result.action == "skipped"
        assert result.message == "no allowed_commands detected"

    def test_creates_project_config(self, tmp_path: Path) -> None:
        data = SyncData(allowed_commands=["myapp:*"])
        writer = CursorPermissionWriter(scope="project")
        result = writer.sync(data, tmp_path)
        assert result.action == "created"
        assert (tmp_path / ".cursor" / "cli.json").exists()

    def test_skips_when_already_configured(self, tmp_path: Path) -> None:
        path = tmp_path / ".cursor" / "cli.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"permissions": {"allow": ["Shell(myapp:*)"]}}))
        data = SyncData(allowed_commands=["myapp:*"])
        writer = CursorPermissionWriter(scope="project")
        result = writer.sync(data, tmp_path)
        assert result.action == "skipped"

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        data = SyncData(allowed_commands=["myapp:*"])
        writer = CursorPermissionWriter(scope="project")
        result = writer.sync(data, tmp_path, dry_run=True)
        assert result.action == "created"
        assert not (tmp_path / ".cursor" / "cli.json").exists()

    def test_global_scope_uses_global_config(self, tmp_path: Path) -> None:
        data = SyncData(allowed_commands=["myapp:*"])
        writer = CursorPermissionWriter(scope="global")
        result = writer.sync(data, tmp_path)
        assert result.action == "created"
        assert _cursor_global(tmp_path).exists()
        assert not (tmp_path / ".cursor" / "cli.json").exists()


# ---------------------------------------------------------------------------
# Pattern translators
# ---------------------------------------------------------------------------


class TestClaudePermissionSyncIdempotency:
    """Verify sync() is idempotent: first call writes, second is a no-op."""

    def test_sync_then_sync_is_skipped(self, tmp_path: Path) -> None:
        data = _make_data(["myapp:*"])
        writer = ClaudePermissionWriter()
        r1 = writer.sync(data, tmp_path)
        assert r1.action == "created"
        r2 = writer.sync(data, tmp_path)
        assert r2.action == "skipped"
        assert r2.message == "already configured"

    def test_sync_with_additional_patterns(self, tmp_path: Path) -> None:
        writer = ClaudePermissionWriter()
        r1 = writer.sync(_make_data(["first:*"]), tmp_path)
        assert r1.action == "created"
        r2 = writer.sync(_make_data(["first:*", "second:*"]), tmp_path)
        assert r2.action == "updated"
        r3 = writer.sync(_make_data(["first:*", "second:*"]), tmp_path)
        assert r3.action == "skipped"


class TestPermissionsRunSyncIntegration:
    """Test permissions through the run_sync() orchestrator."""

    def test_run_sync_creates_both_claude_and_cursor(self, tmp_path: Path) -> None:
        data = _make_data(["myapp:*"])
        results = run_sync(
            data,
            tmp_path,
            concern=SyncConcern.PERMISSIONS,
            installed_tools=[AIToolID.CLAUDE, AIToolID.CURSOR],
        )
        actions = {r.tool_id: r.action for r in results}
        assert actions[AIToolID.CLAUDE] == "created"
        assert actions[AIToolID.CURSOR] == "created"
        assert (tmp_path / ".claude" / "settings.json").exists()
        assert (tmp_path / ".cursor" / "cli.json").exists()

    def test_run_sync_skips_when_already_configured(self, tmp_path: Path) -> None:
        data = _make_data(["myapp:*"])
        run_sync(
            data,
            tmp_path,
            concern=SyncConcern.PERMISSIONS,
            installed_tools=[AIToolID.CLAUDE],
        )
        results = run_sync(
            data,
            tmp_path,
            concern=SyncConcern.PERMISSIONS,
            installed_tools=[AIToolID.CLAUDE],
        )
        assert all(r.action == "skipped" for r in results)


# ---------------------------------------------------------------------------
# Pattern translators
# ---------------------------------------------------------------------------


class TestPatternTranslators:
    def test_canonical_to_claude(self) -> None:
        assert canonical_to_claude("myapp:*") == "Bash(myapp:*)"
        assert canonical_to_claude("./scripts/run.sh") == "Bash(./scripts/run.sh)"

    def test_canonical_to_cursor(self) -> None:
        assert canonical_to_cursor("myapp:*") == "Shell(myapp:*)"
        assert canonical_to_cursor("./scripts/run.sh") == "Shell(./scripts/run.sh)"


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


def _gemini_policy(tmp_path: Path) -> Path:
    return tmp_path / ".gemini" / "policies" / "crossby.toml"


class TestGeminiPermissionWriterCheck:
    def test_returns_false_when_file_missing(self, tmp_path: Path) -> None:
        assert GeminiPermissionWriter.check(tmp_path, ["git:*"]) is False

    def test_returns_true_when_present(self, tmp_path: Path) -> None:
        policy = _gemini_policy(tmp_path)
        policy.parent.mkdir(parents=True)
        policy.write_text('[[rule]]\ntoolName = "run_shell_command"\ncommandPrefix = "git"\n')
        assert GeminiPermissionWriter.check(tmp_path, ["git:*"]) is True

    def test_returns_false_when_pattern_absent(self, tmp_path: Path) -> None:
        policy = _gemini_policy(tmp_path)
        policy.parent.mkdir(parents=True)
        policy.write_text('[[rule]]\ntoolName = "run_shell_command"\ncommandPrefix = "npm"\n')
        assert GeminiPermissionWriter.check(tmp_path, ["git:*"]) is False


class TestGeminiPermissionWriterWrite:
    def test_creates_policy_file(self, tmp_path: Path) -> None:
        GeminiPermissionWriter.write(tmp_path, ["git:*"])
        policy = _gemini_policy(tmp_path)
        assert policy.exists()
        content = policy.read_text()
        assert 'commandPrefix = "git"' in content
        assert 'toolName = "run_shell_command"' in content
        assert 'decision = "allow"' in content
        assert "priority = 100" in content

    def test_multiple_patterns(self, tmp_path: Path) -> None:
        GeminiPermissionWriter.write(tmp_path, ["git:*", "npm:install"])
        content = _gemini_policy(tmp_path).read_text()
        assert 'commandPrefix = "git"' in content
        assert 'commandPrefix = "npm"' in content

    def test_overwrites_previous_content(self, tmp_path: Path) -> None:
        GeminiPermissionWriter.write(tmp_path, ["git:*"])
        GeminiPermissionWriter.write(tmp_path, ["npm:*"])
        content = _gemini_policy(tmp_path).read_text()
        assert 'commandPrefix = "npm"' in content
        assert 'commandPrefix = "git"' not in content

    def test_removes_file_when_empty(self, tmp_path: Path) -> None:
        GeminiPermissionWriter.write(tmp_path, ["git:*"])
        assert _gemini_policy(tmp_path).exists()
        GeminiPermissionWriter.write(tmp_path, [])
        assert not _gemini_policy(tmp_path).exists()

    def test_skips_empty_patterns(self, tmp_path: Path) -> None:
        GeminiPermissionWriter.write(tmp_path, ["git:*", "", "  "])
        content = _gemini_policy(tmp_path).read_text()
        assert content.count("[[rule]]") == 1

    def test_escapes_quotes(self, tmp_path: Path) -> None:
        GeminiPermissionWriter.write(tmp_path, ['"quoted":*'])
        content = _gemini_policy(tmp_path).read_text()
        assert 'commandPrefix = "\\"quoted\\"' in content

    def test_escapes_backslashes(self, tmp_path: Path) -> None:
        GeminiPermissionWriter.write(tmp_path, ["path\\to\\cmd:*"])
        content = _gemini_policy(tmp_path).read_text()
        assert 'commandPrefix = "path\\\\to\\\\cmd"' in content

    def test_pattern_without_args(self, tmp_path: Path) -> None:
        GeminiPermissionWriter.write(tmp_path, ["./scripts/fmt.sh"])
        content = _gemini_policy(tmp_path).read_text()
        assert 'commandPrefix = "./scripts/fmt.sh"' in content

    def test_noop_for_file_that_does_not_exist(self, tmp_path: Path) -> None:
        GeminiPermissionWriter.write(tmp_path, [])
        assert not _gemini_policy(tmp_path).exists()


class TestGeminiPermissionWriterSync:
    def test_skips_when_no_patterns(self, tmp_path: Path) -> None:
        data = SyncData(allowed_commands=[])
        writer = GeminiPermissionWriter()
        result = writer.sync(data, tmp_path)
        assert result.action == "skipped"
        assert result.message == "no allowed_commands detected"

    def test_creates_file(self, tmp_path: Path) -> None:
        data = SyncData(allowed_commands=["git:*"])
        writer = GeminiPermissionWriter()
        result = writer.sync(data, tmp_path)
        assert result.action == "created"
        assert _gemini_policy(tmp_path).exists()

    def test_skips_when_already_configured(self, tmp_path: Path) -> None:
        data = SyncData(allowed_commands=["git:*"])
        writer = GeminiPermissionWriter()
        writer.sync(data, tmp_path)
        result = writer.sync(data, tmp_path)
        assert result.action == "skipped"
        assert result.message == "already configured"

    def test_updated_when_file_exists(self, tmp_path: Path) -> None:
        policy = _gemini_policy(tmp_path)
        policy.parent.mkdir(parents=True)
        policy.write_text('[[rule]]\ntoolName = "run_shell_command"\ncommandPrefix = "npm"\n')
        data = SyncData(allowed_commands=["git:*"])
        writer = GeminiPermissionWriter()
        result = writer.sync(data, tmp_path)
        assert result.action == "updated"

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        data = SyncData(allowed_commands=["git:*"])
        writer = GeminiPermissionWriter()
        result = writer.sync(data, tmp_path, dry_run=True)
        assert result.action == "created"
        assert not _gemini_policy(tmp_path).exists()

    def test_removes_stale_policy(self, tmp_path: Path) -> None:
        """When allowed_commands is empty but policy file exists, remove it."""
        writer = GeminiPermissionWriter()
        writer.sync(SyncData(allowed_commands=["git:*"]), tmp_path)
        assert _gemini_policy(tmp_path).exists()
        result = writer.sync(SyncData(allowed_commands=[]), tmp_path)
        assert result.action == "updated"
        assert result.message == "removed stale policy"
        assert not _gemini_policy(tmp_path).exists()

    def test_dry_run_does_not_remove_stale(self, tmp_path: Path) -> None:
        writer = GeminiPermissionWriter()
        writer.sync(SyncData(allowed_commands=["git:*"]), tmp_path)
        result = writer.sync(SyncData(allowed_commands=[]), tmp_path, dry_run=True)
        assert result.action == "updated"
        assert _gemini_policy(tmp_path).exists()


class TestGeminiPermissionRunSyncIntegration:
    """Test Gemini permissions through run_sync()."""

    def test_run_sync_creates_gemini_policy(self, tmp_path: Path) -> None:
        data = _make_data(["git:*"])
        results = run_sync(
            data,
            tmp_path,
            concern=SyncConcern.PERMISSIONS,
            installed_tools=[AIToolID.GEMINI],
        )
        actions = {r.tool_id: r.action for r in results}
        assert actions[AIToolID.GEMINI] == "created"
        assert _gemini_policy(tmp_path).exists()

    def test_run_sync_creates_all_three(self, tmp_path: Path) -> None:
        data = _make_data(["myapp:*"])
        results = run_sync(
            data,
            tmp_path,
            concern=SyncConcern.PERMISSIONS,
            installed_tools=[AIToolID.CLAUDE, AIToolID.CURSOR, AIToolID.GEMINI],
        )
        actions = {r.tool_id: r.action for r in results}
        assert actions[AIToolID.CLAUDE] == "created"
        assert actions[AIToolID.CURSOR] == "created"
        assert actions[AIToolID.GEMINI] == "created"
