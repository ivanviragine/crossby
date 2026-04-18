"""Unit tests for crossby.config.allowlist_util.configure_json_allowlist."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from crossby.config.allowlist_util import configure_json_allowlist


def _bash(p: str) -> str:
    return f"Bash({p})"


def _shell(p: str) -> str:
    return f"Shell({p})"


_WRITE_TARGET = "crossby.sync.json_utils.write_json_file"


class TestEmptyPatternsShortCircuit:
    def test_no_op_when_patterns_empty(self, tmp_path: Path) -> None:
        config = tmp_path / "settings.json"
        configure_json_allowlist(config, [], pattern_converter=_bash)
        assert not config.exists()

    def test_no_op_does_not_write_existing_file(self, tmp_path: Path) -> None:
        config = tmp_path / "settings.json"
        config.write_text('{"other": 1}\n', encoding="utf-8")
        with patch(_WRITE_TARGET) as mock_write:
            configure_json_allowlist(config, [], pattern_converter=_bash)
            mock_write.assert_not_called()


class TestFreshFileCreation:
    def test_creates_file_and_parent_dirs(self, tmp_path: Path) -> None:
        config = tmp_path / "sub" / "dir" / "settings.json"
        configure_json_allowlist(config, ["myapp:*"], pattern_converter=_bash)
        assert config.is_file()
        data = json.loads(config.read_text(encoding="utf-8"))
        assert data == {"permissions": {"allow": ["Bash(myapp:*)"]}}

    def test_custom_converter_applied(self, tmp_path: Path) -> None:
        config = tmp_path / "cli.json"
        configure_json_allowlist(config, ["myapp:*"], pattern_converter=_shell)
        data = json.loads(config.read_text(encoding="utf-8"))
        assert data["permissions"]["allow"] == ["Shell(myapp:*)"]

    def test_multiple_patterns_all_added(self, tmp_path: Path) -> None:
        config = tmp_path / "settings.json"
        configure_json_allowlist(
            config, ["myapp:*", "./scripts/run.sh:*"], pattern_converter=_bash
        )
        data = json.loads(config.read_text(encoding="utf-8"))
        assert "Bash(myapp:*)" in data["permissions"]["allow"]
        assert "Bash(./scripts/run.sh:*)" in data["permissions"]["allow"]


class TestIdempotency:
    def test_no_duplicate_on_second_call(self, tmp_path: Path) -> None:
        config = tmp_path / "settings.json"
        configure_json_allowlist(config, ["myapp:*"], pattern_converter=_bash)
        configure_json_allowlist(config, ["myapp:*"], pattern_converter=_bash)
        data = json.loads(config.read_text(encoding="utf-8"))
        assert data["permissions"]["allow"].count("Bash(myapp:*)") == 1

    def test_preserves_existing_allow_entries(self, tmp_path: Path) -> None:
        config = tmp_path / "settings.json"
        existing = {"permissions": {"allow": ["Bash(git *)"], "deny": ["Bash(rm *)"]}}
        config.write_text(json.dumps(existing) + "\n", encoding="utf-8")
        configure_json_allowlist(config, ["myapp:*"], pattern_converter=_bash)
        data = json.loads(config.read_text(encoding="utf-8"))
        assert "Bash(git *)" in data["permissions"]["allow"]
        assert "Bash(myapp:*)" in data["permissions"]["allow"]
        assert data["permissions"]["deny"] == ["Bash(rm *)"]

    def test_preserves_other_top_level_keys(self, tmp_path: Path) -> None:
        config = tmp_path / "settings.json"
        existing = {"theme": "dark", "permissions": {"allow": []}}
        config.write_text(json.dumps(existing) + "\n", encoding="utf-8")
        configure_json_allowlist(config, ["myapp:*"], pattern_converter=_bash)
        data = json.loads(config.read_text(encoding="utf-8"))
        assert data["theme"] == "dark"

    def test_no_write_when_already_present(self, tmp_path: Path) -> None:
        config = tmp_path / "settings.json"
        existing = {"permissions": {"allow": ["Bash(myapp:*)"]}}
        config.write_text(json.dumps(existing) + "\n", encoding="utf-8")
        with patch(_WRITE_TARGET) as mock_write:
            configure_json_allowlist(config, ["myapp:*"], pattern_converter=_bash)
            mock_write.assert_not_called()


class TestCorruptedJsonRecovery:
    def test_starts_fresh_on_invalid_json(self, tmp_path: Path) -> None:
        config = tmp_path / "settings.json"
        config.write_text("{not valid json!!", encoding="utf-8")
        configure_json_allowlist(config, ["myapp:*"], pattern_converter=_bash)
        data = json.loads(config.read_text(encoding="utf-8"))
        assert "Bash(myapp:*)" in data["permissions"]["allow"]

    def test_starts_fresh_when_root_is_list(self, tmp_path: Path) -> None:
        config = tmp_path / "settings.json"
        config.write_text("[1, 2, 3]\n", encoding="utf-8")
        configure_json_allowlist(config, ["myapp:*"], pattern_converter=_bash)
        data = json.loads(config.read_text(encoding="utf-8"))
        assert "Bash(myapp:*)" in data["permissions"]["allow"]


class TestRepairBehavior:
    def test_non_dict_permissions_repaired(self, tmp_path: Path) -> None:
        config = tmp_path / "settings.json"
        existing = {"permissions": "invalid", "other": "kept"}
        config.write_text(json.dumps(existing) + "\n", encoding="utf-8")
        configure_json_allowlist(config, ["myapp:*"], pattern_converter=_bash)
        data = json.loads(config.read_text(encoding="utf-8"))
        assert isinstance(data["permissions"], dict)
        assert "Bash(myapp:*)" in data["permissions"]["allow"]
        assert data["other"] == "kept"

    def test_non_list_allow_repaired(self, tmp_path: Path) -> None:
        config = tmp_path / "settings.json"
        existing = {"permissions": {"allow": "not-a-list"}}
        config.write_text(json.dumps(existing) + "\n", encoding="utf-8")
        configure_json_allowlist(config, ["myapp:*"], pattern_converter=_bash)
        data = json.loads(config.read_text(encoding="utf-8"))
        assert isinstance(data["permissions"]["allow"], list)
        assert "Bash(myapp:*)" in data["permissions"]["allow"]
