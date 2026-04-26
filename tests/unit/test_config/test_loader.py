"""Tests for .crossby.yml config loader."""

import warnings

import pytest
import yaml

from crossby.config.loader import (
    ConfigError,
    ensure_yaml_mapping,
    find_config_file,
    load_config,
)
from crossby.models.ai import AIToolID
from crossby.models.config import CrossbyConfig


class TestEnsureYamlMapping:
    def test_dict(self):
        assert ensure_yaml_mapping({"a": 1}) == {"a": 1}

    def test_none(self):
        assert ensure_yaml_mapping(None) is None

    def test_list_raises(self):
        with pytest.raises(ConfigError):
            ensure_yaml_mapping([1, 2])

    def test_scalar_raises(self):
        with pytest.raises(ConfigError):
            ensure_yaml_mapping("hello")


class TestFindConfigFile:
    def test_not_found(self, tmp_path):
        assert find_config_file(tmp_path) is None

    def test_found(self, tmp_path):
        cfg = tmp_path / ".crossby.yml"
        cfg.write_text("version: 1\n")
        assert find_config_file(tmp_path) == cfg

    def test_walk_up(self, tmp_path):
        cfg = tmp_path / ".crossby.yml"
        cfg.write_text("version: 1\n")
        child = tmp_path / "sub" / "deep"
        child.mkdir(parents=True)
        assert find_config_file(child) == cfg


class TestLoadConfig:
    def test_defaults_when_no_file(self, tmp_path):
        config = load_config(tmp_path)
        assert isinstance(config, CrossbyConfig)
        assert config.ai.default_tool is None

    def test_empty_file(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("")
        config = load_config(tmp_path)
        assert isinstance(config, CrossbyConfig)

    def test_full_config(self, tmp_path):
        data = {
            "version": 1,
            "ai": {
                "default_tool": "claude",
                "default_model": "claude-sonnet-4.6",
                "effort": "high",
                "yolo": True,
                "commands": {
                    "plan": {"tool": "copilot", "model": "gpt-5"},
                    "review": {"effort": "low"},
                },
            },
            "models": {
                "claude": {"easy": "haiku", "medium": "sonnet"},
            },
            "profiles": {
                "fast": {"tool": "claude", "effort": "low"},
            },
        }
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        config = load_config(tmp_path)
        assert config.ai.default_tool == "claude"
        assert config.ai.default_model == "claude-sonnet-4.6"
        assert config.ai.effort == "high"
        assert config.ai.yolo is True
        assert "plan" in config.ai.commands
        assert config.ai.commands["plan"].tool == "copilot"
        assert config.ai.commands["plan"].model == "gpt-5"
        assert config.ai.commands["review"].effort == "low"
        assert config.models["claude"].easy == "haiku"
        assert config.profiles["fast"].tool == "claude"
        assert config.profiles["fast"].effort == "low"

    def test_invalid_yaml(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text(": invalid: yaml: [")
        with pytest.raises(ConfigError):
            load_config(tmp_path)

    def test_empty_commands(self, tmp_path):
        data = {"version": 1, "ai": {"default_tool": "claude"}}
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        config = load_config(tmp_path)
        assert config.ai.commands == {}

    def test_models_as_list_raises(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("models:\n  - bad\n")
        with pytest.raises(ConfigError):
            load_config(tmp_path)

    def test_commands_as_list_raises(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("ai:\n  commands:\n    - bad\n")
        with pytest.raises(ConfigError):
            load_config(tmp_path)

    def test_command_entry_as_scalar_raises(self, tmp_path):
        """``ai.commands.plan: 123`` must raise — parallel to ``profiles.<name>``."""
        (tmp_path / ".crossby.yml").write_text(
            "ai:\n  commands:\n    plan: 123\n"
        )
        with pytest.raises(ConfigError, match=r"'ai\.commands\.plan' must be a mapping"):
            load_config(tmp_path)

    def test_models_entry_as_scalar_raises(self, tmp_path):
        """``models.claude: 123`` must raise — was silently dropped before."""
        (tmp_path / ".crossby.yml").write_text(
            "models:\n  claude: 123\n"
        )
        with pytest.raises(ConfigError, match=r"'models\.claude' must be a mapping"):
            load_config(tmp_path)

    def test_ai_as_list_raises(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("ai:\n  - bad\n")
        with pytest.raises(ConfigError):
            load_config(tmp_path)

    def test_ai_as_empty_string_raises(self, tmp_path):
        """Falsy scalars like '' must not be silently coerced to {}."""
        (tmp_path / ".crossby.yml").write_text('ai: ""\n')
        with pytest.raises(ConfigError, match="'ai' must be a mapping"):
            load_config(tmp_path)

    def test_models_as_zero_raises(self, tmp_path):
        """Falsy scalars like 0 must not be silently coerced to {}."""
        (tmp_path / ".crossby.yml").write_text("models: 0\n")
        with pytest.raises(ConfigError, match="'models' must be a mapping"):
            load_config(tmp_path)


class TestSyncDefaults:
    """Parsing of the ``sync_defaults:`` section."""

    def test_missing_section_yields_defaults(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("version: 1\n")
        config = load_config(tmp_path)
        assert config.sync_defaults.from_tool is None
        assert config.sync_defaults.to is None
        assert config.sync_defaults.concern is None
        assert config.get_sync_from() is None
        assert config.get_sync_to() is None
        assert config.get_sync_concern() is None

    def test_full_section_roundtrip(self, tmp_path):
        data = {
            "version": 1,
            "sync_defaults": {
                "from": "claude",
                "to": "cursor",
                "concern": "rules",
            },
        }
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        config = load_config(tmp_path)
        assert config.sync_defaults.from_tool is AIToolID.CLAUDE
        assert config.sync_defaults.to is AIToolID.CURSOR
        assert config.sync_defaults.concern == "rules"
        assert config.get_sync_from() is AIToolID.CLAUDE
        assert config.get_sync_to() is AIToolID.CURSOR
        assert config.get_sync_concern() == "rules"

    def test_invalid_tool_id_raises(self, tmp_path):
        data = {"version": 1, "sync_defaults": {"from": "nosuchtool"}}
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        with pytest.raises(ConfigError, match="Invalid 'sync_defaults'"):
            load_config(tmp_path)

    def test_list_raises(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("sync_defaults:\n  - bad\n")
        with pytest.raises(ConfigError, match="'sync_defaults' must be a mapping"):
            load_config(tmp_path)


class TestHandoffDefaults:
    """Parsing of the ``handoff_defaults:`` section."""

    def test_missing_section_yields_defaults(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("version: 1\n")
        config = load_config(tmp_path)
        assert config.handoff_defaults.from_tool is None
        assert config.handoff_defaults.to is None
        assert config.handoff_defaults.prompt_preset is None
        assert config.handoff_defaults.token_budget is None
        assert config.get_handoff_from() is None
        assert config.get_handoff_to() is None
        assert config.get_handoff_preset() is None
        assert config.get_handoff_token_budget() is None

    def test_full_section_roundtrip(self, tmp_path):
        data = {
            "version": 1,
            "handoff_defaults": {
                "from": "claude",
                "to": "codex",
                "prompt_preset": "cc-compact",
                "token_budget": 16000,
            },
        }
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        config = load_config(tmp_path)
        assert config.handoff_defaults.from_tool is AIToolID.CLAUDE
        assert config.handoff_defaults.to is AIToolID.CODEX
        assert config.handoff_defaults.prompt_preset == "cc-compact"
        assert config.handoff_defaults.token_budget == 16000
        assert config.get_handoff_preset() == "cc-compact"
        assert config.get_handoff_token_budget() == 16000

    def test_unknown_prompt_preset_raises(self, tmp_path):
        data = {
            "version": 1,
            "handoff_defaults": {"prompt_preset": "not-a-preset"},
        }
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        with pytest.raises(ConfigError, match="prompt_preset"):
            load_config(tmp_path)

    def test_invalid_tool_id_raises(self, tmp_path):
        data = {"version": 1, "handoff_defaults": {"to": "nosuchtool"}}
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        with pytest.raises(ConfigError, match="Invalid 'handoff_defaults'"):
            load_config(tmp_path)

    def test_list_raises(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("handoff_defaults:\n  - bad\n")
        with pytest.raises(ConfigError, match="'handoff_defaults' must be a mapping"):
            load_config(tmp_path)

    def test_zero_token_budget_raises(self, tmp_path):
        """``token_budget: 0`` must be rejected at config-load time."""
        (tmp_path / ".crossby.yml").write_text(
            "handoff_defaults:\n  token_budget: 0\n"
        )
        with pytest.raises(ConfigError, match="Invalid 'handoff_defaults'"):
            load_config(tmp_path)

    def test_negative_token_budget_raises(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text(
            "handoff_defaults:\n  token_budget: -100\n"
        )
        with pytest.raises(ConfigError, match="Invalid 'handoff_defaults'"):
            load_config(tmp_path)


class TestDeprecatedSyncKeyUnchanged:
    """Regression guard: the new ``sync_defaults:`` must not reclaim ``sync:``."""

    def test_top_level_sync_still_emits_deprecation(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text(
            "sync:\n  from_tool: claude\n"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_config(tmp_path)
        deprecation_msgs = [
            str(w.message)
            for w in caught
            if issubclass(w.category, DeprecationWarning)
        ]
        assert any("'sync' section" in m for m in deprecation_msgs), deprecation_msgs

    def test_sync_defaults_does_not_trigger_deprecation(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text(
            "sync_defaults:\n  from: claude\n"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_config(tmp_path)
        deprecation_msgs = [
            str(w.message)
            for w in caught
            if issubclass(w.category, DeprecationWarning)
        ]
        assert not deprecation_msgs, deprecation_msgs
