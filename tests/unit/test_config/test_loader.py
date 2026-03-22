"""Tests for .crossby.yml config loader."""

import pytest
import yaml

from crossby.config.loader import (
    ConfigError,
    ensure_yaml_mapping,
    find_config_file,
    load_config,
)
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
            "permissions": {"allowed_commands": ["myapp:*"]},
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
        assert config.permissions.allowed_commands == ["myapp:*"]

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
