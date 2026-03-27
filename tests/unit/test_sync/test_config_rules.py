"""Tests for rules config parsing in config loader."""

import pytest
import yaml

from crossby.config.loader import ConfigError, load_config


class TestRulesConfigParsing:
    def test_no_rules_section(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("version: 1\n")
        config = load_config(tmp_path)
        assert config.rules.enabled is False

    def test_default_rules(self, tmp_path):
        data = {"version": 1, "rules": {}}
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        config = load_config(tmp_path)
        assert config.rules.enabled is True
        assert config.rules.source == "AGENTS.md"
        assert config.rules.strategy == "symlink"
        assert config.rules.gitignore is True
        assert config.rules.targets.claude is True

    def test_custom_source(self, tmp_path):
        data = {"version": 1, "rules": {"source": "CLAUDE.md"}}
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        config = load_config(tmp_path)
        assert config.rules.source == "CLAUDE.md"

    def test_copy_strategy(self, tmp_path):
        data = {"version": 1, "rules": {"strategy": "copy"}}
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        config = load_config(tmp_path)
        assert config.rules.strategy == "copy"

    def test_invalid_strategy(self, tmp_path):
        data = {"version": 1, "rules": {"strategy": "hardlink"}}
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        with pytest.raises(ConfigError, match="strategy"):
            load_config(tmp_path)

    def test_disable_specific_target(self, tmp_path):
        data = {"version": 1, "rules": {"targets": {"cursor": False}}}
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        config = load_config(tmp_path)
        assert config.rules.targets.cursor is False
        assert config.rules.targets.claude is True

    def test_rules_as_list_raises(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("rules:\n  - bad\n")
        with pytest.raises(ConfigError, match="'rules' must be a mapping"):
            load_config(tmp_path)

    def test_targets_as_list_raises(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text(
            "rules:\n  targets:\n    - bad\n"
        )
        with pytest.raises(ConfigError, match="'rules.targets' must be a mapping"):
            load_config(tmp_path)

    def test_gitignore_false(self, tmp_path):
        data = {"version": 1, "rules": {"gitignore": False}}
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        config = load_config(tmp_path)
        assert config.rules.gitignore is False

    def test_gitignore_non_bool_raises(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("rules:\n  gitignore: yes_please\n")
        with pytest.raises(ConfigError, match="'rules.gitignore' must be a boolean"):
            load_config(tmp_path)

    def test_unknown_target_key_raises(self, tmp_path):
        """Typos like 'copliot' must surface as an error, not be silently ignored."""
        (tmp_path / ".crossby.yml").write_text(
            "rules:\n  targets:\n    copliot: false\n"
        )
        with pytest.raises(ConfigError, match="Unknown 'rules.targets' keys"):
            load_config(tmp_path)

    def test_non_bool_target_value_raises(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text(
            "rules:\n  targets:\n    claude: yes_please\n"
        )
        with pytest.raises(ConfigError, match="'rules.targets.claude' must be a boolean"):
            load_config(tmp_path)


class TestAgentsConfigParsing:
    def test_unknown_agents_target_key_raises(self, tmp_path):
        """Typos in agents.targets must surface as an error."""
        (tmp_path / ".crossby.yml").write_text(
            "agents:\n  targets:\n    copliot: true\n"
        )
        with pytest.raises(ConfigError, match="Unknown 'agents.targets' keys"):
            load_config(tmp_path)

    def test_aitoolid_without_agent_path_raises(self, tmp_path):
        """AIToolID values not backed by an agent sync target (e.g. vscode) should be rejected."""
        data = {"version": 1, "agents": {"targets": {"vscode": True}}}
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        with pytest.raises(ConfigError, match="Unknown 'agents.targets' keys"):
            load_config(tmp_path)

    def test_valid_agents_target_key_accepted(self, tmp_path):
        data = {"version": 1, "agents": {"targets": {"claude": True, "cursor": False}}}
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        config = load_config(tmp_path)
        assert config.agents.targets == {"claude": True, "cursor": False}

    def test_agents_strategy_literal_validated(self, tmp_path):
        data = {"version": 1, "agents": {"strategy": "hardlink"}}
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        with pytest.raises(ConfigError, match="strategy"):
            load_config(tmp_path)
