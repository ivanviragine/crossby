"""Tests that deprecated config sections are silently ignored by the loader."""

import warnings

import yaml

from crossby.config.loader import load_config


class TestDeprecatedSectionsIgnored:
    def test_rules_section_ignored(self, tmp_path):
        data = {"version": 1, "rules": {"source": "AGENTS.md", "strategy": "symlink"}}
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = load_config(tmp_path)
        assert config.version == 1
        assert not hasattr(config, "rules")
        assert any("rules" in str(x.message) and "deprecated" in str(x.message) for x in w)

    def test_agents_section_ignored(self, tmp_path):
        data = {"version": 1, "agents": {"targets": {"claude": True}}}
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = load_config(tmp_path)
        assert not hasattr(config, "agents")
        assert any("agents" in str(x.message) for x in w)

    def test_mcp_servers_section_ignored(self, tmp_path):
        data = {"version": 1, "mcp_servers": {"ctx7": {"command": "npx"}}}
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = load_config(tmp_path)
        assert not hasattr(config, "mcp_servers")
        assert any("mcp_servers" in str(x.message) for x in w)

    def test_permissions_section_ignored(self, tmp_path):
        data = {"version": 1, "permissions": {"allowed_commands": ["myapp:*"]}}
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_config(tmp_path)
        assert any("permissions" in str(x.message) for x in w)

    def test_no_warning_when_section_absent(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("version: 1\n")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_config(tmp_path)
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 0

    def test_null_section_no_warning(self, tmp_path):
        """Explicit `rules: null` should not warn."""
        (tmp_path / ".crossby.yml").write_text("version: 1\nrules: null\n")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_config(tmp_path)
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 0
