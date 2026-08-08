"""Tests for .crossby.yml config loader."""

import warnings
from pathlib import Path

import pytest
import yaml

from crossby.config.loader import (
    ConfigError,
    ensure_yaml_mapping,
    find_config_entry,
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

    def test_accept_edits_and_auto_parsed(self, tmp_path):
        data = {
            "version": 1,
            "ai": {
                "default_tool": "claude",
                "accept_edits": True,
                "auto": False,
                "commands": {
                    "implement": {"auto": True},
                },
            },
            "profiles": {
                "guarded": {"tool": "claude", "auto": True, "accept_edits": True},
            },
        }
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        config = load_config(tmp_path)
        assert config.ai.accept_edits is True
        assert config.ai.auto is False
        assert config.ai.commands["implement"].auto is True
        assert config.get_auto("implement") is True
        assert config.profiles["guarded"].auto is True
        assert config.profiles["guarded"].accept_edits is True

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
        (tmp_path / ".crossby.yml").write_text("ai:\n  commands:\n    plan: 123\n")
        with pytest.raises(ConfigError, match=r"'ai\.commands\.plan' must be a mapping"):
            load_config(tmp_path)

    def test_models_entry_as_scalar_raises(self, tmp_path):
        """``models.claude: 123`` must raise — was silently dropped before."""
        (tmp_path / ".crossby.yml").write_text("models:\n  claude: 123\n")
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
        (tmp_path / ".crossby.yml").write_text("handoff_defaults:\n  token_budget: 0\n")
        with pytest.raises(ConfigError, match="Invalid 'handoff_defaults'"):
            load_config(tmp_path)

    def test_negative_token_budget_raises(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("handoff_defaults:\n  token_budget: -100\n")
        with pytest.raises(ConfigError, match="Invalid 'handoff_defaults'"):
            load_config(tmp_path)


class TestScenes:
    """Parsing of the ``scenes:`` section and ``extends`` flattening."""

    def test_missing_section_yields_empty(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("version: 1\n")
        config = load_config(tmp_path)
        assert config.scenes == {}
        assert config.get_scene("nope") is None

    def test_null_section_yields_empty(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("scenes:\n")
        assert load_config(tmp_path).scenes == {}

    def test_full_section_parses(self, tmp_path):
        data = {
            "version": 1,
            "profiles": {"ccyolo": {"tool": "claude", "yolo": True}},
            "scenes": {
                "base": {"skills": {"exclude": ["deploy-*"]}},
                "pr-review": {
                    "description": "Review a pull request",
                    "extends": "base",
                    "profile": "ccyolo",
                    "skills": {"include": ["review-*", "knowledge"]},
                    "agents": {"include": ["code-reviewer"]},
                    "mcp": {"include": ["github"]},
                    "hooks": {"include": ["pre_tool_use:*"]},
                    "permissions": {"include": ["git diff:*", "gh pr *"]},
                },
            },
        }
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        config = load_config(tmp_path)
        assert set(config.scenes) == {"base", "pr-review"}
        scene = config.get_scene("pr-review")
        assert scene is not None
        assert scene.description == "Review a pull request"
        assert scene.profile == "ccyolo"
        assert scene.skills is not None
        assert scene.skills.include == ["review-*", "knowledge"]
        assert scene.permissions is not None
        assert scene.permissions.include == ["git diff:*", "gh pr *"]

    def test_extends_replaces_declared_concern(self, tmp_path):
        data = {
            "scenes": {
                "base": {"skills": {"exclude": ["deploy-*"]}},
                "child": {"extends": "base", "skills": {"include": ["review-*"]}},
            },
        }
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        config = load_config(tmp_path)
        flat = config.get_scene("child")
        assert flat is not None and flat.skills is not None
        assert flat.skills.include == ["review-*"]
        assert flat.skills.exclude == []  # parent's exclude NOT inherited

    def test_extends_inherits_omitted_concern(self, tmp_path):
        data = {
            "scenes": {
                "base": {"skills": {"exclude": ["deploy-*"]}},
                "child": {"extends": "base", "mcp": {"include": ["github"]}},
            },
        }
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        flat = load_config(tmp_path).get_scene("child")
        assert flat is not None and flat.skills is not None
        assert flat.skills.exclude == ["deploy-*"]  # inherited verbatim

    def test_scenes_not_a_mapping_raises(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("scenes:\n  - bad\n")
        with pytest.raises(ConfigError, match="'scenes' must be a mapping"):
            load_config(tmp_path)

    def test_scenes_falsy_scalar_raises(self, tmp_path):
        """A falsy scalar must not be coerced to an empty mapping."""
        (tmp_path / ".crossby.yml").write_text("scenes: false\n")
        with pytest.raises(ConfigError, match="'scenes' must be a mapping"):
            load_config(tmp_path)

    def test_scene_entry_not_a_mapping_raises(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("scenes:\n  pr-review: 123\n")
        with pytest.raises(ConfigError, match=r"'scenes\.pr-review' must be a mapping"):
            load_config(tmp_path)

    def test_scene_entry_null_raises(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("scenes:\n  pr-review:\n")
        with pytest.raises(ConfigError, match=r"'scenes\.pr-review' must be a mapping"):
            load_config(tmp_path)

    def test_selector_not_a_list_raises_with_path(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text(
            "scenes:\n  x:\n    skills:\n      include: not-a-list\n"
        )
        with pytest.raises(ConfigError, match=r"scenes\.x") as exc:
            load_config(tmp_path)
        assert "include" in str(exc.value)

    def test_unknown_concern_key_raises_with_path(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("scenes:\n  x:\n    skils: {}\n")
        with pytest.raises(ConfigError, match=r"scenes\.x"):
            load_config(tmp_path)

    def test_undefined_profile_raises_at_load(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("scenes:\n  x:\n    profile: ghost\n")
        with pytest.raises(ConfigError, match=r"x.*undefined profile.*ghost"):
            load_config(tmp_path)

    def test_valid_profile_reference_loads(self, tmp_path):
        data = {
            "profiles": {"ccyolo": {"tool": "claude"}},
            "scenes": {"x": {"profile": "ccyolo"}},
        }
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        config = load_config(tmp_path)
        assert config.scenes["x"].profile == "ccyolo"

    def test_undefined_parent_raises_via_get_scene(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("scenes:\n  child:\n    extends: ghost\n")
        config = load_config(tmp_path)
        with pytest.raises(ConfigError, match=r"child.*extends undefined scene.*ghost"):
            config.get_scene("child")

    def test_cycle_raises_via_get_scene(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text(
            "scenes:\n  a:\n    extends: b\n  b:\n    extends: a\n"
        )
        config = load_config(tmp_path)
        with pytest.raises(ConfigError, match=r"cycle detected: a -> b -> a"):
            config.get_scene("a")


class TestDeprecatedSyncKeyUnchanged:
    """Regression guard: the new ``sync_defaults:`` must not reclaim ``sync:``."""

    def test_top_level_sync_still_emits_deprecation(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("sync:\n  from_tool: claude\n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_config(tmp_path)
        deprecation_msgs = [
            str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert any("'sync' section" in m for m in deprecation_msgs), deprecation_msgs

    def test_sync_defaults_does_not_trigger_deprecation(self, tmp_path):
        (tmp_path / ".crossby.yml").write_text("sync_defaults:\n  from: claude\n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_config(tmp_path)
        deprecation_msgs = [
            str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert not deprecation_msgs, deprecation_msgs


class TestProfileAllowTools:
    """A profile's ``allow_tools`` list must survive the load (not be dropped)."""

    def test_allow_tools_loaded(self, tmp_path):
        data = {
            "version": 1,
            "profiles": {"cop": {"tool": "copilot", "allow_tools": ["github", "shell(git:*)"]}},
        }
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        config = load_config(tmp_path)
        assert config.get_profile("cop").allow_tools == ["github", "shell(git:*)"]

    def test_allow_tools_defaults_empty(self, tmp_path):
        data = {"version": 1, "profiles": {"cop": {"tool": "copilot"}}}
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        config = load_config(tmp_path)
        assert config.get_profile("cop").allow_tools == []

    def test_allow_tools_explicit_null_treated_as_empty(self, tmp_path):
        # ``allow_tools:`` with no value parses as None — normalized to [] like
        # every other section, not rejected as an invalid list.
        (tmp_path / ".crossby.yml").write_text(
            "version: 1\nprofiles:\n  cop:\n    tool: copilot\n    allow_tools:\n"
        )
        config = load_config(tmp_path)
        assert config.get_profile("cop").allow_tools == []

    def test_allow_tools_must_be_list_of_strings(self, tmp_path):
        data = {"version": 1, "profiles": {"cop": {"allow_tools": [1, 2]}}}
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data))
        with pytest.raises(ConfigError, match="allow_tools"):
            load_config(tmp_path)


class TestBrokenSymlinkBoundary:
    """load_config must stop at the same boundary as find_config_entry.

    A dangling ``.crossby.yml`` symlink is a not-yet-populated config identity —
    parse discovery must not walk past it to an ancestor config, or a subdir run
    would resolve scenes from that ancestor while root discovery (scene_root /
    find_config_entry) stops at the broken-symlink dir, rooting scene state/scan
    there. Both discoveries must agree on the same directory.
    """

    def test_dangling_child_symlink_is_empty_config_rooted_at_child(self, tmp_path):
        # Ancestor holds a real config defining scene 'foo'.
        (tmp_path / ".crossby.yml").write_text("version: 1\nscenes:\n  foo: {}\n")
        child = tmp_path / "project"
        child.mkdir()
        # Child's config is a not-yet-populated (dangling) symlink.
        (child / ".crossby.yml").symlink_to(tmp_path / "nonexistent.yml")
        deep = child / "packages" / "app"
        deep.mkdir(parents=True)

        config = load_config(deep)

        # Rooted at the child (the broken-symlink dir), NOT the ancestor, and
        # the ancestor's scenes never bleed in.
        assert config.project_root is not None
        assert config.config_path is not None
        assert Path(config.project_root) == child.resolve()
        assert Path(config.config_path) == child.resolve() / ".crossby.yml"
        assert config.scenes == {}
        assert config.get_scene("foo") is None
        # Parse discovery and root discovery agree on the boundary.
        entry = find_config_entry(deep)
        assert entry is not None
        assert Path(config.project_root) == entry.parent

    def test_dangling_symlink_without_ancestor_roots_at_link_dir(self, tmp_path):
        # No ancestor config at all: still surfaced as an empty config rooted at
        # the link's dir (config_path set), so authoring writes through it.
        child = tmp_path / "project"
        child.mkdir()
        (child / ".crossby.yml").symlink_to(tmp_path / "nonexistent.yml")

        config = load_config(child)

        assert config.scenes == {}
        assert config.project_root is not None
        assert config.config_path is not None
        assert Path(config.project_root) == child.resolve()
        assert Path(config.config_path) == child.resolve() / ".crossby.yml"

    def test_symlink_to_directory_is_rejected(self, tmp_path):
        # A .crossby.yml symlink whose target *exists* but is a directory is not
        # a readable config and not a dangling identity — it must not be masked
        # as an empty config (a read command would use empty config, and an
        # authoring command would splice into it; write_config_checked would then
        # refuse the target with a clean ConfigWriteError). Reject it up front on
        # the read path so read commands don't run against empty config.
        target_dir = tmp_path / "somedir"
        target_dir.mkdir()
        child = tmp_path / "project"
        child.mkdir()
        (child / ".crossby.yml").symlink_to(target_dir)

        with pytest.raises(ConfigError, match="non-file target"):
            load_config(child)

    def test_symlink_to_directory_does_not_walk_past_to_ancestor(self, tmp_path):
        # The non-file symlink is still a config identity: it is rejected rather
        # than silently resolving the ancestor's real config.
        (tmp_path / ".crossby.yml").write_text("version: 1\nscenes:\n  foo: {}\n")
        target_dir = tmp_path / "somedir"
        target_dir.mkdir()
        child = tmp_path / "project"
        child.mkdir()
        (child / ".crossby.yml").symlink_to(target_dir)

        with pytest.raises(ConfigError, match="non-file target"):
            load_config(child)

    def test_symlink_loop_is_rejected_not_treated_as_dangling(self, tmp_path):
        # A symlink loop has exists() == False just like a dangling link, but it
        # is not a not-yet-populated identity — it must be rejected on the read
        # path, not masked as an empty config (write_config_checked would itself
        # refuse the loop with a clean ConfigWriteError, but only after a read
        # command had already run against the empty config).
        child = tmp_path / "project"
        child.mkdir()
        (child / ".crossby.yml").symlink_to(child / "loop-other")
        (child / "loop-other").symlink_to(child / ".crossby.yml")

        with pytest.raises(ConfigError, match="cannot be resolved"):
            load_config(child)

    def test_direct_directory_config_is_rejected(self, tmp_path):
        # A plain-directory .crossby.yml (not a symlink) is neither a file nor a
        # symlink, so find_config_entry historically walked *past* it. It now
        # stops at it and load_config rejects it — it is not a readable config.
        child = tmp_path / "project"
        child.mkdir()
        (child / ".crossby.yml").mkdir()

        with pytest.raises(ConfigError, match="not a regular file"):
            load_config(child)

    def test_direct_directory_does_not_walk_past_to_ancestor(self, tmp_path):
        # The bug this closes: a direct-directory .crossby.yml in a subproject
        # must not silently resolve the ancestor's real config (which would let a
        # read use it and an authoring command edit it).
        (tmp_path / ".crossby.yml").write_text("version: 1\nscenes:\n  foo: {}\n")
        child = tmp_path / "project"
        child.mkdir()
        (child / ".crossby.yml").mkdir()

        with pytest.raises(ConfigError, match="not a regular file"):
            load_config(child)
