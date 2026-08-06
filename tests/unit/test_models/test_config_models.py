"""Tests for CrossbyConfig generic command map."""

import pytest
from pydantic import ValidationError

from crossby.models.config import (
    AIConfig,
    CommandConfig,
    ComplexityModelMapping,
    CrossbyConfig,
    SceneConfig,
    SceneSelector,
)


class TestCrossbyConfig:
    def test_defaults(self):
        config = CrossbyConfig()
        assert config.version == 1
        assert config.ai.default_tool is None
        assert config.ai.commands == {}
        assert config.profiles == {}

    def test_get_ai_tool_global(self):
        config = CrossbyConfig(ai=AIConfig(default_tool="claude"))
        assert config.get_ai_tool() == "claude"
        assert config.get_ai_tool("unknown") == "claude"

    def test_get_ai_tool_command_override(self):
        config = CrossbyConfig(
            ai=AIConfig(
                default_tool="claude",
                commands={"plan": CommandConfig(tool="copilot")},
            )
        )
        assert config.get_ai_tool("plan") == "copilot"
        assert config.get_ai_tool("implement") == "claude"

    def test_get_model_fallback(self):
        config = CrossbyConfig(
            ai=AIConfig(
                default_model="sonnet-4.6",
                commands={"plan": CommandConfig(model="opus-4.6")},
            )
        )
        assert config.get_model("plan") == "opus-4.6"
        assert config.get_model("other") == "sonnet-4.6"
        assert config.get_model() == "sonnet-4.6"

    def test_get_complexity_model(self):
        config = CrossbyConfig(
            models={
                "claude": ComplexityModelMapping(
                    easy="claude-haiku-4.5",
                    medium="claude-sonnet-4.6",
                )
            }
        )
        assert config.get_complexity_model("claude", "easy") == "claude-haiku-4.5"
        assert config.get_complexity_model("claude", "medium") == "claude-sonnet-4.6"
        assert config.get_complexity_model("unknown", "easy") is None

    def test_get_complexity_effort(self):
        config = CrossbyConfig(
            models={
                "claude": ComplexityModelMapping(
                    easy_effort="low",
                    very_complex_effort="high",
                )
            }
        )
        assert config.get_complexity_effort("claude", "easy") == "low"
        assert config.get_complexity_effort("claude", "very_complex") == "high"
        assert config.get_complexity_effort("claude", "medium") is None
        assert config.get_complexity_effort("unknown", "easy") is None

    def test_get_effort_fallback(self):
        config = CrossbyConfig(
            ai=AIConfig(
                effort="medium",
                commands={"plan": CommandConfig(effort="high")},
            )
        )
        assert config.get_effort("plan") == "high"
        assert config.get_effort("other") == "medium"

    def test_get_yolo_fallback(self):
        config = CrossbyConfig(
            ai=AIConfig(
                yolo=False,
                commands={"implement": CommandConfig(yolo=True)},
            )
        )
        assert config.get_yolo("implement") is True
        assert config.get_yolo("other") is False

    def test_get_accept_edits_fallback(self):
        config = CrossbyConfig(
            ai=AIConfig(
                accept_edits=False,
                commands={"implement": CommandConfig(accept_edits=True)},
            )
        )
        assert config.get_accept_edits("implement") is True
        assert config.get_accept_edits("other") is False
        assert CrossbyConfig().get_accept_edits() is None

    def test_get_auto_fallback(self):
        config = CrossbyConfig(
            ai=AIConfig(
                auto=True,
                commands={"review": CommandConfig(auto=False)},
            )
        )
        assert config.get_auto("review") is False
        assert config.get_auto("other") is True
        assert CrossbyConfig().get_auto() is None

    def test_profile_persists_accept_edits_and_auto(self):
        from crossby.models.config import ProfileConfig

        prof = ProfileConfig(tool="claude", accept_edits=True, auto=True)
        assert prof.accept_edits is True
        assert prof.auto is True

    def test_profiles_empty_default(self):
        config = CrossbyConfig()
        assert config.profiles == {}

    def test_scenes_empty_default(self):
        config = CrossbyConfig()
        assert config.scenes == {}
        assert config.get_scene("anything") is None


class TestSceneSelector:
    def test_defaults(self):
        sel = SceneSelector()
        assert sel.include is None  # absent → "everything", distinct from []
        assert sel.exclude == []

    def test_populated(self):
        sel = SceneSelector(include=["review-*", "knowledge"], exclude=["deploy-*"])
        assert sel.include == ["review-*", "knowledge"]
        assert sel.exclude == ["deploy-*"]

    def test_unknown_key_forbidden(self):
        with pytest.raises(ValidationError):
            SceneSelector.model_validate({"includ": ["typo"]})

    def test_include_must_be_a_list(self):
        with pytest.raises(ValidationError):
            SceneSelector.model_validate({"include": "not-a-list"})


class TestSceneConfig:
    def test_defaults_all_concerns_none(self):
        scene = SceneConfig()
        assert scene.description is None
        assert scene.extends is None
        assert scene.profile is None
        assert scene.skills is None
        assert scene.agents is None
        assert scene.mcp is None
        assert scene.hooks is None
        assert scene.permissions is None

    def test_unknown_concern_key_forbidden(self):
        """A typo'd concern (``skils``) must raise, not vanish silently."""
        with pytest.raises(ValidationError):
            SceneConfig.model_validate({"skils": {"include": ["x"]}})

    def test_rules_is_not_a_selectable_concern(self):
        with pytest.raises(ValidationError):
            SceneConfig.model_validate({"rules": {"include": ["x"]}})

    def test_selectors_parse(self):
        scene = SceneConfig.model_validate(
            {
                "description": "Review a PR",
                "skills": {"include": ["review-*"]},
                "mcp": {"include": ["github"]},
            }
        )
        assert scene.description == "Review a PR"
        assert scene.skills is not None
        assert scene.skills.include == ["review-*"]
        assert scene.mcp is not None
        assert scene.mcp.include == ["github"]


class TestGetSceneFlattening:
    """``get_scene`` folds the ``extends`` chain with per-concern replace."""

    def _config(self, scenes: dict[str, dict[str, object]]) -> CrossbyConfig:
        return CrossbyConfig(
            scenes={name: SceneConfig.model_validate(body) for name, body in scenes.items()}
        )

    def test_child_selector_replaces_parent(self):
        config = self._config(
            {
                "base": {"skills": {"exclude": ["deploy-*"]}},
                "child": {"extends": "base", "skills": {"include": ["review-*"]}},
            }
        )
        flat = config.get_scene("child")
        assert flat is not None
        assert flat.skills is not None
        # Child declared skills → parent's exclude is NOT inherited.
        assert flat.skills.include == ["review-*"]
        assert flat.skills.exclude == []
        # Parent is left untouched.
        assert config.scenes["base"].skills is not None
        assert config.scenes["base"].skills.exclude == ["deploy-*"]

    def test_omitted_concern_inherited_verbatim(self):
        config = self._config(
            {
                "base": {"skills": {"exclude": ["deploy-*"]}, "mcp": {"include": ["github"]}},
                "child": {"extends": "base", "agents": {"include": ["code-reviewer"]}},
            }
        )
        flat = config.get_scene("child")
        assert flat is not None
        assert flat.skills is not None
        assert flat.skills.exclude == ["deploy-*"]  # inherited
        assert flat.mcp is not None
        assert flat.mcp.include == ["github"]  # inherited
        assert flat.agents is not None
        assert flat.agents.include == ["code-reviewer"]  # child's own

    def test_description_and_profile_inherit_then_replace(self):
        config = CrossbyConfig(
            profiles={},
            scenes={
                "base": SceneConfig(description="base desc", profile=None),
                "inherits": SceneConfig(extends="base"),
                "overrides": SceneConfig(extends="base", description="own desc"),
            },
        )
        assert config.get_scene("inherits").description == "base desc"  # type: ignore[union-attr]
        assert config.get_scene("overrides").description == "own desc"  # type: ignore[union-attr]

    def test_three_level_chain(self):
        config = self._config(
            {
                "c": {"mcp": {"include": ["github"]}},
                "b": {"extends": "c", "hooks": {"include": ["pre_tool_use:*"]}},
                "a": {"extends": "b", "skills": {"include": ["review-*"]}},
            }
        )
        flat = config.get_scene("a")
        assert flat is not None
        assert flat.mcp is not None and flat.mcp.include == ["github"]
        assert flat.hooks is not None and flat.hooks.include == ["pre_tool_use:*"]
        assert flat.skills is not None and flat.skills.include == ["review-*"]
        assert flat.extends is None  # flattened result carries no extends

    def test_unknown_scene_returns_none(self):
        config = self._config({"base": {}})
        assert config.get_scene("missing") is None

    def test_undefined_parent_raises(self):
        from crossby.config.loader import ConfigError

        config = self._config({"child": {"extends": "ghost"}})
        with pytest.raises(ConfigError, match=r"child.*extends undefined scene.*ghost"):
            config.get_scene("child")

    def test_self_reference_raises(self):
        from crossby.config.loader import ConfigError

        config = self._config({"loop": {"extends": "loop"}})
        with pytest.raises(ConfigError, match=r"cycle detected: loop -> loop"):
            config.get_scene("loop")

    def test_mutual_cycle_raises_naming_chain(self):
        from crossby.config.loader import ConfigError

        config = self._config({"a": {"extends": "b"}, "b": {"extends": "a"}})
        with pytest.raises(ConfigError, match=r"cycle detected: a -> b -> a"):
            config.get_scene("a")
