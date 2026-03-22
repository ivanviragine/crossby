"""Tests for CrossbyConfig generic command map."""

from crossby.models.config import (
    AIConfig,
    CommandConfig,
    ComplexityModelMapping,
    CrossbyConfig,
)


class TestCrossbyConfig:
    def test_defaults(self):
        config = CrossbyConfig()
        assert config.version == 1
        assert config.ai.default_tool is None
        assert config.ai.commands == {}
        assert config.permissions.allowed_commands == []

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
