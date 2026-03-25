"""Configuration domain models — CrossbyConfig and nested sections.

Matches the .crossby.yml format:

    version: 1
    ai:
      default_tool: claude
      default_model: claude-sonnet-4.6
      effort: medium
      commands:
        plan:
          tool: claude
          model: claude-opus-4.6
        implement:
          tool: copilot
    models:
      claude:
        easy: claude-haiku-4.5
        medium: claude-sonnet-4.6
        complex: claude-sonnet-4.6
        very_complex: claude-opus-4.6
    permissions:
      allowed_commands:
        - "myapp:*"
        - "./scripts/check.sh:*"
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RulesTargetsConfig(BaseModel):
    """Which tool-specific instruction files to generate."""

    claude: bool = True
    cursor: bool = True
    copilot: bool = True
    gemini: bool = True
    codex: bool = True


class RulesConfig(BaseModel):
    """Rules/instructions sync configuration."""

    source: str = "AGENTS.md"
    strategy: Literal["symlink", "copy"] = "symlink"
    gitignore: bool = True
    targets: RulesTargetsConfig = RulesTargetsConfig()


class ComplexityModelMapping(BaseModel):
    """Model IDs for each complexity tier.

    Values are exact model IDs as returned by the tool's get_models().
    Defaults are None — populated at init time by querying the tool.
    """

    easy: str | None = None
    medium: str | None = None
    complex: str | None = None
    very_complex: str | None = None


class CommandConfig(BaseModel):
    """Per-command AI tool and model override."""

    tool: str | None = None
    model: str | None = None
    effort: str | None = None
    yolo: bool | None = None


class AIConfig(BaseModel):
    """AI tool configuration section — generic command map."""

    default_tool: str | None = None
    default_model: str | None = None
    effort: str | None = None
    yolo: bool | None = None
    commands: dict[str, CommandConfig] = {}


class PermissionsConfig(BaseModel):
    """Permission pre-authorization for AI tool sessions.

    Canonical command patterns (e.g. ``"myapp:*"``) are translated to
    tool-specific allowlist flags at launch time.
    """

    allowed_commands: list[str] = []


class CrossbyConfig(BaseModel):
    """Full configuration from .crossby.yml.

    This is the validated, structured representation. The config loader
    parses the YAML file and constructs this model.
    """

    version: int = 1

    ai: AIConfig = AIConfig()
    models: dict[str, ComplexityModelMapping] = {}
    permissions: PermissionsConfig = PermissionsConfig()
    rules: RulesConfig | None = None

    # Resolved values (set after loading, not in YAML)
    config_path: str | None = Field(default=None, exclude=True)
    project_root: str | None = Field(default=None, exclude=True)

    def get_ai_tool(self, command: str | None = None) -> str | None:
        """Get the AI tool for a command, with fallback chain.

        Fallback: command-specific tool → global default_tool → None.
        """
        if command and command in self.ai.commands:
            cmd_config = self.ai.commands[command]
            if cmd_config.tool:
                return cmd_config.tool
        return self.ai.default_tool

    def get_model(self, command: str | None = None) -> str | None:
        """Get the model for a command, with fallback chain.

        Fallback: command-specific model → ai.default_model → None.
        """
        if command and command in self.ai.commands:
            cmd_config = self.ai.commands[command]
            if cmd_config.model:
                return cmd_config.model
        return self.ai.default_model

    def get_complexity_model(self, tool: str, complexity: str) -> str | None:
        """Get model ID for a tool + complexity combination."""
        mapping = self.models.get(tool)
        if mapping:
            return getattr(mapping, complexity, None)
        return None

    def get_effort(self, command: str | None = None) -> str | None:
        """Get the effort level for a command, with fallback chain.

        Fallback: command-specific effort → global ai.effort → None.
        """
        if command and command in self.ai.commands:
            cmd_config = self.ai.commands[command]
            if cmd_config.effort:
                return cmd_config.effort
        return self.ai.effort

    def get_yolo(self, command: str | None = None) -> bool | None:
        """Get the yolo setting for a command, with fallback chain.

        Fallback: command-specific yolo → global ai.yolo → None.
        """
        if command and command in self.ai.commands:
            cmd_config = self.ai.commands[command]
            if cmd_config.yolo is not None:
                return cmd_config.yolo
        return self.ai.yolo
