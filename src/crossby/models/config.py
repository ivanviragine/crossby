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
    profiles:
      ccyolo:
        tool: claude
        yolo: true
        effort: max
      quick:
        tool: cursor
        model: haiku
        effort: low
    sync_defaults:
      from: claude
      to: null
      concern: null
    handoff_defaults:
      from: claude
      to: codex
      prompt_preset: default
      token_budget: 32000
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crossby.models.ai import AIToolID


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server entry.

    A server must have either ``command`` (stdio transport) or ``url``
    (http/sse transport) — not both, not neither.
    """

    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    transport: Literal["stdio", "http", "sse"] = "stdio"
    url: str | None = None
    enabled: bool = True

    @model_validator(mode="after")
    def validate_transport_fields(self) -> "MCPServerConfig":
        has_command = self.command is not None
        has_url = self.url is not None
        if has_command and has_url:
            raise ValueError("MCP server must have 'command' or 'url', not both")
        if not has_command and not has_url:
            raise ValueError("MCP server must have either 'command' (stdio) or 'url' (http/sse)")
        if has_command and self.transport != "stdio":
            raise ValueError(
                f"transport must be 'stdio' when 'command' is set, got '{self.transport}'"
            )
        if has_url and self.transport not in {"http", "sse"}:
            raise ValueError(
                f"transport must be 'http' or 'sse' when 'url' is set, got '{self.transport}'"
            )
        return self


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


class HookEntry(BaseModel):
    """A single canonical hook definition (used by sync readers/writers)."""

    event: str
    command: str
    tools: list[str] = Field(default_factory=list)
    description: str = ""


class ProfileConfig(BaseModel):
    """A saved launch profile (stored in .crossby.yml under ``profiles``)."""

    tool: str | None = None
    model: str | None = None
    effort: str | None = None
    yolo: bool | None = None


class SyncDefaults(BaseModel):
    """Defaults for ``crossby sync`` — all fields optional.

    The YAML key is ``sync_defaults`` (not ``sync``, which is a
    deprecated-and-ignored legacy key handled by the loader).

    The ``from:`` YAML key maps to the Python field ``from_tool``
    because ``from`` is a reserved keyword.
    """

    model_config = ConfigDict(populate_by_name=True)

    from_tool: AIToolID | None = Field(default=None, alias="from")
    to: AIToolID | None = None
    concern: str | None = None


class HandoffDefaults(BaseModel):
    """Defaults for ``crossby handoff`` — all fields optional.

    See :class:`SyncDefaults` for the ``from`` / ``from_tool`` alias note.
    ``prompt_preset`` is validated by the loader (not on this model) to
    avoid a circular import with ``crossby.handoff.prompts``.
    """

    model_config = ConfigDict(populate_by_name=True)

    from_tool: AIToolID | None = Field(default=None, alias="from")
    to: AIToolID | None = None
    prompt_preset: str | None = None
    token_budget: int | None = None


class CrossbyConfig(BaseModel):
    """Full configuration from .crossby.yml.

    This is the validated, structured representation. The config loader
    parses the YAML file and constructs this model.

    Only contains launch preferences (AI defaults, model mappings, profiles).
    Sync data is read directly from tool configs by ``sync.readers``.
    """

    version: int = 1

    ai: AIConfig = AIConfig()
    models: dict[str, ComplexityModelMapping] = {}
    profiles: dict[str, ProfileConfig] = Field(default_factory=dict)
    sync_defaults: SyncDefaults = Field(default_factory=SyncDefaults)
    handoff_defaults: HandoffDefaults = Field(default_factory=HandoffDefaults)

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

    def get_profile(self, name: str) -> ProfileConfig | None:
        """Get a named launch profile."""
        return self.profiles.get(name)

    def get_sync_from(self) -> AIToolID | None:
        """Get the default source tool for ``crossby sync``."""
        return self.sync_defaults.from_tool

    def get_sync_to(self) -> AIToolID | None:
        """Get the default target tool for ``crossby sync`` (``None`` = all installed)."""
        return self.sync_defaults.to

    def get_sync_concern(self) -> str | None:
        """Get the default sync concern (``None`` = all concerns)."""
        return self.sync_defaults.concern

    def get_handoff_from(self) -> AIToolID | None:
        """Get the default source tool for ``crossby handoff``."""
        return self.handoff_defaults.from_tool

    def get_handoff_to(self) -> AIToolID | None:
        """Get the default target tool for ``crossby handoff``."""
        return self.handoff_defaults.to

    def get_handoff_preset(self) -> str | None:
        """Get the default summarization prompt preset for ``crossby handoff``."""
        return self.handoff_defaults.prompt_preset

    def get_handoff_token_budget(self) -> int | None:
        """Get the default token budget for ``crossby handoff``."""
        return self.handoff_defaults.token_budget
