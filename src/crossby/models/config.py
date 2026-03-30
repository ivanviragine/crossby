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
    mcp_servers:
      context7:
        command: npx
        args: ["-y", "@upstash/context7-mcp"]
      postgres:
        transport: http
        url: "http://localhost:8080/mcp"
    agents:
      source: .crossby/agents
      strategy: symlink
      gitignore: true
      targets:
        claude: true
        copilot: true
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


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


class PermissionsConfig(BaseModel):
    """Permission pre-authorization for AI tool sessions.

    Canonical command patterns (e.g. ``"myapp:*"``) are translated to
    tool-specific allowlist flags at launch time.
    """

    allowed_commands: list[str] = []


class SyncConfig(BaseModel):
    """Sync behavior configuration (``sync:`` section in .crossby.yml).

    ``auto``: run sync automatically on ``crossby launch`` (default: true).
    ``tools``: restrict sync to these tool IDs (empty = all installed tools).
    """

    auto: bool = True
    tools: list[str] = []


class RulesTargetsConfig(BaseModel):
    """Which tool-specific instruction files to generate."""

    claude: bool = True
    cursor: bool = True
    copilot: bool = True
    gemini: bool = True
    codex: bool = True


class RulesConfig(BaseModel):
    """Rules/instructions sync configuration."""

    enabled: bool = False
    source: str = "AGENTS.md"
    strategy: Literal["symlink", "copy"] = "symlink"
    gitignore: bool = True
    targets: RulesTargetsConfig = RulesTargetsConfig()


class HookEntry(BaseModel):
    """A single canonical hook definition.

    Canonical format (in .crossby.yml):

        hooks:
          - event: pre_tool_use
            command: "python3 ./scripts/guard.py"
            tools: ["Edit", "Write"]
            description: "Plan write guard"
    """

    event: str
    command: str
    tools: list[str] = Field(default_factory=list)
    description: str = ""


class AgentsConfig(BaseModel):
    """Agents sync configuration (``agents:`` section in .crossby.yml).

    ``enabled``: True when an ``agents:`` section exists in ``.crossby.yml``.
        Writers skip when False (no agents section → nothing to sync).
    ``source``: canonical agent directory (default: ``.crossby/agents``).
    ``strategy``: ``"symlink"`` (default) or ``"copy"``.
    ``gitignore``: manage .gitignore entries for generated dirs (default: true).
    ``targets``: dict of ``{tool_id: bool}`` — empty dict means all installed tools.
    """

    enabled: bool = False
    source: str = ".crossby/agents"
    strategy: Literal["symlink", "copy"] = "symlink"
    gitignore: bool = True
    targets: dict[str, bool] = {}


class CrossbyConfig(BaseModel):
    """Full configuration from .crossby.yml.

    This is the validated, structured representation. The config loader
    parses the YAML file and constructs this model.
    """

    version: int = 1

    ai: AIConfig = AIConfig()
    models: dict[str, ComplexityModelMapping] = {}
    permissions: PermissionsConfig = PermissionsConfig()
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    rules: RulesConfig = RulesConfig()
    sync: SyncConfig = SyncConfig()
    agents: AgentsConfig = AgentsConfig()
    hooks: list[HookEntry] = Field(default_factory=list)

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
