"""Configuration loader — find + parse .crossby.yml (walk up from CWD)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from crossby.models.ai import AIToolID
from crossby.models.config import (
    AIConfig,
    AgentsConfig,
    CommandConfig,
    ComplexityModelMapping,
    CrossbyConfig,
    MCPServerConfig,
    PermissionsConfig,
    RulesConfig,
    RulesTargetsConfig,
    SyncConfig,
)

CONFIG_FILENAME = ".crossby.yml"


class ConfigError(Exception):
    """Raised when .crossby.yml cannot be parsed or has invalid structure."""


def ensure_yaml_mapping(raw: Any) -> dict[str, Any] | None:
    """Validate that parsed YAML is a dict (mapping).

    Returns:
        The dict if raw is a dict, None if raw is None (empty file).

    Raises:
        ConfigError: If raw is a non-dict, non-None value (list, scalar).
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    raise ConfigError("Config must be a YAML mapping (key: value pairs)")


def find_config_file(start: Path | None = None) -> Path | None:
    """Walk up from start (or CWD) looking for .crossby.yml.

    Returns the path to the config file, or None if not found.
    """
    current = (start or Path.cwd()).resolve()

    while True:
        candidate = current / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            break  # Reached filesystem root
        current = parent

    return None


def load_config(start: Path | None = None) -> CrossbyConfig:
    """Find and parse the project config.

    Returns a CrossbyConfig with defaults if no config file exists.
    """
    config_path = find_config_file(start)
    if config_path is None:
        return CrossbyConfig()

    return parse_config_file(config_path)


def parse_config_file(config_path: Path) -> CrossbyConfig:
    """Parse a .crossby.yml file into a CrossbyConfig."""
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {config_path}: {e}") from e

    validated = ensure_yaml_mapping(raw)
    if validated is None:
        # Empty file — treated as defaults
        return CrossbyConfig(
            config_path=str(config_path),
            project_root=str(config_path.parent),
        )

    try:
        return _build_config(validated, config_path)
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise ConfigError(f"Invalid config structure in {config_path}: {e}") from e


def _build_config(raw: dict[str, Any], config_path: Path) -> CrossbyConfig:
    """Build a CrossbyConfig from raw YAML dict."""
    version = raw.get("version", 1)

    # Parse ai section
    ai_raw = raw.get("ai")
    if ai_raw is None:
        ai_raw = {}
    if not isinstance(ai_raw, dict):
        raise ConfigError("'ai' must be a mapping")

    commands: dict[str, CommandConfig] = {}
    commands_raw = ai_raw.get("commands")
    if commands_raw is None:
        commands_raw = {}
    if not isinstance(commands_raw, dict):
        raise ConfigError("'ai.commands' must be a mapping")
    for cmd_name, cmd_raw in commands_raw.items():
        commands[cmd_name] = _parse_command_config(cmd_raw)
    ai = AIConfig(
        default_tool=ai_raw.get("default_tool"),
        default_model=ai_raw.get("default_model"),
        effort=ai_raw.get("effort"),
        yolo=ai_raw.get("yolo"),
        commands=commands,
    )

    # Parse models section (nested: tool -> complexity -> model)
    models_raw = raw.get("models")
    if models_raw is None:
        models_raw = {}
    if not isinstance(models_raw, dict):
        raise ConfigError("'models' must be a mapping")
    models: dict[str, ComplexityModelMapping] = {}
    for tool_name, mapping_raw in models_raw.items():
        if isinstance(mapping_raw, dict):
            models[tool_name] = ComplexityModelMapping(
                easy=mapping_raw.get("easy"),
                medium=mapping_raw.get("medium"),
                complex=mapping_raw.get("complex"),
                very_complex=mapping_raw.get("very_complex"),
            )

    # Parse permissions section
    permissions_raw = raw.get("permissions")
    if permissions_raw is None:
        permissions_raw = {}
    if not isinstance(permissions_raw, dict):
        raise ConfigError("'permissions' must be a mapping")
    permissions = PermissionsConfig(
        allowed_commands=permissions_raw.get("allowed_commands", []),
    )

    # Parse mcp_servers section
    mcp_raw = raw.get("mcp_servers")
    if mcp_raw is None:
        mcp_raw = {}
    if not isinstance(mcp_raw, dict):
        raise ConfigError("'mcp_servers' must be a mapping")
    mcp_servers: dict[str, MCPServerConfig] = {}
    for server_name, server_raw in mcp_raw.items():
        if not isinstance(server_raw, dict):
            raise ConfigError(f"'mcp_servers.{server_name}' must be a mapping")
        try:
            mcp_servers[server_name] = MCPServerConfig(**server_raw)
        except (TypeError, ValueError, ValidationError) as e:
            raise ConfigError(f"Invalid MCP server '{server_name}': {e}") from e

    # Parse sync section
    sync_raw = raw.get("sync")
    if sync_raw is None:
        sync_raw = {}
    if not isinstance(sync_raw, dict):
        raise ConfigError("'sync' must be a mapping")
    sync_auto = sync_raw.get("auto", True)
    if not isinstance(sync_auto, bool):
        raise ConfigError(
            f"'sync.auto' must be a boolean (true/false), got {sync_auto!r}"
        )
    sync = SyncConfig(
        auto=sync_auto,
        tools=sync_raw.get("tools", []),
    )

    # Parse agents section
    agents_raw = raw.get("agents")
    if agents_raw is None:
        agents_raw = {}
    if not isinstance(agents_raw, dict):
        raise ConfigError("'agents' must be a mapping")
    agents_targets_raw = agents_raw.get("targets")
    if agents_targets_raw is None:
        agents_targets_raw = {}
    if not isinstance(agents_targets_raw, dict):
        raise ConfigError("'agents.targets' must be a mapping")
    strategy = agents_raw.get("strategy", "symlink")
    if strategy not in ("symlink", "copy"):
        raise ConfigError(
            f"'agents.strategy' must be one of 'symlink' or 'copy', got {strategy!r}"
        )
    known_agent_targets = {str(t) for t in AIToolID}
    unknown_agent_keys = [k for k in agents_targets_raw if str(k) not in known_agent_targets]
    if unknown_agent_keys:
        unknown_list = ", ".join(sorted(str(k) for k in unknown_agent_keys))
        raise ConfigError(f"Unknown 'agents.targets' keys: {unknown_list}")
    targets: dict[str, bool] = {}
    for k, v in agents_targets_raw.items():
        if not isinstance(v, bool):
            raise ConfigError(
                f"'agents.targets.{k}' must be a boolean (true/false), got {v!r}"
            )
        targets[str(k)] = v
    gitignore_raw = agents_raw.get("gitignore", True)
    if not isinstance(gitignore_raw, bool):
        raise ConfigError(
            f"'agents.gitignore' must be a boolean (true/false), got {gitignore_raw!r}"
        )
    agents = AgentsConfig(
        enabled="agents" in raw and raw.get("agents") is not None,
        source=agents_raw.get("source", ".crossby/agents"),
        strategy=strategy,
        gitignore=gitignore_raw,
        targets=targets,
    )

    # Parse rules section
    rules = _parse_rules_config(raw)

    return CrossbyConfig(
        version=version,
        ai=ai,
        models=models,
        permissions=permissions,
        mcp_servers=mcp_servers,
        rules=rules,
        sync=sync,
        agents=agents,
        config_path=str(config_path),
        project_root=str(config_path.parent),
    )


def _parse_rules_config(raw: dict[str, Any]) -> RulesConfig:
    """Parse the rules section from config YAML."""
    rules_raw = raw.get("rules")
    if rules_raw is None:
        return RulesConfig()
    if not isinstance(rules_raw, dict):
        raise ConfigError("'rules' must be a mapping")

    targets_raw = rules_raw.get("targets")
    targets = RulesTargetsConfig()
    if targets_raw is not None:
        if not isinstance(targets_raw, dict):
            raise ConfigError("'rules.targets' must be a mapping")
        known_target_keys = set(RulesTargetsConfig.model_fields)
        unknown_keys = [k for k in targets_raw.keys() if k not in known_target_keys]
        if unknown_keys:
            unknown_list = ", ".join(sorted(str(k) for k in unknown_keys))
            raise ConfigError(f"Unknown 'rules.targets' keys: {unknown_list}")
        for key, value in targets_raw.items():
            if key in known_target_keys and not isinstance(value, bool):
                raise ConfigError(f"'rules.targets.{key}' must be a boolean")
        targets = RulesTargetsConfig(**{
            k: v for k, v in targets_raw.items() if k in known_target_keys
        })

    strategy = rules_raw.get("strategy", "symlink")
    if strategy not in ("symlink", "copy"):
        raise ConfigError(f"'rules.strategy' must be 'symlink' or 'copy', got '{strategy}'")

    gitignore_raw = rules_raw.get("gitignore", True)
    if not isinstance(gitignore_raw, bool):
        raise ConfigError("'rules.gitignore' must be a boolean")

    return RulesConfig(
        enabled="rules" in raw and raw.get("rules") is not None,
        source=rules_raw.get("source", "AGENTS.md"),
        strategy=strategy,
        gitignore=gitignore_raw,
        targets=targets,
    )


def _parse_command_config(raw: dict[str, Any] | None) -> CommandConfig:
    """Parse a per-command AI config section."""
    if not raw or not isinstance(raw, dict):
        return CommandConfig()
    return CommandConfig(
        tool=raw.get("tool"),
        model=raw.get("model") or None,  # Treat empty string as None
        effort=raw.get("effort"),
        yolo=raw.get("yolo"),
    )
