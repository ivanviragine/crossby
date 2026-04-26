"""Configuration loader — find + parse .crossby.yml (walk up from CWD)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml

from pydantic import ValidationError

from crossby.models.config import (
    AIConfig,
    CommandConfig,
    ComplexityModelMapping,
    CrossbyConfig,
    HandoffDefaults,
    ProfileConfig,
    SyncDefaults,
)

CONFIG_FILENAME = ".crossby.yml"

# Sections removed in the stateless sync refactor — silently ignored.
_DEPRECATED_SECTIONS = frozenset({
    "permissions", "mcp_servers", "rules", "sync", "agents", "hooks",
})


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

    # Warn about deprecated sections (from pre-stateless-sync configs)
    for section in _DEPRECATED_SECTIONS:
        if section in raw and raw[section] is not None:
            warnings.warn(
                f"'{section}' section in .crossby.yml is deprecated and ignored. "
                "Sync now reads directly from tool configs.",
                DeprecationWarning,
                stacklevel=4,
            )

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
        if not isinstance(cmd_raw, dict):
            raise ConfigError(f"'ai.commands.{cmd_name}' must be a mapping")
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
        if not isinstance(mapping_raw, dict):
            raise ConfigError(f"'models.{tool_name}' must be a mapping")
        models[tool_name] = ComplexityModelMapping(
            easy=mapping_raw.get("easy"),
            medium=mapping_raw.get("medium"),
            complex=mapping_raw.get("complex"),
            very_complex=mapping_raw.get("very_complex"),
        )

    # Parse profiles section
    profiles_raw = raw.get("profiles")
    if profiles_raw is None:
        profiles_raw = {}
    if not isinstance(profiles_raw, dict):
        raise ConfigError("'profiles' must be a mapping")
    profiles: dict[str, ProfileConfig] = {}
    for name, profile_raw in profiles_raw.items():
        if not isinstance(profile_raw, dict):
            raise ConfigError(f"'profiles.{name}' must be a mapping")
        profiles[name] = ProfileConfig(
            tool=profile_raw.get("tool"),
            model=profile_raw.get("model"),
            effort=profile_raw.get("effort"),
            yolo=profile_raw.get("yolo"),
        )

    # Parse sync_defaults / handoff_defaults sections
    sync_defaults = _parse_sync_defaults(raw.get("sync_defaults"))
    handoff_defaults = _parse_handoff_defaults(raw.get("handoff_defaults"))

    return CrossbyConfig(
        version=version,
        ai=ai,
        models=models,
        profiles=profiles,
        sync_defaults=sync_defaults,
        handoff_defaults=handoff_defaults,
        config_path=str(config_path),
        project_root=str(config_path.parent),
    )


def _parse_sync_defaults(raw: Any) -> SyncDefaults:
    """Parse the ``sync_defaults`` section. Accepts ``from:`` alias."""
    if raw is None:
        return SyncDefaults()
    if not isinstance(raw, dict):
        raise ConfigError("'sync_defaults' must be a mapping")
    try:
        return SyncDefaults.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid 'sync_defaults': {exc}") from exc


def _parse_handoff_defaults(raw: Any) -> HandoffDefaults:
    """Parse the ``handoff_defaults`` section and validate ``prompt_preset``."""
    if raw is None:
        return HandoffDefaults()
    if not isinstance(raw, dict):
        raise ConfigError("'handoff_defaults' must be a mapping")
    try:
        defaults = HandoffDefaults.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid 'handoff_defaults': {exc}") from exc

    # Validate prompt_preset against the handoff preset registry here (not on
    # the model) to avoid a circular import between models/config.py and
    # handoff/prompts.py.
    if defaults.prompt_preset is not None:
        from crossby.handoff.prompts import PRESETS

        if defaults.prompt_preset not in PRESETS:
            valid = ", ".join(sorted(PRESETS))
            raise ConfigError(
                f"Invalid 'handoff_defaults.prompt_preset': "
                f"{defaults.prompt_preset!r}. Valid presets: {valid}."
            )
    return defaults


def _parse_command_config(raw: dict[str, Any]) -> CommandConfig:
    """Parse a per-command AI config section.

    The caller enforces that ``raw`` is a mapping; an empty dict is allowed
    and yields a ``CommandConfig`` with all-default fields.
    """
    if not raw:
        return CommandConfig()
    return CommandConfig(
        tool=raw.get("tool"),
        model=raw.get("model") or None,  # Treat empty string as None
        effort=raw.get("effort"),
        yolo=raw.get("yolo"),
    )
