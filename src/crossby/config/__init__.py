"""Configuration layer — loading, resolution, and per-tool config management."""

from crossby.config.loader import ConfigError, find_config_file, load_config
from crossby.config.claude_allowlist import configure_plan_hooks as claude_configure_plan_hooks
from crossby.config.cursor_hooks import configure_plan_hooks as cursor_configure_plan_hooks
from crossby.config.copilot_hooks import configure_plan_hooks as copilot_configure_plan_hooks
from crossby.config.gemini_hooks import configure_plan_hooks as gemini_configure_plan_hooks

__all__ = [
    "ConfigError",
    "find_config_file",
    "load_config",
    "claude_configure_plan_hooks",
    "cursor_configure_plan_hooks",
    "copilot_configure_plan_hooks",
    "gemini_configure_plan_hooks",
]
