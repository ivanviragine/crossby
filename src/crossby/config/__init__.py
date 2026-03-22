"""Configuration layer — loading, resolution, and per-tool config management."""

from crossby.config.loader import ConfigError, find_config_file, load_config

__all__ = ["ConfigError", "find_config_file", "load_config"]
