"""Tests for MCPServerConfig model and config loader mcp_servers parsing."""

from __future__ import annotations

import yaml
from pathlib import Path

import pytest

from crossby.config.loader import ConfigError, load_config
from crossby.models.config import MCPServerConfig


class TestMCPServerConfigValidation:
    def test_stdio_requires_command(self) -> None:
        with pytest.raises(Exception):
            MCPServerConfig()  # no command or url

    def test_both_command_and_url_raises(self) -> None:
        with pytest.raises(Exception):
            MCPServerConfig(command="npx", url="http://localhost")

    def test_valid_stdio_minimal(self) -> None:
        s = MCPServerConfig(command="npx")
        assert s.command == "npx"
        assert s.args == []
        assert s.env == {}
        assert s.transport == "stdio"
        assert s.enabled is True

    def test_valid_stdio_full(self) -> None:
        s = MCPServerConfig(
            command="npx",
            args=["-y", "mcp"],
            env={"KEY": "val"},
            transport="stdio",
            enabled=True,
        )
        assert s.args == ["-y", "mcp"]
        assert s.env == {"KEY": "val"}

    def test_valid_http(self) -> None:
        s = MCPServerConfig(transport="http", url="http://localhost:8080/mcp")
        assert s.url == "http://localhost:8080/mcp"
        assert s.command is None

    def test_valid_sse(self) -> None:
        s = MCPServerConfig(transport="sse", url="http://localhost:8080/sse")
        assert s.transport == "sse"

    def test_url_with_stdio_transport_raises(self) -> None:
        with pytest.raises(Exception, match="transport"):
            MCPServerConfig(url="http://localhost", transport="stdio")

    def test_command_with_http_transport_raises(self) -> None:
        with pytest.raises(Exception, match="transport"):
            MCPServerConfig(command="npx", transport="http")

    def test_invalid_transport_value_raises(self) -> None:
        with pytest.raises(Exception):
            MCPServerConfig(command="npx", transport="grpc")  # type: ignore[arg-type]

    def test_disabled(self) -> None:
        s = MCPServerConfig(command="npx", enabled=False)
        assert s.enabled is False

    def test_env_var_reference_preserved(self) -> None:
        s = MCPServerConfig(command="npx", env={"TOKEN": "${MY_TOKEN}"})
        assert s.env["TOKEN"] == "${MY_TOKEN}"


class TestLoaderMCPServers:
    def test_parses_mcp_servers(self, tmp_path: Path) -> None:
        data = {
            "version": 1,
            "mcp_servers": {
                "context7": {
                    "command": "npx",
                    "args": ["-y", "@upstash/context7-mcp"],
                },
                "api": {
                    "transport": "http",
                    "url": "http://localhost:8080/mcp",
                },
            },
        }
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data), encoding="utf-8")
        config = load_config(tmp_path)
        assert "context7" in config.mcp_servers
        assert config.mcp_servers["context7"].command == "npx"
        assert config.mcp_servers["context7"].args == ["-y", "@upstash/context7-mcp"]
        assert config.mcp_servers["api"].transport == "http"
        assert config.mcp_servers["api"].url == "http://localhost:8080/mcp"

    def test_missing_mcp_servers_defaults_empty(self, tmp_path: Path) -> None:
        (tmp_path / ".crossby.yml").write_text("version: 1\n", encoding="utf-8")
        config = load_config(tmp_path)
        assert config.mcp_servers == {}

    def test_mcp_servers_not_a_mapping_raises(self, tmp_path: Path) -> None:
        (tmp_path / ".crossby.yml").write_text("mcp_servers:\n  - bad\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="mcp_servers"):
            load_config(tmp_path)

    def test_mcp_server_invalid_raises(self, tmp_path: Path) -> None:
        data = {"mcp_servers": {"bad": {"command": "npx", "url": "http://x"}}}
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data), encoding="utf-8")
        with pytest.raises(ConfigError, match="bad"):
            load_config(tmp_path)

    def test_mcp_server_neither_command_nor_url_raises(self, tmp_path: Path) -> None:
        data = {"mcp_servers": {"bad": {"transport": "stdio"}}}
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data), encoding="utf-8")
        with pytest.raises(ConfigError, match="bad"):
            load_config(tmp_path)

    def test_mcp_server_not_mapping_raises(self, tmp_path: Path) -> None:
        (tmp_path / ".crossby.yml").write_text(
            "mcp_servers:\n  bad: not-a-mapping\n", encoding="utf-8"
        )
        with pytest.raises(ConfigError):
            load_config(tmp_path)

    def test_env_var_in_mcp_server(self, tmp_path: Path) -> None:
        data = {
            "mcp_servers": {
                "github": {
                    "command": "npx",
                    "env": {"TOKEN": "${GITHUB_TOKEN}"},
                }
            }
        }
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data), encoding="utf-8")
        config = load_config(tmp_path)
        assert config.mcp_servers["github"].env["TOKEN"] == "${GITHUB_TOKEN}"

    def test_disabled_server_in_config(self, tmp_path: Path) -> None:
        data = {
            "mcp_servers": {
                "old": {"command": "npx", "enabled": False},
            }
        }
        (tmp_path / ".crossby.yml").write_text(yaml.dump(data), encoding="utf-8")
        config = load_config(tmp_path)
        assert config.mcp_servers["old"].enabled is False
