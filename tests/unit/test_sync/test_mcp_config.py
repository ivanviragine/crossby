"""Tests for MCPServerConfig model validation."""

from __future__ import annotations

import pytest

from crossby.models.config import MCPServerConfig


class TestMCPServerConfigValidation:
    def test_stdio_requires_command(self) -> None:
        with pytest.raises(ValueError):
            MCPServerConfig()  # no command or url

    def test_both_command_and_url_raises(self) -> None:
        with pytest.raises(ValueError):
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
        with pytest.raises(ValueError):
            MCPServerConfig(command="npx", transport="grpc")  # type: ignore[arg-type]

    def test_disabled(self) -> None:
        s = MCPServerConfig(command="npx", enabled=False)
        assert s.enabled is False

    def test_env_var_reference_preserved(self) -> None:
        s = MCPServerConfig(command="npx", env={"TOKEN": "${MY_TOKEN}"})
        assert s.env["TOKEN"] == "${MY_TOKEN}"
