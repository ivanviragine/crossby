"""Native-file MCP regressions through the real sync CLI and readers."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
import structlog
import tomli_w
from typer.testing import CliRunner

from crossby.cli.main import app
from crossby.models.ai import AIToolID
from crossby.sync.ownership import load_ledger
from crossby.sync.readers import discover_mcp
from crossby.sync.report import REPORT_PATH

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    # CLI initialization caches loggers; keep reader capture tests independent
    # of this module's CLI/discovery order.
    monkeypatch.setattr("crossby.sync.readers.logger", structlog.get_logger())
    monkeypatch.setattr(
        "crossby.ai_tools.base.AbstractAITool.detect_installed",
        lambda: [AIToolID.CLAUDE, AIToolID.CODEX],
    )
    # If discovery resolved values, these would leak into files or reports.
    monkeypatch.setenv("API_TOKEN", "secret-that-must-stay-in-the-environment")
    monkeypatch.setenv("TENANT", "tenant-that-must-stay-in-the-environment")


def _codex(root: Path, servers: dict[str, Any]) -> Path:
    path = root / ".codex/config.toml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(tomli_w.dumps({"mcp_servers": servers}), encoding="utf-8")
    return path


def _sync(root: Path, source: str, target: str, *flags: str) -> str:
    result = runner.invoke(
        app, ["sync", "mcp", "--from", source, "--to", target, "--path", str(root), *flags]
    )
    assert result.exit_code == 0, result.output
    return result.output


def test_native_url_and_authentication(tmp_path: Path) -> None:
    source = _codex(
        tmp_path,
        {
            "remote": {
                "url": "https://example.test/mcp",
                "bearer_token_env_var": "API_TOKEN",
                "http_headers": {"X-Version": "v1", "X-Empty": ""},
                "env_http_headers": {"X-Tenant": "TENANT"},
            }
        },
    )
    before = source.read_bytes()
    _sync(tmp_path, "codex", "claude")
    expected = {
        "type": "http",
        "url": "https://example.test/mcp",
        "headers": {
            "Authorization": "Bearer ${API_TOKEN}",
            "X-Tenant": "${TENANT}",
            "X-Version": "v1",
            "X-Empty": "",
        },
    }
    path = tmp_path / ".mcp.json"
    assert json.loads(path.read_text())["mcpServers"]["remote"] == expected
    assert source.read_bytes() == before
    after = path.read_bytes()
    _sync(tmp_path, "codex", "claude")
    assert path.read_bytes() == after


def test_cli_round_trip_preserves_preexisting_target_auth_and_ownership(tmp_path: Path) -> None:
    servers = {
        "remote": {
            "type": "http",
            "url": "https://example.test/mcp",
            "headers": {
                "Authorization": "Bearer ${API_TOKEN}",
                "X-Tenant": "${TENANT}",
                "X-Version": "v1",
            },
        },
        "stdio": {
            "command": "mcp-server",
            "args": ["--quiet"],
            "env": {"API_TOKEN": "${API_TOKEN}", "MODE": "test", "EMPTY": ""},
        },
        "untouched": {"command": "unrelated-server"},
    }
    target = tmp_path / ".mcp.json"
    target.write_text(json.dumps({"mcpServers": servers, "custom": True}), encoding="utf-8")
    _sync(tmp_path, "claude", "codex")
    native = tomllib.loads((tmp_path / ".codex/config.toml").read_text())["mcp_servers"]
    assert native["remote"]["bearer_token_env_var"] == "API_TOKEN"
    assert native["remote"]["env_http_headers"] == {"X-Tenant": "TENANT"}
    assert native["stdio"]["env_vars"] == ["API_TOKEN"]
    _sync(tmp_path, "codex", "claude")
    assert json.loads(target.read_text()) == {"mcpServers": servers, "custom": True}
    assert load_ledger(tmp_path).mcp(AIToolID.CLAUDE) == frozenset()
    assert load_ledger(tmp_path).mcp(AIToolID.CODEX) == frozenset(servers)


@pytest.mark.parametrize(
    "env_var", ["API_TOKEN", {"name": "API_TOKEN"}, {"name": "API_TOKEN", "source": "local"}]
)
def test_native_stdio_passthrough_and_explicit_env_precedence(
    tmp_path: Path, env_var: str | dict[str, str]
) -> None:
    _codex(
        tmp_path,
        {
            "stdio": {
                "command": "mcp-server",
                "env_vars": [env_var, "OVERRIDE"],
                "env": {"OVERRIDE": "literal", "EMPTY": ""},
            }
        },
    )
    _sync(tmp_path, "codex", "claude")
    entry = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["stdio"]
    assert entry["env"] == {"API_TOKEN": "${API_TOKEN}", "OVERRIDE": "literal", "EMPTY": ""}


@pytest.mark.parametrize(
    "fields",
    [
        {"http_headers_helper": "sensitive-command-value"},
        {"http_headers": {"X-Literal": "${API_TOKEN}"}},
        {
            "http_headers": {"Authorization": "Basic sensitive-auth-value"},
            "bearer_token_env_var": "API_TOKEN",
        },
        {
            "http_headers": {"X-Tenant": "sensitive-fallback-value"},
            "env_http_headers": {"x-tenant": "TENANT"},
        },
        {"bearer_token_env_var": "INVALID-NAME"},
        {"http_headers": {"X-Key": 123}},
        {"auth": "chatgpt"},
        {"oauth": {"client_id": "sensitive-client-value"}},
        {"future_auth": "sensitive-unknown-value"},
        {"url": "https://${HOST}/mcp"},
    ],
)
def test_unsupported_native_http_preserves_target_and_reports_without_values(
    tmp_path: Path, fields: dict[str, Any]
) -> None:
    _codex(tmp_path, {"remote": {"url": "https://example.test/mcp", **fields}})
    target = tmp_path / ".mcp.json"
    target.write_text(
        '{"mcpServers":{"remote":{"type":"http","url":"https://old.test/mcp",'
        '"headers":{"Authorization":"Bearer ${EXISTING}"}}}}',
        encoding="utf-8",
    )
    before = target.read_bytes()
    output = _sync(tmp_path, "codex", "claude")
    assert target.read_bytes() == before
    report = (tmp_path / REPORT_PATH).read_text()
    assert "manual-fix" in report
    assert "remote" in report
    assert "Not Added" in report
    for text in (output, report):
        assert "sensitive-" not in text
        assert "secret-that" not in text
        assert "tenant-that" not in text
    assert load_ledger(tmp_path).mcp(AIToolID.CLAUDE) == frozenset()


@pytest.mark.parametrize(
    "fields",
    [
        {"env_vars": [{"name": "API_TOKEN", "source": "remote"}]},
        {"env": {"KEY": "${OTHER}"}},
        {"env_vars": ["BAD-NAME"]},
        {"env_vars": "API_TOKEN"},
        {"cwd": "/sensitive-path-value"},
        {"experimental_environment": "remote"},
        {"args": ["${API_TOKEN}"]},
    ],
)
def test_unsupported_stdio_is_not_approximated(tmp_path: Path, fields: dict[str, Any]) -> None:
    _codex(tmp_path, {"stdio": {"command": "mcp-server", **fields}})
    _sync(tmp_path, "codex", "claude")
    target = tmp_path / ".mcp.json"
    if target.exists():
        assert "stdio" not in json.loads(target.read_text())["mcpServers"]
    assert "manual-fix" in (tmp_path / REPORT_PATH).read_text()


def test_scoped_codex_discovery_does_not_lose_to_claude(tmp_path: Path) -> None:
    (tmp_path / ".mcp.json").write_text(
        '{"mcpServers":{"remote":{"command":"claude-command"}}}', encoding="utf-8"
    )
    _codex(tmp_path, {"remote": {"url": "https://codex.test/mcp"}})
    assert discover_mcp(tmp_path)["remote"].command == "claude-command"
    assert discover_mcp(tmp_path, AIToolID.CODEX)["remote"].url == "https://codex.test/mcp"


def test_dry_run_native_auth_does_not_write(tmp_path: Path) -> None:
    _codex(
        tmp_path,
        {"remote": {"url": "https://example.test/mcp", "bearer_token_env_var": "API_TOKEN"}},
    )
    _sync(tmp_path, "codex", "claude", "--dry-run")
    assert not (tmp_path / ".mcp.json").exists()


def test_literal_basic_and_env_backed_authorization(tmp_path: Path) -> None:
    _codex(
        tmp_path,
        {
            "basic": {
                "url": "https://basic.test/mcp",
                "http_headers": {"authorization": "Basic abc=="},
            },
            "env": {
                "url": "https://env.test/mcp",
                "env_http_headers": {"Authorization": "API_TOKEN"},
            },
        },
    )
    _sync(tmp_path, "codex", "claude")
    servers = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]
    assert servers["basic"]["headers"] == {"authorization": "Basic abc=="}
    assert servers["env"]["headers"] == {"Authorization": "${API_TOKEN}"}


def test_unsupported_source_is_scoped_and_supported_sibling_still_syncs(tmp_path: Path) -> None:
    _codex(
        tmp_path,
        {
            "bad": {"url": "https://example.test/mcp", "http_headers_helper": "sensitive-value"},
            "good": {"command": "good-server", "env_vars": ["API_TOKEN"]},
        },
    )
    _sync(tmp_path, "codex", "claude")
    servers = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]
    assert set(servers) == {"good"}
    assert "bad" in (tmp_path / REPORT_PATH).read_text()
    # Codex's unsupported sibling must not be reported when Claude is selected.
    _sync(tmp_path, "claude", "codex")
    assert "manual-fix" not in (tmp_path / REPORT_PATH).read_text()


def test_unsupported_source_is_visible_in_plan(tmp_path: Path) -> None:
    _codex(
        tmp_path,
        {"bad": {"command": "mcp-server", "env_vars": [{"name": "KEY", "source": "remote"}]}},
    )
    output = _sync(tmp_path, "codex", "claude", "--plan")
    assert "bad" in output
    assert "manual-fix" in output
    assert not (tmp_path / ".mcp.json").exists()


@pytest.mark.parametrize("unsupported", [False, True])
def test_native_codex_wizard_reports_and_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsupported: bool
) -> None:
    monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)
    fields: dict[str, Any] = {
        "url": "https://example.test/mcp",
        "bearer_token_env_var": "API_TOKEN",
    }
    if unsupported:
        fields["http_headers_helper"] = "sensitive-command-value"
    source = _codex(tmp_path, {"remote": fields})
    before = source.read_bytes()
    result = runner.invoke(app, ["sync", "mcp", "--path", str(tmp_path)])
    assert result.exit_code == 0, result.output
    target = tmp_path / ".mcp.json"
    if unsupported:
        assert source.read_bytes() == before
        assert not target.exists()
        assert "manual-fix" in (tmp_path / REPORT_PATH).read_text()
        assert "sensitive-command-value" not in result.output
    else:
        assert json.loads(target.read_text())["mcpServers"]["remote"]["headers"] == {
            "Authorization": "Bearer ${API_TOKEN}"
        }
    assert load_ledger(tmp_path).mcp(AIToolID.CODEX) == frozenset()


def test_native_url_only_and_crossby_sse_metadata(tmp_path: Path) -> None:
    _codex(
        tmp_path,
        {
            "native": {"url": "https://native.test/mcp"},
            "generated": {"url": "https://generated.test/sse", "transport": "sse"},
        },
    )
    _sync(tmp_path, "codex", "claude")
    servers = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]
    assert servers["native"] == {"type": "http", "url": "https://native.test/mcp"}
    assert servers["generated"] == {"type": "sse", "url": "https://generated.test/sse"}


@pytest.mark.parametrize(
    ("target", "relative_path", "key"),
    [
        ("cursor", ".cursor/mcp.json", "mcpServers"),
        ("copilot", ".vscode/mcp.json", "servers"),
        ("antigravity-cli", ".agents/mcp_config.json", "mcpServers"),
    ],
)
def test_reference_targets_preserve_native_auth_and_sync_literal_siblings(
    tmp_path: Path, target: str, relative_path: str, key: str
) -> None:
    _codex(
        tmp_path,
        {
            "remote": {"url": "https://example.test/mcp", "bearer_token_env_var": "API_TOKEN"},
            "stdio": {"command": "mcp-server", "env_vars": ["API_TOKEN"]},
            "literal": {"command": "literal-server", "env": {"MODE": "literal"}},
        },
    )
    path = tmp_path / relative_path
    path.parent.mkdir()
    existing = {"url": "https://old.test/mcp", "headers": {"Authorization": "Bearer ${env:OLD}"}}
    path.write_text(json.dumps({key: {"remote": existing}}), encoding="utf-8")
    _sync(tmp_path, "codex", target)
    servers = json.loads(path.read_text())[key]
    assert servers["remote"] == existing
    assert "stdio" not in servers
    assert servers["literal"]["env"] == {"MODE": "literal"}
    assert load_ledger(tmp_path).mcp(AIToolID(target)) == frozenset({"literal"})
    report = (tmp_path / REPORT_PATH).read_text()
    assert "manual-fix" in report
    assert "Not Added" in report
    assert "secret-that" not in report


def test_wizard_preserves_codex_references_for_unsupported_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)
    monkeypatch.setattr(
        "crossby.ai_tools.base.AbstractAITool.detect_installed",
        lambda: [AIToolID.CODEX, AIToolID.CURSOR],
    )
    _codex(
        tmp_path,
        {"remote": {"url": "https://example.test/mcp", "bearer_token_env_var": "API_TOKEN"}},
    )
    result = runner.invoke(app, ["sync", "mcp", "--path", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / ".cursor/mcp.json").exists()
    assert "manual-fix" in (tmp_path / REPORT_PATH).read_text()
    assert load_ledger(tmp_path).mcp(AIToolID.CURSOR) == frozenset()
