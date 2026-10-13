"""Cursor reference portability using native files and the real sync CLI."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
import tomli_w
from typer.testing import CliRunner

from crossby.cli.main import app
from crossby.models.ai import AIToolID
from crossby.sync.ownership import load_ledger
from crossby.sync.report import REPORT_PATH

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "crossby.ai_tools.base.AbstractAITool.detect_installed",
        lambda: [AIToolID.CLAUDE, AIToolID.CODEX, AIToolID.CURSOR],
    )
    monkeypatch.setattr(
        "crossby.sync.mcp_discovery._GLOBAL_CLAUDE_JSON_PATH", tmp_path / "home.json"
    )
    monkeypatch.setattr(
        "crossby.sync.mcp_activation.cursor_disabled_path",
        lambda root: root / "cursor-disabled.json",
    )
    monkeypatch.setenv("CROSSBY_TEST_TOKEN", "fake-secret-must-never-be-evaluated")
    monkeypatch.delenv("CROSSBY_TEST_MISSING", raising=False)


def _write(root: Path, tool: str, servers: dict[str, Any]) -> Path:
    relative = {"claude": ".mcp.json", "cursor": ".cursor/mcp.json", "codex": ".codex/config.toml"}
    path = root / relative[tool]
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        tomli_w.dumps({"mcp_servers": servers})
        if tool == "codex"
        else json.dumps({"mcpServers": servers})
    )
    path.write_text(content)
    return path


def _sync(root: Path, source: str, target: str, *flags: str) -> str:
    result = runner.invoke(
        app, ["sync", "mcp", "--from", source, "--to", target, "--path", str(root), *flags]
    )
    assert result.exit_code == 0, result.output
    assert "fake-secret" not in result.output
    return result.output


def _servers(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())["mcpServers"]  # type: ignore[no-any-return]


def test_claude_cursor_roundtrip_plan_repeat_and_missing_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    servers = {
        "remote": {
            "type": "http",
            "url": "https://example.test/mcp",
            "headers": {
                "Authorization": "Bearer ${CROSSBY_TEST_TOKEN}",
                "X-Multi": "pre-${CROSSBY_TEST_TOKEN}/${CROSSBY_TEST_MISSING}-post",
                "X-Literal": "$TOKEN stays literal",
            },
        },
        "local": {
            "command": "mcp-server",
            "env": {
                "CROSSBY_TEST_TOKEN": "${CROSSBY_TEST_TOKEN}",
                "ALIAS": "prefix-${CROSSBY_TEST_MISSING}-suffix",
                "EMPTY": "",
            },
        },
    }
    source = _write(tmp_path, "claude", servers)
    before = source.read_bytes()
    target = tmp_path / ".cursor/mcp.json"
    for flag in ("--plan", "--dry-run"):
        _sync(tmp_path, "claude", "cursor", flag)
        assert not target.exists()
        assert not (tmp_path / ".crossby/owned.json").exists()
    _sync(tmp_path, "claude", "cursor")
    emitted = _servers(target)
    assert emitted["remote"]["headers"]["Authorization"] == "Bearer ${env:CROSSBY_TEST_TOKEN}"
    assert emitted["remote"]["headers"]["X-Multi"] == (
        "pre-${env:CROSSBY_TEST_TOKEN}/${env:CROSSBY_TEST_MISSING}-post"
    )
    assert emitted["local"]["env"]["ALIAS"] == "prefix-${env:CROSSBY_TEST_MISSING}-suffix"
    assert emitted["remote"]["headers"]["X-Literal"] == "$TOKEN stays literal"
    after = target.read_bytes()
    monkeypatch.delenv("CROSSBY_TEST_TOKEN")
    monkeypatch.setenv("CROSSBY_TEST_MISSING", "different-fake-secret")
    _sync(tmp_path, "claude", "cursor")
    assert target.read_bytes() == after
    assert source.read_bytes() == before
    _sync(tmp_path, "cursor", "claude")
    assert _servers(source) == servers
    assert load_ledger(tmp_path).mcp(AIToolID.CLAUDE) == frozenset()
    assert load_ledger(tmp_path).mcp(AIToolID.CURSOR) == frozenset(servers)
    assert "fake-secret" not in target.read_text() + (tmp_path / REPORT_PATH).read_text()


def test_codex_cursor_codex_native_roundtrip(tmp_path: Path) -> None:
    servers = {
        "remote": {
            "url": "https://example.test/mcp",
            "bearer_token_env_var": "CROSSBY_TEST_TOKEN",
            "env_http_headers": {"X-Key": "CROSSBY_TEST_MISSING"},
            "http_headers": {"X-Literal": "plain"},
        },
        "local": {
            "command": "mcp-server",
            "env_vars": ["CROSSBY_TEST_TOKEN"],
            "env": {"MODE": "literal"},
        },
    }
    source = _write(tmp_path, "codex", servers)
    _sync(tmp_path, "codex", "cursor")
    target = tmp_path / ".cursor/mcp.json"
    assert (
        _servers(target)["remote"]["headers"]["Authorization"] == "Bearer ${env:CROSSBY_TEST_TOKEN}"
    )
    assert _servers(target)["local"]["env"]["CROSSBY_TEST_TOKEN"] == "${env:CROSSBY_TEST_TOKEN}"
    _sync(tmp_path, "cursor", "codex")
    native = tomllib.loads(source.read_text())["mcp_servers"]
    assert native["remote"]["bearer_token_env_var"] == "CROSSBY_TEST_TOKEN"
    assert native["remote"]["env_http_headers"] == {"X-Key": "CROSSBY_TEST_MISSING"}
    assert native["local"]["env_vars"] == ["CROSSBY_TEST_TOKEN"]
    assert native["local"]["env"] == {"MODE": "literal"}
    after = source.read_bytes()
    _sync(tmp_path, "cursor", "codex")
    assert source.read_bytes() == after
    assert load_ledger(tmp_path).mcp(AIToolID.CODEX) == frozenset()


@pytest.mark.parametrize(
    "source,target,value",
    [
        ("claude", "cursor", "${CROSSBY_TEST_TOKEN:-sensitive-default}"),
        ("claude", "cursor", "pre-${CROSSBY_TEST_TOKEN:-sensitive-default}-post"),
        ("claude", "cursor", "${env:CROSSBY_TEST_TOKEN}"),
        ("cursor", "claude", "${CROSSBY_TEST_TOKEN}"),  # Cursor literal, Claude reference
        ("cursor", "codex", "${CROSSBY_TEST_TOKEN}"),
        ("cursor", "claude", "${workspaceFolder}"),
        ("cursor", "claude", "${env:CROSSBY_TEST_TOKEN:-sensitive-default}"),
        ("cursor", "claude", "${env:BAD-NAME}"),
        ("cursor", "claude", "${env:CROSSBY_TEST_TOKEN"),
        ("cursor", "codex", "prefix-${env:CROSSBY_TEST_TOKEN}-suffix"),
        ("cursor", "codex", "Bearer  ${env:CROSSBY_TEST_TOKEN}"),
        ("cursor", "claude", "$${env:CROSSBY_TEST_TOKEN}"),
        ("cursor", "claude", r"\${env:CROSSBY_TEST_TOKEN}"),
    ],
)
def test_refusals_preserve_target_ownership_and_report_without_values(
    tmp_path: Path, source: str, target: str, value: str
) -> None:
    source_path = _write(
        tmp_path,
        source,
        {
            "remote": {
                "type": "http",
                "url": "https://example.test/mcp",
                "headers": {"Authorization": value},
            }
        },
    )
    existing = (
        {"url": "https://old.test/mcp", "bearer_token_env_var": "OLD"}
        if target == "codex"
        else {"type": "http", "url": "https://old.test/mcp", "headers": {"Authorization": "old"}}
    )
    target_path = _write(tmp_path, target, {"remote": existing})
    before, source_before = target_path.read_bytes(), source_path.read_bytes()
    for flags in (("--plan",), ("--dry-run",), (), ()):
        output = _sync(tmp_path, source, target, *flags)
        assert "manual-fix" in output
        assert "sensitive-default" not in output
        assert target_path.read_bytes() == before
        assert source_path.read_bytes() == source_before
        assert load_ledger(tmp_path).mcp(AIToolID(target)) == frozenset()
    report = (tmp_path / REPORT_PATH).read_text()
    assert "Not Added" in report and "manual-fix" in report
    assert "sensitive-default" not in report
    assert "fake-secret" not in report


@pytest.mark.parametrize("field", ["command", "args", "url", "env", "headers"])
def test_native_cursor_fields_claude_roundtrip(tmp_path: Path, field: str) -> None:
    value = "prefix-${env:CROSSBY_TEST_TOKEN}-${env:CROSSBY_TEST_MISSING}-suffix"
    entry: dict[str, Any] = {"command": "mcp-server"}
    if field in {"url", "headers"}:
        entry = {"url": "https://example.test/mcp"}  # native URL-only Cursor declaration
    entry[field] = (
        [value] if field == "args" else {"KEY": value} if field in {"env", "headers"} else value
    )
    path = _write(tmp_path, "cursor", {"native": entry})
    _sync(tmp_path, "cursor", "claude")
    emitted = _servers(tmp_path / ".mcp.json")["native"]
    assert "${env:" not in json.dumps(emitted)
    assert "${CROSSBY_TEST_TOKEN}" in json.dumps(emitted)
    _sync(tmp_path, "claude", "cursor")
    returned = _servers(path)["native"]
    assert returned[field] == entry[field]
    if "url" in returned:
        assert returned["type"] == "http"
    after = path.read_bytes()
    _sync(tmp_path, "claude", "cursor")
    assert path.read_bytes() == after


@pytest.mark.parametrize("field", ["command", "args", "url", "env"])
def test_cursor_codex_unsupported_fields_preserve_existing(tmp_path: Path, field: str) -> None:
    value = "${env:CROSSBY_TEST_TOKEN}"
    entry: dict[str, Any] = {"command": "mcp-server"}
    if field == "url":
        entry = {"url": "https://example.test/mcp"}
    entry[field] = [value] if field == "args" else {"ALIAS": value} if field == "env" else value
    _write(tmp_path, "cursor", {"native": entry})
    target = _write(tmp_path, "codex", {"native": {"command": "existing"}})
    before = target.read_bytes()
    assert "manual-fix" in _sync(tmp_path, "cursor", "codex")
    assert target.read_bytes() == before


def test_cursor_source_disable_revokes_only_owned_even_when_nonportable(tmp_path: Path) -> None:
    source = _write(
        tmp_path,
        "cursor",
        {
            "owned": {
                "command": "owned-server",
                "env": {"CROSSBY_TEST_TOKEN": "${env:CROSSBY_TEST_TOKEN}"},
            },
            "user": {"command": "source-user"},
        },
    )
    target = _write(tmp_path, "claude", {"user": {"command": "user-authored"}})
    _sync(tmp_path, "cursor", "claude")
    assert load_ledger(tmp_path).mcp(AIToolID.CLAUDE) == frozenset({"owned"})
    source.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "owned": {"command": "owned-server", "env": {"KEY": "${workspaceFolder}"}},
                    "user": {"command": "source-user"},
                }
            }
        )
    )
    (tmp_path / "cursor-disabled.json").write_text('["owned","user"]')
    before = source.read_bytes()
    _sync(tmp_path, "cursor", "claude")
    assert set(_servers(target)) == {"user"}
    assert source.read_bytes() == before
    assert load_ledger(tmp_path).mcp(AIToolID.CLAUDE) == frozenset()


def test_cursor_target_disable_and_unrelated_entries_survive(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "claude",
        {
            "remote": {
                "type": "http",
                "url": "https://example.test/mcp",
                "headers": {"Authorization": "Bearer ${CROSSBY_TEST_TOKEN}"},
            }
        },
    )
    target = _write(
        tmp_path,
        "cursor",
        {"remote": {"url": "https://old.test/mcp"}, "unrelated": {"command": "user-server"}},
    )
    sidecar = tmp_path / "cursor-disabled.json"
    sidecar.write_text('["remote"]')
    before = sidecar.read_bytes()
    _sync(tmp_path, "claude", "cursor")
    assert sidecar.read_bytes() == before
    assert _servers(target)["unrelated"] == {"command": "user-server"}
    assert load_ledger(tmp_path).mcp(AIToolID.CURSOR) == frozenset()


def test_wizard_codex_cursor_supported_and_refused_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)
    monkeypatch.setattr(
        "crossby.ai_tools.base.AbstractAITool.detect_installed",
        lambda: [AIToolID.CODEX, AIToolID.CURSOR],
    )
    _write(
        tmp_path,
        "codex",
        {
            "good": {
                "url": "https://example.test/mcp",
                "bearer_token_env_var": "CROSSBY_TEST_TOKEN",
            },
            "bad": {"url": "https://example.test/mcp", "http_headers": {"X-Literal": "${TOKEN}"}},
        },
    )
    result = runner.invoke(app, ["sync", "mcp", "--path", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert set(_servers(tmp_path / ".cursor/mcp.json")) == {"good"}
    assert "manual-fix" in (tmp_path / REPORT_PATH).read_text()
    assert load_ledger(tmp_path).mcp(AIToolID.CURSOR) == frozenset({"good"})


@pytest.mark.parametrize(
    "extra",
    [
        {"envFile": "${workspaceFolder}/.env"},
        {"auth": {"CLIENT_SECRET": "${env:CROSSBY_TEST_TOKEN}"}},
    ],
)
def test_native_cursor_unsupported_fields_stay_in_inventory_and_preserve_targets(
    tmp_path: Path, extra: dict[str, Any]
) -> None:
    from crossby.sync.readers import discover_mcp_names

    source = _write(
        tmp_path,
        "cursor",
        {"bad": {"command": "mcp-server", **extra}, "good": {"command": "portable"}},
    )
    target = _write(tmp_path, "claude", {"bad": {"command": "existing"}})
    source_before = source.read_bytes()
    assert discover_mcp_names(tmp_path) == {"bad", "good"}
    assert "manual-fix" in _sync(tmp_path, "cursor", "claude")
    assert _servers(target) == {"bad": {"command": "existing"}, "good": {"command": "portable"}}
    assert source.read_bytes() == source_before
    assert load_ledger(tmp_path).mcp(AIToolID.CLAUDE) == frozenset({"good"})
