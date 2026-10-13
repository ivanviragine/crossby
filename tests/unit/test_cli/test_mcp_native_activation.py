"""Exercise native activation through production CLI, readers, writers and ledger."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from crossby.cli.main import app
from crossby.models.ai import AIToolID
from crossby.sync.mcp_activation import cursor_disabled_path
from crossby.sync.ownership import load_ledger


@pytest.fixture(autouse=True)
def isolated_native_user_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "crossby.sync.mcp_discovery._GLOBAL_CLAUDE_JSON_PATH", tmp_path / "user/.claude.json"
    )
    monkeypatch.setenv("CURSOR_DATA_DIR", str(tmp_path / "cursor-data"))
    monkeypatch.setattr(
        "crossby.sync.mcp_activation._GLOBAL_CLAUDE_SETTINGS_PATH",
        tmp_path / "user/settings.json",
    )
    monkeypatch.setattr(
        "crossby.ai_tools.base.AbstractAITool.detect_installed",
        lambda: ["claude", "codex", "antigravity-cli", "cursor"],
    )


def source_file(
    root: Path, tool: str, *, disabled: object = False, nonportable: bool = False
) -> Path:
    if tool == "codex":
        path = root / ".codex/config.toml"
        body = '[mcp_servers.prod]\ncommand = "prod-server"\n'
        body += (
            "enabled = "
            + (
                "false"
                if disabled is True
                else "true"
                if disabled is False
                else json.dumps(disabled)
            )
            + "\n"
        )
        if nonportable:
            body += 'cwd = "/native-only"\n'
    else:
        path = root / (".agents/mcp_config.json" if tool == "antigravity-cli" else ".mcp.json")
        entry: dict[str, object] = {"command": "prod-server"}
        if tool == "antigravity-cli":
            entry["disabled"] = disabled
        body = json.dumps({"mcpServers": {"prod": entry}})
        if tool == "claude":
            settings = root / ".claude/settings.json"
            settings.parent.mkdir(exist_ok=True)
            settings.write_text(
                json.dumps({"disabledMcpjsonServers": ["prod"] if disabled else []})
            )
    path.parent.mkdir(exist_ok=True)
    path.write_text(body)
    return path


def sync(root: Path, source: str, *extra: str, target: str = "cursor") -> Result:
    return CliRunner().invoke(
        app, ["sync", "mcp", "--from", source, "--to", target, "--path", str(root), *extra]
    )


@pytest.mark.parametrize("source", ["antigravity-cli", "codex", "claude"])
def test_source_disable_revokes_only_owned_copy(tmp_path: Path, source: str) -> None:
    source_file(tmp_path, source)
    first = sync(tmp_path, source)
    assert first.exit_code == 0, first.output
    target = tmp_path / ".cursor/mcp.json"
    assert "prod" in json.loads(target.read_text())["mcpServers"]
    assert load_ledger(tmp_path).mcp(AIToolID.CURSOR) == {"prod"}
    source_file(tmp_path, source, disabled=True)
    result = sync(tmp_path, source)
    assert result.exit_code == 0, result.output
    assert "prod" not in json.loads(target.read_text())["mcpServers"]
    assert not load_ledger(tmp_path).mcp(AIToolID.CURSOR)


@pytest.mark.parametrize("source", ["antigravity-cli", "codex", "claude"])
def test_fresh_disabled_source_is_not_provisioned(tmp_path: Path, source: str) -> None:
    source_file(tmp_path, source, disabled=True)
    result = sync(tmp_path, source)
    assert result.exit_code == 0, result.output
    target = tmp_path / ".cursor/mcp.json"
    assert not target.exists() or "prod" not in json.loads(target.read_text()).get("mcpServers", {})
    assert not load_ledger(tmp_path).mcp(AIToolID.CURSOR)


@pytest.mark.parametrize("source", ["antigravity-cli", "codex", "claude"])
def test_disabled_source_preserves_manual_same_name_target(tmp_path: Path, source: str) -> None:
    source_file(tmp_path, source, disabled=True)
    target = tmp_path / ".cursor/mcp.json"
    target.parent.mkdir()
    target.write_text(json.dumps({"mcpServers": {"prod": {"command": "hand-authored"}}}))
    before = target.read_bytes()
    result = sync(tmp_path, source)
    assert result.exit_code == 0, result.output
    assert target.read_bytes() == before
    assert not load_ledger(tmp_path).mcp(AIToolID.CURSOR)


def test_nonportable_disabled_codex_server_still_revokes_owned_target(tmp_path: Path) -> None:
    source_file(tmp_path, "codex")
    result = sync(tmp_path, "codex")
    assert result.exit_code == 0, result.output
    source_file(tmp_path, "codex", disabled=True, nonportable=True)
    result = sync(tmp_path, "codex")
    assert result.exit_code == 0, result.output
    assert "prod" not in json.loads((tmp_path / ".cursor/mcp.json").read_text())["mcpServers"]
    assert not load_ledger(tmp_path).mcp(AIToolID.CURSOR)


@pytest.mark.parametrize("source", ["antigravity-cli", "codex"])
@pytest.mark.parametrize("malformed", ["true", 0, None])
def test_malformed_activation_cannot_write_or_revoke(
    tmp_path: Path, source: str, malformed: object
) -> None:
    source_file(tmp_path, source)
    result = sync(tmp_path, source)
    assert result.exit_code == 0, result.output
    target = tmp_path / ".cursor/mcp.json"
    ledger = tmp_path / ".crossby/owned.json"
    before_target, before_ledger = target.read_bytes(), ledger.read_bytes()
    source_file(tmp_path, source, disabled=malformed)
    result = sync(tmp_path, source)
    assert result.exit_code == 1, result.output
    assert target.read_bytes() == before_target
    assert ledger.read_bytes() == before_ledger


@pytest.mark.parametrize("layer", ["settings.json", "settings.local.json", "user"])
def test_claude_rejection_wins_over_approvals_and_enable_all(tmp_path: Path, layer: str) -> None:
    source_file(tmp_path, "claude")
    result = sync(tmp_path, "claude")
    assert result.exit_code == 0, result.output
    settings = tmp_path / ".claude/settings.json"
    settings.write_text(
        json.dumps({"enableAllProjectMcpServers": True, "enabledMcpjsonServers": ["prod"]})
    )
    path = tmp_path / ("user/settings.json" if layer == "user" else ".claude/" + layer)
    path.parent.mkdir(exist_ok=True)
    contents = json.loads(path.read_text()) if path.exists() else {}
    contents["disabledMcpjsonServers"] = ["prod"]
    path.write_text(json.dumps(contents))
    before = path.read_bytes()
    result = sync(tmp_path, "claude")
    assert result.exit_code == 0, result.output
    assert "prod" not in json.loads((tmp_path / ".cursor/mcp.json").read_text())["mcpServers"]
    assert path.read_bytes() == before


def test_claude_project_toggle_read_without_importing_user_definitions(tmp_path: Path) -> None:
    source_file(tmp_path, "claude")
    global_config = tmp_path / "user/.claude.json"
    global_config.parent.mkdir()
    global_config.write_text(
        json.dumps(
            {
                "mcpServers": {"private": {"command": "private-server"}},
                "projects": {str(tmp_path.resolve()): {"disabledMcpServers": ["prod"]}},
            }
        )
    )
    result = sync(tmp_path, "claude")
    assert result.exit_code == 0, result.output
    assert not (tmp_path / ".cursor/mcp.json").exists()
    # Metadata reading is not permission to import definitions from the home file.
    assert not load_ledger(tmp_path).mcp(AIToolID.CURSOR)


def test_claude_other_project_toggle_does_not_disable_this_project(tmp_path: Path) -> None:
    source_file(tmp_path, "claude")
    global_config = tmp_path / "user/.claude.json"
    global_config.parent.mkdir()
    global_config.write_text(
        json.dumps({"projects": {"/other/project": {"disabledMcpServers": ["prod"]}}})
    )
    result = sync(tmp_path, "claude")
    assert result.exit_code == 0, result.output
    assert "prod" in json.loads((tmp_path / ".cursor/mcp.json").read_text())["mcpServers"]


@pytest.mark.parametrize(
    "body",
    [
        "{invalid",
        "null",
        "[]",
        '{"disabledMcpjsonServers":null}',
        '{"disabledMcpjsonServers":["prod",1]}',
    ],
)
@pytest.mark.parametrize("mode", [(), ("--dry-run",), ("--plan",), ("--doctor",)])
def test_malformed_claude_activation_preserves_target_and_ledger(
    tmp_path: Path, body: str, mode: tuple[str, ...]
) -> None:
    source_file(tmp_path, "claude")
    result = sync(tmp_path, "claude")
    assert result.exit_code == 0, result.output
    target, ledger = tmp_path / ".cursor/mcp.json", tmp_path / ".crossby/owned.json"
    before = target.read_bytes(), ledger.read_bytes()
    sidecar = tmp_path / ".claude/settings.local.json"
    sidecar.write_text(body)
    result = sync(tmp_path, "claude", *mode)
    assert result.exit_code == 1, result.output
    assert (target.read_bytes(), ledger.read_bytes()) == before
    assert sidecar.read_text() == body


@pytest.mark.parametrize("initially_disabled", [False, True])
def test_cursor_cli_native_sidecar_controls_provisioning_and_owned_revocation(
    tmp_path: Path, initially_disabled: bool
) -> None:
    source = tmp_path / ".cursor/mcp.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"mcpServers": {"prod": {"command": "prod-server"}}}))
    sidecar = cursor_disabled_path(tmp_path)
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps(["prod"] if initially_disabled else []))
    result = sync(tmp_path, "cursor", target="claude")
    assert result.exit_code == 0, result.output
    target = tmp_path / ".mcp.json"
    if not initially_disabled:
        assert "prod" in json.loads(target.read_text())["mcpServers"]
        assert load_ledger(tmp_path).mcp(AIToolID.CLAUDE) == {"prod"}
        sidecar.write_text('["prod"]')
        result = sync(tmp_path, "cursor", target="claude")
        assert result.exit_code == 0, result.output
    assert not target.exists() or "prod" not in json.loads(target.read_text())["mcpServers"]
    assert not load_ledger(tmp_path).mcp(AIToolID.CLAUDE)
    assert json.loads(sidecar.read_text()) == ["prod"]


@pytest.mark.parametrize("body", ["{invalid", "null", "{}", '["prod",1]'])
def test_cursor_malformed_sidecar_does_not_authorize_writes(tmp_path: Path, body: str) -> None:
    source = tmp_path / ".cursor/mcp.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"mcpServers": {"prod": {"command": "prod-server"}}}))
    sidecar = cursor_disabled_path(tmp_path)
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(body)
    result = sync(tmp_path, "cursor", target="claude")
    assert result.exit_code == 1, result.output
    assert not (tmp_path / ".mcp.json").exists()
    assert not (tmp_path / ".crossby/owned.json").exists()


@pytest.mark.parametrize("tool", ["codex", "antigravity-cli"])
def test_refresh_preserves_target_native_user_disable(tmp_path: Path, tool: str) -> None:
    source_file(tmp_path, "claude")
    target = source_file(tmp_path, tool, disabled=True)
    # The incoming connection differs so this exercises the update, not just a no-op.
    target.write_text(target.read_text().replace("prod-server", "old-server"))
    result = sync(tmp_path, "claude", target=tool)
    assert result.exit_code == 0, result.output
    from crossby.sync.readers import discover_mcp

    assert discover_mcp(tmp_path, AIToolID(tool))["prod"].enabled is False
    assert "prod-server" in target.read_text()
    assert not load_ledger(tmp_path).mcp(AIToolID(tool))


@pytest.mark.parametrize("tool", ["codex", "antigravity-cli", "claude"])
def test_scene_disable_survives_cli_sync_and_keeps_its_provenance(
    tmp_path: Path, tool: str
) -> None:
    from crossby.scenes import declare
    from crossby.sync.ownership import save_ledger

    source = source_file(tmp_path, tool)
    result = sync(tmp_path, tool)
    assert result.exit_code == 0, result.output
    ledger = load_ledger(tmp_path)
    activator = {
        "claude": declare.apply_claude_disabled_mcp,
        "codex": declare.apply_codex_disabled_mcp,
        "antigravity-cli": declare.apply_antigravity_disabled_mcp,
    }[tool]
    activated = activator(tmp_path, {"prod"}, ledger)
    assert activated.action != "error"
    save_ledger(tmp_path, ledger)
    scene_before = json.loads((tmp_path / ".crossby/owned.json").read_text())["scene"]
    before_source = source.read_bytes()
    result = sync(tmp_path, tool)
    assert result.exit_code == 0, result.output
    assert source.read_bytes() == before_source
    assert json.loads((tmp_path / ".crossby/owned.json").read_text())["scene"] == scene_before
    assert "prod" not in json.loads((tmp_path / ".cursor/mcp.json").read_text())["mcpServers"]


def test_remote_antigravity_disable_revokes_owned_target(tmp_path: Path) -> None:
    source = source_file(tmp_path, "antigravity-cli")
    result = sync(tmp_path, "antigravity-cli")
    assert result.exit_code == 0, result.output
    source.write_text(
        json.dumps(
            {"mcpServers": {"prod": {"serverUrl": "https://example.test/mcp", "disabled": True}}}
        )
    )
    result = sync(tmp_path, "antigravity-cli")
    assert result.exit_code == 0, result.output
    assert "prod" not in json.loads((tmp_path / ".cursor/mcp.json").read_text())["mcpServers"]


@pytest.mark.parametrize("tool", ["claude", "codex", "antigravity-cli"])
def test_wizard_keeps_disabled_owned_source_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    from crossby.sync.ownership import OwnershipLedger, save_ledger

    source = source_file(tmp_path, tool, disabled=True)
    ledger = OwnershipLedger()
    ledger.record_mcp(AIToolID(tool), {"prod"})
    save_ledger(tmp_path, ledger)
    before = source.read_bytes()
    monkeypatch.setattr("crossby.ui.prompts.confirm", lambda *args, **kwargs: True)
    # Unscoped wizard also dispatches to the winning source tool.
    result = CliRunner().invoke(app, ["sync", "mcp", "--path", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert source.read_bytes() == before
    assert load_ledger(tmp_path).mcp(AIToolID(tool)) == {"prod"}


@pytest.mark.parametrize("mode", [(), ("--dry-run",), ("--plan",), ("--doctor",)])
def test_nonportable_and_ambiguous_activation_never_revokes(
    tmp_path: Path, mode: tuple[str, ...]
) -> None:
    source_file(tmp_path, "codex")
    result = sync(tmp_path, "codex")
    assert result.exit_code == 0, result.output
    target, ledger = tmp_path / ".cursor/mcp.json", tmp_path / ".crossby/owned.json"
    before = target.read_bytes(), ledger.read_bytes()
    source_file(tmp_path, "codex", disabled=0, nonportable=True)
    result = sync(tmp_path, "codex", *mode)
    assert result.exit_code == 1, result.output
    assert (target.read_bytes(), ledger.read_bytes()) == before


def test_cursor_sidecar_uses_git_root_and_data_dir_not_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "Project + name"
    root.mkdir()
    (root / ".git").write_text("gitdir: somewhere")
    nested = root / "nested"
    nested.mkdir()
    monkeypatch.setenv("CURSOR_CONFIG_DIR", str(tmp_path / "config-only"))
    expected = (
        tmp_path
        / "cursor-data/projects"
        / ("-".join(re.findall("[a-zA-Z0-9]+", str(root.resolve()))))
        / "mcp-disabled.json"
    )
    assert cursor_disabled_path(nested) == expected
    source = nested / ".cursor/mcp.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"mcpServers": {"prod": {"command": "prod-server"}}}))
    expected.parent.mkdir(parents=True)
    expected.write_text('["prod"]')
    result = sync(nested, "cursor", target="claude")
    assert result.exit_code == 0, result.output
    assert not (nested / ".mcp.json").exists()


@pytest.mark.parametrize(
    "body",
    ['{"projects":null}', '{"projects":[]}', '{"projects":{"ROOT":{"disabledMcpServers":false}}}'],
)
def test_claude_malformed_project_toggle_blocks_sync(tmp_path: Path, body: str) -> None:
    source_file(tmp_path, "claude")
    global_config = tmp_path / "user/.claude.json"
    global_config.parent.mkdir()
    global_config.write_text(body.replace("ROOT", str(tmp_path.resolve())))
    result = sync(tmp_path, "claude")
    assert result.exit_code == 1, result.output
    assert not (tmp_path / ".cursor/mcp.json").exists()


def test_malformed_activation_in_wizard_is_error_even_without_servers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".mcp.json").write_text("{invalid")
    monkeypatch.setattr("crossby.ui.prompts.confirm", lambda *args, **kwargs: True)
    result = CliRunner().invoke(app, ["sync", "mcp", "--path", str(tmp_path)])
    assert result.exit_code == 1, result.output
    assert not (tmp_path / ".crossby/owned.json").exists()


@pytest.mark.parametrize("source", ["codex", "antigravity-cli"])
def test_disabled_malformed_connection_cannot_authorize_removal(
    tmp_path: Path, source: str
) -> None:
    path = source_file(tmp_path, source)
    result = sync(tmp_path, source)
    assert result.exit_code == 0, result.output
    target, ledger = tmp_path / ".cursor/mcp.json", tmp_path / ".crossby/owned.json"
    before = target.read_bytes(), ledger.read_bytes()
    path = source_file(tmp_path, source, disabled=True)
    path.write_text(path.read_text().replace('"prod-server"', "42"))
    result = sync(tmp_path, source)
    assert result.exit_code == 1, result.output
    assert (target.read_bytes(), ledger.read_bytes()) == before


@pytest.mark.parametrize("target", ["codex", "antigravity-cli", "claude", "cursor"])
def test_malformed_target_activation_cannot_be_erased_or_blessed(
    tmp_path: Path, target: str
) -> None:
    source_tool = "antigravity-cli" if target == "claude" else "claude"
    source_file(tmp_path, source_tool)
    if target in {"codex", "antigravity-cli"}:
        target_path = source_file(tmp_path, target, disabled="false")
    elif target == "claude":
        target_path = tmp_path / ".claude/settings.local.json"
        target_path.parent.mkdir()
        target_path.write_text('{"disabledMcpjsonServers":null}')
    else:
        target_path = cursor_disabled_path(tmp_path)
        target_path.parent.mkdir(parents=True)
        target_path.write_text("null")
    before = target_path.read_bytes()
    result = sync(tmp_path, source_tool, target=target)
    assert result.exit_code == 1, result.output
    assert target_path.read_bytes() == before
    assert not (tmp_path / ".crossby/owned.json").exists()


def test_unaudited_claude_settings_toggle_fields_do_not_acquire_semantics(tmp_path: Path) -> None:
    source_file(tmp_path, "claude")
    settings = tmp_path / ".claude/settings.json"
    # These are project-record fields in ~/.claude.json, not settings fields.
    settings.write_text(json.dumps({"disabledMcpServers": ["prod"], "enabledMcpServers": []}))
    result = sync(tmp_path, "claude")
    assert result.exit_code == 0, result.output
    assert "prod" in json.loads((tmp_path / ".cursor/mcp.json").read_text())["mcpServers"]


@pytest.mark.parametrize("legacy", [False, True])
def test_claude_custom_config_dir_activation_is_not_missed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool
) -> None:
    source_file(tmp_path, "claude")
    custom = tmp_path / "claude-custom"
    custom.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))
    global_config = custom / (".config.json" if legacy else ".claude.json")
    global_config.write_text(
        json.dumps(
            {
                "mcpServers": {"private": {"command": "private-server"}},
                "projects": {str(tmp_path.resolve()): {"disabledMcpServers": ["prod"]}},
            }
        )
    )
    (custom / "settings.json").write_text('{"disabledMcpjsonServers":["private"]}')
    result = sync(tmp_path, "claude")
    assert result.exit_code == 0, result.output
    assert not (tmp_path / ".cursor/mcp.json").exists()
    # Opt-in definitions still come from the same native config home, but
    # project rejection lists do not disable user-scope definitions.
    result = sync(tmp_path, "claude", "--include-user-scope")
    assert result.exit_code == 0, result.output
    assert set(json.loads((tmp_path / ".cursor/mcp.json").read_text())["mcpServers"]) == {"private"}


def test_claude_project_toggle_uses_git_root_for_nested_working_directory(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "nested"
    nested.mkdir()
    source_file(nested, "claude")
    global_config = tmp_path / "user/.claude.json"
    global_config.parent.mkdir()
    global_config.write_text(
        json.dumps({"projects": {str(tmp_path.resolve()): {"disabledMcpServers": ["prod"]}}})
    )
    result = sync(nested, "claude")
    assert result.exit_code == 0, result.output
    assert not (nested / ".cursor/mcp.json").exists()


@pytest.mark.parametrize("losing_source", ["cursor", "codex"])
def test_losing_source_malformed_activation_cannot_authorize_owned_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, losing_source: str
) -> None:
    from crossby.sync.ownership import OwnershipLedger, save_ledger

    source_file(tmp_path, "claude", disabled=True)
    if losing_source == "codex":
        source_file(tmp_path, "codex", disabled=0)
    else:
        cursor = tmp_path / ".cursor/mcp.json"
        cursor.parent.mkdir()
        cursor.write_text('{"mcpServers":{"prod":{"command":"cursor-server"}}}')
        sidecar = cursor_disabled_path(tmp_path)
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text("null")
    target = tmp_path / ".vscode/mcp.json"
    target.parent.mkdir()
    target.write_text('{"servers":{"prod":{"command":"owned-server"}}}')
    ledger = OwnershipLedger()
    ledger.record_mcp(AIToolID.COPILOT, {"prod"})
    save_ledger(tmp_path, ledger)
    ledger_path = tmp_path / ".crossby/owned.json"
    before = target.read_bytes(), ledger_path.read_bytes()
    monkeypatch.setattr(
        "crossby.ai_tools.base.AbstractAITool.detect_installed",
        lambda: ["claude", "copilot", losing_source],
    )
    monkeypatch.setattr("crossby.ui.prompts.confirm", lambda *args, **kwargs: True)
    result = CliRunner().invoke(app, ["sync", "mcp", "--path", str(tmp_path)])
    assert result.exit_code == 1, result.output
    assert (target.read_bytes(), ledger_path.read_bytes()) == before


@pytest.mark.parametrize("transport", [None, "sse"])
def test_cursor_remote_transport_and_native_disable_use_production_flow(
    tmp_path: Path, transport: str | None
) -> None:
    source = tmp_path / ".cursor/mcp.json"
    source.parent.mkdir()
    remote = {"url": "https://example.test/mcp"}
    if transport is not None:
        remote["type"] = transport
    source.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "local": {"command": "local-server"},
                    "remote": remote,
                }
            }
        )
    )
    result = sync(tmp_path, "cursor", target="claude")
    assert result.exit_code == 0, result.output
    target = tmp_path / ".mcp.json"
    servers = json.loads(target.read_text())["mcpServers"]
    assert set(servers) == {"local", "remote"}
    assert servers["remote"]["type"] == (transport or "http")
    assert load_ledger(tmp_path).mcp(AIToolID.CLAUDE) == {"local", "remote"}
    sidecar = cursor_disabled_path(tmp_path)
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text('["remote"]')
    result = sync(tmp_path, "cursor", target="claude")
    assert result.exit_code == 0, result.output
    assert set(json.loads(target.read_text())["mcpServers"]) == {"local"}
    assert load_ledger(tmp_path).mcp(AIToolID.CLAUDE) == {"local"}


def test_fresh_disabled_cursor_url_only_remote_is_not_provisioned(tmp_path: Path) -> None:
    source = tmp_path / ".cursor/mcp.json"
    source.parent.mkdir()
    source.write_text('{"mcpServers":{"remote":{"url":"https://example.test/mcp"}}}')
    sidecar = cursor_disabled_path(tmp_path)
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text('["remote"]')
    result = sync(tmp_path, "cursor", target="claude")
    assert result.exit_code == 0, result.output
    assert not (tmp_path / ".mcp.json").exists()
