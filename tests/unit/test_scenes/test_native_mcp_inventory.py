"""Native MCP portability must not weaken scene restrictions."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
import tomli_w

from crossby.ai_tools.codex import CodexAdapter
from crossby.models.ai import AIToolID
from crossby.models.config import SceneConfig, SceneSelector
from crossby.scenes.engine import apply_scene, clear_scene
from crossby.scenes.launch import SceneLaunchContext, SceneLaunchFallbackError, codex_profile_name
from crossby.services.scene_resolution import resolve_scene
from crossby.sync.base import SyncConcern
from crossby.sync.readers import build_sync_data, scan_project


def _native(root: Path, fields: dict[str, Any]) -> Path:
    path = root / ".codex/config.toml"
    path.parent.mkdir()
    path.write_text(
        tomli_w.dumps(
            {
                "mcp_servers": {
                    "native": {"command": "mcp-server", **fields},
                    "portable": {"command": "other"},
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _context(root: Path, selected: list[str]) -> SceneLaunchContext:
    scan = scan_project(root, [AIToolID.CODEX])
    resolved = resolve_scene(SceneConfig(mcp=SceneSelector(include=selected)), scan, root)
    return SceneLaunchContext("restricted", resolved, root, build_sync_data(root))


@pytest.mark.parametrize(
    "fields",
    [
        {"startup_timeout_sec": 30},
        {"env": {"KEY": "${LITERAL}"}},
        {
            "env_vars": [{"name": "API_TOKEN", "source": "remote"}],
            "experimental_environment": "remote",
        },
    ],
)
def test_nonportable_native_servers_remain_in_launch_and_persistent_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fields: dict[str, Any]
) -> None:
    source = _native(tmp_path, fields)
    original = tomllib.loads(source.read_text())
    context = _context(tmp_path, ["portable"])
    assert "native" not in context.sync_data.mcp_servers
    assert context.deselected_mcp() == {"native"}
    # Selecting the native-only server still matches inventory, without a false
    # "matched no detected item" warning or a portable approximation.
    selected_native = _context(tmp_path, ["native"])
    assert selected_native.selected("mcp") == {"native"}
    assert not selected_native.resolved.warnings

    home = tmp_path / "codex-home"
    home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(home))
    result = CodexAdapter().scene_launch_args(context)
    assert "--profile" in result.args
    profile = home / f"{codex_profile_name(tmp_path, 'restricted')}.config.toml"
    assert tomllib.loads(profile.read_text())["mcp_servers"] == {"native": {"enabled": False}}
    assert tomllib.loads(source.read_text()) == original

    results = apply_scene(context.resolved, tmp_path, tools=[AIToolID.CODEX])
    assert not any(r.action == "error" for r in results)
    entry = tomllib.loads(source.read_text())["mcp_servers"]["native"]
    assert entry == {**original["mcp_servers"]["native"], "enabled": False}
    assert any(r.concern == SyncConcern.MCP and r.action == "updated" for r in results)
    clear_scene(tmp_path, tools=[AIToolID.CODEX])
    assert tomllib.loads(source.read_text()) == original


def test_strict_config_cannot_silently_drop_selected_native_definition(tmp_path: Path) -> None:
    _native(tmp_path, {"startup_timeout_sec": 30})
    context = _context(tmp_path, ["native"])
    assert context.narrows_mcp()
    with pytest.raises(SceneLaunchFallbackError, match="Selected native MCP servers"):
        context.selected_mcp()


def test_exclude_all_restricts_native_only_server(tmp_path: Path) -> None:
    _native(tmp_path, {"startup_timeout_sec": 30})
    context = _context(tmp_path, [])
    assert context.deselected_mcp() == {"native", "portable"}
    results = apply_scene(context.resolved, tmp_path, tools=[AIToolID.CODEX])
    assert not any(r.action == "error" for r in results)
    servers = tomllib.loads((tmp_path / ".codex/config.toml").read_text())["mcp_servers"]
    assert all(s["enabled"] is False for s in servers.values())


@pytest.mark.parametrize("fields", [{"enabled": False}, {"enabled": False, "cwd": "/native-only"}])
def test_native_disable_survives_scene_selection_and_strict_launch(
    tmp_path: Path, fields: dict[str, Any]
) -> None:
    from crossby.ai_tools.claude import ClaudeAdapter
    from crossby.scenes.launch import mcp_json_config

    source = _native(tmp_path, fields)
    before = source.read_bytes()
    context = _context(tmp_path, ["native", "portable"])
    # Selecting a server is not a native enable operation.
    assert context.selected("mcp") == {"native", "portable"}
    assert context.deselected_mcp() == {"native"}
    assert set(context.selected_mcp()) == {"portable"}
    arguments = ClaudeAdapter().scene_launch_args(context)
    assert "--strict-mcp-config" in arguments.args
    import json

    strict = tmp_path / ".crossby/scene/restricted/launch/mcp.json"
    assert set(json.loads(strict.read_text())["mcpServers"]) == {"portable"}
    assert set(json.loads(mcp_json_config(context.mcp_universe()))["mcpServers"]) == {"portable"}
    assert source.read_bytes() == before


def test_malformed_native_activation_refuses_strict_scene_configuration(tmp_path: Path) -> None:
    _native(tmp_path, {"enabled": "false"})
    context = _context(tmp_path, ["native"])
    with pytest.raises(SceneLaunchFallbackError, match="Invalid native MCP activation"):
        context.selected_mcp()
