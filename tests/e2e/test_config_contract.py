"""Deterministic contract tests for `crossby config show`."""

from __future__ import annotations

import pytest
import yaml
from tests.e2e._support import run_crossby

pytestmark = [pytest.mark.contract, pytest.mark.e2e_deterministic]


def test_config_show_reads_parent_config_and_displays_sections(e2e_context) -> None:
    config = {
        "version": 1,
        "ai": {
            "default_tool": "claude",
            "default_model": "claude-sonnet-4.6",
            "effort": "medium",
            "commands": {
                "review": {
                    "tool": "copilot",
                    "effort": "high",
                    "yolo": True,
                }
            },
        },
        "permissions": {"allowed_commands": ["git:*"]},
    }
    (e2e_context.project / ".crossby.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    nested = e2e_context.project / "src" / "nested"
    nested.mkdir(parents=True)

    result = run_crossby(["config", "show"], cwd=nested, env=e2e_context.env)

    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    compact = combined.replace("\n", "")
    assert "CROSSBY Configuration" in combined
    assert str(e2e_context.project / ".crossby.yml") in compact
    assert "Default tool" in combined
    assert "claude" in combined
    assert "review" in combined
    assert "tool=copilot" in combined
    assert "git:*" in combined


def test_config_show_without_config_exits_cleanly(e2e_context) -> None:
    result = run_crossby(["config", "show"], cwd=e2e_context.project, env=e2e_context.env)

    assert result.returncode == 1
    assert "No .crossby.yml found" in result.stderr
