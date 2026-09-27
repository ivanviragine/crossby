"""The CLI end to end, offline (no network commands)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from carbrain.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CARBRAIN_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    return tmp_path


def test_init_loads_catalogs() -> None:
    result = runner.invoke(app, ["init"])
    assert "21 vehicle families" in result.output and "publishers" in result.output


def test_sources_grouped_by_type() -> None:
    result = runner.invoke(app, ["sources"])
    assert result.exit_code == 0
    for heading in ("official_statistics", "specialist_media", "social_platform"):
        assert heading in result.output
    assert "fipe" in result.output and "nothing yet" in result.output


def test_catalog_list_handles_channels_known_only_by_id() -> None:
    result = runner.invoke(app, ["catalog", "list", "--kind", "specialist_media"])
    assert result.exit_code == 0, result.output
    assert "channel id UChUvS75BR7ziAr7WhrecVsA" in result.output  # AutoPapo on YouTube


def test_tool_command() -> None:
    result = runner.invoke(app, ["tool", "find_vehicle", '{"text": "hrv 2023"}'])
    assert result.exit_code == 0
    assert json.loads(result.output)["data"]["matches"][0]["family_id"] == "honda-hr-v"


def test_bad_tool_input_is_a_usage_error() -> None:
    result = runner.invoke(app, ["tool", "ownership_cost", '{"purchase_price": -1}'])
    assert result.exit_code != 0


def test_status_on_empty_database() -> None:
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0 and "Open mapping reviews: 0" in result.output
