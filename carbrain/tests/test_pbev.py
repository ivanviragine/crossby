"""INMETRO PBEV parser, against the real August 2026 table (9-page PDF)."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pdfplumber
import pytest

from carbrain.archive import Snapshot
from carbrain.catalog import load_catalog
from carbrain.config import PACKAGE_DATA
from carbrain.connectors.base import ParseContext
from carbrain.connectors.official import ParseError
from carbrain.connectors.pbev import InmetroPbev, _number, _page_rows, _summary
from carbrain.db import connect
from carbrain.observations import query_facts, write_observations
from carbrain.resolve import Resolver

from .conftest import FIXTURES, fixture_bytes

PDF = FIXTURES / "pbev_2026.pdf"


def _snapshot() -> Snapshot:
    return Snapshot(1, "inmetro_pbev", "https://x/pbev.pdf", "h", PDF, "t", None, True)


@pytest.fixture(scope="module")
def parsed() -> sqlite3.Connection:
    """Parse the full PDF once for the whole module (~13 s)."""
    conn = connect(":memory:")
    resolver = Resolver.from_yaml(PACKAGE_DATA / "families.yaml")
    load_catalog(conn, resolver)
    obs = InmetroPbev().parse(_snapshot(), ParseContext(conn, resolver))
    write_observations(conn, obs, fetched_at="t", snapshot_id=None)
    return conn


def _facts(conn: sqlite3.Connection, key: str, metric: str = "pbev_consumption") -> dict:  # type: ignore[type-arg]
    return {
        (f.dims.get("fuel"), f.dims.get("cycle")): f.value
        for f in query_facts(conn, metric, subject_id=key)
    }


def test_summary_page() -> None:
    with pdfplumber.open(PDF) as pdf:
        assert _summary(_page_rows(pdf.pages[0])) == (2026, date(2026, 8, 14), 976)


def test_coverage(parsed: sqlite3.Connection) -> None:
    n = parsed.execute("SELECT COUNT(*) FROM source_vehicle").fetchone()[0]
    assert 0.95 * 976 <= n <= 976


def test_flex_row(parsed: sqlite3.Connection) -> None:
    got = _facts(parsed, "HONDA|CITY|LX|1.5-16V|CVT|F")
    assert got[("ethanol", "city")] == 9.3 and got[("ethanol", "combined")] == 9.8
    assert got[("gasoline", "highway")] == 15.5 and got[("gasoline", "combined")] == 13.9


def test_ev_row(parsed: sqlite3.Connection) -> None:
    key = "BYD|DOLPHIN MINI|GS 5 EV|ELETRICO|N.A.|E"
    assert _facts(parsed, key) == {
        ("electric_mode", "city"): 58.6,
        ("electric_mode", "highway"): 41.9,
    }
    assert _facts(parsed, key, "pbev_energy") == {(None, None): 0.41}
    assert _facts(parsed, key, "pbev_electric_range") == {(None, None): 280.0}


def test_diesel_is_labeled_diesel(parsed: sqlite3.Connection) -> None:
    got = _facts(parsed, "TOYOTA|HILUX DIESEL 4X4 AT|CHASSI AT|2.8-16V|A-6|D")
    assert got[("diesel", "combined")] == 9.6 and ("gasoline", "combined") not in got


def test_source_gaps_stay_gaps(parsed: sqlite3.Connection) -> None:
    """The Aug 2026 table prints no km/l for the Polo Track; don't invent one."""
    key = "VW|POLO|TRACK 1.0 MPI|1.0-12V|M - 5|F"
    assert _facts(parsed, key) == {}
    assert _facts(parsed, key, "pbev_energy") == {(None, None): 1.48}


def test_identical_labels_are_kept_apart(parsed: sqlite3.Connection) -> None:
    keys = {
        k
        for (k,) in parsed.execute(
            "SELECT external_key FROM source_vehicle "
            "WHERE external_key LIKE 'CAOA CHERY|TIGGO 7|PHEV%'"
        )
    }
    assert keys == {
        "CAOA CHERY|TIGGO 7|PHEV|1.5T -16V|DHT|G",
        "CAOA CHERY|TIGGO 7|PHEV|1.5T -16V|DHT|G#2",
    }


def test_family_mapping(parsed: sqlite3.Connection) -> None:
    rows = parsed.execute(
        "SELECT target_id, external_key FROM external_mapping WHERE source_id = 'inmetro_pbev'"
    ).fetchall()
    families = {r["target_id"] for r in rows}
    assert len(families) == 19 and "volkswagen-gol" not in families  # Gol is discontinued
    track = {r["external_key"] for r in rows if r["target_id"] == "volkswagen-polo-track"}
    assert track == {
        "VW|POLO|TRACK 1.0 MPI|1.0-12V|M - 5|F",
        "VW|POLO|TRACK ROBUST 1.0 MPI|1.0-12V|M - 5|F",
    }


def test_layout_change_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = connect(":memory:")
    resolver = Resolver.from_yaml(PACKAGE_DATA / "families.yaml")
    monkeypatch.setattr(InmetroPbev, "min_coverage", 1.0)
    with pytest.raises(ParseError, match="layout"):
        list(InmetroPbev().parse(_snapshot(), ParseContext(conn, resolver)))


def test_latest_table_link() -> None:
    link = InmetroPbev.latest_table(fixture_bytes("pbev_listing.html").decode())
    assert "mascara-pbev-2026" in link


@pytest.mark.parametrize(
    ("cell", "value"),
    [
        ("9,3", 9.3),
        ("58,59", 58.59),
        ("280", 280.0),
        ("\\", None),
        ("ND", None),
        ("27,", None),
        ("", None),
    ],
)
def test_number(cell: str, value: float | None) -> None:
    assert _number(cell) == value


def test_fixture_is_the_real_file() -> None:
    assert Path(PDF).stat().st_size > 3_000_000


def test_consumption_tool(parsed: sqlite3.Connection) -> None:
    from carbrain.rights import Registry
    from carbrain.tools import ToolContext, call_tool

    ctx = ToolContext(
        parsed,
        Registry.load(PACKAGE_DATA / "sources.yaml"),
        Resolver.from_yaml(PACKAGE_DATA / "families.yaml"),
    )
    result = call_tool(ctx, "consumption", {"family_id": "byd-dolphin-mini"})
    assert result.status == "ok" and result.citations[0].as_of == "2026"
    gs = next(v for v in result.data["versions"] if "GS 5 EV" in v["version"])
    assert gs["electric_range_km"] == 280.0 and gs["energy_mj_per_km"] == 0.41
    track = call_tool(ctx, "consumption", {"family_id": "volkswagen-polo-track"})
    assert all(v["consumption"] is None for v in track.data["versions"])
    assert any("no km/l" in n for n in track.notes)
