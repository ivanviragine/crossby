from __future__ import annotations

import json
import sqlite3
from datetime import date

import pytest

from carbrain.models import Observation, PriceType
from carbrain.observations import write_observations
from carbrain.resolve import Resolver
from carbrain.rights import Registry
from carbrain.tools import ToolContext, _pbev_label, call_tool, tool_definitions

TODAY = date(2026, 9, 28)


@pytest.fixture
def ctx(loaded: sqlite3.Connection, registry: Registry, resolver: Resolver) -> ToolContext:
    # Pretend every source was checked yesterday, so nothing is flagged stale.
    loaded.execute("UPDATE run_log SET finished_at = '2026-09-27T12:00:00+00:00'")
    return ToolContext(loaded, registry, resolver, today=TODAY)


def test_find_vehicle(ctx: ToolContext) -> None:
    result = call_tool(ctx, "find_vehicle", {"text": "quanto custa um polo track 2024?"})
    assert result.status == "ok"
    assert result.data["matches"][0]["family_id"] == "volkswagen-polo-track"
    assert result.data["model_year"] == 2024


def test_find_vehicle_unknown(ctx: ToolContext) -> None:
    assert call_tool(ctx, "find_vehicle", {"text": "Ferrari Roma"}).status == "no_data"


def test_find_vehicle_lists_covered_models_of_a_known_brand(ctx: ToolContext) -> None:
    result = call_tool(ctx, "find_vehicle", {"text": "Toyota Yaris 2026"})
    assert result.status == "no_data"
    assert "Toyota models covered: Corolla, Corolla Cross, Hilux." in result.notes[1]


def test_find_vehicle_flags_approximate_matches(ctx: ToolContext) -> None:
    result = call_tool(ctx, "find_vehicle", {"text": "corola 2026"})
    assert result.data["matches"][0]["match"] == "approximate"
    assert any("confirm the vehicle" in n for n in result.notes)


@pytest.mark.parametrize(
    ("attrs", "label"),
    [
        (  # INMETRO repeats the model in the version column for some Corolla rows
            {
                "model": "COROLLA",
                "version": "COROLLA GRS",
                "engine": "2.0-16V",
                "transmission": "CVT",
            },
            "COROLLA GRS 2.0-16V CVT",
        ),
        (
            {"model": "COROLLA", "version": "ALTIS HV", "engine": "1.8 16V", "transmission": "CVT"},
            "COROLLA ALTIS HV 1.8 16V CVT",
        ),
        ({"model": "POLO", "version": "TRACK 1.0 MPI"}, "POLO TRACK 1.0 MPI"),
    ],
)
def test_pbev_version_label(attrs: dict[str, str], label: str) -> None:
    assert _pbev_label(attrs) == label


def test_fuel_prices_for_a_city(ctx: ToolContext) -> None:
    result = call_tool(ctx, "fuel_prices", {"place": "Curitiba, PR"})
    assert result.status == "ok" and result.data["place"] == "PR/CURITIBA"
    assert {"ETANOL HIDRATADO", "GASOLINA COMUM"} <= set(result.data["prices"])
    (cite,) = result.citations
    assert cite.as_of == "2026-09-20/2026-09-26" and "ANP" in cite.attribution


def test_fuel_prices_fall_back_to_state(ctx: ToolContext) -> None:
    result = call_tool(ctx, "fuel_prices", {"place": "Londrina, PR"})
    assert result.data["place"] == "PR"
    assert any("PR/LONDRINA" in n for n in result.notes)


def test_loan_rate(ctx: ToolContext) -> None:
    result = call_tool(ctx, "loan_rate", {})
    assert result.data == {"month": "2026-07", "monthly_pct": 1.98, "annual_pct": 26.52}
    assert result.citations[0].as_of == "2026-07"


def test_deflate_price(ctx: ToolContext) -> None:
    result = call_tool(ctx, "deflate_price", {"value": 50_000, "from_month": "2025-01"})
    assert result.status == "ok" and result.data["to_month"] == "2026-08"
    assert result.data["adjusted_value"] > 50_000
    old = call_tool(ctx, "deflate_price", {"value": 50_000, "from_month": "2010-01"})
    assert old.status == "no_data"


def test_registrations_accept_short_segment_names(ctx: ToolContext) -> None:
    result = call_tool(ctx, "registrations", {"segment": "automoveis", "months": 3})
    assert result.data["segment"] == "veiculos_leves/automoveis"
    assert [p["month"] for p in result.data["series"]] == ["2025-10", "2025-11", "2025-12"]
    assert "Credit ANFAVEA next to every figure." in result.notes


def test_fleet(ctx: ToolContext) -> None:
    result = call_tool(ctx, "fleet", {"family_id": "byd-dolphin-mini"})
    assert result.status == "ok" and result.data["total"] > 0
    assert result.citations[0].as_of == "2026-07"


def test_fipe_is_restricted_even_when_data_exists(ctx: ToolContext) -> None:
    write_observations(
        ctx.conn,
        [
            Observation(
                source_id="fipe",
                metric="fipe_reference_price",
                subject_type="fipe_code",
                subject_id="005540-9",
                dims={"model_year": 2025, "fuel": "F"},
                as_of_start=date(2026, 9, 1),
                as_of_end=date(2026, 9, 30),
                value=74594.0,
                unit="BRL",
                price_type=PriceType.FIPE_REFERENCE,
            )
        ],
        fetched_at="t",
        snapshot_id=None,
    )
    result = call_tool(ctx, "fipe_price", {"fipe_code": "005540-9"})
    assert result.status == "restricted" and result.data is None
    assert "74594" not in result.model_dump_json()


def test_ownership_cost_uses_cited_prices_and_rates(ctx: ToolContext) -> None:
    result = call_tool(
        ctx,
        "ownership_cost",
        {
            "purchase_price": 120_000,
            "years": 4,
            "km_per_year": 18_000,
            "place": "Curitiba, PR",
            "km_per_l_ethanol": 8.0,
            "km_per_l_gasoline": 11.5,
            "down_payment": 40_000,
            "loan_months": 48,
        },
    )
    assert result.status == "ok"
    sources = {c.source_id for c in result.citations}
    assert sources == {"anp_weekly", "bcb_sgs"}
    assert any("generic assumption" in a for a in result.assumptions)
    assert any("1.98% a month" in a for a in result.assumptions)
    assert any("IPVA" in n for n in result.notes)
    totals = [s["total"] for s in result.data["scenarios"]]
    assert totals == sorted(totals)
    assert result.data["loan"]["months"] == 48


def test_ownership_cost_needs_both_flex_consumptions(ctx: ToolContext) -> None:
    result = call_tool(ctx, "ownership_cost", {"purchase_price": 90_000, "km_per_l_ethanol": 8})
    assert result.status == "no_data"


def test_stale_sources_are_flagged(ctx: ToolContext) -> None:
    ctx.conn.execute(
        "UPDATE run_log SET finished_at = '2026-08-01T00:00:00+00:00' "
        "WHERE source_id = 'anp_weekly'"
    )
    result = call_tool(ctx, "fuel_prices", {"place": "SP"})
    assert any("out of date" in n for n in result.notes)


def test_data_freshness(ctx: ToolContext) -> None:
    rows = call_tool(ctx, "data_freshness", {}).data
    by_id = {r["source_id"]: r for r in rows}
    assert by_id["anp_weekly"]["status"] == "fresh"
    assert by_id["anp_weekly"]["latest_period_end"] == "2026-09-26"


def test_events(ctx: ToolContext) -> None:
    result = call_tool(ctx, "events", {"since": "2026-06-01"})
    assert [e["kind"] for e in result.data] == ["tax", "tax"]


def test_tool_definitions_are_self_contained() -> None:
    defs = tool_definitions()
    assert len({d["name"] for d in defs}) == len(defs)
    blob = json.dumps(defs)
    assert "$ref" not in blob and "$defs" not in blob
    for d in defs:
        assert d["description"] and d["input_schema"]["type"] == "object"


def test_bad_calls(ctx: ToolContext) -> None:
    with pytest.raises(ValueError, match="Unknown tool"):
        call_tool(ctx, "nope", {})
    with pytest.raises(ValueError, match="Invalid input"):
        call_tool(ctx, "ownership_cost", {"purchase_price": -1})
