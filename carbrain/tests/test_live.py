"""Smoke tests against the real sources. Run with: uv run pytest -m live -s"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pytest

from carbrain.archive import RawArchive
from carbrain.connectors.base import Connector, run_connector
from carbrain.connectors.official import Anfavea, AnpWeekly, BcbSgs, IbgeIpca
from carbrain.connectors.vehicles import SenatranFleet
from carbrain.http import Http
from carbrain.observations import query_facts
from carbrain.resolve import Resolver
from carbrain.rights import Registry

pytestmark = pytest.mark.live


def _run(
    connector: Connector,
    seeded: sqlite3.Connection,
    archive: RawArchive,
    registry: Registry,
    resolver: Resolver,
) -> None:
    http = Http(min_interval=0.5)
    result = run_connector(
        connector, conn=seeded, http=http, archive=archive, registry=registry, resolver=resolver
    )
    print(result)
    assert result.status == "ok", result.error


def test_bcb(seeded, archive, registry, resolver) -> None:  # type: ignore[no-untyped-def]
    _run(BcbSgs(), seeded, archive, registry, resolver)
    (latest,) = query_facts(seeded, "vehicle_loan_rate_pf_monthly", latest_only=True)
    assert latest.as_of_start > date.today() - timedelta(days=150)
    assert 0.5 < (latest.value or 0) < 5


def test_ipca(seeded, archive, registry, resolver) -> None:  # type: ignore[no-untyped-def]
    _run(IbgeIpca(), seeded, archive, registry, resolver)
    facts = query_facts(seeded, "ipca_index")
    assert len(facts) > 300 and facts[0].as_of_start.year <= 1980


def test_anp(seeded, archive, registry, resolver) -> None:  # type: ignore[no-untyped-def]
    _run(AnpWeekly(weeks=2), seeded, archive, registry, resolver)
    facts = query_facts(
        seeded,
        "fuel_price_retail_avg",
        subject_id="BR",
        dims={"product": "GASOLINA COMUM"},
    )
    assert len(facts) == 2 and all(3 < (f.value or 0) < 12 for f in facts)
    assert (
        len(query_facts(seeded, "fuel_price_retail_avg", subject_type="uf", latest_only=True)) > 100
    )


def test_anfavea(seeded, archive, registry, resolver) -> None:  # type: ignore[no-untyped-def]
    _run(Anfavea(), seeded, archive, registry, resolver)
    facts = query_facts(seeded, "registrations", subject_id="total", dims={"origin": "total"})
    assert facts and all((f.value or 0) > 50_000 for f in facts)


def test_senatran(seeded, archive, registry, resolver) -> None:  # type: ignore[no-untyped-def]
    _run(SenatranFleet(), seeded, archive, registry, resolver)
    rows = seeded.execute(
        "SELECT subject_id, SUM(value) AS n FROM observation_current "
        "WHERE metric = 'fleet_registered' GROUP BY subject_id ORDER BY n DESC"
    ).fetchall()
    totals = {r["subject_id"]: r["n"] for r in rows}
    print(totals)
    assert len(totals) == len(resolver.families), set(resolver.families) - set(totals)
    assert totals["volkswagen-gol"] > 1_000_000
    inventory = seeded.execute("SELECT COUNT(*) FROM label_inventory").fetchone()[0]
    assert inventory > 30_000


def test_publisher_feeds(seeded, registry, resolver) -> None:  # type: ignore[no-untyped-def]
    from carbrain.config import PACKAGE_DATA
    from carbrain.content.catalog import load_publishers, sync_catalog
    from carbrain.content.runner import sync_content

    sync_catalog(seeded, load_publishers(PACKAGE_DATA / "publishers.yaml"))
    (result,) = sync_content(seeded, registry, resolver, Http(min_interval=0.5), platforms=["rss"])
    print(result)
    assert result.status == "ok" and result.items_new >= 20
