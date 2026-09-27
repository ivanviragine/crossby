from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from carbrain.archive import RawArchive
from carbrain.catalog import load_catalog, load_events
from carbrain.config import PACKAGE_DATA
from carbrain.db import connect
from carbrain.resolve import Resolver
from carbrain.rights import Registry

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    c = connect(":memory:")
    yield c
    c.close()


@pytest.fixture(scope="session")
def resolver() -> Resolver:
    return Resolver.from_yaml(PACKAGE_DATA / "families.yaml")


@pytest.fixture(scope="session")
def registry() -> Registry:
    return Registry.load(PACKAGE_DATA / "sources.yaml")


@pytest.fixture
def seeded(conn: sqlite3.Connection, resolver: Resolver) -> sqlite3.Connection:
    load_catalog(conn, resolver)
    load_events(conn, PACKAGE_DATA / "events.yaml")
    return conn


@pytest.fixture
def archive(conn: sqlite3.Connection, tmp_path: Path) -> RawArchive:
    return RawArchive(conn, tmp_path / "raw")


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def http_with(routes: dict[str, bytes | int]):  # type: ignore[no-untyped-def]
    """Mock HTTP: the first route whose key is a substring of the URL answers."""
    import httpx

    from carbrain.http import Http

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for key, body in routes.items():
            if key in url:
                if isinstance(body, int):
                    return httpx.Response(body)
                return httpx.Response(200, content=body)
        return httpx.Response(404)

    return Http(transport=httpx.MockTransport(handler), sleep=lambda _s: None, min_interval=0)


@pytest.fixture
def loaded(seeded, archive, registry, resolver):  # type: ignore[no-untyped-def]
    """Database filled by running the real connectors over the real-data fixtures."""
    from datetime import date

    from carbrain.connectors.base import run_connector
    from carbrain.connectors.official import Anfavea, AnpWeekly, BcbSgs, IbgeIpca
    from carbrain.connectors.vehicles import SenatranFleet

    http = http_with(
        {
            "sgs.25471": fixture_bytes("bcb_25471.json"),
            "sgs.20749": fixture_bytes("bcb_20749.json"),
            "sidra": fixture_bytes("sidra_ipca_1737.json"),
            "ultimas-semanas": fixture_bytes("anp_listing.html"),
            "resumo_semanal": fixture_bytes("anp_resumo_semanal.xlsx"),
            "siteautoveiculos2025": fixture_bytes("anfavea_2025.xlsx"),
            "package_show": fixture_bytes("ckan_renavam_package.json"),
            "julho_2026.zip": fixture_bytes("senatran_fleet_sample.zip"),
        }
    )
    for connector in (
        BcbSgs(today=date(2026, 9, 27)),
        IbgeIpca(),
        AnpWeekly(),
        Anfavea(years=[2025]),
        SenatranFleet(),
    ):
        result = run_connector(
            connector,
            conn=seeded,
            http=http,
            archive=archive,
            registry=registry,
            resolver=resolver,
        )
        assert result.status == "ok", (connector.source_id, result.error)
    return seeded
