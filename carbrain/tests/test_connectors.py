from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, date, datetime

import httpx
import pytest

from carbrain.archive import RawArchive
from carbrain.catalog import open_reviews
from carbrain.connectors.base import Connector, ParseContext, run_connector
from carbrain.connectors.official import Anfavea, AnpWeekly, BcbSgs, IbgeIpca
from carbrain.connectors.vehicles import (
    FipeDevSample,
    SenatranFleet,
    month_from_text,
    parse_brl,
    parse_fipe_value,
)
from carbrain.http import Http
from carbrain.observations import query_facts
from carbrain.resolve import Resolver
from carbrain.rights import Registry, RightsError

from .conftest import fixture_bytes

Route = Callable[[httpx.Request], httpx.Response]


def http_with(routes: dict[str, bytes | int]) -> Http:
    """Mock HTTP: the first route whose key is a substring of the URL answers."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for key, body in routes.items():
            if key in url:
                if isinstance(body, int):
                    return httpx.Response(body)
                return httpx.Response(200, content=body)
        return httpx.Response(404)

    return Http(transport=httpx.MockTransport(handler), sleep=lambda _s: None, min_interval=0)


def run(
    connector: Connector,
    http: Http,
    seeded: sqlite3.Connection,
    archive: RawArchive,
    registry: Registry,
    resolver: Resolver,
    **kw: bool,
):  # type: ignore[no-untyped-def]
    return run_connector(
        connector,
        conn=seeded,
        http=http,
        archive=archive,
        registry=registry,
        resolver=resolver,
        **kw,
    )


class TestBcb:
    def test_loan_rates(self, seeded, archive, registry, resolver) -> None:  # type: ignore[no-untyped-def]
        http = http_with(
            {
                "sgs.25471": fixture_bytes("bcb_25471.json"),
                "sgs.20749": fixture_bytes("bcb_20749.json"),
            }
        )
        result = run(BcbSgs(today=date(2026, 9, 27)), http, seeded, archive, registry, resolver)
        assert result.status == "ok" and result.files_fetched == 2
        (latest,) = query_facts(seeded, "vehicle_loan_rate_pf_monthly", latest_only=True)
        assert (latest.as_of_start, latest.value, latest.unit) == (date(2026, 7, 1), 1.98, "% a.m.")

    def test_second_run_is_unchanged(self, seeded, archive, registry, resolver) -> None:  # type: ignore[no-untyped-def]
        http = http_with(
            {
                "sgs.25471": fixture_bytes("bcb_25471.json"),
                "sgs.20749": fixture_bytes("bcb_20749.json"),
            }
        )
        connector = BcbSgs(today=date(2026, 9, 27))
        run(connector, http, seeded, archive, registry, resolver)
        again = run(connector, http, seeded, archive, registry, resolver)
        assert again.status == "unchanged" and again.inserted == 0

    def test_api_error_fails_the_run(self, seeded, archive, registry, resolver) -> None:  # type: ignore[no-untyped-def]
        http = http_with({"sgs.": fixture_bytes("bcb_error.json")})
        result = run(BcbSgs(), http, seeded, archive, registry, resolver)
        assert result.status == "error" and "quantidade" in (result.error or "")
        row = seeded.execute("SELECT status, error FROM run_log").fetchone()
        assert row["status"] == "error"


def test_ipca(seeded, archive, registry, resolver) -> None:  # type: ignore[no-untyped-def]
    http = http_with({"sidra": fixture_bytes("sidra_ipca_1737.json")})
    run(IbgeIpca(), http, seeded, archive, registry, resolver)
    facts = query_facts(seeded, "ipca_index")
    assert len(facts) == 30
    assert (facts[-1].as_of_start, facts[-1].value) == (date(2026, 8, 1), 7633.23)
    assert facts[-2].value == 7657.73  # July 2026


class TestAnp:
    def test_listing_keeps_page_order_and_odd_names(self) -> None:
        links = AnpWeekly.links_from_listing(fixture_bytes("anp_listing.html").decode())
        assert links[0].endswith("resumo_semanal_lpc_2026-09-20_2026-09-26.xlsx")
        assert links[-1].endswith("resumo_semanal_lpc_2026-08-30-2026-09-5.xlsx")
        assert len(links) == len(set(links)) == 4
        assert not any("revendas" in link for link in links)

    def test_weekly_prices(self, seeded, archive, registry, resolver) -> None:  # type: ignore[no-untyped-def]
        http = http_with(
            {
                "ultimas-semanas": fixture_bytes("anp_listing.html"),
                "resumo_semanal": fixture_bytes("anp_resumo_semanal.xlsx"),
            }
        )
        result = run(AnpWeekly(weeks=1), http, seeded, archive, registry, resolver)
        assert result.status == "ok", result.error
        week = (date(2026, 9, 20), date(2026, 9, 26))
        brasil = query_facts(
            seeded, "fuel_price_retail_avg", subject_id="BR", dims={"product": "ETANOL HIDRATADO"}
        )
        assert [(f.as_of_start, f.as_of_end) for f in brasil] == [week]
        assert brasil[0].value == 4.1 and brasil[0].unit == "R$/l"
        curitiba = query_facts(seeded, "fuel_price_retail_avg", subject_id="PR/CURITIBA")
        assert curitiba and all(f.subject_type == "municipio" for f in curitiba)
        sp = query_facts(seeded, "fuel_price_retail_avg", subject_type="uf", subject_id="SP")
        assert sp, "state rows are keyed by UF code"
        stations = query_facts(seeded, "fuel_stations_surveyed", subject_id="BR")
        assert all(f.unit == "stations" for f in stations)


class TestAnfavea:
    def test_registrations_by_segment(self, seeded, archive, registry, resolver) -> None:  # type: ignore[no-untyped-def]
        http = http_with({"siteautoveiculos2025": fixture_bytes("anfavea_2025.xlsx")})
        result = run(Anfavea(years=[2025]), http, seeded, archive, registry, resolver)
        assert result.status == "ok", result.error
        jan = date(2025, 1, 1)

        def value(segment: str, origin: str) -> float | None:
            facts = query_facts(
                seeded, "registrations", subject_id=segment, dims={"origin": origin}
            )
            return next(f.value for f in facts if f.as_of_start == jan)

        assert value("total", "total") == 171248
        assert value("veiculos_leves/automoveis", "domestic") == 95887
        assert value("veiculos_leves/automoveis", "imported") == 27520
        assert value("caminhoes/leves", "total") == 772  # 'Leves' under trucks, not light vehicles
        assert result.inserted == 12 * 3 * 11

    def test_unpublished_year_is_skipped(self, seeded, archive, registry, resolver) -> None:  # type: ignore[no-untyped-def]
        http = http_with({"siteautoveiculos2025": fixture_bytes("anfavea_2025.xlsx")})
        result = run(Anfavea(years=[2025, 2027]), http, seeded, archive, registry, resolver)
        assert result.status == "ok" and result.files_fetched == 1


class TestSenatran:
    def test_picks_latest_month(self) -> None:
        import json

        package = json.loads(fixture_bytes("ckan_renavam_package.json"))
        (url,) = SenatranFleet.latest_resources(package, months=1)
        assert url.endswith("marca_e_modelo_ano_julho_2026.zip")
        assert len(SenatranFleet.latest_resources(package, months=5)) == 3

    def test_fleet_by_family(self, seeded, archive, registry, resolver) -> None:  # type: ignore[no-untyped-def]
        http = http_with(
            {
                "package_show": fixture_bytes("ckan_renavam_package.json"),
                "julho_2026.zip": fixture_bytes("senatran_fleet_sample.zip"),
            }
        )
        result = run(SenatranFleet(), http, seeded, archive, registry, resolver)
        assert result.status == "ok", result.error
        july = date(2026, 7, 1)
        mini = query_facts(seeded, "fleet_registered", subject_id="byd-dolphin-mini")
        assert mini and all(f.as_of_start == july for f in mini)
        # Imported (I/BYD) and local (BYD/) labels land in the same family.
        labels = {
            r["external_key"]
            for r in seeded.execute(
                "SELECT external_key FROM external_mapping WHERE target_id = 'byd-dolphin-mini'"
            )
        }
        assert labels == {"I/BYD DOLPHIN MINI GS5EV", "BYD/DOLPHIN MINI GS5EV"}
        # SW4 rows are not counted as Hilux.
        hilux_labels = {
            r["external_key"]
            for r in seeded.execute(
                "SELECT external_key FROM external_mapping WHERE target_id = 'toyota-hilux'"
            )
        }
        assert hilux_labels == {"I/TOYOTA HILUX CD4X4 SRV"}
        # Every label, tracked or not, is inventoried for mapping review.
        inv = {r["label"] for r in seeded.execute("SELECT label FROM label_inventory")}
        assert {"HONDA/CG 160 FAN", "I/TOYOTA HILUX SWSRXA4FD"} <= inv
        dims = mini[0].dims
        assert set(dims) == {"uf", "manufacture_year"} and dims["uf"] in {"SP", "PR"}

    def test_near_misses_go_to_review(self, seeded, resolver) -> None:  # type: ignore[no-untyped-def]
        per_label = {"VW/TCROS HL TSI": 5000.0, "VW/GOLF GTI": 9000.0, "VW/TCROS X": 10.0}
        SenatranFleet()._queue_near_misses(
            ParseContext(seeded, resolver), per_label, dict.fromkeys(per_label)
        )
        queued = {(r.external_key, r.candidate_id) for r in open_reviews(seeded)}
        # Golf is a known lookalike; the small label is below the volume threshold.
        assert queued == {("VW/TCROS HL TSI", "volkswagen-t-cross")}


class TestFipe:
    def test_parse_value(self, seeded, resolver) -> None:  # type: ignore[no-untyped-def]
        import json

        payload = json.loads(fixture_bytes("fipe_value.json"))
        (obs,) = parse_fipe_value(payload, ParseContext(seeded, resolver))
        assert (obs.subject_id, obs.value, obs.price_type) == (
            "005540-9",
            74594.0,
            "fipe_reference",
        )
        assert obs.dims == {"model_year": 2025, "fuel": "F"}
        assert (obs.as_of_start, obs.as_of_end) == (date(2026, 9, 1), date(2026, 9, 30))
        mapped = seeded.execute("SELECT target_id FROM external_mapping WHERE source_id='fipe'")
        assert mapped.fetchone()["target_id"] == "volkswagen-polo-track"

    def test_blocked_without_dev_sample(self, seeded, archive, registry, resolver) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(RightsError):
            run(
                FipeDevSample(resolver, "volkswagen-polo-track", "59"),
                http_with({}),
                seeded,
                archive,
                registry,
                resolver,
            )

    def test_dev_sample_respects_daily_budget(self, seeded, archive, registry, resolver) -> None:  # type: ignore[no-untyped-def]
        seeded.execute(
            "INSERT INTO run_log (source_id, started_at, status, requests) "
            "VALUES ('fipe', ?, 'ok', 19)",
            (datetime.now(UTC).isoformat(),),
        )
        http = http_with(
            {
                "/anos/": fixture_bytes("fipe_value.json"),
                "/anos": b'[{"codigo": "2025-5", "nome": "2025 Flex"}]',
                "/modelos": fixture_bytes("fipe_vw_modelos.json"),
            }
        )
        result = run(
            FipeDevSample(resolver, "volkswagen-polo-track", "59"),
            http,
            seeded,
            archive,
            registry,
            resolver,
            dev_sample=True,
        )
        assert result.status == "error" and "budget" in (result.error or "")
        assert result.requests == 1

    def test_dev_sample_collects(self, seeded, archive, registry, resolver) -> None:  # type: ignore[no-untyped-def]
        http = http_with(
            {
                "/anos/": fixture_bytes("fipe_value.json"),
                "/anos": b'[{"codigo": "2025-5", "nome": "2025 Flex"}]',
                "/modelos": fixture_bytes("fipe_vw_modelos.json"),
            }
        )
        result = run(
            FipeDevSample(resolver, "volkswagen-polo-track", "59", max_models=1),
            http,
            seeded,
            archive,
            registry,
            resolver,
            dev_sample=True,
        )
        assert result.status == "ok", result.error
        assert result.requests == 3
        assert query_facts(seeded, "fipe_reference_price")[0].value == 74594.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("i_frota_por_uf_municipio_marca_e_modelo_ano_julho_2026.zip", (2026, 7)),
        ("setembro de 2026 ", (2026, 9)),
        ("frota_marco_2025.zip", (2025, 3)),
        ("frota_março_2025.zip", (2025, 3)),
    ],
)
def test_month_from_text(text: str, expected: tuple[int, int]) -> None:
    assert month_from_text(text) == expected


def test_parse_brl() -> None:
    assert parse_brl("R$ 74.594,00") == 74594.0
    assert parse_brl("R$ 1.234.567,89") == 1234567.89


def test_anfavea_zero_filled_future_months_are_not_facts() -> None:
    """The 2026 file stores unpublished months as 0 (found in a live run)."""
    months = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    rows = [
        (None, "Emplacamento de autoveículos importados", None),
        (None, None, "Unidades", 2026),
        (None, None, None, *months, "Total Ano"),
        (None, "Total", None, 10, 12, 11, 9, 10, 10, 13, 0, 0, 0, 0, 0, 75),
        (None, "Ônibus", None, 0, 5, 0, 1, 0, 2, 3, 0, 0, 0, 0, 0, 11),
        (None, "Fonte: Renavam", None),
    ]
    obs = list(Anfavea().parse_rows(rows))
    total = [o for o in obs if o.subject_id == "total"]
    assert [o.as_of_start.month for o in total] == [1, 2, 3, 4, 5, 6, 7]
    buses = {o.as_of_start.month: o.value for o in obs if o.subject_id == "onibus"}
    assert buses[1] == 0 and 8 not in buses  # a real zero is kept; unpublished is not


def test_fleet_rows_without_state_are_kept_as_unknown(seeded, archive, registry, resolver) -> None:  # type: ignore[no-untyped-def]
    http = http_with(
        {
            "package_show": fixture_bytes("ckan_renavam_package.json"),
            "julho_2026.zip": fixture_bytes("senatran_fleet_sample.zip"),
        }
    )
    result = run(SenatranFleet(), http, seeded, archive, registry, resolver)
    assert result.status == "ok", result.error
    unknown = query_facts(
        seeded, "fleet_registered", subject_id="byd-dolphin-mini", dims={"uf": "XX"}
    )
    assert [f.value for f in unknown] == [3.0]


def test_monthly_fleet_file_is_not_downloaded_twice(seeded, archive, registry, resolver) -> None:  # type: ignore[no-untyped-def]
    http = http_with(
        {
            "package_show": fixture_bytes("ckan_renavam_package.json"),
            "julho_2026.zip": fixture_bytes("senatran_fleet_sample.zip"),
        }
    )
    first = run(SenatranFleet(), http, seeded, archive, registry, resolver)
    second = run(SenatranFleet(), http, seeded, archive, registry, resolver)
    assert first.requests == 2  # package listing + file
    assert second.requests == 1 and second.status == "unchanged"  # listing only
