from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from carbrain.archive import RawArchive
from carbrain.http import FetchError, Http
from carbrain.models import Observation, RightStatus, Use
from carbrain.observations import query_facts, revisions, write_observations
from carbrain.retention import assert_can_send_to_ai, purge_expired, store_text
from carbrain.rights import Registry, RightsError


def _obs(value: float | None, **kw: object) -> Observation:
    base: dict[str, object] = dict(
        source_id="bcb_sgs",
        metric="vehicle_loan_rate_monthly",
        subject_type="national",
        subject_id="BR",
        as_of_start=date(2026, 7, 1),
        as_of_end=date(2026, 7, 31),
        value=value,
        unit="% a.m.",
    )
    base.update(kw)
    return Observation.model_validate(base)


class TestObservations:
    def test_same_value_is_not_rewritten(self, conn: sqlite3.Connection) -> None:
        first = write_observations(conn, [_obs(1.98)], fetched_at="t1", snapshot_id=None)
        again = write_observations(conn, [_obs(1.98)], fetched_at="t2", snapshot_id=None)
        assert (first.inserted, again.inserted, again.unchanged) == (1, 0, 1)

    def test_changed_value_becomes_a_revision(self, conn: sqlite3.Connection) -> None:
        write_observations(conn, [_obs(1.98)], fetched_at="t1", snapshot_id=None)
        stats = write_observations(conn, [_obs(2.01)], fetched_at="t2", snapshot_id=None)
        assert stats.revised == 1
        (fact,) = query_facts(conn, "vehicle_loan_rate_monthly")
        assert fact.value == 2.01 and fact.revision == 2
        assert [f.value for f in revisions(conn, fact)] == [1.98, 2.01]

    def test_dims_are_order_insensitive_and_filterable(self, conn: sqlite3.Connection) -> None:
        a = _obs(4.1, dims={"product": "ETANOL", "uf": "SP"})
        b = _obs(4.1, dims={"uf": "SP", "product": "ETANOL"})
        c = _obs(6.5, dims={"uf": "SP", "product": "GASOLINA"})
        stats = write_observations(conn, [a, b, c], fetched_at="t", snapshot_id=None)
        assert (stats.inserted, stats.unchanged) == (2, 1)
        facts = query_facts(conn, "vehicle_loan_rate_monthly", dims={"product": "GASOLINA"})
        assert [f.value for f in facts] == [6.5]

    def test_latest_only(self, conn: sqlite3.Connection) -> None:
        june = _obs(1.97, as_of_start=date(2026, 6, 1), as_of_end=date(2026, 6, 30))
        write_observations(conn, [june, _obs(1.98)], fetched_at="t", snapshot_id=None)
        latest = query_facts(conn, "vehicle_loan_rate_monthly", latest_only=True)
        assert [f.as_of_start for f in latest] == [date(2026, 7, 1)]

    def test_period_must_be_ordered(self) -> None:
        with pytest.raises(ValueError):
            _obs(1.0, as_of_start=date(2026, 7, 31), as_of_end=date(2026, 7, 1))


class TestArchive:
    def test_same_content_is_parsed_once(self, archive: RawArchive) -> None:
        s1 = archive.store_bytes("anp_weekly", "u", b"abc", None, "v1")
        assert s1.needs_parse
        archive.mark_parsed(s1, "v1")
        s2 = archive.store_bytes("anp_weekly", "u", b"abc", None, "v1")
        assert s2.id == s1.id and not s2.needs_parse

    def test_new_parser_version_reparses(self, archive: RawArchive) -> None:
        s1 = archive.store_bytes("anp_weekly", "u", b"abc", None, "v1")
        archive.mark_parsed(s1, "v1")
        assert archive.store_bytes("anp_weekly", "u", b"abc", None, "v2").needs_parse

    def test_changed_content_is_a_new_snapshot(self, archive: RawArchive) -> None:
        s1 = archive.store_bytes("anp_weekly", "u", b"abc", None, "v1")
        s2 = archive.store_bytes("anp_weekly", "u", b"abd", None, "v1")
        assert s1.id != s2.id and s2.path.read_bytes() == b"abd"


class TestRights:
    def test_open_data_is_allowed(self, registry: Registry) -> None:
        assert registry.require("senatran_fleet", Use.COLLECT).status is RightStatus.ALLOWED

    def test_fipe_is_blocked_without_contract(self, registry: Registry) -> None:
        with pytest.raises(RightsError, match="pending"):
            registry.require("fipe", Use.COLLECT)
        assert not registry.allows("fipe", Use.DISPLAY_EXCERPTS)

    def test_dev_sample_is_capped_per_day(
        self, registry: Registry, conn: sqlite3.Connection
    ) -> None:
        decision = registry.require("fipe", Use.COLLECT, dev_sample=True, conn=conn)
        assert decision.dev_sample
        conn.execute(
            "INSERT INTO run_log (source_id, started_at, status, requests) VALUES (?, ?, 'ok', ?)",
            ("fipe", datetime.now(UTC).isoformat(), 20),
        )
        with pytest.raises(RightsError, match="limit"):
            registry.require("fipe", Use.COLLECT, dev_sample=True, conn=conn)

    def test_dev_sample_never_unlocks_display(self, registry: Registry) -> None:
        with pytest.raises(RightsError):
            registry.require("fipe", Use.DISPLAY_EXCERPTS, dev_sample=True)

    def test_conditional_returns_the_condition(self, registry: Registry) -> None:
        decision = registry.require("anfavea", Use.PUBLISH_DERIVED)
        assert decision.status is RightStatus.CONDITIONAL
        assert decision.condition and "ANFAVEA" in decision.condition

    def test_unknown_source(self, registry: Registry) -> None:
        with pytest.raises(RightsError, match="Unknown source"):
            registry.require("webmotors_scrape", Use.COLLECT)

    def test_every_source_declares_every_use(self, registry: Registry) -> None:
        for spec in registry:
            for use in Use:
                assert isinstance(spec.rights.status(use), RightStatus)


class TestRetention:
    def test_youtube_comment_text_expires_after_30_days(
        self, conn: sqlite3.Connection, registry: Registry
    ) -> None:
        t0 = datetime(2026, 9, 1, tzinfo=UTC)
        store_text(conn, registry, source_id="youtube", external_id="c1", text="oi", now=t0)
        assert purge_expired(conn, now=t0 + timedelta(days=29)) == 0
        assert purge_expired(conn, now=t0 + timedelta(days=30)) == 1

    def test_pending_source_text_is_refused(
        self, conn: sqlite3.Connection, registry: Registry
    ) -> None:
        with pytest.raises(RightsError):
            store_text(conn, registry, source_id="reclame_aqui", external_id="r1", text="x")

    def test_ai_processing_gate(self, registry: Registry) -> None:
        assert_can_send_to_ai(registry, {"anp_weekly", "senatran_fleet"})
        with pytest.raises(RightsError, match="youtube"):
            assert_can_send_to_ai(registry, {"anp_weekly", "youtube"})


class TestHttp:
    def test_retries_server_errors(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503 if calls["n"] < 3 else 200, text="ok")

        http = Http(transport=httpx.MockTransport(handler), sleep=lambda _s: None, min_interval=0)
        assert http.get("https://x.test/a").text == "ok"
        assert calls["n"] == 3

    def test_client_errors_are_not_retried(self) -> None:
        http = Http(
            transport=httpx.MockTransport(lambda r: httpx.Response(400, json={"erro": "x"})),
            sleep=lambda _s: None,
            min_interval=0,
        )
        with pytest.raises(FetchError, match="HTTP 400"):
            http.get("https://x.test/a")
        assert http.requests == 1

    def test_download_resumes_with_range(self, tmp_path: Path) -> None:
        body = bytes(range(256)) * 40  # 10,240 bytes
        seen_ranges: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            rng = request.headers.get("Range")
            seen_ranges.append(rng)
            if rng is None:
                # Promise the full length but deliver only part of it (a dropped transfer).
                return httpx.Response(
                    200, headers={"Content-Length": str(len(body))}, content=body[:4000]
                )
            start = int(rng.split("=")[1].rstrip("-"))
            return httpx.Response(
                206,
                headers={"Content-Range": f"bytes {start}-{len(body) - 1}/{len(body)}"},
                content=body[start:],
            )

        http = Http(transport=httpx.MockTransport(handler), sleep=lambda _s: None, min_interval=0)
        dest = http.download("https://x.test/big.zip", tmp_path / "big.zip")
        assert dest.read_bytes() == body
        assert seen_ranges == [None, "bytes=4000-"]


def test_same_bytes_from_another_url_are_parsed(archive: RawArchive) -> None:
    """Parsers read meaning from the URL (e.g. the SGS series code)."""
    s1 = archive.store_bytes("bcb_sgs", "https://x/sgs.25471", b"[]", None, "v1")
    archive.mark_parsed(s1, "v1")
    assert archive.store_bytes("bcb_sgs", "https://x/sgs.20749", b"[]", None, "v1").needs_parse
