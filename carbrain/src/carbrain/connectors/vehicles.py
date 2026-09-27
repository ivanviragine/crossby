"""Vehicle-level sources: SENATRAN registered fleet and FIPE reference prices."""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from datetime import date
from typing import Any

from carbrain.archive import Snapshot
from carbrain.catalog import enqueue_review, record_match
from carbrain.connectors.base import Connector, ParseContext
from carbrain.connectors.official import ParseError, month_period
from carbrain.geo import uf_code_or_unknown
from carbrain.http import Http
from carbrain.models import Observation, PriceType
from carbrain.resolve import Resolver, normalize

MONTHS = {
    "janeiro": 1,
    "fevereiro": 2,
    "marco": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}


def month_from_text(text: str) -> tuple[int, int]:
    """'..._julho_2026.zip' or 'setembro de 2026' -> (2026, 7|9)."""
    norm = normalize(text).lower()
    for name, number in MONTHS.items():
        found = re.search(rf"{name}\D{{1,4}}(\d{{4}})", norm)
        if found:
            return int(found.group(1)), number
    raise ParseError(f"No Portuguese month and year in {text!r}")


# --- SENATRAN: registered fleet by brand/model label ------------------------------------

# Unmatched labels with at least this many vehicles and a close fuzzy name go to review.
REVIEW_MIN_VEHICLES = 1000


class SenatranFleet(Connector):
    """Registered fleet by state, municipality, brand/model label and year of manufacture.

    The monthly file is a 136 MB zip holding a 1.2 GB Latin-1 text file, so it is
    downloaded to disk (resumable) and streamed. Only tracked families are stored as
    observations; every label is kept in `label_inventory` with its national total, which
    is what the mapping review works from.
    """

    source_id = "senatran_fleet"
    parser_version = "1"
    large_files = True
    immutable_urls = True
    package_api = (
        "https://dados.transportes.gov.br/api/3/action/package_show"
        "?id=registro-nacional-de-veiculos-automotores-renavam"
    )

    def __init__(self, months: int = 1) -> None:
        self.months = months

    def discover(self, http: Http) -> list[str]:
        return self.latest_resources(http.get(self.package_api).json(), self.months)

    @staticmethod
    def latest_resources(package: dict[str, Any], months: int) -> list[str]:
        """URLs of the brand/model fleet files for the latest `months`, oldest first."""
        dated: list[tuple[tuple[int, int], str]] = []
        for res in package["result"]["resources"]:
            url = res.get("url") or ""
            if "marca_e_modelo" not in url:
                continue
            try:
                dated.append((month_from_text(url.rsplit("/", 1)[-1]), url))
            except ParseError:
                continue
        dated.sort()
        return [url for _, url in dated[-months:]]

    def parse(self, snapshot: Snapshot, ctx: ParseContext) -> Iterator[Observation]:
        year, month = month_from_text(snapshot.url.rsplit("/", 1)[-1])
        as_of_start, as_of_end = month_period(year, month)
        per_family: dict[tuple[str, str, int], float] = defaultdict(float)
        per_label: dict[str, float] = defaultdict(float)
        label_family: dict[str, str | None] = {}

        with zipfile.ZipFile(snapshot.path) as zf:
            names = [n for n in zf.namelist() if n.upper().endswith((".TXT", ".CSV"))]
            if len(names) != 1:
                raise ParseError(f"Expected one text file in the zip, found {zf.namelist()}")
            encoding = _detect_encoding(zf, names[0])
            with zf.open(names[0]) as raw:
                lines = io.TextIOWrapper(raw, encoding=encoding, newline="")
                header = next(lines).rstrip("\r\n").split(";")
                if len(header) != 5 or not header[2].lower().startswith("marca"):
                    raise ParseError(f"Unexpected fleet header: {header}")
                for line in lines:
                    parts = line.rstrip("\r\n").split(";")
                    if len(parts) != 5:
                        continue
                    state, _municipio, label, year_text, qty_text = parts
                    qty = float(qty_text)
                    per_label[label] += qty
                    if label not in label_family:
                        label_family[label] = self._resolve(ctx, label)
                    family = label_family[label]
                    if family is None:
                        continue
                    year_made = int(year_text) if year_text.strip().isdigit() else 0
                    per_family[(family, state, year_made)] += qty

        self._store_inventory(ctx, per_label, as_of_end)
        self._queue_near_misses(ctx, per_label, label_family)
        uf_cache: dict[str, str] = {}
        for (family, state, year_made), qty in sorted(per_family.items()):
            if state not in uf_cache:
                uf_cache[state] = uf_code_or_unknown(state)
            yield Observation(
                source_id=self.source_id,
                metric="fleet_registered",
                subject_type="family",
                subject_id=family,
                dims={"uf": uf_cache[state], "manufacture_year": year_made},
                as_of_start=as_of_start,
                as_of_end=as_of_end,
                value=qty,
                unit="vehicles",
            )

    def _resolve(self, ctx: ParseContext, label: str) -> str | None:
        match = ctx.resolver.resolve_registry(label)
        if match.family_id is None and match.method != "ambiguous":
            return None
        return record_match(ctx.conn, self.source_id, label, match)

    def _store_inventory(self, ctx: ParseContext, per_label: dict[str, float], as_of: date) -> None:
        ctx.conn.executemany(
            "INSERT INTO label_inventory (source_id, label, as_of, quantity) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (source_id, label, as_of) DO UPDATE SET quantity = excluded.quantity",
            [(self.source_id, label, as_of.isoformat(), qty) for label, qty in per_label.items()],
        )
        ctx.conn.commit()

    def _queue_near_misses(
        self,
        ctx: ParseContext,
        per_label: dict[str, float],
        label_family: dict[str, str | None],
    ) -> None:
        for label, qty in per_label.items():
            if qty < REVIEW_MIN_VEHICLES or label_family.get(label) is not None:
                continue
            suggestion = ctx.resolver.suggest_for_label(label)
            if suggestion is None:
                continue
            family_id, score = suggestion
            enqueue_review(
                ctx.conn,
                self.source_id,
                label,
                candidate_id=family_id,
                confidence=score,
                reason=f"No pattern matched; name resembles {family_id} ({qty:.0f} vehicles)",
            )
        ctx.conn.commit()


def _detect_encoding(zf: zipfile.ZipFile, name: str) -> str:
    """UTF-8 if the first MiB decodes as UTF-8, else Latin-1.

    The July 2026 file is UTF-8; older government files are often Latin-1. Mojibake such
    as "MunicÃ­pio" means UTF-8 bytes were read as Latin-1.
    """
    with zf.open(name) as raw:
        head = raw.read(1 << 20)
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        # A multi-byte character cut at the 1 MiB boundary is still UTF-8.
        if exc.start < len(head) - 4:
            return "latin-1"
    return "utf-8"


# --- FIPE: reference prices (blocked until a contract exists) ---------------------------


def parse_brl(text: str) -> float:
    """'R$ 74.594,00' -> 74594.0"""
    digits = re.sub(r"[^\d,]", "", text)
    if not digits:
        raise ParseError(f"Not a BRL amount: {text!r}")
    return float(digits.replace(",", "."))


class FipeDevSample(Connector):
    """A handful of FIPE lookups for parser development, never a mirror of the table.

    Collection rights for FIPE are pending, so this only runs with `--dev-sample`, which
    caps requests per day (see `sources.yaml`). It uses a public third-party wrapper of
    the FIPE lookup because FIPE's own endpoint refuses automated requests.
    """

    source_id = "fipe"
    parser_version = "1"
    api = "https://parallelum.com.br/fipe/api/v1/carros"

    def __init__(
        self, resolver: Resolver, family_id: str, brand_code: str, max_models: int = 2
    ) -> None:
        self.resolver = resolver
        self.family_id = family_id
        self.brand_code = brand_code
        self.max_models = max_models

    def discover(self, http: Http) -> list[str]:
        base = f"{self.api}/marcas/{self.brand_code}/modelos"
        brand = self.resolver.families[self.family_id].brand
        urls: list[str] = []
        for model in http.get(base).json()["modelos"]:
            if len(urls) >= self.max_models:
                break
            if self.resolver.resolve_fipe(brand, model["nome"]).family_id != self.family_id:
                continue
            newest = http.get(f"{base}/{model['codigo']}/anos").json()[0]["codigo"]
            urls.append(f"{base}/{model['codigo']}/anos/{newest}")
        return urls

    def parse(self, snapshot: Snapshot, ctx: ParseContext) -> Iterator[Observation]:
        yield from parse_fipe_value(json.loads(snapshot.path.read_bytes()), ctx, self.source_id)


def parse_fipe_value(
    payload: dict[str, Any], ctx: ParseContext, source_id: str = "fipe"
) -> Iterator[Observation]:
    year, month = month_from_text(payload["MesReferencia"])
    start, end = month_period(year, month)
    code = payload["CodigoFipe"]
    match = ctx.resolver.resolve_fipe(payload["Marca"], payload["Modelo"])
    record_match(ctx.conn, source_id, f"{code}|{payload['Modelo']}", match)
    model_year = int(payload["AnoModelo"])
    yield Observation(
        source_id=source_id,
        metric="fipe_reference_price",
        subject_type="fipe_code",
        subject_id=code,
        dims={
            "model_year": "0km" if model_year == 32000 else model_year,
            "fuel": payload.get("SiglaCombustivel", ""),
        },
        as_of_start=start,
        as_of_end=end,
        value=parse_brl(payload["Valor"]),
        unit="BRL",
        price_type=PriceType.FIPE_REFERENCE,
    )
