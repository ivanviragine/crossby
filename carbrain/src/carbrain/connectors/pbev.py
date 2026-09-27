"""INMETRO PBE Veicular: standardized consumption and energy figures per version.

INMETRO publishes the table only as a PDF (9 landscape pages, ~970 rows). Text
extraction scrambles it, so rows are rebuilt from word positions: column boundaries come
from the table's vertical ruling lines and words are grouped into lines by their vertical
position. Fake-bold text (the same glyph drawn twice, slightly offset) is de-duplicated.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterator
from datetime import date, datetime
from typing import Any, ClassVar

import pdfplumber

from carbrain.archive import Snapshot
from carbrain.catalog import record_match
from carbrain.connectors.base import Connector, ParseContext
from carbrain.connectors.official import ParseError
from carbrain.db import utcnow
from carbrain.http import Http
from carbrain.models import Observation
from carbrain.resolve import normalize

N_COLUMNS = 33
PROPULSION = {"COMBUSTAO", "HIBRIDO", "ELETRICO", "PLUG-IN"}
TEXT_COLUMNS = {
    0: "category",
    1: "brand",
    2: "model",
    3: "version",
    4: "engine",
    5: "propulsion",
    6: "transmission",
    7: "air_conditioning",
    8: "steering",
    9: "fuel",
    30: "pbev_class_category",
    31: "pbev_class_overall",
    32: "conpet_seal",
}
# Column -> (fuel, cycle) for km/l figures. Columns 21-23 hold gasoline, or diesel when
# the fuel code is D. Columns 24-25 are the electric mode in km/l-equivalent (EVs and
# plug-in hybrids). Columns 26-27 (plug-in flex, ethanol basis) are not parsed yet.
KM_PER_L = {
    18: ("ethanol", "city"),
    19: ("ethanol", "highway"),
    20: ("ethanol", "combined"),
    21: ("gasoline", "city"),
    22: ("gasoline", "highway"),
    23: ("gasoline", "combined"),
}
KM_PER_LE = {24: ("electric_mode", "city"), 25: ("electric_mode", "highway")}
ENERGY_COL, RANGE_COL = 28, 29
MONTHS_PT = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
}  # fmt: skip


class InmetroPbev(Connector):
    source_id = "inmetro_pbev"
    parser_version = "1"
    uses_catalog = True
    listing = (
        "https://www.gov.br/inmetro/pt-br/assuntos/regulamentacao/avaliacao-da-conformidade/"
        "programa-brasileiro-de-etiquetagem/tabelas-de-eficiencia-energetica/"
        "veiculos-automotivos-pbe-veicular"
    )
    #: Fewer parsed rows than this share of the declared total means the layout changed.
    min_coverage: ClassVar[float] = 0.95

    def discover(self, http: Http) -> list[str]:
        return [self.latest_table(http.get(self.listing).text)]

    @staticmethod
    def latest_table(html: str) -> str:
        """The PDF whose file name carries the highest year.

        File names don't follow a pattern ("mascara-pbev-2026_19_jan-rev01.pdf" holds the
        2026 table updated in August), so the year in the name is the only usable signal.
        """
        links = set(re.findall(r'href="([^"]+?\.pdf/@@download/file)"', html))
        dated: list[tuple[int, str]] = []
        for link in links:
            name = link.rsplit("/", 3)[-3]
            if "pbe" not in name.lower():
                continue
            years = [int(y) for y in re.findall(r"20\d\d", name)]
            if years:
                dated.append((max(years), link))
        if not dated:
            raise ParseError("No PBEV table link on the INMETRO page")
        return max(dated)[1]

    def parse(self, snapshot: Snapshot, ctx: ParseContext) -> Iterator[Observation]:
        with pdfplumber.open(snapshot.path) as pdf:
            header = _page_rows(pdf.pages[0])
            year, updated, declared = _summary(header)
            rows = [r for page in pdf.pages for r in _page_rows(page) if _is_vehicle(r)]
        if declared and len(rows) < self.min_coverage * declared:
            raise ParseError(
                f"Parsed {len(rows)} of {declared} declared rows; the PDF layout may have changed"
            )
        as_of_start, as_of_end = date(year, 1, 1), date(year, 12, 31)
        published = datetime(updated.year, updated.month, updated.day) if updated else None
        seen: Counter[str] = Counter()
        for row in rows:
            attrs = {name: row[i] for i, name in TEXT_COLUMNS.items()}
            key = "|".join(
                normalize(attrs[k])
                for k in ("brand", "model", "version", "engine", "transmission", "fuel")
            )
            seen[key] += 1
            if seen[key] > 1:  # identical labels with different figures exist
                key = f"{key}#{seen[key]}"
            self._store_vehicle(ctx, key, attrs, as_of_start)
            yield from self._observations(row, attrs, key, as_of_start, as_of_end, published)

    def _store_vehicle(
        self, ctx: ParseContext, key: str, attrs: dict[str, str], as_of: date
    ) -> None:
        ctx.conn.execute(
            "INSERT INTO source_vehicle (source_id, external_key, as_of, attributes, fetched_at) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT (source_id, external_key, as_of) "
            "DO UPDATE SET attributes = excluded.attributes, fetched_at = excluded.fetched_at",
            (
                self.source_id,
                key,
                as_of.isoformat(),
                json.dumps(attrs, ensure_ascii=False),
                utcnow(),
            ),
        )
        # Model plus version: INMETRO files the Polo Track as model "POLO", version "TRACK …".
        match = ctx.resolver.resolve_fipe(attrs["brand"], f"{attrs['model']} {attrs['version']}")
        if match.family_id is not None or match.method == "ambiguous":
            record_match(ctx.conn, self.source_id, key, match)

    def _observations(
        self,
        row: list[str],
        attrs: dict[str, str],
        key: str,
        start: date,
        end: date,
        published: datetime | None,
    ) -> Iterator[Observation]:
        base: dict[str, Any] = {
            "source_id": self.source_id,
            "subject_type": "source_vehicle",
            "subject_id": key,
            "as_of_start": start,
            "as_of_end": end,
            "published_at": published,
        }
        diesel = attrs["fuel"].upper().startswith("D")
        for col, (fuel, cycle) in KM_PER_L.items():
            value = _number(row[col])
            if value is not None:
                fuel_name = "diesel" if diesel and fuel == "gasoline" else fuel
                yield Observation(
                    **base,
                    metric="pbev_consumption",
                    dims={"fuel": fuel_name, "cycle": cycle},
                    value=value,
                    unit="km/l",
                )
        for col, (fuel, cycle) in KM_PER_LE.items():
            value = _number(row[col])
            if value is not None:
                yield Observation(
                    **base,
                    metric="pbev_consumption",
                    dims={"fuel": fuel, "cycle": cycle},
                    value=value,
                    unit="km/le",
                )
        energy = _number(row[ENERGY_COL])
        if energy is not None:
            yield Observation(**base, metric="pbev_energy", value=energy, unit="MJ/km")
        rng = _number(row[RANGE_COL])
        if rng is not None:
            yield Observation(**base, metric="pbev_electric_range", value=rng, unit="km")


def _number(cell: str) -> float | None:
    """'9,3' -> 9.3; '\\', 'ND', '' and cut-off values like '27,' -> None."""
    text = cell.replace(" ", "")
    if not re.fullmatch(r"\d+(,\d+)?", text):
        return None
    return float(text.replace(",", "."))


def _is_vehicle(row: list[str]) -> bool:
    return (
        len(row) == N_COLUMNS
        and bool(row[1])
        and normalize(row[1]) != "MARCA"
        and normalize(row[5]) in PROPULSION
    )


def _summary(rows: list[list[str]]) -> tuple[int, date | None, int | None]:
    """Table year, update date and declared number of rows from the first page."""
    text = " ".join(" ".join(r) for r in rows)
    squashed = text.replace(" ", "")
    year_match = re.search(r"TabelaAno(20\d\d)", squashed)
    if year_match is None:
        raise ParseError("No 'Tabela Ano <year>' on the first page")
    declared = _declared_rows(rows)
    updated_match = re.search(r"ATUALIZA[CÇ][AÃ]O(\d{1,2})/([a-z]{3})/(\d{2})", squashed)
    updated = None
    if updated_match:
        day, month, yy = updated_match.groups()
        updated = date(2000 + int(yy), MONTHS_PT[month], int(day))
    return int(year_match.group(1)), updated, declared


def _declared_rows(rows: list[list[str]]) -> int | None:
    """The number in the cell just before 'Modelos/Versões' on the summary page."""
    for row in rows:
        cells = [c.replace(" ", "") for c in row]
        for i, cell in enumerate(cells):
            if normalize(cell).startswith("MODELOS/VERS"):
                before = [c for c in cells[:i] if c]
                if before and before[-1].isdigit():
                    return int(before[-1])
    return None


def _page_rows(page: Any) -> list[list[str]]:
    """Table rows of one page, rebuilt from word positions."""
    page = _dedupe(page)
    xs = sorted(
        {round(e["x0"], 1) for e in page.edges if e["orientation"] == "v" and e["height"] > 20}
    )
    bounds: list[float] = []
    for x in xs:
        if not bounds or x - bounds[-1] > 2:
            bounds.append(x)
    words = sorted(
        page.extract_words(x_tolerance=1.5, y_tolerance=2), key=lambda w: (w["top"], w["x0"])
    )
    lines: list[list[dict[str, Any]]] = []
    line_top: float | None = None
    for w in words:
        if line_top is None or w["top"] - line_top > 2.5:
            lines.append([])
            line_top = w["top"]
        lines[-1].append(w)
    rows = []
    for line in lines:
        cells = [""] * max(len(bounds) - 1, 0)
        for w in sorted(line, key=lambda w: w["x0"]):
            center = (w["x0"] + w["x1"]) / 2
            for i in range(len(bounds) - 1):
                if bounds[i] <= center < bounds[i + 1]:
                    cells[i] = f"{cells[i]} {w['text']}".strip()
                    break
        rows.append(cells)
    return rows


def _dedupe(page: Any) -> Any:
    """Drop glyphs drawn twice within 1 pt (fake bold). Much faster than dedupe_chars()."""
    kept: dict[tuple[str, int], list[float]] = {}
    keep: set[int] = set()
    for c in sorted(page.chars, key=lambda c: (c["top"], c["x0"])):
        slot = kept.setdefault((c["text"], round(c["top"])), [])
        if any(abs(c["x0"] - x) < 1 for x in slot):
            continue
        slot.append(c["x0"])
        keep.add(id(c))
    return page.filter(lambda o: o.get("object_type") != "char" or id(o) in keep)
