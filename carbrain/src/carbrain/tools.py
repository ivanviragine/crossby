"""Tools for the chatbot and the post generator.

Every tool returns a `ToolResult` with a status (`ok`, `no_data`, `restricted`), the data,
citations (source, reference period, when it was checked) and notes. A source's data is
only returned if the rights registry allows showing it; stale sources are flagged.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from carbrain.calc.finance import annual_from_monthly, deflate, loan
from carbrain.calc.fuel import cost_per_km, flex_choice
from carbrain.calc.ownership import Financing, OwnershipInput, Range, ownership_cost
from carbrain.geo import UF_CODES, municipio_id, uf_code
from carbrain.models import Use
from carbrain.observations import Fact, query_facts
from carbrain.resolve import Resolver, normalize
from carbrain.rights import Registry

Status = Literal["ok", "no_data", "restricted"]

#: Generic depreciation assumption used only when the caller gives none. Not specific
#: to any car; replace with FIPE-based curves once FIPE data may be shown.
DEFAULT_DEPRECIATION = Range(low=8, base=12, high=16)


class Citation(BaseModel):
    source_id: str
    source: str
    attribution: str
    as_of: str
    checked_at: str | None = None
    url: str


class ToolResult(BaseModel):
    status: Status
    data: Any = None
    citations: list[Citation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


@dataclass
class ToolContext:
    conn: sqlite3.Connection
    registry: Registry
    resolver: Resolver
    today: date = field(default_factory=date.today)


class _Cite:
    """Collects citations and rights notes while a tool runs."""

    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx
        self.citations: dict[tuple[str, str], Citation] = {}
        self.notes: list[str] = []

    def allowed(self, source_id: str) -> bool:
        decision = self.ctx.registry.check(source_id, Use.DISPLAY_EXCERPTS)
        if not self.ctx.registry.allows(source_id, Use.DISPLAY_EXCERPTS):
            return False
        if decision.condition and decision.condition not in self.notes:
            self.notes.append(decision.condition)
        return True

    def add(self, facts: list[Fact]) -> None:
        for fact in facts:
            spec = self.ctx.registry.get(fact.source_id)
            as_of = _period(fact.as_of_start, fact.as_of_end)
            key = (fact.source_id, as_of)
            if key not in self.citations:
                self.citations[key] = Citation(
                    source_id=fact.source_id,
                    source=spec.name,
                    attribution=spec.attribution,
                    as_of=as_of,
                    checked_at=_last_success(self.ctx.conn, fact.source_id),
                    url=spec.url,
                )
                stale = _stale_note(self.ctx, fact.source_id)
                if stale and stale not in self.notes:
                    self.notes.append(stale)

    def result(self, status: Status, data: Any = None, **kw: Any) -> ToolResult:
        notes = [*kw.pop("notes", []), *self.notes]
        return ToolResult(
            status=status,
            data=data,
            citations=list(self.citations.values()),
            notes=notes,
            **kw,
        )


def _period(start: date, end: date) -> str:
    if (start.month, start.day, end.month, end.day) == (1, 1, 12, 31) and start.year == end.year:
        return str(start.year)
    if start.day == 1 and (end + timedelta(days=1)).day == 1 and start.month == end.month:
        return start.isoformat()[:7]
    return start.isoformat() if start == end else f"{start.isoformat()}/{end.isoformat()}"


def _last_success(conn: sqlite3.Connection, source_id: str) -> str | None:
    row = conn.execute(
        "SELECT MAX(finished_at) FROM run_log "
        "WHERE source_id = ? AND status IN ('ok', 'unchanged')",
        (source_id,),
    ).fetchone()
    return str(row[0]) if row and row[0] else None


def _stale_note(ctx: ToolContext, source_id: str) -> str | None:
    spec = ctx.registry.get(source_id)
    last = _last_success(ctx.conn, source_id)
    if last is None:
        return None
    age = (ctx.today - datetime.fromisoformat(last).date()).days
    if age > spec.freshness_days:
        return (
            f"{spec.name} was last checked {age} days ago (target: every "
            f"{spec.freshness_days} days); figures may be out of date."
        )
    return None


# --- tools --------------------------------------------------------------------------------


class FindVehicleInput(BaseModel):
    text: str = Field(description="What the user wrote, e.g. 'polo track 2024' or 'hrv'.")


def find_vehicle(ctx: ToolContext, inp: FindVehicleInput) -> ToolResult:
    """Identify which vehicle families (and model year, if given) the user means."""
    matches = [m for m in ctx.resolver.find(inp.text) if m.family_id]
    if not matches:
        notes = [
            f"No covered vehicle family matches. The catalog has {len(ctx.resolver.families)}."
        ]
        for brand_id in sorted(ctx.resolver.brands_in(inp.text)):
            names = sorted(f.name for f in ctx.resolver.families.values() if f.brand == brand_id)
            brand = ctx.resolver.brands[brand_id].name
            notes.append(f"{brand} models covered: {', '.join(names)}. Other models are not.")
        return ToolResult(status="no_data", notes=notes)
    year = re.search(r"\b(19[89]\d|20[0-4]\d)\b", inp.text)
    rows = []
    for m in matches:
        fam = ctx.conn.execute(
            "SELECT f.id, b.name AS brand, f.name, f.segment, f.body, f.status, f.powertrains "
            "FROM family f JOIN brand b ON b.id = f.brand_id WHERE f.id = ?",
            (m.family_id,),
        ).fetchone()
        if fam is None:
            continue
        rows.append(
            {
                "family_id": fam["id"],
                "brand": fam["brand"],
                "name": fam["name"],
                "segment": fam["segment"],
                "status": fam["status"],
                "powertrains": json.loads(fam["powertrains"]),
                "confidence": round(m.confidence, 2),
                "match": "approximate" if m.method == "fuzzy" else "name",
            }
        )
    notes = []
    if any(r["match"] == "approximate" for r in rows):
        notes.append(
            "An 'approximate' match comes from a similar spelling: confirm the vehicle with the "
            "user before quoting its data."
        )
    if len(rows) > 1 and rows[0]["confidence"] - rows[1]["confidence"] < 0.05:
        notes.append(
            "The text mentions several families: compare them, or ask which one the user means."
        )
    notes.append(
        "Versions and trims are not in the catalog yet; ask for the version if it matters."
    )
    return ToolResult(
        status="ok",
        data={"matches": rows, "model_year": int(year.group(1)) if year else None},
        notes=notes,
    )


class FleetInput(BaseModel):
    family_id: str = Field(description="Family ID from find_vehicle, e.g. 'byd-dolphin-mini'.")
    uf: str | None = Field(default=None, description="Two-letter state code to filter by.")


def fleet(ctx: ToolContext, inp: FleetInput) -> ToolResult:
    """Registered fleet of a vehicle family: total, by state and by year of manufacture."""
    cite = _Cite(ctx)
    if not cite.allowed("senatran_fleet"):
        return cite.result("restricted")
    dims: dict[str, str | int] = {"uf": uf_code(inp.uf)} if inp.uf else {}
    facts = query_facts(
        ctx.conn, "fleet_registered", subject_id=inp.family_id, dims=dims, latest_only=True
    )
    if not facts:
        return cite.result("no_data")
    cite.add(facts)
    by_uf: dict[str, float] = {}
    by_year: dict[int, float] = {}
    for f in facts:
        by_uf[str(f.dims["uf"])] = by_uf.get(str(f.dims["uf"]), 0) + (f.value or 0)
        year = int(f.dims["manufacture_year"])
        by_year[year] = by_year.get(year, 0) + (f.value or 0)
    unknown_year = by_year.pop(0, 0.0)
    recent: dict[str, float] = {str(y): by_year[y] for y in sorted(by_year)[-8:]}
    if unknown_year:
        recent["unknown"] = unknown_year
    return cite.result(
        "ok",
        {
            "family_id": inp.family_id,
            "total": sum(by_uf.values()),
            "by_state_top10": dict(sorted(by_uf.items(), key=lambda kv: -kv[1])[:10]),
            "by_manufacture_year_recent": recent,
        },
        notes=[
            "Counts vehicles registered in RENAVAM, including some no longer on the road.",
            "State 'XX' means the registry has no state for those vehicles.",
        ],
    )


class RegistrationsInput(BaseModel):
    segment: str = Field(
        default="veiculos_leves/automoveis",
        description=(
            "ANFAVEA segment: total, veiculos_leves, veiculos_leves/automoveis, "
            "veiculos_leves/comerciais_leves, caminhoes, onibus."
        ),
    )
    origin: Literal["total", "domestic", "imported"] = "total"
    months: int = Field(default=12, ge=1, le=60)


def registrations(ctx: ToolContext, inp: RegistrationsInput) -> ToolResult:
    """Monthly new-vehicle registrations by segment (ANFAVEA), with year-on-year change."""
    cite = _Cite(ctx)
    if not cite.allowed("anfavea"):
        return cite.result("restricted")
    candidates = [inp.segment, f"veiculos_leves/{inp.segment}", f"caminhoes/{inp.segment}"]
    for segment in candidates:
        facts = query_facts(
            ctx.conn, "registrations", subject_id=segment, dims={"origin": inp.origin}
        )
        if facts:
            break
    else:
        return cite.result("no_data", notes=[f"No ANFAVEA data for segment '{inp.segment}'."])
    by_month = {f.as_of_start: f for f in facts}
    shown = facts[-inp.months :]
    cite.add(shown)
    series = []
    for f in shown:
        last_year = by_month.get(f.as_of_start.replace(year=f.as_of_start.year - 1))
        yoy = (
            round(((f.value or 0) / last_year.value - 1) * 100, 1)
            if last_year and last_year.value
            else None
        )
        series.append({"month": f.as_of_start.isoformat()[:7], "units": f.value, "yoy_pct": yoy})
    return cite.result(
        "ok",
        {"segment": segment, "origin": inp.origin, "series": series},
        notes=[
            "Registrations (emplacamentos) include direct sales to fleets, rental companies "
            "and PCD buyers; they are not retail sales."
        ],
    )


class FuelPricesInput(BaseModel):
    place: str = Field(
        default="BR",
        description="'BR', a state ('SP' or 'São Paulo') or a city ('Curitiba, PR').",
    )


def fuel_prices(ctx: ToolContext, inp: FuelPricesInput) -> ToolResult:
    """Latest weekly average pump prices by fuel for a city, state or Brazil (ANP)."""
    cite = _Cite(ctx)
    if not cite.allowed("anp_weekly"):
        return cite.result("restricted")
    found = _latest_fuel_facts(ctx, inp.place)
    if found is None:
        return cite.result("no_data", notes=[f"No ANP prices for '{inp.place}'."])
    place_used, facts, notes = found
    cite.add(facts)
    prices = {
        str(f.dims["product"]): {"avg": f.value, "unit": f.unit}
        for f in facts
        if f.metric == "fuel_price_retail_avg"
    }
    return cite.result("ok", {"place": place_used, "prices": prices}, notes=notes)


def _place_chain(place: str) -> list[tuple[str, str]]:
    """(subject_type, subject_id) from most to least specific."""
    text = normalize(place)
    if text in ("", "BR", "BRASIL", "BRAZIL"):
        return [("national", "BR")]
    city, uf = None, None
    if "/" in text:
        uf, city = (p.strip() for p in text.split("/", 1))
    elif "," in text:
        city, uf = (p.strip() for p in text.rsplit(",", 1))
    else:
        uf = text
    try:
        uf = uf_code(uf) if uf else None
    except KeyError:
        uf = None
    chain: list[tuple[str, str]] = []
    if city and uf:
        chain.append(("municipio", municipio_id(uf, city)))
    if uf in UF_CODES:
        chain.append(("uf", uf))
    chain.append(("national", "BR"))
    return chain


def _latest_fuel_facts(ctx: ToolContext, place: str) -> tuple[str, list[Fact], list[str]] | None:
    notes: list[str] = []
    for subject_type, subject_id in _place_chain(place):
        facts = [
            f
            for metric in ("fuel_price_retail_avg",)
            for f in query_facts(
                ctx.conn,
                metric,
                subject_type=subject_type,
                subject_id=subject_id,
                latest_only=True,
            )
        ]
        if facts:
            return subject_id, facts, notes
        notes.append(f"No ANP survey for {subject_id}; using a wider area.")
    return None


class LoanRateInput(BaseModel):
    pass


def loan_rate(ctx: ToolContext, inp: LoanRateInput) -> ToolResult:
    """Latest average interest rate on vehicle loans to individuals (Banco Central)."""
    cite = _Cite(ctx)
    if not cite.allowed("bcb_sgs"):
        return cite.result("restricted")
    monthly = query_facts(ctx.conn, "vehicle_loan_rate_pf_monthly", latest_only=True)
    annual = query_facts(ctx.conn, "vehicle_loan_rate_pf_annual", latest_only=True)
    if not monthly:
        return cite.result("no_data")
    cite.add(monthly + annual)
    return cite.result(
        "ok",
        {
            "month": monthly[0].as_of_start.isoformat()[:7],
            "monthly_pct": monthly[0].value,
            "annual_pct": annual[0].value if annual else None,
        },
        notes=["Market average for new loans; an individual offer can differ a lot."],
    )


class DeflateInput(BaseModel):
    value: float = Field(description="Amount in reais.")
    from_month: str = Field(description="Month the amount refers to, 'YYYY-MM'.")
    to_month: str | None = Field(
        default=None, description="Target month 'YYYY-MM'; default: latest IPCA month."
    )


def deflate_price(ctx: ToolContext, inp: DeflateInput) -> ToolResult:
    """Express an amount from one month in reais of another month, using IPCA."""
    cite = _Cite(ctx)
    if not cite.allowed("ibge_ipca"):
        return cite.result("restricted")
    facts = query_facts(ctx.conn, "ipca_index")
    if not facts:
        return cite.result("no_data")
    index = {f.as_of_start: f.value or 0.0 for f in facts}
    start = _month(inp.from_month)
    end = _month(inp.to_month) if inp.to_month else facts[-1].as_of_start
    try:
        real = deflate(inp.value, start, end, index)
    except KeyError as exc:
        return cite.result("no_data", notes=[str(exc)])
    cite.add([f for f in facts if f.as_of_start in (start, end)])
    return cite.result(
        "ok",
        {
            "value": inp.value,
            "from_month": start.isoformat()[:7],
            "to_month": end.isoformat()[:7],
            "adjusted_value": round(real, 2),
            "inflation_pct": round((index[end] / index[start] - 1) * 100, 2),
        },
    )


def _month(text: str) -> date:
    year, month = (int(p) for p in text.split("-")[:2])
    return date(year, month, 1)


class FlexChoiceInput(BaseModel):
    km_per_l_ethanol: float = Field(gt=0)
    km_per_l_gasoline: float = Field(gt=0)
    place: str = "BR"


def flex_fuel_choice(ctx: ToolContext, inp: FlexChoiceInput) -> ToolResult:
    """Cheaper fuel for a flex car at local prices, using the car's own consumption."""
    prices = fuel_prices(ctx, FuelPricesInput(place=inp.place))
    if prices.status != "ok":
        return prices
    p = prices.data["prices"]
    if "ETANOL HIDRATADO" not in p or "GASOLINA COMUM" not in p:
        return ToolResult(status="no_data", notes=["Ethanol or gasoline price missing."])
    choice = flex_choice(
        inp.km_per_l_ethanol,
        inp.km_per_l_gasoline,
        p["ETANOL HIDRATADO"]["avg"],
        p["GASOLINA COMUM"]["avg"],
    )
    return prices.model_copy(
        update={
            "data": {
                "place": prices.data["place"],
                "best": choice.best,
                "ethanol_cost_per_km": round(choice.ethanol_per_km, 3),
                "gasoline_cost_per_km": round(choice.gasoline_per_km, 3),
                "price_ratio": round(choice.price_ratio, 3),
                "breakeven_ratio": round(choice.breakeven_ratio, 3),
                "saving_per_1000_km": round(choice.saving_per_1000_km, 2),
            },
            "assumptions": [
                "Consumption figures as passed in (from the consumption tool or the user); "
                "cite their source too."
            ],
        }
    )


class OwnershipCostInput(BaseModel):
    purchase_price: float = Field(gt=0, description="Price the user expects to pay, in reais.")
    years: int = Field(default=4, gt=0, le=15)
    km_per_year: float = Field(default=12000, gt=0)
    place: str = Field(default="BR", description="City or state, for fuel prices.")
    fuel: Literal["flex", "gasoline", "ethanol", "diesel"] = "flex"
    km_per_l_ethanol: float | None = Field(default=None, gt=0)
    km_per_l_gasoline: float | None = Field(default=None, gt=0)
    km_per_l_diesel: float | None = Field(default=None, gt=0)
    annual_depreciation_pct: Range | None = None
    insurance_per_year: Range | None = None
    maintenance_per_year: Range | None = None
    ipva_rate_pct: float | None = Field(default=None, ge=0, le=10)
    down_payment: float | None = Field(default=None, ge=0)
    loan_months: int | None = Field(default=None, gt=0, le=120)
    loan_monthly_rate_pct: float | None = Field(default=None, ge=0)


def ownership_cost_tool(ctx: ToolContext, inp: OwnershipCostInput) -> ToolResult:
    """Cost of owning a car for N years: low/base/high scenarios with every assumption."""
    cite = _Cite(ctx)
    assumptions: list[str] = []
    notes: list[str] = []

    prices = fuel_prices(ctx, FuelPricesInput(place=inp.place))
    if prices.status != "ok":
        return prices
    for c in prices.citations:
        cite.citations[(c.source_id, c.as_of)] = c
    notes.extend(prices.notes)
    p = {k: v["avg"] for k, v in prices.data["prices"].items()}

    energy: float
    if inp.fuel == "flex":
        if "ETANOL HIDRATADO" not in p or "GASOLINA COMUM" not in p:
            return ToolResult(status="no_data", notes=["Ethanol or gasoline price missing."])
        if inp.km_per_l_ethanol is None or inp.km_per_l_gasoline is None:
            return ToolResult(
                status="no_data",
                notes=["Flex cars need km/l on ethanol and on gasoline (ask the user or PBEV)."],
            )
        choice = flex_choice(
            inp.km_per_l_ethanol,
            inp.km_per_l_gasoline,
            p["ETANOL HIDRATADO"],
            p["GASOLINA COMUM"],
        )
        energy = min(choice.ethanol_per_km, choice.gasoline_per_km)
        assumptions.append(f"Runs on {choice.best}, the cheaper fuel at current local prices.")
    else:
        product, km_per_l = {
            "gasoline": ("GASOLINA COMUM", inp.km_per_l_gasoline),
            "ethanol": ("ETANOL HIDRATADO", inp.km_per_l_ethanol),
            "diesel": ("OLEO DIESEL S10", inp.km_per_l_diesel),
        }[inp.fuel]
        if km_per_l is None or product not in p:
            return ToolResult(status="no_data", notes=[f"Need km/l and a price for {inp.fuel}."])
        energy = cost_per_km(p[product], km_per_l)
    assumptions.append("Fuel prices stay at the latest weekly ANP average.")

    depreciation = inp.annual_depreciation_pct
    if depreciation is None:
        depreciation = DEFAULT_DEPRECIATION
        assumptions.append(
            "Depreciation of 8% / 12% / 16% a year is a generic assumption, not specific to "
            "this car. Model-specific curves need FIPE data, which can't be shown yet."
        )

    financing = None
    if inp.down_payment is not None and inp.loan_months is not None:
        rate = inp.loan_monthly_rate_pct
        if rate is None:
            bcb = loan_rate(ctx, LoanRateInput())
            if bcb.status != "ok":
                return bcb
            rate = float(bcb.data["monthly_pct"])
            for c in bcb.citations:
                cite.citations[(c.source_id, c.as_of)] = c
            assumptions.append(
                f"Loan at the market average of {rate}% a month "
                f"({annual_from_monthly(rate):.1f}% a year)."
            )
        financing = Financing(
            down_payment=inp.down_payment, monthly_rate_pct=rate, months=inp.loan_months
        )

    result = ownership_cost(
        OwnershipInput(
            purchase_price=inp.purchase_price,
            years=inp.years,
            km_per_year=inp.km_per_year,
            energy_cost_per_km=Range.of(energy),
            annual_depreciation_pct=depreciation,
            insurance_per_year=inp.insurance_per_year,
            maintenance_per_year=inp.maintenance_per_year,
            ipva_rate_pct=inp.ipva_rate_pct,
            financing=financing,
        )
    )
    data: dict[str, Any] = {"scenarios": result.summary(), "excluded": result.excluded}
    if financing is not None:
        plan = loan(
            inp.purchase_price, financing.down_payment, financing.monthly_rate_pct, financing.months
        )
        data["loan"] = {
            "installment": round(plan.installment, 2),
            "months": plan.months,
            "total_interest": round(plan.total_interest, 2),
        }
    if result.excluded:
        notes.append("Not included: " + "; ".join(result.excluded) + ".")
    return cite.result("ok", data, notes=notes, assumptions=assumptions)


class ConsumptionInput(BaseModel):
    family_id: str = Field(description="Family ID from find_vehicle, e.g. 'fiat-strada'.")


def consumption(ctx: ToolContext, inp: ConsumptionInput) -> ToolResult:
    """Standardized consumption per version (INMETRO PBEV): km/l, energy use, EV range."""
    cite = _Cite(ctx)
    if not cite.allowed("inmetro_pbev"):
        return cite.result("restricted")
    keys = [
        r["external_key"]
        for r in ctx.conn.execute(
            "SELECT external_key FROM external_mapping WHERE source_id = 'inmetro_pbev' "
            "AND target_id = ? AND status != 'rejected' ORDER BY external_key",
            (inp.family_id,),
        )
    ]
    versions = []
    for key in keys:
        attrs_row = ctx.conn.execute(
            "SELECT attributes FROM source_vehicle WHERE source_id = 'inmetro_pbev' "
            "AND external_key = ? ORDER BY as_of DESC LIMIT 1",
            (key,),
        ).fetchone()
        facts = [
            f
            for metric in ("pbev_consumption", "pbev_energy", "pbev_electric_range")
            for f in query_facts(ctx.conn, metric, subject_id=key, latest_only=True)
        ]
        cite.add(facts)
        attrs = json.loads(attrs_row["attributes"]) if attrs_row else {}
        km: dict[str, dict[str, float | None]] = {}
        for f in facts:
            if f.metric == "pbev_consumption":
                km.setdefault(f"{f.dims['fuel']} ({f.unit})", {})[str(f.dims["cycle"])] = f.value
        versions.append(
            {
                "version": _pbev_label(attrs),
                "fuel_code": attrs.get("fuel"),
                "consumption": km or None,
                "energy_mj_per_km": next(
                    (f.value for f in facts if f.metric == "pbev_energy"), None
                ),
                "electric_range_km": next(
                    (f.value for f in facts if f.metric == "pbev_electric_range"), None
                ),
                "pbev_class_in_category": attrs.get("pbev_class_category") or None,
            }
        )
    if not versions:
        return cite.result("no_data", notes=["No PBEV rows mapped to this family."])
    notes = [
        "Standardized lab figures, comparable across cars; real-world consumption varies "
        "with driving and conditions.",
        "Fuel codes: F flex, G gasoline, D diesel, E electric, E100 ethanol only.",
    ]
    if any(v["consumption"] is None for v in versions):
        notes.append("Some versions have no km/l in the published table (shown as null).")
    return cite.result("ok", {"family_id": inp.family_id, "versions": versions}, notes=notes)


class InformationSourcesInput(BaseModel):
    kind: Literal["specialist_media", "creator"] | None = None
    focus: str | None = Field(
        default=None,
        description="e.g. reviews, buying_advice, ev, mechanics, used_cars, instrumented_tests",
    )
    platform: Literal["website", "rss", "youtube", "instagram"] | None = None


def information_sources(ctx: ToolContext, inp: InformationSourcesInput) -> ToolResult:
    """Specialist media and creators worth following, filtered by kind, focus or platform."""
    rows = ctx.conn.execute("SELECT * FROM publisher ORDER BY kind DESC, name").fetchall()
    out = []
    for pub in rows:
        focus = json.loads(pub["focus"])
        if (inp.kind and pub["kind"] != inp.kind) or (inp.focus and inp.focus not in focus):
            continue
        channels = [
            {
                "platform": ch["platform"],
                "handle": ch["handle"],
                "url": ch["url"],
                "external_id": ch["external_id"],
                "status": ch["status"],
            }
            for ch in ctx.conn.execute(
                "SELECT platform, handle, url, external_id, status FROM channel "
                "WHERE publisher_id = ? "
                "ORDER BY platform",
                (pub["id"],),
            )
            if not inp.platform or ch["platform"] == inp.platform
        ]
        if inp.platform and not channels:
            continue
        out.append(
            {
                "id": pub["id"],
                "name": pub["name"],
                "kind": pub["kind"],
                "focus": focus,
                "evidence_kind": json.loads(pub["evidence_kind"]),
                "channels": channels,
            }
        )
    if not out:
        return ToolResult(status="no_data")
    return ToolResult(
        status="ok",
        data=out,
        notes=[
            "Handles with status 'web_evidence' come from dated third-party lists and may "
            "have changed; 'api_verified' ones were confirmed through the platform's API.",
            "Weigh 'entertainment' sources as buzz, not as evidence about a car.",
        ],
    )


class ExpertContentInput(BaseModel):
    family_id: str = Field(description="Family ID from find_vehicle.")
    days: int = Field(default=30, ge=1, le=365)
    kind: Literal["specialist_media", "creator"] | None = None


def expert_content(ctx: ToolContext, inp: ExpertContentInput) -> ToolResult:
    """Recent headlines and videos from specialist media and creators about a vehicle."""
    since = (ctx.today - timedelta(days=inp.days)).isoformat()
    rows = ctx.conn.execute(
        "SELECT c.source_id, c.title, c.url, c.published_at, c.kind AS item_kind, "
        "p.name AS publisher, p.kind, p.evidence_kind FROM content_mention m "
        "JOIN content_item c ON c.id = m.content_id JOIN channel ch ON ch.id = c.channel_id "
        "JOIN publisher p ON p.id = ch.publisher_id "
        "WHERE m.family_id = ? AND COALESCE(c.published_at, c.fetched_at) >= ? "
        "ORDER BY COALESCE(c.published_at, c.fetched_at) DESC",
        (inp.family_id, since),
    ).fetchall()
    cite = _Cite(ctx)
    items, hidden = [], 0
    for r in rows:
        if inp.kind and r["kind"] != inp.kind:
            continue
        if not cite.allowed(r["source_id"]):
            hidden += 1
            continue
        items.append(
            {
                "publisher": r["publisher"],
                "kind": r["kind"],
                "evidence_kind": json.loads(r["evidence_kind"]),
                "title": r["title"],
                "url": r["url"],
                "published_at": r["published_at"],
            }
        )
        key = (r["publisher"], r["url"] or r["title"])
        cite.citations[key] = Citation(
            source_id=r["source_id"],
            source=r["publisher"],
            attribution=f"Fonte: {r['publisher']}",
            as_of=(r["published_at"] or "")[:10],
            checked_at=_last_success(ctx.conn, r["source_id"]),
            url=r["url"] or "",
        )
    notes = []
    if hidden:
        notes.append(f"{hidden} item(s) from sources without display rights were left out.")
    if not items:
        return cite.result("no_data", notes=notes)
    # What may be shown (headline and link only) comes from the rights registry's conditions.
    return cite.result("ok", items[:20], notes=notes)


def _pbev_label(attrs: dict[str, str]) -> str:
    """'COROLLA GRS 2.0-16V CVT'. INMETRO sometimes repeats the model in the version
    column ('COROLLA' + 'COROLLA GRS'), so the model is not written twice."""
    model, version = attrs.get("model", ""), attrs.get("version", "")
    head = version if version == model or version.startswith(f"{model} ") else f"{model} {version}"
    parts = (head, attrs.get("engine", ""), attrs.get("transmission", ""))
    return " ".join(p for p in parts if p).strip()


class FipePriceInput(BaseModel):
    fipe_code: str
    model_year: int | None = None


def fipe_price(ctx: ToolContext, inp: FipePriceInput) -> ToolResult:
    """FIPE reference price. Restricted until FIPE confirms display rights."""
    cite = _Cite(ctx)
    if not cite.allowed("fipe"):
        return cite.result(
            "restricted",
            notes=[
                "FIPE prices can't be shown yet: display rights are pending a contract "
                "with FIPE. Suggest the user check veiculos.fipe.org.br directly."
            ],
        )
    dims: dict[str, str | int] = {"model_year": inp.model_year} if inp.model_year else {}
    facts = query_facts(
        ctx.conn, "fipe_reference_price", subject_id=inp.fipe_code, dims=dims, latest_only=True
    )
    if not facts:
        return cite.result("no_data")
    cite.add(facts)
    return cite.result(
        "ok", [{"model_year": f.dims["model_year"], "price": f.value} for f in facts]
    )


class EventsInput(BaseModel):
    since: str | None = Field(default=None, description="Only events on or after 'YYYY-MM-DD'.")


def events(ctx: ToolContext, inp: EventsInput) -> ToolResult:
    """Tax changes, rating-protocol changes and other breaks that explain market moves."""
    rows = ctx.conn.execute(
        "SELECT date, kind, title, scope, source_url FROM event WHERE date >= ? ORDER BY date",
        (inp.since or "1900-01-01",),
    ).fetchall()
    if not rows:
        return ToolResult(status="no_data")
    return ToolResult(status="ok", data=[dict(r) for r in rows])


class FreshnessInput(BaseModel):
    pass


def data_freshness(ctx: ToolContext, inp: FreshnessInput) -> ToolResult:
    """When each source was last checked and the latest period its data describes."""
    rows = []
    for spec in ctx.registry:
        last = _last_success(ctx.conn, spec.id)
        latest = ctx.conn.execute(
            "SELECT MAX(as_of_end) FROM observation WHERE source_id = ?", (spec.id,)
        ).fetchone()[0]
        if last is None and latest is None:
            continue
        age = (ctx.today - datetime.fromisoformat(last).date()).days if last else None
        rows.append(
            {
                "source_id": spec.id,
                "source": spec.name,
                "last_checked": last,
                "latest_period_end": latest,
                "status": "stale" if age is None or age > spec.freshness_days else "fresh",
            }
        )
    return ToolResult(status="ok" if rows else "no_data", data=rows)


# --- registry of tools --------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[[ToolContext, Any], ToolResult]


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec("find_vehicle", find_vehicle.__doc__ or "", FindVehicleInput, find_vehicle),
        ToolSpec("fleet", fleet.__doc__ or "", FleetInput, fleet),
        ToolSpec("registrations", registrations.__doc__ or "", RegistrationsInput, registrations),
        ToolSpec("fuel_prices", fuel_prices.__doc__ or "", FuelPricesInput, fuel_prices),
        ToolSpec("loan_rate", loan_rate.__doc__ or "", LoanRateInput, loan_rate),
        ToolSpec("deflate_price", deflate_price.__doc__ or "", DeflateInput, deflate_price),
        ToolSpec(
            "flex_fuel_choice", flex_fuel_choice.__doc__ or "", FlexChoiceInput, flex_fuel_choice
        ),
        ToolSpec(
            "ownership_cost",
            ownership_cost_tool.__doc__ or "",
            OwnershipCostInput,
            ownership_cost_tool,
        ),
        ToolSpec("consumption", consumption.__doc__ or "", ConsumptionInput, consumption),
        ToolSpec("fipe_price", fipe_price.__doc__ or "", FipePriceInput, fipe_price),
        ToolSpec("events", events.__doc__ or "", EventsInput, events),
        ToolSpec(
            "information_sources",
            information_sources.__doc__ or "",
            InformationSourcesInput,
            information_sources,
        ),
        ToolSpec(
            "expert_content", expert_content.__doc__ or "", ExpertContentInput, expert_content
        ),
        ToolSpec("data_freshness", data_freshness.__doc__ or "", FreshnessInput, data_freshness),
    ]
}


def call_tool(ctx: ToolContext, name: str, arguments: dict[str, Any]) -> ToolResult:
    """Validate arguments and run a tool. Unknown tools and bad input raise ValueError."""
    spec = TOOLS.get(name)
    if spec is None:
        raise ValueError(f"Unknown tool '{name}'. Available: {', '.join(TOOLS)}")
    try:
        inp = spec.input_model.model_validate(arguments)
    except ValidationError as exc:
        raise ValueError(f"Invalid input for {name}: {exc}") from exc
    return spec.fn(ctx, inp)


def tool_definitions() -> list[dict[str, Any]]:
    """Tool definitions in the shape the Claude Messages API expects."""
    return [
        {
            "name": spec.name,
            "description": " ".join(spec.description.split()),
            "input_schema": inline_refs(spec.input_model.model_json_schema()),
        }
        for spec in TOOLS.values()
    ]


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace `$ref`s to `$defs` with the definitions themselves."""
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                return walk(dict(defs[ref.split("/")[-1]]))
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    result: dict[str, Any] = walk(schema)
    return result
