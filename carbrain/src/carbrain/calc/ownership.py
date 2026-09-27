"""Ownership-cost scenarios (low / base / high) with every assumption listed.

Nothing is hidden: each input is either given by the caller, looked up from a cited
source by the tool layer, or a labeled assumption with a range.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field, model_validator

from carbrain.calc.finance import loan


class Range(BaseModel):
    """A value with an optional range. A single number means low = base = high."""

    low: float
    base: float
    high: float

    @classmethod
    def of(cls, value: float | Range) -> Range:
        return value if isinstance(value, Range) else cls(low=value, base=value, high=value)

    @model_validator(mode="after")
    def _ordered(self) -> Range:
        if not self.low <= self.base <= self.high:
            raise ValueError("expected low <= base <= high")
        return self


class Financing(BaseModel):
    down_payment: float = Field(ge=0)
    monthly_rate_pct: float = Field(ge=0)
    months: int = Field(gt=0, le=120)


class OwnershipInput(BaseModel):
    purchase_price: float = Field(gt=0)
    years: int = Field(gt=0, le=15)
    km_per_year: float = Field(gt=0)
    energy_cost_per_km: Range
    annual_depreciation_pct: Range
    insurance_per_year: Range | None = None
    maintenance_per_year: Range | None = None
    ipva_rate_pct: float | None = Field(default=None, ge=0, le=10)
    financing: Financing | None = None


@dataclass(frozen=True)
class Scenario:
    name: str
    components: dict[str, float]
    resale_value: float

    @property
    def total(self) -> float:
        return sum(self.components.values())


@dataclass
class OwnershipResult:
    months: int
    km: float
    scenarios: list[Scenario]
    excluded: list[str] = field(default_factory=list)

    def summary(self) -> list[dict[str, object]]:
        return [
            {
                "scenario": s.name,
                "total": round(s.total, 2),
                "per_month": round(s.total / self.months, 2),
                "per_km": round(s.total / self.km, 3),
                "resale_value": round(s.resale_value, 2),
                "components": {k: round(v, 2) for k, v in s.components.items()},
            }
            for s in self.scenarios
        ]


def ownership_cost(inp: OwnershipInput) -> OwnershipResult:
    """Total cost of keeping a car for `years`, excluding the resale value recovered.

    Depreciation = price - resale value after `years` at the annual rate. IPVA is charged
    each year on the car's estimated value at the start of that year (an approximation of
    the venal value states use). Financing adds interest only; the principal is the car.
    """
    months = inp.years * 12
    km = inp.km_per_year * inp.years
    excluded: list[str] = []
    if inp.insurance_per_year is None:
        excluded.append("insurance (no estimate given)")
    if inp.maintenance_per_year is None:
        excluded.append("maintenance (no estimate given)")
    if inp.ipva_rate_pct is None:
        excluded.append("IPVA (state rate not given)")

    interest = 0.0
    if inp.financing is not None:
        f = inp.financing
        interest = loan(
            inp.purchase_price, f.down_payment, f.monthly_rate_pct, f.months
        ).total_interest

    scenarios = []
    for name in ("low", "base", "high"):
        # "low" means the cheapest plausible outcome: slower depreciation, lower costs.
        dep_rate = getattr(inp.annual_depreciation_pct, name) / 100
        values = [inp.purchase_price * (1 - dep_rate) ** y for y in range(inp.years + 1)]
        components = {
            "depreciation": inp.purchase_price - values[-1],
            "energy": getattr(inp.energy_cost_per_km, name) * km,
        }
        if inp.ipva_rate_pct is not None:
            components["ipva"] = sum(v * inp.ipva_rate_pct / 100 for v in values[:-1])
        if inp.insurance_per_year is not None:
            components["insurance"] = getattr(inp.insurance_per_year, name) * inp.years
        if inp.maintenance_per_year is not None:
            components["maintenance"] = getattr(inp.maintenance_per_year, name) * inp.years
        if inp.financing is not None:
            components["financing_interest"] = interest
        scenarios.append(Scenario(name, components, values[-1]))
    return OwnershipResult(months, km, scenarios, excluded)
