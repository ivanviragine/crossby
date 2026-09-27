from __future__ import annotations

from datetime import date

import pytest

from carbrain.calc.finance import (
    annual_from_monthly,
    deflate,
    installment,
    loan,
    monthly_from_annual,
)
from carbrain.calc.fuel import cost_per_km, ev_cost_per_km, flex_choice
from carbrain.calc.ownership import Financing, OwnershipInput, Range, ownership_cost


def test_installment_textbook_case() -> None:
    assert installment(1000, 1.0, 12) == pytest.approx(88.8488, abs=1e-4)
    assert installment(1200, 0.0, 12) == 100


def test_rate_conversion_round_trip() -> None:
    assert monthly_from_annual(12.682503) == pytest.approx(1.0, abs=1e-6)
    assert annual_from_monthly(monthly_from_annual(26.9)) == pytest.approx(26.9)


def test_loan_totals() -> None:
    plan = loan(120_000, 40_000, 1.98, 48)
    assert plan.principal == 80_000
    assert plan.total_paid == pytest.approx(plan.installment * 48)
    assert plan.total_interest == pytest.approx(plan.total_paid - 80_000)
    with pytest.raises(ValueError):
        loan(100, 200, 1, 10)


def test_deflate() -> None:
    index = {date(2025, 1, 1): 100.0, date(2026, 1, 1): 104.5}
    assert deflate(1000, date(2025, 1, 15), date(2026, 1, 1), index) == pytest.approx(1045)
    with pytest.raises(KeyError, match="2024-01"):
        deflate(1000, date(2024, 1, 1), date(2026, 1, 1), index)


def test_flex_choice_uses_the_cars_own_ratio() -> None:
    choice = flex_choice(8.0, 11.5, 4.10, 6.55)
    assert choice.best == "ethanol"
    assert choice.breakeven_ratio == pytest.approx(8.0 / 11.5)
    assert choice.price_ratio == pytest.approx(4.10 / 6.55)
    # A car whose ethanol consumption is much worse flips the answer at the same prices.
    assert flex_choice(6.5, 11.5, 4.10, 6.55).best == "gasoline"


def test_cost_per_km() -> None:
    assert cost_per_km(6.0, 12.0) == 0.5
    assert ev_cost_per_km(0.9, 12.0) == pytest.approx(0.108)
    with pytest.raises(ValueError):
        cost_per_km(6.0, 0)


def test_ownership_cost_components() -> None:
    result = ownership_cost(
        OwnershipInput(
            purchase_price=100_000,
            years=1,
            km_per_year=10_000,
            energy_cost_per_km=Range.of(0.5),
            annual_depreciation_pct=Range.of(10),
            insurance_per_year=Range.of(3_000),
            maintenance_per_year=Range.of(1_000),
            ipva_rate_pct=4,
        )
    )
    base = result.summary()[1]
    assert base["components"] == {
        "depreciation": 10_000,
        "energy": 5_000,
        "ipva": 4_000,
        "insurance": 3_000,
        "maintenance": 1_000,
    }
    assert base["total"] == 23_000 and base["per_month"] == pytest.approx(1916.67)
    assert result.excluded == []


def test_ownership_scenarios_are_ordered_and_exclusions_listed() -> None:
    result = ownership_cost(
        OwnershipInput(
            purchase_price=120_000,
            years=4,
            km_per_year=18_000,
            energy_cost_per_km=Range.of(0.45),
            annual_depreciation_pct=Range(low=8, base=12, high=16),
            financing=Financing(down_payment=40_000, monthly_rate_pct=1.98, months=48),
        )
    )
    totals = [s["total"] for s in result.summary()]
    assert totals == sorted(totals)
    assert any("IPVA" in e for e in result.excluded)
    interest = loan(120_000, 40_000, 1.98, 48).total_interest
    assert result.scenarios[0].components["financing_interest"] == pytest.approx(interest)


def test_range_must_be_ordered() -> None:
    with pytest.raises(ValueError):
        Range(low=10, base=5, high=20)
