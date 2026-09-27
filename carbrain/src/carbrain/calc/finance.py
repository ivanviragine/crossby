"""Loans (Tabela Price), rate conversion and inflation adjustment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


def monthly_from_annual(annual_pct: float) -> float:
    """Effective annual rate (% a.a.) -> equivalent monthly rate (% a.m.)."""
    return float((1 + annual_pct / 100) ** (1 / 12) - 1) * 100


def annual_from_monthly(monthly_pct: float) -> float:
    return float((1 + monthly_pct / 100) ** 12 - 1) * 100


def installment(principal: float, monthly_pct: float, months: int) -> float:
    """Fixed monthly payment under the Tabela Price (French amortization) system."""
    if months <= 0:
        raise ValueError("months must be positive")
    if principal <= 0:
        return 0.0
    i = monthly_pct / 100
    if i == 0:
        return principal / months
    return principal * i / (1 - (1 + i) ** -months)


@dataclass(frozen=True)
class Loan:
    principal: float
    monthly_pct: float
    months: int
    installment: float
    total_paid: float
    total_interest: float


def loan(price: float, down_payment: float, monthly_pct: float, months: int) -> Loan:
    if not 0 <= down_payment <= price:
        raise ValueError("down payment must be between 0 and the price")
    principal = price - down_payment
    pmt = installment(principal, monthly_pct, months)
    total = pmt * months
    return Loan(principal, monthly_pct, months, pmt, total, total - principal)


def deflate(value: float, from_month: date, to_month: date, index: dict[date, float]) -> float:
    """Express `value` (in `from_month` reais) in `to_month` reais using a price index.

    `index` maps the first day of each month to the index number (e.g. IPCA).
    """
    start, end = from_month.replace(day=1), to_month.replace(day=1)
    missing = [m.isoformat()[:7] for m in (start, end) if m not in index]
    if missing:
        raise KeyError(f"No index value for {', '.join(missing)}")
    return value * index[end] / index[start]
