"""Fuel and energy cost per km, and the flex-fuel choice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


def cost_per_km(price_per_unit: float, km_per_unit: float) -> float:
    """R$ per km for a fuel price (R$/l) and a consumption (km/l)."""
    if km_per_unit <= 0:
        raise ValueError("consumption must be positive")
    return price_per_unit / km_per_unit


def ev_cost_per_km(price_per_kwh: float, kwh_per_100km: float) -> float:
    if kwh_per_100km <= 0:
        raise ValueError("consumption must be positive")
    return price_per_kwh * kwh_per_100km / 100


@dataclass(frozen=True)
class FlexChoice:
    best: Literal["ethanol", "gasoline"]
    ethanol_per_km: float
    gasoline_per_km: float
    price_ratio: float
    breakeven_ratio: float
    saving_per_1000_km: float


def flex_choice(
    km_per_l_ethanol: float,
    km_per_l_gasoline: float,
    price_ethanol: float,
    price_gasoline: float,
) -> FlexChoice:
    """Cheaper fuel for a flex car, using the car's own consumption ratio.

    Ethanol pays off when its price is below `breakeven_ratio` times the gasoline price, where the
    ratio is this car's ethanol/gasoline consumption. The popular "70% rule" is only an
    average of this ratio across cars.
    """
    ethanol = cost_per_km(price_ethanol, km_per_l_ethanol)
    gasoline = cost_per_km(price_gasoline, km_per_l_gasoline)
    best: Literal["ethanol", "gasoline"] = "ethanol" if ethanol < gasoline else "gasoline"
    return FlexChoice(
        best=best,
        ethanol_per_km=ethanol,
        gasoline_per_km=gasoline,
        price_ratio=price_ethanol / price_gasoline,
        breakeven_ratio=km_per_l_ethanol / km_per_l_gasoline,
        saving_per_1000_km=abs(ethanol - gasoline) * 1000,
    )
