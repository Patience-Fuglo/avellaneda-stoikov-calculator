"""Sensitivity analysis: how does the Avellaneda-Stoikov quote respond as
risk aversion and volatility change, one at a time, holding everything
else fixed?

Two different kinds of input, treated differently:
- Risk aversion (gamma) is the market maker's own design/preference
  parameter -- not something observed in data, so it's swept across a
  standard illustrative range.
- Volatility (sigma) gets swept across the REAL range of realized
  volatility a real name has actually shown in its own price history
  (see the demo script), not an arbitrary made-up range.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .pricing import optimal_spread, reservation_price


def sensitivity_to_risk_aversion(
    mid_price: float,
    inventory: float,
    volatility: float,
    time_remaining: float,
    order_arrival_sensitivity: float,
    risk_aversion_values: Sequence[float],
) -> pd.DataFrame:
    """Reservation-price skew and spread at each risk-aversion value,
    holding inventory/volatility/time/order-arrival-sensitivity fixed."""
    rows = []
    for gamma in risk_aversion_values:
        r = reservation_price(mid_price, inventory, gamma, volatility, time_remaining)
        delta = optimal_spread(gamma, volatility, time_remaining, order_arrival_sensitivity)
        rows.append({"risk_aversion": gamma, "reservation_price": r, "skew": r - mid_price, "spread": delta})
    return pd.DataFrame(rows)


def sensitivity_to_volatility(
    mid_price: float,
    inventory: float,
    risk_aversion: float,
    time_remaining: float,
    order_arrival_sensitivity: float,
    volatility_values: Sequence[float],
) -> pd.DataFrame:
    """Reservation-price skew and spread at each volatility value, holding
    inventory/risk-aversion/time/order-arrival-sensitivity fixed."""
    rows = []
    for sigma in volatility_values:
        r = reservation_price(mid_price, inventory, risk_aversion, sigma, time_remaining)
        delta = optimal_spread(risk_aversion, sigma, time_remaining, order_arrival_sensitivity)
        rows.append({"volatility": sigma, "reservation_price": r, "skew": r - mid_price, "spread": delta})
    return pd.DataFrame(rows)
