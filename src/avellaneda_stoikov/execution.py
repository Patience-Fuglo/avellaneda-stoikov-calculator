"""Almgren-Chriss (2000) optimal execution, derived and implemented from
scratch: given a large position that needs liquidating, how fast should
you trade?

Two real costs pull in opposite directions:
- Market impact: trading fast pushes the price against you.
- Risk exposure: trading slow leaves you exposed to the price drifting
  against you while the position is still open.

The closed-form optimal holdings trajectory (continuous-time
approximation, linear temporary impact, no permanent impact -- permanent
impact adds a fixed cost that doesn't change which trajectory is optimal,
only the total cost of following it):

    x(t) = X * sinh(kappa * (T - t)) / sinh(kappa * T)

where kappa = sqrt(risk_aversion * volatility^2 / temporary_impact) sets
how front-loaded the optimal schedule is. The same risk-aversion idea
(gamma) as the reservation-price model reappears here: more risk-averse
means a larger kappa, means a more front-loaded (faster) schedule to
escape risk exposure sooner.

As risk_aversion -> 0, kappa -> 0 and the schedule degenerates to a
straight line (a plain TWAP: trade at a constant rate) -- with nothing
left to balance against impact cost, there's no reason to front-load.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def optimal_execution_trajectory(
    total_shares: float,
    time_horizon: float,
    risk_aversion: float,
    volatility: float,
    temporary_impact: float,
    n_steps: int = 50,
) -> pd.DataFrame:
    """The optimal holdings trajectory for liquidating ``total_shares``
    over ``time_horizon``, balancing market impact against risk exposure.

    Returns a DataFrame with ``time`` and ``remaining_shares`` (holdings
    still left to sell at each step), from ``total_shares`` at t=0 down
    to 0 at t=time_horizon.
    """
    if total_shares <= 0:
        raise ValueError("total_shares must be positive")
    if time_horizon <= 0:
        raise ValueError("time_horizon must be positive")
    if risk_aversion < 0:
        raise ValueError("risk_aversion cannot be negative")
    if volatility < 0:
        raise ValueError("volatility cannot be negative")
    if temporary_impact <= 0:
        raise ValueError("temporary_impact must be positive")

    times = np.linspace(0.0, time_horizon, n_steps + 1)

    kappa_sq = risk_aversion * (volatility**2) / temporary_impact
    if kappa_sq < 1e-12:
        # degenerate case: no risk aversion -> a straight-line TWAP,
        # the sinh ratio's limit as kappa -> 0 (L'Hopital: sinh(kx)/kx -> 1)
        remaining = total_shares * (1.0 - times / time_horizon)
    else:
        kappa = np.sqrt(kappa_sq)
        remaining = total_shares * np.sinh(kappa * (time_horizon - times)) / np.sinh(kappa * time_horizon)

    return pd.DataFrame({"time": times, "remaining_shares": remaining})
