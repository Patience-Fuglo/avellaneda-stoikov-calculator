"""Avellaneda-Stoikov (2008) reservation price and optimal spread,
derived and implemented from scratch.

The question this answers: as a market maker holding some inventory,
where should you center your quotes, and how wide should the spread be?

Two closed-form results from the original paper drive both answers:

Reservation price -- the fair price you'd quote, adjusted for the risk
of already holding ``inventory`` units of the asset:

    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

Optimal spread -- how wide to quote around that reservation price,
balancing inventory risk against how sensitive fill probability is to
quote aggressiveness:

    delta(t) = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def reservation_price(
    mid_price: float,
    inventory: float,
    risk_aversion: float,
    volatility: float,
    time_remaining: float,
) -> float:
    """The price at which the market maker is indifferent between holding
    ``inventory`` units of the asset or not.

    Positive ``inventory`` (long) shifts the reservation price below the
    real market ``mid_price`` -- the market maker would rather sell than
    buy more, so they'd accept a lower price for that inventory right now.
    Negative ``inventory`` (short) shifts it above. Zero inventory means
    the reservation price equals the mid-price exactly -- there's no
    position to adjust for.

    ``time_remaining`` (``T - t``) is real exposure duration, not decay:
    more time left in the trading session means more time for the price
    to move against the held inventory, so the adjustment is larger; it
    shrinks to zero as the session ends.
    """
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")
    if volatility < 0:
        raise ValueError("volatility cannot be negative")
    if time_remaining < 0:
        raise ValueError("time_remaining cannot be negative")
    return mid_price - inventory * risk_aversion * (volatility**2) * time_remaining


def optimal_spread(
    risk_aversion: float,
    volatility: float,
    time_remaining: float,
    order_arrival_sensitivity: float,
) -> float:
    """How wide to quote around the reservation price.

    Two terms, two different real forces:
    - ``gamma * sigma^2 * (T - t)``: the same inventory-risk exposure term
      as the reservation price -- more risk left to carry means a wider
      cushion is needed.
    - ``(2/gamma) * ln(1 + gamma/k)``: a floor spread driven by how
      sensitive real order-arrival probability is to quote aggressiveness
      (``order_arrival_sensitivity``, k) -- quoting too tight barely
      improves fill odds once k is small, so there's no reason to give
      away more edge than that.

    Never negative or zero even at ``time_remaining == 0``: the second
    term is a real floor that doesn't depend on remaining risk exposure.
    """
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")
    if volatility < 0:
        raise ValueError("volatility cannot be negative")
    if time_remaining < 0:
        raise ValueError("time_remaining cannot be negative")
    if order_arrival_sensitivity <= 0:
        raise ValueError("order_arrival_sensitivity (k) must be positive")

    inventory_risk_term = risk_aversion * (volatility**2) * time_remaining
    fill_probability_floor = (2 / risk_aversion) * math.log(1 + risk_aversion / order_arrival_sensitivity)
    return inventory_risk_term + fill_probability_floor


@dataclass
class Quote:
    reservation_price: float
    spread: float
    bid: float
    ask: float


def quote(
    mid_price: float,
    inventory: float,
    risk_aversion: float,
    volatility: float,
    time_remaining: float,
    order_arrival_sensitivity: float,
) -> Quote:
    """The full real quote a market maker would post: reservation price,
    the spread around it, and the resulting bid/ask, symmetric around the
    reservation price (not the raw mid-price).
    """
    r = reservation_price(mid_price, inventory, risk_aversion, volatility, time_remaining)
    delta = optimal_spread(risk_aversion, volatility, time_remaining, order_arrival_sensitivity)
    return Quote(reservation_price=r, spread=delta, bid=r - delta / 2, ask=r + delta / 2)
