# avellaneda-stoikov-calculator

Reservation price and optimal spread for a market maker, derived and
implemented from scratch (Avellaneda & Stoikov, 2008): where should a
market maker center their quotes, and how wide should the spread be,
given their current inventory, risk aversion, the asset's volatility, and
how much time is left in the trading session?

## Status

| Module | Status |
|---|---|
| Reservation price + optimal spread | done |
| Sensitivity analysis (risk aversion, volatility) | done |

## Reservation price + optimal spread

`src/avellaneda_stoikov/pricing.py`

**Reservation price** is the price at which a market maker is
indifferent between holding their current inventory or not — the fair
mid-price, shifted by the risk of already holding a position. Positive
(long) inventory shifts it below the real market mid-price; negative
(short) inventory shifts it above; zero inventory leaves it unchanged.

**Optimal spread** balances two real forces: the inventory-risk exposure
still left to carry (`gamma * sigma^2 * (T-t)`), and a floor spread
driven by how sensitive real order-arrival probability is to quote
aggressiveness (`(2/gamma) * ln(1 + gamma/k)`). The spread never
disappears, even at the very last instant of the session — the floor
term doesn't depend on time remaining.

```python
from avellaneda_stoikov import quote

q = quote(
    mid_price=358.97, inventory=5000, risk_aversion=0.1,
    volatility=0.02914, time_remaining=1.0, order_arrival_sensitivity=1.5,
)
# q.reservation_price, q.spread, q.bid, q.ask
```

Cross-checked against the standard worked example from the original 2008
paper (`gamma=0.1, sigma=2, T-t=1, k=1.5` → spread ≈ 1.69077), computed
independently in the test suite rather than just trusting the formula
transcription.

**Real result:** using real TSLA daily volatility (0.02914, i.e. ~46.3%
annualized, from 281 real trading days) as the volatility input, a
modest real inventory (50 shares) produces a barely-visible reservation
price skew — the fill-probability floor term dominates the spread at
this volatility level, not inventory risk. At 5,000 shares, the same
real volatility input produces a real, clearly visible ~$0.42 skew. Same
formula, same real data — the skew's visibility scales directly with how
much inventory risk is actually being carried. The spread itself never
changes with inventory in this model; only the reservation price does.

Run the real-data demo:

```bash
pip install -e .
python scripts/demo_quote.py
```

Run the tests:

```bash
pytest tests/
```

## Sensitivity analysis

`src/avellaneda_stoikov/sensitivity.py`

How does the quote respond as risk aversion (`gamma`) or volatility
(`sigma`) change, one at a time? `gamma` is swept across a standard
illustrative range (it's the market maker's own design preference, not
observed data); `sigma` is swept across TSLA's own **real** observed
20-day rolling volatility range, not an arbitrary made-up range.

```python
from avellaneda_stoikov import sensitivity_to_risk_aversion, sensitivity_to_volatility

gamma_result = sensitivity_to_risk_aversion(mid_price, inventory, volatility, time_remaining, k, gamma_values)
vol_result = sensitivity_to_volatility(mid_price, inventory, risk_aversion, time_remaining, k, sigma_values)
```

**Real result, both sweeps confirm the closed-form relationships
directly:** the reservation-price skew grows linearly with `gamma`
(doubling it doubles the skew) and quadratically with `sigma` (doubling
it quadruples the skew) — both cross-checked in the test suite, not just
asserted.

**A genuinely surprising real finding, verified by decomposing the
spread formula's two terms rather than assumed:** at TSLA's real,
current volatility, total spread actually *decreases* as `gamma`
increases (1.3245 → 1.1513 across a 0.02–0.5 sweep) — the opposite of
the common textbook claim that more risk-averse market makers always
quote wider spreads. At this real volatility level the inventory-risk
term (`gamma * sigma^2 * (T-t)`) is negligible (0.00002 to 0.0005) next
to the floor term (`(2/gamma) * ln(1+gamma/k)`), which itself *shrinks*
as `gamma` grows — so the floor term's shrinkage dominates the total.
At high volatility (`sigma=2`, the paper's own example scale), the
inventory-risk term dominates instead and spread *increases* with
`gamma`, matching the textbook intuition. Neither direction is
universally correct — which one holds depends entirely on the real
volatility regime, confirmed here with both a real market's data and a
direct test of both regimes (`test_spread_vs_risk_aversion_direction_depends_on_volatility_regime`).

Run the real-data demo:

```bash
python scripts/demo_sensitivity.py
```
