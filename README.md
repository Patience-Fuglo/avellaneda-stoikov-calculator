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
