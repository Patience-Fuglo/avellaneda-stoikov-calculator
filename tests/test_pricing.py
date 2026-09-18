import math

import pytest

from avellaneda_stoikov import Quote, optimal_spread, quote, reservation_price


# ---- reservation price -----------------------------------------------------


def test_reservation_price_equals_mid_price_at_zero_inventory():
    r = reservation_price(mid_price=100.0, inventory=0.0, risk_aversion=0.1, volatility=2.0, time_remaining=1.0)
    assert r == pytest.approx(100.0)


def test_reservation_price_shifts_below_mid_for_long_inventory():
    r = reservation_price(mid_price=100.0, inventory=5.0, risk_aversion=0.1, volatility=2.0, time_remaining=1.0)
    # hand-computed: 100 - 5*0.1*4*1 = 100 - 2 = 98
    assert r == pytest.approx(98.0)


def test_reservation_price_shifts_above_mid_for_short_inventory():
    r = reservation_price(mid_price=100.0, inventory=-5.0, risk_aversion=0.1, volatility=2.0, time_remaining=1.0)
    assert r == pytest.approx(102.0)


def test_reservation_price_adjustment_shrinks_as_time_remaining_shrinks():
    r_early = reservation_price(100.0, inventory=5.0, risk_aversion=0.1, volatility=2.0, time_remaining=1.0)
    r_late = reservation_price(100.0, inventory=5.0, risk_aversion=0.1, volatility=2.0, time_remaining=0.1)
    # less time left -> smaller adjustment -> closer to the real mid-price
    assert abs(r_late - 100.0) < abs(r_early - 100.0)


def test_reservation_price_rejects_non_positive_risk_aversion():
    with pytest.raises(ValueError):
        reservation_price(100.0, inventory=1.0, risk_aversion=0.0, volatility=2.0, time_remaining=1.0)


def test_reservation_price_rejects_negative_volatility():
    with pytest.raises(ValueError):
        reservation_price(100.0, inventory=1.0, risk_aversion=0.1, volatility=-2.0, time_remaining=1.0)


# ---- optimal spread ---------------------------------------------------------


def test_optimal_spread_matches_hand_computed_textbook_example():
    # gamma=0.1, sigma=2, T-t=1, k=1.5 -- the standard worked example from
    # the original Avellaneda-Stoikov paper, hand-computed independently here:
    # inventory term: 0.1 * 4 * 1 = 0.4
    # floor term: (2/0.1) * ln(1 + 0.1/1.5) = 20 * ln(1.06666...) = 20 * 0.0645385... = 1.29077...
    # total: 0.4 + 1.29077... = 1.69077...
    delta = optimal_spread(risk_aversion=0.1, volatility=2.0, time_remaining=1.0, order_arrival_sensitivity=1.5)
    assert delta == pytest.approx(0.4 + 20 * math.log(1 + 0.1 / 1.5), abs=1e-9)
    assert delta == pytest.approx(1.69077, abs=1e-4)


def test_optimal_spread_stays_positive_even_at_zero_time_remaining():
    # the fill-probability floor term doesn't depend on time_remaining --
    # a real market maker still needs SOME edge even at the very last instant
    delta = optimal_spread(risk_aversion=0.1, volatility=2.0, time_remaining=0.0, order_arrival_sensitivity=1.5)
    assert delta > 0


def test_optimal_spread_widens_with_more_time_remaining():
    delta_early = optimal_spread(risk_aversion=0.1, volatility=2.0, time_remaining=1.0, order_arrival_sensitivity=1.5)
    delta_late = optimal_spread(risk_aversion=0.1, volatility=2.0, time_remaining=0.1, order_arrival_sensitivity=1.5)
    assert delta_early > delta_late


def test_optimal_spread_rejects_non_positive_order_arrival_sensitivity():
    with pytest.raises(ValueError):
        optimal_spread(risk_aversion=0.1, volatility=2.0, time_remaining=1.0, order_arrival_sensitivity=0.0)


# ---- full quote --------------------------------------------------------------


def test_quote_bid_ask_are_symmetric_around_reservation_price_not_mid_price():
    q = quote(
        mid_price=100.0, inventory=5.0, risk_aversion=0.1, volatility=2.0,
        time_remaining=1.0, order_arrival_sensitivity=1.5,
    )
    assert isinstance(q, Quote)
    assert q.reservation_price == pytest.approx(98.0)
    midpoint_of_quote = (q.bid + q.ask) / 2
    assert midpoint_of_quote == pytest.approx(q.reservation_price)
    # NOT centered on the real market mid-price, since inventory shifted it
    assert midpoint_of_quote != pytest.approx(100.0)


def test_quote_ask_minus_bid_equals_optimal_spread():
    q = quote(
        mid_price=100.0, inventory=0.0, risk_aversion=0.1, volatility=2.0,
        time_remaining=1.0, order_arrival_sensitivity=1.5,
    )
    assert (q.ask - q.bid) == pytest.approx(q.spread)
