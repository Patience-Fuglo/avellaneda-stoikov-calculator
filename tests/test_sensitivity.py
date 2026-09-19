import pytest

from avellaneda_stoikov import sensitivity_to_risk_aversion, sensitivity_to_volatility


def test_sensitivity_to_risk_aversion_returns_one_row_per_value():
    result = sensitivity_to_risk_aversion(
        mid_price=100.0, inventory=50.0, volatility=0.03, time_remaining=1.0,
        order_arrival_sensitivity=1.5, risk_aversion_values=[0.05, 0.1, 0.2, 0.5],
    )
    assert len(result) == 4
    assert list(result.columns) == ["risk_aversion", "reservation_price", "skew", "spread"]


def test_sensitivity_to_risk_aversion_skew_magnitude_grows_with_gamma():
    # skew = -inventory * gamma * sigma^2 * (T-t) -- linear in gamma, so a
    # long position's skew should grow strictly more negative as gamma rises
    result = sensitivity_to_risk_aversion(
        mid_price=100.0, inventory=50.0, volatility=0.03, time_remaining=1.0,
        order_arrival_sensitivity=1.5, risk_aversion_values=[0.05, 0.1, 0.2],
    )
    skews = result["skew"].tolist()
    assert skews[0] > skews[1] > skews[2]  # increasingly negative (long inventory)
    assert all(s < 0 for s in skews)


def test_sensitivity_to_risk_aversion_skew_doubles_when_gamma_doubles():
    # real cross-check of the linear relationship, not just "monotonic"
    result = sensitivity_to_risk_aversion(
        mid_price=100.0, inventory=50.0, volatility=0.03, time_remaining=1.0,
        order_arrival_sensitivity=1.5, risk_aversion_values=[0.1, 0.2],
    )
    skew_at_01, skew_at_02 = result["skew"].tolist()
    assert skew_at_02 == pytest.approx(2 * skew_at_01)


def test_sensitivity_to_volatility_returns_one_row_per_value():
    result = sensitivity_to_volatility(
        mid_price=100.0, inventory=50.0, risk_aversion=0.1, time_remaining=1.0,
        order_arrival_sensitivity=1.5, volatility_values=[0.01, 0.02, 0.03],
    )
    assert len(result) == 3
    assert list(result.columns) == ["volatility", "reservation_price", "skew", "spread"]


def test_sensitivity_to_volatility_skew_quadruples_when_sigma_doubles():
    # skew depends on sigma^2, so doubling sigma should quadruple the
    # magnitude of the skew -- a real, checkable, non-linear relationship
    result = sensitivity_to_volatility(
        mid_price=100.0, inventory=50.0, risk_aversion=0.1, time_remaining=1.0,
        order_arrival_sensitivity=1.5, volatility_values=[0.02, 0.04],
    )
    skew_at_02, skew_at_04 = result["skew"].tolist()
    assert skew_at_04 == pytest.approx(4 * skew_at_02)


def test_spread_vs_risk_aversion_direction_depends_on_volatility_regime():
    # a genuinely surprising real finding, verified here rather than just
    # asserted: at low (real-world) volatility the inventory-risk term is
    # negligible next to the floor term, which itself SHRINKS as gamma
    # grows -- so spread actually decreases with gamma. At high volatility
    # (the paper's own example scale), the inventory-risk term dominates
    # instead and spread increases with gamma, matching the common
    # textbook intuition. Both are real; neither is universally true.
    low_vol_result = sensitivity_to_risk_aversion(
        mid_price=100.0, inventory=0.0, volatility=0.03, time_remaining=1.0,
        order_arrival_sensitivity=1.5, risk_aversion_values=[0.02, 0.5],
    )
    assert low_vol_result["spread"].iloc[0] > low_vol_result["spread"].iloc[1]

    high_vol_result = sensitivity_to_risk_aversion(
        mid_price=100.0, inventory=0.0, volatility=2.0, time_remaining=1.0,
        order_arrival_sensitivity=1.5, risk_aversion_values=[0.02, 0.5],
    )
    assert high_vol_result["spread"].iloc[0] < high_vol_result["spread"].iloc[1]


def test_sensitivity_to_volatility_spread_widens_with_volatility():
    result = sensitivity_to_volatility(
        mid_price=100.0, inventory=50.0, risk_aversion=0.1, time_remaining=1.0,
        order_arrival_sensitivity=1.5, volatility_values=[0.01, 0.02, 0.03],
    )
    spreads = result["spread"].tolist()
    assert spreads[0] < spreads[1] < spreads[2]
