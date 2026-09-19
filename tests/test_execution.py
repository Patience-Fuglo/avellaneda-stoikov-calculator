import numpy as np
import pytest

from avellaneda_stoikov import optimal_execution_trajectory


def test_trajectory_starts_at_total_shares_and_ends_at_zero():
    result = optimal_execution_trajectory(
        total_shares=5000, time_horizon=1.0, risk_aversion=1e-6,
        volatility=0.03, temporary_impact=0.1, n_steps=10,
    )
    assert result["remaining_shares"].iloc[0] == pytest.approx(5000)
    assert result["remaining_shares"].iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_trajectory_is_monotonically_decreasing():
    result = optimal_execution_trajectory(
        total_shares=5000, time_horizon=1.0, risk_aversion=0.5,
        volatility=0.03, temporary_impact=0.1, n_steps=20,
    )
    assert result["remaining_shares"].is_monotonic_decreasing


def test_zero_risk_aversion_degenerates_to_a_straight_line_twap():
    result = optimal_execution_trajectory(
        total_shares=1000, time_horizon=1.0, risk_aversion=0.0,
        volatility=0.03, temporary_impact=0.1, n_steps=4,
    )
    # a plain TWAP: equal 250-share decrements at each of the 4 steps
    expected = [1000, 750, 500, 250, 0]
    for actual, exp in zip(result["remaining_shares"], expected):
        assert actual == pytest.approx(exp, abs=1e-6)


def test_higher_risk_aversion_front_loads_the_schedule():
    # at the midpoint in time, a more risk-averse trajectory should have
    # SOLD MORE already (fewer shares remaining) than a less risk-averse one
    low_gamma = optimal_execution_trajectory(
        total_shares=5000, time_horizon=1.0, risk_aversion=0.01,
        volatility=0.03, temporary_impact=0.1, n_steps=10,
    )
    high_gamma = optimal_execution_trajectory(
        total_shares=5000, time_horizon=1.0, risk_aversion=5.0,
        volatility=0.03, temporary_impact=0.1, n_steps=10,
    )
    midpoint_idx = 5  # t = 0.5
    assert high_gamma["remaining_shares"].iloc[midpoint_idx] < low_gamma["remaining_shares"].iloc[midpoint_idx]


def test_rejects_non_positive_total_shares():
    with pytest.raises(ValueError):
        optimal_execution_trajectory(0, 1.0, 0.1, 0.03, 0.1)


def test_rejects_non_positive_time_horizon():
    with pytest.raises(ValueError):
        optimal_execution_trajectory(1000, 0.0, 0.1, 0.03, 0.1)


def test_rejects_negative_risk_aversion():
    with pytest.raises(ValueError):
        optimal_execution_trajectory(1000, 1.0, -0.1, 0.03, 0.1)


def test_rejects_non_positive_temporary_impact():
    with pytest.raises(ValueError):
        optimal_execution_trajectory(1000, 1.0, 0.1, 0.03, 0.0)
