"""Tests for efficiency calculations."""

from src.efficiency import _compute_possessions, _get_recency_weights

import pandas as pd


def test_possessions_formula() -> None:
    """Test the possession estimate formula."""
    # FGA=60, OR=10, TO=15, FTA=20
    # = 60 - 10 + 15 + 0.475 * 20 = 74.5
    result = _compute_possessions(60, 10, 15, 20)
    assert abs(result - 74.5) < 1e-10


def test_possessions_zero_fta() -> None:
    """Test possessions with no free throws."""
    # FGA=50, OR=8, TO=12, FTA=0
    # = 50 - 8 + 12 + 0 = 54
    result = _compute_possessions(50, 8, 12, 0)
    assert abs(result - 54.0) < 1e-10


def test_recency_weights_late_season() -> None:
    """Late season games should get 1.5x weight."""
    days = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    weights = _get_recency_weights(days)
    # Threshold = 10 + 0.7 * 90 = 73
    # Days >= 73 get 1.5, rest get 1.0
    assert weights.iloc[0] == 1.0   # Day 10
    assert weights.iloc[-1] == 1.5  # Day 100
    assert weights.iloc[-2] == 1.5  # Day 90
    assert weights.iloc[-3] == 1.5  # Day 80


def test_recency_weights_single_game() -> None:
    """Single game should get weight 1.0."""
    days = pd.Series([50])
    weights = _get_recency_weights(days)
    assert weights.iloc[0] == 1.0
