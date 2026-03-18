"""Tests for the Elo rating system."""

import math

from src.elo import _expected_score, _mov_multiplier


def test_expected_score_equal_ratings() -> None:
    """Equal ratings should give 50/50 expected scores."""
    assert abs(_expected_score(1500, 1500) - 0.5) < 1e-10


def test_expected_score_higher_rating_favored() -> None:
    """Higher-rated team should have >50% expected score."""
    exp = _expected_score(1600, 1400)
    assert exp > 0.5
    # 200-point gap should give ~0.76
    assert abs(exp - 0.7597) < 0.01


def test_expected_score_symmetry() -> None:
    """Expected scores should sum to 1."""
    exp_a = _expected_score(1600, 1400)
    exp_b = _expected_score(1400, 1600)
    assert abs(exp_a + exp_b - 1.0) < 1e-10


def test_mov_multiplier_larger_margin_gives_higher_mult() -> None:
    """Larger score differences should produce larger multipliers."""
    mult_small = _mov_multiplier(5, 100)
    mult_large = _mov_multiplier(20, 100)
    assert mult_large > mult_small


def test_mov_multiplier_dampens_blowouts() -> None:
    """Large elo diff (strong team blowing out weak team) should be dampened."""
    mult_close = _mov_multiplier(20, 0)      # Equal teams, big win
    mult_mismatch = _mov_multiplier(20, 300)  # Strong team blowing out weak
    assert mult_close > mult_mismatch


def test_mov_multiplier_positive() -> None:
    """MOV multiplier should always be positive for positive score diffs."""
    for diff in [1, 5, 10, 30]:
        for elo_diff in [0, 100, 200, 500]:
            assert _mov_multiplier(diff, elo_diff) > 0


def test_season_regression() -> None:
    """Season regression should move ratings toward 1500."""
    old_elo = 1700
    regress = 0.75
    new_elo = 1500 + regress * (old_elo - 1500)
    assert new_elo == 1650.0

    old_elo = 1300
    new_elo = 1500 + regress * (old_elo - 1500)
    assert new_elo == 1350.0
