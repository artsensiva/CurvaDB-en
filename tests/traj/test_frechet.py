"""Тесты дискретной метрики Фреше."""

import numpy as np

from traj.frechet import distance, distance_within, lower_bound


def test_identical_is_zero():
    P = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [5.0, 5.0]])
    assert distance(P, P) == 0.0


def test_symmetry():
    rng = np.random.default_rng(1)
    P = rng.random((20, 2)) * 100
    Q = rng.random((15, 2)) * 100
    assert np.isclose(distance(P, Q), distance(Q, P))


def test_shift_invariance():
    rng = np.random.default_rng(2)
    P = rng.random((10, 2)) * 50
    Q = rng.random((12, 2)) * 50
    shift = np.array([37.0, -12.5])
    assert np.isclose(distance(P, Q), distance(P + shift, Q + shift))


def test_lower_bound_never_exceeds_exact_distance():
    rng = np.random.default_rng(3)
    for _ in range(10):
        P = rng.random((8, 2)) * 100
        Q = rng.random((11, 2)) * 100
        assert lower_bound(P, Q) <= distance(P, Q) + 1e-9


def test_distance_within_matches_full_when_threshold_generous():
    rng = np.random.default_rng(4)
    P = rng.random((15, 2)) * 100
    Q = rng.random((15, 2)) * 100
    d = distance(P, Q)
    val, exact = distance_within(P, Q, threshold=d + 10.0)
    assert exact
    assert np.isclose(val, d)


def test_distance_within_flags_inexact_below_threshold():
    rng = np.random.default_rng(5)
    P = rng.random((15, 2)) * 1000
    Q = rng.random((15, 2)) * 1000 + 5000.0  # заведомо далеко
    val, exact = distance_within(P, Q, threshold=1.0)
    assert not exact
    assert val > 1.0
