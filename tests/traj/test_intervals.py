"""Tests for traj.intervals: section 2.5's interval queries, per
docs/specs/step7_B_certified_store.md. Spec section 9's own mandatory property test:
the interval must contain the exact distance between the ORIGINAL polylines on 10,000
random pairs."""

from __future__ import annotations

import numpy as np

from traj.frechet_cont import distance
from traj.intervals import bbox_lower_bound, cheap_lower_bound, decide_range, endpoint_lower_bound


def _random_polyline(rng: np.random.Generator, n_min: int = 2, n_max: int = 8, scale: float = 50.0) -> np.ndarray:
    n = int(rng.integers(n_min, n_max + 1))
    return rng.uniform(-scale, scale, size=(n, 2))


def _perturbed(rng: np.random.Generator, raw: np.ndarray, max_amp: float) -> tuple[np.ndarray, float]:
    """A "representation" of `raw`: same-index perturbation (identity correspondence),
    each point moved by up to `max_amp`. Returns (representation, honest_eps), where
    honest_eps is the perturbation's OWN max per-point norm -- a valid upper bound on
    d_F(raw, representation) via the identity correspondence itself (a valid, if not
    necessarily optimal, monotone coupling), not an assumed/made-up certificate."""
    delta = rng.uniform(-max_amp, max_amp, size=raw.shape) / np.sqrt(2.0)
    rep = raw + delta
    eps = float(np.max(np.hypot(*(rep - raw).T))) if len(raw) else 0.0
    return rep, eps


def test_interval_contains_true_distance_10000_pairs():
    """spec section 9: for 10,000 random (query, candidate) pairs, the interval
    [D-sum_eps, D+sum_eps] (D = d_F between the two "representations", sum_eps =
    the sum of two independently-honest certificates) must contain the TRUE d_F
    between the ORIGINAL (uncompressed) polylines."""
    rng = np.random.default_rng(20260918)
    for _ in range(10_000):
        q_raw = _random_polyline(rng)
        A_raw = _random_polyline(rng)
        q_lin, eps_q = _perturbed(rng, q_raw, max_amp=float(rng.uniform(0.0, 5.0)))
        A_lin, eps_A = _perturbed(rng, A_raw, max_amp=float(rng.uniform(0.0, 5.0)))
        sum_eps = eps_q + eps_A

        D = distance(q_lin, A_lin, tol=1e-6)
        true_d = distance(q_raw, A_raw, tol=1e-6)
        assert D - sum_eps - 1e-4 <= true_d <= D + sum_eps + 1e-4


def test_endpoint_lower_bound_is_valid():
    rng = np.random.default_rng(1)
    for _ in range(500):
        P = _random_polyline(rng)
        Q = _random_polyline(rng)
        lb = endpoint_lower_bound(P, Q)
        true_d = distance(P, Q, tol=1e-6)
        assert lb <= true_d + 1e-9


def test_bbox_lower_bound_is_valid():
    rng = np.random.default_rng(2)
    for _ in range(500):
        P = _random_polyline(rng)
        Q = _random_polyline(rng)
        lb = bbox_lower_bound(P, Q)
        true_d = distance(P, Q, tol=1e-6)
        assert lb <= true_d + 1e-9


def test_decide_range_outcomes_are_consistent_with_ground_truth():
    """accept must imply true d_F(q,A) <= r; either rejection outcome must imply
    true d_F(q,A) > r; refine makes no promise either way (that's the point)."""
    rng = np.random.default_rng(3)
    n_accept = n_reject = n_refine = 0
    for _ in range(3_000):
        q_raw = _random_polyline(rng)
        A_raw = _random_polyline(rng)
        q_lin, eps_q = _perturbed(rng, q_raw, max_amp=float(rng.uniform(0.0, 5.0)))
        A_lin, eps_A = _perturbed(rng, A_raw, max_amp=float(rng.uniform(0.0, 5.0)))
        sum_eps = eps_q + eps_A
        r = float(rng.uniform(0.0, 80.0))

        cheap_lb = cheap_lower_bound(q_lin, A_lin)
        outcome = decide_range(q_lin, A_lin, r, sum_eps, cheap_lb=cheap_lb)
        true_d = distance(q_raw, A_raw, tol=1e-6)

        if outcome == "accept":
            n_accept += 1
            assert true_d <= r + 1e-4
        elif outcome in ("reject_cheap", "reject_interval"):
            n_reject += 1
            assert true_d > r - 1e-4
        else:
            assert outcome == "refine"
            n_refine += 1

    assert n_accept + n_reject + n_refine == 3_000
    # sanity: at these random r/eps scales, all three outcomes should occur
    assert n_accept > 0 and n_reject > 0 and n_refine > 0


def test_decide_range_cheap_filter_rejects_without_decide():
    """Two polylines far enough apart that the endpoint/bbox filter alone exceeds
    r + sum_eps must be rejected as "reject_cheap", not "reject_interval" -- the
    two are reported separately (mandatory correction 2)."""
    P = np.array([[0.0, 0.0], [1.0, 0.0]])
    Q = np.array([[1000.0, 0.0], [1001.0, 0.0]])
    cheap_lb = cheap_lower_bound(P, Q)
    assert cheap_lb > 10.0  # well above any r+sum_eps used below
    outcome = decide_range(P, Q, r=5.0, sum_eps=1.0, cheap_lb=cheap_lb)
    assert outcome == "reject_cheap"


def test_decide_range_negative_lower_threshold_never_calls_decide_with_negative_eps():
    """r - sum_eps < 0 must not raise (decide() rejects a negative eps) -- the
    accept check is skipped outright in that case."""
    P = np.array([[0.0, 0.0], [1.0, 0.0]])
    Q = np.array([[0.0, 0.0], [1.0, 0.1]])
    outcome = decide_range(P, Q, r=1.0, sum_eps=5.0)  # r - sum_eps = -4.0
    assert outcome in ("accept", "reject_cheap", "reject_interval", "refine")
