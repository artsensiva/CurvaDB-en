"""Tests for traj.certify: section 2.2 (exact polyline certificate) and section 2.4
(certified linearization, the spline fallback path), per
docs/specs/step7_B_certified_store.md. Fast/small-scale here; the full-corpus S1/S2
validation lives in benchmarks/step7_certify.py (spec section 5's own file layout)."""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.interpolate import BSpline, make_lsq_spline

from traj.certify import (
    certified_linearize,
    certify_polyline,
    certify_spline_linearization,
    hausdorff_lower_bound,
)
from traj.frechet_cont import distance_upper

from ._frechet_cont_mpmath import distance_mp

_coord = st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False, width=64)


def _polyline(min_pts=4, max_pts=12):
    return st.lists(st.tuples(_coord, _coord), min_size=min_pts, max_size=max_pts).map(
        lambda pts: np.array(pts, dtype=np.float64)
    )


def _kept_indices(rng: np.random.Generator, n: int) -> np.ndarray:
    """A random subset of {0, ..., n-1} always including both endpoints."""
    n_extra = int(rng.integers(0, max(n - 2, 1)))
    middle = rng.choice(np.arange(1, n - 1), size=min(n_extra, max(n - 2, 0)), replace=False) if n > 2 else np.empty(0, dtype=int)
    kept = np.unique(np.concatenate([[0], middle, [n - 1]])).astype(int)
    return kept


_settings = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@given(A=_polyline(min_pts=4, max_pts=10), seed=st.integers(min_value=0, max_value=2**31 - 1))
@_settings
def test_certify_polyline_matches_mpmath_per_piece(A, seed):
    """Genuine independent high-precision verification at small scale: eps_A must
    be >= the mpmath reference for the worst piece (chain vs. its own single
    segment), matching S1's criterion (spec section 8), minus a small slack for
    distance_upper's own tol and the margin's own (tiny) size."""
    rng = np.random.default_rng(seed)
    kept = _kept_indices(rng, len(A))
    eps_A = certify_polyline(A, kept, eta=1e-6)

    worst_ref = 0.0
    for k in range(len(kept) - 1):
        i0, i1 = int(kept[k]), int(kept[k + 1])
        piece = A[i0 : i1 + 1]
        segment = A[[i0, i1]]
        ref = float(distance_mp(piece.tolist(), segment.tolist(), tol=1e-9, dps=40))
        worst_ref = max(worst_ref, ref)
    assert eps_A >= worst_ref - 1e-5


@given(A=_polyline(min_pts=4, max_pts=10), seed=st.integers(min_value=0, max_value=2**31 - 1))
@_settings
def test_certify_polyline_bounds_bracket_lower_bound(A, seed):
    """eps_A must never fall below the Hausdorff-based lower bound (spec section
    2.5) -- a coarser, cheaper sanity check applicable even where mpmath isn't run,
    and the same quantity benchmarks/step7_certify.py reports as eps_A/LB."""
    rng = np.random.default_rng(seed)
    kept = _kept_indices(rng, len(A))
    eps_A = certify_polyline(A, kept, eta=1e-6)
    lb = hausdorff_lower_bound(A, A[kept])
    assert eps_A >= lb - 1e-6


def test_certify_spline_linearization_inf_when_uncertified():
    """Item 0.4 (docs/reviews/step7_M0.md): must return (inf, False), never a
    finite eps_A, when the linearization at the certificate's own lam/max_levels
    isn't fully certified. Forced via a deliberately tiny max_levels."""
    rng = np.random.default_rng(1)
    t_pts = np.linspace(0.0, 20.0, 40)
    theta = t_pts / 20.0 * 3 * np.pi
    xy_true = np.column_stack([50.0 * np.cos(theta), 50.0 * np.sin(theta)])
    xy_noisy = xy_true + rng.normal(0.0, 0.3, xy_true.shape)
    k = 3
    internal = t_pts[5:-5:5]
    full_knots = np.concatenate([np.full(k + 1, t_pts[0]), internal, np.full(k + 1, t_pts[-1])])
    bs = make_lsq_spline(t_pts, xy_noisy, full_knots, k=k)

    eps_A, ok = certify_spline_linearization(xy_noisy, bs, lam=1e-6, eta=1e-3, max_levels=1)
    assert ok is False
    assert eps_A == float("inf")


def test_certify_spline_linearization_finite_when_certified():
    rng = np.random.default_rng(2)
    t_pts = np.linspace(0.0, 20.0, 40)
    theta = t_pts / 20.0 * 3 * np.pi
    xy_true = np.column_stack([50.0 * np.cos(theta), 50.0 * np.sin(theta)])
    xy_noisy = xy_true + rng.normal(0.0, 0.3, xy_true.shape)
    k = 3
    internal = t_pts[5:-5:5]
    full_knots = np.concatenate([np.full(k + 1, t_pts[0]), internal, np.full(k + 1, t_pts[-1])])
    bs = make_lsq_spline(t_pts, xy_noisy, full_knots, k=k)

    eps_A, ok = certify_spline_linearization(xy_noisy, bs, lam=1.0, eta=1e-3, max_levels=12)
    assert ok is True
    assert 0.0 < eps_A < float("inf")


@given(
    lam=st.floats(min_value=1e-3, max_value=2.0, allow_nan=False),
    n_interior=st.integers(min_value=0, max_value=6),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@_settings
def test_certified_linearize_within_lam_of_spline(lam, n_interior, seed):
    """When fully_certified, Lin(A') really must lie within lam of the spline
    (spec section 2.4's own contract) -- checked directly against a dense sample
    of the spline itself, at points not necessarily on Lin(A')'s own vertices."""
    rng = np.random.default_rng(seed)
    k = 3
    interior = np.sort(rng.uniform(1.0, 9.0, n_interior)) if n_interior > 0 else np.empty(0)
    t = np.concatenate([np.full(k + 1, 0.0), interior, np.full(k + 1, 10.0)])
    n_ctrl = len(t) - k - 1
    c = rng.random((n_ctrl, 2)) * 20.0 - 10.0
    bs = BSpline(t, c, k)

    lin_vertices, fully_certified = certified_linearize(bs, lam, max_levels=12)
    if not fully_certified:
        return  # nothing to check; certify_spline_linearization handles this via inf

    us = np.linspace(0.0, 10.0, 300)
    dense = bs(us)
    d = distance_upper(dense, lin_vertices, tol=1e-3)
    assert d <= lam + 1e-2  # small slack: distance_upper's own tol + de Casteljau's discretization of "dense"
