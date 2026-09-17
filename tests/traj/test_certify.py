"""Tests for traj.certify: section 2.2 (exact polyline certificate), section 2.3
(monotone projection-matching spline certificate, M2), and section 2.4 (certified
linearization, the spline fallback path, M1), per
docs/specs/step7_B_certified_store.md. Fast/small-scale here; the full-corpus S1/S2
validation lives in benchmarks/step7_certify.py (spec section 5's own file layout)."""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.interpolate import BSpline, make_lsq_spline

from traj.certify import (
    _certify_projection_piece,
    certified_linearize,
    certify_polyline,
    certify_spline,
    certify_spline_linearization,
    certify_spline_projection,
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


# --- M1.1 property tests (docs/reviews/step7_M1.md): root-splitting + small-ball ---


def _single_segment_bspline(control_points: np.ndarray) -> BSpline:
    k = control_points.shape[0] - 1
    t = np.concatenate([np.zeros(k + 1), np.ones(k + 1)])
    return BSpline(t, control_points, k)


def _assert_certified_and_within_lam(control_points, lam, slack=1e-2):
    """Shared check for the non-adversarial M1.1 cases: must certify, and the
    certified linearization must genuinely be within lam of the spline on a dense
    sample (not just trusted because fully_certified says so)."""
    bs = _single_segment_bspline(control_points)
    lin_vertices, fully_certified = certified_linearize(bs, lam, max_levels=12)
    assert fully_certified is True
    dense = bs(np.linspace(0.0, 1.0, 300))
    d = distance_upper(dense, lin_vertices, tol=1e-4)
    assert d <= lam + slack


def test_certified_linearize_reversal():
    """Genuine sign change in the chord-projected derivative (confirmed this
    session: derivative-projection Bernstein coefficients [6,-9,6], two real
    roots at u ~ 0.276, 0.724) -- fails _certified_ok's monotonicity check at any
    lam under the OLD (blind-bisection) algorithm, which needs ~20 levels to
    resolve; root-splitting resolves it in 2 levels at lam=0.5 (verified directly:
    _certify_segment with levels_left=1 still fails, levels_left=2 succeeds --
    the boundary-margin guard in _certify_segment's docstring means a single
    split doesn't always suffice even when clean roots exist, but it's still a
    large improvement over ~20)."""
    seg = np.array([[0.0, 0.0], [2.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    _assert_certified_and_within_lam(seg, lam=0.5)


def test_certified_linearize_stop():
    """Near-duplicate control points (small but non-degenerate chord) with a real,
    bounded wander -- exercises the small-ball rule (2*rho <= lam)."""
    seg = np.array([[0.0, 0.0], [0.05, 0.08], [-0.06, 0.04], [0.001, 0.0]])
    a, b = seg[0], seg[-1]
    assert 1e-12 <= float(np.hypot(*(b - a))) < 1e-2  # non-degenerate but tiny chord
    _assert_certified_and_within_lam(seg, lam=0.2)


def test_certified_linearize_loop():
    """Exactly-degenerate chord (closes back to its own start) -- exercises the
    existing curve-vs-point path; confirms root-splitting correctly skips this
    case (chord_len < 1e-12 guard) rather than attempting a spurious split."""
    seg = np.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 2.0], [0.0, 0.0]])
    _assert_certified_and_within_lam(seg, lam=2.5)


def test_certified_linearize_near_degenerate_loop():
    """User's mandatory addition (correction 3): seg[0] and seg[-1] at distance
    1e-6 (NOT degenerate by the 1e-12 threshold, so the normal chord/unit-direction
    path runs), with the loop's actual extent (rho) larger than lam/2 (so the
    small-ball rule does NOT trivially apply either) -- the numerically hardest
    case, where the chord direction is technically well-defined but practically
    ill-conditioned. Only requires: no crash/hang, and the result is either
    fully_certified with an independently-verified correct bound, or an honest
    fully_certified=False -- never a silent wrong answer."""
    lam = 0.2
    seg = np.array([[0.0, 0.0], [0.3, 0.25], [-0.28, 0.22], [1e-6, 0.0]])
    a, b = seg[0], seg[-1]
    chord_len = float(np.hypot(*(b - a)))
    rho = max(float(np.hypot(*(p - a))) for p in seg)
    assert chord_len >= 1e-12  # not treated as degenerate
    assert 2.0 * rho > lam  # small-ball rule does not trivially apply

    bs = _single_segment_bspline(seg)
    lin_vertices, fully_certified = certified_linearize(bs, lam, max_levels=12)  # must not hang/crash

    if fully_certified:
        dense = bs(np.linspace(0.0, 1.0, 300))
        d = distance_upper(dense, lin_vertices, tol=1e-4)
        assert d <= lam + 1e-2
    else:
        assert fully_certified is False  # honest refusal is an acceptable outcome


# --- M2 property tests (spec section 9): section 2.3, monotone projection matching ---


def _random_spline(rng: np.random.Generator, k: int = 3, n_interior: int = 4):
    interior = np.sort(rng.uniform(1.0, 9.0, n_interior)) if n_interior > 0 else np.empty(0)
    knots = np.concatenate([np.full(k + 1, 0.0), interior, np.full(k + 1, 10.0)])
    n_ctrl = len(knots) - k - 1
    c = rng.random((n_ctrl, 2)) * 20.0 - 10.0
    return BSpline(knots, c, k)


@given(
    n_interior=st.integers(min_value=0, max_value=5),
    n_points=st.integers(min_value=4, max_value=15),
    noise=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@_settings
def test_certify_spline_projection_matches_dense_sampling(n_interior, n_points, noise, seed):
    """spec section 9: for a certified piece, the explicit correspondence (spec
    2.3 item 3's clamp formula), sampled densely by parameter, has cost <= eps_A;
    independently, the sampled projection s(u) is non-decreasing (spec 2.3 item 2's
    monotonicity condition, checked here directly on samples, not just via the
    analytic Bernstein-coefficient argument certify_spline_projection used)."""
    rng = np.random.default_rng(seed)
    bs = _random_spline(rng, n_interior=n_interior)
    t_pts = np.linspace(0.0, 10.0, n_points)
    A = np.asarray(bs(t_pts)) + rng.normal(0.0, noise, (n_points, 2))

    res = certify_spline_projection(A, bs, max_levels=12)

    for p in res.pieces:
        if p.u_k1 <= p.u_k or p.L_k < 1e-12 or not p.certified:
            continue
        V_k = A[p.k]
        e_k, L_k = p.e_k, p.L_k
        grid = np.linspace(p.u_k, p.u_k1, 300)
        prev_s = None
        for u in grid:
            Cu = np.asarray(bs(u))
            s = float((Cu - V_k) @ e_k)
            if prev_s is not None:
                assert s >= prev_s - 1e-6 * max(L_k, 1.0)
            prev_s = s
            matched = V_k + float(np.clip(s, 0.0, L_k)) * e_k
            cost = float(np.hypot(*(Cu - matched)))
            assert cost <= res.eps_A + 1e-6


@given(n_interior=st.integers(min_value=0, max_value=5), n_points=st.integers(min_value=4, max_value=15), seed=st.integers(min_value=0, max_value=2**31 - 1))
@_settings
def test_certify_spline_never_returns_finite_eps_a_when_uncertified(n_interior, n_points, seed):
    """certify_spline's contract: either a method succeeded (2.3 or 2.4 fallback)
    with a finite eps_A, or even the fallback failed (2.4_fallback_uncertified)
    and eps_A is inf -- never a finite number without a method that backs it."""
    rng = np.random.default_rng(seed)
    bs = _random_spline(rng, n_interior=n_interior)
    t_pts = np.linspace(0.0, 10.0, n_points)
    A = np.asarray(bs(t_pts)) + rng.normal(0.0, 0.5, (n_points, 2))

    eps_A, method = certify_spline(A, bs, max_levels=12)
    assert method in ("2.3", "2.4_fallback", "2.4_fallback_uncertified")
    if method == "2.4_fallback_uncertified":
        assert eps_A == float("inf")
    else:
        assert 0.0 <= eps_A < float("inf")


def test_certify_projection_piece_degenerate_point_vs_segment():
    """Mandatory correction 2: u_k == u_{k+1} (the correspondence search's
    monotonicity clamp bound it -- e.g. two source vertices very close together
    relative to the spline's own curvature). The piece degenerates to the single
    spline point C(u_k) matched against the WHOLE segment S_k; cost must equal
    the exact max(|C(u_k)-V_k|, |C(u_k)-V_{k+1}|), and every point on S_k must be
    within that cost of C(u_k) (this piece's own dense-sampling check, since S_k
    itself has no interior parameter to vary)."""
    bs = _random_spline(np.random.default_rng(11), n_interior=3)
    A = np.array([[0.0, 0.0], [3.0, 4.0], [3.5, 4.2], [10.0, 0.0]])
    u = np.array([0.0, 0.2, 0.2, 10.0])  # u_1 == u_2 by construction
    delta = np.array([float(np.hypot(*(np.asarray(bs(uu)) - A[i]))) for i, uu in enumerate(u)])

    p = _certify_projection_piece(bs, A, u, delta, 1, max_levels=12)
    assert p.degenerate == "point_vs_segment"
    assert p.certified is True

    Cu = np.asarray(bs(u[1]))
    V_k, V_k1 = A[1], A[2]
    expected = max(float(np.hypot(*(Cu - V_k))), float(np.hypot(*(Cu - V_k1))))
    assert abs(p.cost - expected) < 1e-9
    for tt in np.linspace(0.0, 1.0, 50):
        Q = V_k + tt * (V_k1 - V_k)
        assert float(np.hypot(*(Q - Cu))) <= p.cost + 1e-9


def test_certify_projection_piece_degenerate_curve_vs_point():
    """Mandatory correction 2: |S_k| < 1e-12 (two coincident source vertices).
    The piece is matched against the single point V_k; cost must equal
    max_i |P_i - V_k| over every control point of every leaf Bezier segment in
    the piece, and this must genuinely bound the dense-sampled curve-to-point
    distance across the whole piece (this piece's own dense-sampling check)."""
    bs = _random_spline(np.random.default_rng(5), n_interior=3)
    A = np.array([[0.0, 0.0], [2.0, 3.0], [2.0, 3.0], [10.0, 0.0]])  # V_1 == V_2
    u = np.array([0.0, 0.3, 0.7, 10.0])  # u_1 != u_2 -- a genuine curve extent
    delta = np.array([float(np.hypot(*(np.asarray(bs(uu)) - A[i]))) for i, uu in enumerate(u)])

    p = _certify_projection_piece(bs, A, u, delta, 1, max_levels=12)
    assert p.degenerate == "curve_vs_point"
    assert p.certified is True

    V_k = A[1]
    for uu in np.linspace(u[1], u[2], 300):
        Cu = np.asarray(bs(uu))
        assert float(np.hypot(*(Cu - V_k))) <= p.cost + 1e-9
