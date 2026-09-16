"""Property tests for the continuous Frechet distance (traj.frechet_cont), per
docs/specs/step7_B_certified_store.md section 9."""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from traj.frechet import distance as discrete_distance
from traj.frechet_cont import decide, distance, distance_upper

from ._frechet_cont_mpmath import distance_mp

# Warm up numba's JIT before hypothesis starts timing examples.
distance(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[0.0, 1.0], [1.0, 0.0]]))

_coord = st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False, width=64)


def _polyline(min_pts=2, max_pts=8):
    return st.lists(st.tuples(_coord, _coord), min_size=min_pts, max_size=max_pts).map(
        lambda pts: np.array(pts, dtype=np.float64)
    )


def _insert_collinear_vertices(P: np.ndarray, rng: np.random.Generator, n_extra: int) -> np.ndarray:
    """Insert n_extra exact-linear-interpolation points on random segments of P."""
    n = P.shape[0] - 1
    inserts = []  # (segment_index, t, point)
    for _ in range(n_extra):
        seg = int(rng.integers(0, n))
        t = float(rng.uniform(0.01, 0.99))
        point = P[seg] + t * (P[seg + 1] - P[seg])
        inserts.append((seg, t, point))
    inserts.sort(key=lambda x: (x[0], x[1]))

    out = [P[0]]
    insert_idx = 0
    for seg in range(n):
        while insert_idx < len(inserts) and inserts[insert_idx][0] == seg:
            out.append(inserts[insert_idx][2])
            insert_idx += 1
        out.append(P[seg + 1])
    return np.array(out, dtype=np.float64)


def _densify(P: np.ndarray, points_per_segment: int) -> np.ndarray:
    """Resample P with extra exact-linear-interpolation points; same curve, denser."""
    n = P.shape[0] - 1
    out = [P[0]]
    for seg in range(n):
        for k in range(1, points_per_segment + 1):
            t = k / points_per_segment
            out.append(P[seg] + t * (P[seg + 1] - P[seg]))
    return np.array(out, dtype=np.float64)


def _resample_by_step(P: np.ndarray, h: float) -> np.ndarray:
    """Resample P (same curve) so consecutive points are at most h apart (arc-length),
    always keeping the original vertices. Unlike _densify's fixed per-segment count,
    this makes h a meaningful, curve-independent discretization step."""
    n = P.shape[0] - 1
    out = [P[0]]
    for seg in range(n):
        seg_len = float(np.hypot(*(P[seg + 1] - P[seg])))
        count = min(int(np.ceil(seg_len / h)) if h > 0.0 else 1, 500)
        count = max(count, 1)
        for k in range(1, count + 1):
            t = k / count
            out.append(P[seg] + t * (P[seg + 1] - P[seg]))
    return np.array(out, dtype=np.float64)


def _bbox_diagonal(*polylines: np.ndarray) -> float:
    stacked = np.vstack(polylines)
    span = np.max(stacked, axis=0) - np.min(stacked, axis=0)
    return float(np.hypot(*span))


# --- example-based sanity tests (mirror tests/traj/test_frechet.py's style) ---


def test_identical_is_zero():
    P = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [5.0, 5.0]])
    assert distance(P, P) == 0.0


def test_parallel_segments_offset_is_exact():
    P = np.array([[0.0, 0.0], [1.0, 0.0]])
    Q = np.array([[0.0, 3.0], [1.0, 3.0]])
    assert abs(distance(P, Q, tol=1e-9) - 3.0) < 1e-8


def test_single_segment_vs_zigzag_n_equals_1():
    """Exercises the n=1 (single-segment P) case, which relies on the general
    O(nm) algorithm degenerating to O(m) automatically -- no special path."""
    P = np.array([[0.0, 0.0], [10.0, 0.0]])
    Q = np.array([[0.0, 0.0], [2.0, 1.0], [4.0, -1.0], [6.0, 1.0], [8.0, -1.0], [10.0, 0.0]])
    d = distance(P, Q, tol=1e-9)
    assert abs(d - 1.0) < 1e-8
    assert decide(P, Q, 1.5)
    assert not decide(P, Q, 0.9)


def test_duplicate_consecutive_vertex_matches_without_it():
    P_dup = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    P_clean = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    Q = np.array([[0.0, 1.0], [2.0, 1.0]])
    assert abs(distance(P_dup, Q) - distance(P_clean, Q)) < 1e-6


def test_decide_zero_negative_eps_raises():
    P = np.array([[0.0, 0.0], [1.0, 0.0]])
    Q = np.array([[0.0, 1.0], [1.0, 1.0]])
    try:
        decide(P, Q, -1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- hypothesis property tests (spec section 9) ---

_settings = settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@given(P=_polyline(), Q=_polyline())
@_settings
def test_decide_distance_consistency(P, Q):
    d = distance(P, Q, tol=1e-6)
    assert decide(P, Q, d + 1e-4)
    if d > 1e-4:
        assert not decide(P, Q, d - 1e-4)


@given(P=_polyline(), Q=_polyline())
@_settings
def test_symmetry(P, Q):
    assert np.isclose(distance(P, Q), distance(Q, P), atol=1e-6)


@given(P=_polyline(), Q=_polyline(), R=_polyline())
@_settings
def test_triangle_inequality(P, Q, R):
    d_pr = distance(P, R, tol=1e-9)
    d_pq = distance(P, Q, tol=1e-9)
    d_qr = distance(Q, R, tol=1e-9)
    assert d_pr <= d_pq + d_qr + 1e-9


@given(P=_polyline(min_pts=2, max_pts=6), n_extra=st.integers(min_value=1, max_value=4), seed=st.integers(min_value=0, max_value=2**31 - 1))
@_settings
def test_reparametrization_invariance(P, n_extra, seed):
    # Mathematically the distance is exactly 0 (the inserted points are exact linear
    # interpolations, so both curves trace the same point set). It can't be asserted
    # bit-exact, or even to a tiny fixed tolerance like 1e-6/1e-9: decide()/distance()
    # are deliberately strict at the discriminant's Delta~=0 boundary (a point exactly
    # on a segment's line is a repeated root, and float64 cancellation in B*B-4*A*C can
    # push the residual a hair negative) -- reporting "not yet feasible" a fraction of
    # an eps early, never the other way, since that's the only safe direction for a
    # value used as a certified upper bound (see frechet_cont._free_interval's
    # docstring; an earlier clamp attempted to "fix" this by widening the discriminant's
    # feasible region, which instead made distance() UNDERESTIMATE d_F by ~1e-4 at
    # realistic (~1e5 m) coordinate scale -- unsafe, reverted). An empirical 1000-trial
    # sweep at this test's coordinate range (+-50) found the resulting conservative
    # slack up to ~1.5e-6; 1e-4 gives ample margin above that while still catching a
    # real regression (which would show up at the ~1e-2+ scale, see the GeoLife-scale
    # variant below for where this slack actually becomes certificate-relevant).
    rng = np.random.default_rng(seed)
    P_reparam = _insert_collinear_vertices(P, rng, n_extra)
    assert distance(P, P_reparam, tol=1e-6) < 1e-4


@given(
    P=st.lists(
        st.tuples(
            st.floats(min_value=-5e4, max_value=5e4, allow_nan=False, allow_infinity=False, width=64),
            st.floats(min_value=-5e4, max_value=5e4, allow_nan=False, allow_infinity=False, width=64),
        ),
        min_size=2,
        max_size=6,
    ).map(lambda pts: np.array(pts, dtype=np.float64)),
    n_extra=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_reparametrization_invariance_at_geolife_scale(P, n_extra, seed):
    """GeoLife's projected coordinates run to ~1e4-1e5 m (traj.io's equirectangular
    projection around a shared centroid) -- the conservative slack from strict Delta~=0
    handling scales with coordinate magnitude (an empirical sweep at this scale found
    up to ~1.5e-3 m), so this needs its own, much larger, tolerance than the small-scale
    version above. Still well within the spec's own certificate precision (eta=1mm) --
    exactly the kind of slack M1's certify.py margin (spec section 3.4) must absorb."""
    rng = np.random.default_rng(seed)
    P_reparam = _insert_collinear_vertices(P, rng, n_extra)
    assert distance(P, P_reparam, tol=1e-3) < 1e-2


@given(P=_polyline(max_pts=8), Q=_polyline(max_pts=8))
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_independent_bracket_via_discrete_frechet(P, Q):
    """Two-sided bracket on the true continuous distance from a genuinely different
    algorithm (traj.frechet's Eiter-Mannila DP, not sharing this module's free-space-
    diagram code, unlike the mpmath oracle which re-derives the same recurrence).

    Upper side: discrete Frechet distance on ANY sampling of two curves is a
    mathematically guaranteed upper bound on their continuous Frechet distance -- a
    discrete vertex-to-vertex coupling is a restricted special case of the continuous
    reparametrization space, so the DP minimizing over the smaller space can only find
    a value >= the one minimizing over the larger space. Holds at any density, so
    distance_upper (the certificate-grade bound) must not exceed it either.

    Lower side: resampling each curve so consecutive points are at most h apart means
    rounding the optimal continuous coupling to the nearest sample moves each matched
    point by at most h, so discrete Frechet on that resampling is at most continuous+h
    -- i.e. continuous >= discrete - h.
    """
    size = max(_bbox_diagonal(P, Q), 1e-9)
    h = 0.005 * size
    d_disc = discrete_distance(_resample_by_step(P, h), _resample_by_step(Q, h))
    tol = 1e-9
    # distance_upper's own decide_conservative check requires genuine clearance beyond
    # eps itself (see _free_interval_conservative), so it carries an intrinsic floor
    # proportional to the curves' own local scale, even for a zero true distance --
    # confirmed empirically: distance_upper(P, P) ~= 2.38e-7 * scale, not 0 (still
    # "microns or smaller" for realistic geometry, per the margin redesign's target,
    # but bigger than the 1e-9 bisection tol, so this comparison needs a
    # correspondingly scaled slack rather than a fixed constant).
    assert distance_upper(P, Q, tol=tol) <= d_disc + 4e-7 * size + 2e-9
    assert d_disc - h <= distance(P, Q, tol=tol) + tol


@given(P=_polyline(min_pts=2, max_pts=5), Q=_polyline(min_pts=2, max_pts=5))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_matches_mpmath_oracle(P, Q):
    d_fast = distance(P, Q, tol=1e-6)
    d_ref = float(distance_mp(P.tolist(), Q.tolist(), tol=1e-9, dps=40))
    assert abs(d_fast - d_ref) < 1e-5


@given(P=_polyline(min_pts=2, max_pts=6), Q=_polyline(min_pts=2, max_pts=6))
@_settings
def test_decide_monotone_in_eps(P, Q):
    d = distance(P, Q, tol=1e-6)
    samples = sorted({max(0.0, d - 1.0), max(0.0, d - 0.01), d + 0.01, d + 1.0, d + 10.0})
    results = [decide(P, Q, e) for e in samples]
    assert results == sorted(results)


@given(
    P=_polyline(min_pts=2, max_pts=5).map(lambda p: p * 4.0),  # modest local scale (~+-200)
    offset=st.tuples(
        st.floats(min_value=1e4, max_value=1e5) | st.floats(min_value=1e4, max_value=1e5).map(lambda v: -v),
        st.floats(min_value=1e4, max_value=1e5) | st.floats(min_value=1e4, max_value=1e5).map(lambda v: -v),
    ),
    delta_mag=st.floats(min_value=1e-3, max_value=10.0),
    delta_angle=st.floats(min_value=0.0, max_value=2 * np.pi, allow_nan=False),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_distance_upper_vs_mpmath_at_geolife_scale(P, offset, delta_mag, delta_angle):
    """distance_upper must not underestimate the trusted mpmath oracle at GeoLife-
    realistic absolute coordinate scale (offset ~1e4-1e5 m) with certificate-relevant
    eps/true-distance values (~1e-3 to 10 m). Q is P translated by a known vector, so
    the true distance is exactly delta_mag by construction -- also confirms
    translation locality (point 1): if recentering were broken, this would drift with
    the offset."""
    P_shifted = P + np.array(offset)
    delta_vec = delta_mag * np.array([np.cos(delta_angle), np.sin(delta_angle)])
    Q_shifted = P_shifted + delta_vec
    d_up = distance_upper(P_shifted, Q_shifted, tol=1e-3)
    d_ref = float(distance_mp(P_shifted.tolist(), Q_shifted.tolist(), tol=1e-6, dps=40))
    assert d_up >= d_ref - 1e-9
    assert abs(d_ref - delta_mag) < 1e-3  # confirms the closed-form construction itself


def test_distance_upper_no_floor_at_geolife_scale():
    """Regression guard for the review's mandatory correction: an earlier version of
    decide_conservative computed its margin from the overall coordinate magnitude,
    giving a ~6-12mm floor regardless of how small the true distance was -- wrong on
    principle, since Frechet distance is translation-invariant. The fix (local,
    per-cell margin from each cell's own |a-p|+|b-a|+eps, plus recentering on a common
    origin) must not reintroduce any such floor: a short segment, offset by 1e5 m,
    with a true distance of exactly 1e-3 m, should have distance_upper within
    microns of the mpmath oracle, not millimeters."""
    origin_offset = np.array([1e5, -1e5])
    P = np.array([[0.0, 0.0], [5.0, 0.0]]) + origin_offset
    delta_mag = 1e-3
    Q = P + np.array([delta_mag, 0.0])
    d_up = distance_upper(P, Q, tol=1e-9)
    d_ref = float(distance_mp(P.tolist(), Q.tolist(), tol=1e-12, dps=40))
    assert d_up - d_ref <= 1e-6
