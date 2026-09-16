"""Property tests for the continuous Frechet distance (traj.frechet_cont), per
docs/specs/step7_B_certified_store.md section 9."""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from traj.frechet_cont import decide, distance

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
    # bit-exact: floating-point interpolation isn't bit-exact collinear, leaving ~1e-15
    # noise that makes decide(P, Q, 0.0) legitimately return False. 1e-9 is tight enough
    # to catch a broken/missing early-return (which would surface at the ~1e-6 tol scale
    # or a coordinate-scale error), while tolerating that noise.
    rng = np.random.default_rng(seed)
    P_reparam = _insert_collinear_vertices(P, rng, n_extra)
    assert distance(P, P_reparam, tol=1e-9) < 1e-6


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
