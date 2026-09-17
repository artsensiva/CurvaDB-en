"""Property tests for traj.bezier, per docs/specs/step7_B_certified_store.md
section 9: knot insertion and Bezier extraction must not change the curve."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.interpolate import BSpline

from traj.bezier import (
    bezier_segments,
    bezier_segments_in_range,
    de_casteljau_split,
    derivative_control_points,
    evaluate_bezier,
    insert_knot,
)


def _random_bspline(rng: np.random.Generator, k: int = 3, n_interior: int = 4, n_ctrl_extra: int = 0):
    """A random valid cubic (or given degree) B-spline on domain [0, 10]."""
    interior = np.sort(rng.uniform(0.5, 9.5, n_interior)) if n_interior > 0 else np.empty(0)
    t = np.concatenate([np.full(k + 1, 0.0), interior, np.full(k + 1, 10.0)])
    n_ctrl = len(t) - k - 1 + n_ctrl_extra
    c = rng.random((n_ctrl, 2)) * 20.0 - 10.0
    return BSpline(t, c[: len(t) - k - 1], k)


_bspline_strategy = st.builds(
    lambda seed, n_interior: _random_bspline(np.random.default_rng(seed), n_interior=n_interior),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_interior=st.integers(min_value=0, max_value=6),
)

_settings = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@given(bs=_bspline_strategy, seed=st.integers(min_value=0, max_value=2**31 - 1))
@_settings
def test_insert_knot_preserves_curve(bs, seed):
    rng = np.random.default_rng(seed)
    domain_lo, domain_hi = float(bs.t[bs.k]), float(bs.t[-bs.k - 1])
    u_bar = float(rng.uniform(domain_lo, domain_hi))
    new_t, new_c = insert_knot(bs.t, bs.c, bs.k, u_bar)
    new_bs = BSpline(new_t, new_c, bs.k)

    us = np.linspace(domain_lo, domain_hi, 200)
    L = max(float(np.max(np.abs(bs.c))), 1.0)
    assert np.max(np.abs(new_bs(us) - bs(us))) <= 1e-12 * L
    assert len(new_c) == len(bs.c) + 1


@given(bs=_bspline_strategy)
@_settings
def test_bezier_segments_preserve_curve(bs):
    segments = bezier_segments(bs)
    domain_lo, domain_hi = float(bs.t[bs.k]), float(bs.t[-bs.k - 1])
    interior = np.unique(bs.t[bs.k + 1 : len(bs.t) - bs.k - 1])
    breaks = np.concatenate([[domain_lo], interior, [domain_hi]])
    assert len(segments) == len(breaks) - 1

    L = max(float(np.max(np.abs(bs.c))), 1.0)
    max_diff = 0.0
    for j, seg in enumerate(segments):
        u0, u1 = breaks[j], breaks[j + 1]
        for tt in np.linspace(0.0, 1.0, 11):
            u = u0 + tt * (u1 - u0)
            if u1 > u0:
                v_bs = bs(u)
            else:
                v_bs = bs(u0)  # degenerate zero-width span (repeated knot)
            v_bez = evaluate_bezier(seg, tt)
            max_diff = max(max_diff, float(np.max(np.abs(v_bs - v_bez))))
    assert max_diff <= 1e-12 * L


@given(
    p_degree=st.integers(min_value=1, max_value=5),
    t_split=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@_settings
def test_de_casteljau_split_matches_original(p_degree, t_split, seed):
    rng = np.random.default_rng(seed)
    P = rng.random((p_degree + 1, 2)) * 20.0 - 10.0
    left, right = de_casteljau_split(P, t_split)

    L = max(float(np.max(np.abs(P))), 1.0)
    max_diff = 0.0
    for tt in np.linspace(0.0, 1.0, 11):
        v_orig = evaluate_bezier(P, tt * t_split)
        v_left = evaluate_bezier(left, tt)
        max_diff = max(max_diff, float(np.max(np.abs(v_orig - v_left))))
    for tt in np.linspace(0.0, 1.0, 11):
        v_orig = evaluate_bezier(P, t_split + tt * (1.0 - t_split))
        v_right = evaluate_bezier(right, tt)
        max_diff = max(max_diff, float(np.max(np.abs(v_orig - v_right))))
    assert max_diff <= 1e-9 * L


@given(bs=_bspline_strategy)
@_settings
def test_derivative_control_points_match_scipy(bs):
    """Independent cross-check: the Bezier derivative's evaluation (scaled by
    1/(u1-u0) for the segment's local-to-global parameter change) must match
    scipy's own bs.derivative() on the same segment's domain."""
    segments = bezier_segments(bs)
    domain_lo, domain_hi = float(bs.t[bs.k]), float(bs.t[-bs.k - 1])
    interior = np.unique(bs.t[bs.k + 1 : len(bs.t) - bs.k - 1])
    breaks = np.concatenate([[domain_lo], interior, [domain_hi]])
    dbs = bs.derivative()

    L = max(float(np.max(np.abs(bs.c))), 1.0)
    max_diff = 0.0
    for j, seg in enumerate(segments):
        u0, u1 = breaks[j], breaks[j + 1]
        if u1 <= u0:
            continue
        dseg = derivative_control_points(seg)
        for tt in np.linspace(0.05, 0.95, 5):
            u = u0 + tt * (u1 - u0)
            v_scipy = dbs(u)
            v_bez = evaluate_bezier(dseg, tt) / (u1 - u0)
            max_diff = max(max_diff, float(np.max(np.abs(v_scipy - v_bez))))
    assert max_diff <= 1e-9 * L


@given(
    bs=_bspline_strategy,
    lo_frac=st.floats(min_value=0.0, max_value=0.9, allow_nan=False),
    span_frac=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
)
@_settings
def test_bezier_segments_in_range_preserves_curve(bs, lo_frac, span_frac):
    """spec section 9 (M2): restricting bezier_segments to an arbitrary [u_lo, u_hi]
    (not necessarily knot-aligned) must reproduce the same curve, dense-sampled,
    as evaluating bs directly."""
    domain_lo, domain_hi = float(bs.t[bs.k]), float(bs.t[-bs.k - 1])
    span = domain_hi - domain_lo
    u_lo = domain_lo + lo_frac * span
    u_hi = min(u_lo + span_frac * span * (1.0 - lo_frac), domain_hi)
    if u_hi <= u_lo:
        return

    segments = bezier_segments_in_range(bs, u_lo, u_hi)

    t_after = np.asarray(bs.t, dtype=float)
    interior_between = np.unique(t_after[(t_after > u_lo) & (t_after < u_hi)])
    breaks = np.concatenate([[u_lo], interior_between, [u_hi]])
    assert len(segments) == len(breaks) - 1

    L = max(float(np.max(np.abs(bs.c))), 1.0)
    max_diff = 0.0
    for j, seg in enumerate(segments):
        u0, u1 = breaks[j], breaks[j + 1]
        if u1 <= u0:
            continue
        for tt in np.linspace(0.0, 1.0, 11):
            u = u0 + tt * (u1 - u0)
            v_bs = bs(u)
            v_bez = evaluate_bezier(seg, tt)
            max_diff = max(max_diff, float(np.max(np.abs(v_bs - v_bez))))
    assert max_diff <= 1e-9 * L


def test_bezier_segments_in_range_full_domain_matches_bezier_segments():
    rng = np.random.default_rng(7)
    bs = _random_bspline(rng, n_interior=4)
    domain_lo, domain_hi = float(bs.t[bs.k]), float(bs.t[-bs.k - 1])
    a = bezier_segments_in_range(bs, domain_lo, domain_hi)
    b = bezier_segments(bs)
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        assert np.allclose(sa, sb)


def test_bezier_segments_in_range_rejects_zero_width():
    rng = np.random.default_rng(3)
    bs = _random_bspline(rng, n_interior=2)
    u = 0.5 * (float(bs.t[bs.k]) + float(bs.t[-bs.k - 1]))
    with pytest.raises(ValueError):
        bezier_segments_in_range(bs, u, u)


def test_insert_knot_raises_multiplicity():
    """A repeated insertion at the same value raises that knot's multiplicity by
    one each time, without changing the curve."""
    k = 3
    t = np.array([0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 10.0, 10.0, 10.0, 10.0])
    c = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, -1.0], [3.0, 3.0], [4.0, 0.0], [5.0, 1.0]])
    bs = BSpline(t, c, k)
    us = np.linspace(0.0, 10.0, 100)
    orig = bs(us)

    new_t, new_c = insert_knot(t, c, k, 3.0)
    assert np.sum(new_t == 3.0) == 2
    new_bs = BSpline(new_t, new_c, k)
    assert np.max(np.abs(new_bs(us) - orig)) <= 1e-12 * np.max(np.abs(c))
