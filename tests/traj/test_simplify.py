"""Tests for traj.simplify: degenerate-time guard (ADR-0020, same class of bug
as ADR-0014's dense_max_error mode/domain mismatch -- a time-aware function
silently returning a meaningless-but-plausible result on a degenerate time
axis instead of failing loudly)."""

from __future__ import annotations

import numpy as np
import pytest

from traj.io import Track
from traj.simplify import simplify_sed_with_indices


def _track(t: np.ndarray, xy: np.ndarray) -> Track:
    return Track(track_id="t", lat=np.zeros(len(xy)), lon=np.zeros(len(xy)), t=t, xy=xy)


def _wiggly_xy(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = np.column_stack([np.linspace(0.0, 100.0, n), np.zeros(n)])
    xy[1:-1, 1] += rng.normal(0.0, 5.0, n - 2)
    return xy


def test_simplify_sed_rejects_zero_span_time():
    """The exact bug found in benchmarks/step7_m4.py's _polyline_repr: a
    constant (all-zero) time array. On the pre-guard code this silently
    degrades SED's synchronized-position interpolation (every point's
    "synchronized" target collapses to the segment's own start point, since
    frac = (t[i]-t0)/(t1-t0) hits the t1<=t0 fallback everywhere) instead of
    raising -- this test fails on that code."""
    n = 20
    xy = _wiggly_xy(n)
    t_zero = np.zeros(n)
    with pytest.raises(ValueError, match="zero or negative span"):
        simplify_sed_with_indices(_track(t_zero, xy), tol=5.0)


def test_simplify_sed_rejects_non_monotonic_time():
    n = 20
    xy = _wiggly_xy(n, seed=1)
    t = np.linspace(0.0, 100.0, n)
    t[10], t[11] = t[11], t[10]  # swap -> a later index at an earlier time
    with pytest.raises(ValueError, match="not non-decreasing"):
        simplify_sed_with_indices(_track(t, xy), tol=5.0)


def test_simplify_sed_accepts_genuine_increasing_time():
    n = 20
    xy = _wiggly_xy(n, seed=2)
    t = np.linspace(0.0, 100.0, n)
    kept_xy, kept_idx = simplify_sed_with_indices(_track(t, xy), tol=5.0)
    assert len(kept_idx) >= 2
    assert kept_idx[0] == 0 and kept_idx[-1] == n - 1


def test_simplify_sed_accepts_two_point_zero_span_track():
    """A 2-point track has no interior point for _seds to interpolate at all
    (the recursion's `i1 - i0 < 2` guard skips it immediately) -- a degenerate
    (equal) t is harmless here, so the length-2 case is exempted rather than
    rejected."""
    xy = np.array([[0.0, 0.0], [1.0, 1.0]])
    t = np.array([0.0, 0.0])
    kept_xy, kept_idx = simplify_sed_with_indices(_track(t, xy), tol=5.0)
    assert list(kept_idx) == [0, 1]
