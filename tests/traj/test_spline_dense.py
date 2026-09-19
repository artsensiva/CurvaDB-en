"""Honest fitting contract: error <= tol on a DENSE grid between
neighboring points, not just at the points themselves (see
src/traj/spline.py, benchmarks/results/step0_diagnostics.md)."""

import numpy as np
import pytest

from traj.io import Track
from traj.spline import dense_check, dense_max_error, fit
from traj.spline_lsq import fit_adaptive

TOL = 10.0


def _sharp_turn_track(seed: int = 0) -> Track:
    """A sharp ~90° turn + uneven dt (1..25s) -- a case where the
    interpolating spline tends to overshoot between sparse knots near the
    corner."""
    rng = np.random.default_rng(seed)
    n1, n2 = 25, 25
    leg1 = np.column_stack([np.linspace(0, 500, n1), np.zeros(n1)])
    leg2 = np.column_stack([np.full(n2, 500.0), np.linspace(0, 500, n2)])
    xy = np.vstack([leg1, leg2[1:]]) + rng.normal(0.0, 1.0, (n1 + n2 - 1, 2))
    n = len(xy)
    dt = rng.uniform(1.0, 25.0, size=n - 1)
    t = np.concatenate([[0.0], np.cumsum(dt)])
    zeros = np.zeros(n)
    return Track(track_id="sharp_turn", lat=zeros, lon=zeros, t=t, xy=xy)


def _stop_and_go_track(seed: int = 0) -> Track:
    """Random heading changes + clusters of nearby points (stops), dt
    within 1..29s (as after cleaning by src/traj/clean.py, where dt <=
    30s -- no split occurs)."""
    rng = np.random.default_rng(seed)
    pos = np.array([0.0, 0.0])
    t_cursor = 0.0
    times = [0.0]
    points = [pos.copy()]
    for _ in range(8):
        heading = rng.uniform(0, 2 * np.pi)
        step = rng.uniform(20.0, 150.0)
        pos = pos + step * np.array([np.cos(heading), np.sin(heading)])
        t_cursor += rng.uniform(5.0, 29.0)
        points.append(pos.copy() + rng.normal(0.0, 1.0, 2))
        times.append(t_cursor)
        for _ in range(int(rng.integers(2, 5))):
            t_cursor += rng.uniform(1.0, 5.0)
            points.append(pos.copy() + rng.normal(0.0, 0.5, 2))
            times.append(t_cursor)
    xy = np.array(points)
    t = np.array(times)
    n = len(xy)
    zeros = np.zeros(n)
    return Track(track_id="stop_and_go", lat=zeros, lon=zeros, t=t, xy=xy)


def test_dense_error_within_tol_sharp_turn():
    for seed in range(5):
        track = _sharp_turn_track(seed)
        sp = fit(track, tol=TOL)
        max_err, _ = dense_check(track, sp)
        assert max_err <= TOL + 1e-6, f"seed={seed} max_err={max_err}"


def test_dense_error_within_tol_stop_and_go():
    for seed in range(5):
        track = _stop_and_go_track(seed)
        sp = fit(track, tol=TOL)
        max_err, _ = dense_check(track, sp)
        assert max_err <= TOL + 1e-6, f"seed={seed} max_err={max_err}"


def test_densification_can_trigger():
    """For at least some hard tracks the fitter really does add synthetic
    knots (n_points_used > n_raw) -- checks that the knot-adding code path
    actually executes, not just that it's unreachable."""
    triggered = False
    for seed in range(10):
        track = _sharp_turn_track(seed)
        sp = fit(track, tol=TOL)
        if sp.n_points_used > len(track.t):
            triggered = True
            break
    assert triggered


def test_sp_max_error_matches_dense_check():
    """sp.max_error, computed inside fit(), must match the independent
    dense_check on the same track."""
    track = _sharp_turn_track(seed=1)
    sp = fit(track, tol=TOL)
    max_err, _ = dense_check(track, sp)
    assert abs(sp.max_error - max_err) < 1e-6


def test_dense_max_error_rejects_mismatched_domain():
    """ADR-0014: spline_lsq.py's LsqSplineFit.tck has knots in REAL time
    (make_lsq_spline fits directly on t, unlike splprep's normalized-u
    convention). Calling dense_max_error with mode="time" on such a tck --
    exactly the mistake benchmarks/step7_certify.py's fit_validity() made
    -- evaluates the spline in the wrong sliver of its domain instead of
    across the real track; it must raise, not silently return a wrong
    number."""
    track = _sharp_turn_track(seed=2)
    lsq_fit = fit_adaptive(track, tol=TOL)
    with pytest.raises(ValueError, match="doesn't match mode"):
        dense_max_error(track.t, track.xy, lsq_fit.tck, mode="time")
    # mode="raw" is the correct call for this tck -- must NOT raise.
    dense_max_error(track.t, track.xy, lsq_fit.tck, mode="raw")
