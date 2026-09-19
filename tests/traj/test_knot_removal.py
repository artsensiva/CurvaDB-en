"""Tests for traj.knot_removal (step8 spec section 2.4, ADR-0023): the
mandatory property from spec section 9 that's this module's own job --
"knot removal never returns a representation with certified error above
tol." DP-vs-brute-force (spec 9's other knot-removal-adjacent property)
belongs to M2's dynamic program, not this module."""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from traj.knot_removal import remove_knots


def _smooth_curve(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 100.0, n)
    theta = t / 100.0 * (2.0 * np.pi)
    r = 200.0 + 5.0 * np.sin(3.0 * theta)
    xy = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    xy += rng.normal(0.0, 1e-3, xy.shape)  # break exact degeneracies, negligible vs. any tol used
    return t, xy


def test_remove_knots_never_exceeds_tol_at_loose_tol():
    t, xy = _smooth_curve()
    tol = 5.0
    result = remove_knots(t, xy, xy, target_tol=tol)
    assert result.converged
    assert result.eps_A <= tol


def test_remove_knots_actually_removes_knots_at_loose_tol():
    """A smooth, gentle curve at a generous tol should end up needing fewer
    knots than the dense starting fit -- otherwise removal isn't doing
    anything."""
    t, xy = _smooth_curve()
    result = remove_knots(t, xy, xy, target_tol=5.0)
    assert result.converged
    assert result.n_removed > 0


def test_remove_knots_tight_tol_still_certifies_or_reports_unconverged():
    """At a very tight tol, remove_knots may not converge (the starting
    dense fit itself might not certify) -- but it must never silently
    report a passing result with eps_A > tol."""
    t, xy = _smooth_curve(n=60)
    result = remove_knots(t, xy, xy, target_tol=1e-6)
    if result.converged:
        assert result.eps_A <= 1e-6
    else:
        assert result.eps_A > 1e-6


def test_remove_knots_reconstruction_matches_points_within_tol():
    t, xy = _smooth_curve()
    tol = 3.0
    result = remove_knots(t, xy, xy, target_tol=tol)
    assert result.converged
    recon = result.bs(result.u)
    err = float(np.max(np.hypot(*(recon - xy).T)))
    assert err <= tol + 1e-9


@given(seed=st.integers(min_value=0, max_value=1000))
@settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_remove_knots_property_never_exceeds_tol(seed):
    t, xy = _smooth_curve(n=80, seed=seed)
    tol = 4.0
    result = remove_knots(t, xy, xy, target_tol=tol)
    if result.converged:
        assert result.eps_A <= tol
