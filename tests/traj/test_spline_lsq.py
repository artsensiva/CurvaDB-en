"""Smoke tests for the LSQ spline (src/traj/spline_lsq.py): a ground-truth
curve fitter with no tethering to the polyline -- the honest error is the
direct residual at the points (fit_uniform/fit_adaptive) or the error on a
dense grid against the ground truth (fit_oracle)."""

import numpy as np

from traj.io import Track
from traj.spline_lsq import fit_adaptive, fit_oracle, fit_uniform, reconstruct


def _noisy_arc_track(n: int = 120, noise_std: float = 0.5, seed: int = 0) -> Track:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 100.0, n)
    theta = t / 100.0 * 3 * np.pi
    xy_true = np.column_stack([100.0 * np.cos(theta), 100.0 * np.sin(theta)])
    xy = xy_true + rng.normal(0.0, noise_std, xy_true.shape)
    zeros = np.zeros(n)
    return Track(track_id="synthetic", lat=zeros, lon=zeros, t=t, xy=xy)


def test_fit_uniform_reaches_generous_tol():
    track = _noisy_arc_track()
    sp = fit_uniform(track, tol=5.0)
    assert sp.converged
    assert sp.max_error <= 5.0 + 1e-6
    assert sp.knot_mode == "uniform"


def test_fit_adaptive_reaches_generous_tol():
    track = _noisy_arc_track()
    sp = fit_adaptive(track, tol=5.0)
    assert sp.converged
    assert sp.max_error <= 5.0 + 1e-6
    assert sp.knot_mode == "adaptive"


def test_bisection_is_minimal_ish():
    """A tighter tol must not yield FEWER control points than a looser one
    (bisection searches for a minimal m, but the problem is monotonically
    harder)."""
    track = _noisy_arc_track()
    loose = fit_uniform(track, tol=10.0)
    tight = fit_uniform(track, tol=2.0)
    assert loose.converged and tight.converged
    assert tight.n_control_points >= loose.n_control_points


def test_fit_oracle_matches_true_curve():
    t_dense = np.linspace(0.0, 100.0, 1500)
    theta = t_dense / 100.0 * 3 * np.pi
    xy_true = np.column_stack([100.0 * np.cos(theta), 100.0 * np.sin(theta)])
    sp = fit_oracle(t_dense, xy_true, tol=0.5)
    assert sp.converged
    assert sp.max_error <= 0.5 + 1e-6
    recon = reconstruct(sp, t_dense)
    assert np.max(np.hypot(*(recon - xy_true).T)) <= 0.5 + 1e-6


def test_straight_line_needs_no_internal_knots():
    t = np.linspace(0.0, 10.0, 20)
    xy = np.column_stack([5.0 * t, 2.0 * t])
    track = Track(track_id="line", lat=np.zeros(20), lon=np.zeros(20), t=t, xy=xy)
    sp = fit_uniform(track, tol=0.01)
    assert sp.converged
    assert sp.n_internal_knots == 0
    assert sp.max_error < 1e-6


def test_unreachable_tol_reports_not_converged():
    """At a very tight tol on noisy data, the fitter honestly fails to
    converge (it doesn't force it via the knot count, see the m_max cap
    in _bisect_fit) -- a negative result is acceptable here."""
    track = _noisy_arc_track(n=60, noise_std=3.0)
    sp = fit_uniform(track, tol=1e-6)
    assert not sp.converged
