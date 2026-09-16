"""Spline fitting tests: approximation error <= tol."""

import numpy as np

from traj.io import Track
from traj.spline import fit


def _synthetic_track(n: int = 80, seed: int = 0) -> Track:
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.uniform(0.5, 2.0, size=n))
    t -= t[0]
    theta = np.linspace(0.0, 4 * np.pi, n)
    x = 100.0 * np.cos(theta) + rng.normal(0.0, 1.0, n)
    y = 100.0 * np.sin(theta) + rng.normal(0.0, 1.0, n)
    xy = np.column_stack([x, y])
    zeros = np.zeros(n)
    return Track(track_id="synthetic", lat=zeros, lon=zeros, t=t, xy=xy)


def test_spline_max_error_within_tol():
    track = _synthetic_track()
    sp = fit(track, tol=10.0)
    assert sp.converged
    assert sp.max_error <= 10.0 + 1e-6


def test_spline_max_error_within_tol_various_shapes():
    for seed in range(5):
        track = _synthetic_track(n=60 + seed * 10, seed=seed)
        sp = fit(track, tol=5.0)
        assert sp.max_error <= 5.0 + 1e-6
