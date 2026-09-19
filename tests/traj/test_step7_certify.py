"""Tests for benchmarks/step7_certify.py's fit_validity() (ADR-0010/ADR-0014): the
full-corpus S2 validation script lives in benchmarks/ per spec section 5's file
layout, but fit_validity() has no home in src/traj/, so it's imported directly here
(mirroring step7_certify.py's own reverse sys.path insertion of tests/traj/)."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks"))

from step7_certify import fit_validity  # noqa: E402
from traj.io import Track  # noqa: E402
from traj.spline_lsq import fit_adaptive, fit_uniform  # noqa: E402


def _smooth_noiseless_track(n: int = 120) -> Track:
    t = np.linspace(0.0, 100.0, n)
    theta = t / 100.0 * 3 * np.pi
    xy = np.column_stack([100.0 * np.cos(theta), 100.0 * np.sin(theta)])
    zeros = np.zeros(n)
    return Track(track_id="smooth", lat=zeros, lon=zeros, t=t, xy=xy)


def test_fit_validity_accepts_good_fits_both_fitters():
    """ADR-0014: with the dense_mode="raw" fix, a genuinely good spline_lsq
    fit of a noiseless smooth curve must be reported valid -- confirms the
    fixed check doesn't just fail to crash, but correctly accepts a fit
    it should accept (closing the loop with
    test_spline_dense.test_dense_max_error_rejects_mismatched_domain, which
    confirms it correctly REJECTS a mismatched call)."""
    track = _smooth_noiseless_track()
    for fitter in (fit_adaptive, fit_uniform):
        fit = fitter(track, tol=10.0)
        valid, reason = fit_validity(track, fit, dense_mode="raw")
        assert valid, f"{fitter.__name__}: unexpectedly invalid, reason={reason}"
