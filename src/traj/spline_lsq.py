"""Fitting trajectories with cubic B-splines by least squares directly on
x(t), y(t) (`scipy.interpolate.make_lsq_spline`) -- WITHOUT tethering to
the noisy points' polyline (unlike `spline.fit()`; see its docstring and
docs/prompts/step3.md: the "stay near the polyline" contract prevents the
old fitter from smoothing noise). The spline isn't required to pass
through the noisy points -- so error here is the DIRECT Euclidean residual
at the points themselves (t_i, xy_i), not point-to-segment distance to the
polyline (`spline.dense_max_error`).

Two ways of placing internal knots:
- "uniform" -- evenly spaced by index along t;
- "adaptive" -- two-pass: a uniform trial fit with m knots, residuals at
  the points, redistribute the same m knots by cumulative sum of
  |residuals| (more knots where the trial fit describes the data worst),
  refit. The true curve's curvature is NOT used -- only the residuals of
  the noisy trial fit.

Internal knots are always chosen as a SUBSET of the actual t samples
(rather than arbitrary real-valued positions) -- this automatically
satisfies the Schoenberg-Whitney condition (at least one data sample
between any two neighboring knots) required by `make_lsq_spline`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline

DEGREE = 3
MAX_ITER = 40


@dataclass
class LsqSplineFit:
    tck: tuple  # (knots, [cx, cy], k) -- shape-compatible with SplineFit.tck
    t_min: float
    t_max: float
    max_error: float  # max Euclidean residual at the points (t_i, xy_i), NOT to the polyline
    converged: bool
    n_control_points: int
    knot_mode: str  # "uniform" | "adaptive" | "oracle"
    n_internal_knots: int


def _pick_indices(n: int, m: int, weights: np.ndarray | None = None) -> np.ndarray:
    """m strictly distinct indices in (0, n-1) -- internal knots as a
    subset of data samples (guarantees the Schoenberg-Whitney condition).
    weights (length n, optional) -- point weights biasing indices toward
    higher-weight regions (adaptive placement); None -- uniform."""
    if m <= 0:
        return np.empty(0, dtype=int)
    if weights is None:
        pos = np.linspace(0.0, n - 1, m + 2)[1:-1]
    else:
        cum = np.concatenate([[0.0], np.cumsum(weights)])
        targets = np.linspace(0.0, cum[-1], m + 2)[1:-1]
        pos = np.interp(targets, cum, np.arange(n + 1, dtype=float))
    idx = np.clip(np.round(pos).astype(int), 1, n - 2)
    idx = np.unique(idx)
    if len(idx) < m:
        free = np.setdiff1d(np.arange(1, n - 1), idx)
        need = m - len(idx)
        if len(free) >= need > 0:
            extra = free[np.linspace(0, len(free) - 1, need).round().astype(int)]
            idx = np.unique(np.concatenate([idx, extra]))
    return idx


def _full_knot_vector(t_min: float, t_max: float, internal_knots: np.ndarray, k: int) -> np.ndarray:
    return np.concatenate([np.full(k + 1, t_min), internal_knots, np.full(k + 1, t_max)])


def _lsq_fit(t: np.ndarray, xy: np.ndarray, k: int, m: int, weights: np.ndarray | None = None) -> BSpline:
    idx = _pick_indices(len(t), m, weights=weights)
    internal_knots = t[idx]
    full_knots = _full_knot_vector(float(t[0]), float(t[-1]), internal_knots, k)
    return make_lsq_spline(t, xy, full_knots, k=k)


def _residual_max(t: np.ndarray, xy: np.ndarray, bspline: BSpline) -> float:
    pred = bspline(t)
    return float(np.max(np.hypot(*(pred - xy).T)))


def _build_uniform(t: np.ndarray, xy: np.ndarray, k: int, m: int) -> tuple[BSpline, float]:
    bs = _lsq_fit(t, xy, k, m, weights=None)
    return bs, _residual_max(t, xy, bs)


def _build_adaptive(t: np.ndarray, xy: np.ndarray, k: int, m: int) -> tuple[BSpline, float]:
    if m <= 0:
        return _build_uniform(t, xy, k, m)
    bs0 = _lsq_fit(t, xy, k, m, weights=None)
    residual0 = np.hypot(*(bs0(t) - xy).T)
    bs1 = _lsq_fit(t, xy, k, m, weights=residual0)
    return bs1, _residual_max(t, xy, bs1)


def _make_fit(bs: BSpline, t: np.ndarray, err: float, converged: bool, knot_mode: str, m: int) -> LsqSplineFit:
    tck = (bs.t, [bs.c[:, 0], bs.c[:, 1]], bs.k)
    return LsqSplineFit(
        tck=tck,
        t_min=float(t[0]),
        t_max=float(t[-1]),
        max_error=float(err),
        converged=converged,
        n_control_points=len(tck[1][0]),
        knot_mode=knot_mode,
        n_internal_knots=m,
    )


def _bisect_fit(t: np.ndarray, xy: np.ndarray, tol: float, k: int, max_iter: int, knot_mode: str) -> LsqSplineFit:
    """Grow the number of internal knots m (exponentially) until the
    honest error (direct residual at the points) is <= tol, then bisect
    down to the minimal m -- analogous to growing `s`/bisection in
    spline.fit().

    m_max -- the theoretical ceiling (n-k-2, the point at which the least
    squares system stops being overdetermined). Near that boundary
    (near-interpolation on noisy data), the error occasionally jumps by
    1-2 orders of magnitude -- so growth tracks the BEST (not the last)
    result; if tol is unreachable, that best result is returned rather
    than a potentially degraded attempt right at m_max."""
    n = len(t)
    m_max = max(n - k - 2, 0)
    build = _build_adaptive if knot_mode == "adaptive" else _build_uniform

    best_bs, best_err = build(t, xy, k, 0)
    best_m = 0
    if best_err <= tol or m_max == 0:
        return _make_fit(best_bs, t, best_err, best_err <= tol, knot_mode, 0)

    m_lo, m_hi = 0, 1
    success = None  # (bs, err, m) -- first m to reach tol
    grown = 0
    while grown < max_iter:
        try:
            bs_hi, err_hi = build(t, xy, k, m_hi)
        except (ValueError, np.linalg.LinAlgError):
            bs_hi, err_hi = None, float("inf")
        if bs_hi is not None and err_hi < best_err:
            best_bs, best_err, best_m = bs_hi, err_hi, m_hi
        if err_hi <= tol:
            success = (bs_hi, err_hi, m_hi)
            break
        if m_hi >= m_max:
            break
        m_lo, m_hi = m_hi, min(m_hi * 2, m_max)
        grown += 1

    if success is None:
        # tol is unreachable for m <= m_max -- return the best result found
        return _make_fit(best_bs, t, best_err, False, knot_mode, best_m)

    best_bs, best_err, best_m = success
    lo_b, hi_b = m_lo, best_m
    for _ in range(max_iter):
        if hi_b - lo_b <= 1:
            break
        m_mid = (lo_b + hi_b) // 2
        try:
            bs_mid, err_mid = build(t, xy, k, m_mid)
        except (ValueError, np.linalg.LinAlgError):
            err_mid = float("inf")
        if err_mid <= tol:
            hi_b, best_bs, best_err, best_m = m_mid, bs_mid, err_mid, m_mid
        else:
            lo_b = m_mid
    return _make_fit(best_bs, t, best_err, True, knot_mode, best_m)


def fit_uniform(track, tol: float, k: int = DEGREE, max_iter: int = MAX_ITER) -> LsqSplineFit:
    """Uniform (by sample index) internal knots, bisecting their count
    down to the honest residual <= tol at the track's own (noisy) points."""
    t = np.asarray(track.t, dtype=float)
    xy = np.asarray(track.xy, dtype=float)
    return _bisect_fit(t, xy, tol, k, max_iter, knot_mode="uniform")


def fit_adaptive(track, tol: float, k: int = DEGREE, max_iter: int = MAX_ITER) -> LsqSplineFit:
    """Like fit_uniform, but at each candidate m -- a two-pass fit (trial
    uniform fit -> residuals -> redistribute knots by cumulative residual
    -> refit)."""
    t = np.asarray(track.t, dtype=float)
    xy = np.asarray(track.xy, dtype=float)
    return _bisect_fit(t, xy, tol, k, max_iter, knot_mode="adaptive")


def fit_oracle(t_dense: np.ndarray, true_xy_dense: np.ndarray, tol: float, k: int = DEGREE, max_iter: int = MAX_ITER) -> LsqSplineFit:
    """Fits DIRECTLY to the dense ground-truth (noise-free) curve, uniform
    knots, bisecting m by the error on that same dense grid. Knows nothing
    about noise or observation sparsity -- a reference lower bound on
    geometry representation at a given tol (used only for the oracle table
    in the benchmark, not part of the criteria)."""
    t = np.asarray(t_dense, dtype=float)
    xy = np.asarray(true_xy_dense, dtype=float)
    return _bisect_fit(t, xy, tol, k, max_iter, knot_mode="oracle")


def reconstruct(fit: LsqSplineFit, t) -> np.ndarray:
    """Reconstructs the fit spline's xy at times t."""
    knots, c_list, k = fit.tck
    c = np.column_stack(c_list)
    bs = BSpline(knots, c, k)
    return bs(np.asarray(t, dtype=float))
