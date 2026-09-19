"""Step8 spec section 2.4: knot removal for spline segments -- start from a
dense/near-interpolating spline, greedily remove the knot whose removal
least disturbs the fit, refit by least squares, stop when no further
removal keeps the CERTIFIED error within tol (Lyche-Moerken-style removal
order, certified stopping).

Cost budget (ADR-0023): a naive design calling `certify_spline` once per
CANDIDATE at every removal step is O(m) certify calls per segment, and
`certify_spline` is far too expensive per call for M2's scale (measured
directly, `benchmarks/step8_m1_budget.py`) to spend that many calls on every
one of the many segments M2's DP will evaluate. This module instead:

1. Ranks candidate knots ONCE (`_rank_removal_order`) by a cheap LSQ-residual
   local cost (no certification involved) -- a documented simplification of
   textbook Lyche-Moerken, which re-ranks adaptively after every removal;
   this module ranks once and removes in that fixed order, trading some
   removal-order optimality for a large constant-factor speedup.
2. Bisects (`remove_knots`) over how many knots (in that fixed order) can be
   removed while still certifying `<= target_tol` -- O(log m) real
   `certify_spline` calls, not O(m). Bisection assumes the certified error is
   roughly non-decreasing as more knots are removed in this order; if that
   assumption is violated for a specific track, the search may settle on a
   suboptimal removal count, but every ACCEPTED candidate is independently
   certified before being returned -- never a false accept (the same
   heuristic-search-plus-independent-verification split as ADR-0008).
3. Adds a small, bounded local-refinement pass after bisection (try removing
   a few more knots one at a time, stop at the first failure) to recover a
   little of the adaptive-greedy quality near the boundary cheaply.

ADR-0023 also fixes the DEFAULT `lam_fallback` (0.1, not ADR-0022's 0.001 --
too slow at M2's per-segment call volume) and the starting dense fit's own
internal tol (`0.5 * target_tol`, via `traj.spline.fit`, ADR-0022's chosen
fitter) -- both measured, not guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline

from . import spline as spline_module
from .certify import certify_spline
from .io import Track

DEGREE = 3
DEFAULT_LAM_FALLBACK = 0.1  # ADR-0023
DEFAULT_INTERNAL_TOL_RATIO = 0.5  # ADR-0023
DEFAULT_MAX_LOCAL_REFINE = 10


@dataclass
class KnotRemovalFit:
    bs: BSpline
    eps_A: float  # certify_spline's certified upper bound, against `A`
    n_internal_knots: int
    n_control_points: int
    n: int  # n_knots (full vector) + 2*n_control_points, matching M0's convention
    converged: bool  # False if even the starting dense fit fails to certify <= target_tol
    n_removed: int
    u: np.ndarray  # the parametrization actually used (spline.fit()'s "time" convention)


def _full_knot_vector(u_min: float, u_max: float, internal: np.ndarray, k: int) -> np.ndarray:
    return np.concatenate([np.full(k + 1, u_min), internal, np.full(k + 1, u_max)])


def _lsq_refit(u: np.ndarray, xy: np.ndarray, internal_knots: np.ndarray, k: int) -> BSpline:
    """`make_lsq_spline` can succeed (no exception) yet return NaN/Inf
    coefficients for a near-degenerate knot set (e.g. removal leaving too
    little data support between neighboring knots) -- raises ValueError in
    that case too, so every caller's existing except-ValueError handling
    covers it without a separate check at each call site."""
    full_knots = _full_knot_vector(float(u[0]), float(u[-1]), internal_knots, k)
    bs = make_lsq_spline(u, xy, full_knots, k=k)
    if not np.all(np.isfinite(bs.c)):
        raise ValueError("LSQ refit produced non-finite control points (degenerate knot removal)")
    return bs


def _rank_removal_order(u: np.ndarray, xy: np.ndarray, internal_knots: np.ndarray, k: int) -> np.ndarray:
    """One-shot local ranking: for each interior knot, the LSQ residual
    (direct max Euclidean deviation at the points, no certify_spline) of
    removing THAT ONE knot alone from the full set, cheapest first."""
    costs = np.empty(len(internal_knots))
    for i in range(len(internal_knots)):
        trial = np.delete(internal_knots, i)
        try:
            trial_bs = _lsq_refit(u, xy, trial, k)
        except (ValueError, np.linalg.LinAlgError):
            costs[i] = np.inf
            continue
        pred = trial_bs(u)
        costs[i] = float(np.max(np.hypot(*(pred - xy).T)))
    return np.argsort(costs)


def remove_knots(
    t: np.ndarray,
    xy: np.ndarray,
    A: np.ndarray,
    target_tol: float,
    lam_fallback: float = DEFAULT_LAM_FALLBACK,
    internal_tol_ratio: float = DEFAULT_INTERNAL_TOL_RATIO,
    k: int = DEGREE,
    max_local_refine: int = DEFAULT_MAX_LOCAL_REFINE,
) -> KnotRemovalFit:
    """Greedy knot removal with certified stopping (spec section 2.4).

    `t`, `xy`: the points the spline is fit TO -- the recorded track's own
    samples for the plain LSQ-free-knot method, or a dense true-curve sample
    for the free-knot oracle (spec 2.6).
    `A`: the polyline `certify_spline` certifies error against (same
    convention as `certify.certify_spline(A, bs)`). M1's own callers always
    pass `A=xy` (LSQ-free-knot certifies against the same recorded points it
    fits; the oracle certifies against the same dense true-curve sample it
    fits) -- kept as a separate parameter for M2, where a DP segment's `A`
    may differ from what a candidate was constructed from.
    """
    if k != spline_module.DEGREE:
        raise ValueError(f"remove_knots only supports k={spline_module.DEGREE} (traj.spline.fit()'s own fixed degree), got {k}")
    track_like = Track(track_id="knot_removal", lat=np.zeros(len(t)), lon=np.zeros(len(t)), t=np.asarray(t, dtype=float), xy=np.asarray(xy, dtype=float))
    dense = spline_module.fit(track_like, tol=internal_tol_ratio * target_tol, parametrization="time")

    u = spline_module._param_u(track_like.t, track_like.xy, "time")
    knots, c_list, k_fit = dense.tck
    interior = np.asarray(knots[k_fit + 1 : len(knots) - (k_fit + 1)], dtype=float)
    bs = BSpline(knots, np.column_stack(c_list), k_fit)
    eps_A, _method = certify_spline(A, bs, lam_fallback=lam_fallback)

    def _n(n_internal: int, n_control: int) -> int:
        return (n_internal + 2 * (k_fit + 1)) + 2 * n_control

    if eps_A > target_tol:
        return KnotRemovalFit(bs, eps_A, len(interior), len(c_list[0]), _n(len(interior), len(c_list[0])), False, 0, u)
    if len(interior) == 0:
        return KnotRemovalFit(bs, eps_A, 0, len(c_list[0]), _n(0, len(c_list[0])), True, 0, u)

    order = _rank_removal_order(u, xy, interior, k_fit)
    ranked_interior = interior[order]

    cache: dict[int, tuple[BSpline, float] | None] = {}

    def try_remove(m_count: int):
        if m_count in cache:
            return cache[m_count]
        remaining = np.sort(ranked_interior[m_count:])
        try:
            trial_bs = _lsq_refit(u, xy, remaining, k_fit)
        except (ValueError, np.linalg.LinAlgError):
            cache[m_count] = None
            return None
        trial_eps, _method = certify_spline(A, trial_bs, lam_fallback=lam_fallback)
        result = (trial_bs, trial_eps)
        cache[m_count] = result
        return result

    best_bs, best_eps, best_removed = bs, eps_A, 0
    lo, hi = 0, len(ranked_interior)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        result = try_remove(mid)
        if result is not None and result[1] <= target_tol:
            lo, best_bs, best_eps, best_removed = mid, result[0], result[1], mid
        else:
            hi = mid

    for _ in range(max_local_refine):
        if best_removed >= len(ranked_interior):
            break
        result = try_remove(best_removed + 1)
        if result is not None and result[1] <= target_tol:
            best_removed += 1
            best_bs, best_eps = result
        else:
            break

    n_control_points = best_bs.c.shape[0]
    n_internal_final = len(interior) - best_removed
    return KnotRemovalFit(best_bs, best_eps, n_internal_final, n_control_points, _n(n_internal_final, n_control_points), True, best_removed, u)
