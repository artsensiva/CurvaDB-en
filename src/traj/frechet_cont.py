"""Continuous Frechet distance between polylines (Alt-Godau free-space diagram)."""

from __future__ import annotations

import numpy as np
from numba import njit

_MACHINE_EPS = float(np.finfo(np.float64).eps)


@njit(cache=True)
def _free_interval(ax, ay, bx, by, px, py, eps2):
    """Interval of t in [0,1] with |a + t*(b-a) - p|^2 <= eps2. (lo, hi), lo > hi if empty.

    Deliberately strict (no discriminant slack): a point exactly on the infinite
    line through a,b gives a mathematically repeated root (discriminant == 0
    exactly), and float64 cancellation in B*B - 4*A*C can push that residual
    slightly negative, reporting "empty" a hair early. An earlier version of this
    function clamped small negative residuals to feasible to paper over that --
    which is unsafe: it made decide(eps) return True for eps measurably *below*
    the true distance at realistic (~1e5 m) coordinate scale, i.e. distance()
    could underestimate d_F, violating the certificate requirement (spec S1:
    eps_A >= d_F). Rounding the other way (reporting empty a hair early) only
    ever makes decide()/distance() report a value >= the true distance -- the
    safe direction. Any additional slack needed for certificate tightness
    belongs at the certify.py layer (M1), per the spec's own floating-point
    rigor convention (docs/specs/00_overview.md section 3.4), not here.
    """
    dx = bx - ax
    dy = by - ay
    A = dx * dx + dy * dy
    ex = ax - px
    ey = ay - py
    if A < 1e-20:
        return (0.0, 1.0) if (ex * ex + ey * ey) <= eps2 else (1.0, 0.0)
    B = 2.0 * (dx * ex + dy * ey)
    C = ex * ex + ey * ey - eps2
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return (1.0, 0.0)
    sq = disc**0.5
    t_lo = (-B - sq) / (2.0 * A)
    t_hi = (-B + sq) / (2.0 * A)
    lo = t_lo if t_lo > 0.0 else 0.0
    hi = t_hi if t_hi < 1.0 else 1.0
    return (lo, hi)


@njit(cache=True)
def _decide_core(P, Q, eps2):
    """Alt-Godau reachability DP. P has n segments, Q has m segments."""
    n = P.shape[0] - 1
    m = Q.shape[0] - 1

    dx0 = P[0, 0] - Q[0, 0]
    dy0 = P[0, 1] - Q[0, 1]
    feasible00 = (dx0 * dx0 + dy0 * dy0) <= eps2

    dxn = P[n, 0] - Q[m, 0]
    dyn = P[n, 1] - Q[m, 1]
    feasible_nm = (dxn * dxn + dyn * dyn) <= eps2
    if not feasible_nm or not feasible00:
        return False

    # LeftFree[i][j]: vertical edge x=i vs Q segment j -- shape (n+1, m)
    # BotFree[i][j]: horizontal edge y=j vs P segment i -- shape (n, m+1)
    left_lo = np.empty((n + 1, m), dtype=np.float64)
    left_hi = np.empty((n + 1, m), dtype=np.float64)
    for i in range(n + 1):
        for j in range(m):
            lo, hi = _free_interval(Q[j, 0], Q[j, 1], Q[j + 1, 0], Q[j + 1, 1], P[i, 0], P[i, 1], eps2)
            left_lo[i, j] = lo
            left_hi[i, j] = hi

    bot_lo = np.empty((n, m + 1), dtype=np.float64)
    bot_hi = np.empty((n, m + 1), dtype=np.float64)
    for i in range(n):
        for j in range(m + 1):
            lo, hi = _free_interval(P[i, 0], P[i, 1], P[i + 1, 0], P[i + 1, 1], Q[j, 0], Q[j, 1], eps2)
            bot_lo[i, j] = lo
            bot_hi[i, j] = hi

    # Reachable sub-intervals: L[i][j] subset of LeftFree[i][j] (i=0..n, j=0..m-1);
    # B[i][j] subset of BotFree[i][j] (i=0..n-1, j=0..m). Only the reachable lo is
    # tracked -- reachable hi always equals the free interval's own hi.
    l_reach = np.full((n + 1, m), np.inf, dtype=np.float64)
    l_ok = np.zeros((n + 1, m), dtype=np.bool_)
    b_reach = np.full((n, m + 1), np.inf, dtype=np.float64)
    b_ok = np.zeros((n, m + 1), dtype=np.bool_)

    for i in range(n + 1):
        for j in range(m):
            lo, hi = left_lo[i, j], left_hi[i, j]
            if lo > hi:
                continue
            entire = False
            if i == 0 and j == 0 and feasible00:
                entire = True
            if not entire and i >= 1 and b_ok[i - 1, j]:
                entire = True
            if not entire and j >= 1 and l_ok[i, j - 1]:
                prev_hi = left_hi[i, j - 1]
                if prev_hi == 1.0 and lo == 0.0:
                    entire = True
            if entire:
                l_ok[i, j] = True
                l_reach[i, j] = lo
            elif i >= 1 and l_ok[i - 1, j]:
                new_lo = l_reach[i - 1, j]
                if new_lo < lo:
                    new_lo = lo
                if new_lo <= hi:
                    l_ok[i, j] = True
                    l_reach[i, j] = new_lo

        if i >= n:
            continue
        for j in range(m + 1):
            lo, hi = bot_lo[i, j], bot_hi[i, j]
            if lo > hi:
                continue
            entire = False
            if i == 0 and j == 0 and feasible00:
                entire = True
            if not entire and j >= 1 and l_ok[i, j - 1]:
                entire = True
            if not entire and i >= 1 and b_ok[i - 1, j]:
                prev_hi = bot_hi[i - 1, j]
                if prev_hi == 1.0 and lo == 0.0:
                    entire = True
            if entire:
                b_ok[i, j] = True
                b_reach[i, j] = lo
            elif j >= 1 and b_ok[i, j - 1]:
                new_lo = b_reach[i, j - 1]
                if new_lo < lo:
                    new_lo = lo
                if new_lo <= hi:
                    b_ok[i, j] = True
                    b_reach[i, j] = new_lo

    return bool(l_ok[n, m - 1] or b_ok[n - 1, m])


def decide(P: np.ndarray, Q: np.ndarray, eps: float) -> bool:
    """Decide whether the continuous Frechet distance between polylines P and Q is <= eps."""
    if eps < 0.0:
        raise ValueError("eps must be non-negative")
    P = np.ascontiguousarray(P, dtype=np.float64)
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    if P.shape[0] < 2 or Q.shape[0] < 2:
        raise ValueError("P and Q must each have at least 2 vertices (1 segment)")
    # Frechet distance is translation-invariant; recentering on a common origin keeps
    # every coordinate fed into the DP small (local), regardless of how far P, Q sit
    # from an arbitrary absolute origin (e.g. GeoLife's shared projection centroid).
    origin = P[0].copy()
    return _decide_core(P - origin, Q - origin, eps * eps)


def distance(P: np.ndarray, Q: np.ndarray, tol: float = 1e-6) -> float:
    """Continuous Frechet distance between polylines P and Q, via bisection on eps.

    Returns the certified upper end of the bisection bracket (hi), so the result is
    always >= the true distance, with slack at most `tol`. Default tol=1e-6 for
    general use; callers needing the spec's certificate convention (eta=1mm) should
    pass tol=1e-3 explicitly.
    """
    P = np.ascontiguousarray(P, dtype=np.float64)
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    if P.shape[0] < 2 or Q.shape[0] < 2:
        raise ValueError("P and Q must each have at least 2 vertices (1 segment)")

    if decide(P, Q, 0.0):
        return 0.0

    d_start = float(np.hypot(*(P[0] - Q[0])))
    d_end = float(np.hypot(*(P[-1] - Q[-1])))
    hi = max(d_start, d_end)
    if hi <= 0.0:
        hi = tol
    while not decide(P, Q, hi):
        hi *= 2.0

    lo = 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if decide(P, Q, mid):
            hi = mid
        else:
            lo = mid
    return hi


@njit(cache=True)
def _free_interval_conservative(ax, ay, bx, by, px, py, eps):
    """Like _free_interval, but tests a reduced eps2 -- eps2 minus a margin computed
    from THIS cell's own local geometry, not any global scale. Translation-invariant
    by construction: a, b, p are already local (small) once the caller has recentered
    P, Q, so |a-p| and |b-a| stay small (meters, not the ~1e5 m absolute coordinate
    offset a curve might sit at) regardless of where the curve is in the world.
    """
    dx = bx - ax
    dy = by - ay
    A = dx * dx + dy * dy
    ex = ax - px
    ey = ay - py
    off_len = (ex * ex + ey * ey) ** 0.5
    seg_len = A**0.5
    local_scale = off_len + seg_len + eps
    margin = 64.0 * _MACHINE_EPS * local_scale * local_scale
    eps2 = eps * eps - margin
    if A < 1e-20:
        return (0.0, 1.0) if (ex * ex + ey * ey) <= eps2 else (1.0, 0.0)
    B = 2.0 * (dx * ex + dy * ey)
    C = ex * ex + ey * ey - eps2
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return (1.0, 0.0)
    sq = disc**0.5
    t_lo = (-B - sq) / (2.0 * A)
    t_hi = (-B + sq) / (2.0 * A)
    lo = t_lo if t_lo > 0.0 else 0.0
    hi = t_hi if t_hi < 1.0 else 1.0
    return (lo, hi)


@njit(cache=True)
def _decide_core_conservative(P, Q, eps):
    """Mirrors _decide_core exactly, except: the two corner feasibility checks and
    every cell's free interval use a local (per-corner / per-cell) margin instead of
    a raw eps2 comparison -- see _free_interval_conservative's docstring."""
    n = P.shape[0] - 1
    m = Q.shape[0] - 1

    dx0 = P[0, 0] - Q[0, 0]
    dy0 = P[0, 1] - Q[0, 1]
    d0 = (dx0 * dx0 + dy0 * dy0) ** 0.5
    margin0 = 64.0 * _MACHINE_EPS * (d0 + eps) * (d0 + eps)
    feasible00 = d0 * d0 <= eps * eps - margin0

    dxn = P[n, 0] - Q[m, 0]
    dyn = P[n, 1] - Q[m, 1]
    dn = (dxn * dxn + dyn * dyn) ** 0.5
    marginn = 64.0 * _MACHINE_EPS * (dn + eps) * (dn + eps)
    feasible_nm = dn * dn <= eps * eps - marginn
    if not feasible_nm or not feasible00:
        return False

    left_lo = np.empty((n + 1, m), dtype=np.float64)
    left_hi = np.empty((n + 1, m), dtype=np.float64)
    for i in range(n + 1):
        for j in range(m):
            lo, hi = _free_interval_conservative(Q[j, 0], Q[j, 1], Q[j + 1, 0], Q[j + 1, 1], P[i, 0], P[i, 1], eps)
            left_lo[i, j] = lo
            left_hi[i, j] = hi

    bot_lo = np.empty((n, m + 1), dtype=np.float64)
    bot_hi = np.empty((n, m + 1), dtype=np.float64)
    for i in range(n):
        for j in range(m + 1):
            lo, hi = _free_interval_conservative(P[i, 0], P[i, 1], P[i + 1, 0], P[i + 1, 1], Q[j, 0], Q[j, 1], eps)
            bot_lo[i, j] = lo
            bot_hi[i, j] = hi

    l_reach = np.full((n + 1, m), np.inf, dtype=np.float64)
    l_ok = np.zeros((n + 1, m), dtype=np.bool_)
    b_reach = np.full((n, m + 1), np.inf, dtype=np.float64)
    b_ok = np.zeros((n, m + 1), dtype=np.bool_)

    for i in range(n + 1):
        for j in range(m):
            lo, hi = left_lo[i, j], left_hi[i, j]
            if lo > hi:
                continue
            entire = False
            if i == 0 and j == 0 and feasible00:
                entire = True
            if not entire and i >= 1 and b_ok[i - 1, j]:
                entire = True
            if not entire and j >= 1 and l_ok[i, j - 1]:
                prev_hi = left_hi[i, j - 1]
                if prev_hi == 1.0 and lo == 0.0:
                    entire = True
            if entire:
                l_ok[i, j] = True
                l_reach[i, j] = lo
            elif i >= 1 and l_ok[i - 1, j]:
                new_lo = l_reach[i - 1, j]
                if new_lo < lo:
                    new_lo = lo
                if new_lo <= hi:
                    l_ok[i, j] = True
                    l_reach[i, j] = new_lo

        if i >= n:
            continue
        for j in range(m + 1):
            lo, hi = bot_lo[i, j], bot_hi[i, j]
            if lo > hi:
                continue
            entire = False
            if i == 0 and j == 0 and feasible00:
                entire = True
            if not entire and j >= 1 and l_ok[i, j - 1]:
                entire = True
            if not entire and i >= 1 and b_ok[i - 1, j]:
                prev_hi = bot_hi[i - 1, j]
                if prev_hi == 1.0 and lo == 0.0:
                    entire = True
            if entire:
                b_ok[i, j] = True
                b_reach[i, j] = lo
            elif j >= 1 and b_ok[i, j - 1]:
                new_lo = b_reach[i, j - 1]
                if new_lo < lo:
                    new_lo = lo
                if new_lo <= hi:
                    b_ok[i, j] = True
                    b_reach[i, j] = new_lo

    return bool(l_ok[n, m - 1] or b_ok[n - 1, m])


def decide_conservative(P: np.ndarray, Q: np.ndarray, eps: float) -> bool:
    """Stricter than decide(): also requires clearance beyond the natural float64
    cancellation noise at the Delta~=0 boundary, via a per-cell local margin (see
    _free_interval_conservative's docstring). Used by distance_upper to verify its
    bisected bound has a genuine, quantified margin -- not just "happened to round
    the safe way," which decide()/distance() alone don't independently re-verify.
    """
    if eps < 0.0:
        raise ValueError("eps must be non-negative")
    P = np.ascontiguousarray(P, dtype=np.float64)
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    if P.shape[0] < 2 or Q.shape[0] < 2:
        raise ValueError("P and Q must each have at least 2 vertices (1 segment)")
    origin = P[0].copy()
    return _decide_core_conservative(P - origin, Q - origin, eps)


def distance_upper(P: np.ndarray, Q: np.ndarray, tol: float = 1e-6) -> float:
    """Certificate-grade upper bound on the continuous Frechet distance, for M1's
    certify.py. Recenters once, bisects via distance() (which recenters again
    trivially, since P[0] is now the origin), then verifies the result against
    decide_conservative -- growing further (doubling, then a short bisection
    refinement) if the plain bisected bound doesn't clear the stricter check.
    distance() alone is safe on average (see _free_interval's docstring) but not
    independently re-verified the way this function's result is; use this, not
    distance(), wherever the result feeds a certificate.
    """
    P = np.ascontiguousarray(P, dtype=np.float64)
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    if P.shape[0] < 2 or Q.shape[0] < 2:
        raise ValueError("P and Q must each have at least 2 vertices (1 segment)")
    origin = P[0].copy()
    P = P - origin
    Q = Q - origin

    hi = distance(P, Q, tol)
    if decide_conservative(P, Q, hi):
        return hi

    step = max(hi, tol)
    while not decide_conservative(P, Q, hi + step):
        step *= 2.0
    lo = hi
    hi = hi + step
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if decide_conservative(P, Q, mid):
            hi = mid
        else:
            lo = mid
    return hi
