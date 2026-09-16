"""Discrete Frechet distance (Eiter-Mannila) with numba and early exit."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _dist(a, b) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


@njit(cache=True, fastmath=True)
def _discrete_frechet_dp(P: np.ndarray, Q: np.ndarray, threshold: float):
    """Classic Eiter-Mannila DP.

    Early exit: ca[i][j] is the minimum over monotone paths of the "cost"
    (max of pairwise distances) of reaching cell (i, j). Any monotone path
    to (n-1, m-1) passes through exactly one cell in each row i, and its
    cost is at least min_j ca[i][j] (the cost of reaching that cell). So if
    min_j ca[i][j] > threshold, the final distance is also > threshold --
    computation can be aborted without finishing the remaining rows.
    """
    n = P.shape[0]
    m = Q.shape[0]
    ca = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        row_min = np.inf
        for j in range(m):
            d = _dist(P[i], Q[j])
            if i == 0 and j == 0:
                v = d
            elif i == 0:
                v = max(ca[0, j - 1], d)
            elif j == 0:
                v = max(ca[i - 1, 0], d)
            else:
                prev = ca[i - 1, j]
                if ca[i - 1, j - 1] < prev:
                    prev = ca[i - 1, j - 1]
                if ca[i, j - 1] < prev:
                    prev = ca[i, j - 1]
                v = max(prev, d)
            ca[i, j] = v
            if v < row_min:
                row_min = v
        if row_min > threshold:
            return row_min, False
    return ca[n - 1, m - 1], True


def lower_bound(P: np.ndarray, Q: np.ndarray) -> float:
    """max(|start_a-start_b|, |end_a-end_b|) -- a lower bound on Frechet distance."""
    d_start = float(np.hypot(*(P[0] - Q[0])))
    d_end = float(np.hypot(*(P[-1] - Q[-1])))
    return max(d_start, d_end)


def distance(P: np.ndarray, Q: np.ndarray) -> float:
    """Exact discrete Frechet distance between polylines P and Q."""
    P = np.ascontiguousarray(P, dtype=np.float64)
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    value, _ = _discrete_frechet_dp(P, Q, np.inf)
    return float(value)


def distance_within(P: np.ndarray, Q: np.ndarray, threshold: float) -> tuple[float, bool]:
    """Frechet distance with early exit at `threshold`.

    Returns (value, exact). If lower_bound already exceeds threshold, the
    DP doesn't run at all. If the DP aborts early, the returned value is a
    valid lower bound (>= threshold), exact=False. When exact=True the
    value is exact (it could still be <= or > threshold).
    """
    lb = lower_bound(P, Q)
    if lb > threshold:
        return lb, False
    P = np.ascontiguousarray(P, dtype=np.float64)
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    value, exact = _discrete_frechet_dp(P, Q, threshold)
    return float(value), exact
