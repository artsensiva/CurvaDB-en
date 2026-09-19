"""B-spline Bezier machinery: knot insertion (Boehm), Bezier segment extraction,
de Casteljau subdivision, derivative control points. General infrastructure used by
certify.py's certified linearization (spec section 2.4) and, later, the spline
projection-matching certificate (spec section 2.3, M2).

Operates on scipy.interpolate.BSpline objects (bs.t: knots, bs.c: control points
shape (n_coef, dim), bs.k: degree) -- the same representation spline.py/spline_lsq.py
already produce/reconstruct.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


def insert_knot(t: np.ndarray, c: np.ndarray, k: int, u_bar: float) -> tuple[np.ndarray, np.ndarray]:
    """Boehm's algorithm: insert u_bar once into knot vector t (degree k, control
    points c, shape (n+1, dim)). Returns (new_t, new_c) with one more knot and one
    more control point. Works whether or not u_bar coincides with an existing knot
    (raises its multiplicity if so).

    Q_i = P_i                                  for i <= k_span - k
    Q_i = a_i*P_i + (1-a_i)*P_{i-1}             for k_span-k+1 <= i <= k_span
          a_i = (u_bar - t[i]) / (t[i+k] - t[i])
    Q_i = P_{i-1}                               for i >= k_span + 1

    where k_span is the index with t[k_span] <= u_bar < t[k_span+1] (the last
    occurrence of u_bar itself, if u_bar is already a knot).
    """
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    n = c.shape[0] - 1

    k_span = n
    for i in range(k, n + 1):
        if t[i] <= u_bar < t[i + 1]:
            k_span = i
            break

    new_c = np.empty((c.shape[0] + 1,) + c.shape[1:], dtype=float)
    for i in range(0, k_span - k + 1):
        new_c[i] = c[i]
    for i in range(k_span - k + 1, k_span + 1):
        denom = t[i + k] - t[i]
        alpha = (u_bar - t[i]) / denom if denom > 0.0 else 0.0
        new_c[i] = alpha * c[i] + (1.0 - alpha) * c[i - 1]
    for i in range(k_span + 1, c.shape[0] + 1):
        new_c[i] = c[i - 1]

    new_t = np.concatenate([t[: k_span + 1], [u_bar], t[k_span + 1 :]])
    return new_t, new_c


def bezier_segments(bs: BSpline) -> list[np.ndarray]:
    """Raises every distinct interior knot to multiplicity k (via repeated
    insert_knot), then slices the resulting control points into Bezier segments.
    With m distinct interior knots there are m+1 segments, each k+1 control points;
    consecutive segments share one boundary control point."""
    t = np.asarray(bs.t, dtype=float)
    c = np.asarray(bs.c, dtype=float)
    k = bs.k
    n_knots = len(t)
    interior = t[k + 1 : n_knots - k - 1]
    if len(interior) == 0:
        return [c.copy()]

    values, counts = np.unique(interior, return_counts=True)
    for u, mult in zip(values, counts):
        for _ in range(k - int(mult)):
            t, c = insert_knot(t, c, k, float(u))

    m = len(values)
    return [c[j * k : j * k + k + 1].copy() for j in range(m + 1)]


def _raise_multiplicity(t: np.ndarray, c: np.ndarray, k: int, u_bar: float, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Inserts u_bar (exact float equality tracking -- insert_knot's returned knot
    vector always contains u_bar bit-for-bit, so repeated calls accumulate multiplicity
    correctly) until it has multiplicity target, whatever its starting multiplicity."""
    mult = int(np.sum(t == u_bar))
    for _ in range(target - mult):
        t, c = insert_knot(t, c, k, u_bar)
    return t, c


def bezier_segments_in_range(bs: BSpline, u_lo: float, u_hi: float) -> list[np.ndarray]:
    """Like bezier_segments, but restricted to [u_lo, u_hi] (spec section 2.3, M2's
    per-original-vertex pieces) -- u_lo/u_hi need not be existing knots. Raises both to
    full clamped multiplicity k+1 (the standard Bezier-extraction technique: treating
    them as fresh clamped sub-domain boundaries doesn't change the curve's values, only
    the knot vector's redundancy) and every interior knot strictly between them to
    multiplicity k, then slices the control points between u_lo's and u_hi's first
    occurrences into m+1 segments of k+1 points each. Verified against dense sampling
    (max error ~1e-15) for arbitrary, knot-aligned, and full-range [u_lo, u_hi].

    Requires u_hi > u_lo (a zero-width piece has no Bezier segment to extract -- callers
    with u_k == u_{k+1}, spec section 2.3's degenerate case, must special-case it
    themselves, matching a point-vs-segment formula rather than a curve-vs-segment one).
    """
    if u_hi <= u_lo:
        raise ValueError(f"bezier_segments_in_range requires u_hi > u_lo, got {u_lo=} {u_hi=}")

    t = np.asarray(bs.t, dtype=float)
    c = np.asarray(bs.c, dtype=float)
    k = bs.k

    t, c = _raise_multiplicity(t, c, k, u_lo, k + 1)
    t, c = _raise_multiplicity(t, c, k, u_hi, k + 1)

    n_knots = len(t)
    interior_all = t[k + 1 : n_knots - k - 1]
    between = np.unique(interior_all[(interior_all > u_lo) & (interior_all < u_hi)])
    for u in between:
        t, c = _raise_multiplicity(t, c, k, float(u), k)

    lo_first = int(np.searchsorted(t, u_lo, side="left"))
    hi_first = int(np.searchsorted(t, u_hi, side="left"))
    seg_c = c[lo_first : hi_first + 1]
    m_plus_1 = (len(seg_c) - 1) // k
    return [seg_c[j * k : j * k + k + 1].copy() for j in range(m_plus_1)]


def de_casteljau_split(P: np.ndarray, t: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Splits a single Bezier segment (control points P, shape (p+1, dim)) at
    parameter t into (left, right) child segments, each shape (p+1, dim)."""
    P = np.asarray(P, dtype=float)
    p = P.shape[0] - 1
    tri = [P.copy()]
    cur = P
    for _ in range(p):
        cur = (1.0 - t) * cur[:-1] + t * cur[1:]
        tri.append(cur)
    left = np.array([tri[level][0] for level in range(p + 1)])
    right = np.array([tri[level][-1] for level in range(p + 1)])[::-1]
    return left, right


def derivative_control_points(P: np.ndarray) -> np.ndarray:
    """Degree p-1 derivative Bezier's control points: p*(P[i+1]-P[i])."""
    P = np.asarray(P, dtype=float)
    p = P.shape[0] - 1
    return p * (P[1:] - P[:-1])


def evaluate_bezier(P: np.ndarray, t: float) -> np.ndarray:
    """De Casteljau evaluation of a single Bezier segment at parameter t in [0,1]."""
    cur = np.asarray(P, dtype=float)
    while cur.shape[0] > 1:
        cur = (1.0 - t) * cur[:-1] + t * cur[1:]
    return cur[0]
