"""Polyline simplification: classic Douglas-Peucker (shapely.simplify) and
a time-aware variant with SED (synchronized Euclidean distance)."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString

DEFAULT_TOL = 10.0


def simplify(track, tol: float = DEFAULT_TOL) -> np.ndarray:
    """Simplifies a track's polyline (xy, meters) with the Douglas-Peucker algorithm.

    preserve_topology=False gives the classic DP (GEOS), not the
    topology-preserving variant.
    """
    line = LineString(track.xy)
    simplified = line.simplify(tol, preserve_topology=False)
    return np.asarray(simplified.coords)


def simplify_with_indices(track, tol: float = DEFAULT_TOL) -> tuple[np.ndarray, np.ndarray]:
    """Like simplify(), but also returns the indices of the retained
    points in track.xy/track.t (DP never creates new points, only removes
    vertices, so this mapping is always possible)."""
    xy = simplify(track, tol)
    orig = track.xy
    idx = np.empty(len(xy), dtype=int)
    cursor = -1
    for i, p in enumerate(xy):
        matches = np.flatnonzero((orig[:, 0] == p[0]) & (orig[:, 1] == p[1]))
        matches = matches[matches > cursor]
        cursor = int(matches[0])
        idx[i] = cursor
    return xy, idx


def _seds(t: np.ndarray, xy: np.ndarray, i0: int, i1: int) -> tuple[np.ndarray, np.ndarray]:
    """SED of each point in (i0, i1) to the segment [i0, i1] -- distance
    not to the nearest point of the segment (as in DP), but to the
    position interpolated by TIME FRACTION (Meratnia & de By, "TD-TR"):
    assumes motion between i0 and i1 was uniform in time along the line."""
    idx = np.arange(i0 + 1, i1)
    t0, t1 = t[i0], t[i1]
    frac = (t[idx] - t0) / (t1 - t0) if t1 > t0 else np.zeros(len(idx))
    interp = xy[i0] + frac[:, None] * (xy[i1] - xy[i0])
    d = np.hypot(*(xy[idx] - interp).T)
    return idx, d


def simplify_sed_with_indices(track, tol: float = DEFAULT_TOL) -> tuple[np.ndarray, np.ndarray]:
    """Time-aware simplification: a top-down algorithm in the spirit of
    Douglas-Peucker, but the split criterion is SED (see _seds) rather
    than perpendicular distance to the segment. Stricter than spatial DP
    at the same tol (it also accounts for uneven spacing of points in
    time), so it usually needs more points for the same tol."""
    t, xy = track.t, track.xy
    n = len(t)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 - i0 < 2:
            continue
        idx, d = _seds(t, xy, i0, i1)
        if len(d) == 0:
            continue
        j = int(np.argmax(d))
        if d[j] > tol:
            best_i = int(idx[j])
            keep[best_i] = True
            stack.append((i0, best_i))
            stack.append((best_i, i1))
    kept_idx = np.nonzero(keep)[0]
    return xy[kept_idx], kept_idx
