"""Упрощение ломаной алгоритмом Дугласа-Пекера (shapely.simplify)."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString

DEFAULT_TOL = 10.0


def simplify(track, tol: float = DEFAULT_TOL) -> np.ndarray:
    """Упрощает ломаную трека (xy, метры) алгоритмом Дугласа-Пекера.

    preserve_topology=False даёт классический DP (GEOS), а не
    топологически-сохраняющий вариант.
    """
    line = LineString(track.xy)
    simplified = line.simplify(tol, preserve_topology=False)
    return np.asarray(simplified.coords)


def simplify_with_indices(track, tol: float = DEFAULT_TOL) -> tuple[np.ndarray, np.ndarray]:
    """Как simplify(), но также возвращает индексы сохранённых точек в
    track.xy/track.t (DP не создаёт новых точек, только удаляет вершины,
    поэтому такое сопоставление всегда возможно)."""
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
