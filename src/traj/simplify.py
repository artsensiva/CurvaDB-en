"""Упрощение ломаной: классический Дуглас-Пекер (shapely.simplify) и
time-aware вариант с SED (synchronized Euclidean distance)."""

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


def _seds(t: np.ndarray, xy: np.ndarray, i0: int, i1: int) -> tuple[np.ndarray, np.ndarray]:
    """SED каждой точки (i0, i1) до отрезка [i0, i1] — расстояние не до
    ближайшей точки отрезка (как в DP), а до позиции, интерполированной
    по ДОЛЕ ВРЕМЕНИ (Meratnia & de By, "TD-TR"): считаем, что между i0 и
    i1 движение было бы равномерным по времени вдоль прямой."""
    idx = np.arange(i0 + 1, i1)
    t0, t1 = t[i0], t[i1]
    frac = (t[idx] - t0) / (t1 - t0) if t1 > t0 else np.zeros(len(idx))
    interp = xy[i0] + frac[:, None] * (xy[i1] - xy[i0])
    d = np.hypot(*(xy[idx] - interp).T)
    return idx, d


def simplify_sed_with_indices(track, tol: float = DEFAULT_TOL) -> tuple[np.ndarray, np.ndarray]:
    """Time-aware упрощение: top-down алгоритм в духе Дугласа-Пекера, но
    критерий разреза — SED (см. _seds), а не перпендикулярное расстояние
    до отрезка. Строже пространственного DP при том же tol (учитывает,
    что раскладка точек во времени тоже могла быть неравномерной), поэтому
    обычно требует больше точек для того же tol."""
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
