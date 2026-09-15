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
