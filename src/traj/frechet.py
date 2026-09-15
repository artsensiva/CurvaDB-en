"""Дискретная метрика Фреше (Eiter-Mannila) с numba и ранним выходом."""

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
    """Классическая ДП Eiter-Mannila.

    Ранний выход: ca[i][j] — минимальная по монотонным путям "цена" (максимум
    попарных расстояний) достижения ячейки (i, j). Любой монотонный путь до
    (n-1, m-1) проходит через ровно одну ячейку в каждой строке i, а его цена
    не меньше min_j ca[i][j] (цены достижения этой ячейки). Значит если
    min_j ca[i][j] > threshold, итоговое расстояние тоже > threshold — можно
    прервать вычисление, не досчитывая оставшиеся строки.
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
    """max(|start_a-start_b|, |end_a-end_b|) — нижняя граница Фреше."""
    d_start = float(np.hypot(*(P[0] - Q[0])))
    d_end = float(np.hypot(*(P[-1] - Q[-1])))
    return max(d_start, d_end)


def distance(P: np.ndarray, Q: np.ndarray) -> float:
    """Точная дискретная метрика Фреше между полилиниями P и Q."""
    P = np.ascontiguousarray(P, dtype=np.float64)
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    value, _ = _discrete_frechet_dp(P, Q, np.inf)
    return float(value)


def distance_within(P: np.ndarray, Q: np.ndarray, threshold: float) -> tuple[float, bool]:
    """Фреше с ранним выходом по `threshold`.

    Возвращает (значение, exact). Если lower_bound уже превышает threshold,
    ДП не запускается вовсе. Если ДП прерывается раньше срока, возвращаемое
    значение — валидная нижняя граница (>= threshold), exact=False. При
    exact=True значение точное (могло оказаться и <=, и > threshold).
    """
    lb = lower_bound(P, Q)
    if lb > threshold:
        return lb, False
    P = np.ascontiguousarray(P, dtype=np.float64)
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    value, exact = _discrete_frechet_dp(P, Q, threshold)
    return float(value), exact
