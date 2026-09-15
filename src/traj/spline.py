"""Фиттинг траекторий кубическими B-сплайнами (scipy.interpolate.splprep).

Честный контракт: max_error — это ошибка на ГУСТОЙ сетке (>= PTS_PER_INTERVAL
точек на интервал между каждой парой соседних точек исходного трека),
измеренная как point-to-segment расстояние до прямого сегмента между этими
двумя точками — а не только в самих точках (см.
benchmarks/results/step0_diagnostics.md: старая версия контролировала
ошибку исключительно в узлах и пропускала колебания между ними у 82%
треков).

Если сплайн при s=0 (интерполяция) всё равно нарушает tol между какой-то
парой соседних точек, в этот интервал добавляется synthetic-узел — точка
на прямой между исходными точками в месте максимального отклонения
(адаптивное сгущение, до MAX_DENSIFY_ROUNDS раундов). Только когда густая
проверка проходит на s=0, включается прежний механизм роста `s`
(бисекция) для компактности — но уже с густой ошибкой как критерием, а не
ошибкой только в узлах.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splev, splprep

DEFAULT_TOL = 10.0
DEGREE = 3
MAX_ITER = 40
S_HI_CAP_MULT = 1e6
PTS_PER_INTERVAL = 10  # >= точек густой сетки на интервал между соседними точками
MAX_DENSIFY_ROUNDS = 8
MAX_POINTS_MULT = 6  # верхняя граница числа точек (raw + synthetic) = MULT * n_raw


@dataclass
class SplineFit:
    tck: tuple
    t_min: float
    t_max: float
    s: float
    max_error: float  # густая ошибка (между метками, не только в них)
    converged: bool
    n_control_points: int
    parametrization: str = "time"
    n_points_used: int = 0  # raw + synthetic-узлы, добавленные при фиттинге


def _param_u(t: np.ndarray, xy: np.ndarray, mode: str) -> np.ndarray:
    if mode == "chord":
        seg = np.maximum(np.hypot(*np.diff(xy, axis=0).T), 1e-6)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        return cum / total if total > 0 else np.linspace(0.0, 1.0, len(xy))
    if mode != "time":
        raise ValueError(f"неизвестная параметризация: {mode!r}")
    t_min, t_max = float(t.min()), float(t.max())
    span = t_max - t_min
    return (t - t_min) / span if span > 0 else np.linspace(0.0, 1.0, len(t))


def _point_seg_dist(px: np.ndarray, py: np.ndarray, ax: float, ay: float, bx: float, by: float) -> np.ndarray:
    """Point-to-segment расстояние (векторизовано по px, py; сегмент [a, b] скаляры)."""
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby
    if ab2 == 0.0:
        return np.hypot(px - ax, py - ay)
    tt = np.clip(((px - ax) * abx + (py - ay) * aby) / ab2, 0.0, 1.0)
    cx, cy = ax + tt * abx, ay + tt * aby
    return np.hypot(px - cx, py - cy)


def _dense_scan(
    t0: np.ndarray, xy0: np.ndarray, tck: tuple, mode: str, pts_per_interval: int
) -> tuple[np.ndarray, np.ndarray]:
    """Для каждой пары соседних точек (t0, xy0) — густая выборка сплайна на
    их интервале параметра u и max point-to-segment расстояние до прямого
    сегмента между ними. Возвращает (per_interval_max, доля_u_худшей_точки)."""
    u0 = _param_u(t0, xy0, mode)
    n = len(t0)
    per_interval_max = np.empty(max(n - 1, 0))
    worst_frac = np.empty(max(n - 1, 0))
    for i in range(n - 1):
        grid = np.linspace(u0[i], u0[i + 1], pts_per_interval)
        xs, ys = splev(grid, tck)
        d = _point_seg_dist(xs, ys, xy0[i, 0], xy0[i, 1], xy0[i + 1, 0], xy0[i + 1, 1])
        j = int(np.argmax(d))
        per_interval_max[i] = float(d[j])
        denom = grid[-1] - grid[0]
        worst_frac[i] = float((grid[j] - grid[0]) / denom) if denom > 0 else 0.5
    return per_interval_max, worst_frac


def dense_check(track, spline: SplineFit, pts_per_interval: int = PTS_PER_INTERVAL) -> tuple[float, np.ndarray]:
    """Независимая проверка густой ошибки на исходном треке (не зависит от
    внутренних synthetic-узлов, использованных при фиттинге). Возвращает
    (max_error, per_interval_max)."""
    t = np.asarray(track.t, dtype=float)
    xy = np.asarray(track.xy, dtype=float)
    per_interval_max, _ = _dense_scan(t, xy, spline.tck, spline.parametrization, pts_per_interval)
    max_error = float(per_interval_max.max()) if len(per_interval_max) else 0.0
    return max_error, per_interval_max


def fit(
    track,
    tol: float = DEFAULT_TOL,
    max_iter: int = MAX_ITER,
    parametrization: str = "time",
    pts_per_interval: int = PTS_PER_INTERVAL,
    max_densify_rounds: int = MAX_DENSIFY_ROUNDS,
    max_points_mult: int = MAX_POINTS_MULT,
) -> SplineFit:
    """Подбирает кубический B-сплайн так, чтобы ошибка на ГУСТОЙ сетке между
    каждой парой соседних исходных точек была <= tol (см. docstring модуля).

    1. s=0 (интерполяция) на исходных точках; если густая проверка не
       проходит — добавляем по одной synthetic-точке в месте максимального
       отклонения в каждом ещё нарушенном интервале (адаптивное сгущение,
       линейная интерполяция t и xy вдоль прямого сегмента между исходными
       точками), повторяем до max_densify_rounds раз или пока не упрёмся в
       max_points_mult * n_raw точек.
    2. Как только s=0 на (возможно дополненном) наборе точек проходит
       густую проверку — растим `s` (как раньше: экспоненциальный рост +
       бисекция), но критерий теперь густая ошибка, а не ошибка в узлах.

    Если даже после исчерпания бюджета sinthetic-узлов густая ошибка > tol,
    возвращается лучший найденный (s=0) вариант с converged=False.
    """
    t0 = np.asarray(track.t, dtype=float)
    xy0 = np.asarray(track.xy, dtype=float)
    n0 = len(t0)
    t_min, t_max = float(t0.min()), float(t0.max())
    k = DEGREE

    def try_fit(t_arr: np.ndarray, xy_arr: np.ndarray, s: float) -> tuple:
        u_arr = _param_u(t_arr, xy_arr, parametrization)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tck, _ = splprep([xy_arr[:, 0], xy_arr[:, 1]], u=u_arr, k=k, s=s)
        return tck

    t_aug, xy_aug = t0.copy(), xy0.copy()
    tck0 = None
    dense_max_error = float("inf")
    dense_ok = False

    for round_i in range(max_densify_rounds + 1):
        tck0 = try_fit(t_aug, xy_aug, 0.0)
        per_interval_max, worst_frac = _dense_scan(t0, xy0, tck0, parametrization, pts_per_interval)
        dense_max_error = float(per_interval_max.max()) if len(per_interval_max) else 0.0

        if dense_max_error <= tol:
            dense_ok = True
            break
        if round_i == max_densify_rounds or len(t_aug) >= max_points_mult * n0:
            break

        bad = np.nonzero(per_interval_max > tol)[0]
        new_t, new_xy = [], []
        for i in bad:
            frac = min(max(float(worst_frac[i]), 1e-3), 1 - 1e-3)
            new_t.append(t0[i] + frac * (t0[i + 1] - t0[i]))
            new_xy.append(xy0[i] + frac * (xy0[i + 1] - xy0[i]))

        t_aug = np.concatenate([t_aug, new_t])
        xy_aug = np.concatenate([xy_aug, np.array(new_xy)], axis=0)
        order = np.argsort(t_aug, kind="stable")
        t_aug, xy_aug = t_aug[order], xy_aug[order]

    if not dense_ok:
        return SplineFit(
            tck0, t_min, t_max, 0.0, dense_max_error, False, len(tck0[1][0]),
            parametrization=parametrization, n_points_used=len(t_aug),
        )

    def dense_err_for_s(s: float) -> tuple:
        tck = try_fit(t_aug, xy_aug, s)
        per_int, _ = _dense_scan(t0, xy0, tck, parametrization, pts_per_interval)
        err = float(per_int.max()) if len(per_int) else 0.0
        return tck, err

    s_lo, tck_lo, err_lo = 0.0, tck0, dense_max_error
    s_span = max(t_max - t_min, 1.0)
    s_cap = s_span * S_HI_CAP_MULT
    s_hi = s_span * 1e-3
    tck_hi, err_hi = tck_lo, err_lo
    grown = 0
    while err_hi <= tol and s_hi < s_cap and grown < max_iter:
        try:
            tck_hi, err_hi = dense_err_for_s(s_hi)
        except Exception:
            err_hi = float("inf")
            break
        if err_hi <= tol:
            s_lo, tck_lo, err_lo = s_hi, tck_hi, err_hi
            s_hi *= 4.0
        grown += 1

    if err_hi <= tol:
        return SplineFit(
            tck_hi, t_min, t_max, s_hi, err_hi, True, len(tck_hi[1][0]),
            parametrization=parametrization, n_points_used=len(t_aug),
        )

    best_s, best_tck, best_err = s_lo, tck_lo, err_lo
    bisect_converged = False
    for _ in range(max_iter):
        if (s_hi - s_lo) < max(s_hi, 1.0) * 1e-4:
            bisect_converged = True
            break
        s_mid = (s_lo + s_hi) / 2.0
        try:
            tck_mid, err_mid = dense_err_for_s(s_mid)
        except Exception:
            s_hi = s_mid
            continue
        if err_mid <= tol:
            s_lo, best_s, best_tck, best_err = s_mid, s_mid, tck_mid, err_mid
        else:
            s_hi = s_mid

    return SplineFit(
        best_tck, t_min, t_max, best_s, best_err, bisect_converged, len(best_tck[1][0]),
        parametrization=parametrization, n_points_used=len(t_aug),
    )


def sample(spline: SplineFit, n_points: int) -> np.ndarray:
    """Равномерная выборка n_points точек вдоль параметра u в [0, 1]."""
    u = np.linspace(0.0, 1.0, n_points)
    xs, ys = splev(u, spline.tck)
    return np.column_stack([xs, ys])


def derivatives(spline: SplineFit, t) -> tuple[np.ndarray, np.ndarray]:
    """Скорость и ускорение (м/с, м/с^2) в реальном времени `t` (секунды).

    Требует parametrization="time" (линейная связь u<->t); для "chord"
    восстановление кинематики не реализовано (нужна отдельная инверсия
    t(u), не нужная за пределами сравнения параметризаций в step1)."""
    if spline.parametrization != "time":
        raise NotImplementedError(
            f"derivatives() не поддерживает parametrization={spline.parametrization!r}"
        )
    t = np.asarray(t, dtype=float)
    span = spline.t_max - spline.t_min
    if span > 0:
        u = np.clip((t - spline.t_min) / span, 0.0, 1.0)
        scale1 = 1.0 / span
    else:
        u = np.zeros_like(t)
        scale1 = 0.0
    scale2 = scale1 ** 2

    dx, dy = splev(u, spline.tck, der=1)
    d2x, d2y = splev(u, spline.tck, der=2)
    velocity = np.column_stack([dx, dy]) * scale1
    acceleration = np.column_stack([d2x, d2y]) * scale2
    return velocity, acceleration
