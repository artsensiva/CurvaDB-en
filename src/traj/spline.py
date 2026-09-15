"""Фиттинг траекторий кубическими B-сплайнами (scipy.interpolate.splprep)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splev, splprep

DEFAULT_TOL = 10.0
DEGREE = 3
MAX_ITER = 40
S_HI_CAP_MULT = 1e6


@dataclass
class SplineFit:
    tck: tuple
    t_min: float
    t_max: float
    s: float
    max_error: float
    converged: bool
    n_control_points: int


def _max_error(x: np.ndarray, y: np.ndarray, u: np.ndarray, tck: tuple) -> float:
    xs, ys = splev(u, tck)
    return float(np.hypot(xs - x, ys - y).max())


def fit(track, tol: float = DEFAULT_TOL, max_iter: int = MAX_ITER) -> SplineFit:
    """Подбирает сглаживающий параметр `s` так, чтобы максимальная ошибка
    аппроксимации в исходных временных метках была <= tol метров.

    Параметризация — нормированное время (u = (t - t_min) / (t_max - t_min)).
    s=0 (интерполяция через все точки) — нижняя граница поиска, далее
    экспоненциальный рост верхней границы и бисекция между последним
    "хорошим" (ошибка <= tol) и первым "плохим" s. Если бисекция не
    сходится за max_iter шагов, возвращается лучшее найденное s
    (наибольшее с ошибкой <= tol) и converged=False.
    """
    x = track.xy[:, 0]
    y = track.xy[:, 1]
    t = track.t
    t_min, t_max = float(t.min()), float(t.max())
    span = t_max - t_min
    u = (t - t_min) / span if span > 0 else np.linspace(0.0, 1.0, len(t))
    k = DEGREE

    def try_fit(s: float):
        # FITPACK иногда не сходится к целевому fp=s за отведённые итерации
        # для конкретного пробного s из нашего поиска — это его внутренний
        # критерий, а не наш; мы всё равно валидируем итоговую ошибку сами.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tck, _ = splprep([x, y], u=u, k=k, s=s)
        return tck, _max_error(x, y, u, tck)

    tck0, err0 = try_fit(0.0)
    if err0 > tol:
        # даже интерполяция через все точки не укладывается в tol —
        # численная особенность, используем что есть
        return SplineFit(tck0, t_min, t_max, 0.0, err0, False, len(tck0[1][0]))

    s_lo, tck_lo, err_lo = 0.0, tck0, err0
    s_cap = max(span, 1.0) * S_HI_CAP_MULT
    s_hi = max(span, 1.0) * 1e-3
    tck_hi, err_hi = tck_lo, err_lo
    grown = 0
    while err_hi <= tol and s_hi < s_cap and grown < max_iter:
        try:
            tck_hi, err_hi = try_fit(s_hi)
        except Exception:
            err_hi = float("inf")
            break
        if err_hi <= tol:
            s_lo, tck_lo, err_lo = s_hi, tck_hi, err_hi
            s_hi *= 4.0
        grown += 1

    if err_hi <= tol:
        # верхняя граница поиска исчерпана, а ошибка всё ещё в допуске —
        # берём максимально сглаженный вариант
        return SplineFit(tck_hi, t_min, t_max, s_hi, err_hi, True, len(tck_hi[1][0]))

    best_s, best_tck, best_err = s_lo, tck_lo, err_lo
    converged = False
    for _ in range(max_iter):
        if (s_hi - s_lo) < max(s_hi, 1.0) * 1e-4:
            converged = True
            break
        s_mid = (s_lo + s_hi) / 2.0
        try:
            tck_mid, err_mid = try_fit(s_mid)
        except Exception:
            s_hi = s_mid
            continue
        if err_mid <= tol:
            s_lo, best_s, best_tck, best_err = s_mid, s_mid, tck_mid, err_mid
        else:
            s_hi = s_mid

    return SplineFit(best_tck, t_min, t_max, best_s, best_err, converged, len(best_tck[1][0]))


def sample(spline: SplineFit, n_points: int) -> np.ndarray:
    """Равномерная выборка n_points точек вдоль параметра u в [0, 1]."""
    u = np.linspace(0.0, 1.0, n_points)
    xs, ys = splev(u, spline.tck)
    return np.column_stack([xs, ys])


def derivatives(spline: SplineFit, t) -> tuple[np.ndarray, np.ndarray]:
    """Скорость и ускорение (м/с, м/с^2) в реальном времени `t` (секунды)."""
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
