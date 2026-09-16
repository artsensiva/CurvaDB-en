"""Фиттинг траекторий кубическими B-сплайнами методом наименьших квадратов
по x(t), y(t) напрямую (`scipy.interpolate.make_lsq_spline`) -- БЕЗ
привязки к ломаной зашумлённых точек (в отличие от `spline.fit()`, см.
его docstring и docs/prompts/step3.md: контракт "держаться у ломаной"
не даёт старому фиттеру сглаживать шум). Сплайн не обязан проходить
через зашумлённые точки -- поэтому ошибка здесь считается ПРЯМЫМ
Евклидовым остатком в самих точках (t_i, xy_i), а не point-to-segment
до ломаной (`spline.dense_max_error`).

Два способа расстановки внутренних узлов:
- "uniform" -- равномерно по индексу вдоль t;
- "adaptive" -- двухпроходно: равномерный пробный фит с m узлами,
  остатки в точках, перераспределение тех же m узлов по кумулятивной
  сумме |остатков| (там, где пробный фит хуже всего описывает данные --
  больше узлов), повторный фит. Кривизна истинной кривой НЕ используется
  -- только остатки зашумлённого пробного фита.

Внутренние узлы всегда выбираются как ПОДМНОЖЕСТВО реальных отсчётов t
(а не произвольные вещественные позиции) -- это автоматически
удовлетворяет условию Шёнберга-Уитни (между любыми соседними узлами
есть хотя бы один отсчёт данных), которое требует `make_lsq_spline`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline

DEGREE = 3
MAX_ITER = 40


@dataclass
class LsqSplineFit:
    tck: tuple  # (knots, [cx, cy], k) -- совместимо по форме со SplineFit.tck
    t_min: float
    t_max: float
    max_error: float  # max Евклидов остаток в точках (t_i, xy_i), НЕ до ломаной
    converged: bool
    n_control_points: int
    knot_mode: str  # "uniform" | "adaptive" | "oracle"
    n_internal_knots: int


def _pick_indices(n: int, m: int, weights: np.ndarray | None = None) -> np.ndarray:
    """m строго различных индексов в (0, n-1) -- внутренние узлы как
    подмножество отсчётов данных (гарантирует условие Шёнберга-Уитни).
    weights (длина n, опционально) -- веса точек для смещения индексов к
    зонам с большим весом (адаптивная расстановка); None -- равномерно."""
    if m <= 0:
        return np.empty(0, dtype=int)
    if weights is None:
        pos = np.linspace(0.0, n - 1, m + 2)[1:-1]
    else:
        cum = np.concatenate([[0.0], np.cumsum(weights)])
        targets = np.linspace(0.0, cum[-1], m + 2)[1:-1]
        pos = np.interp(targets, cum, np.arange(n + 1, dtype=float))
    idx = np.clip(np.round(pos).astype(int), 1, n - 2)
    idx = np.unique(idx)
    if len(idx) < m:
        free = np.setdiff1d(np.arange(1, n - 1), idx)
        need = m - len(idx)
        if len(free) >= need > 0:
            extra = free[np.linspace(0, len(free) - 1, need).round().astype(int)]
            idx = np.unique(np.concatenate([idx, extra]))
    return idx


def _full_knot_vector(t_min: float, t_max: float, internal_knots: np.ndarray, k: int) -> np.ndarray:
    return np.concatenate([np.full(k + 1, t_min), internal_knots, np.full(k + 1, t_max)])


def _lsq_fit(t: np.ndarray, xy: np.ndarray, k: int, m: int, weights: np.ndarray | None = None) -> BSpline:
    idx = _pick_indices(len(t), m, weights=weights)
    internal_knots = t[idx]
    full_knots = _full_knot_vector(float(t[0]), float(t[-1]), internal_knots, k)
    return make_lsq_spline(t, xy, full_knots, k=k)


def _residual_max(t: np.ndarray, xy: np.ndarray, bspline: BSpline) -> float:
    pred = bspline(t)
    return float(np.max(np.hypot(*(pred - xy).T)))


def _build_uniform(t: np.ndarray, xy: np.ndarray, k: int, m: int) -> tuple[BSpline, float]:
    bs = _lsq_fit(t, xy, k, m, weights=None)
    return bs, _residual_max(t, xy, bs)


def _build_adaptive(t: np.ndarray, xy: np.ndarray, k: int, m: int) -> tuple[BSpline, float]:
    if m <= 0:
        return _build_uniform(t, xy, k, m)
    bs0 = _lsq_fit(t, xy, k, m, weights=None)
    residual0 = np.hypot(*(bs0(t) - xy).T)
    bs1 = _lsq_fit(t, xy, k, m, weights=residual0)
    return bs1, _residual_max(t, xy, bs1)


def _make_fit(bs: BSpline, t: np.ndarray, err: float, converged: bool, knot_mode: str, m: int) -> LsqSplineFit:
    tck = (bs.t, [bs.c[:, 0], bs.c[:, 1]], bs.k)
    return LsqSplineFit(
        tck=tck,
        t_min=float(t[0]),
        t_max=float(t[-1]),
        max_error=float(err),
        converged=converged,
        n_control_points=len(tck[1][0]),
        knot_mode=knot_mode,
        n_internal_knots=m,
    )


def _bisect_fit(t: np.ndarray, xy: np.ndarray, tol: float, k: int, max_iter: int, knot_mode: str) -> LsqSplineFit:
    """Растим число внутренних узлов m (экспоненциально), пока честная
    ошибка (прямой остаток в точках) не станет <= tol, затем бисекция на
    минимальное m -- по аналогии с ростом `s`/бисекцией в spline.fit().

    m_max -- теоретический потолок (n-k-2, столько узлов ещё оставляет
    систему МНК переопределённой). У самой границы (почти-интерполяция
    на зашумлённых данных) изредка возникает численный разрыв ошибки на
    1-2 порядка -- поэтому в ходе роста отслеживается ЛУЧШИЙ (не
    последний) результат; если tol недостижим, возвращается именно он,
    а не потенциально испорченная попытка у самой границы m_max."""
    n = len(t)
    m_max = max(n - k - 2, 0)
    build = _build_adaptive if knot_mode == "adaptive" else _build_uniform

    best_bs, best_err = build(t, xy, k, 0)
    best_m = 0
    if best_err <= tol or m_max == 0:
        return _make_fit(best_bs, t, best_err, best_err <= tol, knot_mode, 0)

    m_lo, m_hi = 0, 1
    success = None  # (bs, err, m) -- первое m, достигшее tol
    grown = 0
    while grown < max_iter:
        try:
            bs_hi, err_hi = build(t, xy, k, m_hi)
        except (ValueError, np.linalg.LinAlgError):
            bs_hi, err_hi = None, float("inf")
        if bs_hi is not None and err_hi < best_err:
            best_bs, best_err, best_m = bs_hi, err_hi, m_hi
        if err_hi <= tol:
            success = (bs_hi, err_hi, m_hi)
            break
        if m_hi >= m_max:
            break
        m_lo, m_hi = m_hi, min(m_hi * 2, m_max)
        grown += 1

    if success is None:
        # tol недостижим при m <= m_max -- лучший найденный результат
        return _make_fit(best_bs, t, best_err, False, knot_mode, best_m)

    best_bs, best_err, best_m = success
    lo_b, hi_b = m_lo, best_m
    for _ in range(max_iter):
        if hi_b - lo_b <= 1:
            break
        m_mid = (lo_b + hi_b) // 2
        try:
            bs_mid, err_mid = build(t, xy, k, m_mid)
        except (ValueError, np.linalg.LinAlgError):
            err_mid = float("inf")
        if err_mid <= tol:
            hi_b, best_bs, best_err, best_m = m_mid, bs_mid, err_mid, m_mid
        else:
            lo_b = m_mid
    return _make_fit(best_bs, t, best_err, True, knot_mode, best_m)


def fit_uniform(track, tol: float, k: int = DEGREE, max_iter: int = MAX_ITER) -> LsqSplineFit:
    """Равномерные (по индексу отсчётов) внутренние узлы, бисекция по их
    числу до честного остатка <= tol в самих (зашумлённых) точках трека."""
    t = np.asarray(track.t, dtype=float)
    xy = np.asarray(track.xy, dtype=float)
    return _bisect_fit(t, xy, tol, k, max_iter, knot_mode="uniform")


def fit_adaptive(track, tol: float, k: int = DEGREE, max_iter: int = MAX_ITER) -> LsqSplineFit:
    """Как fit_uniform, но на каждом кандидате m -- двухпроходный фит
    (пробный равномерный -> остатки -> перераспределение узлов по
    кумулятивному остатку -> повторный фит)."""
    t = np.asarray(track.t, dtype=float)
    xy = np.asarray(track.xy, dtype=float)
    return _bisect_fit(t, xy, tol, k, max_iter, knot_mode="adaptive")


def fit_oracle(t_dense: np.ndarray, true_xy_dense: np.ndarray, tol: float, k: int = DEGREE, max_iter: int = MAX_ITER) -> LsqSplineFit:
    """Фит НАПРЯМУЮ на плотную истинную (бесшумную) кривую, равномерные
    узлы, бисекция m по ошибке на той же густой сетке. Не знает о шуме
    или разрежённости наблюдений -- справочная нижняя граница
    представления геометрии при заданном tol (используется только для
    оракульской таблицы в бенчмарке, не участвует в критериях)."""
    t = np.asarray(t_dense, dtype=float)
    xy = np.asarray(true_xy_dense, dtype=float)
    return _bisect_fit(t, xy, tol, k, max_iter, knot_mode="oracle")


def reconstruct(fit: LsqSplineFit, t) -> np.ndarray:
    """Восстанавливает xy сплайна fit в моментах времени t."""
    knots, c_list, k = fit.tck
    c = np.column_stack(c_list)
    bs = BSpline(knots, c, k)
    return bs(np.asarray(t, dtype=float))
