"""Честный контракт фиттинга: ошибка <= tol на ГУСТОЙ сетке между
соседними точками, а не только в самих точках (см. src/traj/spline.py,
benchmarks/results/step0_diagnostics.md)."""

import numpy as np

from traj.io import Track
from traj.spline import dense_check, fit

TOL = 10.0


def _sharp_turn_track(seed: int = 0) -> Track:
    """Резкий поворот ~90° + неравномерный dt (1..25с) — случай, где
    интерполирующий сплайн склонен переколебаться между разреженными
    узлами вблизи угла."""
    rng = np.random.default_rng(seed)
    n1, n2 = 25, 25
    leg1 = np.column_stack([np.linspace(0, 500, n1), np.zeros(n1)])
    leg2 = np.column_stack([np.full(n2, 500.0), np.linspace(0, 500, n2)])
    xy = np.vstack([leg1, leg2[1:]]) + rng.normal(0.0, 1.0, (n1 + n2 - 1, 2))
    n = len(xy)
    dt = rng.uniform(1.0, 25.0, size=n - 1)
    t = np.concatenate([[0.0], np.cumsum(dt)])
    zeros = np.zeros(n)
    return Track(track_id="sharp_turn", lat=zeros, lon=zeros, t=t, xy=xy)


def _stop_and_go_track(seed: int = 0) -> Track:
    """Случайные броски направления + кластеры близких точек (стоянки),
    dt в пределах 1..29с (как после чистки src/traj/clean.py, где dt <=
    30с — резки не происходит)."""
    rng = np.random.default_rng(seed)
    pos = np.array([0.0, 0.0])
    t_cursor = 0.0
    times = [0.0]
    points = [pos.copy()]
    for _ in range(8):
        heading = rng.uniform(0, 2 * np.pi)
        step = rng.uniform(20.0, 150.0)
        pos = pos + step * np.array([np.cos(heading), np.sin(heading)])
        t_cursor += rng.uniform(5.0, 29.0)
        points.append(pos.copy() + rng.normal(0.0, 1.0, 2))
        times.append(t_cursor)
        for _ in range(int(rng.integers(2, 5))):
            t_cursor += rng.uniform(1.0, 5.0)
            points.append(pos.copy() + rng.normal(0.0, 0.5, 2))
            times.append(t_cursor)
    xy = np.array(points)
    t = np.array(times)
    n = len(xy)
    zeros = np.zeros(n)
    return Track(track_id="stop_and_go", lat=zeros, lon=zeros, t=t, xy=xy)


def test_dense_error_within_tol_sharp_turn():
    for seed in range(5):
        track = _sharp_turn_track(seed)
        sp = fit(track, tol=TOL)
        max_err, _ = dense_check(track, sp)
        assert max_err <= TOL + 1e-6, f"seed={seed} max_err={max_err}"


def test_dense_error_within_tol_stop_and_go():
    for seed in range(5):
        track = _stop_and_go_track(seed)
        sp = fit(track, tol=TOL)
        max_err, _ = dense_check(track, sp)
        assert max_err <= TOL + 1e-6, f"seed={seed} max_err={max_err}"


def test_densification_can_trigger():
    """Хотя бы для части сложных треков фиттинг реально добавляет
    synthetic-узлы (n_points_used > n_raw) — проверяем, что путь кода с
    добавлением узлов действительно исполняется, а не просто недостижим."""
    triggered = False
    for seed in range(10):
        track = _sharp_turn_track(seed)
        sp = fit(track, tol=TOL)
        if sp.n_points_used > len(track.t):
            triggered = True
            break
    assert triggered


def test_sp_max_error_matches_dense_check():
    """sp.max_error, посчитанный внутри fit(), должен совпадать с
    независимой проверкой dense_check на том же треке."""
    track = _sharp_turn_track(seed=1)
    sp = fit(track, tol=TOL)
    max_err, _ = dense_check(track, sp)
    assert abs(sp.max_error - max_err) < 1e-6
