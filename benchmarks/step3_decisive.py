"""Step3, п.2: решающий эксперимент -- LSQ-сплайн (равномерные и
адаптивные узлы, src/traj/spline_lsq.py) против DP+SED на синтетике
step2 (дорожная геометрия: прямые + клотоидные повороты) ПЛЮС
переменная скорость (разгоны/торможения/остановки), при разных шагах
наблюдений dt.

Честность подбора параметров -- как в step2_crossover.py: целевой tol не
передаётся методам напрямую, для каждого метода перебирается его
собственный внутренний параметр (tol на зашумлённых точках), честная
ошибка -- max отклонение реконструкции от ИСТИННОЙ (бесшумной) кривой на
густой сетке. В отличие от step2 (лог-сетка из N_INTERNAL_GRID точек),
здесь перебор -- НЕПРЕРЫВНАЯ БИСЕКЦИЯ (см. `_bisect_reachable`).

Оракул (LSQ-сплайн, посаженный напрямую на истинную бесшумную кривую,
`spline_lsq.fit_oracle`) считается ОДИН раз на пару (seed, tol) --
не зависит от sigma/dt, в критериях K1-K3 не участвует, справочная
таблица нижней границы представления геометрии.

Размер -- единая схема для ВСЕХ методов: параметры (время в секундах,
координаты в метрах) -> округление до см/сантисекунд -> int32 ->
дельта-кодирование -> конкатенация потоков -> zlib.compress. `zlib` --
единственная новая зависимость, из stdlib.

Запуск:
    venv/bin/python benchmarks/step3_decisive.py --pilot   # 3 трека, лимит 2 минуты
    venv/bin/python benchmarks/step3_decisive.py           # полный прогон, пишет results/step3.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zlib
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from traj.io import Track  # noqa: E402
from traj.simplify import simplify_sed_with_indices  # noqa: E402
from traj import spline_lsq  # noqa: E402
from step2_crossover import (  # noqa: E402
    _build_path,
    _dp_reconstruct,
    _path_xy,
    _random_road_segments,
    true_curve_error,
)
from _report_utils import upsert_section  # noqa: E402

SEED = 42
N_TRACKS = 15
SIGMA_LIST = [0.0, 0.1, 1.0, 5.0]
TOL_LIST = [0.5, 2.0, 10.0]
DT_LIST = [1.0, 5.0, 15.0]
LENGTH_RANGE_M = (1000.0, 5000.0)
PTS_PER_SEC = 10  # густая сетка для честной ошибки (>= 10 точек/с)
BISECT_MAX_ITER = 30
BYTE_SCALE = 100.0  # см (координаты) / сантисекунды (время)

N_TRACKS_PILOT = 3

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_MD = os.path.join(RESULTS_DIR, "step3.md")
SECTION1_HEADER = "## 1. LSQ-сплайн истинной кривой: контракт"
SECTION2_HEADER = "## 2. Таблицы по dt (сплайн uniform/adaptive vs DP+SED, оракул отдельно)"

METHODS = ("dp_sed", "spline_uniform", "spline_adaptive")

# ------------------------------------------------------- переменная скорость --

CRUISE_RANGE = (5.0, 20.0)  # м/с
ACCEL_RANGE = (1.0, 2.5)  # м/с^2, разгон
DECEL_RANGE = (1.5, 3.0)  # м/с^2, торможение
CRUISE_DURATION_RANGE = (15.0, 90.0)  # с
STOP_DURATION_RANGE = (2.0, 8.0)  # с
STOP_PROB = 0.4


def _variable_speed_profile(rng: np.random.Generator, total_length_m: float):
    """Кусочно-постоянное ускорение: разгон -> крейсерская скорость ->
    (с вероятностью STOP_PROB) торможение до остановки -> пауза ->
    повтор, пока не пройдена total_length_m. Возвращает (pieces,
    duration): pieces -- список (t_start, s_start, v_start, a, dur);
    duration подобрана так, что s(duration) == total_length_m ТОЧНО
    (первый кусок, где кумулятивная длина достигает total_length_m,
    обрезается по формуле кинематики, остальные отбрасываются)."""
    pieces = []
    t_cur = s_cur = v_cur = 0.0
    while s_cur < total_length_m:
        cruise_speed = float(rng.uniform(*CRUISE_RANGE))
        dv = cruise_speed - v_cur
        if abs(dv) > 1e-9:
            a = float(rng.uniform(*ACCEL_RANGE)) * float(np.sign(dv))
            dur = abs(dv / a)
            pieces.append((t_cur, s_cur, v_cur, a, dur))
            s_cur += v_cur * dur + 0.5 * a * dur * dur
            t_cur += dur
            v_cur = cruise_speed

        cruise_dur = float(rng.uniform(*CRUISE_DURATION_RANGE))
        pieces.append((t_cur, s_cur, v_cur, 0.0, cruise_dur))
        s_cur += v_cur * cruise_dur
        t_cur += cruise_dur

        if s_cur < total_length_m and rng.uniform() < STOP_PROB:
            decel = float(rng.uniform(*DECEL_RANGE))
            dur = v_cur / decel
            pieces.append((t_cur, s_cur, v_cur, -decel, dur))
            s_cur += v_cur * dur - 0.5 * decel * dur * dur
            t_cur += dur
            v_cur = 0.0
            stop_dur = float(rng.uniform(*STOP_DURATION_RANGE))
            pieces.append((t_cur, s_cur, 0.0, 0.0, stop_dur))
            t_cur += stop_dur

    cut_idx = len(pieces) - 1
    for i, (t_start, s_start, v_start, a, dur) in enumerate(pieces):
        s_end = s_start + v_start * dur + 0.5 * a * dur * dur
        if s_end >= total_length_m:
            cut_idx = i
            break
    t_start, s_start, v_start, a, dur = pieces[cut_idx]
    target = total_length_m - s_start
    if abs(a) < 1e-9:
        dt = target / v_start if v_start > 1e-9 else 0.0
    else:
        disc = max(v_start * v_start + 2.0 * a * target, 0.0)
        dt = (-v_start + np.sqrt(disc)) / a
    dt = float(np.clip(dt, 0.0, dur))
    pieces = pieces[: cut_idx + 1]
    pieces[-1] = (t_start, s_start, v_start, a, dt)
    duration = t_start + dt
    return pieces, duration


def _s_of_t(pieces, t_query: np.ndarray) -> np.ndarray:
    duration = pieces[-1][0] + pieces[-1][4]
    t_query = np.clip(np.asarray(t_query, dtype=float), 0.0, duration)
    starts = np.array([p[0] for p in pieces])
    idx = np.clip(np.searchsorted(starts, t_query, side="right") - 1, 0, len(pieces) - 1)
    out = np.empty_like(t_query)
    for i in range(len(pieces)):
        mask = idx == i
        if not mask.any():
            continue
        t_start, s_start, v_start, a, dur = pieces[i]
        dt = np.clip(t_query[mask] - t_start, 0.0, dur)
        out[mask] = s_start + v_start * dt + 0.5 * a * dt * dt
    return out


def _geometry_for_seed(seed: int, length_range_m=LENGTH_RANGE_M):
    """Дорожная геометрия (как step2_crossover) + профиль переменной
    скорости -- всё из ОДНОГО geom_rng(seed), не зависит от sigma/dt."""
    geom_rng = np.random.default_rng(seed)
    target_length = float(geom_rng.uniform(*length_range_m))
    segments = _random_road_segments(geom_rng, target_length)
    starts, total_length = _build_path(segments)
    pieces, duration = _variable_speed_profile(geom_rng, total_length)

    def true_xy(t):
        s = _s_of_t(pieces, t)
        return _path_xy(starts, total_length, s)

    return true_xy, duration


def make_variable_speed_track(seed: int, noise_std: float, dt: float, length_range_m=LENGTH_RANGE_M):
    true_xy, duration = _geometry_for_seed(seed, length_range_m)
    t = np.arange(0.0, duration, dt)
    if len(t) == 0 or t[-1] != duration:
        t = np.concatenate([t, [duration]])
    xy_true = true_xy(t)
    if noise_std > 0.0:
        noise_rng = np.random.default_rng((seed, int(round(noise_std * 1000)), int(round(dt * 1000))))
        xy_obs = xy_true + noise_rng.normal(0.0, noise_std, xy_true.shape)
    else:
        xy_obs = xy_true.copy()
    zeros = np.zeros(len(t))
    track = Track(track_id=f"vs{seed}_s{noise_std}_dt{dt}", lat=zeros, lon=zeros, t=t, xy=xy_obs)
    return track, true_xy, duration


def _dense_grid(duration: float) -> np.ndarray:
    dt_dense = 1.0 / PTS_PER_SEC
    t_dense = np.arange(0.0, duration, dt_dense)
    if len(t_dense) == 0 or t_dense[-1] != duration:
        t_dense = np.concatenate([t_dense, [duration]])
    return t_dense


# --------------------------------------------------------------- размер --

INT32_SAFE_BOUND = 2_000_000_000.0


def _stream_bytes(values: np.ndarray, scale: float = BYTE_SCALE) -> bytes:
    """float64 (метры или секунды) -> округление -> int32 -> дельты.

    Клип перед round/astype -- изредка LSQ-фиттер на грани m_max
    возвращает численно разъехавшийся (нефизично большой) сплайн; сам
    кандидат всё равно отбраковывается на этапе honest-проверки (его
    err относительно истины гигантский и никогда не <= target_tol), но
    БЕЗ клипа np.round(huge).astype(int32) кидает RuntimeWarning
    (переполнение) -- клип делает это тихо и детерминированно."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return b""
    scaled = np.clip(values * scale, -INT32_SAFE_BOUND, INT32_SAFE_BOUND)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=INT32_SAFE_BOUND, neginf=-INT32_SAFE_BOUND)
    ints = np.round(scaled).astype(np.int32)
    deltas = np.empty_like(ints)
    deltas[0] = ints[0]
    deltas[1:] = np.diff(ints)
    return deltas.tobytes()


def compressed_size(streams: list[np.ndarray]) -> int:
    raw = b"".join(_stream_bytes(s) for s in streams)
    return len(zlib.compress(raw))


# ------------------------------------------------------------ методы --

def _build_dp_sed(track, itol: float, t_dense, true_xy_dense):
    _, idx = simplify_sed_with_indices(track, itol)
    recon = _dp_reconstruct(t_dense, track.t[idx], track.xy[idx])
    err = true_curve_error(true_xy_dense, recon)
    nbytes = compressed_size([track.t[idx], track.xy[idx, 0], track.xy[idx, 1]])
    n = len(idx) * 3
    return {"n": n, "bytes": nbytes, "err": err}


def _build_spline_lsq(track, itol: float, t_dense, true_xy_dense, mode: str):
    fit_fn = spline_lsq.fit_uniform if mode == "uniform" else spline_lsq.fit_adaptive
    sp = fit_fn(track, itol)
    recon = spline_lsq.reconstruct(sp, t_dense)
    err = true_curve_error(true_xy_dense, recon)
    knots, c_list, _k = sp.tck
    nbytes = compressed_size([knots, c_list[0], c_list[1]])
    n = len(knots) + 2 * len(c_list[0])
    return {"n": n, "bytes": nbytes, "err": err}


COARSE_SCAN_POINTS = 8


def _bisect_between(build_fn, target_tol: float, lo: float, lo_result: dict, hi: float, max_iter: int = BISECT_MAX_ITER) -> dict:
    """Уточняющая бисекция МЕЖДУ lo (проходит target_tol) и hi (не
    проходит) на МАКСИМАЛЬНЫЙ itol (минимум байт), при котором ещё
    err <= target_tol -- только внутри уже найденного грубым перебором
    прохода/провала, без предположения о монотонности err(itol) за
    пределами этой пары."""
    best = lo_result
    lo_b, hi_b = lo, hi
    for _ in range(max_iter):
        if hi_b / lo_b < 1.01:
            break
        mid = (lo_b * hi_b) ** 0.5
        cand = build_fn(mid)
        if cand["err"] <= target_tol:
            lo_b, best = mid, cand
        else:
            hi_b = mid
    return best


def search_min_params(build_fn, target_tol: float) -> dict:
    """Грубый лог-скан (COARSE_SCAN_POINTS точек, как в step2) +
    уточняющая бисекция вокруг лучшей (минимум байт) проходящей точки.

    ВАЖНО: err(itol) для LSQ-сплайна НЕ обязательно монотонна -- слишком
    маленький itol (много узлов, почти интерполяция по шумным и РЕДКИМ
    (dt>=5с) точкам) может дать колебание сплайна МЕЖДУ соседними
    отсчётами (проверено вручную: честная ошибка на густой сетке резко
    росла при этом, хотя остаток в самих точках оставался маленьким --
    внутренний критерий фиттера его не видит, см. коммит spline_lsq.py).
    Поэтому реализуемость проверяется по ЛЮБОЙ точке грубого скана, а не
    по одной самой строгой -- бисекция уточняет только ЛОКАЛЬНО, между
    найденной лучшей проходящей точкой и её ближайшим непроходящим
    соседом по возрастанию itol."""
    grid = target_tol * np.logspace(-2, 1, COARSE_SCAN_POINTS)
    results = [(float(g), build_fn(float(g))) for g in grid]
    passing = [(g, r) for g, r in results if r["err"] <= target_tol]
    if not passing:
        return {"reachable": False}

    g0, r0 = min(passing, key=lambda gr: gr[1]["bytes"])
    hi_bound = next((g for g, r in results if g > g0 and r["err"] > target_tol), None)
    if hi_bound is None:
        return {"reachable": True, **r0}
    best = _bisect_between(build_fn, target_tol, g0, r0, hi_bound)
    return {"reachable": True, **best}


def evaluate_cell(track, true_xy_dense, t_dense, target_tol: float) -> dict:
    return {
        "dp_sed": search_min_params(lambda itol: _build_dp_sed(track, itol, t_dense, true_xy_dense), target_tol),
        "spline_uniform": search_min_params(
            lambda itol: _build_spline_lsq(track, itol, t_dense, true_xy_dense, "uniform"), target_tol
        ),
        "spline_adaptive": search_min_params(
            lambda itol: _build_spline_lsq(track, itol, t_dense, true_xy_dense, "adaptive"), target_tol
        ),
    }


def evaluate_oracle(t_dense, true_xy_dense, target_tol: float) -> dict:
    sp = spline_lsq.fit_oracle(t_dense, true_xy_dense, target_tol)
    knots, c_list, _k = sp.tck
    nbytes = compressed_size([knots, c_list[0], c_list[1]])
    n = len(knots) + 2 * len(c_list[0])
    return {"reachable": sp.converged, "n": n, "bytes": nbytes, "err": sp.max_error}


# ------------------------------------------------------------------ run --

@dataclass
class RunResult:
    cells: dict  # (dt, sigma, tol) -> {method: [per-track dict, ...]}
    oracle: dict  # (seed, tol) -> {"n", "bytes", "err", "reachable"}


def run_grid(n_tracks, sigma_list, tol_list, dt_list, seed=SEED, verbose=True) -> RunResult:
    rng_master = np.random.default_rng(seed)
    geom_seeds = [int(s) for s in rng_master.integers(0, 1_000_000, size=n_tracks)]

    cells = {(dt, s, t): {m: [] for m in METHODS} for dt in dt_list for s in sigma_list for t in tol_list}
    oracle = {}

    for gseed in geom_seeds:
        true_xy, duration = _geometry_for_seed(gseed)
        t_dense_oracle = _dense_grid(duration)
        true_xy_dense_oracle = true_xy(t_dense_oracle)
        for tol in tol_list:
            oracle[(gseed, tol)] = evaluate_oracle(t_dense_oracle, true_xy_dense_oracle, tol)

        for dt in dt_list:
            for sigma in sigma_list:
                track, _true_xy, _duration = make_variable_speed_track(gseed, sigma, dt)
                t_dense = _dense_grid(duration)
                true_xy_dense = true_xy(t_dense)
                for tol in tol_list:
                    res = evaluate_cell(track, true_xy_dense, t_dense, tol)
                    for m in METHODS:
                        cells[(dt, sigma, tol)][m].append(res[m])
        if verbose:
            print(f"трек seed={gseed} готов")
    return RunResult(cells=cells, oracle=oracle)


# --------------------------------------------------------------- отчёт --

def _summarize_method(entries: list[dict]) -> str:
    reachable = [e for e in entries if e["reachable"]]
    n_total = len(entries)
    n_reach = len(reachable)
    if n_reach == 0:
        return "недостижимо"
    mean_bytes = np.mean([e["bytes"] for e in reachable])
    mean_n = np.mean([e["n"] for e in reachable])
    suffix = "" if n_reach == n_total else f", {n_reach}/{n_total} достижимо"
    return f"{mean_bytes:.0f}Б (n={mean_n:.1f}){suffix}"


def format_section1() -> str:
    return "\n".join(
        [
            f"{SECTION1_HEADER}\n",
            (
                "`src/traj/spline_lsq.py` -- LSQ-сплайн x(t)/y(t) напрямую "
                "(`make_lsq_spline`), без привязки к ломаной (в отличие от "
                "`spline.fit()`). Внутренние узлы -- подмножество реальных "
                "отсчётов t (удовлетворяет условию Шёнберга-Уитни "
                "автоматически): `uniform` -- равномерно по индексу; "
                "`adaptive` -- двухпроходно (пробный равномерный фит -> "
                "остатки в точках -> перераспределение узлов по кумулятивному "
                "остатку -> повторный фит). Бисекция по числу внутренних "
                "узлов m растит его до теоретического предела (n-k-2), но "
                "отслеживает ЛУЧШУЮ (не последнюю) ошибку по ходу роста -- у "
                "самой границы (почти-интерполяция шумных и/или редких точек) "
                "изредка возникает численный разрыв ошибки на 1-2 порядка, "
                "невидимый для остатка в ТРЕНИРОВОЧНЫХ точках (см. коммит "
                "spline_lsq.py). Честная ошибка внутри фиттера -- ПРЯМОЙ "
                "Евклидов остаток в самих точках (t_i, xy_i), не "
                "point-to-segment до ломаной -- и именно поэтому НЕ видит "
                "колебания сплайна МЕЖДУ точками (см. следующий абзац).\n"
            ),
            (
                "`fit_oracle` -- тот же фиттер, посаженный НАПРЯМУЮ на "
                "плотную истинную (бесшумную) кривую; не знает о шуме или "
                "разрежённости наблюдений -- справочная нижняя граница "
                "представления геометрии сплайном при заданном tol, "
                "считается один раз на (seed, tol), в критериях K1-K3 не "
                "участвует.\n"
            ),
            (
                "Подбор внутреннего параметра метода (tol на зашумлённых "
                "точках) -- грубый лог-скан (8 точек, как в step2) + "
                "уточняющая бисекция вокруг лучшей проходящей точки "
                "(`search_min_params`/`_bisect_between` в step3_decisive.py), "
                "а НЕ чистая бисекция от самого строгого internal_tol: err(itol) "
                "для LSQ-сплайна не монотонна -- слишком маленький itol на "
                "редких (dt>=5с) точках может дать сплайн, который проходит "
                "близко к отсчётам, но дико колеблется МЕЖДУ ними (честная "
                "ошибка на густой сетке при этом на порядки хуже, хотя остаток "
                "в самих точках маленький -- внутренний критерий фиттера этого "
                "не видит; обнаружено эмпирически при отладке). Грубый скан "
                "устойчив к этой немонотонности, чистая бисекция от самого "
                "строгого конца иногда ошибочно помечала клетку недостижимой. "
                "Размер -- единая схема для всех методов: время/координаты -> "
                "округление до см/сантисекунд -> int32 (с клипом от переполнения) "
                "-> дельты -> zlib.compress.\n"
            ),
        ]
    ) + "\n"


def format_section2(result: RunResult, sigma_list, tol_list, dt_list, n_tracks) -> str:
    lines = [f"{SECTION2_HEADER}\n", f"{n_tracks} треков, seed={SEED}.\n"]
    header = "| sigma \\ tol | " + " | ".join(f"{t:g}" for t in tol_list) + " |"

    for dt in dt_list:
        lines.append(f"### dt = {dt:g} с\n")
        for method_key, title in (
            ("dp_sed", "DP+SED"),
            ("spline_uniform", "LSQ-сплайн (uniform)"),
            ("spline_adaptive", "LSQ-сплайн (adaptive)"),
        ):
            lines.append(f"#### {title}: байты (n параметров), доля достижимых треков")
            lines.append(header)
            lines.append("|" + "---|" * (len(tol_list) + 1))
            for sigma in sigma_list:
                row = [f"{sigma:g}"] + [
                    _summarize_method(result.cells[(dt, sigma, tol)][method_key]) for tol in tol_list
                ]
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    lines.append("### Оракул (LSQ-сплайн по истинной кривой, справочно -- один раз на (seed, tol), не зависит от sigma/dt)\n")
    lines.append("| tol | среднее n | средние байты | доля достижимых (сходимость бисекции) |")
    lines.append("|---|---|---|---|")
    geom_seeds = sorted({s for (s, _tol) in result.oracle})
    for tol in tol_list:
        entries = [result.oracle[(s, tol)] for s in geom_seeds]
        reach = [e for e in entries if e["reachable"]]
        if reach:
            mean_n = np.mean([e["n"] for e in reach])
            mean_bytes = np.mean([e["bytes"] for e in reach])
            lines.append(f"| {tol:g} | {mean_n:.1f} | {mean_bytes:.0f} | {len(reach)}/{len(entries)} |")
        else:
            lines.append(f"| {tol:g} | -- | -- | 0/{len(entries)} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="бюджетный прогон (3 трека), без записи в .md")
    parser.add_argument("--n-tracks", type=int, default=None, help="переопределить N_TRACKS полного прогона")
    args = parser.parse_args()

    if args.pilot:
        print(f"ПИЛОТ: {N_TRACKS_PILOT} треков, sigma={SIGMA_LIST}, tol={TOL_LIST}, dt={DT_LIST}")
        t0 = time.time()
        result = run_grid(N_TRACKS_PILOT, SIGMA_LIST, TOL_LIST, DT_LIST)
        elapsed = time.time() - t0
        n_cells_pilot = N_TRACKS_PILOT * len(SIGMA_LIST) * len(TOL_LIST) * len(DT_LIST)
        n_cells_full = N_TRACKS * len(SIGMA_LIST) * len(TOL_LIST) * len(DT_LIST)
        extrapolated = elapsed * (n_cells_full / n_cells_pilot)
        print(f"\nпилот занял {elapsed:.1f}с ({n_cells_pilot} (трек x sigma x tol x dt) комбинаций)")
        print(f"экстраполяция на полную сетку ({n_cells_full} комбинаций): ~{extrapolated:.0f}с (~{extrapolated / 60:.1f} мин)")
        print(format_section2(result, SIGMA_LIST, TOL_LIST, DT_LIST, N_TRACKS_PILOT))
        return

    n_tracks = args.n_tracks if args.n_tracks is not None else N_TRACKS
    print(f"ПОЛНЫЙ ПРОГОН: {n_tracks} треков, sigma={SIGMA_LIST}, tol={TOL_LIST}, dt={DT_LIST}")
    t0 = time.time()
    result = run_grid(n_tracks, SIGMA_LIST, TOL_LIST, DT_LIST)
    elapsed = time.time() - t0
    print(f"\nполный прогон занял {elapsed:.1f}с ({elapsed / 60:.1f} мин)")

    body1 = format_section1()
    body2 = format_section2(result, SIGMA_LIST, TOL_LIST, DT_LIST, n_tracks)
    if n_tracks != N_TRACKS:
        body2 += (
            f"\n**Ограничение бюджета времени:** N_TRACKS уменьшен с {N_TRACKS} до "
            f"{n_tracks} по результатам пилота/полного прогона. Полный прогон занял "
            f"{elapsed:.1f}с ({elapsed / 60:.1f} мин) на {n_tracks} треках.\n"
        )
    upsert_section(OUT_MD, SECTION1_HEADER, body1)
    upsert_section(OUT_MD, SECTION2_HEADER, body2)
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
