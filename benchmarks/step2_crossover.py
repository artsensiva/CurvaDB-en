"""Step 2, п.1-2: где отношение шум/tol переворачивает сжатие в пользу
сплайна на ГЛАДКИХ (дорожно-подобных) синтетических траекториях.

Синтетика: прямые + дуги окружности, соединённые клотоидами (спираль
Эйлера, замкнутая форма через интегралы Френеля) — кривизна и курс
непрерывны на всех стыках, в отличие от step1_kinematics.py (там
повороты — дуги без клотоидных переходов). Скорость постоянна на весь
трек (геометрия сжатия изучается отдельно от кинематики — она уже
разобрана в step1 §4).

Честное сравнение: целевой tol из сетки — это допуск на ошибку
ОТНОСИТЕЛЬНО ИСТИННОЙ (бесшумной) кривой на густой сетке, а не
параметр, передаваемый методам напрямую. У каждого метода (DP,
DP+SED, сплайн-time, сплайн-chord) свой внутренний параметр
(порог на ЗАШУМЛЁННЫХ точках); для каждой клетки (sigma, tol, метод,
трек) перебором внутреннего параметра ищется МИНИМАЛЬНОЕ число
параметров представления, при котором честная ошибка <= целевого tol.
Недостижимые клетки (даже самая генерозная точка перебора не проходит
tol — типично при sigma >= tol) помечаются явно, без траты времени на
дальнейший перебор.

Запуск:
    venv/bin/python benchmarks/step2_crossover.py --pilot   # бюджетная проверка (5 треков, малая сетка)
    venv/bin/python benchmarks/step2_crossover.py           # полный прогон, пишет results/step2.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splev
from scipy.special import fresnel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from traj.io import Track  # noqa: E402
from traj.simplify import simplify_sed_with_indices, simplify_with_indices  # noqa: E402
from traj.spline import fit as spline_fit  # noqa: E402
from traj.spline import _param_u  # noqa: E402
from _report_utils import upsert_section  # noqa: E402

SEED = 42
N_TRACKS = 30
SIGMA_LIST = [0.0, 0.02, 0.1, 0.5, 2.0, 5.0]
TOL_LIST = [0.05, 0.2, 1.0, 5.0, 20.0]
LENGTH_RANGE_M = (1000.0, 5000.0)
SPEED_RANGE_MPS = (8.0, 20.0)
PTS_PER_SEC = 10  # густая сетка для честной ошибки: >= 10 точек/с
N_INTERNAL_GRID = 8  # точек перебора внутреннего параметра метода на клетку
# Пониженная точность fit() ТОЛЬКО для перебора сетки (не влияет на honest
# error -- она считается независимо на густой сетке против истинной кривой,
# см. true_curve_error); полные max_iter/max_densify_rounds были на два
# порядка медленнее без заметного изменения n_control_points или true_error
# (проверено вручную на нескольких клетках).
SPLINE_SEARCH_MAX_ITER = 15
SPLINE_SEARCH_MAX_DENSIFY_ROUNDS = 3

SIGMA_PILOT = [0.0, 0.1, 2.0]
TOL_PILOT = [0.2, 1.0, 5.0]
N_TRACKS_PILOT = 5
LENGTH_RANGE_PILOT_M = (900.0, 1100.0)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_MD = os.path.join(RESULTS_DIR, "step2.md")
SECTION1_HEADER = "## 1. Синтетика, метрика ошибки, честный подбор параметра"
SECTION2_HEADER = "## 2. Сжатие: карта отношения сплайн/DP+SED по (шум, tol)"


# ------------------------------------------------------------- geometry --

@dataclass
class Segment:
    kind: str  # "line" | "arc" | "clothoid"
    length: float
    k0: float = 0.0
    k1: float = 0.0


def _clothoid_xy(s, x0, y0, theta0, k0, a):
    """Позиция вдоль клотоиды (curvature(u) = k0 + a*u) на длине дуги s
    (может быть массивом), замкнутая форма через интегралы Френеля."""
    s = np.asarray(s, dtype=float)
    if abs(a) < 1e-12:
        return x0 + np.cos(theta0) * s, y0 + np.sin(theta0) * s
    v0 = k0 / a
    v1 = v0 + s
    theta_dd = theta0 - k0 * k0 / (2.0 * a)
    scale = np.sqrt(np.pi / abs(a))
    sign_a = np.sign(a)
    root = np.sqrt(abs(a) / np.pi)
    s0, c0 = fresnel(v0 * root)
    s1, c1 = fresnel(v1 * root)
    dC = c1 - c0
    dS = sign_a * (s1 - s0)
    ix = scale * (np.cos(theta_dd) * dC - np.sin(theta_dd) * dS)
    iy = scale * (np.sin(theta_dd) * dC + np.cos(theta_dd) * dS)
    return x0 + ix, y0 + iy


def _segment_xy(seg: Segment, s_local, x0: float, y0: float, theta0: float):
    if seg.kind == "line":
        return x0 + np.cos(theta0) * s_local, y0 + np.sin(theta0) * s_local
    if seg.kind == "arc":
        k = seg.k0
        theta = theta0 + k * s_local
        return x0 + (np.sin(theta) - np.sin(theta0)) / k, y0 - (np.cos(theta) - np.cos(theta0)) / k
    if seg.kind == "clothoid":
        a = (seg.k1 - seg.k0) / seg.length
        return _clothoid_xy(s_local, x0, y0, theta0, seg.k0, a)
    raise ValueError(seg.kind)


def _segment_end_state(seg: Segment, x0: float, y0: float, theta0: float):
    x1, y1 = _segment_xy(seg, np.array([seg.length]), x0, y0, theta0)
    if seg.kind == "line":
        theta1 = theta0
    elif seg.kind == "arc":
        theta1 = theta0 + seg.k0 * seg.length
    else:
        a = (seg.k1 - seg.k0) / seg.length
        theta1 = theta0 + seg.k0 * seg.length + a * seg.length ** 2 / 2.0
    return float(x1[0]), float(y1[0]), theta1


def _random_road_segments(rng: np.random.Generator, target_length_m: float) -> list[Segment]:
    """Прямые (50-400м) чередуются с поворотами: клотоида-вход (0->k),
    дуга постоянной кривизны, клотоида-выход (k->0). Кривизна и курс
    непрерывны на всех стыках по построению."""
    segments: list[Segment] = []
    total = 0.0
    while total < target_length_m:
        line_len = float(rng.uniform(50.0, 400.0))
        segments.append(Segment("line", line_len))
        total += line_len
        if total >= target_length_m:
            break
        radius = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(30.0, 150.0))
        k = 1.0 / radius
        clothoid_len = float(np.clip(abs(radius) * 0.4, 15.0, 80.0))
        angle = float(rng.uniform(np.pi / 6.0, 5.0 * np.pi / 6.0))
        arc_len = abs(radius) * angle
        segments.append(Segment("clothoid", clothoid_len, 0.0, k))
        segments.append(Segment("arc", arc_len, k, k))
        segments.append(Segment("clothoid", clothoid_len, k, 0.0))
        total += 2 * clothoid_len + arc_len
    return segments


def _build_path(segments: list[Segment]):
    starts = []
    x, y, theta = 0.0, 0.0, 0.0
    s_cursor = 0.0
    for seg in segments:
        starts.append((s_cursor, x, y, theta, seg))
        x, y, theta = _segment_end_state(seg, x, y, theta)
        s_cursor += seg.length
    return starts, s_cursor


def _path_xy(starts, total_length: float, s_query: np.ndarray) -> np.ndarray:
    s_query = np.clip(np.asarray(s_query, dtype=float), 0.0, total_length)
    boundaries = np.array([st[0] for st in starts] + [total_length])
    idx = np.clip(np.searchsorted(boundaries, s_query, side="right") - 1, 0, len(starts) - 1)
    out = np.empty((len(s_query), 2))
    for i in range(len(starts)):
        mask = idx == i
        if not mask.any():
            continue
        s0, x0, y0, theta0, seg = starts[i]
        s_local = np.clip(s_query[mask] - s0, 0.0, seg.length)
        xs, ys = _segment_xy(seg, s_local, x0, y0, theta0)
        out[mask, 0] = xs
        out[mask, 1] = ys
    return out


# ------------------------------------------------------------ synthetic --

def make_synthetic_track(
    seed: int,
    noise_std: float,
    length_range_m: tuple[float, float] = LENGTH_RANGE_M,
    speed_range_mps: tuple[float, float] = SPEED_RANGE_MPS,
):
    """Возвращает (шумный трек с шагом 1с, true_xy(t) -- векторизованная
    функция истинной (бесшумной) кривой, duration). Геометрия и скорость
    зависят только от `seed` (не от `noise_std`) -- одна и та же форма
    пути используется на всех уровнях шума; шум использует отдельный
    rng, производный от (seed, noise_std), чтобы не коррелировать между
    разными sigma одного трека."""
    geom_rng = np.random.default_rng(seed)
    target_length = float(geom_rng.uniform(*length_range_m))
    segments = _random_road_segments(geom_rng, target_length)
    starts, total_length = _build_path(segments)
    speed = float(geom_rng.uniform(*speed_range_mps))
    duration = total_length / speed

    def true_xy(t):
        t = np.asarray(t, dtype=float)
        return _path_xy(starts, total_length, speed * t)

    t = np.arange(0.0, duration, 1.0)
    if t[-1] != duration:
        t = np.concatenate([t, [duration]])
    xy_true = true_xy(t)
    if noise_std > 0.0:
        noise_rng = np.random.default_rng((seed, int(round(noise_std * 1000))))
        xy_obs = xy_true + noise_rng.normal(0.0, noise_std, xy_true.shape)
    else:
        xy_obs = xy_true.copy()

    zeros = np.zeros(len(t))
    track = Track(track_id=f"xover{seed}_s{noise_std}", lat=zeros, lon=zeros, t=t, xy=xy_obs)
    return track, true_xy, duration


# ---------------------------------------------------------------- error --

def true_curve_error(true_xy_dense: np.ndarray, recon_xy_dense: np.ndarray) -> float:
    return float(np.max(np.hypot(*(recon_xy_dense - true_xy_dense).T)))


def _dp_reconstruct(t_dense, t_kept, xy_kept):
    x = np.interp(t_dense, t_kept, xy_kept[:, 0])
    y = np.interp(t_dense, t_kept, xy_kept[:, 1])
    return np.column_stack([x, y])


def _spline_time_reconstruct(t_dense, sp):
    span = sp.t_max - sp.t_min
    u = (t_dense - sp.t_min) / span if span > 0 else np.zeros_like(t_dense)
    xs, ys = splev(np.clip(u, 0.0, 1.0), sp.tck)
    return np.column_stack([xs, ys])


def _spline_chord_reconstruct(t_dense, sp, track):
    u0 = _param_u(track.t, track.xy, "chord")
    u = np.interp(t_dense, track.t, u0)
    xs, ys = splev(np.clip(u, 0.0, 1.0), sp.tck)
    return np.column_stack([xs, ys])


# --------------------------------------------------------- build funcs --

def _build_dp(track, internal_tol, t_dense, true_xy_dense, sed):
    fn = simplify_sed_with_indices if sed else simplify_with_indices
    _, idx = fn(track, internal_tol)
    recon = _dp_reconstruct(t_dense, track.t[idx], track.xy[idx])
    err = true_curve_error(true_xy_dense, recon)
    n = len(idx)
    return n, n * 3 * 8, err


def _build_spline(track, internal_tol, t_dense, true_xy_dense, mode):
    sp = spline_fit(
        track,
        tol=internal_tol,
        parametrization=mode,
        max_iter=SPLINE_SEARCH_MAX_ITER,
        max_densify_rounds=SPLINE_SEARCH_MAX_DENSIFY_ROUNDS,
    )
    if mode == "time":
        recon = _spline_time_reconstruct(t_dense, sp)
    else:
        recon = _spline_chord_reconstruct(t_dense, sp, track)
    err = true_curve_error(true_xy_dense, recon)
    n_knots = len(sp.tck[0])
    n_coeffs = len(sp.tck[1][0])
    n = n_knots + 2 * n_coeffs
    nbytes = n * 8 + (16 if mode == "time" else 0)
    return n, nbytes, err


def search_min_params(build_fn, target_tol: float, internal_tol_grid: np.ndarray) -> dict:
    """internal_tol_grid -- по ВОЗРАСТАНИЮ (от самого генерозного/малого
    internal_tol к самому скупому/большому). Первая точка -- проверка
    достижимости (лучший случай метода); если она не проходит tol,
    остальные точки сетки не считаются."""
    best = None
    for i, itol in enumerate(internal_tol_grid):
        n, nbytes, err = build_fn(float(itol))
        if i == 0 and err > target_tol:
            return {"reachable": False}
        if err <= target_tol and (best is None or n < best["n"]):
            best = {"reachable": True, "n": n, "bytes": nbytes, "true_error": err, "internal_tol": float(itol)}
    return best if best is not None else {"reachable": False}


METHODS = ("dp", "dp_sed", "spline_time", "spline_chord")


def evaluate_cell(track, true_xy_dense, t_dense, target_tol: float) -> dict:
    grid = target_tol * np.logspace(-2, 1, N_INTERNAL_GRID)
    results = {}
    results["dp"] = search_min_params(
        lambda itol: _build_dp(track, itol, t_dense, true_xy_dense, sed=False), target_tol, grid
    )
    results["dp_sed"] = search_min_params(
        lambda itol: _build_dp(track, itol, t_dense, true_xy_dense, sed=True), target_tol, grid
    )
    results["spline_time"] = search_min_params(
        lambda itol: _build_spline(track, itol, t_dense, true_xy_dense, "time"), target_tol, grid
    )
    results["spline_chord"] = search_min_params(
        lambda itol: _build_spline(track, itol, t_dense, true_xy_dense, "chord"), target_tol, grid
    )
    return results


# ------------------------------------------------------------------ run --

def run_grid(n_tracks, sigma_list, tol_list, length_range_m, seed=SEED, verbose=True):
    rng_master = np.random.default_rng(seed)
    geom_seeds = [int(s) for s in rng_master.integers(0, 1_000_000, size=n_tracks)]

    # cells[(sigma, tol)][method] -> list of per-track results (or None if unreachable)
    cells = {(s, t): {m: [] for m in METHODS} for s in sigma_list for t in tol_list}

    for gseed in geom_seeds:
        for sigma in sigma_list:
            track, true_xy, duration = make_synthetic_track(gseed, sigma, length_range_m=length_range_m)
            dt_dense = 1.0 / PTS_PER_SEC
            t_dense = np.arange(0.0, duration, dt_dense)
            if len(t_dense) == 0 or t_dense[-1] != duration:
                t_dense = np.concatenate([t_dense, [duration]])
            true_xy_dense = true_xy(t_dense)
            for tol in tol_list:
                res = evaluate_cell(track, true_xy_dense, t_dense, tol)
                for m in METHODS:
                    cells[(sigma, tol)][m].append(res[m])
        if verbose:
            print(f"трек seed={gseed} готов")
    return cells


def _summarize_method(entries: list[dict]) -> str:
    reachable = [e for e in entries if e["reachable"]]
    n_total = len(entries)
    n_reach = len(reachable)
    if n_reach == 0:
        return "недостижимо"
    mean_bytes = np.mean([e["bytes"] for e in reachable])
    mean_n = np.mean([e["n"] for e in reachable])
    if n_reach == n_total:
        return f"{mean_bytes:.0f}Б (n={mean_n:.1f})"
    return f"{mean_bytes:.0f}Б (n={mean_n:.1f}), {n_reach}/{n_total} достижимо"


def format_section1() -> str:
    lines = [
        f"{SECTION1_HEADER}\n",
        (
            "Синтетика: дорожная модель из прямых (50-400м) и поворотов "
            "(радиус 30-150м, угол 30-150°), каждый поворот -- клотоида-вход "
            "(кривизна 0->k, спираль Эйлера через интегралы Френеля), дуга "
            "постоянной кривизны, клотоида-выход (k->0) -- курс и кривизна "
            f"непрерывны на всех стыках. Длина трека {LENGTH_RANGE_M[0]/1000:.0f}-"
            f"{LENGTH_RANGE_M[1]/1000:.0f} км, скорость постоянна на трек "
            f"({SPEED_RANGE_MPS[0]:.0f}-{SPEED_RANGE_MPS[1]:.0f} м/с), шаг "
            "наблюдений 1с, seed=42.\n"
        ),
        (
            "Ошибка ВСЕХ методов -- max отклонение реконструкции от ИСТИННОЙ "
            "(бесшумной) кривой на густой сетке (>=10 точек/с). DP/DP+SED "
            "реконструируются кусочно-линейно ПО ВРЕМЕНИ между сохранёнными "
            "вершинами; сплайн (time) -- прямая связь u=(t-t_min)/(t_max-t_min); "
            "сплайн (chord) -- точной связи t<->u нет, используется линейная "
            "интерполяция по узлам шумного трека (приближение, см. код).\n"
        ),
        (
            "Честный подбор: целевой tol НЕ передаётся методам напрямую. Для "
            "каждого метода отдельно перебирается его собственный внутренний "
            f"параметр (на зашумлённых точках) по {N_INTERNAL_GRID} "
            "логарифмически распределённым значениям относительно целевого "
            "tol; отчитывается минимальное число параметров среди прошедших "
            "честную проверку (ошибка <= tol). Если даже самая генерозная "
            "точка перебора не проходит tol (типично sigma >= tol) -- клетка "
            "помечается недостижимой без дальнейшего перебора.\n"
        ),
    ]
    return "\n".join(lines) + "\n"


def _pair_summary(sed_entries: list[dict], spl_entries: list[dict], n_tracks: int):
    """Сравнивает DP+SED и сплайн ПОПАРНО, по одним и тем же трекам --
    честно различает три исхода: оба достижимы (ratio по общему
    подмножеству), достижим только один метод (это тоже "победа" --
    другой метод не может держать tol НИ ПРИ КАКОМ числе параметров), и
    недостижимо оба. Возвращает (текст_клетки, ratio_или_None,
    "spline_only"|"sed_only"|"both"|"neither")."""
    both = [(s, p) for s, p in zip(sed_entries, spl_entries) if s["reachable"] and p["reachable"]]
    spline_only = [p for s, p in zip(sed_entries, spl_entries) if p["reachable"] and not s["reachable"]]
    sed_only = [s for s, p in zip(sed_entries, spl_entries) if s["reachable"] and not p["reachable"]]

    if not both and not spline_only and not sed_only:
        return "недостижимо (оба)", None, "neither"

    parts = []
    ratio = None
    kind = "both"
    if both:
        sed_bytes = np.mean([s["bytes"] for s, _ in both])
        spl_bytes = np.mean([p["bytes"] for _, p in both])
        ratio = spl_bytes / sed_bytes
        suffix = f" ({len(both)}/{n_tracks})" if len(both) != n_tracks else ""
        parts.append(f"**{ratio:.2f}x**{suffix}")
    if spline_only:
        kind = "spline_only" if not both else kind
        parts.append(f"сплайн-только: {len(spline_only)}/{n_tracks} (DP+SED недостижим ни при каком n)")
    if sed_only:
        kind = "sed_only" if not both and not spline_only else kind
        parts.append(f"DP+SED-только: {len(sed_only)}/{n_tracks} (сплайн недостижим ни при каком n)")
    return "; ".join(parts), ratio, kind


def format_section2(cells, sigma_list, tol_list, n_tracks) -> str:
    lines = [f"{SECTION2_HEADER}\n", f"{n_tracks} треков, seed={SEED}.\n"]
    header = "| sigma \\ tol | " + " | ".join(f"{t:g}" for t in tol_list) + " |"

    for method_key, title in (
        ("dp", "DP"),
        ("dp_sed", "DP+SED"),
        ("spline_time", "сплайн (time)"),
        ("spline_chord", "сплайн (chord) -- нижняя граница, без учёта восстановления времени"),
    ):
        lines.append(f"### {title}: байты (n параметров), доля достижимых треков")
        lines.append(header)
        lines.append("|" + "---|" * (len(tol_list) + 1))
        for sigma in sigma_list:
            row = [f"{sigma:g}"] + [_summarize_method(cells[(sigma, tol)][method_key]) for tol in tol_list]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append(
        "### Сравнение попарно по трекам: сплайн (time) vs DP+SED "
        "(отношение байт по общему подмножеству достижимых треков; "
        "\"только один метод\" -- другой метод не укладывается в tol "
        "НИ ПРИ КАКОМ числе параметров на этой клетке)"
    )
    lines.append(header)
    lines.append("|" + "---|" * (len(tol_list) + 1))
    cell_kind = {}
    ratio_map = {}
    for sigma in sigma_list:
        row = [f"{sigma:g}"]
        for tol in tol_list:
            text, ratio, kind = _pair_summary(cells[(sigma, tol)]["dp_sed"], cells[(sigma, tol)]["spline_time"], n_tracks)
            row.append(text)
            ratio_map[(sigma, tol)] = ratio
            cell_kind[(sigma, tol)] = kind
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    crossovers = []
    for sigma in sigma_list:
        wins = [
            tol
            for tol in tol_list
            if cell_kind.get((sigma, tol)) == "spline_only"
            or (ratio_map.get((sigma, tol)) is not None and ratio_map[(sigma, tol)] < 1.0)
        ]
        if wins:
            crossovers.append(f"sigma={sigma:g}: сплайн выигрывает (или единственный достижим) при tol>={min(wins):g}")
    if crossovers:
        lines.append("Точка(и) пересечения: " + "; ".join(crossovers) + ".\n")
    else:
        lines.append(
            "Пересечения в этой сетке НЕТ: на всех клетках, где сравнение "
            "возможно, DP+SED компактнее сплайна или оба недостижимы "
            "(отрицательный результат, не подгонялось).\n"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="бюджетный прогон (5 треков, малая сетка), без записи в .md")
    parser.add_argument("--n-tracks", type=int, default=None, help="переопределить N_TRACKS полного прогона")
    args = parser.parse_args()

    if args.pilot:
        print(
            f"ПИЛОТ: {N_TRACKS_PILOT} треков, длина ~1км, "
            f"sigma={SIGMA_PILOT}, tol={TOL_PILOT}"
        )
        t0 = time.time()
        cells = run_grid(N_TRACKS_PILOT, SIGMA_PILOT, TOL_PILOT, LENGTH_RANGE_PILOT_M)
        elapsed = time.time() - t0
        n_cells_pilot = N_TRACKS_PILOT * len(SIGMA_PILOT) * len(TOL_PILOT)
        n_cells_full = N_TRACKS * len(SIGMA_LIST) * len(TOL_LIST)
        extrapolated = elapsed * (n_cells_full / n_cells_pilot)
        print(f"\nпилот занял {elapsed:.1f}с ({n_cells_pilot} (трек x sigma x tol) комбинаций)")
        print(
            f"экстраполяция на полную сетку ({n_cells_full} комбинаций, x"
            f"{n_cells_full / n_cells_pilot:.1f}): ~{extrapolated:.0f}с (~{extrapolated / 60:.1f} мин)"
        )
        print(format_section2(cells, SIGMA_PILOT, TOL_PILOT, N_TRACKS_PILOT))
        return

    n_tracks = args.n_tracks if args.n_tracks is not None else N_TRACKS
    print(f"ПОЛНЫЙ ПРОГОН: {n_tracks} треков, sigma={SIGMA_LIST}, tol={TOL_LIST}")
    t0 = time.time()
    cells = run_grid(n_tracks, SIGMA_LIST, TOL_LIST, LENGTH_RANGE_M)
    elapsed = time.time() - t0
    print(f"\nполный прогон занял {elapsed:.1f}с ({elapsed/60:.1f} мин)")

    body1 = format_section1()
    body2 = format_section2(cells, SIGMA_LIST, TOL_LIST, n_tracks)
    if n_tracks != N_TRACKS:
        body2 += (
            f"\n**Ограничение бюджета времени:** N_TRACKS уменьшен с {N_TRACKS} до "
            f"{n_tracks} по результатам пилотного прогона (см. вывод `--pilot`), "
            "чтобы полный прогон укладывался в разумное время. Полный прогон занял "
            f"{elapsed:.1f}с ({elapsed/60:.1f} мин) на {n_tracks} треках.\n"
        )
    upsert_section(OUT_MD, SECTION1_HEADER, body1)
    upsert_section(OUT_MD, SECTION2_HEADER, body2)
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
