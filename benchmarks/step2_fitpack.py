"""Step 2, п.3: неоптимальность FITPACK -- сколько контрольных точек даёт
`splprep` (через `traj.spline.fit`) против минимального числа РАВНОМЕРНЫХ
узлов, при котором `make_lsq_spline` всё ещё держит tol (честная густая
ошибка, `traj.spline.dense_max_error`).

5 треков из той же синтетики, что и step2_crossover.py (общий генератор
`make_synthetic_track`, те же первые 5 seed из мастер-seed=42),
sigma=0.1, tol=1.0 -- типичная средняя клетка сетки step2 (не самая
простая и не самая жёсткая).

Запуск: venv/bin/python benchmarks/step2_fitpack.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.interpolate import make_lsq_spline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from traj.spline import DEGREE, dense_max_error  # noqa: E402
from traj.spline import _param_u  # noqa: E402
from traj.spline import fit as spline_fit  # noqa: E402
from _report_utils import upsert_section  # noqa: E402
from step2_crossover import SEED, N_TRACKS, make_synthetic_track  # noqa: E402

N_FITPACK_TRACKS = 5
SIGMA = 0.1
TOL = 1.0
MAX_N_INTERIOR = 300
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_MD = os.path.join(RESULTS_DIR, "step2.md")
SECTION_HEADER = "## 3. Неоптимальность FITPACK: splprep vs минимальные равномерные узлы"


def _uniform_clamped_knots(u_min: float, u_max: float, n_interior: int, k: int) -> np.ndarray:
    interior = np.linspace(u_min, u_max, n_interior + 2)[1:-1] if n_interior > 0 else np.array([])
    return np.concatenate([[u_min] * (k + 1), interior, [u_max] * (k + 1)])


def min_uniform_knots(t: np.ndarray, xy: np.ndarray, tol: float, mode: str = "time", k: int = DEGREE):
    """Растим число внутренних равномерных узлов n_int = 0, 1, 2, ...
    пока честная густая ошибка (dense_max_error) не станет <= tol.
    Возвращает (n_control_points, n_interior, dense_error) или
    (None, None, None), если не уложились за MAX_N_INTERIOR."""
    u = _param_u(t, xy, mode)
    u_min, u_max = float(u.min()), float(u.max())
    for n_int in range(0, MAX_N_INTERIOR + 1):
        knots = _uniform_clamped_knots(u_min, u_max, n_int, k)
        if len(knots) - k - 1 > len(u):
            break  # больше параметров, чем точек -- дальше только хуже обусловлено
        try:
            bx = make_lsq_spline(u, xy[:, 0], knots, k=k)
            by = make_lsq_spline(u, xy[:, 1], knots, k=k)
        except Exception:
            continue
        tck = (bx.t, [bx.c, by.c], k)
        err = dense_max_error(t, xy, tck, mode=mode)
        if err <= tol:
            return n_int + k + 1, n_int, err
    return None, None, None


def main() -> None:
    rng_master = np.random.default_rng(SEED)
    geom_seeds = [int(s) for s in rng_master.integers(0, 1_000_000, size=N_TRACKS)][:N_FITPACK_TRACKS]

    rows = []
    for gseed in geom_seeds:
        track, _true_xy, _duration = make_synthetic_track(gseed, SIGMA)
        sp = spline_fit(track, tol=TOL, parametrization="time")
        n_cp_lsq, n_int, err_lsq = min_uniform_knots(track.t, track.xy, TOL, mode="time")
        rows.append(
            {
                "seed": gseed,
                "n_points": len(track.t),
                "n_cp_splprep": sp.n_control_points,
                "splprep_err": sp.max_error,
                "n_cp_lsq": n_cp_lsq,
                "n_interior_lsq": n_int,
                "lsq_err": err_lsq,
            }
        )
        print(
            f"seed={gseed}: n_points={len(track.t)} splprep n_cp={sp.n_control_points} "
            f"(err={sp.max_error:.3f}) | lsq minimal n_cp={n_cp_lsq} (n_int={n_int}, err={err_lsq})"
        )

    lines = [
        f"{SECTION_HEADER}\n",
        (
            f"{N_FITPACK_TRACKS} треков из синтетики step2_crossover.py (те же "
            f"первые {N_FITPACK_TRACKS} seed из мастер-seed={SEED}), sigma={SIGMA:g}м, "
            f"tol={TOL:g}м, параметризация временем. `splprep` -- через "
            "`traj.spline.fit` (честный густой контроль, адаптивная densify + "
            "рост `s`). `make_lsq_spline` -- РАВНОМЕРНЫЕ узлы, число внутренних "
            "узлов растится с 0 до первого прохождения той же честной густой "
            "проверки (`traj.spline.dense_max_error`).\n"
        ),
        "| seed | n точек | n control points (splprep) | n control points (lsq, равномерные) | отношение |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        ratio = (
            f"{r['n_cp_splprep'] / r['n_cp_lsq']:.2f}x" if r["n_cp_lsq"] else "н/д (не уложились)"
        )
        lines.append(
            f"| {r['seed']} | {r['n_points']} | {r['n_cp_splprep']} | "
            f"{r['n_cp_lsq'] if r['n_cp_lsq'] else 'н/д'} | {ratio} |"
        )
    lines.append("")

    valid = [r for r in rows if r["n_cp_lsq"]]
    if valid:
        mean_splprep = np.mean([r["n_cp_splprep"] for r in valid])
        mean_lsq = np.mean([r["n_cp_lsq"] for r in valid])
        lines.append(
            f"В среднем по {len(valid)}/{N_FITPACK_TRACKS} трекам: splprep "
            f"{mean_splprep:.1f} контрольных точек, минимальные равномерные "
            f"узлы -- {mean_lsq:.1f} ({mean_splprep / mean_lsq:.2f}x). "
            + (
                "splprep использует заметно больше параметров, чем нужно для "
                "той же гарантии -- часть проигрыша сплайна DP+SED в step1/step2 "
                "может быть артефактом FITPACK, а не фундаментальным свойством "
                "кубических B-сплайнов."
                if mean_splprep > mean_lsq * 1.1
                else "splprep не хуже равномерных узлов на этих треках -- "
                "проигрыш сплайна DP+SED не объясняется неоптимальностью FITPACK."
            )
            + "\n"
        )

    body = "\n".join(lines) + "\n"
    upsert_section(OUT_MD, SECTION_HEADER, body)
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
