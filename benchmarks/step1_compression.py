"""Step 1, п.5: сжатие (гипотеза A) — где исправленный сплайн может
выигрывать у DP по байтам, при разных tol и кодировках.

DP (пространственный, только расстояние до отрезка) vs DP+SED
(time-aware, synchronized Euclidean distance) vs исправленный сплайн;
байты ВКЛЮЧАЯ метки времени; tol = 2, 5, 10, 20, 50 м; float64 и int32
(позиция в см, время в целых секундах).

Запуск: venv/bin/python benchmarks/step1_compression.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from traj.clean import load_clean_tracks  # noqa: E402
from traj.simplify import simplify_sed_with_indices, simplify_with_indices  # noqa: E402
from traj.spline import fit as spline_fit  # noqa: E402
from _report_utils import upsert_section  # noqa: E402

SEED = 42
N_TRACKS = 200
N_SUBSET = 60  # подвыборка очищенных треков (сплайн-фиттинг на 5 tol дорог)
TOL_LIST = [2.0, 5.0, 10.0, 20.0, 50.0]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_MD = os.path.join(RESULTS_DIR, "step1.md")
SECTION_HEADER = "## 5. Сжатие (гипотеза A): где сплайн выигрывает у DP по байтам"


def _dp_bytes(n_vertices: int, encoding: str) -> int:
    """x, y, t на вершину."""
    return n_vertices * 3 * (8 if encoding == "float64" else 4)


def _raw_bytes(n_points: int, encoding: str) -> int:
    return n_points * 3 * (8 if encoding == "float64" else 4)


def _spline_bytes(sp, encoding: str) -> int:
    n_knots = len(sp.tck[0])
    n_coeffs = len(sp.tck[1][0])
    itemsize = 8 if encoding == "float64" else 4
    # + t_min, t_max -- без них узлы (доля [0,1]) нельзя перевести обратно в реальное время
    return (n_knots + 2 * n_coeffs) * itemsize + 2 * itemsize


def main() -> None:
    all_tracks, _clean_stats = load_clean_tracks(n=N_TRACKS, seed=SEED)
    rng = np.random.default_rng(SEED)
    subset_idx = rng.choice(len(all_tracks), size=min(N_SUBSET, len(all_tracks)), replace=False)
    tracks = [all_tracks[i] for i in subset_idx]
    print(f"треков в подвыборке: {len(tracks)} (из {len(all_tracks)} очищенных)")

    rows = []
    for tol in TOL_LIST:
        t0 = time.time()
        n_raw, n_dp, n_sed, n_knots, n_cp = [], [], [], [], []
        n_spline_fail = 0
        for tr in tracks:
            n_raw.append(len(tr.t))

            _dp_xy, dp_idx = simplify_with_indices(tr, tol)
            n_dp.append(len(dp_idx))

            _sed_xy, sed_idx = simplify_sed_with_indices(tr, tol)
            n_sed.append(len(sed_idx))

            sp = spline_fit(tr, tol=tol, parametrization="time")
            if not sp.converged and sp.max_error > tol:
                n_spline_fail += 1
            n_knots.append(len(sp.tck[0]))
            n_cp.append(sp.n_control_points)

        elapsed = time.time() - t0

        def mean_bytes(fn, arr, encoding):
            return float(np.mean([fn(x, encoding) for x in arr]))

        spline_bytes_f64 = float(
            np.mean([(k + 2 * c) * 8 + 16 for k, c in zip(n_knots, n_cp)])
        )
        spline_bytes_i32 = float(
            np.mean([(k + 2 * c) * 4 + 8 for k, c in zip(n_knots, n_cp)])
        )

        row = {
            "tol": tol,
            "elapsed": elapsed,
            "n_spline_fail": n_spline_fail,
            "raw_f64": mean_bytes(_raw_bytes, n_raw, "float64"),
            "raw_i32": mean_bytes(_raw_bytes, n_raw, "int32"),
            "dp_f64": mean_bytes(_dp_bytes, n_dp, "float64"),
            "dp_i32": mean_bytes(_dp_bytes, n_dp, "int32"),
            "sed_f64": mean_bytes(_dp_bytes, n_sed, "float64"),
            "sed_i32": mean_bytes(_dp_bytes, n_sed, "int32"),
            "spline_f64": spline_bytes_f64,
            "spline_i32": spline_bytes_i32,
            "mean_n_dp": float(np.mean(n_dp)),
            "mean_n_sed": float(np.mean(n_sed)),
            "mean_n_cp": float(np.mean(n_cp)),
        }
        rows.append(row)
        print(
            f"tol={tol}: dp_f64={row['dp_f64']:.0f}B sed_f64={row['sed_f64']:.0f}B "
            f"spline_f64={row['spline_f64']:.0f}B (n_dp={row['mean_n_dp']:.1f} "
            f"n_sed={row['mean_n_sed']:.1f} n_cp={row['mean_n_cp']:.1f}) "
            f"spline_dense_fail={n_spline_fail}/{len(tracks)} elapsed={elapsed:.1f}s"
        )

    lines = [
        f"{SECTION_HEADER}\n",
        (
            f"{len(tracks)} очищенных треков (подвыборка из {len(all_tracks)}, seed={SEED}), "
            "float64 = позиция+время float64 (24 байт/точку); int32 = позиция в "
            "см + время в целых секундах, int32 (12 байт/точку). DP — "
            "пространственный (perpendicular distance), DP+SED — time-aware "
            "(synchronized Euclidean distance, см. src/traj/simplify.py). "
            "Байты сплайна включают t_min/t_max — только так узлы (доля [0,1]) "
            "переводятся обратно в реальное время; отдельных меток времени на "
            "вершину сплайну не нужно.\n"
        ),
        "| tol, м | raw, Б (f64/i32) | DP, Б (f64/i32) | DP+SED, Б (f64/i32) | сплайн, Б (f64/i32) | n DP / n SED / n control points |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['tol']:.0f} | {row['raw_f64']:.0f} / {row['raw_i32']:.0f} | "
            f"{row['dp_f64']:.0f} / {row['dp_i32']:.0f} | "
            f"{row['sed_f64']:.0f} / {row['sed_i32']:.0f} | "
            f"{row['spline_f64']:.0f} / {row['spline_i32']:.0f} | "
            f"{row['mean_n_dp']:.1f} / {row['mean_n_sed']:.1f} / {row['mean_n_cp']:.1f} |"
        )
    lines.append("")

    n_fail_total = sum(r["n_spline_fail"] for r in rows)
    best_by_tol = []
    for row in rows:
        candidates = {"DP": row["dp_f64"], "DP+SED": row["sed_f64"], "сплайн": row["spline_f64"]}
        best = min(candidates, key=candidates.get)
        best_by_tol.append(f"tol={row['tol']:.0f}: {best}")
    lines.append(
        "Наименьший размер (float64) по tol: " + "; ".join(best_by_tol) + ".\n"
    )
    if n_fail_total:
        lines.append(
            f"ВНИМАНИЕ: у {n_fail_total} (трек×tol) фиттингов густая ошибка "
            "превысила tol (за пределами гарантии из п.2 при этих значениях "
            "tol) — не подгонялось, см. числа выше.\n"
        )

    body = "\n".join(lines) + "\n"
    upsert_section(OUT_MD, SECTION_HEADER, body)
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
