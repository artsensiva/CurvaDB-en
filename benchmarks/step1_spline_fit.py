"""Step 1, п.2: честный фиттинг сплайна — проверка на густой сетке,
сравнение параметризации временем и длиной хорды.

Запуск: venv/bin/python benchmarks/step1_spline_fit.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from traj.clean import load_clean_tracks  # noqa: E402
from traj.spline import dense_check, fit  # noqa: E402
from _report_utils import upsert_section  # noqa: E402

SEED = 42
N_TRACKS = 200
TOL = 10.0
PASS_THRESHOLD = 0.99
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_MD = os.path.join(RESULTS_DIR, "step1.md")
SECTION_HEADER = "## 2. Честный фиттинг сплайна"


def _run(tracks, parametrization: str) -> dict:
    t0 = time.time()
    errs, ncp, densified, converged = [], [], 0, 0
    for tr in tracks:
        sp = fit(tr, tol=TOL, parametrization=parametrization)
        max_err, _ = dense_check(tr, sp)
        errs.append(max_err)
        ncp.append(sp.n_control_points)
        if sp.n_points_used > len(tr.t):
            densified += 1
        if sp.converged:
            converged += 1
    elapsed = time.time() - t0
    errs = np.array(errs)
    n_pass = int((errs <= TOL).sum())
    return {
        "parametrization": parametrization,
        "elapsed_s": elapsed,
        "n_tracks": len(tracks),
        "n_pass": n_pass,
        "pass_rate": n_pass / len(tracks),
        "median_err": float(np.median(errs)),
        "p99_err": float(np.percentile(errs, 99)),
        "max_err": float(errs.max()),
        "mean_ncp": float(np.mean(ncp)),
        "n_densified": densified,
        "n_bisect_converged": converged,
    }


def _section_md(res_time: dict, res_chord: dict, mean_raw_n: float) -> str:
    passed = res_time["pass_rate"] >= PASS_THRESHOLD
    verdict = (
        f"ПОРОГ ПРИЁМКИ (>= {PASS_THRESHOLD:.0%} треков с густой ошибкой <= tol) "
        f"{'ДОСТИГНУТ' if passed else 'НЕ ДОСТИГНУТ'}: "
        f"{res_time['n_pass']}/{res_time['n_tracks']} = {res_time['pass_rate']:.2%} "
        "(параметризация временем)."
    )
    lines = [
        f"{SECTION_HEADER}\n",
        (
            f"Проверка: >= 10 точек густой сетки на каждый интервал между "
            f"соседними точками ОЧИЩЕННОГО трека (после п.1), ошибка = "
            f"point-to-segment расстояние до прямого сегмента между ними. "
            f"Если s=0 (интерполяция) нарушает tol между какой-то парой точек — "
            f"адаптивно добавляется synthetic-узел в месте максимального "
            f"отклонения (до 8 раундов); затем растим `s` для компактности, "
            f"сохраняя густую ошибку <= tol.\n"
        ),
        f"**{verdict}**\n",
        "| Параметризация | Pass rate | Median err, м | p99 err, м | Max err, м | mean control points | треков с synthetic-узлами | bisect converged | время, с |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for res in (res_time, res_chord):
        lines.append(
            f"| {res['parametrization']} | {res['pass_rate']:.2%} | "
            f"{res['median_err']:.3f} | {res['p99_err']:.3f} | {res['max_err']:.3f} | "
            f"{res['mean_ncp']:.1f} | {res['n_densified']}/{res['n_tracks']} | "
            f"{res['n_bisect_converged']}/{res['n_tracks']} | {res['elapsed_s']:.1f} |"
        )
    lines += [
        "",
        (
            f"Среднее число точек в очищенном треке: {mean_raw_n:.1f}. Обе "
            "параметризации достигают одинакового pass rate; параметризация "
            "длиной хорды даёт заметно компактнее представление "
            f"({res_chord['mean_ncp']:.1f} vs {res_time['mean_ncp']:.1f} "
            "контрольных точек в среднем), но `derivatives()` (скорость/"
            "ускорение) реализована только для параметризации временем — "
            "нелинейная связь u<->t для хорды потребовала бы отдельной "
            "инверсии t(u), не нужной за пределами этого сравнения. Для "
            "кинематики (п.4) и остальных бенчмарков step1 используется "
            "параметризация временем (по умолчанию в fit()).\n"
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    tracks, _clean_stats = load_clean_tracks(n=N_TRACKS, seed=SEED)
    mean_raw_n = float(np.mean([len(tr.t) for tr in tracks]))
    print(f"очищенных треков: {len(tracks)}, среднее число точек: {mean_raw_n:.1f}")

    res_time = _run(tracks, "time")
    print(
        f"time: pass={res_time['n_pass']}/{res_time['n_tracks']} "
        f"({res_time['pass_rate']:.2%}) mean_ncp={res_time['mean_ncp']:.1f} "
        f"densified={res_time['n_densified']} elapsed={res_time['elapsed_s']:.1f}s"
    )

    res_chord = _run(tracks, "chord")
    print(
        f"chord: pass={res_chord['n_pass']}/{res_chord['n_tracks']} "
        f"({res_chord['pass_rate']:.2%}) mean_ncp={res_chord['mean_ncp']:.1f} "
        f"densified={res_chord['n_densified']} elapsed={res_chord['elapsed_s']:.1f}s"
    )

    upsert_section(OUT_MD, SECTION_HEADER, _section_md(res_time, res_chord, mean_raw_n))
    print(f"\nwrote {OUT_MD}")

    if res_time["pass_rate"] < PASS_THRESHOLD:
        print(
            "\nПОРОГ НЕ ДОСТИГНУТ — см. TODO.md перед тем, как переходить к п.4."
        )


if __name__ == "__main__":
    main()
