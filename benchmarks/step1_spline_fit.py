"""Step 1, item 2: honest spline fitting -- dense-grid check, comparing
time and chord-length parametrization.

Run: venv/bin/python benchmarks/step1_spline_fit.py
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
SECTION_HEADER = "## 2. Honest spline fitting"


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
        f"ACCEPTANCE THRESHOLD (>= {PASS_THRESHOLD:.0%} of tracks with dense error <= tol) "
        f"{'REACHED' if passed else 'NOT REACHED'}: "
        f"{res_time['n_pass']}/{res_time['n_tracks']} = {res_time['pass_rate']:.2%} "
        "(time parametrization)."
    )
    lines = [
        f"{SECTION_HEADER}\n",
        (
            f"Check: >= 10 dense-grid points per interval between "
            f"neighboring points of the CLEANED track (after item 1), error = "
            f"point-to-segment distance to the straight segment between them. "
            f"If s=0 (interpolation) violates tol between some pair of points, "
            f"a synthetic knot is adaptively added at the location of maximum "
            f"deviation (up to 8 rounds); then `s` is grown for compactness, "
            f"keeping the dense error <= tol.\n"
        ),
        f"**{verdict}**\n",
        "| Parametrization | Pass rate | Median err, m | p99 err, m | Max err, m | mean control points | tracks with synthetic knots | bisect converged | time, s |",
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
            f"Mean number of points in a cleaned track: {mean_raw_n:.1f}. Both "
            "parametrizations reach the same pass rate; chord-length "
            "parametrization gives a noticeably more compact representation "
            f"({res_chord['mean_ncp']:.1f} vs {res_time['mean_ncp']:.1f} "
            "control points on average), but `derivatives()` (velocity/"
            "acceleration) is only implemented for time parametrization -- "
            "the nonlinear u<->t relationship for chord would require a "
            "separate inversion t(u), not needed beyond this comparison. For "
            "kinematics (item 4) and the rest of the step1 benchmarks, time "
            "parametrization is used (the default in fit()).\n"
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    tracks, _clean_stats = load_clean_tracks(n=N_TRACKS, seed=SEED)
    mean_raw_n = float(np.mean([len(tr.t) for tr in tracks]))
    print(f"cleaned tracks: {len(tracks)}, mean number of points: {mean_raw_n:.1f}")

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
            "\nTHRESHOLD NOT REACHED -- see TODO.md before moving on to item 4."
        )


if __name__ == "__main__":
    main()
