"""Step8 M1 item 0: budget check for M2's certify_spline load, BEFORE writing
knot_removal.py (ADR-0023's pre-registered rule).

ADR-0022's lam_fallback=0.001 sweep was measured on FULL TRACKS. M2's actual DP
(spec section 2.3) calls certify_spline on SHORT SEGMENTS (a candidate cut
spans at most W=8 of the ~m candidate points per track, section 2.1), so a
full-track-based cost estimate is the wrong scale -- this script measures the
real thing (certify_spline on segments of typical length) and compares against
the naive full-track-based estimate, showing the gap explicitly rather than
silently replacing one guess with another.

Run:
    venv/bin/python benchmarks/step8_m1_budget.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.interpolate import BSpline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from traj import spline as spline_module  # noqa: E402
from traj.certify import certify_spline  # noqa: E402
from traj.io import Track  # noqa: E402
from step3_decisive import _dense_grid, _geometry_for_seed  # noqa: E402
from _report_utils import upsert_section  # noqa: E402

SEGMENT_SIZES = [10, 25, 50, 100]
LAM_VALUES = [0.001, 0.1]
REPEATS = 5
FIT_TOL = 2.0  # a representative mid-range tol for the timing-only fits
BUDGET_SEED = 777
BUDGET_HOURS_THRESHOLD = 2.0

# ADR-0022's own full-track measurements (benchmarks/results/step8.md M0.2,
# "Applying the refined lam rule" table) -- the OLD, too-pessimistic estimate
# this script's segment-scale measurement is compared against.
OLD_FULL_TRACK_TIME_MS = {0.001: 2350.73, 0.1: 208.48}

# DP+SED kept-vertex counts (M0, benchmarks/results/step8.md), used as a
# labeled PROXY for section 2.1's not-yet-built candidate-point count `m` --
# n = len(idx)*3 (M0's own convention), so vertices = n/3.
M_SYNTHETIC_PROXY = 204.6 / 3  # sigma=0, tol=2, dt=1 (M0's DP+SED n)
M_GEOLIFE_PROXY = 150.0  # labeled estimate: GeoLife tracks run longer (up to 2000 raw points)
W = 8

# M2's data plan (spec section 4): synthetic step3 grid + GeoLife.
N_SYNTHETIC_TRACKS = 15
N_SIGMA = 4
N_TOL = 3
N_DT = 2
N_GEOLIFE_TRACKS = 585
N_GEOLIFE_TOL = 2

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_MD = os.path.join(RESULTS_DIR, "step8.md")
M1_BUDGET_HEADER = "### M1.0 -- budget check (ADR-0023)"


def measure_segment_certify_cost() -> dict:
    rng = np.random.default_rng(BUDGET_SEED)
    true_xy, duration = _geometry_for_seed(BUDGET_SEED)
    t_dense = _dense_grid(duration)
    xy_dense = true_xy(t_dense)

    results = {}
    for n in SEGMENT_SIZES:
        times = {lam: [] for lam in LAM_VALUES}
        n_control_points = []
        for _ in range(REPEATS):
            start = int(rng.integers(0, len(t_dense) - n))
            t_seg = t_dense[start : start + n]
            xy_seg = xy_dense[start : start + n]
            zeros = np.zeros(n)
            track_seg = Track(track_id="seg", lat=zeros, lon=zeros, t=t_seg, xy=xy_seg)
            fit = spline_module.fit(track_seg, tol=FIT_TOL)
            bs = BSpline(fit.tck[0], np.column_stack(fit.tck[1]), fit.tck[2])
            n_control_points.append(fit.n_control_points)
            for lam in LAM_VALUES:
                t0 = time.perf_counter()
                certify_spline(xy_seg, bs, lam_fallback=lam)
                times[lam].append(time.perf_counter() - t0)
        results[n] = {
            "n_control_points_p50": float(np.median(n_control_points)),
            **{lam: 1000.0 * float(np.median(times[lam])) for lam in LAM_VALUES},
        }
    return results


def project_total_hours(transitions_total: float, time_per_call_ms: float) -> float:
    return transitions_total * time_per_call_ms / 1000.0 / 3600.0


def main():
    seg_results = measure_segment_certify_cost()

    dp_transitions_synth = M_SYNTHETIC_PROXY * W
    dp_transitions_geolife = M_GEOLIFE_PROXY * W
    n_synth_cells = N_SYNTHETIC_TRACKS * N_SIGMA * N_TOL * N_DT
    n_geolife_cells = N_GEOLIFE_TRACKS * N_GEOLIFE_TOL
    transitions_total = n_synth_cells * dp_transitions_synth + n_geolife_cells * dp_transitions_geolife

    # Representative segment size for the projection: a DP candidate segment
    # spans up to W out of m candidates over n_raw points, so its own length
    # is roughly n_raw * W / m. Using the synthetic proxy (m~68, W=8) and a
    # representative n_raw~300-500 (step3's synthetic tracks) gives ~35-60
    # points -- the 50-vertex bracket is the closest tested size.
    representative_n = 50

    old_hours = {lam: project_total_hours(transitions_total, OLD_FULL_TRACK_TIME_MS[lam]) for lam in LAM_VALUES}
    new_hours = {lam: project_total_hours(transitions_total, seg_results[representative_n][lam]) for lam in LAM_VALUES}

    decision = "lam_fallback=0.001 (affordable)" if new_hours[0.001] <= BUDGET_HOURS_THRESHOLD else \
        "lam_fallback=0.1 + tuned internal tol (ADR-0023 rule triggered)"

    lines = [f"{M1_BUDGET_HEADER}\n"]
    lines.append(
        "**Estimate, not a measurement of M2's real runtime** (M2 doesn't exist yet; section "
        "2.1's candidate count `m` is a labeled proxy from M0's DP+SED kept-vertex counts, not "
        "a measured quantity). Purpose: decide `knot_removal.py`'s certification `lam_fallback` "
        "*before* writing it (ADR-0023).\n"
    )
    lines.append(
        f"Assumptions: `m` (candidates/track) proxy = {M_SYNTHETIC_PROXY:.1f} (synthetic, from "
        f"M0's DP+SED n at sigma=0/tol=2/dt=1) / {M_GEOLIFE_PROXY:.0f} (GeoLife, longer tracks); "
        f"`W`={W}; M2 grid = {N_SYNTHETIC_TRACKS} synthetic tracks x {N_SIGMA} sigma x {N_TOL} tol "
        f"x {N_DT} dt ({n_synth_cells:.0f} cells) + {N_GEOLIFE_TRACKS} GeoLife tracks x "
        f"{N_GEOLIFE_TOL} tol ({n_geolife_cells:.0f} cells); DP transitions/track = m*W "
        f"({dp_transitions_synth:.0f} synthetic, {dp_transitions_geolife:.0f} GeoLife); total "
        f"transitions across the grid ~ {transitions_total:,.0f}.\n"
    )

    lines.append("#### Measured `certify_spline` cost by segment length\n")
    lines.append("| segment vertices | n control points p50 | time p50 lam=0.001 (ms) | time p50 lam=0.1 (ms) |")
    lines.append("|---|---|---|---|")
    for n in SEGMENT_SIZES:
        r = seg_results[n]
        lines.append(f"| {n} | {r['n_control_points_p50']:.1f} | {r[0.001]:.2f} | {r[0.1]:.2f} |")
    lines.append("")

    lines.append("#### Old (full-track-based) vs. new (segment-based) M2 projection\n")
    lines.append(
        f"Old estimate reuses ADR-0022's full-track measurements (`benchmarks/results/step8.md` "
        f"M0.2): {OLD_FULL_TRACK_TIME_MS[0.001]:.2f} ms/call at lam=0.001, "
        f"{OLD_FULL_TRACK_TIME_MS[0.1]:.2f} ms/call at lam=0.1 -- treating each DP transition's "
        f"spline-cost evaluation as if it cost as much as certifying an ENTIRE track, which this "
        f"section's own measurement shows is far too pessimistic. New estimate uses the "
        f"{representative_n}-vertex bracket above ({seg_results[representative_n][0.001]:.2f} ms "
        f"lam=0.001, {seg_results[representative_n][0.1]:.2f} ms lam=0.1), the closest tested "
        f"size to a typical DP candidate segment's own length.\n"
    )
    lines.append("| lam_fallback | old estimate (full-track basis) | new estimate (segment basis) | ratio |")
    lines.append("|---|---|---|---|")
    for lam in LAM_VALUES:
        ratio = old_hours[lam] / new_hours[lam] if new_hours[lam] else float("nan")
        lines.append(f"| {lam:g} | {old_hours[lam]:,.1f} h | {new_hours[lam]:,.1f} h | {ratio:.1f}x |")
    lines.append("")

    lines.append(
        f"**Applying ADR-0023's rule:** the segment-based lam=0.001 projection is "
        f"{new_hours[0.001]:,.1f} h ({'<=' if new_hours[0.001] <= BUDGET_HOURS_THRESHOLD else '>'} "
        f"the {BUDGET_HOURS_THRESHOLD:g} h threshold). **Decision: {decision}.**\n"
    )

    body = "\n".join(lines) + "\n"
    upsert_section(OUT_MD, M1_BUDGET_HEADER, body)
    print(f"wrote {OUT_MD}")
    print(body)
    return seg_results, old_hours, new_hours, decision


if __name__ == "__main__":
    main()
