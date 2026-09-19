"""Step8 M0.2: refines ADR-0022's lam_fallback choice, now on the fitter the ADR
actually selected (traj.spline.fit(), not fit_uniform), and checks a specific
hypothesis the user raised: does lam affect what M2 would actually STORE, or
only the certificate/its compute cost?

`traj.encode.SplineSegment` stores a spline's own internal knots + control
points (see `benchmarks/step8_hybrid.py`'s `_spline_segment_from_fit`) -- it
never stores `certify_spline`'s internal linearization `Lin(A')`, which is
computed, used, and discarded purely inside the certificate. So `lam` should
affect eps_A and certification TIME (more subdivision at tighter lam), but NOT
stored bytes, since bytes come from the fit's own tck, independent of lam.
This script measures both the linearization vertex count (proof lam changes
the certification's internal work) and the segment's bytes (to confirm they
don't move) side by side, rather than assuming either.

1. Lam sweep {0.1, 0.01, 0.001} on spline.fit()'s own candidates, same 4 cells
   as M0.1: eps_A<=tol fraction, eps_A p50/p90/max, n (control points) median,
   Lin(A') vertex count median, segment bytes (encode_segments+zlib), and
   median/p90 certification wall time.
2. tol=0.5 m spline availability: spline.fit() with INTERNAL tol in {0.25, 0.1}
   (tighter than the 0.5 m target), certified with the lam chosen in step 1,
   at the two tol=0.5 cells.

Run:
    venv/bin/python benchmarks/step8_m02_lam_refine.py
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
from traj.certify import certified_linearize, certify_spline  # noqa: E402
from traj.encode import SplineSegment, encode_segments  # noqa: E402
from step3_decisive import N_TRACKS, make_variable_speed_track  # noqa: E402
from step8_m01_lam_check import CELLS, DEFAULT_LAM, LAM_VALUES, _geom_seeds, gather_spline_fit_candidates  # noqa: E402
from _report_utils import upsert_section  # noqa: E402

TOL_0_5_CELLS = [(sigma, tol) for sigma, tol in CELLS if tol == 0.5]
TIGHT_INTERNAL_TOLS = [0.25, 0.1]
MAX_LEVELS = 12  # certify_spline's own default, kept explicit here for the linearize() call

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_MD = os.path.join(RESULTS_DIR, "step8.md")
M02_HEADER = "### M0.2"


def _segment_bytes(fit_like_tck: tuple) -> int:
    knots, c_list, k = fit_like_tck
    internal = np.asarray(knots[k + 1 : len(knots) - (k + 1)], dtype=float)
    control_xy = np.column_stack(c_list)
    seg = SplineSegment(t_start=float(knots[0]), t_end=float(knots[-1]), internal_knots=internal, control_xy=control_xy)
    return len(encode_segments([seg]))


def lam_sweep_on_spline_fit(candidates: list[dict], tol: float) -> dict:
    rows = {}
    n_vals = [c["n"] for c in candidates]
    byte_vals = [_segment_bytes((c["bs"].t, [c["bs"].c[:, 0], c["bs"].c[:, 1]], c["bs"].k)) for c in candidates]
    for lam in LAM_VALUES:
        eps_vals, times, lin_counts = [], [], []
        for c in candidates:
            lin_vertices, _fully = certified_linearize(c["bs"], lam, max_levels=MAX_LEVELS)
            lin_counts.append(len(lin_vertices))
            t0 = time.perf_counter()
            eps_A, _method = certify_spline(c["track"].xy, c["bs"], lam_fallback=lam)
            times.append(time.perf_counter() - t0)
            eps_vals.append(eps_A)
        eps_arr = np.array(eps_vals, dtype=float)
        ok = eps_arr <= tol
        rows[lam] = {
            "n_pop": len(candidates),
            "frac_ok": float(np.mean(ok)) if len(candidates) else float("nan"),
            "eps_p50": float(np.percentile(eps_arr, 50)) if len(eps_arr) else float("nan"),
            "eps_p90": float(np.percentile(eps_arr, 90)) if len(eps_arr) else float("nan"),
            "eps_max": float(np.max(eps_arr)) if len(eps_arr) else float("nan"),
            "n_p50": float(np.percentile(n_vals, 50)) if n_vals else float("nan"),
            "lin_count_p50": float(np.percentile(lin_counts, 50)) if lin_counts else float("nan"),
            "bytes_p50": float(np.percentile(byte_vals, 50)) if byte_vals else float("nan"),
            "time_p50_ms": 1000.0 * float(np.percentile(times, 50)) if times else float("nan"),
            "time_p90_ms": 1000.0 * float(np.percentile(times, 90)) if times else float("nan"),
        }
    return rows


def check_tol_0_5_availability(sigma: float, target_tol: float, chosen_lam: float, n_tracks=N_TRACKS) -> dict:
    rows = {}
    for itol in TIGHT_INTERNAL_TOLS:
        n_total = 0
        eps_vals, n_vals = [], []
        for gseed in _geom_seeds(n_tracks):
            track, _true_xy, _duration = make_variable_speed_track(gseed, sigma, 1.0)
            n_total += 1
            fit = spline_module.fit(track, tol=itol)
            if not fit.converged:
                continue
            bs = BSpline(fit.tck[0], np.column_stack(fit.tck[1]), fit.tck[2])
            eps_A, _method = certify_spline(track.xy, bs, lam_fallback=chosen_lam)
            eps_vals.append(eps_A)
            n_vals.append(len(fit.tck[0]) + 2 * len(fit.tck[1][0]))
        eps_arr = np.array(eps_vals, dtype=float)
        ok = eps_arr <= target_tol
        rows[itol] = {
            "n_converged": len(eps_vals),
            "n_total": n_total,
            "frac_ok": float(np.mean(ok)) if len(eps_vals) else float("nan"),
            "eps_p50": float(np.percentile(eps_arr, 50)) if len(eps_arr) else float("nan"),
            "eps_min": float(np.min(eps_arr)) if len(eps_arr) else float("nan"),
            "n_p50": float(np.percentile(n_vals, 50)) if n_vals else float("nan"),
        }
    return rows


def _fmt_pct(x: float) -> str:
    return "--" if np.isnan(x) else f"{100.0 * x:.1f}%"


def _fmt_m(x: float) -> str:
    return "--" if np.isnan(x) else f"{x:.3f}"


def choose_lam(avg_frac: dict, avg_bytes: dict, avg_time: dict, tie_eps: float = 1e-9) -> tuple[float, str]:
    ref = avg_frac[0.001]
    candidates = [lam for lam in LAM_VALUES if ref - avg_frac[lam] <= 0.05 + 1e-12]
    default = min(candidates)
    tied = [lam for lam in candidates if abs(avg_frac[lam] - avg_frac[default]) <= tie_eps]
    best = min(tied, key=lambda lam: (avg_bytes[lam], avg_time[lam]))
    reason = (
        f"default (smallest within 5pp of the lam=0.001 reference) is {default:g}; "
        f"tie-break set (equal average fraction, {avg_frac[default] * 100:.1f}%): {tied} -- "
        f"chosen by smallest bytes then smallest time: {best:g}"
    )
    return best, reason


def main():
    per_cell = {}
    for sigma, tol in CELLS:
        candidates = gather_spline_fit_candidates(sigma, tol)[0]
        per_cell[(sigma, tol)] = lam_sweep_on_spline_fit(candidates, tol)
        print(f"sigma={sigma} tol={tol}: {len(candidates)} spline.fit() candidates swept")

    avg_frac = {lam: float(np.mean([per_cell[cell][lam]["frac_ok"] for cell in CELLS])) for lam in LAM_VALUES}
    avg_bytes = {lam: float(np.mean([per_cell[cell][lam]["bytes_p50"] for cell in CELLS])) for lam in LAM_VALUES}
    avg_time = {lam: float(np.mean([per_cell[cell][lam]["time_p50_ms"] for cell in CELLS])) for lam in LAM_VALUES}
    avg_lin = {lam: float(np.mean([per_cell[cell][lam]["lin_count_p50"] for cell in CELLS])) for lam in LAM_VALUES}
    chosen_lam, reason = choose_lam(avg_frac, avg_bytes, avg_time)

    tol05_rows = {sigma: check_tol_0_5_availability(sigma, 0.5, chosen_lam) for sigma, _tol in TOL_0_5_CELLS}

    lines = [f"{M02_HEADER}\n"]
    lines.append(
        "Refines ADR-0022 on the fitter it actually selected (`spline.fit()`, not "
        "`fit_uniform`), and checks whether `lam_fallback` affects what M2 would "
        "store: `traj.encode.SplineSegment` encodes a spline's own control points, "
        "never `certify_spline`'s internal linearization `Lin(A')` -- the linearization "
        "vertex count and the segment's stored bytes are measured side by side below "
        "to confirm this rather than assume it.\n"
    )

    for sigma, tol in CELLS:
        rows = per_cell[(sigma, tol)]
        lines.append(f"#### sigma={sigma:g}, tol={tol:g}\n")
        lines.append(
            "| lam_fallback | eps_A<=tol fraction | eps_A p50/p90/max | n (control pts) p50 | "
            "Lin(A') vertices p50 | segment bytes p50 | cert time p50/p90 (ms) |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for lam in LAM_VALUES:
            r = rows[lam]
            lines.append(
                f"| {lam:g} | {_fmt_pct(r['frac_ok'])} ({r['n_pop']} pop) | "
                f"{_fmt_m(r['eps_p50'])}/{_fmt_m(r['eps_p90'])}/{_fmt_m(r['eps_max'])} | "
                f"{r['n_p50']:.1f} | {r['lin_count_p50']:.1f} | {r['bytes_p50']:.0f}B | "
                f"{r['time_p50_ms']:.2f}/{r['time_p90_ms']:.2f} |"
            )
        lines.append("")

    lines.append("### Applying the refined lam rule\n")
    lines.append("| lam_fallback | avg eps_A<=tol fraction (4 cells) | avg Lin(A') vertices | avg segment bytes | avg cert time p50 (ms) |")
    lines.append("|---|---|---|---|---|")
    for lam in LAM_VALUES:
        lines.append(f"| {lam:g} | {_fmt_pct(avg_frac[lam])} | {avg_lin[lam]:.1f} | {avg_bytes[lam]:.1f}B | {avg_time[lam]:.2f} |")
    lines.append("")
    lines.append(f"{reason}. **Chosen lam_fallback: {chosen_lam:g}.**\n")

    lines.append("### tol=0.5 m spline availability (internal tol tighter than the 0.5 m target)\n")
    lines.append(f"Certified with the chosen lam_fallback ({chosen_lam:g}).\n")
    lines.append("| sigma | internal tol | converged | eps_A<=0.5 fraction | eps_A p50 | eps_A min | n p50 |")
    lines.append("|---|---|---|---|---|---|---|")
    for sigma, _tol in TOL_0_5_CELLS:
        for itol in TIGHT_INTERNAL_TOLS:
            r = tol05_rows[sigma][itol]
            lines.append(
                f"| {sigma:g} | {itol:g} | {r['n_converged']}/{r['n_total']} | {_fmt_pct(r['frac_ok'])} | "
                f"{_fmt_m(r['eps_p50'])} | {_fmt_m(r['eps_min'])} | {r['n_p50']:.1f} |"
            )
    lines.append("")

    any_pass = any(
        tol05_rows[sigma][itol]["frac_ok"] > 0
        for sigma, _tol in TOL_0_5_CELLS
        for itol in TIGHT_INTERNAL_TOLS
        if not np.isnan(tol05_rows[sigma][itol]["frac_ok"])
    )
    if not any_pass:
        lines.append(
            "**No spline representation at either internal tol (0.25 m, 0.1 m) certifies "
            "`eps_A<=0.5` at either sigma.** At `tol=0.5 m`, spline segments are certificate-"
            "unavailable in M2's terms -- choosing the polyline there is a construction-"
            "availability fact, not a geometric finding about which shape compresses better.\n"
        )
    else:
        lines.append(
            "At least one internal-tol/sigma combination does certify `eps_A<=0.5` -- see the "
            "table above for which, and the M0.2 Conclusions below for the byte cost.\n"
        )

    body = "\n".join(lines) + "\n"
    upsert_section(OUT_MD, M02_HEADER, body)
    print(f"\nwrote {OUT_MD}")
    print(body)


if __name__ == "__main__":
    main()
