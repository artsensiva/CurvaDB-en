"""Step8 M1 item 0 (continued): picks knot_removal.py's internal-tol fraction
at the chosen lam_fallback=0.1 (ADR-0023's budget decision), on the same four
cells M0.1/M0.2 used. Reference is M0.2's own lam=0.001-at-ratio=1.0 average
fraction (70.0%, benchmarks/results/step8.md's M0.2 "Applying the refined lam
rule" table) -- reused directly, not recomputed.

Run:
    venv/bin/python benchmarks/step8_m1_internal_tol.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.interpolate import BSpline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from traj import spline as spline_module  # noqa: E402
from traj.certify import certify_spline  # noqa: E402
from step3_decisive import N_TRACKS, make_variable_speed_track  # noqa: E402
from step8_m01_lam_check import CELLS, _geom_seeds  # noqa: E402
from _report_utils import upsert_section  # noqa: E402

RATIOS = [1.0, 0.5, 0.25, 0.1]
LAM = 0.1
REFERENCE_FRACTION = 0.70  # M0.2: average fraction at lam=0.001, ratio=1.0
TIE_TOLERANCE_PP = 0.05

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_MD = os.path.join(RESULTS_DIR, "step8.md")
HEADER = "### M1.0 -- internal-tol fraction at lam=0.1 (ADR-0023)"


def evaluate_ratio(sigma: float, target_tol: float, ratio: float, n_tracks=N_TRACKS) -> dict:
    eps_vals, n_vals = [], []
    n_total = 0
    for gseed in _geom_seeds(n_tracks):
        track, _true_xy, _duration = make_variable_speed_track(gseed, sigma, 1.0)
        n_total += 1
        fit = spline_module.fit(track, tol=ratio * target_tol)
        if not fit.converged:
            continue
        bs = BSpline(fit.tck[0], np.column_stack(fit.tck[1]), fit.tck[2])
        eps_A, _method = certify_spline(track.xy, bs, lam_fallback=LAM)
        eps_vals.append(eps_A)
        n_vals.append(len(fit.tck[0]) + 2 * len(fit.tck[1][0]))
    eps_arr = np.array(eps_vals, dtype=float)
    ok = eps_arr <= target_tol
    return {
        "n_converged": len(eps_vals),
        "n_total": n_total,
        "frac_ok": float(np.mean(ok)) if len(eps_vals) else float("nan"),
        "n_p50": float(np.median(n_vals)) if n_vals else float("nan"),
    }


def main():
    per_ratio_frac = {ratio: [] for ratio in RATIOS}
    per_ratio_n = {ratio: [] for ratio in RATIOS}
    lines = [f"{HEADER}\n"]
    lines.append(
        f"Reference: M0.2's average `eps_A<=tol` fraction at lam_fallback=0.001, internal tol = "
        f"target tol (ratio=1.0), across the same 4 cells: **{100 * REFERENCE_FRACTION:.1f}%** "
        f"(reused from `benchmarks/results/step8.md`'s M0.2 section, not recomputed).\n"
    )
    for sigma, tol in CELLS:
        lines.append(f"#### sigma={sigma:g}, tol={tol:g} (lam_fallback={LAM:g})\n")
        lines.append("| internal-tol ratio | converged | eps_A<=tol fraction | n p50 |")
        lines.append("|---|---|---|---|")
        for ratio in RATIOS:
            r = evaluate_ratio(sigma, tol, ratio)
            per_ratio_frac[ratio].append(r["frac_ok"])
            per_ratio_n[ratio].append(r["n_p50"])
            lines.append(f"| {ratio:g} | {r['n_converged']}/{r['n_total']} | {100 * r['frac_ok']:.1f}% | {r['n_p50']:.1f} |")
        lines.append("")

    avg_frac = {ratio: float(np.mean(per_ratio_frac[ratio])) for ratio in RATIOS}
    avg_n = {ratio: float(np.mean(per_ratio_n[ratio])) for ratio in RATIOS}

    candidates = [r for r in RATIOS if REFERENCE_FRACTION - avg_frac[r] <= TIE_TOLERANCE_PP + 1e-12]
    chosen = max(candidates) if candidates else min(RATIOS, key=lambda r: -avg_frac[r])

    lines.append("#### Applying ADR-0023's internal-tol rule\n")
    lines.append("| ratio | avg eps_A<=tol fraction (4 cells) | avg n p50 | within 5pp of reference? |")
    lines.append("|---|---|---|---|")
    for ratio in RATIOS:
        within = "yes" if REFERENCE_FRACTION - avg_frac[ratio] <= TIE_TOLERANCE_PP + 1e-12 else "no"
        lines.append(f"| {ratio:g} | {100 * avg_frac[ratio]:.1f}% | {avg_n[ratio]:.1f} | {within} |")
    lines.append("")
    lines.append(
        f"**Chosen internal-tol ratio: {chosen:g}** (largest ratio -- i.e. cheapest construction "
        f"-- whose average fraction stays within 5pp of the {100 * REFERENCE_FRACTION:.1f}% "
        f"reference)." + ("" if candidates else " No ratio met the 5pp bar; falling back to the "
        "highest-fraction ratio tested instead.") + "\n"
    )

    body = "\n".join(lines) + "\n"
    upsert_section(OUT_MD, HEADER, body)
    print(f"wrote {OUT_MD}")
    print(body)
    return chosen, avg_frac, avg_n


if __name__ == "__main__":
    main()
