"""Step8 M0.1: separates two candidate explanations for M0's open question (the
LSQ-uniform spline's certified eps_A<=tol fraction collapsing at tight tol,
docs/decisions/ADR-0022):

1. certify_spline's fixed lam_fallback margin (default 0.1 m), added directly to
   eps_A regardless of how good the fit is.
2. spline_lsq.fit_uniform only controlling error AT sample points, letting the
   spline oscillate BETWEEN them in a way certify_spline's Frechet-based eps_A
   is sensitive to but fit_uniform's own search never sees.

Two held-fixed sweeps on the four cells M0 flagged:
  - lam_fallback in {0.1, 0.01, 0.001}, fitter held at fit_uniform.
  - fitter in {fit_uniform, traj.spline.fit()} (ADR-0013's dense-error-between-
    samples fitter), lam_fallback held at the default 0.1.
n (parameter count) is reported alongside every eps_A<=tol fraction, so a
fit()-vs-fit_uniform difference can't be mistaken for "more control points."

ADR-0022's decision rule (fitter choice, lam_fallback choice) was written BEFORE
this script ran; this script only produces the numbers the rule is applied to.

Run:
    venv/bin/python benchmarks/step8_m01_lam_check.py
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
from step3_decisive import (  # noqa: E402
    N_TRACKS,
    SEED,
    _dense_grid,
    _geometry_for_seed,
    make_variable_speed_track,
    search_min_params,
)
from step8_hybrid import _build_spline_uniform  # noqa: E402
from _report_utils import upsert_section  # noqa: E402

CELLS = [(0.0, 0.5), (0.1, 0.5), (1.0, 2.0), (5.0, 10.0)]
LAM_VALUES = [0.001, 0.01, 0.1]
DEFAULT_LAM = 0.1

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_MD = os.path.join(RESULTS_DIR, "step8.md")
M01_HEADER = "### M0.1"


def _geom_seeds(n_tracks=N_TRACKS):
    rng_master = np.random.default_rng(SEED)
    return [int(s) for s in rng_master.integers(0, 1_000_000, size=n_tracks)]


def gather_fit_uniform_candidates(sigma: float, tol: float, n_tracks=N_TRACKS) -> list[dict]:
    """Per track where step3's own (unchanged) honest-curve-error criterion
    accepts fit_uniform at this cell -- same population M0 reported."""
    out = []
    for gseed in _geom_seeds(n_tracks):
        true_xy, duration = _geometry_for_seed(gseed)
        track, _true_xy, _duration = make_variable_speed_track(gseed, sigma, 1.0)
        t_dense = _dense_grid(duration)
        true_xy_dense = true_xy(t_dense)
        res = search_min_params(lambda itol: _build_spline_uniform(track, itol, t_dense, true_xy_dense), tol)
        if res.get("reachable"):
            fit = res["fit"]
            bs = BSpline(fit.tck[0], np.column_stack(fit.tck[1]), fit.tck[2])
            out.append({"track": track, "bs": bs, "n": res["n"]})
    return out


def gather_spline_fit_candidates(sigma: float, tol: float, n_tracks=N_TRACKS) -> tuple[list[dict], int]:
    """spline.fit()'s OWN convergence criterion (dense error vs. its own
    recorded polyline, ADR-0013) -- a DIFFERENT gate than fit_uniform's,
    reported explicitly rather than conflated with it."""
    out = []
    n_total = 0
    for gseed in _geom_seeds(n_tracks):
        track, _true_xy, _duration = make_variable_speed_track(gseed, sigma, 1.0)
        n_total += 1
        fit = spline_module.fit(track, tol=tol)
        if not fit.converged:
            continue
        bs = BSpline(fit.tck[0], np.column_stack(fit.tck[1]), fit.tck[2])
        n_params = len(fit.tck[0]) + 2 * len(fit.tck[1][0])
        out.append({"track": track, "bs": bs, "n": n_params})
    return out, n_total


def lam_sweep(candidates: list[dict], tol: float) -> dict:
    rows = {}
    for lam in LAM_VALUES:
        eps_vals = [certify_spline(c["track"].xy, c["bs"], lam_fallback=lam)[0] for c in candidates]
        eps_arr = np.array(eps_vals, dtype=float)
        ok = eps_arr <= tol
        rows[lam] = {
            "n_pop": len(candidates),
            "frac_ok": float(np.mean(ok)) if len(candidates) else float("nan"),
            "eps_p50": float(np.percentile(eps_arr, 50)) if len(eps_arr) else float("nan"),
            "eps_p90": float(np.percentile(eps_arr, 90)) if len(eps_arr) else float("nan"),
            "eps_max": float(np.max(eps_arr)) if len(eps_arr) else float("nan"),
        }
    return rows


def certified_summary(candidates: list[dict], tol: float, lam: float) -> dict:
    eps_vals = [certify_spline(c["track"].xy, c["bs"], lam_fallback=lam)[0] for c in candidates]
    eps_arr = np.array(eps_vals, dtype=float)
    ok = eps_arr <= tol
    n_vals = [c["n"] for c in candidates]
    return {
        "n_pop": len(candidates),
        "frac_ok": float(np.mean(ok)) if len(candidates) else float("nan"),
        "eps_p50": float(np.percentile(eps_arr, 50)) if len(eps_arr) else float("nan"),
        "eps_p90": float(np.percentile(eps_arr, 90)) if len(eps_arr) else float("nan"),
        "eps_max": float(np.max(eps_arr)) if len(eps_arr) else float("nan"),
        "n_p50": float(np.percentile(n_vals, 50)) if n_vals else float("nan"),
        "n_p90": float(np.percentile(n_vals, 90)) if n_vals else float("nan"),
        "n_values": n_vals,
    }


def run() -> dict:
    per_cell = {}
    for sigma, tol in CELLS:
        fu_candidates = gather_fit_uniform_candidates(sigma, tol)
        sf_candidates, sf_total = gather_spline_fit_candidates(sigma, tol)

        lam_rows = lam_sweep(fu_candidates, tol)
        fu_summary = certified_summary(fu_candidates, tol, DEFAULT_LAM)
        sf_summary = certified_summary(sf_candidates, tol, DEFAULT_LAM)
        sf_summary["n_total_tracks"] = sf_total

        per_cell[(sigma, tol)] = {
            "lam_sweep": lam_rows,
            "fit_uniform": fu_summary,
            "spline_fit": sf_summary,
        }
        print(f"sigma={sigma} tol={tol}: fit_uniform {fu_summary['n_pop']}/{N_TRACKS} reachable, "
              f"spline.fit() {sf_summary['n_pop']}/{sf_total} converged")
    return per_cell


def _fmt_pct(x: float) -> str:
    return "--" if np.isnan(x) else f"{100.0 * x:.1f}%"


def _fmt_m(x: float) -> str:
    return "--" if np.isnan(x) else f"{x:.3f}"


def format_m01(per_cell: dict) -> str:
    lines = [f"{M01_HEADER}\n"]
    lines.append(
        "Investigates the M0 open question (ADR-0022): does the spline's certified "
        "`eps_A<=tol` shortfall at tight `tol` come from `certify_spline`'s fixed "
        "`lam_fallback` margin, or from `fit_uniform`'s between-sample oscillation "
        "(uncontrolled by its own fitting criterion, but visible to `eps_A`'s "
        "Frechet-based check)? Four cells, same 15-track/seed=42 generator as M0.\n"
    )

    avg_frac_by_lam = {lam: [] for lam in LAM_VALUES}
    avg_frac_fu, avg_frac_sf = [], []
    pooled_n_fu, pooled_n_sf = [], []

    for sigma, tol in CELLS:
        cell = per_cell[(sigma, tol)]
        lines.append(f"#### sigma={sigma:g}, tol={tol:g}\n")

        lines.append("`lam_fallback` sweep (fitter = fit_uniform):\n")
        lines.append("| lam_fallback | eps_A<=tol fraction | eps_A p50 | eps_A p90 | eps_A max |")
        lines.append("|---|---|---|---|---|")
        for lam in LAM_VALUES:
            row = cell["lam_sweep"][lam]
            avg_frac_by_lam[lam].append(row["frac_ok"])
            lines.append(
                f"| {lam:g} | {row['n_pop']} pop: {_fmt_pct(row['frac_ok'])} | "
                f"{_fmt_m(row['eps_p50'])} | {_fmt_m(row['eps_p90'])} | {_fmt_m(row['eps_max'])} |"
            )
        lines.append("")

        fu, sf = cell["fit_uniform"], cell["spline_fit"]
        avg_frac_fu.append(fu["frac_ok"])
        avg_frac_sf.append(sf["frac_ok"])
        pooled_n_fu.extend(fu["n_values"])
        pooled_n_sf.extend(sf["n_values"])

        lines.append(f"Fitter comparison (lam_fallback = {DEFAULT_LAM:g} fixed):\n")
        lines.append(
            "| Fitter | Own population (own criterion) | eps_A<=tol fraction | "
            "eps_A p50/p90/max | n median | n p90 |"
        )
        lines.append("|---|---|---|---|---|---|")
        lines.append(
            f"| fit_uniform | {fu['n_pop']}/{N_TRACKS} (honest true-curve error, step3's criterion) | "
            f"{_fmt_pct(fu['frac_ok'])} | {_fmt_m(fu['eps_p50'])}/{_fmt_m(fu['eps_p90'])}/{_fmt_m(fu['eps_max'])} | "
            f"{fu['n_p50']:.1f} | {fu['n_p90']:.1f} |"
        )
        lines.append(
            f"| spline.fit() | {sf['n_pop']}/{sf['n_total_tracks']} (own dense-vs-own-polyline convergence) | "
            f"{_fmt_pct(sf['frac_ok'])} | {_fmt_m(sf['eps_p50'])}/{_fmt_m(sf['eps_p90'])}/{_fmt_m(sf['eps_max'])} | "
            f"{sf['n_p50']:.1f} | {sf['n_p90']:.1f} |"
        )
        lines.append("")

    ref = np.mean(avg_frac_by_lam[0.001])
    chosen_lam = None
    for lam in LAM_VALUES:  # ascending order (LAM_VALUES is already ascending)
        avg = np.mean(avg_frac_by_lam[lam])
        if ref - avg <= 0.05 + 1e-12:
            chosen_lam = lam
            break
    fu_overall = np.mean(avg_frac_fu)
    sf_overall = np.mean(avg_frac_sf)
    gap_pp = 100.0 * (sf_overall - fu_overall)
    median_n_fu = float(np.median(pooled_n_fu)) if pooled_n_fu else float("nan")
    median_n_sf = float(np.median(pooled_n_sf)) if pooled_n_sf else float("nan")

    if abs(gap_pp) >= 20.0:
        fitter_decision = "spline.fit()" if gap_pp > 0 else "fit_uniform"
        fitter_reason = f"average fraction gap is {gap_pp:+.1f}pp, >= 20pp threshold -- decided outright"
    else:
        fitter_decision = "spline.fit()" if median_n_sf < median_n_fu else "fit_uniform"
        fitter_reason = (
            f"average fraction gap is {gap_pp:+.1f}pp, under the 20pp threshold -- "
            f"tie-break by pooled median n (fit_uniform={median_n_fu:.1f}, spline.fit()={median_n_sf:.1f})"
        )

    lines.append("### Applying ADR-0022's pre-registered rule\n")
    lines.append(
        f"`lam_fallback` sweep, average `eps_A<=tol` fraction across the 4 cells: "
        f"lam=0.001 -> {_fmt_pct(np.mean(avg_frac_by_lam[0.001]))} (reference), "
        f"lam=0.01 -> {_fmt_pct(np.mean(avg_frac_by_lam[0.01]))}, "
        f"lam=0.1 -> {_fmt_pct(np.mean(avg_frac_by_lam[0.1]))}. "
        f"**Chosen lam_fallback: {chosen_lam:g}** (smallest tested value within 5pp of the "
        f"lam=0.001 reference).\n"
    )
    lines.append(
        f"Fitter comparison, average `eps_A<=tol` fraction across the 4 cells (lam_fallback="
        f"{DEFAULT_LAM:g}): fit_uniform -> {_fmt_pct(fu_overall)}, spline.fit() -> "
        f"{_fmt_pct(sf_overall)} ({fitter_reason}). **Chosen fitter: {fitter_decision}.**\n"
    )
    lam_swing_pp = 100.0 * (np.mean(avg_frac_by_lam[0.001]) - np.mean(avg_frac_by_lam[0.1]))
    lines.append(
        f"For scale: sweeping `lam_fallback` alone (fit_uniform fixed) moves the average fraction "
        f"by {lam_swing_pp:+.1f}pp (lam=0.1 -> lam=0.001); switching fitter alone (lam=0.1 fixed) "
        f"moves it by {gap_pp:+.1f}pp (fit_uniform -> spline.fit()). Whichever swing is larger in "
        f"magnitude is the dominant mechanism; interpretation in the M0.1 Conclusions below.\n"
    )

    return "\n".join(lines) + "\n"


def main():
    per_cell = run()
    body = format_m01(per_cell)
    upsert_section(OUT_MD, M01_HEADER, body)
    print(f"\nwrote {OUT_MD}")
    print(body)


if __name__ == "__main__":
    main()
