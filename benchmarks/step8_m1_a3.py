"""Step8 M1: LSQ-free-knot, the free-knot oracle (spec section 2.6), and
criterion A3 (spec section 8) -- "does even the free-knot spline oracle beat
DP+SED by 20% on a smooth, noiseless curve, at all?" H1's ceiling check.

Scope: M1's own artifact (the milestone table) is A3 alone, which spec
section 8 states specifically "без шума, dt=1" -- sigma=0, dt=1, matching
M0's own grid exactly, so DP+SED's numbers are REUSED from
benchmarks/results/step8.md's M0 section, not recomputed.

Mandatory correction 1: reachability for ALL THREE methods (oracle, DP+SED,
LSQ-free-knot) uses the SAME criterion -- step3's own honest error against
the true noise-free curve on a dense grid, `<= target_tol` (ADR-0021: the
certificate stays an ADDITIONAL check, reported alongside, never the gate).
The oracle fits directly to a DENSE true-curve sample (spec 2.6: "без шума и
разреженности" -- no noise, no sparsity), not the sparse dt=1 samples.

Mandatory correction 2: a cell is "valid" for A3 using spec section 8's
definition VERBATIM -- both methods A3 actually COMPARES (oracle, DP+SED)
reach >=80% of tracks. No additional conditions.

Run:
    venv/bin/python benchmarks/step8_m1_a3.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from traj import spline as spline_module  # noqa: E402
from traj.encode import SplineSegment, encode_segments  # noqa: E402
from traj.knot_removal import remove_knots  # noqa: E402
from step2_crossover import true_curve_error  # noqa: E402
from step3_decisive import N_TRACKS, SEED, TOL_LIST, _dense_grid, _geometry_for_seed, make_variable_speed_track  # noqa: E402
from _report_utils import upsert_section  # noqa: E402

SIGMA = 0.0
DT = 1.0
A3_THRESHOLD = 0.80
VALID_CELL_FRACTION = 0.80

# Reused directly from benchmarks/results/step8.md's M0 section (sigma=0,
# dt=1 -- same generator/seed) -- NOT recomputed, per this project's "reuse,
# don't redo" discipline and per correction 1's "same basis" requirement
# (M0's own numbers already use step3's honest-curve criterion).
DP_SED_M0 = {
    0.5: {"bytes": 481, "reachable": 5, "total": N_TRACKS},
    2.0: {"bytes": 470, "reachable": 15, "total": N_TRACKS},
    10.0: {"bytes": 282, "reachable": 15, "total": N_TRACKS},
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_MD = os.path.join(RESULTS_DIR, "step8.md")
M1_HEADER = "## M1"


def _segment_bytes_from_bs(bs) -> int:
    knots, k = bs.t, bs.k
    internal = np.asarray(knots[k + 1 : len(knots) - (k + 1)], dtype=float)
    seg = SplineSegment(t_start=float(knots[0]), t_end=float(knots[-1]), internal_knots=internal, control_xy=np.asarray(bs.c))
    return len(encode_segments([seg]))


def _dense_u(t_dense: np.ndarray) -> np.ndarray:
    dummy_xy = np.zeros((len(t_dense), 2))
    return spline_module._param_u(t_dense, dummy_xy, "time")


def evaluate_lsq_free_knot(track, t_dense, true_xy_dense, target_tol) -> dict:
    t0 = time.perf_counter()
    result = remove_knots(track.t, track.xy, track.xy, target_tol=target_tol)
    elapsed = time.perf_counter() - t0
    if not result.converged:
        return {"reachable": False, "eps_A": result.eps_A, "time": elapsed}
    recon = result.bs(_dense_u(t_dense))
    err = true_curve_error(true_xy_dense, recon)
    return {
        "reachable": err <= target_tol,
        "err": err,
        "eps_A": result.eps_A,
        "cert_ok": result.eps_A <= target_tol,
        "n": result.n,
        "bytes": _segment_bytes_from_bs(result.bs),
        "time": elapsed,
    }


def evaluate_oracle(t_dense, true_xy_dense, target_tol) -> dict:
    t0 = time.perf_counter()
    result = remove_knots(t_dense, true_xy_dense, true_xy_dense, target_tol=target_tol)
    elapsed = time.perf_counter() - t0
    if not result.converged:
        return {"reachable": False, "eps_A": result.eps_A, "time": elapsed}
    recon = result.bs(result.u)
    err = true_curve_error(true_xy_dense, recon)
    return {
        "reachable": err <= target_tol,
        "err": err,
        "eps_A": result.eps_A,
        "cert_ok": result.eps_A <= target_tol,
        "n": result.n,
        "bytes": _segment_bytes_from_bs(result.bs),
        "time": elapsed,
    }


def run_grid(n_tracks=N_TRACKS) -> dict:
    rng_master = np.random.default_rng(SEED)
    geom_seeds = [int(s) for s in rng_master.integers(0, 1_000_000, size=n_tracks)]
    cells = {tol: {"oracle": [], "lsq_free_knot": []} for tol in TOL_LIST}

    for gseed in geom_seeds:
        true_xy, duration = _geometry_for_seed(gseed)
        t_dense = _dense_grid(duration)
        true_xy_dense = true_xy(t_dense)
        track, _true_xy, _duration = make_variable_speed_track(gseed, SIGMA, DT)
        for tol in TOL_LIST:
            cells[tol]["oracle"].append(evaluate_oracle(t_dense, true_xy_dense, tol))
            cells[tol]["lsq_free_knot"].append(evaluate_lsq_free_knot(track, t_dense, true_xy_dense, tol))
        print(f"track seed={gseed} done")
    return cells


def _summarize(entries: list[dict]) -> dict:
    reachable = [e for e in entries if e["reachable"]]
    n_total = len(entries)
    n_reach = len(reachable)
    out = {"n_total": n_total, "n_reach": n_reach, "frac_reach": n_reach / n_total if n_total else float("nan")}
    if n_reach:
        out["mean_bytes"] = float(np.mean([e["bytes"] for e in reachable]))
        out["mean_n"] = float(np.mean([e["n"] for e in reachable]))
        out["n_p50"] = float(np.median([e["n"] for e in reachable]))
        out["n_p90"] = float(np.percentile([e["n"] for e in reachable], 90))
        out["time_p50"] = float(np.median([e["time"] for e in reachable]))
        out["time_p90"] = float(np.percentile([e["time"] for e in reachable], 90))
        cert_ok = [e for e in reachable if e.get("cert_ok")]
        out["cert_frac"] = len(cert_ok) / n_reach
    return out


def main():
    cells = run_grid()
    summaries = {tol: {m: _summarize(cells[tol][m]) for m in ("oracle", "lsq_free_knot")} for tol in TOL_LIST}

    lines = [f"{M1_HEADER}\n"]
    lines.append(
        "Scope: M1's own artifact (spec section 7's milestone table) is criterion A3 alone, "
        "which spec section 8 states specifically for `sigma=0, dt=1` -- matching M0's own grid "
        "(`benchmarks/results/step8.md`'s M0 section), whose DP+SED numbers are REUSED directly "
        "below, not recomputed. Reachability for all three methods (oracle, DP+SED, "
        "LSQ-free-knot) uses the SAME criterion: honest error against the true noise-free curve "
        "on a dense grid, `<= target_tol` -- identical to step3's/M0's own gate (ADR-0021: the "
        "certificate, `certify_spline`'s `eps_A`, stays an additional, non-gating check, reported "
        "alongside as `cert_frac` below). The oracle fits directly to a DENSE true-curve sample "
        "(spec 2.6 -- no noise, no sparsity), not the sparse dt=1 samples the other two methods "
        "see. `n`/time distributions are over each method's own reachable population.\n"
    )

    lines.append("### Methods (sigma=0, dt=1, 15 tracks, seed=42)\n")
    lines.append("| tol | method | bytes (mean) | n mean/p50/p90 | reachable | cert eps_A<=tol fraction | time p50/p90 (ms) |")
    lines.append("|---|---|---|---|---|---|---|")
    for tol in TOL_LIST:
        dp = DP_SED_M0[tol]
        lines.append(f"| {tol:g} | DP+SED (M0, reused) | {dp['bytes']}B | -- | {dp['reachable']}/{dp['total']} | -- | -- |")
        for m_key, title in (("oracle", "Oracle (free knots, true curve)"), ("lsq_free_knot", "LSQ-free-knot (recorded track)")):
            s = summaries[tol][m_key]
            if s["n_reach"] == 0:
                lines.append(f"| {tol:g} | {title} | -- | -- | 0/{s['n_total']} | -- | -- |")
                continue
            lines.append(
                f"| {tol:g} | {title} | {s['mean_bytes']:.0f}B | {s['mean_n']:.1f}/{s['n_p50']:.1f}/{s['n_p90']:.1f} | "
                f"{s['n_reach']}/{s['n_total']} | {100 * s['cert_frac']:.1f}% | {1000 * s['time_p50']:.1f}/{1000 * s['time_p90']:.1f} |"
            )
    lines.append("")

    valid_cells = []
    for tol in TOL_LIST:
        dp = DP_SED_M0[tol]
        dp_frac = dp["reachable"] / dp["total"]
        oracle_frac = summaries[tol]["oracle"]["frac_reach"]
        valid = dp_frac >= VALID_CELL_FRACTION and oracle_frac >= VALID_CELL_FRACTION
        valid_cells.append((tol, valid, dp_frac, oracle_frac))

    lines.append("### Valid-cell check for A3 (spec section 8, verbatim: both COMPARED methods -- oracle, DP+SED -- reach >=80%)\n")
    lines.append("| tol | DP+SED reachable fraction | oracle reachable fraction | valid for A3? |")
    lines.append("|---|---|---|---|")
    for tol, valid, dp_frac, oracle_frac in valid_cells:
        lines.append(f"| {tol:g} | {100 * dp_frac:.1f}% | {100 * oracle_frac:.1f}% | {'yes' if valid else 'no'} |")
    lines.append("")

    a3_rows = []
    for tol, valid, _dp_frac, _oracle_frac in valid_cells:
        if not valid:
            continue
        dp_bytes = DP_SED_M0[tol]["bytes"]
        oracle_bytes = summaries[tol]["oracle"]["mean_bytes"]
        ratio = oracle_bytes / dp_bytes
        passed = ratio <= A3_THRESHOLD
        a3_rows.append((tol, dp_bytes, oracle_bytes, ratio, passed))

    a3_passed = any(row[4] for row in a3_rows)

    lines.append("### Criterion A3 (spec section 8)\n")
    lines.append("| # | Criterion | Threshold | Actual | Passed |")
    lines.append("|---|---|---|---|---|")
    if a3_rows:
        detail = "; ".join(f"tol={tol:g}: {ratio:.3f}x{' (pass)' if passed else ''}" for tol, _db, _ob, ratio, passed in a3_rows)
        lines.append(
            f"| A3 | oracle bytes <= 0.80 x DP+SED bytes, in >=1 valid cell (sigma=0, dt=1) | "
            f"<=0.80x | {detail} | {'yes' if a3_passed else 'no'} |"
        )
    else:
        lines.append("| A3 | oracle bytes <= 0.80 x DP+SED bytes, in >=1 valid cell (sigma=0, dt=1) | <=0.80x | no valid cells | no |")
    lines.append("")

    body = "\n".join(lines) + "\n"
    upsert_section(OUT_MD, M1_HEADER, body)
    print(f"wrote {OUT_MD}")
    print(body)
    return summaries, valid_cells, a3_rows, a3_passed


if __name__ == "__main__":
    main()
