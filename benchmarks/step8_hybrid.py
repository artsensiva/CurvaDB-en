"""Step8 direction A (docs/specs/step8_A_hybrid.md), the section-3 grid: DP+SED,
LSQ-uniform, LSQ-free-knot, hybrid, and their oracles, incrementally filled in
across M0-M4. This module only adds M0.

M0's job (spec section 7 + section 11's literal prompt): re-verify step3's own
DP+SED / LSQ-uniform byte counts through the new section-2.5 unified encoding
(traj.encode), on step3's EXACT synthetic grid (dt=1, seed=42, 15 tracks, all
sigma x tol) -- reusing step3_decisive.py's generator and selection logic
unchanged, not reimplementing them.

Reachability criterion is UNCHANGED from step3: honest error against the true
noise-free curve on a dense grid (`true_curve_error`), exactly as
`step3_decisive.search_min_params` already computes it -- this keeps step8's
numbers comparable to benchmarks/results/step3.md, and the spec's own H1
acceptance rule (section 8, criterion A3) is phrased in these same terms.
Step7's certificates (`certify_polyline`/`certify_spline`, default section 2.4,
ADR-0018) are a SEPARATE, additional check: for whichever candidate the
UNCHANGED step3 criterion selects as the cell's answer, `eps_A` against the
recorded/sampled track is computed once (not swept), and the fraction of
reachable cells with `eps_A <= target_tol` is reported as a correctness metric
-- not a selection criterion (docs/decisions/ADR-0021).

Run:
    venv/bin/python benchmarks/step8_hybrid.py --pilot   # 3 tracks
    venv/bin/python benchmarks/step8_hybrid.py           # full run, writes results/step8.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zlib

import numpy as np
from scipy.interpolate import BSpline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from traj import spline_lsq  # noqa: E402
from traj.certify import certify_polyline, certify_spline  # noqa: E402
from traj.encode import LineSegment, SplineSegment, encode_segments  # noqa: E402
from traj.simplify import simplify_sed_with_indices  # noqa: E402
from step3_decisive import (  # noqa: E402
    N_TRACKS,
    N_TRACKS_PILOT,
    SEED,
    SIGMA_LIST,
    TOL_LIST,
    _dense_grid,
    _dp_reconstruct,
    _geometry_for_seed,
    make_variable_speed_track,
    search_min_params,
    true_curve_error,
)
from _report_utils import upsert_section  # noqa: E402

DT = 1.0
METHODS = ("dp_sed", "spline_uniform")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_MD = os.path.join(RESULTS_DIR, "step8.md")
M0_HEADER = "## M0"


# ------------------------------------------------------------ segments --

def _spline_segment_from_fit(fit: spline_lsq.LsqSplineFit) -> SplineSegment:
    knots, c_list, k = fit.tck
    internal = np.asarray(knots[k + 1 : len(knots) - (k + 1)], dtype=float)
    control_xy = np.column_stack(c_list)
    return SplineSegment(t_start=fit.t_min, t_end=fit.t_max, internal_knots=internal, control_xy=control_xy)


# ------------------------------------------------------------ builders --
# Identical cost profile to step3_decisive.py's own _build_dp_sed/_build_spline_lsq
# (no certify_* call here -- that stays a POST-selection, once-per-reachable-cell
# step, done in evaluate_cell below, so the coarse-scan/bisection search itself
# is exactly as fast as step3's).

def _build_dp_sed(track, itol: float, t_dense, true_xy_dense):
    _, idx = simplify_sed_with_indices(track, itol)
    recon = _dp_reconstruct(t_dense, track.t[idx], track.xy[idx])
    err = true_curve_error(true_xy_dense, recon)
    nbytes = len(encode_segments([LineSegment(t=track.t[idx], xy=track.xy[idx])]))
    n = len(idx) * 3
    return {"n": n, "bytes": nbytes, "err": err, "idx": idx}


def _build_spline_uniform(track, itol: float, t_dense, true_xy_dense):
    fit = spline_lsq.fit_uniform(track, itol)
    recon = spline_lsq.reconstruct(fit, t_dense)
    err = true_curve_error(true_xy_dense, recon)
    nbytes = len(encode_segments([_spline_segment_from_fit(fit)]))
    n = len(fit.tck[0]) + 2 * len(fit.tck[1][0])
    return {"n": n, "bytes": nbytes, "err": err, "fit": fit}


def evaluate_cell(track, true_xy_dense, t_dense, target_tol: float) -> dict:
    dp = search_min_params(lambda itol: _build_dp_sed(track, itol, t_dense, true_xy_dense), target_tol)
    if dp.get("reachable"):
        eps_A = certify_polyline(track.xy, dp["idx"])
        dp["eps_A"] = eps_A
        dp["eps_ok"] = eps_A <= target_tol

    sp = search_min_params(lambda itol: _build_spline_uniform(track, itol, t_dense, true_xy_dense), target_tol)
    if sp.get("reachable"):
        fit = sp["fit"]
        bs = BSpline(fit.tck[0], np.column_stack(fit.tck[1]), fit.tck[2])
        eps_A, method = certify_spline(track.xy, bs)
        sp["eps_A"] = eps_A
        sp["eps_A_method"] = method
        sp["eps_ok"] = eps_A <= target_tol

    return {"dp_sed": dp, "spline_uniform": sp}


# ------------------------------------------------------------------ run --

def run_grid(n_tracks, sigma_list, tol_list, seed=SEED, verbose=True) -> dict:
    rng_master = np.random.default_rng(seed)
    geom_seeds = [int(s) for s in rng_master.integers(0, 1_000_000, size=n_tracks)]

    cells = {(s, t): {m: [] for m in METHODS} for s in sigma_list for t in tol_list}

    for gseed in geom_seeds:
        true_xy, duration = _geometry_for_seed(gseed)
        for sigma in sigma_list:
            track, _true_xy, _duration = make_variable_speed_track(gseed, sigma, DT)
            t_dense = _dense_grid(duration)
            true_xy_dense = true_xy(t_dense)
            for tol in tol_list:
                res = evaluate_cell(track, true_xy_dense, t_dense, tol)
                for m in METHODS:
                    cells[(sigma, tol)][m].append(res[m])
        if verbose:
            print(f"track seed={gseed} done")
    return cells


# --------------------------------------------------------- segment overhead --

def measure_segment_overhead(total_points: int = 200, counts=(1, 2, 5, 10, 20), seed: int = 123) -> list[dict]:
    """Fixed total payload (total_points vertices), cut into `counts[i]` line
    segments sharing boundary vertices per 2.5's dedup rule -- the only thing
    that changes between rows is how many segment headers get paid for, since
    the dedup means shared boundary vertices are never double-stored. Needed
    for M2, where a track can be cut into many segments."""
    rng = np.random.default_rng(seed)
    t_all = np.sort(rng.uniform(0.0, 1000.0, total_points))
    xy_all = rng.uniform(-1000.0, 1000.0, (total_points, 2))

    rows = []
    baseline_raw = None
    for n_seg in counts:
        cut_idx = np.unique(np.linspace(0, total_points - 1, n_seg + 1).round().astype(int))
        segments = [
            LineSegment(t=t_all[cut_idx[i] : cut_idx[i + 1] + 1], xy=xy_all[cut_idx[i] : cut_idx[i + 1] + 1])
            for i in range(len(cut_idx) - 1)
        ]
        blob = encode_segments(segments)
        raw = zlib.decompress(blob)
        if baseline_raw is None:
            baseline_raw = len(raw)
        header_bytes = len(raw) - baseline_raw
        rows.append(
            {
                "n_segments": len(segments),
                "pre_zlib_bytes": len(raw),
                "post_zlib_bytes": len(blob),
                "header_bytes": header_bytes,
                "header_fraction": header_bytes / len(raw) if len(raw) else 0.0,
            }
        )
    return rows


# --------------------------------------------------------------- report --

def _summarize_method(entries: list[dict]) -> tuple[str, str]:
    reachable = [e for e in entries if e.get("reachable")]
    n_total = len(entries)
    n_reach = len(reachable)
    if n_reach == 0:
        return "unreachable", "--"
    mean_bytes = np.mean([e["bytes"] for e in reachable])
    mean_n = np.mean([e["n"] for e in reachable])
    suffix = "" if n_reach == n_total else f", {n_reach}/{n_total} reachable"
    bytes_str = f"{mean_bytes:.0f}B (n={mean_n:.1f}){suffix}"
    eps_ok = [e for e in reachable if e.get("eps_ok")]
    eps_str = f"{len(eps_ok)}/{n_reach} ({100.0 * len(eps_ok) / n_reach:.1f}%)"
    return bytes_str, eps_str


def format_m0(cells: dict, sigma_list, tol_list, n_tracks, overhead_rows: list[dict], elapsed: float) -> str:
    lines = [
        f"{M0_HEADER}\n",
        (
            "Unified section-2.5 encoding (`src/traj/encode.py`) re-verifying step3's own "
            "DP+SED / LSQ-uniform byte counts, dt=1 s only, same 15-track/seed=42 synthetic "
            "generator and the same reachability criterion as `step3_decisive.py` (honest "
            "error against the true noise-free curve on a dense grid, `<= target_tol` -- "
            "UNCHANGED from step3, see docs/decisions/ADR-0021). Step7's certificates "
            "(`certify_polyline`/`certify_spline`, section 2.4 default, ADR-0018) are computed "
            "once per reachable cell's CHOSEN representation, as a separate, additional "
            "correctness check (`eps_A <= target_tol`), not a selection criterion.\n"
        ),
        f"{n_tracks} tracks, seed={SEED}, dt=1 s. Full grid elapsed: {elapsed:.1f}s.\n",
        "### Reachability (step3 criterion, unchanged) + certified correctness (ADR-0021)\n",
    ]
    header = "| sigma \\ tol | " + " | ".join(f"{t:g}" for t in tol_list) + " |"

    for method_key, title in (("dp_sed", "DP+SED"), ("spline_uniform", "LSQ spline (uniform)")):
        lines.append(f"#### {title}: bytes (n parameters), reachable fraction; certified `eps_A<=tol` fraction")
        lines.append(header)
        lines.append("|" + "---|" * (len(tol_list) + 1))
        for sigma in sigma_list:
            row = [f"{sigma:g}"]
            for tol in tol_list:
                bytes_str, eps_str = _summarize_method(cells[(sigma, tol)][method_key])
                row.append(f"{bytes_str}; cert {eps_str}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("### Segment-format overhead (synthetic, no real tracks)\n")
    lines.append(
        "Fixed 200-vertex payload cut into 1/2/5/10/20 line segments sharing boundary "
        "vertices (2.5's dedup rule) -- only the per-segment header (`[type][count varint]`) "
        "cost varies; the point payload itself is identical across rows.\n"
    )
    lines.append("| segments | pre-zlib bytes | post-zlib bytes | header bytes | header fraction (pre-zlib) |")
    lines.append("|---|---|---|---|---|")
    for row in overhead_rows:
        lines.append(
            f"| {row['n_segments']} | {row['pre_zlib_bytes']} | {row['post_zlib_bytes']} | "
            f"{row['header_bytes']} | {100.0 * row['header_fraction']:.2f}% |"
        )
    lines.append("")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="budget run (3 tracks), doesn't write to .md")
    parser.add_argument("--n-tracks", type=int, default=None, help="override N_TRACKS for the full run")
    args = parser.parse_args()

    overhead_rows = measure_segment_overhead()

    if args.pilot:
        print(f"PILOT: {N_TRACKS_PILOT} tracks, sigma={SIGMA_LIST}, tol={TOL_LIST}, dt={DT}")
        t0 = time.time()
        cells = run_grid(N_TRACKS_PILOT, SIGMA_LIST, TOL_LIST)
        elapsed = time.time() - t0
        n_cells_pilot = N_TRACKS_PILOT * len(SIGMA_LIST) * len(TOL_LIST)
        n_cells_full = N_TRACKS * len(SIGMA_LIST) * len(TOL_LIST)
        extrapolated = elapsed * (n_cells_full / n_cells_pilot)
        print(f"\npilot took {elapsed:.1f}s ({n_cells_pilot} (track x sigma x tol) combinations)")
        print(f"extrapolated to the full grid ({n_cells_full} combinations): ~{extrapolated:.0f}s (~{extrapolated / 60:.1f} min)")
        print(format_m0(cells, SIGMA_LIST, TOL_LIST, N_TRACKS_PILOT, overhead_rows, elapsed))
        return

    n_tracks = args.n_tracks if args.n_tracks is not None else N_TRACKS
    print(f"FULL RUN: {n_tracks} tracks, sigma={SIGMA_LIST}, tol={TOL_LIST}, dt={DT}")
    t0 = time.time()
    cells = run_grid(n_tracks, SIGMA_LIST, TOL_LIST)
    elapsed = time.time() - t0
    print(f"\nfull run took {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    body = format_m0(cells, SIGMA_LIST, TOL_LIST, n_tracks, overhead_rows, elapsed)
    if n_tracks != N_TRACKS:
        body += (
            f"\n**Time budget constraint:** N_TRACKS was reduced from {N_TRACKS} to "
            f"{n_tracks} based on the pilot/full-run results. The full run took "
            f"{elapsed:.1f}s ({elapsed / 60:.1f} min) on {n_tracks} tracks.\n"
        )
    upsert_section(OUT_MD, M0_HEADER, body)
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
