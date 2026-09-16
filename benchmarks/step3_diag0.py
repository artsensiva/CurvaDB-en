"""Step3, item 0: sanity-checking step2's accounting -- where the mean n
values of exactly 200.0 and 600.0 come from.

Reproduces EXACTLY the same procedure as step2_crossover.evaluate_cell/
search_min_params for the spline_time method (a log grid of
N_INTERNAL_GRID internal_tol values relative to target_tol, honest error
-- true_curve_error against the ground-truth curve), but additionally
prints, per track, the WHOLE internal_tol -> (n, err) grid, not just the
chosen minimum -- to see where the round mean n comes from.

n = n_knots + 2*n_coeffs (see step2_crossover._build_spline) -- counts
scalars (float64), not bytes directly.

Run: venv/bin/python benchmarks/step3_diag0.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from step2_crossover import (  # noqa: E402
    LENGTH_RANGE_M,
    N_INTERNAL_GRID,
    N_TRACKS,
    PTS_PER_SEC,
    SEED,
    _build_spline,
    make_synthetic_track,
)
from traj.spline import fit as spline_fit  # noqa: E402
from _report_utils import upsert_section  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_MD = os.path.join(RESULTS_DIR, "step3.md")
SECTION_HEADER = "## 0. Sanity-checking step2's accounting -- where n=200.0 and n=600.0 come from"

CELLS = [(0.02, 0.2), (5.0, 20.0)]


def spline_build_with_tck(track, internal_tol, t_dense, true_xy_dense):
    """Like step2_crossover._build_spline(mode='time'), but additionally
    returns n_knots/n_coeffs/sp directly (that function only returns
    n, nbytes, err)."""
    from step2_crossover import SPLINE_SEARCH_MAX_DENSIFY_ROUNDS, SPLINE_SEARCH_MAX_ITER, _spline_time_reconstruct
    from step2_crossover import true_curve_error as _true_err

    sp = spline_fit(
        track,
        tol=internal_tol,
        parametrization="time",
        max_iter=SPLINE_SEARCH_MAX_ITER,
        max_densify_rounds=SPLINE_SEARCH_MAX_DENSIFY_ROUNDS,
    )
    recon = _spline_time_reconstruct(t_dense, sp)
    err = _true_err(true_xy_dense, recon)
    n_knots = len(sp.tck[0])
    n_coeffs = len(sp.tck[1][0])
    n = n_knots + 2 * n_coeffs
    return n, err, n_knots, n_coeffs, sp


def main():
    rng_master = np.random.default_rng(SEED)
    geom_seeds = [int(s) for s in rng_master.integers(0, 1_000_000, size=N_TRACKS)]

    lines = [f"{SECTION_HEADER}\n"]
    lines.append(
        "Reproduces `step2_crossover.evaluate_cell`'s procedure for the "
        "spline_time method verbatim (a log grid of internal_tol with "
        f"{N_INTERNAL_GRID} points relative to target_tol, honest error -- "
        "true_curve_error against the ground-truth curve on a dense grid), "
        "but shows the WHOLE internal_tol grid per track, not just the "
        "chosen minimum n. n = n_knots + 2*n_coeffs (float64 scalars, NOT bytes).\n"
    )

    for sigma, tol in CELLS:
        lines.append(f"### sigma={sigma:g}, tol={tol:g}\n")
        grid = tol * np.logspace(-2, 1, N_INTERNAL_GRID)
        lines.append("internal_tol grid (m): " + ", ".join(f"{g:.4g}" for g in grid) + "\n")
        lines.append("| seed | n_raw | reachable | chosen itol | n (chosen) | n_knots | n_coeffs | err (chosen) | full grid n@itol (err<=tol?) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")

        chosen_n_values = []
        for gseed in geom_seeds:
            track, true_xy, duration = make_synthetic_track(gseed, sigma, length_range_m=LENGTH_RANGE_M)
            dt_dense = 1.0 / PTS_PER_SEC
            t_dense = np.arange(0.0, duration, dt_dense)
            if len(t_dense) == 0 or t_dense[-1] != duration:
                t_dense = np.concatenate([t_dense, [duration]])
            true_xy_dense = true_xy(t_dense)

            grid_results = []
            for itol in grid:
                n, err, n_knots, n_coeffs, sp = spline_build_with_tck(track, float(itol), t_dense, true_xy_dense)
                grid_results.append((float(itol), n, err, n_knots, n_coeffs))

            reachable = grid_results[0][2] <= tol
            if not reachable:
                chosen_n_values.append(None)
                grid_str = "; ".join(f"{n}@{itol:.3g}({'ok' if err <= tol else 'no'})" for itol, n, err, _, _ in grid_results)
                lines.append(f"| {gseed} | {len(track.t)} | no (first grid point fails) | -- | -- | -- | -- | -- | {grid_str} |")
                continue

            best = min((g for g in grid_results if g[2] <= tol), key=lambda g: g[1])
            chosen_n_values.append(best[1])
            grid_str = "; ".join(f"{n}@{itol:.3g}({'ok' if err <= tol else 'no'})" for itol, n, err, _, _ in grid_results)
            lines.append(
                f"| {gseed} | {len(track.t)} | yes | {best[0]:.4g} | {best[1]} | {best[3]} | {best[4]} | "
                f"{best[2]:.4g} | {grid_str} |"
            )

        reach_vals = [v for v in chosen_n_values if v is not None]
        if reach_vals:
            all_equal = len(set(reach_vals)) == 1
            mean_n = float(np.mean(reach_vals))
            lines.append(
                f"\nReachable: {len(reach_vals)}/{len(chosen_n_values)}; "
                f"mean n = {mean_n:.4g}; n across reachable tracks is "
                f"{'ALL EQUAL' if all_equal else 'different'}: {sorted(set(reach_vals))}.\n"
            )
        else:
            lines.append("\nNo track is reachable in this cell.\n")

    body = "\n".join(lines) + "\n"
    upsert_section(OUT_MD, SECTION_HEADER, body)
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
