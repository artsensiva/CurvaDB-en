"""Step 2, item 3: FITPACK suboptimality -- how many control points
`splprep` (via `traj.spline.fit`) uses compared to the minimal number of
UNIFORM knots at which `make_lsq_spline` still holds tol (honest dense
error, `traj.spline.dense_max_error`).

5 tracks from the same synthetic data as step2_crossover.py (shared
generator `make_synthetic_track`, the same first 5 seeds from
master-seed=42), sigma=0.1, tol=1.0 -- a typical mid-range cell of the
step2 grid (neither the easiest nor the hardest).

Run: venv/bin/python benchmarks/step2_fitpack.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.interpolate import make_lsq_spline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from traj.spline import DEGREE, dense_max_error  # noqa: E402
from traj.spline import _param_u  # noqa: E402
from traj.spline import fit as spline_fit  # noqa: E402
from _report_utils import upsert_section  # noqa: E402
from step2_crossover import SEED, N_TRACKS, make_synthetic_track  # noqa: E402

N_FITPACK_TRACKS = 5
SIGMA = 0.1
TOL = 1.0
MAX_N_INTERIOR = 300
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_MD = os.path.join(RESULTS_DIR, "step2.md")
SECTION_HEADER = "## 3. FITPACK suboptimality: splprep vs minimal uniform knots"


def _uniform_clamped_knots(u_min: float, u_max: float, n_interior: int, k: int) -> np.ndarray:
    interior = np.linspace(u_min, u_max, n_interior + 2)[1:-1] if n_interior > 0 else np.array([])
    return np.concatenate([[u_min] * (k + 1), interior, [u_max] * (k + 1)])


def min_uniform_knots(t: np.ndarray, xy: np.ndarray, tol: float, mode: str = "time", k: int = DEGREE):
    """Grows the number of interior uniform knots n_int = 0, 1, 2, ...
    until the honest dense error (dense_max_error) becomes <= tol.
    Returns (n_control_points, n_interior, dense_error) or
    (None, None, None) if it didn't fit within MAX_N_INTERIOR."""
    u = _param_u(t, xy, mode)
    u_min, u_max = float(u.min()), float(u.max())
    for n_int in range(0, MAX_N_INTERIOR + 1):
        knots = _uniform_clamped_knots(u_min, u_max, n_int, k)
        if len(knots) - k - 1 > len(u):
            break  # more parameters than points -- only gets worse-conditioned from here
        try:
            bx = make_lsq_spline(u, xy[:, 0], knots, k=k)
            by = make_lsq_spline(u, xy[:, 1], knots, k=k)
        except Exception:
            continue
        tck = (bx.t, [bx.c, by.c], k)
        err = dense_max_error(t, xy, tck, mode=mode)
        if err <= tol:
            return n_int + k + 1, n_int, err
    return None, None, None


def main() -> None:
    rng_master = np.random.default_rng(SEED)
    geom_seeds = [int(s) for s in rng_master.integers(0, 1_000_000, size=N_TRACKS)][:N_FITPACK_TRACKS]

    rows = []
    for gseed in geom_seeds:
        track, _true_xy, _duration = make_synthetic_track(gseed, SIGMA)
        sp = spline_fit(track, tol=TOL, parametrization="time")
        n_cp_lsq, n_int, err_lsq = min_uniform_knots(track.t, track.xy, TOL, mode="time")
        rows.append(
            {
                "seed": gseed,
                "n_points": len(track.t),
                "n_cp_splprep": sp.n_control_points,
                "splprep_err": sp.max_error,
                "n_cp_lsq": n_cp_lsq,
                "n_interior_lsq": n_int,
                "lsq_err": err_lsq,
            }
        )
        print(
            f"seed={gseed}: n_points={len(track.t)} splprep n_cp={sp.n_control_points} "
            f"(err={sp.max_error:.3f}) | lsq minimal n_cp={n_cp_lsq} (n_int={n_int}, err={err_lsq})"
        )

    lines = [
        f"{SECTION_HEADER}\n",
        (
            f"{N_FITPACK_TRACKS} tracks from step2_crossover.py's synthetic data "
            f"(the same first {N_FITPACK_TRACKS} seeds from master-seed={SEED}), sigma={SIGMA:g}m, "
            f"tol={TOL:g}m, time parametrization. `splprep` -- via "
            "`traj.spline.fit` (honest dense control, adaptive densify + "
            "growing `s`). `make_lsq_spline` -- UNIFORM knots, the number of "
            "interior knots grown from 0 until it first passes the same "
            "honest dense check (`traj.spline.dense_max_error`).\n"
        ),
        "| seed | n points | n control points (splprep) | n control points (lsq, uniform) | ratio |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        ratio = (
            f"{r['n_cp_splprep'] / r['n_cp_lsq']:.2f}x" if r["n_cp_lsq"] else "n/a (didn't fit)"
        )
        lines.append(
            f"| {r['seed']} | {r['n_points']} | {r['n_cp_splprep']} | "
            f"{r['n_cp_lsq'] if r['n_cp_lsq'] else 'n/a'} | {ratio} |"
        )
    lines.append("")

    valid = [r for r in rows if r["n_cp_lsq"]]
    if valid:
        mean_splprep = np.mean([r["n_cp_splprep"] for r in valid])
        mean_lsq = np.mean([r["n_cp_lsq"] for r in valid])
        lines.append(
            f"On average across {len(valid)}/{N_FITPACK_TRACKS} tracks: splprep "
            f"uses {mean_splprep:.1f} control points, minimal uniform "
            f"knots -- {mean_lsq:.1f} ({mean_splprep / mean_lsq:.2f}x). "
            + (
                "splprep uses noticeably more parameters than needed for the "
                "same guarantee -- part of the spline's loss to DP+SED in "
                "step1/step2 may be a FITPACK artifact, not a fundamental "
                "property of cubic B-splines."
                if mean_splprep > mean_lsq * 1.1
                else "splprep is no worse than uniform knots on these tracks -- "
                "the spline's loss to DP+SED is not explained by FITPACK suboptimality."
            )
            + "\n"
        )

    body = "\n".join(lines) + "\n"
    upsert_section(OUT_MD, SECTION_HEADER, body)
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
