"""Step7 M1.3 (docs/reviews/step7_M1_2.md finding 4): verify the backtracking
hypothesis for the eps_A/LB tail's 5 worst polyline tracks, instead of asserting
"LB is loose" without checking. k* is the piece that actually DETERMINES the
track's eps_A (argmax_k cert_k -- certify_polyline's own eps_A = max_k cert_k, so
this is the piece whose cost eps_A literally equals, not a noisy ratio-based
guess: an early ratio-based selection was tried first and found to be dominated
by degenerate 2-point pieces, where both cert_k and LB_hausdorff_k are ~0 and
their ratio is meaningless noise).

For that piece:
  - cert_k = distance_upper(piece, segment, tol=S1_ETA) + margins.
  - lb_hausdorff_local = hausdorff_lower_bound(piece, segment) (this piece's OWN
    local Hausdorff, not the whole-track one).
  - LB_back = max_{i<j, s_i>s_j} (s_i - s_j) / 2, over projections s_i of the
    piece's points onto the piece's own chord -- a Frechet-specific lower bound:
    a monotone (index-order-preserving) coupling between the piece and its chord
    must pay at least (s_i-s_j)/2 leash wherever the projection goes backward.
  - Hypothesis check: cert_k <= max(lb_hausdorff_local, LB_back) * 1.05.
  - Fraction of consecutive points with decreasing projection: secondary diagnostic.
  - eps_A (whole polyline) vs. its S1 mpmath reference (dps=40, tol=1e-6): is the
    certificate tight against ground truth?
  - whole-track eps_A / global Hausdorff LB (the original ratio being explained).

Run: venv/bin/python benchmarks/step7_m13_tail.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests", "traj"))
sys.path.insert(0, os.path.dirname(__file__))

from traj.certify import _bbox_diagonal, _with_margins, certify_polyline, hausdorff_lower_bound  # noqa: E402
from traj.clean import load_clean_tracks  # noqa: E402
from traj.frechet_cont import distance_upper  # noqa: E402
from traj.simplify import simplify_sed_with_indices  # noqa: E402
from _frechet_cont_mpmath import distance_mp  # noqa: E402
from step7_certify import DP_SED_TOL, N_TRACKS, S1_ETA, S1_REF_DPS, S1_REF_TOL, SEED  # noqa: E402

WORST_TRACKS = [472, 475, 196, 476, 482]  # already identified by whole-track eps_A/LB ratio


def _piece_projection(piece: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    length = float(np.hypot(*d))
    if length < 1e-12:
        return np.zeros(len(piece))
    return (piece - a) @ d / length


def _lb_back(s: np.ndarray) -> float:
    """max_{i<j, s_i>s_j} (s_i - s_j) / 2."""
    n = len(s)
    best = 0.0
    running_max = s[0]
    for j in range(1, n):
        if running_max > s[j]:
            best = max(best, (running_max - s[j]) / 2.0)
        running_max = max(running_max, s[j])
    return best


def analyze_track(idx: int, tr) -> dict:
    _, kept = simplify_sed_with_indices(tr, tol=DP_SED_TOL)
    best_cert, best_k = -1.0, None
    for k in range(len(kept) - 1):
        i0, i1 = int(kept[k]), int(kept[k + 1])
        piece = tr.xy[i0 : i1 + 1]
        d_upper = distance_upper(piece, tr.xy[[i0, i1]], tol=S1_ETA)
        cert_k = _with_margins(d_upper, piece[0], _bbox_diagonal(piece))
        if cert_k > best_cert:
            best_cert, best_k = cert_k, k

    i0, i1 = int(kept[best_k]), int(kept[best_k + 1])
    piece = tr.xy[i0 : i1 + 1]
    a, b = tr.xy[i0], tr.xy[i1]
    lb_hausdorff_local = hausdorff_lower_bound(piece, tr.xy[[i0, i1]])
    s = _piece_projection(piece, a, b)
    lb_back = _lb_back(s)
    frac_decreasing = float(np.mean(np.diff(s) < 0)) if len(s) > 1 else 0.0

    eps_A = certify_polyline(tr.xy, kept, eta=S1_ETA)
    global_lb = hausdorff_lower_bound(tr.xy, tr.xy[kept])
    worst_ref = 0.0
    for k in range(len(kept) - 1):
        i0k, i1k = int(kept[k]), int(kept[k + 1])
        piece_k = tr.xy[i0k : i1k + 1]
        segment_k = tr.xy[[i0k, i1k]]
        ref = float(distance_mp(piece_k.tolist(), segment_k.tolist(), tol=S1_REF_TOL, dps=S1_REF_DPS))
        worst_ref = max(worst_ref, ref)

    bound = max(lb_hausdorff_local, lb_back) * 1.05
    confirmed = best_cert <= bound

    return {
        "idx": idx, "k_star": best_k, "i0": i0, "i1": i1, "n_points": i1 - i0 + 1,
        "cert_k_star": best_cert, "lb_hausdorff_local": lb_hausdorff_local, "lb_back": lb_back,
        "frac_decreasing": frac_decreasing, "bound_1_05": bound, "confirmed": confirmed,
        "eps_A": eps_A, "mpmath_reference": worst_ref, "global_lb": global_lb,
        "global_ratio": eps_A / global_lb,
    }


def main() -> None:
    tracks, _ = load_clean_tracks(n=N_TRACKS, seed=SEED)
    for idx in WORST_TRACKS:
        r = analyze_track(idx, tracks[idx])
        print(
            f"idx={idx} global_ratio={r['global_ratio']:.3f} (eps_A={r['eps_A']:.4f} "
            f"global_LB={r['global_lb']:.4f}) | mpmath_ref={r['mpmath_reference']:.4f} "
            f"eps_A==mpmath: {abs(r['eps_A'] - r['mpmath_reference']) < 1e-6} | "
            f"k*=[{r['i0']}:{r['i1']}] n_points={r['n_points']} cert_k*={r['cert_k_star']:.4f} "
            f"lb_hausdorff_local={r['lb_hausdorff_local']:.4f} lb_back={r['lb_back']:.4f} "
            f"frac_decreasing={r['frac_decreasing']:.3f} bound_1.05={r['bound_1_05']:.4f} "
            f"confirmed={r['confirmed']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
