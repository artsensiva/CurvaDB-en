"""Step 0: an honest comparison of the raw polyline, the DP polyline, and
the spline.

Question: does the cubic B-spline beat the Douglas-Peucker-simplified
polyline on at least one axis (bytes, latency, recall, velocity/
acceleration error). A negative result is a result too.

Run: venv/bin/python benchmarks/step0.py
"""

from __future__ import annotations

import bisect
import csv
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splev
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from traj import frechet, simplify, spline  # noqa: E402
from traj.io import load_tracks  # noqa: E402

TOL = 10.0
SEED = 42
N_QUERIES = 30
K = 10
MAX_ARC_POINTS = 3000  # guard against rare tracks with anomalous GPS jumps
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


@dataclass
class TrackRepr:
    track_id: str
    raw: np.ndarray
    dp: np.ndarray
    dp_t: np.ndarray
    spline_same: np.ndarray
    spline_arclen: np.ndarray
    raw_bytes: int
    dp_bytes: int
    spline_bytes: int
    n_dp: int
    n_spline_cp: int
    spline_converged: bool
    spline_max_error: float
    v_err_spline: float
    a_err_spline: float
    v_err_dp: float
    a_err_dp: float


def _spline_storage_bytes(sp: spline.SplineFit) -> int:
    """Spline footprint: knots t + 2 coefficient arrays (x, y), float64."""
    knots = len(sp.tck[0])
    coeffs = len(sp.tck[1][0])
    return (knots + 2 * coeffs) * 8


def _arclength_resample(sp: spline.SplineFit, tol: float, n_fine: int = 4000) -> np.ndarray:
    """Discretizes the spline with a step of ~tol along arc length."""
    u_fine = np.linspace(0.0, 1.0, n_fine)
    xf, yf = splev(u_fine, sp.tck)
    seg = np.hypot(np.diff(xf), np.diff(yf))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    n_pts = max(2, min(int(round(total / tol)) + 1, MAX_ARC_POINTS))
    target = np.linspace(0.0, total, n_pts)
    u_t = np.interp(target, cum, u_fine)
    xt, yt = splev(u_t, sp.tck)
    return np.column_stack([xt, yt])


def _smoothed_reference_derivatives(xy: np.ndarray, t: np.ndarray):
    """Velocity/acceleration via finite differences on the Savitzky-Golay-
    smoothed raw track -- a reference for "true" kinematics without GPS
    noise."""
    n = len(t)
    window = min(11, n - (1 - n % 2))
    smoothed = np.column_stack(
        [
            savgol_filter(xy[:, 0], window, 2, mode="interp"),
            savgol_filter(xy[:, 1], window, 2, mode="interp"),
        ]
    )
    v = np.gradient(smoothed, t, axis=0)
    a = np.gradient(v, t, axis=0)
    return v, a


def _dp_derivatives(dp_xy: np.ndarray, dp_t: np.ndarray, t: np.ndarray):
    """Piecewise-constant velocity of the DP polyline; acceleration is
    identically 0 between vertices (the nature of a polyline)."""
    seg = np.searchsorted(dp_t, t, side="right") - 1
    seg = np.clip(seg, 0, len(dp_t) - 2)
    dt = dp_t[seg + 1] - dp_t[seg]
    dxy = dp_xy[seg + 1] - dp_xy[seg]
    v = dxy / dt[:, None]
    a = np.zeros_like(v)
    return v, a


def build_representation(track, tol: float) -> TrackRepr:
    raw = track.xy
    dp_xy, dp_idx = simplify.simplify_with_indices(track, tol)
    dp_t = track.t[dp_idx]
    sp = spline.fit(track, tol)
    spline_same = spline.sample(sp, len(dp_xy))
    spline_arclen = _arclength_resample(sp, tol)

    ref_v, ref_a = _smoothed_reference_derivatives(raw, track.t)
    sp_v, sp_a = spline.derivatives(sp, track.t)
    dp_v, dp_a = _dp_derivatives(dp_xy, dp_t, track.t)

    return TrackRepr(
        track_id=track.track_id,
        raw=raw,
        dp=dp_xy,
        dp_t=dp_t,
        spline_same=spline_same,
        spline_arclen=spline_arclen,
        raw_bytes=raw.size * 8,
        dp_bytes=dp_xy.size * 8,
        spline_bytes=_spline_storage_bytes(sp),
        n_dp=len(dp_xy),
        n_spline_cp=sp.n_control_points,
        spline_converged=sp.converged,
        spline_max_error=sp.max_error,
        v_err_spline=float(np.median(np.linalg.norm(sp_v - ref_v, axis=1))),
        a_err_spline=float(np.median(np.linalg.norm(sp_a - ref_a, axis=1))),
        v_err_dp=float(np.median(np.linalg.norm(dp_v - ref_v, axis=1))),
        a_err_dp=float(np.median(np.linalg.norm(dp_a - ref_a, axis=1))),
    )


def full_scan_top10(query_xy: np.ndarray, corpus, exclude_idx: int, k: int = K):
    """Top-k by Frechet distance via full corpus scan; distance_within
    with the current worst of the top-k as the threshold just speeds up
    discarding hopeless candidates -- every candidate is still visited."""
    best: list[tuple[float, int]] = []
    for idx, cand_xy in corpus:
        if idx == exclude_idx:
            continue
        threshold = best[-1][0] if len(best) == k else float("inf")
        val, exact = frechet.distance_within(query_xy, cand_xy, threshold)
        if not exact:
            continue
        if len(best) < k:
            bisect.insort(best, (val, idx))
        elif val < best[-1][0]:
            bisect.insort(best, (val, idx))
            best.pop()
    return best


def recall_at_k(reference_ids: set, got, k: int = K) -> float:
    got_ids = {idx for _, idx in got}
    return len(reference_ids & got_ids) / k


def run(
    n_tracks: int,
    tol: float = TOL,
    seed: int = SEED,
    n_queries: int = N_QUERIES,
    out_md: str | None = None,
    out_csv: str | None = None,
) -> dict:
    t0 = time.time()
    tracks = load_tracks(n=n_tracks, seed=seed)
    print(f"[n={n_tracks}] loaded {len(tracks)} tracks in {time.time() - t0:.1f}s")

    t0 = time.time()
    reprs = [build_representation(tr, tol) for tr in tracks]
    print(f"[n={n_tracks}] built DP/spline representations in {time.time() - t0:.1f}s")

    rng = np.random.default_rng(seed)
    query_idx = rng.choice(len(tracks), size=min(n_queries, len(tracks)), replace=False)

    reprs_by_kind = {
        "raw": [(i, r.raw) for i, r in enumerate(reprs)],
        "dp": [(i, r.dp) for i, r in enumerate(reprs)],
        "spline_same": [(i, r.spline_same) for i, r in enumerate(reprs)],
        "spline_arclen": [(i, r.spline_arclen) for i, r in enumerate(reprs)],
    }

    latencies: dict[str, list] = {k: [] for k in reprs_by_kind}
    recalls: dict[str, list] = {k: [] for k in reprs_by_kind if k != "raw"}
    raw_top10_cache: dict[int, set] = {}

    for kind, corpus in reprs_by_kind.items():
        t_start = time.time()
        for qi in query_idx:
            qi = int(qi)
            query_xy = corpus[qi][1]
            t_q = time.time()
            got = full_scan_top10(query_xy, corpus, exclude_idx=qi)
            latencies[kind].append(time.time() - t_q)
            if kind == "raw":
                raw_top10_cache[qi] = {idx for _, idx in got}
            else:
                recalls[kind].append(recall_at_k(raw_top10_cache[qi], got))
        print(
            f"[n={n_tracks}] {kind}: full scan of {len(query_idx)} queries "
            f"in {time.time() - t_start:.1f}s"
        )

    def pct_ms(xs, p):
        return float(np.percentile(xs, p) * 1000.0)

    stats = {
        "n_tracks": len(tracks),
        "tol": tol,
        "lengths": [len(tr.lat) for tr in tracks],
        "n_dp_mean": float(np.mean([r.n_dp for r in reprs])),
        "n_spline_cp_mean": float(np.mean([r.n_spline_cp for r in reprs])),
        "n_nonconverged": int(sum(1 for r in reprs if not r.spline_converged)),
        "bytes_raw": float(np.mean([r.raw_bytes for r in reprs])),
        "bytes_dp": float(np.mean([r.dp_bytes for r in reprs])),
        "bytes_spline": float(np.mean([r.spline_bytes for r in reprs])),
        "v_err_spline": float(np.mean([r.v_err_spline for r in reprs])),
        "a_err_spline": float(np.mean([r.a_err_spline for r in reprs])),
        "v_err_dp": float(np.mean([r.v_err_dp for r in reprs])),
        "a_err_dp": float(np.mean([r.a_err_dp for r in reprs])),
        "latency_p50": {k: pct_ms(v, 50) for k, v in latencies.items()},
        "latency_p95": {k: pct_ms(v, 95) for k, v in latencies.items()},
        "recall_at_10": {k: float(np.mean(v)) for k, v in recalls.items()},
    }

    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "track_id", "n_raw", "n_dp", "n_spline_cp", "spline_converged",
                    "spline_max_error_m", "raw_bytes", "dp_bytes", "spline_bytes",
                    "v_err_spline_mps", "a_err_spline_mps2", "v_err_dp_mps", "a_err_dp_mps2",
                ]
            )
            for tr, r in zip(tracks, reprs):
                w.writerow(
                    [
                        r.track_id, len(tr.lat), r.n_dp, r.n_spline_cp, r.spline_converged,
                        round(r.spline_max_error, 3), r.raw_bytes, r.dp_bytes, r.spline_bytes,
                        round(r.v_err_spline, 4), round(r.a_err_spline, 4),
                        round(r.v_err_dp, 4), round(r.a_err_dp, 4),
                    ]
                )

    if out_md:
        _write_md(out_md, stats)

    return stats


def _write_md(path: str, s: dict) -> None:
    lengths = s["lengths"]
    lines = [
        "# Step 0: spline vs DP polyline vs raw polyline\n",
        f"Tracks: {s['n_tracks']}, tol = {s['tol']} m.\n",
        (
            f"Track length (points): min={min(lengths)}, "
            f"median={int(np.median(lengths))}, max={max(lengths)}, "
            f"mean={np.mean(lengths):.0f}.\n"
        ),
        (
            f"Mean number of DP polyline vertices: {s['n_dp_mean']:.1f}. "
            f"Mean number of spline control points: {s['n_spline_cp_mean']:.1f}. "
            f"Non-converged fits: {s['n_nonconverged']}/{s['n_tracks']}.\n"
        ),
        "\n## Bytes per track (float64)\n",
        "| Representation | Bytes/track |",
        "|---|---|",
        f"| raw | {s['bytes_raw']:.0f} |",
        f"| DP polyline | {s['bytes_dp']:.0f} |",
        f"| spline (knots + coefficients) | {s['bytes_spline']:.0f} |",
        "\n## Top-10 latency, full scan, 30 queries (ms)\n",
        "| Representation | p50 | p95 |",
        "|---|---|---|",
    ]
    for kind in ["raw", "dp", "spline_same", "spline_arclen"]:
        lines.append(f"| {kind} | {s['latency_p50'][kind]:.2f} | {s['latency_p95'][kind]:.2f} |")

    lines += [
        "\n## Recall@10 against Frechet distance on raw tracks\n",
        "| Representation | Recall@10 |",
        "|---|---|",
    ]
    for kind in ["dp", "spline_same", "spline_arclen"]:
        lines.append(f"| {kind} | {s['recall_at_10'][kind]:.3f} |")

    lines += [
        "\n## Velocity/acceleration error against finite differences "
        "on the smoothed raw track\n",
        "| Representation | Velocity, m/s (median) | Acceleration, m/s² (median) |",
        "|---|---|---|",
        f"| spline | {s['v_err_spline']:.3f} | {s['a_err_spline']:.3f} |",
        f"| DP polyline | {s['v_err_dp']:.3f} | {s['a_err_dp']:.3f} |",
        (
            "\nNote: the DP polyline is piecewise-linear, so its "
            "acceleration is identically 0 between vertices -- the DP "
            "acceleration error shows how large the real acceleration is "
            "that a polyline fundamentally cannot represent.\n"
        ),
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run(n_tracks=50, tol=TOL, seed=SEED, n_queries=N_QUERIES)
    run(
        n_tracks=200,
        tol=TOL,
        seed=SEED,
        n_queries=N_QUERIES,
        out_md=os.path.join(RESULTS_DIR, "step0.md"),
        out_csv=os.path.join(RESULTS_DIR, "step0.csv"),
    )
