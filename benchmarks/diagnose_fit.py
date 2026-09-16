"""Spline fitting diagnostics: why the spline's recall@10 is lower than DP's.

Consolidates diagnostic scripts from a previous session:
- §1: why spline_same and spline_arclen give the same recall (0.707) --
  a coincidence of point counts, or an effect of the Frechet metric on
  different sampling densities of the same curve.
- §2: dense check of spline error between timestamps (20 points per
  interval) -- arc-length ratio, max deviation from the raw polyline,
  fraction of tracks violating tol/2tol/5tol/10tol.
- §3: same for the DP polyline (structural check of raw-point deviation).
- §4 (new): max dt and dt spread (std, IQR) within a track, correlation
  with the causes already found (max_gap, min_dt, span, n_raw).

Results reproduce benchmarks/results/step0_diagnostics.md; the script is
meant to be re-run after fixes (clean.py, spline.py) to check the effect.

Run: venv/bin/python benchmarks/diagnose_fit.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.interpolate import splev
from shapely.geometry import LineString, Point

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from traj import spline as splmod  # noqa: E402
from traj.io import load_tracks  # noqa: E402
from traj.simplify import simplify_with_indices  # noqa: E402

from step0 import TOL, SEED, N_QUERIES, build_representation, full_scan_top10, recall_at_k  # noqa: E402

PTS_PER_INTERVAL = 20  # points per interval between neighboring timestamps


def raw_polyline_len(xy: np.ndarray) -> float:
    return float(np.hypot(*np.diff(xy, axis=0).T).sum())


def check_resampling_identity(tracks, reprs) -> None:
    """§1: sensitivity of recall to the sampling density of a single curve."""
    print("=== point counts for first 5 tracks ===")
    for r in reprs[:5]:
        print(
            r.track_id, "n_dp:", r.n_dp,
            "spline_same:", len(r.spline_same), "spline_arclen:", len(r.spline_arclen),
        )

    same_count = sum(1 for r in reprs if len(r.spline_same) == len(r.spline_arclen))
    print(f"\ntracks where len(spline_same) == len(spline_arclen): {same_count}/{len(reprs)}")

    for r in reprs[:5]:
        same_len = len(r.spline_same) == len(r.spline_arclen)
        identical = same_len and np.allclose(r.spline_same, r.spline_arclen)
        print(r.track_id, "same_len:", same_len, "identical_arrays:", identical)

    rng = np.random.default_rng(SEED)
    query_idx = rng.choice(len(tracks), size=min(N_QUERIES, len(tracks)), replace=False)

    reprs_by_kind = {
        "raw": [(i, r.raw) for i, r in enumerate(reprs)],
        "spline_same": [(i, r.spline_same) for i, r in enumerate(reprs)],
        "spline_arclen": [(i, r.spline_arclen) for i, r in enumerate(reprs)],
    }

    raw_top10 = {}
    for qi in query_idx:
        qi = int(qi)
        got = full_scan_top10(reprs_by_kind["raw"][qi][1], reprs_by_kind["raw"], exclude_idx=qi)
        raw_top10[qi] = {idx for _, idx in got}

    print("\n=== per-query recall: spline_same vs spline_arclen ===")
    recalls_same, recalls_arc, identical_topk_count = [], [], 0
    for qi in query_idx:
        qi = int(qi)
        got_same = full_scan_top10(
            reprs_by_kind["spline_same"][qi][1], reprs_by_kind["spline_same"], exclude_idx=qi
        )
        got_arc = full_scan_top10(
            reprs_by_kind["spline_arclen"][qi][1], reprs_by_kind["spline_arclen"], exclude_idx=qi
        )
        rec_same = recall_at_k(raw_top10[qi], got_same)
        rec_arc = recall_at_k(raw_top10[qi], got_arc)
        recalls_same.append(rec_same)
        recalls_arc.append(rec_arc)
        ids_same = tuple(sorted(idx for _, idx in got_same))
        ids_arc = tuple(sorted(idx for _, idx in got_arc))
        identical_topk = ids_same == ids_arc
        if identical_topk:
            identical_topk_count += 1
        print(
            f"query {qi}: recall_same={rec_same:.2f} recall_arc={rec_arc:.2f} "
            f"identical_top10_ids={identical_topk}"
        )

    print(f"\nmean recall_same={np.mean(recalls_same):.4f} mean recall_arc={np.mean(recalls_arc):.4f}")
    print(f"queries with identical top10 id sets between same/arclen: {identical_topk_count}/{len(query_idx)}")


def check_dense_error(tracks) -> tuple[list[dict], list[dict]]:
    """§2/§3/§4: dense error of the spline between timestamps, DP error, dt analysis."""
    results_spline = []
    results_dp = []

    for tr in tracks:
        sp = splmod.fit(tr, tol=TOL)
        t = tr.t
        span = sp.t_max - sp.t_min
        u_raw = (t - sp.t_min) / span if span > 0 else np.linspace(0.0, 1.0, len(t))

        u_dense_parts = []
        for i in range(len(u_raw) - 1):
            seg = np.linspace(u_raw[i], u_raw[i + 1], PTS_PER_INTERVAL + 1)
            u_dense_parts.append(seg[:-1] if i < len(u_raw) - 2 else seg)
        u_dense = np.concatenate(u_dense_parts)

        xs, ys = splev(u_dense, sp.tck)
        dense_xy = np.column_stack([xs, ys])

        dense_len = raw_polyline_len(dense_xy)
        raw_len = raw_polyline_len(tr.xy)
        ratio = dense_len / raw_len if raw_len > 0 else float("nan")

        raw_line = LineString(tr.xy)
        dists = np.array([raw_line.distance(Point(p)) for p in dense_xy])
        max_dist = float(dists.max())

        raw_diffs = np.hypot(*np.diff(tr.xy, axis=0).T)
        max_gap_m = float(raw_diffs.max())
        dt = np.diff(t)
        min_dt = float(dt.min())
        max_dt = float(dt.max())
        std_dt = float(dt.std())
        iqr_dt = float(np.percentile(dt, 75) - np.percentile(dt, 25))
        span_dist = float(np.hypot(*(tr.xy[-1] - tr.xy[0])))

        results_spline.append(
            dict(
                track_id=tr.track_id, ratio=ratio, max_dist=max_dist, n_raw=len(tr.lat),
                max_gap_m=max_gap_m, min_dt=min_dt, max_dt=max_dt, std_dt=std_dt,
                iqr_dt=iqr_dt, span_dist=span_dist, fit_max_error=sp.max_error,
                converged=sp.converged,
            )
        )

        dp_xy, dp_idx = simplify_with_indices(tr, TOL)
        dp_line = LineString(dp_xy)
        dp_dists = np.array([dp_line.distance(Point(p)) for p in tr.xy])
        max_dist_dp = float(dp_dists.max())
        results_dp.append(
            dict(track_id=tr.track_id, max_dist_dp=max_dist_dp, n_raw=len(tr.lat), n_dp=len(dp_xy))
        )

    ratios = np.array([r["ratio"] for r in results_spline])
    max_dists = np.array([r["max_dist"] for r in results_spline])

    print("=== spline: ratio dense_arclen / raw_arclen ===")
    print(
        f"median={np.median(ratios):.4f} p90={np.percentile(ratios, 90):.4f} "
        f"max={ratios.max():.4f} min={ratios.min():.4f}"
    )

    print("\n=== spline: max distance dense-spline -> raw polyline (m) ===")
    print(
        f"median={np.median(max_dists):.3f} p75={np.percentile(max_dists, 75):.3f} "
        f"p90={np.percentile(max_dists, 90):.3f} p99={np.percentile(max_dists, 99):.3f} "
        f"max={max_dists.max():.3f}"
    )

    for mult in (1, 2, 5, 10):
        n_exceed = int((max_dists > mult * TOL).sum())
        print(f"tracks where max_dist > {mult}*tol({TOL}m): {n_exceed}/{len(results_spline)}")

    max_gaps = np.array([r["max_gap_m"] for r in results_spline])
    min_dts = np.array([r["min_dt"] for r in results_spline])
    max_dts = np.array([r["max_dt"] for r in results_spline])
    std_dts = np.array([r["std_dt"] for r in results_spline])
    iqr_dts = np.array([r["iqr_dt"] for r in results_spline])
    span_dists = np.array([r["span_dist"] for r in results_spline])
    n_raws = np.array([r["n_raw"] for r in results_spline])

    print("\n=== correlations (log10 max_dist vs factor) ===")
    log_md = np.log10(max_dists + 1e-6)
    for name, arr in [
        ("max_gap", max_gaps), ("min_dt", min_dts), ("max_dt", max_dts),
        ("std_dt", std_dts), ("iqr_dt", iqr_dts), ("span_start_end", span_dists),
        ("log n_raw", np.log10(n_raws)),
    ]:
        print(f"corr(log max_dist, {name}):", np.corrcoef(log_md, arr)[0, 1])

    print("\n=== dt: max and spread within a track (across all 200 tracks) ===")
    print(f"max_dt: median={np.median(max_dts):.2f}s p90={np.percentile(max_dts, 90):.2f}s max={max_dts.max():.2f}s")
    print(f"std_dt: median={np.median(std_dts):.2f}s p90={np.percentile(std_dts, 90):.2f}s max={std_dts.max():.2f}s")
    print(f"iqr_dt: median={np.median(iqr_dts):.2f}s p90={np.percentile(iqr_dts, 90):.2f}s max={iqr_dts.max():.2f}s")

    order = np.argsort(-max_dists)[:5]
    print("\n=== 5 worst tracks by max_dist (spline vs raw polyline) ===")
    for i in order:
        r = results_spline[i]
        print(
            f"{r['track_id']}: max_dist={r['max_dist']:.2f}m ratio={r['ratio']:.3f} n_raw={r['n_raw']} "
            f"max_gap={r['max_gap_m']:.1f}m min_dt={r['min_dt']:.2f}s max_dt={r['max_dt']:.2f}s "
            f"std_dt={r['std_dt']:.2f}s span_start_end={r['span_dist']:.1f}m "
            f"fit_max_error(at t_i)={r['fit_max_error']:.2f}m converged={r['converged']}"
        )

    max_dists_dp = np.array([r["max_dist_dp"] for r in results_dp])
    print("\n=== DP: max distance raw points -> DP polyline (m) ===")
    print(f"median={np.median(max_dists_dp):.3f} p90={np.percentile(max_dists_dp, 90):.3f} max={max_dists_dp.max():.3f}")
    n_exceed_tol_dp = int((max_dists_dp > TOL).sum())
    print(f"tracks where max_dist_dp > tol({TOL}m): {n_exceed_tol_dp}/{len(results_dp)}")

    order_dp = np.argsort(-max_dists_dp)[:5]
    print("\n=== 5 worst tracks by max_dist (DP) ===")
    for i in order_dp:
        r = results_dp[i]
        print(f"{r['track_id']}: max_dist_dp={r['max_dist_dp']:.3f}m n_raw={r['n_raw']} n_dp={r['n_dp']}")

    return results_spline, results_dp


def main() -> None:
    tracks = load_tracks(n=200, seed=SEED)
    reprs = [build_representation(tr, TOL) for tr in tracks]

    print("#" * 70)
    print("# §1: spline_same vs spline_arclen -- effect of sampling density")
    print("#" * 70)
    check_resampling_identity(tracks, reprs)

    print("\n" + "#" * 70)
    print("# §2-4: dense error of spline/DP between timestamps + dt analysis")
    print("#" * 70)
    check_dense_error(tracks)


if __name__ == "__main__":
    main()
