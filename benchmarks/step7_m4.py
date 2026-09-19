"""Step7 M4 (docs/specs/step7_B_certified_store.md sections 1 and 8): the final
metrics needed for the M4 report and docs/phases/step7_summary.md.

1. The mandatory missing number (spec section 8): the approximate (uncertified)
   method's miss rate (misses / true positives) and false-positive rate (false
   positives / the approximate method's OWN predicted-positive count), per r
   and representation. M3's grid didn't track the two denominators needed
   (total true positives, approximate's own predicted-positive count) --
   added here (traj.query's _new_rep_stats/run_correctness_grid gained
   n_ground_truth_positive/approx_n_predicted_positive) and the full grid is
   recomputed (same deterministic seeds as M3 -- the corpus/query set
   reproduce identically; verified against M3's own saved numbers below).
2. Trade-off curve (polylines only): DP+SED tol in {1,2,5,10,20} m, on a
   100-track/200-query subsample (kept small deliberately, for time budget --
   this is a product-decision exploration, not a full-corpus validation).
   median eps_A, bytes/track (a simple, documented quantize-then-zlib
   estimate -- not a production codec), and the refine rate (1 - S5, measured
   the same post-cheap-filter way as M3) at r=50 and r=200.
3. Per-QUERY latency (not per-pair, unlike M3's dedicated timing sample) at
   r=200: each of {certified, filter_uncompressed, approximate} timed as ONE
   pass over the WHOLE candidate corpus per query (a single perf_counter
   bracket per query, not one per candidate -- avoids per-call timer overhead
   dominating at this many candidates), then median/p95 taken ACROSS queries.

Run: venv/bin/python benchmarks/step7_m4.py
"""

from __future__ import annotations

import os
import pickle
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from traj.certify import certify_polyline  # noqa: E402
from traj.frechet_cont import decide  # noqa: E402
from traj.intervals import cheap_lower_bound, decide_range  # noqa: E402
from traj.io import Track  # noqa: E402
from traj.simplify import simplify_sed_with_indices  # noqa: E402

import step7_query as Q  # noqa: E402

OUT_PKL = "/tmp/m4_report.pkl"
OUT_LOG = "/tmp/m4_report.log"

TOL_VALUES = [1.0, 2.0, 5.0, 10.0, 20.0]
TRADEOFF_N_TRACKS = 100
TRADEOFF_N_QUERIES = 200
TRADEOFF_SEED = 20260919
TRADEOFF_RS = [50.0, 200.0]

PER_QUERY_R = 200.0
QUANTIZE_PRECISION_CM = 1.0  # 1 cm grid -- see quantized_bytes


# --- item 1: recomputed correctness grid (miss/false-positive rates) ---------------


def recompute_correctness_grid():
    tracks, clean_stats = Q.load_clean_tracks(n=Q.N_TRACKS, seed=Q.SEED)
    full_corpus, round_log = Q.assemble_full_corpus(tracks)
    query_indices = Q.select_queries(full_corpus)
    grid = Q.run_correctness_grid(full_corpus, query_indices)
    return full_corpus, query_indices, grid


def approx_error_rates(grid: dict) -> dict:
    """Mandatory correction 1 (docs/reviews-equivalent instruction): both
    fractions, per r/representation --
      miss_rate = approx misses / (all true-positive pairs)
      false_positive_rate = approx false positives / (pairs the approximate
                             method itself predicted positive)
    """
    out = {}
    for r, reps in grid["results"].items():
        out[r] = {}
        for rep, stats in reps.items():
            tp = stats["n_ground_truth_positive"]
            pred_pos = stats["approx_n_predicted_positive"]
            miss_rate = stats["approx_n_miss"] / tp if tp else float("nan")
            fp_rate = stats["approx_n_false_positive"] / pred_pos if pred_pos else float("nan")
            out[r][rep] = {
                "n_ground_truth_positive": tp,
                "approx_n_predicted_positive": pred_pos,
                "approx_n_miss": stats["approx_n_miss"],
                "approx_n_false_positive": stats["approx_n_false_positive"],
                "miss_rate": miss_rate,
                "false_positive_rate": fp_rate,
            }
    return out


# --- item 2: trade-off curve (polylines only) ---------------------------------------


def _polyline_repr(xy: np.ndarray, t: np.ndarray, tol: float) -> tuple[np.ndarray | None, float | None]:
    """ADR-0020: `t` must be the track's REAL timestamps, not a placeholder --
    simplify_sed_with_indices is time-aware (SED = synchronized Euclidean
    distance) and, before ADR-0020's guard existed, a degenerate `t` (this
    function used to pass `np.zeros(len(xy))`) silently broke its
    time-synchronized interpolation instead of raising, causing systematic
    vertex over-retention almost independent of `tol` -- exactly the "size
    barely depends on tol" anomaly this ADR investigates and fixes."""
    track = Track(track_id="q", lat=np.zeros(len(xy)), lon=np.zeros(len(xy)), t=t, xy=xy)
    _, kept_idx = simplify_sed_with_indices(track, tol=tol)
    if len(kept_idx) < 2:
        return None, None
    kept = xy[kept_idx]
    eps = certify_polyline(xy, kept_idx, eta=Q.S1_ETA)
    return kept, eps


def quantized_bytes(kept: np.ndarray, precision_cm: float = QUANTIZE_PRECISION_CM) -> int:
    """A simple, explicit storage estimate: round each coordinate to a
    `precision_cm` grid, store as int32, zlib-compress (level 9). Not a
    production codec (no delta-coding, no header optimization) -- just a
    documented, reproducible number for the tol-vs-size trade-off."""
    q = np.round(kept * (100.0 / precision_cm)).astype(np.int32)
    return len(zlib.compress(q.tobytes(), level=9))


def build_tradeoff_sample(tracks: list, seed: int = TRADEOFF_SEED, n_total: int = TRADEOFF_N_TRACKS, n_duplicates: int = 30) -> list[tuple[np.ndarray, np.ndarray]]:
    """A plain random sample of tracks (GeoLife, different users/times) has
    essentially no pairs within r=50 m of each other at all (M3: >99.9% of
    all pairs are cheap-filter-rejected even WITH near-duplicates injected) --
    without engineered close pairs, S5 at r=50 would be undefined (no
    candidates survive the cheap filter to measure a refine rate on). Mixes
    in `n_duplicates` near-duplicates (the same controlled-translation
    generator as step7_query.py, targeting r=50 or r=200) of OTHER tracks in
    the same sample, so both the near-duplicate and its source are present.

    Returns (xy, t) pairs, not bare xy arrays (ADR-0020): a near-duplicate is
    a spatial perturbation of its source's own points, so it reuses the
    SOURCE's real timestamps -- matching step7_query.py's own convention
    (_append_duplicate stores the source's `t`), not a placeholder."""
    rng = np.random.default_rng(seed)
    n_base = n_total - n_duplicates
    base_idx = rng.choice(len(tracks), size=min(n_base, len(tracks)), replace=False)
    sample: list[tuple[np.ndarray, np.ndarray]] = [(tracks[int(i)].xy, tracks[int(i)].t) for i in base_idx]

    for _ in range(n_duplicates):
        src_xy, src_t = sample[int(rng.integers(0, len(sample)))]
        target_r = float(rng.choice(TRADEOFF_RS))
        target_distance = max(1.0, target_r * float(rng.uniform(0.8, 1.2)))
        dup_xy = Q.make_near_duplicate(rng, src_xy, target_distance)
        sample.append((dup_xy, src_t))

    return sample


def run_tradeoff_curve(xy_t_sample: list[tuple[np.ndarray, np.ndarray]], n_queries: int = TRADEOFF_N_QUERIES, seed: int = TRADEOFF_SEED) -> list[dict]:
    results = []
    for tol in TOL_VALUES:
        entries = []
        for xy, t in xy_t_sample:
            kept, eps = _polyline_repr(xy, t, tol)
            entries.append({"xy": xy, "kept": kept, "eps": eps})

        eps_list = [e["eps"] for e in entries if e["eps"] is not None]
        bytes_list = [quantized_bytes(e["kept"]) for e in entries if e["kept"] is not None]
        median_eps = float(np.median(eps_list)) if eps_list else float("nan")
        median_bytes = float(np.median(bytes_list)) if bytes_list else float("nan")

        rng = np.random.default_rng(seed)
        n_q = min(n_queries, len(entries))
        query_local_idx = rng.choice(len(entries), size=n_q, replace=False)

        after_cheap = {r: 0 for r in TRADEOFF_RS}
        resolved_after_cheap = {r: 0 for r in TRADEOFF_RS}
        for qi in query_local_idx:
            q = entries[int(qi)]
            if q["kept"] is None or q["eps"] is None:
                continue
            for ci, cand in enumerate(entries):
                if ci == qi or cand["kept"] is None or cand["eps"] is None:
                    continue
                sum_eps = q["eps"] + cand["eps"]
                cheap_lb = cheap_lower_bound(q["kept"], cand["kept"])
                for r in TRADEOFF_RS:
                    outcome = decide_range(q["kept"], cand["kept"], r, sum_eps, cheap_lb=cheap_lb)
                    if outcome == "reject_cheap":
                        continue
                    after_cheap[r] += 1
                    if outcome != "refine":
                        resolved_after_cheap[r] += 1

        s5 = {r: (resolved_after_cheap[r] / after_cheap[r] if after_cheap[r] else float("nan")) for r in TRADEOFF_RS}
        refine_rate = {r: 1.0 - s5[r] for r in TRADEOFF_RS}

        row = {
            "tol": tol,
            "median_eps_A": median_eps,
            "median_bytes_per_track": median_bytes,
            "s5_r50": s5[50.0],
            "s5_r200": s5[200.0],
            "refine_r50": refine_rate[50.0],
            "refine_r200": refine_rate[200.0],
        }
        results.append(row)
        print(f"  tol={tol}: median_eps_A={median_eps:.3f} median_bytes={median_bytes:.1f} "
              f"S5(r=50)={s5[50.0]:.3f} S5(r=200)={s5[200.0]:.3f}", flush=True)
    return results


# --- item 3: per-query latency (r=200) -----------------------------------------------


def run_per_query_latency(full_corpus: list[dict], query_indices: list[int], r: float = PER_QUERY_R) -> dict:
    certified_times: dict[str, list[float]] = {"polyline": [], "spline": []}
    approx_times: dict[str, list[float]] = {"polyline": [], "spline": []}
    filter_uncompressed_times: list[float] = []

    t_start = time.time()
    for qi, q_idx in enumerate(query_indices):
        q = full_corpus[q_idx]

        t0 = time.perf_counter()
        for cand in full_corpus:
            if cand["idx"] == q_idx:
                continue
            if cheap_lower_bound(q["xy"], cand["xy"]) <= r:
                decide(q["xy"], cand["xy"], r)
        filter_uncompressed_times.append(time.perf_counter() - t0)

        for rep in ("polyline", "spline"):
            q_lin, q_eps = Q._lin_of(q, rep), Q._eps_of(q, rep)
            if q_lin is None or q_eps is None:
                continue

            t0 = time.perf_counter()
            for cand in full_corpus:
                if cand["idx"] == q_idx:
                    continue
                a_lin = Q._lin_of(cand, rep)
                if a_lin is None:
                    continue
                decide(q_lin, a_lin, r)
            approx_times[rep].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            for cand in full_corpus:
                if cand["idx"] == q_idx:
                    continue
                a_lin, a_eps = Q._lin_of(cand, rep), Q._eps_of(cand, rep)
                if a_lin is None or a_eps is None:
                    continue
                sum_eps = q_eps + a_eps
                cheap_lb = cheap_lower_bound(q_lin, a_lin)
                outcome = decide_range(q_lin, a_lin, r, sum_eps, cheap_lb=cheap_lb)
                if outcome == "refine":
                    decide(q["xy"], cand["xy"], r)
            certified_times[rep].append(time.perf_counter() - t0)

        if (qi + 1) % 100 == 0:
            print(f"  per-query latency: {qi + 1}/{len(query_indices)} queries, elapsed {time.time() - t_start:.1f}s", flush=True)

    return {"certified": certified_times, "approx": approx_times, "filter_uncompressed": filter_uncompressed_times,
            "elapsed": time.time() - t_start}


# --- main -----------------------------------------------------------------------


def main() -> None:
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log("=== item 1: recomputing correctness grid (miss/false-positive rates) ===")
    full_corpus, query_indices, grid = recompute_correctness_grid()
    log(f"grid recomputed, elapsed {grid['elapsed']:.1f}s")
    rates = approx_error_rates(grid)
    for r, reps in rates.items():
        for rep, vals in reps.items():
            log(f"  r={r} rep={rep}: miss_rate={vals['miss_rate']:.5f} "
                f"false_positive_rate={vals['false_positive_rate']:.5f} "
                f"(tp={vals['n_ground_truth_positive']}, approx_pred_pos={vals['approx_n_predicted_positive']})")

    log("=== item 2: trade-off curve (polylines, 100 tracks / 200 queries) ===")
    tracks, _ = Q.load_clean_tracks(n=Q.N_TRACKS, seed=Q.SEED)
    xy_sample = build_tradeoff_sample(tracks)
    log(f"tradeoff sample: {len(xy_sample)} tracks ({TRADEOFF_N_TRACKS - 30} base + 30 near-duplicates "
        f"targeting r in {TRADEOFF_RS})")
    tradeoff = run_tradeoff_curve(xy_sample)

    log("=== item 3: per-query latency (r=200) ===")
    per_query = run_per_query_latency(full_corpus, query_indices)
    log(f"per-query latency done, elapsed {per_query['elapsed']:.1f}s")

    with open(OUT_PKL, "wb") as f:
        pickle.dump({"approx_error_rates": rates, "tradeoff": tradeoff, "per_query": per_query}, f)
    with open(OUT_LOG, "w") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
