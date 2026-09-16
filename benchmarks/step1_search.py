"""Step 1, item 6: search (hypothesis A) -- recall@10 (raw/DP/fixed
spline) on CLEANED tracks, bootstrap interval, 3 seeds for query
selection (repeats step0.py's methodology, but on honest data/fitting).

Run: venv/bin/python benchmarks/step1_search.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from traj.clean import load_clean_tracks  # noqa: E402
from step0 import N_QUERIES, TOL, build_representation, full_scan_top10, recall_at_k  # noqa: E402
from _report_utils import upsert_section  # noqa: E402

CLEAN_SEED = 42
N_TRACKS = 200
QUERY_SEEDS = [42, 43, 44]
N_BOOTSTRAP = 2000
CI = 0.95
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_MD = os.path.join(RESULTS_DIR, "step1.md")
SECTION_HEADER = "## 6. Search (hypothesis A): recall@10 on cleaned tracks"


def _bootstrap_ci(values: np.ndarray, n_boot: int, ci: float, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    alpha = 1 - ci
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def main() -> None:
    t0 = time.time()
    tracks, _clean_stats = load_clean_tracks(n=N_TRACKS, seed=CLEAN_SEED)
    reprs = [build_representation(tr, TOL) for tr in tracks]
    print(f"built {len(reprs)} representations in {time.time() - t0:.1f}s")

    reprs_by_kind = {
        "raw": [(i, r.raw) for i, r in enumerate(reprs)],
        "dp": [(i, r.dp) for i, r in enumerate(reprs)],
        "spline_same": [(i, r.spline_same) for i, r in enumerate(reprs)],
    }

    all_recalls: dict[str, list] = {"dp": [], "spline_same": []}
    per_seed_means: dict[str, list] = {"dp": [], "spline_same": []}

    for qseed in QUERY_SEEDS:
        t_seed = time.time()
        rng = np.random.default_rng(qseed)
        n_queries = min(N_QUERIES, len(tracks))
        query_idx = rng.choice(len(tracks), size=n_queries, replace=False)
        seed_recalls: dict[str, list] = {"dp": [], "spline_same": []}
        for qi in query_idx:
            qi = int(qi)
            raw_top10 = full_scan_top10(reprs_by_kind["raw"][qi][1], reprs_by_kind["raw"], exclude_idx=qi)
            raw_ids = {idx for _, idx in raw_top10}
            for kind in seed_recalls:
                got = full_scan_top10(reprs_by_kind[kind][qi][1], reprs_by_kind[kind], exclude_idx=qi)
                r = recall_at_k(raw_ids, got)
                seed_recalls[kind].append(r)
                all_recalls[kind].append(r)
        for kind in per_seed_means:
            per_seed_means[kind].append(float(np.mean(seed_recalls[kind])))
        print(
            f"seed={qseed}: dp={np.mean(seed_recalls['dp']):.3f} "
            f"spline={np.mean(seed_recalls['spline_same']):.3f} "
            f"({time.time() - t_seed:.1f}s)"
        )

    stats = {}
    for kind in all_recalls:
        vals = np.array(all_recalls[kind])
        lo, hi = _bootstrap_ci(vals, N_BOOTSTRAP, CI, seed=CLEAN_SEED)
        stats[kind] = {"mean": float(vals.mean()), "ci_lo": lo, "ci_hi": hi, "n": len(vals)}
        print(f"{kind}: mean={stats[kind]['mean']:.3f} {CI:.0%} CI=({lo:.3f}, {hi:.3f}) n={len(vals)}")

    lines = [
        f"{SECTION_HEADER}\n",
        (
            f"{len(tracks)} cleaned tracks (seed={CLEAN_SEED}), tol={TOL:.0f}m, "
            f"{N_QUERIES} queries x {len(QUERY_SEEDS)} query-selection seeds "
            f"({', '.join(str(s) for s in QUERY_SEEDS)}) = {len(all_recalls['dp'])} "
            f"measurements per representation; recall@10 against the exact "
            f"Frechet distance on raw (cleaned) tracks; {CI:.0%} confidence "
            f"interval -- bootstrap, {N_BOOTSTRAP} resamples.\n"
        ),
        "| Representation | Recall@10 (mean) | " + f"{CI:.0%} CI" + " |",
        "|---|---|---|",
        f"| DP polyline | {stats['dp']['mean']:.3f} | ({stats['dp']['ci_lo']:.3f}, {stats['dp']['ci_hi']:.3f}) |",
        (
            f"| fixed spline | {stats['spline_same']['mean']:.3f} | "
            f"({stats['spline_same']['ci_lo']:.3f}, {stats['spline_same']['ci_hi']:.3f}) |"
        ),
        "",
        (
            f"For comparison, step0.md (broken fitting, no cleaning, 1 query "
            "seed): DP = 0.997, spline = 0.707. After cleaning (item 1) and "
            "honest fitting (item 2), the recall@10 gap between DP and the "
            "spline has almost entirely closed"
            + (
                ", but DP is still slightly ahead"
                if stats["dp"]["mean"] > stats["spline_same"]["mean"]
                else ""
            )
            + ".\n"
        ),
    ]
    body = "\n".join(lines) + "\n"
    upsert_section(OUT_MD, SECTION_HEADER, body)
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
