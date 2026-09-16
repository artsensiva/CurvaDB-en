"""Step 1, item 3: spline recall@10 on tracks with dense error <= 2*tol
(after cleaning in item 1 and honest fitting in item 2) -- checking the
cause of low recall in step0 (already known: spline oscillation between
timestamps).

Run: venv/bin/python benchmarks/step1_recall_check.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from traj.clean import load_clean_tracks  # noqa: E402
from step0 import N_QUERIES, TOL, build_representation, full_scan_top10, recall_at_k  # noqa: E402
from _report_utils import upsert_section  # noqa: E402

SEED = 42
N_TRACKS = 200
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_MD = os.path.join(RESULTS_DIR, "step1.md")
SECTION_HEADER = "## 3. Recall@10 on tracks with dense error <= 2·tol"


def main() -> None:
    tracks, _clean_stats = load_clean_tracks(n=N_TRACKS, seed=SEED)
    reprs = [build_representation(tr, TOL) for tr in tracks]

    good_idx = [i for i, r in enumerate(reprs) if r.spline_max_error <= 2 * TOL]
    print(f"tracks with dense error <= 2*tol: {len(good_idx)}/{len(reprs)}")

    rng = np.random.default_rng(SEED)
    pool = np.array(good_idx)
    n_queries = min(N_QUERIES, len(pool))
    query_idx = rng.choice(pool, size=n_queries, replace=False)

    reprs_by_kind = {
        "raw": [(i, r.raw) for i, r in enumerate(reprs)],
        "dp": [(i, r.dp) for i, r in enumerate(reprs)],
        "spline_same": [(i, r.spline_same) for i, r in enumerate(reprs)],
    }

    recalls: dict[str, list] = {"dp": [], "spline_same": []}
    for qi in query_idx:
        qi = int(qi)
        raw_top10 = full_scan_top10(reprs_by_kind["raw"][qi][1], reprs_by_kind["raw"], exclude_idx=qi)
        raw_ids = {idx for _, idx in raw_top10}
        for kind in recalls:
            got = full_scan_top10(reprs_by_kind[kind][qi][1], reprs_by_kind[kind], exclude_idx=qi)
            recalls[kind].append(recall_at_k(raw_ids, got))

    mean_recall_dp = float(np.mean(recalls["dp"]))
    mean_recall_spline = float(np.mean(recalls["spline_same"]))
    print(f"recall@10 dp={mean_recall_dp:.3f} spline={mean_recall_spline:.3f} (n_queries={n_queries})")

    line = (
        f"On {len(good_idx)}/{len(reprs)} cleaned tracks with spline dense error <= 2·tol "
        f"({n_queries} queries from that same group, corpus -- all {len(reprs)} tracks): "
        f"recall@10 DP = {mean_recall_dp:.3f}, recall@10 fixed spline = "
        f"{mean_recall_spline:.3f} -- vs 0.997/0.707 in step0.md (broken fitting, "
        "no data cleaning)."
    )
    body = f"{SECTION_HEADER}\n\n{line}\n"
    upsert_section(OUT_MD, SECTION_HEADER, body)
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
