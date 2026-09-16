"""Step 1, item 1: track cleaning -- how many tracks/points were dropped.

Run: venv/bin/python benchmarks/step1_clean.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from traj.clean import BEIJING_BBOX, MAX_DT_S, MAX_SPEED_MPS, load_clean_tracks  # noqa: E402
from _report_utils import upsert_section  # noqa: E402

SEED = 42
N_TRACKS = 200
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_MD = os.path.join(RESULTS_DIR, "step1.md")
TITLE = "# Step 1: data cleaning, honest spline fitting, A/B\n"
SECTION_HEADER = "## 1. Track cleaning"


def _section_md(stats) -> str:
    lines = [
        f"{SECTION_HEADER}\n",
        (
            f"Parameters: a time gap > {MAX_DT_S:.0f}s OR speed > "
            f"{MAX_SPEED_MPS:.0f} m/s -- split; single-point segments -- "
            f"drop; Beijing bbox {BEIJING_BBOX} (lat_min, lat_max, lon_min, "
            "lon_max); then refilter to 50..2000 points per segment.\n"
        ),
        f"Loaded raw tracks: {stats.n_tracks_in}, points: {stats.n_points_in}.\n",
        "| Stage | Value |",
        "|---|---|",
        f"| extra segments from break-splitting | {stats.n_split_segments} |",
        f"| single-point outliers removed | {stats.n_outliers_removed} |",
        f"| points dropped outside Beijing bbox | {stats.n_points_dropped_bbox} |",
        f"| segments dropped by length filter (not 50..2000) | {stats.n_dropped_short_or_long} |",
        f"| **total: tracks** | {stats.n_tracks_out} (out of {stats.n_tracks_in} original) |",
        f"| **total: points** | {stats.n_points_out} (out of {stats.n_points_in} original) |",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    _tracks, stats = load_clean_tracks(n=N_TRACKS, seed=SEED)
    print(
        f"in: {stats.n_tracks_in} tracks / {stats.n_points_in} points -> "
        f"out: {stats.n_tracks_out} tracks / {stats.n_points_out} points"
    )
    print(
        f"split_segments={stats.n_split_segments} outliers={stats.n_outliers_removed} "
        f"bbox_dropped_points={stats.n_points_dropped_bbox} "
        f"length_refilter_dropped={stats.n_dropped_short_or_long}"
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(OUT_MD):
        with open(OUT_MD, "w") as f:
            f.write(TITLE + "\n")
    upsert_section(OUT_MD, SECTION_HEADER, _section_md(stats))
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
