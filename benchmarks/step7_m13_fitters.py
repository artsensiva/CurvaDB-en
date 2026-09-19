"""Step7 M1.3 (docs/reviews/step7_M1_2.md, ADR-0013): corrected full-corpus S2 runs
for both candidate fitters, sequentially, in one process -- fit_adaptive (ADR-0014's
dense_mode="raw" fix) and spline.fit() (dense_mode="time", ADR-0013's comparison).
Both share tol=10.0 (step1's own value for these 585 tracks).

Run: venv/bin/python benchmarks/step7_m13_fitters.py
"""

from __future__ import annotations

import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from traj.spline import fit as spline_fit  # noqa: E402
from step7_certify import N_TRACKS, SEED, fit_adaptive, load_clean_tracks, run_s2  # noqa: E402

OUT_PKL = "/tmp/m13_s2_fitters.pkl"
OUT_LOG = "/tmp/m13_s2_fitters.log"


def main() -> None:
    log_lines = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log("Loading and cleaning tracks...")
    tracks, clean_stats = load_clean_tracks(n=N_TRACKS, seed=SEED)
    log(f"{len(tracks)} cleaned track segments from {clean_stats.n_tracks_in} raw tracks.")

    t0 = time.time()
    log("=== fit_adaptive (dense_mode=raw, ADR-0014 fix) ===")
    result_adaptive = run_s2(tracks, "fit_adaptive-corrected", fitter=fit_adaptive, dense_mode="raw")
    log(f"fit_adaptive: {result_adaptive}")
    log(f"elapsed so far: {time.time() - t0:.1f}s")

    log("=== spline.fit() (dense_mode=time, ADR-0013 comparison) ===")
    result_spline_fit = run_s2(tracks, "spline.fit()", fitter=spline_fit, dense_mode="time")
    log(f"spline.fit(): {result_spline_fit}")
    log(f"total elapsed: {time.time() - t0:.1f}s")

    with open(OUT_PKL, "wb") as f:
        pickle.dump({"fit_adaptive": result_adaptive, "spline_fit": result_spline_fit}, f)
    with open(OUT_LOG, "w") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
