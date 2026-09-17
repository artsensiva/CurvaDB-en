# TODO

The research part (step0-step3) is complete — the final writeup is in
[docs/findings.md](docs/findings.md). Open (untested) questions are
there, in the "Not investigated" section.

- RESOLVED (step1, item 1): `benchmarks/step0.py`'s `MAX_ARC_POINTS` cap
  was a workaround for GPS jumps in the raw data. `src/traj/clean.py`
  now splits tracks at breaks (dt > 30s OR speed > 70 m/s) before
  fitting/benchmarking — fixing the cause, not just the symptom.
  `step0.py` itself is untouched (the "before/after" comparison lives in
  step0_diagnostics.md and step1.md); all step1_*.py benchmarks use
  `load_clean_tracks()`.

- The acceptance threshold for honest fitting (step1, item 2, >= 99% of
  tracks with dense error <= tol) was reached with margin (100%,
  585/585) — no blockers, nothing to record.

- `benchmarks/step1_compression.py` and `step1_search.py` use
  subsamples of the cleaned tracks (60 and all 585, respectively) with
  different seeds — running on the full corpus takes longer (~4-15
  minutes per script) if desired, but the conclusions (item 7) are
  robust to sample size (verified at 585/585 for search).

- RESOLVED (step7 M0 review, `docs/reviews/step7_M0.md`, finding 4): rolling-row
  (`O(n+m)` memory) DP kernels `_decide_core_rolling` /
  `_decide_core_conservative_rolling` added in `src/traj/frechet_cont.py` (M1),
  used automatically by `decide()`/`decide_conservative()` when `n*m >
  5_000_000` (S2's spline-vs-original-track certificates can reach that size).
  Verified identical to the original `O(nm)` kernels via
  `test_rolling_dp_matches_full_dp` (200 random cases, exact boolean match).

- OPEN (step7 M1.2, `docs/decisions/ADR-0012-s2-invalid-fit-threshold-sensitivity.md`):
  `src/traj/spline_lsq.py`'s fitters (`fit_adaptive`, `fit_uniform`) have no
  mechanism to control error *between* samples, only *at* them, unlike
  `src/traj/spline.py`'s `fit()`. On real GeoLife tracks this gives a very
  high "invalid fit" rate under S2's validity check (93.8% at the fixed
  10x/100x threshold, still 17.9% at a 1000x/1000x comparison threshold) --
  not a threshold-calibration problem, a structural gap in the fitter. A
  proper fix (a dense-error-aware fitting mode for `spline_lsq.py`, mirroring
  `spline.py`'s densify mechanism, or reconsidering `S2_FIT_TOL` itself) is
  out of scope for M1.2 (about the certification algorithm, not the fitting
  methodology) -- open for a future milestone.
