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

- RESOLVED (step7 M1.2/M1.3, `docs/decisions/ADR-0010-invalid-fit-category.md`,
  `docs/decisions/ADR-0012-s2-invalid-fit-threshold-sensitivity.md`,
  `docs/decisions/ADR-0013-spline-fit-for-s2.md`,
  `docs/decisions/ADR-0014-dense-error-domain-bug.md`): M1.2 reported a very
  high "invalid fit" rate for `src/traj/spline_lsq.py`'s fitters (93.8%/17.9%
  at two thresholds) and attributed it entirely to a structural gap (no
  dense-error control between samples, unlike `src/traj/spline.py`'s
  `fit()`). M1.3 found that measurement itself was partly broken (ADR-0014:
  `fit_validity()`'s dense-check evaluated `spline_lsq` fits at the wrong
  parametrization domain) -- the real invalid rate is lower but still
  nonzero (~18.3% on a 60-track sample after the fix), confirming the
  underlying architectural gap is real, just smaller than first reported.
  Resolved via ADR-0013: `spline.py`'s `fit()` (dense-error-controlled by
  construction) is compared against `spline_lsq` by a pre-registered rule;
  the outcome and full-corpus numbers are in
  `benchmarks/results/step7.md`'s M1.3 section. `spline_lsq.py`'s fitters
  themselves are unchanged -- no dense-error-aware fitting mode was added to
  them, since the resolution was a fitter *choice*, not a fix to
  `spline_lsq.py` -- record that as a still-open item only if a future
  milestone specifically needs `spline_lsq`'s fitters (not `spline.fit()`)
  to control dense error.
