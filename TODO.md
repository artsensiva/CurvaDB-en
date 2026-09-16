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
