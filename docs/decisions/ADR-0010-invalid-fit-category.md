# ADR-0010: "Invalid fit" category for S2, with a fixed validity threshold

- Status: Accepted
- Date: 2026-09-17

## Context

M1.1's diagnostics (`docs/reviews/step7_M1.md` item 0) found that the raw chord-length/speed
numbers for the 46 originally-uncertified S2 tracks were physically absurd (mean chord ~2.4e13 m,
speeds up to ~4.6e17 m/s). Root cause: at least 26/46 have a `fit_adaptive` spline fit with
wildly unstable control points (confirmed directly on one track: control points spanning ~1e12
while the track itself is ~2km) -- `spline_lsq.py`'s own documented near-interpolation numerical
instability, not a certification-algorithm problem. Before M1.2, these tracks were silently
lumped into the same "fallback" bucket as genuine geometric edge cases the root-splitting fix
(ADR-0008) targets, conflating two unrelated failure modes and making S2's pass/fallback counts
misleading (a "fallback" reads as "the certification algorithm couldn't handle this geometry,"
not "the upstream fit is garbage before certification was even attempted").

## Options considered

- **Leave as-is** (unstable fits counted as "fallback"): simplest, but keeps conflating "spline
  fit is garbage" with "certification algorithm hit a real geometric edge case" -- exactly the
  ambiguity that made the 46-track number hard to interpret in M1.1.
- **Exclude unstable fits from the corpus entirely** (don't attempt certification): loses the
  information that these tracks exist and why, and risks the exclusion criterion silently
  drifting depending on when/how it's applied.
- **A third, explicit "invalid fit" category, checked before certification is attempted**: the
  track is neither a pass nor a certification failure -- it never reaches certification at all,
  because the *representation itself* doesn't honestly describe the track. Reported separately.

## Decision

Add a validity check, run before `certify_spline_linearization` for every track in S2. A fit is
**invalid** if either of two fixed conditions fails (checked in this order; either failure is
sufficient):

1. **Dense-grid deviation**: the spline's maximum point-to-track-polyline distance on a dense
   parameter grid (reusing `traj.spline.dense_max_error`'s existing point-to-segment methodology,
   `mode="time"`, `pts_per_interval=10` -- the same "honest error between samples, not just at
   them" check `spline.py`'s own `fit()` already applies, which `spline_lsq.py`'s fitters do
   *not* apply on their own) exceeds **10 x the fitter's own target tolerance** (`S2_FIT_TOL =
   5.0` m, so the threshold is 50 m).
2. **Control-point bound**: any control point lies more than **100 bounding-box diagonals** of
   the track's own extent away from the track's first point.

These thresholds are fixed **before** the full-corpus run and must not be adjusted after seeing
its results. If they turn out to be a poor choice (too strict, too lenient, or miscategorizing
tracks a human would judge differently), the correction is a **new ADR** with a **new run**, and
both sets of results are reported side by side -- not a silent edit to this one.

Invalid-fit tracks are excluded from S2's pass/fallback counts entirely and reported as a third,
separate bucket.

## Consequences

- S2's pass/fallback/invalid-fit breakdown is now interpretable: "fallback" means the
  certification algorithm genuinely couldn't certify a *valid* representation within budget;
  "invalid fit" means the representation itself doesn't honestly describe the track, independent
  of certification.
- Effect on S2's guarantee: none -- `certify_spline_linearization`'s own correctness
  (`eps_A >= inf` contract, ADR-0008) is unaffected; this only changes what gets *counted* as a
  meaningful attempt.
- Actual counts from the full 585-track run: see `benchmarks/results/step7.md`'s M1 section
  (filled in after the run this ADR's decision governs).

**Update (M1.3, ADR-0014):** the dense-grid deviation check's implementation
(`benchmarks/step7_certify.py`'s `fit_validity()`) had a parametrization-domain bug when applied
to `spline_lsq.py` fits -- it evaluated the spline at the wrong part of its domain, inflating (or
occasionally deflating) the measured deviation by orders of magnitude. **This ADR's own decision
(the two-condition validity check, and its `10x`/`100x` multiplier thresholds) is unaffected and
unchanged** -- the bug was in the measurement feeding condition 1, not in the check's design. But
the *specific counts this ADR reported* (93.8% invalid on the full corpus) are unreliable as
measurements; see ADR-0014 for the bug, the fix, and corrected numbers.

## Links

`docs/reviews/step7_M1.md`; `docs/reviews/step7_M1_2.md`; `benchmarks/results/step7.md` (M1.1,
M1.2 and M1.3 sections); `src/traj/spline.py`'s `dense_max_error`; ADR-0008; ADR-0011 (fitter
choice, same investigation); ADR-0014 (the measurement bug affecting this ADR's reported
numbers).
