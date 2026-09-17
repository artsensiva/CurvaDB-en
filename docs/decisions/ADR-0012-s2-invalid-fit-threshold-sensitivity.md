# ADR-0012: S2 invalid-fit rate is a fitter property, not a threshold-calibration artifact

- Status: Superseded by ADR-0014
- Date: 2026-09-17

**Superseded (M1.3):** this ADR's entire sensitivity analysis was computed from
`fit_validity()`'s dense-deviation measurement, which ADR-0014 found to be buggy for
`spline_lsq.py` fits (wrong parametrization domain -- deviations off by orders of magnitude,
inconsistently in either direction). The conclusion below ("not a calibration artifact, a
structural property") is therefore unsound as stated -- it was built on broken numbers, not a
merely-conservative reading of good ones. See ADR-0014 for the bug and corrected full-corpus
numbers, and ADR-0013 for the resulting fitter comparison. Kept here, unedited below, as the
record of what was believed and why -- not deleted.

## Context

ADR-0010 fixed S2's validity thresholds (dense-grid deviation `<= 10x` the fitter's tol,
control points within `100x` the track's bbox diagonal) before running the full 585-track
corpus, with an explicit instruction: if the thresholds turn out to be a poor choice, write a
new ADR and re-run with an alternative, reporting both results side by side rather than silently
editing ADR-0010. The full run under those fixed thresholds gave **36/585 (6.2%) pass, 0
fallback, 549/585 (93.8%) invalid_fit** -- high enough to ask whether `10x`/`100x` were simply
too strict.

## Investigation

Computed the invalid-fraction sensitivity to the deviation multiplier directly from the full
run's cached raw per-track data (no re-fitting needed):

| Multiplier | Invalid fraction |
|---|---|
| 10x (ADR-0010) | 0.938 (549/585) |
| 20x | 0.879 |
| 50x | 0.776 |
| 100x | 0.655 |
| 200x | 0.496 |
| 500x | 0.262 |
| 1000x | 0.171 |
| 2000x | 0.109 |

The deviation distribution itself: median 989 m (~198x `S2_FIT_TOL=5`), p90 10,698 m, p99
33,418,784 m, max 3.86e17 m. This is not a distribution with a clear "reasonable" cutoff a few
multiples above `10x` -- it has an extremely heavy tail, and even a **1000x** multiplier (500m
absolute, one hundred times the fit tolerance's own order of magnitude) still leaves 17% invalid.

Ran the full corpus again at `1000x`/`1000x` (both factors) to get real certification numbers,
not just an extrapolation: **480/585 (82.1%) pass, 0 fallback, 105/585 (17.9%) invalid_fit**.
Notably, `n_fallback = 0` under **both** thresholds -- every track whose fit is valid gets
certified; the M1.1 root-splitting fix (ADR-0008) resolves genuine geometric edge cases
completely once numerically unstable fits are excluded from consideration, closing the loop on
M1.1's own diagnosis (most of that milestone's 46 uncertified tracks were unstable fits, not
algorithm-level failures).

## Decision

**Do not change ADR-0010's thresholds.** The sensitivity analysis shows the high invalid rate is
not an artifact of an arbitrarily-strict multiplier choice -- it persists (at a still-substantial
18%) even at a multiplier two orders of magnitude more generous, and the underlying cause is
structural: `spline_lsq.py`'s fitters (`fit_adaptive`/`fit_uniform`, see ADR-0011) bisect down to
the *minimum* knot count satisfying error *at the sample points only* (`docstring`: "the spline
isn't required to pass through the noisy points"), with no mechanism to control deviation
*between* samples -- unlike `spline.py`'s `fit()`, which was built specifically to close that gap
(`docs/findings.md`'s step0/step1 history). Picking a specific "generous enough" multiplier to
report a nicer pass rate would misrepresent this as a calibration problem when it is not one.

Both result sets (ADR-0010's `10x`/`100x` and this ADR's `1000x`/`1000x` comparison) are reported
side by side in `benchmarks/results/step7.md`'s M1 section, as instructed.

## Consequences

- S2's "valid fit" corpus is genuinely small under any defensible threshold -- this is now a
  documented, understood property of the current fitting approach on real GeoLife data, not an
  open question.
- Properly fixing this (a dense-error-aware fitting mode for `spline_lsq.py`, mirroring
  `spline.py`'s densify mechanism, or reconsidering `S2_FIT_TOL` itself) is out of scope for M1.2
  (which is about the certification algorithm, not the fitting methodology) -- recorded as an
  open item for a future milestone rather than addressed here.
- The `n_fallback = 0` result at both thresholds is independent confirmation that ADR-0008's fix
  is working as intended: certification itself is not the bottleneck once given a valid
  representation.

## Links

`benchmarks/results/step7.md` (M1.2 section); ADR-0010; ADR-0008; ADR-0011; `docs/findings.md`
(step0/step1 history on dense-error control); `TODO.md`.
