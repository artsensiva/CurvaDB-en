# ADR-0013: spec deviation -- compare `spline.fit()` against `spline_lsq` for S2

- Status: Accepted (comparison procedure fixed now; outcome filled in below after the run)
- Date: 2026-09-17

## Context

`docs/reviews/step7_M1_2.md` finding 1: `src/traj/spline.py`'s `fit()` already exists and
dense-error-controls by construction (step1: 585/585 within `tol=10.0` on these exact 585
tracks, on a dense grid against the polyline, not just at samples). M1/M1.2 used
`spline_lsq.py`'s `fit_adaptive` for S2 without considering it. Even with ADR-0014's measurement
bug fixed, a real (smaller, ~18.3% on a 60-track sample) invalid rate persists for
`spline_lsq.py` -- a genuine, if less dramatic than first reported, finding: `spline_lsq.py`'s
fitters only control error *at* the sample points (`spline_lsq.py`'s own documented contract),
not on a dense grid between them.

**Both fitters use the same `tol = 10.0`** (step1's own value for these 585 tracks) -- not two
different tol values. This number means a different thing per fitter, recorded explicitly so it
is never misread as an apples-to-apples error metric: `fit_adaptive`/`fit_uniform`'s `max_error`
is the direct residual *at* the sample points; `spline.fit()`'s `max_error` is the dense-grid
point-to-*polyline* deviation. Both are bounded by the same numeric budget; they are not the same
quantity.

## Options considered

- Keep `fit_adaptive` only, ignore the review finding.
- Switch outright to `spline.fit()` without comparing.
- **Run both, decide by a rule fixed before seeing the results** -- chosen, consistent with this
  project's established discipline (ADR-0010's thresholds, fixed before its run).

## Decision

**Comparison procedure (fixed now, before the corrected full-corpus runs)**, not a foregone
conclusion:

1. Run both fitters (`fit_adaptive`, `spline.fit()`) on all 585 tracks at `tol=10.0`, under
   ADR-0010's validity check (with ADR-0014's `dense_mode` fix: `"raw"` for `fit_adaptive`,
   `"time"` for `spline.fit()`), same 10x/100x multiplier.
2. The fitter with the **higher fraction of valid fits** is preferred.
3. If the two are within **5 percentage points** of each other, fall back to two indicators among
   *passing* tracks: the median `eps_A / tol` ratio, and the absolute median `eps_A`. If both
   indicators agree on the same fitter, that fitter is preferred. If they disagree, this ADR
   records the conflict explicitly (not silently broken) and keeps `spline.fit()` as the default
   going forward, since its `eps_A` is directly commensurate with a dense-error-controlled
   representation by construction.

The spec (`docs/specs/step7_B_certified_store.md` §2.1) names `spline_lsq` as S2's module --
using `spline.py`'s `fit()` instead (if steps 2-3 select it) is a deviation from the literal spec
text, recorded in `docs/ROADMAP.md` section 8 (spec text itself untouched).

### Outcome (filled in after the corrected full-corpus run, `benchmarks/results/step7.md` M1.3)

<!-- FILLED IN AFTER THE RUN -->

## Consequences

- If `spline.fit()` is selected, S2's fitter no longer matches the spec's literal module name --
  a documented, deliberate deviation, not a silent one.
- Either way, ADR-0010's validity check and multiplier are reused unchanged; only the fitter (and
  its shared `tol=10.0`) changes.
- `TODO.md`'s open item about `spline_lsq.py`'s dense-error-control gap is resolved by this
  decision (a fitter switch), not left dangling.

## Links

`docs/reviews/step7_M1_2.md`; ADR-0010 (validity check, reused); ADR-0011 (fit_adaptive vs.
fit_uniform, superseded in relevance by this broader comparison); ADR-0014 (the measurement bug
this comparison's correctness depends on); `docs/ROADMAP.md` section 8;
`benchmarks/results/step7.md` (M1.3 section).
