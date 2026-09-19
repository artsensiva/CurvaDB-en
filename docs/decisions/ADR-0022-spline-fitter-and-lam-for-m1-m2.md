# ADR-0022: spline segment fitter and `lam_fallback` for M1/M2, decided before the run

- Status: Accepted (decision rule fixed now, before M0.1's script was run; outcome filled in
  below after the run, same discipline as ADR-0013)
- Date: 2026-09-19

## Context

`benchmarks/results/step8.md`'s M0 Conclusions flagged an open question: the certified
`eps_A <= target_tol` fraction for the LSQ-uniform spline collapses at tight `tol` relative to
sigma/curve complexity (down to 0% in two cells), even in cells step3's own honest-curve-error
criterion accepts as reachable. Two candidate explanations were named, not yet separated:

1. `certify_spline`'s default `lam_fallback = 0.1` m adds a fixed additive margin to `eps_A`
   (the section-2.4 formula: `eps_A = d_F(A, Lin(A')) + lam [+ margins]`) that can by itself
   approach or exceed a tight `target_tol`.
2. `spline_lsq.fit_uniform` only controls error *at* the sample points (its own documented
   contract, `src/traj/spline_lsq.py`'s module docstring) -- it can oscillate *between* samples
   without that showing up in the fitting criterion, and `certify_spline`'s Fréchet-based `eps_A`
   *is* sensitive to such between-sample excursions, unlike `fit_uniform`'s own search.

M0.1 (`benchmarks/step8_m01_lam_check.py`) separates the two: a `lam_fallback` sweep
(`{0.1, 0.01, 0.001}`) holding the fitter fixed at `fit_uniform`, and a fitter comparison
(`fit_uniform` vs. `traj.spline.fit()`, ADR-0013's dense-between-samples-error-controlled
fitter) holding `lam_fallback` fixed at the default `0.1`, on the four cells M0 flagged:
`(sigma=0, tol=0.5)`, `(sigma=0.1, tol=0.5)`, `(sigma=1, tol=2)`, `(sigma=5, tol=10)`. Per the
mandatory correction to this ADR's own plan: `n` (parameter count, median/p90) is reported
alongside every `eps_A <= tol` fraction, so a `fit()` advantage cannot be mistaken for "it just
uses more control points" without that being visible directly.

## Decision rule (fixed now, before the run -- mirrors ADR-0013's step7 precedent)

**Fitter choice for spline segments in M1/M2:**

1. Compute each fitter's `eps_A <= target_tol` fraction (at `lam_fallback = 0.1`) per cell, then
   average across the four cells.
2. If one fitter's average fraction is higher by **at least 20 percentage points**, that fitter
   is selected outright.
3. Otherwise (the gap is under 20 points), the fitter with the **smaller pooled median `n`**
   (parameter count, pooled across the four cells' own accepted/converged populations) is
   selected.

**`lam_fallback` for spline segments in M1/M2:**

Using `fit_uniform`'s per-cell `eps_A <= target_tol` fractions at each tested `lam_fallback`,
averaged across the four cells: take `lam = 0.001`'s average fraction as the reference. Among
`{0.001, 0.01, 0.1}`, `lam_fallback` is set to the **smallest** value whose average fraction does
not fall more than **5 percentage points** below that reference. (Since `0.001` trivially
satisfies a 0-point drop relative to itself, this rule resolves to `0.001` unless a *larger*
`lam` is needed to avoid a certification-feasibility failure at very tight `lam` -- i.e. unless
the data itself shows `lam=0.001`'s fraction is *not* the best of the three, in which case the
smallest-`lam`-within-5-points-of-the-reference reading is applied to whichever value the
reference actually turns out to be. The rule is applied mechanically to whatever the run
produces, not adjusted after seeing the numbers.)

Both rules are applied to the same run, mechanically, before looking at whether the resulting
choice matches any prior expectation.

## Outcome (`benchmarks/step8_m01_lam_check.py`, `benchmarks/results/step8.md`'s M0.1 section)

**`lam_fallback` rule:** average `eps_A<=tol` fraction across the 4 cells is 4.3% at *every*
tested value (0.001, 0.01, 0.1) -- the additive margin never moves the outcome, because in every
failing case the base `d_F(A, Lin(A'))` term already exceeds `target_tol` before `lam` is added
(e.g. `sigma=5/tol=10`: `eps_A` p50 ~13 m against a 10 m tol). Applying the rule mechanically:
`lam = 0.001` (the reference itself, trivially within 5pp of itself; no larger `lam` was needed
to avoid a certification-feasibility failure at the tightest tested value -- the data showed no
such failure). **`lam_fallback = 0.001` for M1/M2's spline segments.**

**Fitter rule:** average fraction is 4.3% for `fit_uniform` vs. 36.7% for `spline.fit()` (lam
fixed at 0.1) -- a +32.4pp gap, decisively above the 20pp threshold. **Rule step 1 decides it
outright; `spline.fit()` is selected**, without needing the `n`-based tie-break.

**Caveat surfaced by the mandatory `n` check** (not part of the mechanical rule, but material to
how the decision is used going forward): the two cells driving most of that 32.4pp gap
(`sigma=1/tol=2`: +73.3pp; `sigma=5/tol=10`: +66.7pp) are also the two cells where `spline.fit()`
uses 3.3x-4.5x more parameters than `fit_uniform` (n median 412 vs. 125.5, and 436 vs. 97.0) --
part of the win there is plausibly `spline.fit()`'s own adaptive densification spending more
knots, not purely eliminating between-sample oscillation. The one cell where `spline.fit()` uses
*fewer* parameters (`sigma=0/tol=0.5`: 274.0 vs. 359.5) still shows a materially tighter `eps_A`
(p50 0.595 vs. 0.789), which is clean, parameter-count-independent evidence that between-sample
oscillation control is a real, non-trivial contributor -- not the whole story, but not an
artifact either. Full breakdown: `benchmarks/results/step8.md`'s M0.1 Conclusions.

## Refinement (M0.2, `benchmarks/step8_m02_lam_refine.py`)

M0.1's `lam_fallback` sweep ran on `fit_uniform`, not on `spline.fit()` -- the fitter this ADR
actually selected. M0.2 repeats the sweep on `spline.fit()` itself, adds two measurements the
user asked for explicitly (the `Lin(A')` linearization vertex count, and the segment's actual
encoded bytes), and checks `tol=0.5 m` spline availability with a tighter internal construction
tol.

**What `SplineSegment` stores, and what `lam` can and can't affect:** `traj.encode.SplineSegment`
encodes a spline's own internal knots and control points (from the fit's `tck`), never
`certify_spline`'s internal linearization `Lin(A')` -- that linearization is produced, used, and
discarded entirely inside the certificate computation. Confirmed empirically, not assumed:
segment bytes are bit-identical across all three tested `lam` values in every one of the 4 cells
(e.g. 532B/532B/532B at `sigma=0/tol=0.5`). **`lam_fallback` can only ever affect `eps_A` and
certification time, never what M2 would actually store.**

**`lam_fallback`'s effect on `spline.fit()` is real, unlike on `fit_uniform`.** Average
`eps_A<=tol` fraction across the 4 cells: 70.0% (lam=0.001), 60.0% (lam=0.01), 36.7% (lam=0.1) --
a genuine 33.3pp spread, the opposite of M0.1's ~0pp finding on `fit_uniform`. Mechanism: once
the fitter itself controls the *base* Fréchet term tightly (which is exactly what `spline.fit()`
is designed to do), the *additive* `lam` margin becomes the dominant remaining lever -- the two
M0.1/M0.2 findings are consistent, not contradictory, once read as "lam matters once the base
error stops swamping it."

**Applying the refined rule:** lam=0.001's average fraction (70.0%) is not tied with lam=0.01
(60.0%, 10pp below) or lam=0.1 (36.7%, 33.3pp below) -- both exceed the 5pp tie-break window, so
the rule's primary criterion decides outright; the bytes/time tie-break (bytes tied at 678.8B
average regardless, so it would fall through to time) is never reached. **`lam_fallback = 0.001`
is confirmed** -- same value as M0.1's naive pick, but now for a substantive reason (0.001 is
measurably best, not merely "smallest, trivially tied with itself").

**Real cost this ADR did not originally weigh:** median certification time per track, averaged
across the 4 cells, is 2350.7 ms at lam=0.001 vs. 208.5 ms at lam=0.1 -- an ~11.3x spread (worst
cell, `sigma=5/tol=10`: 4679.8 ms median, 8730.2 ms p90). The rule as written never consults this
number here, because the fraction gap alone already decides it. This is flagged as a
**consequence to weigh explicitly once M1/M2's actual DP cost function is implemented**, not
silently accepted: `cost(i, j)` (spec section 2.2) is evaluated `O(m*W)` times per track (spec
section 2.3), and a multi-second-per-candidate certification cost at `lam=0.001` may force a
revisit of this choice once real runtimes are measured -- this ADR's decision stands for now, but
is explicitly not final on cost grounds.

**`tol=0.5 m` spline availability, revised from what the naive check suggested:** calling
`spline.fit()` directly at the 0.5 m target certifies at most 53.3% of tracks (at lam=0.001,
worse at larger lam) -- suggestive of unavailability. But targeting a substantially tighter
*internal* tol first (0.25 m or 0.1 m) and certifying with lam=0.001 reaches **100%** certified
pass at every internal-tol/sigma combination tested, at a real but bounded parameter cost
(~1.5x-3x more control points than the naive target-tol attempt). **Spline segments ARE
certificate-available at `tol=0.5 m`** -- the earlier open question is resolved in the opposite
direction from the hypothesis it was testing. This also validates spec section 2.4's actual
planned M1 construction (start dense, remove knots down to the certified limit) over the cruder
direct-fit-at-target approach used for comparison here.

## Consequences

- `src/traj/knot_removal.py` (M1) starts its greedy removal from a dense/interpolating LSQ
  spline and certifies each candidate via `certify_spline(..., lam_fallback=0.001)`, not the
  0.1 default used elsewhere in this codebase (ADR-0018's `certify_spline` default remains 0.1
  for callers that don't override it -- this ADR only fixes the value M1/M2's spline-segment
  code path passes explicitly).
- M1/M2's spline-segment cost accounting (spec section 2.2's `cost(i, j)`) should track `n`
  alongside the certified pass/fail, not just bytes after the fact -- the `n` confound found here
  means a spline segment "passing" the certificate is not by itself evidence it's cheap; M2's DP
  cost function already compares bytes directly, so this is a reporting/interpretation note, not
  a code change.
- **Revised by M0.2** (this consequence, as originally written after M0.1, is superseded, not
  just extended): very tight `tol` (~0.5 m) is hard for `spline.fit()` called *directly at the
  target* (at most 53.3% certified even at the now-chosen lam=0.001), but not because splines are
  unavailable there -- calling `spline.fit()` at a tighter *internal* tol (0.25 m or 0.1 m) and
  certifying with lam=0.001 reaches 100% certified pass. M1/M2's knot-removal construction
  (spec 2.4: start dense, remove down to the certified limit) should therefore reach `tol=0.5 m`
  spline segments fine in practice; a direct-fit-at-target attempt should not be read as evidence
  otherwise. DP+SED vs. spline at `tol=0.5 m` remains a genuine `cost(i,j)` trade-off for M2 to
  decide, not a construction-availability default toward the polyline.
- The spec's literal wording for M1 (`docs/specs/step8_A_hybrid.md` section 2.4: "начать с
  интерполирующего или плотного LSQ-сплайна") doesn't name a specific fitter for that starting
  point; this ADR resolves the ambiguity in favor of `spline.fit()`'s dense-error-controlled
  construction over `spline_lsq.fit_uniform`'s at-sample-only control, for the reasons above.

## Links

`benchmarks/step8_m01_lam_check.py`, `benchmarks/step8_m02_lam_refine.py`;
`benchmarks/results/step8.md` (M0 Conclusions, M0.1, M0.2); ADR-0013 (the same pre-registered-rule
discipline, for `spline.fit()` vs. `spline_lsq` in step7); ADR-0018 (section 2.4 as the default
spline certificate, whose formula's `lam` term this ADR tunes); ADR-0021 (the
certificate-is-additional-not-gating split this investigation depends on);
`docs/specs/step8_A_hybrid.md` section 2.4 (knot removal for spline segments, the M1 consumer of
this decision).
