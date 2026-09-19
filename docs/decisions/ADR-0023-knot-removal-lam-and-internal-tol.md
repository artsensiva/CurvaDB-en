# ADR-0023: `knot_removal.py`'s certification `lam_fallback` and internal-tol strategy

- Status: Accepted (decision rule fixed now, before the budget measurement script ran;
  outcome filled in below after the run, same discipline as ADR-0013/ADR-0022)
- Date: 2026-09-19

## Context

M2's DP (`docs/specs/step8_A_hybrid.md` section 2.3) evaluates `cost(i, j)` for `O(m*W)`
candidate cuts per track (`W=8` default), and a spline candidate's cost requires certifying it
via `certify_spline`. ADR-0022 picked `lam_fallback=0.001` for the fitter it selected
(`spline.fit()`), based on a sweep measured on FULL TRACKS. Before writing `knot_removal.py`,
M1 needs to know whether that choice is affordable at M2's actual scale, where `cost(i,j)` is
evaluated on short segments, not full tracks -- and if not, what to use instead.

## Decision rule (fixed now, before the budget script ran)

1. Measure `certify_spline`'s wall time on representative segments of length 10, 25, 50, 100
   vertices (not full tracks -- a full-track-based estimate was computed first specifically to
   show why it is too pessimistic to be useful, see Outcome), at both `lam_fallback=0.001` and
   `lam_fallback=0.1`.
2. Project M2's total `certify_spline` time for the full grid (spec section 4's data plan) as
   `(estimated DP transitions per track) * (measured per-segment time)`, using the segment-length
   bracket closest to a typical DP candidate segment's own size.
3. **If the `lam=0.001` projection exceeds 2 hours for the full grid**, `knot_removal.py`
   certifies at `lam_fallback=0.1` instead, and compensates for the coarser margin by building
   its starting dense spline via `spline.fit()` at an **internal tol tighter than the target**
   (a tested fraction of `target_tol`, following M0.2's own demonstrated pattern for `tol=0.5 m`
   spline availability) -- the fraction is chosen as the largest tested value whose certified
   pass rate (at `lam=0.1`) stays within 5 percentage points of using `lam=0.001` directly with
   the fit built at the target tol itself (mirroring ADR-0022's own 5pp tie-break convention).
4. Otherwise (`lam=0.001` is affordable), `knot_removal.py` keeps ADR-0022's `lam_fallback=0.001`
   with no internal-tol adjustment.

The rule is applied mechanically to whatever the measurement produces.

## Outcome (`benchmarks/step8_m1_budget.py`, `benchmarks/step8_m1_internal_tol.py`)

**Segment-scale measurement vs. the naive full-track-based estimate:** `certify_spline` on a
50-vertex segment (the bracket closest to a typical DP candidate segment's own length, given the
`m*W` estimate below) costs 24.23 ms at lam=0.001 and 2.70 ms at lam=0.1 -- both far below the
full-track figures ADR-0022 measured (2350.73 ms / 208.48 ms). Projected over the full M2 grid
(15 synthetic tracks x 4 sigma x 3 tol x 2 dt + 585 GeoLife tracks x 2 tol = 1530 cells, `m*W`
DP transitions/cell using `m` proxies of 68.2/track [synthetic, from M0's DP+SED vertex count]
and 150/track [GeoLife, labeled estimate]): the **old, full-track-based estimate said 1045.0 h at
lam=0.001 and 92.7 h at lam=0.1 -- both absurdly over budget, an artifact of treating a short
segment's certification as if it cost the same as certifying an entire track.** The **new,
segment-based estimate is 10.8 h at lam=0.001 and 1.2 h at lam=0.1** -- a 97x/77x reduction from
the naive figure, and a genuinely different qualitative conclusion (lam=0.1 is affordable; lam=
0.001 is not, but by hours, not centuries).

**Rule step 3 triggers:** the segment-based lam=0.001 projection (10.8 h) exceeds the 2 h
threshold. `knot_removal.py` certifies at `lam_fallback=0.1`.

**Internal-tol fraction, tested on the same 4 cells at lam=0.1:** ratio=1.0 (fit directly at the
target tol) gives only 36.7% average certified pass -- far below the 70.0% lam=0.001 reference.
Ratios 0.5, 0.25, and 0.1 all reach **100.0%** average certified pass at lam=0.1 (all comfortably
within the 5pp bar). **Chosen: ratio=0.5** (the largest -- cheapest -- of the three, per the
rule), at an average `n` of 639.2 control points vs. 366.2 at ratio=1.0 (~1.75x more parameters
for a fully reliable certificate, vs. ADR-0022's `lam=0.001`-at-ratio=1.0 path being both slower
*and* only 70% reliable).

**`knot_removal.py`'s starting dense spline is therefore built via `spline.fit(track_or_dense_
curve, tol=0.5 * target_tol)`, and every candidate along the removal path is certified via
`certify_spline(..., lam_fallback=0.1)`.**

Caveat: the segment-scale measurement used a single representative mid-range fitting tol
(`FIT_TOL=2.0`) for the timing-only fits, which converged to the minimal 4-control-point spline
at every tested segment length (10-100 vertices) -- these are absolute-time order-of-magnitude
figures for a go/no-go decision, not a precise runtime model for every `target_tol`; the internal-
tol experiment (which *does* vary `target_tol` per cell, sigma=0/0.1/1/5 with tol=0.5/2/10)
carries the actual decision that matters for correctness, and is not subject to this caveat.

## Consequences

- `src/traj/knot_removal.py` (M1) starts from `spline.fit(x, tol=0.5*target_tol)` and certifies
  every candidate along the removal path with `certify_spline(..., lam_fallback=0.1)` --
  overriding ADR-0022's `lam_fallback=0.001` specifically for this code path, for cost reasons,
  not correctness reasons (ADR-0022's choice remains correct in its own, full-track, M1.1-style
  context; this ADR narrows scope to the segment-scale, high-call-volume context M2 will
  actually use).
- M2's DP cost function should budget roughly 1.2 h of `certify_spline` time for the full grid
  at this configuration (segment-based estimate) -- still a rough figure (see the caveat above),
  but two orders of magnitude more realistic than a full-track-based guess would have been.
- The ~1.75x parameter-count cost of the tighter internal construction is a real cost M2's
  `cost(i,j)` byte accounting already captures directly (bytes, not `n`, gate the DP's choice
  between line and spline candidates) -- no separate adjustment needed, just noted so the
  parameter-count-vs-certificate-reliability trade-off isn't read as free.

## Links

`benchmarks/step8_m1_budget.py`; `benchmarks/step8_m1_internal_tol.py`; `src/traj/knot_removal.py`;
ADR-0022 (the fitter and the full-track-based `lam_fallback` sweep this ADR refines to segment
scale); `benchmarks/results/step8.md` (M1 section); `docs/specs/step8_A_hybrid.md` sections
2.2-2.4 (cost accounting, DP, knot removal).
