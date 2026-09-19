# step7 phase summary (gate G1)

Certified curve store (`docs/specs/step7_B_certified_store.md`): can `d_F` between an original
track and a compressed representation be certified strictly and densely, and does that let
range queries skip reading the source data? Full detail: `benchmarks/results/step7.md`
(M0-M4); decision history: `docs/decisions/` (19 ADRs).

## What was checked

- **M0**: continuous Fréchet distance (Alt-Godau), cross-checked against an independent `mpmath`
  oracle. **M1**: exact polyline certificates (section 2.2) and certified linearization (section
  2.4, the spline fallback). **M2**: the spec's primary spline certificate (section 2.3, monotone
  projection matching) -- full 585-track validation. **M3**: interval range queries (section 2.5)
  at full scale (1000 queries x 3 ranges x 2 representations, 6M combinations, plus 415
  near-duplicates engineered for the hard `r +/- 2*sum_eps` regime). **M4**: the remaining
  mandatory spec-8 number, a `tol`-vs-certificate-vs-size trade-off curve, and per-query latency.

## Outcome, S1-S6 (spec section 8)

| # | Criterion | Result |
|---|---|---|
| S1 | Polyline certificate correctness | **100%** (585/585 vs. mpmath) |
| S2 | Spline certificate correctness | **100%** (585/585, section 2.4) |
| S3 (density) | median `eps_A/LB` | **passes**: 1.0 (polylines, `<=1.2`), 1.019 (splines, `<=2`) |
| S3 (fallback rate) | `<=10%` of splines via 2.4 | **FAILED for section 2.3**: 83.6% -- resolved by ADR-0018 (2.4 made primary), not revised |
| S4 | 0 misses / 0 false positives | **100%** (0/0 across 6,000,000 combinations, M3) |
| S5 | `>=80%` at `r=200`, either representation (spec's own literal wording) | **passes**: 84.8% (polyline), 83.7% (spline) |
| S5, extended (M3's own choice, beyond spec's literal `r=200`-only scope) | same bar, checked at `r=50`/`r=1000` too | `r=1000` passes (95-96%); `r=50` **falls short** (58-62% at the M1/M2 operating `tol`; recoverable to 93.3% at a tighter `tol=1`, at a real 5.1x storage cost, M4/ADR-0020) |
| S6a | certificate `<=16` bytes/track | **passes** (8 bytes) |
| S6b | query time `<=2x` approximate search | **FAILED**: 2.7x-4.4x (structural: two `decide()` calls vs. one) |

Stop condition (S5 `<50%` for all representations) **not triggered** -- worst case clears 50%.

## Gate decision (G1)

**Open.** G1's own checklist (`docs/ROADMAP.md`) requires S1, S2, S3-density, and S4 -- all four
pass. S3's fallback-rate criterion and S6b are real, documented failures of the spec's *original*
design (section 2.3 as primary; no query-time overhead budget for a two-sided interval check),
not implementation defects (M2/M3's own correctness checks independently confirm this), and are
recorded as fixed limitations, not reopened. See `docs/ROADMAP.md` section 8 for the exact
decision text and ADR-0018 for the architecture change (section 2.4 primary, section 2.3 kept as
a verified non-default alternative).

## What's reusable

`src/traj/frechet_cont.py` (Alt-Godau + certificate-grade `decide_conservative`/`distance_upper`),
`bezier.py` (knot insertion, Bezier extraction, de Casteljau), `certify.py` (all three certificate
paths, `certify_spline`'s `use_projection` switch), `intervals.py` (the decide-only interval rule
+ cheap P3-style filters) are general-purpose and depend only on `numpy`/`scipy`/`numba` -- usable
directly by step8 and the `certigeo` core extraction (phase 3) without changes.

## What we don't know

Whether the S5/`r=50` gap is worth closing in a real product by defaulting to a tighter `tol`
(M4's trade-off curve gives one data point, not a full sweep against real query-radius
distributions); how S6b's overhead behaves under a real query load (not just this benchmark's
synthetic GeoLife-based one); whether section 2.3's structural fallback-rate problem is fixable
with a different piece-decomposition strategy (not attempted -- out of scope once ADR-0018 made
2.4 primary). None of these block G1; they are candidate follow-ups for step8 or a future
revisit, not open correctness questions.
