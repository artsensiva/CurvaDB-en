# ADR-0011: Spline fitter choice for S2 (`fit_adaptive` vs. `fit_uniform`)

- Status: Accepted
- Date: 2026-09-17

## Context

S2 needs a B-spline representation of each track to certify (spec section 2.1: "LSQ-сплайн
(`src/traj/spline_lsq.py`)" -- the spec names the *module*, not which of its two internal
knot-placement modes to use). M1's original implementation used `fit_adaptive` without an
explicit justification for that choice over `fit_uniform`. Investigated for M1.2.

## Options considered

- **`fit_adaptive`** (two-pass: uniform trial fit -> residuals -> redistribute knots by
  cumulative residual -> refit): places more knots where a *noisy trial fit* residual is
  largest -- not necessarily where the true curve's curvature is highest (`docs/findings.md`'s
  own methodology note on this).
- **`fit_uniform`** (evenly-spaced knots): simpler, no residual-driven redistribution.

## Decision

Ran both on a random 60-track sample (seed 7, from the standard 585-track corpus), comparing
each fit's dense-grid deviation from its track (`traj.spline.dense_max_error`, the same check
ADR-0010 uses for validity):

| Fitter | Invalid rate (10x tol threshold) | Deviation median | p90 | max |
|---|---|---|---|---|
| `fit_adaptive` | 55/60 (91.7%) | 976.8 | 7345.1 | 1,291,308.6 |
| `fit_uniform` | 55/60 (91.7%) | 921.9 | 4903.8 | 21,068,422.6 |

No meaningful difference in invalid rate -- both fail the validity threshold for the large
majority of real tracks (see ADR-0010's consequences and `benchmarks/results/step7.md`'s M1.2
section for why this is a real, expected finding: `spline_lsq.py`'s fitters, unlike `spline.py`'s
`fit()`, have no mechanism at all to control error *between* samples, only *at* them). Where they
differ, `fit_adaptive` has a smaller worst-case deviation (1.3e6 vs. 2.1e7) -- `fit_uniform`'s
uniform knot placement occasionally does noticeably worse in the tail on real, irregularly-sampled
GPS data.

Kept `fit_adaptive` -- no compelling reason to switch, and it has a better (if still bad) tail.

## Consequences

- S2's fitter choice does not explain the high invalid-fit rate found in M1.2 -- that's a
  property of `spline_lsq.py`'s fitters in general on this kind of data, not a specific fitter's
  fault. Switching fitters would not have fixed it.
- If `spline_lsq.py` ever gains a dense-error-aware fitting mode (mirroring `spline.py`'s
  densify/growing-knots mechanism), this decision should be revisited.

## Links

`benchmarks/results/step7.md` (M1.2 section); ADR-0010 (the validity check this comparison used).
