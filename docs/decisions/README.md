# Architecture decision records

One ADR per technical decision that affects the certificate guarantee, an algorithm's
correctness contract, or a stack/tooling choice made instead of an available alternative.
Written or updated in the same commit as the code change it documents (per `CLAUDE.md`'s
"Documentation rules"). An ADR is never deleted once a decision it documents ships — if a later
decision replaces it, the later ADR's "Links" section points back, and this one's status becomes
`Superseded by ADR-NNNN`.

## Template

```markdown
# ADR-NNNN: <short title>

- Status: Proposed | Accepted | Superseded by ADR-NNNN
- Date: YYYY-MM-DD

## Context

What problem or question forced this decision. What was true before it.

## Options considered

The real alternatives, briefly, with why each was or wasn't taken.

## Decision

What was actually done.

## Consequences

Including explicitly: what this does to the certificate/correctness guarantee (tighter,
looser, unaffected, newly established) -- not just "what changed" in the code.

## Links

Commits, reports (`benchmarks/results/step7.md`), reviews (`docs/reviews/`), other ADRs.
```

## Index

| # | Title | Status |
|---|---|---|
| [0001](ADR-0001-continuous-frechet-metric.md) | Continuous Fréchet metric instead of discrete | Accepted |
| [0002](ADR-0002-no-fastmath-in-frechet-cont.md) | No `fastmath` in `frechet_cont.py`'s numba kernels | Accepted |
| [0003](ADR-0003-discriminant-clamp.md) | Discriminant clamp at the `Delta~=0` boundary | Superseded by ADR-0004 |
| [0004](ADR-0004-strict-discriminant-check.md) | Strict discriminant check (no clamp) | Accepted |
| [0005](ADR-0005-global-scale-margin.md) | Global-coordinate-scale margin in `decide_conservative` | Superseded by ADR-0006 |
| [0006](ADR-0006-recentering-and-local-margin.md) | Recentering + local per-cell margin in `decide_conservative`/`distance_upper` | Accepted |
| [0007](ADR-0007-rolling-row-dp.md) | Rolling-row (`O(n+m)`) DP kernels above `n*m > 5,000,000` | Accepted |
| [0008](ADR-0008-root-splitting-small-ball.md) | Root-splitting per recursion level + small-ball rule + boundary guard for `certified_linearize` | Accepted |
| [0009](ADR-0009-mpmath-iv-instead-of-flint.md) | `mpmath.iv` instead of `python-flint` | Accepted |
| [0010](ADR-0010-invalid-fit-category.md) | "Invalid fit" category for S2, with a fixed validity threshold | Accepted |
| [0011](ADR-0011-fitter-choice-for-s2.md) | Spline fitter choice for S2 (`fit_adaptive` vs. `fit_uniform`) | Accepted |
| [0012](ADR-0012-s2-invalid-fit-threshold-sensitivity.md) | S2 invalid-fit rate is a fitter property, not a threshold-calibration artifact | Superseded by ADR-0014 |
| [0013](ADR-0013-spline-fit-for-s2.md) | Spec deviation: compare `spline.fit()` against `spline_lsq` for S2 | Accepted |
| [0014](ADR-0014-dense-error-domain-bug.md) | `dense_max_error`'s parametrization-domain bug (ADR-0010/ADR-0012's numbers invalid) | Accepted |
| [0015](ADR-0015-correspondence-point-search.md) | Correspondence-point search for section 2.3 (coarse scan + safeguarded Newton) | Accepted |
| [0016](ADR-0016-projection-certificate-structure.md) | Section 2.3 certificate structure (fixed direction, measured tube, early exit) | Accepted |
| [0017](ADR-0017-lb-needs-adaptive-dense-sample.md) | `hausdorff_lower_bound`'s dense spline sample must be adaptive, not fixed-count-uniform | Accepted |
| [0018](ADR-0018-2-4-primary-spline-certificate.md) | Section 2.4 becomes the primary spline certificate; section 2.3 excluded from the pipeline | Accepted |
| [0019](ADR-0019-interval-queries-design.md) | Interval queries (section 2.5): `eps_A` already equals `epsilon+lambda`, decide-only rule | Accepted |
| [0020](ADR-0020-degenerate-time-breaks-sed.md) | A degenerate time array silently breaks SED simplification (M4's trade-off curve) | Accepted |
| [0021](ADR-0021-certificate-is-additional-not-gating.md) | Step8's reachability gate stays step3's honest-curve error; certificates are additional | Accepted |
| [0022](ADR-0022-spline-fitter-and-lam-for-m1-m2.md) | Spline segment fitter and `lam_fallback` for M1/M2 (pre-registered rule) | Accepted |
| [0023](ADR-0023-knot-removal-lam-and-internal-tol.md) | `knot_removal.py`'s certification `lam_fallback` and internal-tol strategy (segment-scale budget) | Accepted |
