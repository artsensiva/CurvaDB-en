# ADR-0003: Discriminant clamp at the `Delta~=0` boundary

- Status: Superseded by ADR-0004
- Date: 2026-09-16

## Context

A point exactly on the infinite line through a segment's endpoints gives the point-vs-segment
quadratic a mathematically repeated root (`Delta == 0` exactly). Computing `B*B - 4*A*C` in
float64 can push that residual slightly negative from cancellation alone, making `_free_interval`
report "empty" for a point that's genuinely touching the segment. This surfaced as a real (not
hypothetical) failure in the reparametrization-invariance property test on an axis-aligned
configuration.

## Options considered

- **Clamp**: if the discriminant is negative but within a `64 * eps_machine * (B^2 + |4AC|)`
  margin of zero, treat it as zero (feasible) instead of empty.
- **Leave strict**: report "empty" whenever the discriminant is negative, even by a hair.

## Decision (later reversed)

Implemented the clamp in M0 (commit 4aff3fd), reasoning that widening the feasible region was
"the safe direction" for a value meant to become a certified upper bound.

## Consequences (why this was wrong)

The review (`docs/reviews/step7_M0.md`) caught that this reasoning was backwards: the margin
scales with the *absolute* magnitude of `B`, `A`, `C`, which scale with the *raw coordinate
offset* the curve happens to sit at (e.g. ~1e5 m for GeoLife's shared projection centroid) --
concretely verified, this made `decide(P, Q, eps)` return `True` for `eps` up to **~1e-4 below**
the true distance at that scale, i.e. `distance()` could genuinely *underestimate* `d_F`,
violating the certificate requirement (spec S1: `eps_A >= d_F`). Reverted in favor of a strict
check (ADR-0004).

## Links

Commit 4aff3fd (introduced); `docs/reviews/step7_M0.md` (finding 1); commit c4ed9c3 (reverted);
ADR-0004.
