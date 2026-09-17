# ADR-0006: Recentering + local per-cell margin in `decide_conservative`/`distance_upper`

- Status: Accepted
- Date: 2026-09-16

## Context

Same problem as ADR-0005 (need a margin for `decide_conservative` to verify against), after that
global-scale design was rejected for coupling the margin to an arbitrary absolute coordinate
offset rather than the curve's own geometry.

## Options considered

- Global scale (ADR-0005) -- rejected.
- **Recenter + local per-cell margin**: `decide`, `decide_conservative`, `distance_upper` all
  subtract a common origin (`P[0]`) from `P` and `Q` before any computation (so every coordinate
  the DP ever sees is local/small regardless of absolute position), and
  `decide_conservative`'s margin is computed *per point-vs-segment cell* from that cell's own
  local geometry: `margin = 64*eps_machine*(|a-p| + |b-a| + eps)^2`.

## Decision

Implemented the recenter + local-margin design.

## Consequences

- Verified numerically: at realistic local scale (1-500m after recentering), the resulting `eps`
  slack is ~2.4e-7 m per meter of local scale, stable across 6 orders of magnitude of test
  geometry -- sub-micron to tens of microns, not the ~6-12mm ADR-0005 would have given.
  Concretely, for the scenario the correction specified (5m-segment curve, offset 1e5m, true
  distance 1e-3m): `distance_upper - distance_mp = 9.5e-10` m.
- `decide_conservative` carries an intrinsic floor proportional to local scale even for an
  exactly-zero true distance (it requires genuine clearance beyond `eps` itself), but that floor
  is the same sub-micron-per-meter order, not a certificate-relevant concern for realistic
  single-segment GPS geometry.
- `decide()`'s external contract is unchanged (same signature, same result up to float precision)
  -- recentering is translation-invariant by construction, so this is a pure precision
  improvement, never a behavior change.

## Links

Commit 8992db3; `benchmarks/results/step7.md` (M0.1 round 2 section); ADR-0005 (superseded by
this).
