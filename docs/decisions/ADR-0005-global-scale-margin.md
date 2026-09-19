# ADR-0005: Global-coordinate-scale margin in `decide_conservative`

- Status: Superseded by ADR-0006
- Date: 2026-09-16

## Context

Building `distance_upper` (a certificate-grade bound that independently re-verifies its own
bisected result via a stricter `decide_conservative` check, rather than just trusting `decide`'s
average-case-safe rounding) needed a margin to subtract from `eps^2` before testing feasibility,
so that "certified" really means "verified with genuine clearance," not "happened to round the
safe way."

## Options considered

- **Margin from the overall coordinate magnitude of `P`, `Q`** (`scale = max(|P|, |Q|, 1.0)`,
  `margin = 64 * eps_machine * scale^2`): simple, one scale value per call.
  Coordinate magnitude for a GeoLife-projected curve is dominated by the shared projection
  centroid offset (~1e5 m), which is unrelated to the curve's own geometric size.
- **Margin from local, per-cell geometry** (see ADR-0006): more design/implementation work, but
  ties the margin to the actual quantities the discriminant computation depends on.

## Decision (never shipped)

A global-scale margin was designed and about to be implemented for `distance_upper`, but was
caught and rejected during the same planning pass, before any code was written or committed:
Fréchet distance is translation-invariant, so no precision floor should depend on an arbitrary
absolute coordinate offset. Numerically verified before rejecting it: this design would have
given a **~6-12mm floor** on `distance_upper`'s result regardless of how small the true distance
was, at GeoLife's actual coordinate scale.

## Consequences

None shipped -- caught in planning. Recorded here anyway (per the "why" this project cares about,
`docs/findings.md`'s own methodology notes) so this specific design isn't reconsidered later
without remembering why it was rejected the first time.

## Links

Conversation/planning discussion for step7 M0.1 round 2 (no commit -- rejected before
implementation); ADR-0006 (the design that replaced it, which *was* implemented).
