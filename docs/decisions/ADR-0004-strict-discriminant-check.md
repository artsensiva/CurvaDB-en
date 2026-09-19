# ADR-0004: Strict discriminant check (no clamp)

- Status: Accepted
- Date: 2026-09-16

## Context

ADR-0003's clamp made `decide()`/`distance()` unsafe (could underestimate `d_F` by ~1e-4 m at
GeoLife coordinate scale). The underlying numerical issue it was trying to paper over is real
(float64 cancellation right at a genuinely repeated root), but the fix direction was wrong.

## Options considered

- **Keep clamping, shrink the margin**: still couples the margin to absolute coordinate
  magnitude, just less aggressively -- doesn't fix the principle (Fréchet distance is
  translation-invariant; nothing about the true answer depends on where the curve sits in an
  arbitrary coordinate frame).
- **Revert to strict** (`disc < 0.0` always means empty): the only remaining slack is *ordinary*
  float64 cancellation noise in computing `B*B-4*A*C` itself -- unavoidable, but empirically much
  smaller (microns to sub-micron at realistic geometry, vs. millimeters with the clamp), and,
  crucially, it can only ever make `decide` report "not yet feasible" a hair early -- never the
  unsafe direction.

## Decision

Reverted to the strict check. Any additional slack a certificate needs belongs at the
`certify.py` layer (see ADR-0006), not inside the raw metric.

## Consequences

- `decide()`/`distance()` are safe again: verified directly that the underestimation window
  collapses from ~1e-4 (clamp) back to ordinary float64 precision (~1e-9 relative) at the same
  GeoLife scale.
- Reintroduces a small, unavoidable conservative slack at genuinely-touching configurations
  (documented and budgeted for by test tolerances, e.g.
  `test_reparametrization_invariance_at_geolife_scale`) -- this is the *safe* direction (only
  ever overestimates), so it doesn't threaten the certificate guarantee.

## Links

Commit c4ed9c3; `benchmarks/results/step7.md` (M0.1 section); ADR-0003 (superseded by this).
