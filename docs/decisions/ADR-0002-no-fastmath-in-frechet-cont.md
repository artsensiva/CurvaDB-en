# ADR-0002: No `fastmath` in `frechet_cont.py`'s numba kernels

- Status: Accepted
- Date: 2026-09-16

## Context

`src/traj/frechet.py` (discrete Fréchet, used only for benchmarking/cross-checks) uses
`@njit(cache=True, fastmath=True)` throughout. When writing `frechet_cont.py`'s free-space-diagram
DP, the same convention was available for the taking, but this DP's correctness depends on exact
comparisons of interval endpoints (`lo <= hi`, `hi == 1.0`, `lo(next) <= hi(prev)`) right at the
discriminant's `Delta ~= 0` boundary, unlike `frechet.py`'s DP which only ever takes `max`/`min`
of plain distances (safe under reassociation).

## Options considered

- **Match the sibling module's convention** (`fastmath=True`): consistent style, modest speed
  gain, but `fastmath` permits reassociation, denormal flush-to-zero, and relaxed NaN/Inf
  handling -- exactly the kind of rounding difference that could flip one of the boundary
  comparisons this DP's correctness leans on.
- **No `fastmath`**: gives up some numba-level speed, keeps IEEE754-standard rounding behavior
  for every comparison the DP makes.

## Decision

`@njit(cache=True)` without `fastmath` for `_free_interval`, `_decide_core`, and every kernel
added since (`_free_interval_conservative`, `_decide_core_conservative`, the rolling-row
variants).

## Consequences

- Certificates built on this metric (`distance_upper`, and everything in `certify.py`) don't
  inherit a hidden risk of `fastmath`-induced sign flips at exactly the boundary where the
  discriminant's rigor matters most.
- Some raw speed left on the table relative to `frechet.py`'s convention; not measured
  separately, judged not worth the correctness risk for a certificate-computing kernel (as
  opposed to `frechet.py`, which only ever feeds benchmark numbers, never a certificate).

## Links

`benchmarks/results/step7.md` (M0 section, "What was built"); commit 4aff3fd.
