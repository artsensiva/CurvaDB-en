# ADR-0008: Root-splitting per recursion level + small-ball rule + boundary guard for `certified_linearize`

- Status: Accepted
- Date: 2026-09-17

## Context

Spec section 2.4 claims the certified-linearization path is "always applicable"
(`Путь всегда применим`). Empirically false: 46/585 (7.9%) real tracks got no certificate at
`lam=0.1, max_levels=12` under the original design (blind `t=0.5` de Casteljau bisection,
re-testing tube+monotonicity against each half's own chord). Diagnosed cause: blind bisection can
need far more levels than the budget allows to resolve a genuine sign change in the derivative's
chord-projection (confirmed: a synthetic reversal case needed ~20 levels, not 12), and
near-stationary segments make the chord *direction* itself ill-conditioned regardless of depth.

## Options considered

- **Raise `max_levels`**: doesn't help persistent sign changes (verified: escalating a real
  failing case from 12 to 20 levels didn't resolve it) and just delays the same problem.
- **Root-split once, at the top-level segment's own chord, before recursing**: simpler, but
  monotonicity along the *original* chord doesn't imply monotonicity along a sub-piece's own
  (possibly quite different) chord after splitting -- doesn't actually fix the general case.
- **Root-split at every recursion level, relative to the current piece's own chord** (the option
  taken): finds roots of `<C'(u), e>` (`e` = that piece's own chord direction) and splits there,
  falling back to plain bisection when the chord is degenerate or no root exists.

## Decision

Implemented per-level root-splitting, plus two refinements found necessary during
implementation, not anticipated by the initial design:

1. **Small-ball rule**: if a piece's control points all lie within `rho` of its own start and
   `2*rho <= lam`, it's certified without checking monotonicity at all (any pairing between a
   curve point and a chord point is then at most `2*rho` apart, regardless of monotonicity) --
   handles near-stationary segments where a chord direction is meaningless.
2. **Boundary-margin guard**: a root is only used as a split point if it falls in
   `[0.05, 1-0.05]`; otherwise, fall back to plain `t=0.5` bisection. Found necessary after the
   mandatory 100-sample regression check (re-running S2 on previously-*passing* tracks) caught a
   real bug: without this guard, a root chasing progressively closer to one endpoint at every
   level burns the whole recursion budget on a razor-thin sliver each level, without shrinking
   the actual problem -- 51/100 tracks regressed. With the guard: 0/100.

Root-finding (`_projection_roots_in_unit_interval`) is used **only** to pick split points. The
sole certification authority is `_certified_ok` (renamed from `_tube_and_monotone_ok`), checked
fresh after every split; the heuristic never bypasses it, so the *rigor* of what counts as
certified is unaffected by how the split point was chosen -- only how quickly (or whether, within
`max_levels`) a genuinely certifiable piece gets found.

## Consequences

- 22/46 formerly-uncertified tracks now get a valid certificate (vs. 0/46 before) -- closely
  matches the 20/46 tracks independently classified as having a "reasonable" (non-diverged)
  spline fit (see ADR-0010), confirming the fix targets genuine geometric edge cases and not the
  separate spline-fit-instability issue.
- Recursion depth for the newly-certifying tracks is still fairly deep in practice (mean 9.9,
  vs. the synthetic reversal case's 2) -- real GPS-derived geometry is noisier than a clean
  synthetic sign change. The meaningful outcome is going from *no certificate at any depth* to
  *a valid one*, not a dramatic depth reduction.
- The certificate guarantee itself is unaffected in the strict sense (`_certified_ok` never
  bypassed) -- this ADR only changes *how many* tracks can be certified within a given
  `max_levels` budget, never *what "certified" means*.

## Links

Commits 5fd4b5f (initial), 730cb77 (boundary-margin fix), 882bbaf (diagnostic tooling fix);
`docs/reviews/step7_M1.md`; `benchmarks/results/step7.md` (M1.1 section).
