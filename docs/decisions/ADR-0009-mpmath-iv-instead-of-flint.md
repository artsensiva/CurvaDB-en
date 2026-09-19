# ADR-0009: `mpmath.iv` instead of `python-flint`

- Status: Accepted
- Date: 2026-09-16

## Context

Spec section 6 ("Стек") names `python-flint` as the preferred library for control-recompute
interval arithmetic (`docs/specs/00_overview.md` section 3.4's floating-point rigor convention),
with an explicit fallback clause: if no wheel is available for the project's Python version, use
`mpmath.iv` instead and record the substitution.

## Options considered

- **`python-flint`**: confirmed unavailable -- `import flint` fails in `venv/` (Python 3.14.4, no
  published wheel for that version at the time of checking).
- **`mpmath.iv`**: already a project dependency (added for the high-precision oracle,
  `tests/traj/_frechet_cont_mpmath.py`), provides interval arithmetic, slower than flint's `arb`
  type but adequate for the spot-checks the spec calls for (not a hot path).

## Decision

Use `mpmath.iv` wherever the spec calls for control-recompute interval arithmetic. Not yet
exercised in the codebase as of M1 (no milestone has needed an interval-arithmetic spot-check
yet -- M0/M1 used mpmath's plain high-precision arithmetic, not its interval type); this ADR
records the fallback decision for whenever M1's `certify.py` or a later milestone needs it.

## Consequences

- No functional difference expected for correctness (both give rigorous interval bounds); flint's
  `arb` would likely be faster if it becomes a bottleneck later, worth revisiting if
  interval-arithmetic spot-checks turn out to run often enough for that to matter.
- Recorded explicitly per the spec's own instruction to note the substitution.

## Links

`docs/specs/step7_B_certified_store.md` section 6; `benchmarks/results/step7.md` (M0 section,
"Stack decision recorded").
