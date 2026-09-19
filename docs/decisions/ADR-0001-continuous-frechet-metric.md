# ADR-0001: Continuous Fréchet metric instead of discrete

- Status: Accepted
- Date: 2026-09-16

## Context

Step7's certified store needs a distance metric between a stored curve and its approximation
that doesn't depend on how densely either one happens to be sampled. `docs/findings.md`'s step0
history is the direct lesson: a naive comparison using the *discrete* Fréchet distance
(`src/traj/frechet.py`, Eiter-Mannila) made the spline representation look far worse than DP
purely because of a methodology bug (error checked only at original timestamps, not between
them) -- but the deeper, permanent issue discrete Fréchet has is that it is a property of the
*sample points chosen*, not of the underlying curves: resampling either curve more densely
changes the discrete distance even when the curves themselves haven't changed.

## Options considered

- **Discrete Fréchet on the stored curves' own vertices** (already implemented, used throughout
  step0-step3 for evaluation): fast (numba, `O(nm)`), but the certificate would then be a
  statement about a specific vertex sampling, not about the curves -- exactly the property that
  caused step0's methodology bug.
- **Continuous Fréchet via the Alt-Godau free-space-diagram algorithm**: `O(nm)` decision, `O(nm
  log(range/tol))` for the distance via bisection on `eps`; the result doesn't depend on sample
  density, only on the curves' actual shape. Matches spec section 2's explicit design ("непрерывная
  метрика вместо дискретной (step0)") and `docs/specs/00_overview.md` section 3.3.

## Decision

Implement the continuous metric (`src/traj/frechet_cont.py`: `decide`/`distance`) via Alt-Godau,
and keep the existing discrete implementation (`src/traj/frechet.py`) only as an independent
cross-check tool in tests (e.g. `test_independent_bracket_via_discrete_frechet`), never as the
metric certificates are computed against.

## Consequences

- Certificates (`certify.py`) are statements about the true curves, not about whatever vertex
  density a particular representation happens to have -- this is the core guarantee the whole
  certified-store idea depends on.
- Implementation cost: a nontrivial free-space-diagram DP (see ADR-0002/0003/0004 for the
  correctness issues that surfaced building it), versus the already-existing, already-tested
  discrete implementation. Judged worth it given the guarantee discrete Fréchet cannot provide.

## Links

`benchmarks/results/step7.md` (M0 section); `docs/specs/step7_B_certified_store.md` section 2,
`docs/specs/00_overview.md` section 3.3; commit 4aff3fd.
