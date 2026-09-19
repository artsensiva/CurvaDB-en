# ADR-0007: Rolling-row (`O(n+m)`) DP kernels above `n*m > 5,000,000`

- Status: Accepted
- Date: 2026-09-17

## Context

`_decide_core`/`_decide_core_conservative` materialize full `O(nm)` arrays
(`left_lo/hi`, `bot_lo/hi`, the reachability arrays). Fine for M0/M1's per-track certificate
computation at typical sizes, but M1's S2 certifies an original track against a *linearized
spline*, where both `n` (original track segments) and `m` (linearized-spline segments) can be
large enough that `n*m` becomes a real memory concern -- flagged by the M0 review (finding 4) as
relevant once an actual large-`n*m` workload existed.

## Options considered

- **Leave the `O(nm)` kernels as the only path**: simplest, but risks large memory use (or worse)
  for the largest tracks once S2 exercises them at scale.
- **Always use a rolling-row (`O(m)` memory) kernel**: removes the risk uniformly, but means
  replacing the *already heavily-validated* (multiple rounds of scrutiny across M0/M0.1) `O(nm)`
  kernels entirely -- unnecessary risk to a well-tested hot path for the common (small `n*m`)
  case.
- **Add rolling-row kernels as an alternate path, auto-selected above a size threshold**: keeps
  the existing kernels untouched below the threshold; only the rare large-`n*m` case takes the
  new path.

## Decision

Added `_decide_core_rolling`/`_decide_core_conservative_rolling` (same recurrence, only the
current and previous DP row kept, `O(m)` memory). `decide()`/`decide_conservative()`
automatically switch to the rolling kernel when `(P.shape[0]-1) * (Q.shape[0]-1) > 5_000_000`.

## Consequences

- No change to correctness or behavior below the threshold (untouched code path).
- Above the threshold, a *new* code path with its own correctness risk (a rolling-row
  transformation is exactly the kind of change that can silently drop a boundary case) --
  mitigated by a mandatory equivalence test (`test_rolling_dp_matches_full_dp`, 200 random
  `(n,m,eps)` cases plus ad-hoc stress sweeps, 0 mismatches) before trusting it.
- The threshold (5,000,000) is a judgment call, not derived from a memory budget calculation --
  revisit if M3's query workload profile turns out to need a different cutoff.

## Links

Commit a2dfcd3; `TODO.md` (the finding this resolves); `docs/reviews/step7_M0.md` (finding 4).
