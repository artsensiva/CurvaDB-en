# ADR-0015: correspondence-point search for section 2.3 (coarse scan + safeguarded Newton)

- Status: Accepted
- Date: 2026-09-18

## Context

Spec section 2.3 item 1 requires, for each original vertex `V_k`, a parameter `u_k` on the
spline (the nearest point, "Newton's method from the previous `u_{k-1}`"), with the constraint
`u_0=0 <= u_1 <= ... <= u_n=1`. `u_0`/`u_n` are fixed to the spline's own domain bounds
(`bs.t[0]`/`bs.t[-1]`); `u_k` for interior `k` must be searched.

## What was tried, and what broke

**Plain clamped Newton** (`f(u)=|C(u)-V_k|^2`, clamped into `[u_{k-1}, u_max]` at every step,
started from `u_{k-1}`): worked for simple cases, but on a real track produced a catastrophic
failure. Four consecutive vertices collapsed onto the same early `u` (their true nearest points
were behind `u_{k-1}`, so clamping correctly held them there) with growing residuals -- then the
*next* vertex's search, starting from that same stuck `u`, took one poorly-conditioned Newton
step (tiny second derivative near an inflection), overshot to `u_max`, and *falsely converged*
there: the post-clip position stopped changing between iterations, tripping the step-size
convergence check, even though `u_max` was a far worse point (residual 88 m) than where the
search started (residual 83 m). The resulting piece spanned `[0.07, 1.0]` -- nearly the entire
spline -- and took 100-1000x longer to (fail to) certify than a normal piece.

**Fix 1 -- safeguarded (backtracking) Newton**: a step is only *accepted* if it does not increase
the squared residual; otherwise the step is halved (up to 20 times) before giving up and stopping
at the last accepted point. Since the search always starts at `u_guess = u_lo` (this module's own
calling convention), the result can never be worse than `u_lo` by construction -- removes the
false-convergence pathology entirely (confirmed: the same track no longer produces the runaway
piece). But this alone does not fix everything: a *purely local* method starting from a stuck
point has no way to discover a much better match far away in parameter space if the local
gradient doesn't point there -- confirmed on the same track: vertex 20's true nearest point sits
at distance 0.34 m, but safeguarded Newton from the previous (already stuck) vertex's `u`
converges to a *different, disconnected* local optimum at distance 83 m, since the real minimum
is in a basin the local method can never reach from that starting point.

**Fix 2 -- coarse pre-scan**: 50 evenly spaced samples across `[u_{k-1}, u_max]`, keeping
whichever of {best sample, `u_guess`} has the smaller residual as Newton's actual starting point,
before running safeguarded Newton to refine it. Cost is a *fixed* 50 extra spline evaluations per
vertex, independent of how large the remaining range is -- confirmed this finds vertex 20's real
minimum (distance 0.34, not 83) and, combined with Fix 1, brings per-track certification time on
several real tracks from a >100 s hang down to well under 1 s.

## Decision

Correspondence-point search is: 50-point coarse scan of `[u_lo, u_hi]`, pick the better of the
best sample and the caller's own `u_guess`, then safeguarded (backtracking) Newton refinement
from there, clamped to `[u_lo, u_hi]` throughout. `u_lo = u_{k-1}`, `u_hi = u_max` (the spline's
own domain end) for every interior vertex -- monotonicity is guaranteed by this range restriction
alone, not by any property of the search method.

**Optimality is explicitly not required for the certificate's validity** -- only correctness.
Whatever `u_k` this search returns, `certify_spline_projection` independently *re-measures*
`delta_k = |C(u_k) - V_k|` by direct evaluation and folds it into the cost bound honestly; a
worse-than-optimal `u_k` only makes the certificate looser (larger `eps_A` or, in the extreme, a
piece that fails to certify and falls back), never wrong. This is why Fix 1/Fix 2 are framed as
*quality/performance* fixes, not correctness fixes -- the pre-fix code was already producing valid
(if occasionally absurdly loose and catastrophically slow) certificates, never a silently wrong
number.

## Consequences

- Per-vertex cost is `O(1)` (50 fixed samples + a small bounded number of Newton/backtracking
  iterations), not `O(n)` or worse -- scales to the full 585-track corpus.
- The certificate's tightness now depends on how good this heuristic search is, not just on
  whether *a* valid `u_k` exists -- a materially better search (e.g. a wider coarse grid, or a
  proper global optimizer) could tighten `eps_A` further in a future milestone if warranted; not
  pursued here since 50 samples already fixed the concrete pathology found.
- Degenerate case `u_k == u_{k+1}` (the search can still return the same `u` for two consecutive
  vertices, e.g. when both are genuinely closest to the same spline point) is handled explicitly
  by `certify_spline_projection`'s per-piece logic (ADR-0016), not by this search itself.

## Links

`src/traj/certify.py`'s `_nearest_point_on_spline`/`_correspondence_points`; ADR-0016 (the
per-piece certificate this feeds); ADR-0008 (the analogous "root-finding is heuristic,
verification is separate" discipline for section 2.4, same project convention applied here to a
different search problem).
