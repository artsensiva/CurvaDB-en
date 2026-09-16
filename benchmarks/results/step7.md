# Step7 (direction B): certified curve store

## M0 -- continuous Frechet metric

**Artifact target (spec table, section 7):** correctness of the metric.

Scope: `src/traj/frechet_cont.py` (Alt-Godau free-space-diagram decision + bisection distance,
numba), an independent `mpmath` high-precision oracle for cross-checking, and the property
tests mandated by spec section 9. No certificates (`certify.py`) or spline logic yet -- those are
M1/M2. The S1-S6 acceptance criteria (spec section 8) attach to later milestones, not M0; this
section reports metric-correctness test results instead.

### What was built

- `decide(P, Q, eps) -> bool` and `distance(P, Q, tol=1e-6) -> float`: the continuous Frechet
  distance between polylines, via the classic free-space-diagram reachability DP, `O(nm)` time,
  numba-jitted (no `fastmath` -- the DP's correctness hinges on exact interval-endpoint
  comparisons near the discriminant's `Delta ~= 0` boundary, where `fastmath`'s relaxed rounding
  is most likely to flip a result). `distance()` bisects on `eps` and returns the certified upper
  end (`hi`), matching the spec's `eps_A = hi` convention (section 2.2).
- The single-segment-vs-polyline `O(m)` case (spec section 5's file description) needs no
  separate function: setting `n=1` makes the general algorithm's outer loop run once, degenerating
  to `O(m)` automatically -- verified by a dedicated test (`test_single_segment_vs_zigzag_n_equals_1`).
- `_free_interval` is deliberately **strict** at the discriminant's `Delta ~= 0` boundary (a point
  exactly on a segment's *line* is a mathematically repeated root, and float64 cancellation in
  `B*B - 4*A*C` can push the residual a hair negative) -- reporting "not yet feasible" a fraction
  of an `eps` early is the only safe direction for a value meant to be a certified upper bound
  later. See "M0.1" below: an earlier version of this got the direction backwards.
- `tests/traj/_frechet_cont_mpmath.py`: an independent high-precision oracle (pure Python +
  `mpmath`, no shared code with the numba implementation) used only in tests, to cross-check the
  DP's correctness rather than merely re-running the same logic at higher precision.

### Test results (spec section 9 property tests + defensive extras)

| Criterion | Threshold | Actual | Passed |
|---|---|---|---|
| decide/distance consistency | `decide(d+1e-4)` true, `decide(d-1e-4)` false, 100 random cases | 100/100 | yes |
| Symmetry | `d_F(P,Q) == d_F(Q,P)`, 100 random cases | 100/100 | yes |
| Triangle inequality | `d_F(P,R) <= d_F(P,Q)+d_F(Q,R)+1e-9`, 100 random triples | 100/100 | yes |
| Reparametrization invariance (coords +-50) | inserting exact-collinear vertices leaves `d_F` < 1e-4 (see M0.1), 100 random cases | 100/100 | yes |
| Reparametrization invariance (GeoLife scale, +-5e4) | same, `d_F` < 1e-2, 30 random cases | 30/30 | yes |
| Bounded above by dense discrete Frechet (independent algorithm) | `d_F <= discrete_Frechet(densify(P,25), densify(Q,25)) + 1e-6`, 30 random cases | 30/30 | yes |
| Cross-check vs. mpmath oracle (independent implementation) | `\|distance - distance_mp\| < 1e-5`, small polylines (n,m<=6) | worst deviation 9.97e-10 over 60+20 cases | yes |
| eps-monotonicity (defensive, not in spec) | `decide` results non-decreasing in `eps`, 100 random cases | 100/100 | yes |
| n=1 special case (segment vs. polyline) | O(m) path shares code with general algorithm, no divergence | exact match (distance 1.0 on hand-computed example) | yes |
| Duplicate consecutive vertex | zero-length segment doesn't crash, distance matches de-duplicated polyline | matches (diff < 1e-6) | yes |

`venv/bin/pytest tests/traj/test_frechet_cont.py -v`: 13/13 passed, ~2.5-3.5s (numba JIT warm-up
included), stable across repeated runs (checked 4x with a cleared hypothesis example cache). Full
`tests/traj/` suite: 31/31 passed, no regressions in the existing `frechet.py`, `spline.py`,
`spline_lsq.py` tests.

### M0.1 -- review fixes (`docs/reviews/step7_M0.md`)

The review caught a real correctness bug in M0's original discriminant-cancellation fix. That
version clamped small negative discriminant residuals (near the `Delta ~= 0` boundary) to zero
*to make `decide` more permissive* -- reasoning that a wider free interval was "the safe
direction." That reasoning was backwards. Concretely verified: at GeoLife-realistic coordinate
scale (~1e5 m), the clamp made `decide(P, Q, eps)` return `True` for `eps` up to **~1e-4 below**
the true distance -- meaning `distance()` could genuinely *underestimate* `d_F`, which directly
violates the certificate requirement (spec S1: `eps_A >= d_F`). Fixed by reverting to a strict
discriminant check (no clamp) -- findings and resolutions below.

| # | Finding | Resolution |
|---|---|---|
| 1 | Clamp direction unsafe for certified upper bounds | Reverted; `_free_interval` is now strictly conservative (only ever reports "not yet feasible" a hair early, never the reverse) |
| 2 | mpmath oracle shares the same DP recurrence -- doesn't catch algorithmic errors | Added `test_bounded_above_by_dense_discrete_frechet`: cross-checks against `traj.frechet`'s discrete Frechet (a genuinely different algorithm) on densely resampled curves, using the fact that discrete Frechet on *any* sampling is a guaranteed upper bound on continuous Frechet |
| 3 | No tests at realistic (GeoLife) coordinate scale | Added `test_reparametrization_invariance_at_geolife_scale` (coords +-5e4 m) |
| 4 | `decide()` allocates `O(nm)` arrays per call -- relevant to M3's query workload | Recorded in `TODO.md`; not fixed now, no query benchmark yet to size the tradeoff against |

Reverting the clamp reintroduces a much smaller, *unavoidable* float64 slack at the `Delta ~= 0`
boundary -- ordinary cancellation noise in `B*B - 4*A*C`, not an artificially introduced margin.
Quantified empirically (1000-trial and 500-trial sweeps of the reparametrization-invariance
construction): up to **~1.5e-6** at coordinate scale +-50 (the existing property tests' range),
up to **~1.5e-3** at GeoLife scale (~5e4 m) -- both always in the safe (overestimate) direction.
The GeoLife-scale figure is close to the spec's own certificate precision convention
(`eta = 1mm`, section 2.2) -- exactly the kind of slack M1's `certify.py` margin
(`delta = 64 * eps_machine * L`, section 3.4) needs to absorb, confirming that margin belongs at
the certificate layer, not inside this raw metric.

### Stack decision recorded (spec section 6)

`python-flint` has no wheel for Python 3.14 (`import flint` fails in `venv/`, confirmed). Per the
spec's own fallback clause, interval-arithmetic spot-checks (needed starting M1's `certify.py`,
per `docs/specs/00_overview.md` section 3.4) will use `mpmath.iv` instead. Not needed for M0
itself -- `hypothesis` (6.168.0) and `mpmath` (1.4.1) were added to `requirements-research.txt`
and are sufficient for this milestone's property tests and oracle.
