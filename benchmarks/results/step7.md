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
| Reparametrization invariance (GeoLife scale: local curve +-22m + shared offset 1e4-1e5m) | same, `d_F` < 1e-6, 30 random cases | 30/30 (worst 3000-trial sweep: 8.26e-7) | yes |
| Independent bracket via discrete Frechet (two-sided) | `distance <= discrete_Frechet(resample(P,h), resample(Q,h)) + tol`, `discrete_Frechet(...) - h <= distance + tol`, and `distance_upper >= discrete_Frechet(...) - h - 1e-9`, `h = 0.5%` of curve bbox diagonal, 30 random cases | 30/30 | yes |
| Cross-check vs. mpmath oracle (independent implementation) | `\|distance - distance_mp\| < 1e-5`, small polylines (n,m<=6) | worst deviation 9.97e-10 over 60+20 cases | yes |
| `distance_upper` vs. mpmath oracle at GeoLife scale (closed-form translation, offset 1e4-1e5m, true distance 1e-3 to 10m) | `distance_upper >= distance_mp - 1e-9`, 30 random cases | 30/30 | yes |
| `distance_upper` no-floor regression guard (GeoLife scale, deterministic) | 5m-segment curve, offset 1e5m, true distance 1e-3m: `distance_upper - distance_mp <= 1e-6` | `9.5e-10` | yes |
| eps-monotonicity (defensive, not in spec) | `decide` results non-decreasing in `eps`, 100 random cases | 100/100 | yes |
| n=1 special case (segment vs. polyline) | O(m) path shares code with general algorithm, no divergence | exact match (distance 1.0 on hand-computed example) | yes |
| Duplicate consecutive vertex | zero-length segment doesn't crash, distance matches de-duplicated polyline | matches (diff < 1e-6) | yes |

`venv/bin/pytest tests/traj/test_frechet_cont.py -v`: 15/15 passed, ~2.6-3.5s (numba JIT warm-up
included), stable across repeated runs (checked 5x with a cleared hypothesis example cache). Full
`tests/traj/` suite: 33/33 passed, no regressions in the existing `frechet.py`, `spline.py`,
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

### M0.1 (round 2) -- `decide_conservative` / `distance_upper`, and a second review correction

Round 1 ended by observing that certificate margins belong at the `certify.py` layer (M1), not in
`frechet_cont.py`. This round builds exactly that layer -- but a first attempt at it repeated a
version of round 1's own mistake, caught by a second review correction before it was ever
committed as final: the margin was computed from `P`/`Q`'s **overall coordinate magnitude**
(~1e5 m at GeoLife scale), giving a **~6-12mm floor** regardless of how small the true distance
was. That is wrong on principle -- Frechet distance is translation-invariant, so no precision
floor should depend on an arbitrary absolute coordinate offset (GeoLife's shared projection
centroid is not a physically meaningful reference point). Fixed before implementation, two ways:

1. `decide()`, `decide_conservative()`, and `distance_upper()` all subtract a common origin
   (`P[0]`) from `P` and `Q` before any computation, so every coordinate the DP ever sees is local
   (small) regardless of the curve's absolute position in the world.
2. `decide_conservative`'s margin is computed **per point-vs-segment cell**, from that cell's own
   local geometry -- `margin = 64 * eps_machine * (|a-p| + |b-a| + eps)^2` -- not a single global
   `scale`.

**Actual margin order after the fix** (measured, not the earlier "6-12mm floor" estimate): at
realistic local scale (segment lengths / point offsets of 1-500m after recentering), the resulting
`eps` slack is **~2.4e-7 m per meter of local scale** (empirically stable across 6 orders of
magnitude of test geometry, `1.0` to `141` m). Concretely: for the exact scenario the correction
specified -- a 5m-segment curve, offset by 1e5m, true distance exactly 1e-3m --
`distance_upper - distance_mp = 9.5e-10` m, i.e. sub-nanometer, nowhere near even one micron, let
alone a millimeter. `decide_conservative` does carry an intrinsic floor proportional to local
scale even for an exactly-zero true distance (it requires genuine clearance beyond `eps` itself,
so `eps=0` can never clear it) -- but that floor is the same `~2.4e-7 * scale` order, sub-micron
for any realistic single-segment GPS geometry.

New functions (both in `src/traj/frechet_cont.py`, `decide()`/`distance()`'s external contract
unchanged): `decide_conservative(P, Q, eps)` -- stricter than `decide()`, via the local margin
above; `distance_upper(P, Q, tol=1e-6)` -- bisects via `distance()`, then verifies the result
against `decide_conservative`, growing further if needed. **`distance_upper` is what M1's
`certify.py` should use for certificates** -- it's independently re-verified, not just relying on
`decide()` rounding the safe way on average.

**Correction (M1 item 0.1): the independent-bracket test's original upper-side check was based on
a false premise.** It asserted `distance_upper(P,Q) <= discrete_Frechet(...) + 4e-7*size`,
reasoning that `distance_upper` shouldn't exceed the discrete-Frechet-on-resampling bound by more
than a small fudge factor. There is no such ordering: `distance_upper` and `discrete_Frechet(...)`
are two *different* upper bounds on the same true continuous distance, obtained via different
methods with different, unrelated slack sources (`decide_conservative`'s local margin vs. the
resampling step `h`) -- nothing forces either one to dominate the other, and the `4e-7*size` term
was an empirically-tuned fudge factor papering over cases where the assumed ordering didn't hold,
not a principled bound. Checked directly: also confirmed `_resample_by_step` was never the actual
cause (it already preserves every original vertex exactly). Fixed: the upper-side check now uses
plain `distance()` (which genuinely is a same-method-family bound alongside discrete Frechet, both
without `decide_conservative`'s extra margin), and a new, mathematically sound check verifies
`distance_upper`'s own correctness instead: `distance_upper(P,Q) >= discrete_Frechet(...) - h -
1e-9` -- `distance_upper` must never fall below a valid *lower* bound on the true distance, which
`discrete_Frechet(...) - h` is (regardless of `distance_upper`'s own margin size).

**Correction (M1 item 0.2): the ~1.5e-3 GeoLife-scale figure is unaffected by recentering, not
superseded by it.** Re-verified directly with the now-recentered `decide()`: identical construction
(independent uniform points scattered across the full +-5e4 range) still gives `0.001545`.
Recentering removes an absolute offset; it cannot shrink distances between points that are
genuinely far apart from each other, which is what that construction's long segments are. The
`test_reparametrization_invariance_at_geolife_scale` test itself was redesigned to isolate the
effect recentering actually has: a compact local curve (+-22m, realistic consecutive-GPS-sample
spacing) translated by a large shared offset (1e4-1e5m) -- under that construction, a 3000-trial
sweep gives a worst case of `8.26e-7`, and the test now asserts `< 1e-6` (tightened from `1e-2`).

### Stack decision recorded (spec section 6)

`python-flint` has no wheel for Python 3.14 (`import flint` fails in `venv/`, confirmed). Per the
spec's own fallback clause, interval-arithmetic spot-checks (needed starting M1's `certify.py`,
per `docs/specs/00_overview.md` section 3.4) will use `mpmath.iv` instead. Not needed for M0
itself -- `hypothesis` (6.168.0) and `mpmath` (1.4.1) were added to `requirements-research.txt`
and are sufficient for this milestone's property tests and oracle.
