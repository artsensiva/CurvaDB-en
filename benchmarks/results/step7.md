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

## M1 -- polyline and linearization certificates

Corpus: `load_clean_tracks(n=200, seed=42)` -> 585 cleaned track segments from 200 raw tracks (the project's standard corpus, per `traj.io.DEFAULT_N_TRACKS`).

| Criterion | Threshold | Actual | Passed |
|---|---|---|---|
| S1: polyline certificates (`certify_polyline`, eta=1e-06) vs. mpmath reference (per piece, dps=40, tol=1e-06) | `eps_A >= reference - 1e-06` for 100% of tracks | 585/585 (worst margin -5.000e-07) | yes |
| S2: spline certificates via certified linearization (`certify_spline_linearization`, lam=0.1, eta=0.001) vs. near-exact reference (lam=0.0001) | `eps_A >= reference - 0.0001`, `eps_A` finite, for 100% of tracks with an available reference | 539/585 (46 fallback [uncertified linearization at lam=0.1], 0 reference-unavailable) | yes |

**This S2 row is the original M1 measurement, kept as the historical record of that first run --
it predates the "invalid fit" category (M1.2, ADR-0010) and the fitter-comparison/bug-fix (M1.3,
ADR-0013/ADR-0014) and is superseded by them for anything about *current* S2 coverage.** See the
M1.3 section below and the M1 Conclusions for the current picture (`spline.fit()`, 585/585 valid,
585/585 pass) -- this row's "46 fallback, yes" should not be read as still describing today's S2.

**Informational (not a gate):** median `eps_A/LB` for polylines (LB = directed Hausdorff distance both ways, spec section 2.5) = **1.000** -- a preliminary reading of spec S3's density criterion (median <= 1.2 for polylines) for gate G1; S3 itself is not an M1 gate and is not enforced here.

**S2 pilot** (first 40 tracks, measured separately before the full run): 383.3s wall time, peak Python-tracked memory 236.5 MB (`tracemalloc`). Linear extrapolation to the full 585-track corpus predicted ~5605s; the actual full run took **2217.6s** (~37 minutes) -- the pilot's first-40 tracks were evidently not representative of the corpus average (front-loaded by some slower/larger tracks), so the projection overestimated by ~2.5x. Noted for future pilots: track order isn't a reliable proxy for per-track cost here: consider a random subsample for timing estimates instead of a fixed prefix.

**Fallback / reference-unavailable tracks are a real, expected outcome** (spec section 10's acknowledged risk: spline loops relative to a chord break the monotonicity test at any subdivision depth) -- not a bug. `certify_spline_linearization` returns `(inf, False)` for these (item 0.4), and they are excluded from the S2 pass/fail count, not silently treated as passes.

`decide()`/`decide_conservative()`'s rolling-row kernels (item 4) were exercised transparently wherever a track's `n * m` (original vs. linearized-spline segment count) exceeded 5,000,000 during S2 -- no separate accounting needed, both code paths are verified equivalent (`test_rolling_dp_matches_full_dp`).

### M1.1 -- spec section 2.4 correction (`docs/reviews/step7_M1.md`, `docs/ROADMAP.md` section 8)

**Spec error confirmed**: section 2.4's claim that the certified-linearization path is "always
applicable" (`Путь всегда применим`) is empirically false -- 46/585 (7.9%) tracks got no
certificate at `lam=0.1, max_levels=12`. Root cause (diagnosed before any code change): blind
`t=0.5` bisection can take far more levels than the budget allows to resolve a genuine sign
change in the chord-projected derivative (a synthetic case needed ~20 levels, not 12), and
near-stationary segments make the chord *direction* itself ill-conditioned regardless of depth.

**Diagnostics (item 0, run against the pre-fix code, all 585 tracks):**

| Statistic | Value |
|---|---|
| eps_A/LB (polylines): p50 / p90 / p99 / max / frac exactly 1.0 | 1.0000 / 1.1611 / 2.0710 / 6.5847 / 0.817 |
| 46 failures by check | monotonicity: 36, tube: 1, tube+monotonicity: 9 |
| Tracks exceeding the rolling-DP threshold (`n*m > 5,000,000`) during S2 | 0/585 |

**Unanticipated finding**: the raw chord-length/speed numbers for the 46 failures are physically
absurd (mean chord ~2.4e13 m, speeds up to ~4.6e17 m/s) -- not a diagnostic bug. At least 26/46
have a `fit_adaptive` spline fit with wildly unstable control points (confirmed directly: one
track's control points span +-1e12 while the track itself is ~2km), a separate, pre-existing
numerical-instability issue in `spline_lsq.py` (already flagged in its own docstring), out of
scope for this fix. Classifying all 46 by whether the fit's control points stay within 100x the
track's own extent: **26/46 unstable fits** (this fix can't help -- there is no reasonable
certificate to find), **20/46 reasonable fits** (plausible genuine geometric edge cases this fix
should address).

**Fix**: `_certify_segment` now finds roots of `<C'(u), e>` relative to the *current* piece's own
chord at every recursion level (not just once at the top), falling back to plain bisection when
the chord is degenerate, no root exists, or every root is within `0.05` of an endpoint. That last
guard was not part of the original design -- it was added after the mandatory 100-sample
regression check (re-running S2 on previously-*passing* tracks) caught a real bug: root-splitting
without it caused **51/100 regressions** (a root chasing progressively closer to one endpoint at
every level, burning the recursion budget on a razor-thin sliver each time without shrinking the
problematic remainder). After the boundary-margin fix: **0/100 regressions** (100/100 pass,
re-verified). `_certified_ok` also gained the small-ball rule (certifies without checking
monotonicity when the whole piece fits in a ball of diameter `<= lam`).

**Re-validation results:**

| Check | Result |
|---|---|
| 46 former failures, re-tested | **22/46 now get a finite certificate** (vs. 0/46 before) -- closely matches the 20/46 "reasonable fit" classification, confirming the fix targets genuine geometric cases, not the separate spline-instability issue; 24/46 still fallback |
| Recursion depth, 46 tracks' worst segment, before (old algorithm) | mean 6.41, max 12 |
| Recursion depth, 46 tracks' worst segment, after (new algorithm), split by outcome | now-certifying (22): mean 9.9, max 12; still-fallback (24): mean 12.0, max 12 (exhausts the budget, expected for genuinely unresolvable/unstable cases) |
| Random 100-sample of previously-*passing* tracks, re-tested | **100/100 pass, 0 fallback** -- no regression |

Depth alone understates the fix's value: the "now-certifying" cohort still averages a fairly deep
9.9 levels (real GPS-derived geometry is noisier than the synthetic reversal test case, which
resolves in 2), but the meaningful outcome is 22 tracks moving from *no certificate at any depth*
to *a valid one* -- something the old algorithm could never produce regardless of `max_levels`.

**Full-corpus re-run: not executed at M1.1 time, per the timing budget.** The 100-sample took
703.0s (7.03s/track, includes root-finding overhead vs. the original algorithm) -- extrapolated to
the full 585-track corpus, **~4113s (~68.5 minutes), exceeding the 40-minute budget**. Per the
instruction to record this decision rather than force a long run blind: the full corpus was not
re-run at that time. Combining the confirmed no-regression 539 passes with the 22 newly-certifying
tracks gave an **estimated (not directly verified) 561/585** S2 pass count post-fix -- reported as
an estimate, not a fact. M1.2 (below) replaces this estimate with an actual full-corpus run.

### M1.2 -- ADR system, S2 fit-validity gate, full-corpus S2 re-run, eps_A/LB tail

**ADR system**: retroactive ADRs 0001-0011 written for the full M0-M1.1 decision history
(`docs/decisions/`), including two "Superseded" ADRs for decisions that were later reversed
(discriminant clamp -> strict check; global-scale margin -> recentering + local margin -- the
latter never shipped, caught in planning, documented anyway since the reasoning is real). ADR-0012
(below) is this milestone's own decision, written the normal way (before/alongside the code and
run it governs, not retroactively).

**S2 fit-validity gate (ADR-0010)**: M1.1's estimate silently mixed two different failure modes
under "fallback" -- the certification algorithm failing on a *geometrically valid* linearization,
and the certification algorithm being handed a *numerically garbage* spline fit (at least 26/46 of
M1.1's original failures had `fit_adaptive` control points diverging by orders of magnitude from
the track itself). `fit_validity()` (`benchmarks/step7_certify.py`) now runs before certification
is attempted for every S2 track, checking two fixed thresholds (set before this milestone's run,
per ADR-0010, and not adjusted afterward):

1. dense-grid deviation of the spline from the track (`traj.spline.dense_max_error`, reusing the
   same "error between samples, not just at them" check `spline.py`'s own fitter already applies,
   which `spline_lsq.py`'s fitters do not) `<= 10 x S2_FIT_TOL` (50 m);
2. every control point within `100x` the track's own bbox diagonal of its first point.

A track failing either check is **invalid fit**: excluded from S2's pass/fallback counts entirely,
neither a pass nor a certification failure, because certification was never meaningfully attempted
on an honest representation.

**Fitter choice (ADR-0011)**: M1's original choice of `fit_adaptive` over `fit_uniform` had no
recorded justification. A 60-track comparison (seed 7) found no meaningful difference in invalid
rate (91.7% both) but a better tail for `fit_adaptive` (max deviation 1.3e6 vs. 2.1e7) -- kept.

**Full-corpus S2 re-run under the fixed ADR-0010 thresholds** (585/585 tracks, `fit_adaptive`):

| Result | Count |
|---|---|
| pass | 36/585 (6.2%) |
| fallback (valid fit, certification exhausted budget) | 0/585 |
| fit error (fitter raised an exception) | 0/585 |
| reference unavailable | 0/585 |
| invalid fit | 549/585 (93.8%), all `dense_deviation` |

Elapsed: 70.3s.

**Threshold-sensitivity check (ADR-0012)**: 93.8% invalid was high enough to ask whether ADR-0010's
thresholds were simply too strict, rather than editing them after the fact per the plan's
contingency (new ADR + new run + both results reported). Computed the invalid-fraction sensitivity
to the deviation multiplier directly from the cached per-track deviations (no re-fitting): 10x =
0.938, 20x = 0.879, 50x = 0.776, 100x = 0.655, 200x = 0.496, 500x = 0.262, 1000x = 0.171, 2000x =
0.109 -- a heavy-tailed distribution (median deviation ~989 m, p99 ~3.3e7 m, max ~3.9e17 m) with no
natural "reasonable" cutoff a few multiples above `10x`. Ran the full corpus again at an explicit
`1000x`/`1000x` comparison threshold to get real numbers rather than an extrapolation:

| Result | Count |
|---|---|
| pass | 480/585 (82.1%) |
| fallback | 0/585 |
| fit error | 0/585 |
| reference unavailable | 0/585 |
| invalid fit | 105/585 (17.9%) -- 100 `dense_deviation`, 5 `control_point_bound` |

Elapsed: 2235.8s (~37.3 minutes).

**Decision (ADR-0012): keep ADR-0010's thresholds unchanged.** The high invalid rate is a
structural property of `spline_lsq.py`'s fitters on real GeoLife data (no dense-error control
between samples), not an artifact of an arbitrarily strict multiplier -- it persists at a still
substantial 17.9% even two orders of magnitude more generous. Picking a "nicer" threshold post hoc
would misrepresent a real methodology gap as a calibration problem. The gap is recorded as an open
item in `TODO.md` for a future milestone, out of scope for M1.2 (which is about the certification
algorithm, not the fitting methodology).

**`n_fallback = 0` at both thresholds** is the headline result for the certification algorithm
itself: every track with a valid fit gets certified, at either threshold. This closes M1.1's own
open question -- the 46 originally-uncertified tracks were overwhelmingly a fit-quality problem,
not an algorithm-level one; ADR-0008's root-splitting fix has zero remaining fallbacks once given
a valid representation.

**eps_A/LB tail, 5 worst polyline tracks -- verified, not asserted (M1.3,
`docs/reviews/step7_M1_2.md` finding 4, `benchmarks/step7_m13_tail.py`).** M1.2's text asserted
"the lower bound is loose" without checking. For each of the 5 tracks, `k*` is the piece that
literally *determines* `eps_A` (`argmax_k cert_k`, matching `certify_polyline`'s own
`eps_A = max_k cert_k` -- not a piece-size or piece-local-ratio proxy, both of which were tried
first and found to be dominated by noise from degenerate 2-point pieces where both the piece's
own certified cost and its own Hausdorff bound are near zero).

| Track idx | global ratio | `eps_A` | mpmath ref | `eps_A == ref`? | `k*` range | `cert_k*` | `LB_hausdorff` (k*, local) | `LB_back` (k*) | frac. decreasing | confirmed (`<= *1.05`) |
|---|---|---|---|---|---|---|---|---|---|---|
| 472 | 6.585 | 7.3739 | 7.3739 | yes | [172:176] | 7.3739 | 7.3739 | 0.0000 | 0.000 | yes |
| 475 | 2.645 | 8.9990 | 8.9990 | yes | [129:179] | 8.9990 | 8.9990 | 2.4090 | 0.360 | yes |
| 196 | 2.428 | 6.1966 | 6.1966 | yes | [3:35] | 6.1966 | 6.1966 | 1.2238 | 0.188 | yes |
| 476 | 2.198 | 5.1885 | 5.1885 | yes | [30:40] | 5.1885 | 5.1885 | 0.3216 | 0.200 | yes |
| 482 | 2.181 | 6.4740 | 6.4740 | yes | [87:131] | 6.4740 | 6.4740 | 2.6360 | 0.295 | yes |

Two things are now directly verified, not assumed:

1. **`eps_A` exactly equals the mpmath ground-truth reference for all 5 tracks** -- the S1
   certificate is not loose relative to the true continuous Fréchet distance. This directly
   confirms the review's "not a loose certificate" premise.
2. **`cert_k*` exactly equals that piece's own local `LB_hausdorff` in all 5 cases**, regardless of
   how much the piece backtracks (`frac_decreasing` up to 36%) -- a structural fact, not a
   coincidence: the Fréchet distance from a curve to a single straight *segment* (as opposed to a
   multi-vertex polyline) always equals the max point-to-segment distance, because the segment's
   own free parametrization can pause/backtrack at zero cost to absorb any backtracking the curve
   does. So `LB_back` (real, and sometimes substantial -- 2.41 m for idx=475) never actually
   inflates a piece's own certified cost beyond its own local Hausdorff bound for *this*
   certificate structure (piece vs. its 2-point segment). The "confirmed" check
   (`cert_k* <= max(LB_hausdorff_local, LB_back) * 1.05`) holds for all 5, trivially so, since
   `cert_k*` and `LB_hausdorff_local` are already equal.

**So the mechanism explaining the whole-track `eps_A`/`LB` gap is refined, not simply confirmed
as "within-piece backtracking":** it is a **self-proximity** effect at the *global* level.
`LB` (`hausdorff_lower_bound`, spec 2.5) is computed against the *entire* simplified polyline, so
a point in piece `k*` can be matched, for the purpose of that lower bound, to *any* other,
non-corresponding segment elsewhere in the polyline -- and for idx=472 specifically, it is: that
piece's own local Hausdorff/Fréchet cost is 7.3739, `frac_decreasing = 0` (no backtracking
whatsoever within the piece -- its own chord-projection is strictly increasing), yet the
whole-track `LB` is only 1.1199, a 6.6x gap. The only way a *non-backtracking* piece can produce
that gap is if some *other*, temporally non-corresponding part of the 163-vertex simplified
polyline happens to pass within ~1.12 m of these points in space -- i.e. the track loops close to
another part of itself. The order-preserving Fréchet coupling cannot use that spatial shortcut (it
must honestly match piece `k*` to its own corresponding segment); Hausdorff's any-to-any matching
can. For idx=475/196/476/482, both effects likely coexist (real within-piece backtracking,
`frac_decreasing` 18.8-36%, plus some self-proximity) but only the global self-proximity effect
can be responsible for the *size* of the gap, since the piece-local backtracking is proven above
to cost nothing extra against a straight 2-segment. **Net: `eps_A` is not loose (verified against
mpmath); the informational `LB` is a genuinely weaker bound than a naive "backtracking within one
piece" story would suggest, and the correct mechanism is the track's self-proximity to other,
non-corresponding parts of its own path -- itself a direct consequence of Fréchet's order
constraint vs. Hausdorff's order-free matching, just not localized to a single piece.**

### M1.3 -- fitter comparison (`docs/reviews/step7_M1_2.md`, ADR-0013/ADR-0014)

**Bug fix first (ADR-0014):** `fit_validity()`'s dense-grid deviation check evaluated
`spline_lsq.py` fits at the wrong parametrization domain (real-time-domain knots evaluated as if
normalized to `u ∈ [0,1]`) -- M1.2's 93.8%/17.9% invalid-fit numbers were therefore largely
measurement artifacts, not a reliable finding. Fixed by adding a `"raw"` mode to
`traj.spline._param_u` and a domain-consistency guard (`_check_tck_domain`) that raises
`ValueError` on any tck/mode mismatch, verified on a 60-track sample (buggy 54/60 (90%) invalid
vs. corrected 11/60 (18.3%)) and with a regression test that fails on the pre-fix code. ADR-0010's
thresholds are unaffected (the bug was in the measurement, not the check's design); ADR-0012's
sensitivity analysis is superseded (it was computed from the same broken measurement).

**Fitter comparison (ADR-0013):** the review's finding 1 -- `spline.py`'s `fit()` already exists
and dense-error-controls by construction, `spline_lsq.py`'s fitters don't -- is checked by running
both fitters on the corrected full 585-track corpus, same `tol=10.0` (step1's own value), same
ADR-0010 multiplier, decided by a rule fixed *before* the run:

| Fitter | Valid | Pass | Invalid (reason) | Elapsed | `eps_A` p50/p90/p99/max | `eps_A/tol` median/p90/p99/max |
|---|---|---|---|---|---|---|
| `fit_adaptive` (corrected) | 527/585 (90.1%) | 527/585 | 58 (`dense_deviation`) | 2602.4s | 7.54 / 15.76 / 50.62 / 87.52 | 0.754 / 1.576 / 5.062 / 8.752 |
| `spline.fit()` | 585/585 (100.0%) | 585/585 | 0 | 3367.2s | 8.11 / 10.58 / 15.87 / 23.83 | 0.811 / 1.058 / 1.587 / 2.383 |

The valid-fraction gap (9.9 points) exceeds ADR-0013's 5-point tie-break threshold, so the rule
decides at step 1: **`spline.fit()` is selected as S2's primary fitter.** (Had it reached the
tie-break, the two indicators would have disagreed -- `fit_adaptive` has a lower median
`eps_A/tol` but also a lower absolute median `eps_A`, an inherently ambiguous comparison the rule
resolves by default toward `spline.fit()`; moot here.) This is a deliberate deviation from spec
section 2.1's literal "S2 uses `spline_lsq`", recorded in `docs/ROADMAP.md` section 8.
`fit_adaptive` remains available as the comparison variant.

`spline.fit()`'s much tighter `eps_A/tol` tail (max 2.38 vs. 8.75) is exactly what "dense-error
control by construction" predicts: `fit_adaptive`'s handful of surviving-but-marginal fits (valid
under the `10x` deviation threshold, i.e. `<=100` m, but closer to that ceiling than to `tol`
itself) drag its tail up, while `spline.fit()`'s dense-check-during-fitting keeps every valid fit
close to `tol` by construction.

### M1 Conclusions

**Results vs. criteria**: S1's gate criterion passes at 100% (585/585 against the mpmath
reference, worst margin -5.000e-07). S2's gate criterion, taken *literally* as first measured
(`certify_spline_linearization` on `fit_adaptive` at `tol=5.0`, no invalid-fit distinction yet),
passed at 539/585 with 46 acknowledged, spec-flagged (section 10) fallbacks -- but that specific
number is a snapshot of the *first* run, superseded for anything about *current* S2 behavior by
M1.2's invalid-fit distinction and M1.3's bug fix and fitter switch (below). There is no single
"S2 coverage" percentage that stays fixed across M1.1/M1.2/M1.3 -- it depends on which fitter,
which validity threshold, and (M1.3) a since-fixed measurement bug, so each is stated explicitly
rather than collapsed into one headline number:

| Stage | Fitter | Valid/total | Notes |
|---|---|---|---|
| M1 (original) | `fit_adaptive`, `tol=5.0` | -- (539/585 pass, 46 fallback; no invalid-fit split yet) | superseded by M1.2's split |
| M1.2, ADR-0010 threshold | `fit_adaptive`, `tol=5.0` | 36/585 (6.2%) valid -- **measurement later found buggy (ADR-0014)** | not a reliable number as reported |
| M1.2, ADR-0012 alt. threshold | `fit_adaptive`, `tol=5.0` | 480/585 (82.1%) valid -- **measurement later found buggy (ADR-0014)** | not a reliable number as reported |
| M1.3, corrected | `fit_adaptive`, `tol=10.0` | 527/585 (90.1%) valid, 527/585 pass | bug-fixed (ADR-0014), still `spline_lsq`'s own architectural gap |
| M1.3, corrected | `spline.fit()`, `tol=10.0` | 585/585 (100.0%) valid, 585/585 pass | **current primary fitter (ADR-0013)** |

**What's confirmed about certification *correctness*** (independent of which fitter or threshold):
`n_fallback = 0` in every M1.2/M1.3 full-corpus run, at every threshold, for both fitters -- the
certification algorithm itself (`certify_spline_linearization`, `_certify_segment`'s
root-splitting + small-ball rule, ADR-0008) has no remaining failures once given *any* numerically
valid spline representation. This closes M1.1's own open question: the original 46 uncertified
tracks were overwhelmingly a fit-quality problem, never an algorithm-level one. Also confirmed:
`eps_A` exactly matches the mpmath ground-truth reference for the 5 worst S1 eps_A/LB tracks (the
S1 certificate itself is not loose, M1.3's tail verification above).

**What's confirmed about *coverage and density*** (fitter- and threshold-dependent, unlike
correctness): with the bug fixed and `spline.fit()` selected as the primary fitter (ADR-0013),
S2's *current* coverage is **585/585 (100%) valid and passing** -- a materially better position
than M1.2's headline "93.8% invalid" ever suggested, because that headline was itself
measurement-broken (ADR-0014). `fit_adaptive`'s own (bug-fixed) coverage, 90.1%, confirms
`spline_lsq.py`'s architectural gap (no dense-error control between samples) is real but smaller
than M1.2 reported. The S1 eps_A/LB density tail (5 worst tracks, informational, not a gate) is
explained by global self-proximity of the track's own path, not by a loose certificate or by
naive within-piece backtracking (M1.3 tail verification above).

**Decisions**: ADR-0010 (invalid-fit category + fixed thresholds -- design unaffected by the later
bug), ADR-0011 (fit_adaptive vs. fit_uniform, superseded in relevance by ADR-0013's broader
comparison), ADR-0012 (sensitivity analysis, **superseded by ADR-0014** -- built on the buggy
measurement), ADR-0013 (fitter comparison rule and outcome: `spline.fit()` selected), ADR-0014
(the parametrization-domain bug and its fix). Fourteen ADRs total now cover the M0-M1.3 decision
history (`docs/decisions/README.md`).

**Open issues**: `spline_lsq.py`'s fitters themselves are unchanged -- no dense-error-aware
fitting mode was added to them; the resolution was a fitter *choice* (`spline.fit()` instead), not
a fix to `spline_lsq.py`. If a future milestone specifically needs `spline_lsq`'s fitters (not
`spline.fit()`) to control dense error, that would be new, separate work (`TODO.md`).

`docs/phases/step7_summary.md` (per `CLAUDE.md`'s Documentation rules, one page per gate) is not
yet written -- it will be created at gate G1, after M3, once the full step7 phase's scope is
complete, not at this intermediate milestone.

## M2 -- spline certificate via monotone projection matching (spec section 2.3)

Corpus: the same 585 cleaned track segments as M1, `spline.fit()` (`tol=10.0`, ADR-0013's
selected S2 fitter). `certify_spline_projection` (spec 2.3) is tried first; `certify_spline`
falls back to the existing `certify_spline_linearization` (2.4, M1, unchanged) per track when 2.3
can't fully certify. Implementation: `src/traj/certify.py` (ADR-0015: correspondence-point
search; ADR-0016: certificate structure); `src/traj/bezier.py`'s `bezier_segments_in_range`; new
property tests in `tests/traj/test_certify.py`; full-corpus run in
`benchmarks/step7_m2_projection.py`.

**S1's methodology is three genuine checks, not the near-exact-linearization reference M1 used
for S2** (a mid-implementation correction): that reference is itself only an upper bound
(`d_F(A,Lin)+lam`), so a genuinely tighter 2.3 certificate can legitimately be *smaller* than it
-- comparing against it as a correctness floor would be invalid.

| Criterion | Threshold | Actual | Passed |
|---|---|---|---|
| S1a: `eps_A(combined) >= LB` (`hausdorff_lower_bound`, genuine lower bound) | 100% of tracks | 585/585 (0 violations) | yes |
| S1b: `eps_A(combined) >= distance_mp(A, Lin(A',1.0)) - 1.0 - 1e-3` (reverse triangle inequality, genuine lower bound) | 100% of a 30-track sample | 30/30 | yes |
| S1c: direct dense-sampling verification of the spec's own clamp-cost formula + monotonicity (spec section 9), on every 2.3-certified track (not a sample) | 100% of 96 2.3-certified tracks | 96/96 (0 violations) | yes |
| S2: combined-pipeline `eps_A` vs. near-exact-linearization reference (`lam=1e-4`) `- 1e-4` | 100% of tracks with an available reference | 585/585 (0 failures, reference available for all 585) | yes |
| S3 (density): median `eps_A/LB` for splines `<= 2` | median `<= 2` | median 1.019 (p90 1.738, p99 2.554, max 7.497) | yes |
| S3 (fallback rate): fraction of splines (tracks) that fell back to 2.4 | `<= 10%` | **489/585 = 83.6%** | **no** |

**S1b's slack (30-track mpmath sample) is `1.0` m, not the originally-envisioned `1e-6`** --
found infeasible empirically before running against real data (fixed before the run, not tuned
after seeing results): `tests/traj/_frechet_cont_mpmath.py`'s own docstring already says "too
slow for anything but small fixtures (n, m <= ~6)"; confirmed directly (`n=264, m=100`, 26,400
DP cells, did not complete in 120s -- `distance_mp`'s bisection needs ~25-30 `decide_mp` calls to
reach a tight tolerance, each `O(n*m)` in arbitrary-precision arithmetic). Fixed parameters
(`benchmarks/step7_m2_projection.py`): `lam_ref=1.0`, mpmath `tol=1e-3`, `dps=25`, and the
30-track sample is drawn only from tracks where `n * m(lam_ref=1.0) <= 3000` (a *tractability*
filter, checked before knowing what `distance_mp` would return, not a favorable-results filter).
A slack of `1.0` m is still meaningfully tight relative to typical `eps_A` values (~5-15 m) --
30/30 tracks pass regardless.

**S1a's LB computation needed its own fix mid-run** (ADR-0017): a first attempt used a
fixed-count uniform spline sample (~500-700 points) for `LB`, which produced 12 *false* S1a
violations (certificates that looked smaller than `LB` by up to 3.3 m). Investigated directly:
the uniform sample was simply too coarse to resolve a locally curvy stretch on those specific
splines (confirmed: `hausdorff_lower_bound(Lin(A'), uniform_708_points) = 3.59`, when it should
be `<= lam=0.1` if the sample faithfully represented the curve; a 2000-point sample already
dropped this to `0.59`). Fixed by reusing the near-exact-linearization reference's own
adaptively-placed vertices (already computed for S2) as the dense sample instead -- all 12
violations resolved (`LB` dropped below `eps_A` in every case once measured correctly), and a
vectorized point-to-polyline distance computation (`_fast_hausdorff_lower_bound`) was needed to
make this tractable at the resulting scale (`m` up to ~10,000). This is a benchmark-methodology
fix; `certify_spline_projection`/`certify_spline_linearization` themselves were never wrong.

### 2.3 vs. 2.4 comparison (96 tracks where 2.3 succeeds)

| Quantity | median | p90 | p99 | max | min |
|---|---|---|---|---|---|
| `eps_A(2.3) / eps_A(2.4)` | **1.683** | 1.963 | 3.744 | 6.863 | 0.991 |
| `eps_A(2.3) / near-exact reference` (informational density metric) | 1.703 | 1.995 | -- | 6.964 | 1.000 |
| time, section 2.3 (s) | 0.103 | 0.507 | -- | 13.666 | -- |
| time, section 2.4 (s) | 0.070 | 0.349 | -- | 12.590 | -- |

Per-track certification time, full combined pipeline (fit + 2.3 + 2.4-for-comparison + near-exact
reference): median 2.84 s, p90 12.70 s, max 213.9 s (the near-exact reference at `lam=1e-4`
dominates this -- see M3 carry-over note below).

**Correction to the plan's own expectation**: `eps_A(2.3)/reference`'s minimum is exactly
`1.000`, never below -- both `eps_A(2.3)` and the near-exact reference upper-bound the *same*
true `d_F(A, spline)`, and the reference is specifically built to be near-exact (a very fine,
certified linearization), so `eps_A(2.3) >= reference` is the expected mathematical relationship,
not `< 1`. The milestone's plan described `< 1` as the expected/good outcome for this ratio; the
data corrects that -- the ratio instead measures how much slack 2.3's specific correspondence
construction carries above the best achievable bound (typically 70%, sometimes up to 6x).

### 30-track mpmath sample (S1b), `eps_A` vs. both lower bounds

| idx | n | m (`lam_ref=1`) | `eps_A` | `LB` (near-exact ref) | `eps_A/LB` | `distance_mp` | mpmath LB | `eps_A`/mpmath LB |
|---|---|---|---|---|---|---|---|---|
| 265 | 56 | 35 | 6.309 | 1.253 | **5.035** | 6.209 | 5.208 | 1.211 |
| 134 | 67 | 6 | 18.427 | 9.518 | 1.936 | 9.518 | 8.517 | 2.163 |
| 375 | 83 | 9 | 17.961 | 9.238 | 1.944 | 9.238 | 8.237 | 2.180 |
| 373 | 116 | 13 | 10.083 | 5.057 | 1.994 | 5.057 | 4.056 | 2.486 |
| (26 more, all `eps_A >= mpmath_lower_bound`) | | | | | | | | |

All 30/30 satisfy `eps_A >= distance_mp - 1.0 - 1e-3` (S1b). Track 265's `eps_A/LB = 5.0` is the
same *self-proximity* mechanism M1.3's tail analysis already identified for polylines (the
whole-track vertex-Hausdorff bound benefits from matching to a spatially-close but
non-corresponding part of the track, a shortcut the order-preserving certificate cannot take) --
not a new phenomenon, and not evidence against `eps_A`'s correctness (`eps_A/mpmath_lower_bound`
for the same track is a much more modest 1.211, since the mpmath lower bound uses a full,
non-simplified linearization rather than sparse vertices).

### Fallback structure

Fallback is a **per-track** decision (spec 2.3 item 5: if *any* piece can't certify, the whole
track uses 2.4) but the *piece*-level failure rate is low: 2,639/141,363 pieces (**1.87%**). With
an average of ~242 pieces per track, even a small per-piece failure probability compounds
multiplicatively across a track's pieces -- `(1 - 0.0187)^242 ≈ 0.011`, i.e. only ~1% of tracks
would be expected to have *zero* failing pieces if failures were independent and identically
likely per piece, roughly consistent with the observed 16.4% success rate once piece failures'
real clustering (some tracks are uniformly easy, others have a genuinely hard, curvy stretch) is
accounted for. This is a structural consequence of the per-track fallback granularity combined
with a nonzero (if individually small) piece failure rate -- not evidence of an implementation
defect, and not fixable by raising `max_levels` alone (ADR-0016: a piece with no interior root at
all is provably, permanently non-monotone relative to its fixed direction, regardless of budget).

### M2 Conclusions

**Results vs. criteria**: S1 (all three genuine checks) and S2 pass at 100%. S3's density
threshold (median `eps_A/LB <= 2`) passes (1.019). **S3's fallback-rate threshold (`<=10%`)
fails**: 83.6% of tracks fall back to section 2.4, nearly 8x the threshold.

**Key negative finding of this milestone**: `eps_A(2.3)`'s median is **1.68x larger** than
`eps_A(2.4)` on the very tracks where 2.3 succeeds -- section 2.3, the spec's primary, more
elaborate spline certificate, is *not* typically tighter than section 2.4's simpler
certified-linearization fallback in practice, and section 2.3 fails to certify at all on the
large majority (83.6%) of real tracks. Per `CLAUDE.md` ("a negative benchmark result is a normal
result -- don't force it"), this is reported as a genuine finding, not adjusted away: the spec's
own architecture (2.4 as "always applicable, but coarser," section 2.4's own text) turns out, on
real GeoLife data, to be both more *available* and typically *tighter* than the primary path it
was meant to back up.

**What we now know**: the certificate is *correct* wherever it applies -- confirmed three
independent ways (genuine lower bounds via `LB` and mpmath, and direct dense-sampling
verification of the spec's own formula on every successful track). The certificate is
*available* on only 16.4% of real tracks, and even there is usually looser than the simpler
fallback. The mechanism (ADR-0016) is structural: 2.3 certifies against the ORIGINAL track's
fixed per-vertex segments, so a single genuinely non-monotone piece (real GPS noise or curvature
relative to a specific short segment) forces the whole track to fall back; with ~242 pieces per
track on average, this is common even at a low (1.87%) per-piece rate.

**Decisions**: ADR-0015 (correspondence-point search: coarse scan + safeguarded Newton, fixing
two real pathologies found while smoke-testing), ADR-0016 (certificate structure: fixed
direction, measured tube, early exit on unsplittable pieces), ADR-0017 (LB needs an adaptively
dense sample, not a fixed uniform count). Seventeen ADRs total now cover the M0-M2 decision
history (`docs/decisions/README.md`).

**Open issues / M3 carry-overs**: the near-exact-linearization reference (`lam=1e-4`) dominates
the combined-pipeline per-track time reported above (median 2.84s, max 213.9s) -- M3's query load
will be much heavier and does not need this reference at all (it exists only for this milestone's
own S2/density accounting), so M3's own timing budget should be planned around 2.3's and 2.4's
own costs directly (medians under 0.1s each), not this milestone's reporting overhead. Given S3's
fallback-rate failure, M2 does not by itself justify preferring section 2.3 over relying on 2.4
as the default in a real system -- a design question for M3/M4 to weigh against 2.3's per-track
tightness where it *does* succeed. `docs/phases/step7_summary.md` remains deferred to gate G1
after M3 (unchanged from M1/M1.2/M1.3's own notes).

## M3 -- interval range queries (spec section 2.5)

Corpus: the same 585 cleaned tracks plus 415 "near-duplicate" tracks (a controlled translation
perturbation with a measured, not assumed, `d_F` to their source -- see below), 1000 total, both
representations (DP+SED polyline, spline.fit()+section 2.4 only per ADR-0018, `lam=0.1` fixed --
M1/M2's `lam=1e-4` near-exact reference is never used at query time). 1000 queries (drawn from
this corpus) x 3 ranges (`r in {50, 200, 1000}` m) x 2 representations, against every other
corpus entry (999 candidates each). `src/traj/intervals.py` (ADR-0019) implements the
accept/reject/refine rule via `decide()` only. Implementation: `benchmarks/step7_query.py`.

### Near-duplicate generation and hard-regime coverage (mandatory correction 3)

A near-duplicate is a uniform translation of a source track by a vector of a chosen target
magnitude, plus small jitter -- the identity (same-index) correspondence gives every point the
same cost, so `d_F <= target_distance` by construction, and for a generic (non-self-similar)
track shape the true `d_F` lands close to the target; **every resulting `d_F` is independently
measured** via `traj.frechet_cont.distance` on the raw polylines, never assumed from the target.
(An earlier version used a *local* bump on a sub-range of points instead -- found empirically to
be a much less reliable way to hit a target distance, since a local perturbation's contribution
to the Fréchet distance depends on how it interacts with the optimal matching, not just its own
size; switched to the global-translation approach before the real run, not after seeing bad
results from it.)

Generation ran in rounds, checking after each whether every `r` has `>=10%` of (duplicate,
source) pairs with true `d_F` inside `r +/- 2*sum_eps` (the hardest regime for the interval
rule), for **both** representations independently -- retargeting under-covered `r`'s with a
tighter jitter in later rounds:

| Round | Duplicates | Coverage (polyline / spline), `r=50` | `r=200` | `r=1000` |
|---|---|---|---|---|
| 0 (broad) | 60 | 8.3% / 15.0% | 3.3% / 3.3% | 1.7% / 1.7% |
| 1 (targeted) | 80 | 12.5% / 17.5% | 13.75% / 13.75% | 8.75% / 8.75% |
| 2 (targeted) | 100 | **17.0% / 21.0%** | **19.0% / 19.0%** | **12.0% / 12.0%** |

All six `(r, representation)` combinations clear the `>=10%` bar by round 2 -- no further
targeted rounds needed. An additional 315 general-volume near-duplicates (broad amplitude, not
aimed at any specific `r`) were then added purely to reach 1000 distinct corpus entries (needed
for a full 1000-query set, since 585 base + 100 coverage-targeted duplicates = 685 < 1000) --
these do not count toward or dilute the coverage numbers above, which are measured only against
the 100 coverage-targeted duplicates.

### Criteria (spec section 8)

| Criterion | Threshold | Actual | Passed |
|---|---|---|---|
| S4: 0 misses, 0 false positives (certified method vs. full brute force) | 1000 queries x 3 r's x 2 representations | 0/0 in all 6 million (query, candidate, r, representation) combinations | yes |
| S5: fraction of candidates resolved without reading originals, **among post-cheap-filter survivors** (mandatory correction 1) | `>= 80%` for at least one representation, per `r` | `r=50`: 61.7% (polyline) / 57.9% (spline); `r=200`: 84.8% / 83.7%; `r=1000`: 95.6% / 95.0% | **`r=50`: no; `r=200`, `r=1000`: yes** |
| S5, informational: fraction resolved without reading originals, **among ALL candidates** | -- (reference only) | 99.92-99.97% for every `r`/representation (the cheap pre-filter alone resolves the large majority, see below) | -- |
| S6a: certificate size | `<= 16 bytes/track` | 8 bytes/track (one float64, either representation -- ADR-0018's `eps_A` already bundles `lam`) | yes |
| S6b: query time vs. approximate search without certificates | `<= 2x` | polyline 2.66x; spline 4.38x (median latency ratio) | **no** |
| Stop condition: S5 `< 50%` for **all** representations | direction closes if triggered | worst case (r=50) is 57.9%/61.7%, still `> 50%` | **not triggered** |

### Rejections: cheap filter vs. interval rule (mandatory correction 2)

Both sides of `decide_range`'s interval rule are checked explicitly (ADR-0019); rejections are
split into `reject_cheap` (endpoint/bbox lower bound alone, no `decide()` call) and
`reject_interval` (the interval rule's own upper-threshold check). Across all three `r`'s and
both representations, **99.94-99.99% of all rejections come from the free cheap filter alone** --
the interval rule itself (an actual `decide()` call) only ever has to reject the small residue the
cheap filter couldn't already dismiss (e.g. `r=1000`, polyline: 982,160 cheap rejections vs. 612
interval-rule rejections). This is expected given the corpus spans geographically and temporally
disjoint GeoLife users/days -- most candidate pairs are simply nowhere near each other.

### Latency and interval width (informational)

Dedicated timing sample (4000 pairs, not reused from the correctness grid -- each method run
through its own real code path):

| Method | Median | p95 |
|---|---|---|
| brute (raw, no filter) | 11.5 us | 22.1 us |
| filter_uncompressed (cheap filter + raw decide) | 43.0 us | 123.7 us |
| approximate, polyline (no certificate) | 8.2 us | 11.5 us |
| **certified, polyline** | **21.9 us** | **33.1 us** |
| approximate, spline (no certificate) | 12.6 us | 32.5 us |
| **certified, spline** | **55.0 us** | **188.6 us** |

The certified method's overhead over the approximate one (S6b, above) comes from `decide_range`
needing up to two `decide()` calls (accept-check, then reject-check) plus the cheap-bound
computation, vs. the approximate method's single, unconditional call -- a structural consequence
of checking both sides of the interval explicitly, not a specific inefficiency to fix.

Interval width (`2*sum_eps`, the full width of `[D-sum_eps, D+sum_eps]`, over 200,000 sampled
pairs per representation): polyline median 30.8 m (p90 36.0 m, p99 38.2 m, max 39.7 m); spline
median 31.9 m (p90 40.3 m, p99 51.9 m, max 86.8 m) -- consistent with M1/M2's own established
`eps_A` scales for these representations (tighter for polylines, a somewhat heavier tail for
splines' section 2.4 certificates).

Refine rate (fraction of candidates needing to read the originals, i.e. `1 - S5(post-filter)`):
`r=50` 38.3%/42.1%, `r=200` 15.2%/16.3%, `r=1000` 4.4%/5.0% (polyline/spline) -- mirrors S5's own
`r`-dependence directly (smaller `r` means `sum_eps` is a larger fraction of `r`, widening the
*relative* size of the ambiguous interval band).

### M3 Conclusions

**Results vs. criteria**: S4 passes perfectly (0 misses, 0 false positives, the whole point of
the certified store). S6a (certificate size) passes trivially. **S5 fails at `r=50`** for both
representations (58-62% vs. the `>=80%` threshold) while passing comfortably at `r=200` and
`r=1000`. **S6b (query time `<=2x` approximate) fails** for both representations (2.7x-4.4x). The
stop condition (S5 `<50%` for *all* representations) is **not triggered** -- even the worst case
clears 50%, so per spec section 8 the direction does not close on this basis.

**Key negative findings, reported plainly** (`CLAUDE.md`: "a negative benchmark result is a
normal result"): (1) at short range (`r=50`, comparable in scale to the certificates'
own ~15 m/side interval half-width), the interval rule resolves a *minority* of post-filter
candidates without reading originals -- the certified store's main promised benefit (avoiding
the originals) degrades exactly where queries are tightest, a direct, structural consequence of
`sum_eps` not shrinking with `r`. (2) The certified method is consistently slower than the
uncertified approximate one by more than the spec's `2x` budget, because checking both sides of
the interval costs up to two `decide()` calls where the approximate method needs only one --
inherent to explicitly guaranteeing correctness (S4's 0/0 result), not a fixable inefficiency in
this implementation.

**What we now know**: the certified interval query pipeline is *exactly correct* (S4, confirmed
at full 6-million-pair scale) and its overhead is *concentrated and explicable* (cheap filters
alone resolve >99.9% of all rejections; the remaining cost is the interval rule's own two-sided
check). Its *practical* usefulness is range-dependent: it clearly earns its keep at `r=200` and
`r=1000` (S5 passes, most candidates resolved cheaply) but is less compelling at `r=50`, where
reading the originals is needed for close to half the post-filter candidates anyway.

**Decisions**: ADR-0019 (interval rule design, decide-only, the `eps_A = epsilon+lambda` mapping).
Nineteen ADRs total now cover the M0-M3 decision history (`docs/decisions/README.md`).

**Open issues**: whether the certified store is worth its S6b overhead at small `r` specifically
is a product/architecture question for M4's summary, not resolved here -- M3's job was to measure
it honestly, which it does. `docs/phases/step7_summary.md` is created at gate G1, which M3
completes the acceptance-criteria portion of; M4's summary report is the next and final step7
milestone.
