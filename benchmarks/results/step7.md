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
- **A numerical robustness fix found during testing:** a point lying exactly on the *line* through
  a segment gives a mathematically repeated root (`Delta == 0` exactly), but computing
  `B*B - 4*A*C` in float64 can push the residual slightly negative from cancellation alone,
  wrongly reporting "empty" for a point that's genuinely touching the segment. This surfaced as a
  real (not merely hypothetical) failure in the reparametrization-invariance property test on an
  axis-aligned configuration. Fixed by clamping discriminant residuals within a
  `64 * eps_machine`-scaled margin to zero rather than declaring empty -- the safe direction for a
  certified upper bound, and the same convention the spec's own floating-point rigor section
  (`docs/specs/00_overview.md` section 3.4, `delta = 64 * eps_machine * L`) prescribes for
  certificates generally.
- `tests/traj/_frechet_cont_mpmath.py`: an independent high-precision oracle (pure Python +
  `mpmath`, no shared code with the numba implementation) used only in tests, to cross-check the
  DP's correctness rather than merely re-running the same logic at higher precision.

### Test results (spec section 9 property tests + defensive extras)

| Criterion | Threshold | Actual | Passed |
|---|---|---|---|
| decide/distance consistency | `decide(d+1e-4)` true, `decide(d-1e-4)` false, 100 random cases | 100/100 | yes |
| Symmetry | `d_F(P,Q) == d_F(Q,P)`, 100 random cases | 100/100 | yes |
| Triangle inequality | `d_F(P,R) <= d_F(P,Q)+d_F(Q,R)+1e-9`, 100 random triples | 100/100 | yes |
| Reparametrization invariance | inserting exact-collinear vertices leaves `d_F` at 0 (< 1e-6), 100 random cases + 500-case stress sweep | 600/600 | yes |
| Cross-check vs. mpmath oracle (independent implementation) | `\|distance - distance_mp\| < 1e-5`, small polylines (n,m<=6) | worst deviation 9.97e-10 over 60+20 cases | yes |
| eps-monotonicity (defensive, not in spec) | `decide` results non-decreasing in `eps`, 100 random cases | 100/100 | yes |
| n=1 special case (segment vs. polyline) | O(m) path shares code with general algorithm, no divergence | exact match (distance 1.0 on hand-computed example) | yes |
| Duplicate consecutive vertex | zero-length segment doesn't crash, distance matches de-duplicated polyline | matches (diff < 1e-6) | yes |

`venv/bin/pytest tests/traj/test_frechet_cont.py -v`: 11/11 passed, ~2.5s (numba JIT warm-up
included). Full `tests/traj/` suite: 29/29 passed, no regressions in the existing `frechet.py`,
`spline.py`, `spline_lsq.py` tests.

### Stack decision recorded (spec section 6)

`python-flint` has no wheel for Python 3.14 (`import flint` fails in `venv/`, confirmed). Per the
spec's own fallback clause, interval-arithmetic spot-checks (needed starting M1's `certify.py`,
per `docs/specs/00_overview.md` section 3.4) will use `mpmath.iv` instead. Not needed for M0
itself -- `hypothesis` (6.168.0) and `mpmath` (1.4.1) were added to `requirements-research.txt`
and are sufficient for this milestone's property tests and oracle.
