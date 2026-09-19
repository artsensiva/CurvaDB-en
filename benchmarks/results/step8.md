## M0

Unified section-2.5 encoding (`src/traj/encode.py`) re-verifying step3's own DP+SED / LSQ-uniform byte counts, dt=1 s only, same 15-track/seed=42 synthetic generator and the same reachability criterion as `step3_decisive.py` (honest error against the true noise-free curve on a dense grid, `<= target_tol` -- UNCHANGED from step3, see docs/decisions/ADR-0021). Step7's certificates (`certify_polyline`/`certify_spline`, section 2.4 default, ADR-0018) are computed once per reachable cell's CHOSEN representation, as a separate, additional correctness check (`eps_A <= target_tol`), not a selection criterion.

15 tracks, seed=42, dt=1 s. Full grid elapsed: 36.5s.

### Reachability (step3 criterion, unchanged) + certified correctness (ADR-0021)

#### DP+SED: bytes (n parameters), reachable fraction; certified `eps_A<=tol` fraction
| sigma \ tol | 0.5 | 2 | 10 |
|---|---|---|---|
| 0 | 481B (n=264.0), 5/15 reachable; cert 5/5 (100.0%) | 470B (n=204.6); cert 15/15 (100.0%) | 282B (n=95.0); cert 15/15 (100.0%) |
| 0.1 | 426B (n=228.0), 1/15 reachable; cert 1/1 (100.0%) | 481B (n=206.6); cert 13/15 (86.7%) | 281B (n=94.2); cert 14/15 (93.3%) |
| 1 | unreachable; cert -- | unreachable; cert -- | 285B (n=92.8); cert 9/15 (60.0%) |
| 5 | unreachable; cert -- | unreachable; cert -- | unreachable; cert -- |

#### LSQ spline (uniform): bytes (n parameters), reachable fraction; certified `eps_A<=tol` fraction
| sigma \ tol | 0.5 | 2 | 10 |
|---|---|---|---|
| 0 | 617B (n=386.7), 14/15 reachable; cert 1/14 (7.1%) | 438B (n=228.2); cert 15/15 (100.0%) | 275B (n=114.6); cert 15/15 (100.0%) |
| 0.1 | 553B (n=320.8), 10/15 reachable; cert 1/10 (10.0%) | 444B (n=230.0); cert 14/15 (93.3%) | 275B (n=114.8); cert 15/15 (100.0%) |
| 1 | unreachable; cert -- | 272B (n=124.0), 4/15 reachable; cert 0/4 (0.0%) | 268B (n=111.0); cert 12/15 (80.0%) |
| 5 | unreachable; cert -- | unreachable; cert -- | 250B (n=101.7), 7/15 reachable; cert 0/7 (0.0%) |

### Comparison against `benchmarks/results/step3.md`'s dt=1 tables

Reachability fractions and `n` (parameter counts) match **exactly** in every one of the 7
non-empty DP+SED cells and 9 non-empty LSQ-uniform cells -- confirming the unchanged generator,
seed, and reachability criterion (ADR-0021) reproduce step3's own selection decision cell for
cell, not just on average.

| Method | Cell (sigma, tol) | step3.md bytes | step8 M0 bytes | delta |
|---|---|---|---|---|
| DP+SED | 0, 0.5 | 479 | 481 | +2 |
| DP+SED | 0, 2 | 467 | 470 | +3 |
| DP+SED | 0, 10 | 278 | 282 | +4 |
| DP+SED | 0.1, 0.5 | 413 | 426 | +13 |
| DP+SED | 0.1, 2 | 478 | 481 | +3 |
| DP+SED | 0.1, 10 | 277 | 281 | +4 |
| DP+SED | 1, 10 | 282 | 285 | +3 |
| LSQ-uniform | 0, 0.5 | 615 | 617 | +2 |
| LSQ-uniform | 0, 2 | 438 | 438 | 0 |
| LSQ-uniform | 0, 10 | 273 | 275 | +2 |
| LSQ-uniform | 0.1, 0.5 | 551 | 553 | +2 |
| LSQ-uniform | 0.1, 2 | 445 | 444 | -1 |
| LSQ-uniform | 0.1, 10 | 274 | 275 | +1 |
| LSQ-uniform | 1, 2 | 271 | 272 | +1 |
| LSQ-uniform | 1, 10 | 266 | 268 | +2 |
| LSQ-uniform | 5, 10 | 249 | 250 | +1 |

All deltas are small (0 to +4B) and fully explained by `encode.py`'s explicit per-segment framing
(`[type: 1 byte][count: varint]`, 2-3 bytes) that step3's ad hoc scheme (direct concatenation of
the t/x/y delta streams, no framing at all) never paid, plus minor zlib-boundary effects from the
different byte layout -- negligible relative to cell sizes of 250-620B. The one outlier, DP+SED at
`(sigma=0.1, tol=0.5)` (+13B), is a **single-track cell** (1/15 reachable both times): with one
data point there is no averaging to smooth out a particular track's own zlib boundary behavior, so
a larger-than-typical delta here is sampling noise, not a sign of a systematic issue -- every
other, better-sampled cell stays within +0 to +4B.

### Segment-format overhead (synthetic, no real tracks)

Fixed 200-vertex payload cut into 1/2/5/10/20 line segments sharing boundary vertices (2.5's dedup rule) -- only the per-segment header (`[type][count varint]`) cost varies; the point payload itself is identical across rows.

| segments | pre-zlib bytes | post-zlib bytes | header bytes | header fraction (pre-zlib) |
|---|---|---|---|---|
| 1 | 2404 | 1743 | 0 | 0.00% |
| 2 | 2405 | 1752 | 1 | 0.04% |
| 5 | 2411 | 1755 | 7 | 0.29% |
| 10 | 2421 | 1771 | 17 | 0.70% |
| 20 | 2441 | 1796 | 37 | 1.52% |

Post-zlib, the relative overhead shrinks further (zlib partially absorbs the repeated
type/count-varint bytes across segments): 1743B -> 1796B is a 3.0% increase in COMPRESSED size
going from 1 to 20 segments, even though pre-zlib header bytes alone already reach 1.52% of the
uncompressed payload. Framing overhead scales roughly linearly with segment count (~1-2B/segment
pre-zlib) and stays a small fraction of typical per-track payload sizes even at 20 segments --
M2's hybrid cutting does not need a special byte budget purely for segment-count framing.

### M0 Conclusions

**Encoding re-verification (spec section 7's M0 artifact: "match step3 or an explained
discrepancy"): matched.** Reachability and `n` reproduce `benchmarks/results/step3.md`'s dt=1
tables exactly, cell for cell, for both DP+SED and LSQ-uniform (ADR-0021: the reachability gate
was deliberately kept identical to step3's, not replaced by the new certificates). Byte counts
differ by a small, fully explained amount (0 to +4B in every multi-track cell; one +13B
single-track outlier, sampling noise) -- `traj.encode`'s explicit per-segment framing costs a
few bytes step3's ad hoc scheme didn't pay, nothing more.

**The certified-correctness split (ADR-0021) is already informative.** DP+SED's `eps_A <=
target_tol` fraction stays high (86.7-100%) everywhere it's reachable except one cell
(sigma=1/tol=10: 60.0%). The LSQ-uniform spline's fraction, in contrast, drops sharply at TIGHT
`tol` relative to sigma/curve complexity: 7.1% (sigma=0, tol=0.5), 10.0% (sigma=0.1, tol=0.5), 0%
(sigma=1, tol=2), 0% (sigma=5, tol=10) -- cells step3's own honest-error criterion accepts as
reachable, yet the certificate against the actually-recorded track regularly exceeds the same
`target_tol`. A plausible mechanism, not yet verified in depth: `certify_spline`'s default
`lam_fallback=0.1` m adds a fixed ~10 cm margin on top of the measured linearization/Fréchet
distance (plus the certificate's own rounding margins), which can by itself approach or exceed a
`target_tol` as tight as 0.5 m or 2 m -- `certify_polyline` carries no comparable additive
constant, consistent with its much higher pass fraction on the same grid. This is an **open
question carried into M1/M2**, not resolved here: spec section 2.2 requires `cost(i, j)` to be
gated by the certificate (`eps <= tol`), unlike M0's purely informational tally, so a spline
segment's usable `tol` range may be narrower in practice than its honest-error-based reachability
would suggest.

**Segment-format overhead is small.** 1.52% pre-zlib / ~3% post-zlib at 20 segments over a
200-vertex payload (see table above) -- M2's hybrid cutting doesn't need special byte-budget
precautions purely from per-segment framing.

**Decisions:** ADR-0021 (the certificate is an additional check, not a reachability gate; segment
type stored as 1 byte, not 1 bit).

**Open issues, carried into M1:** knot removal (spec 2.4), LSQ-free-knot fitting, the free-knot
spline oracle (spec 2.6); the `lam_fallback`-vs-tight-`tol` question above, worth checking before
M2's `cost(i, j)` starts gating on the certificate directly.

### M0.1

Investigates the M0 open question (ADR-0022): does the spline's certified `eps_A<=tol` shortfall at tight `tol` come from `certify_spline`'s fixed `lam_fallback` margin, or from `fit_uniform`'s between-sample oscillation (uncontrolled by its own fitting criterion, but visible to `eps_A`'s Frechet-based check)? Four cells, same 15-track/seed=42 generator as M0.

#### sigma=0, tol=0.5

`lam_fallback` sweep (fitter = fit_uniform):

| lam_fallback | eps_A<=tol fraction | eps_A p50 | eps_A p90 | eps_A max |
|---|---|---|---|---|
| 0.001 | 14 pop: 7.1% | 0.697 | 1.229 | 1.327 |
| 0.01 | 14 pop: 7.1% | 0.703 | 1.237 | 1.334 |
| 0.1 | 14 pop: 7.1% | 0.789 | 1.317 | 1.414 |

Fitter comparison (lam_fallback = 0.1 fixed):

| Fitter | Own population (own criterion) | eps_A<=tol fraction | eps_A p50/p90/max | n median | n p90 |
|---|---|---|---|---|---|
| fit_uniform | 14/15 (honest true-curve error, step3's criterion) | 7.1% | 0.789/1.317/1.414 | 359.5 | 651.1 |
| spline.fit() | 15/15 (own dense-vs-own-polyline convergence) | 0.0% | 0.595/0.606/0.612 | 274.0 | 515.2 |

#### sigma=0.1, tol=0.5

`lam_fallback` sweep (fitter = fit_uniform):

| lam_fallback | eps_A<=tol fraction | eps_A p50 | eps_A p90 | eps_A max |
|---|---|---|---|---|
| 0.001 | 10 pop: 10.0% | 0.670 | 0.826 | 1.230 |
| 0.01 | 10 pop: 10.0% | 0.679 | 0.833 | 1.239 |
| 0.1 | 10 pop: 10.0% | 0.755 | 0.922 | 1.320 |

Fitter comparison (lam_fallback = 0.1 fixed):

| Fitter | Own population (own criterion) | eps_A<=tol fraction | eps_A p50/p90/max | n median | n p90 |
|---|---|---|---|---|---|
| fit_uniform | 10/15 (honest true-curve error, step3's criterion) | 10.0% | 0.755/0.922/1.320 | 245.5 | 595.0 |
| spline.fit() | 15/15 (own dense-vs-own-polyline convergence) | 6.7% | 0.589/0.603/0.613 | 343.0 | 739.0 |

#### sigma=1, tol=2

`lam_fallback` sweep (fitter = fit_uniform):

| lam_fallback | eps_A<=tol fraction | eps_A p50 | eps_A p90 | eps_A max |
|---|---|---|---|---|
| 0.001 | 4 pop: 0.0% | 2.425 | 2.899 | 3.020 |
| 0.01 | 4 pop: 0.0% | 2.435 | 2.911 | 3.033 |
| 0.1 | 4 pop: 0.0% | 2.537 | 2.998 | 3.128 |

Fitter comparison (lam_fallback = 0.1 fixed):

| Fitter | Own population (own criterion) | eps_A<=tol fraction | eps_A p50/p90/max | n median | n p90 |
|---|---|---|---|---|---|
| fit_uniform | 4/15 (honest true-curve error, step3's criterion) | 0.0% | 2.537/2.998/3.128 | 125.5 | 170.5 |
| spline.fit() | 15/15 (own dense-vs-own-polyline convergence) | 73.3% | 1.875/2.074/2.247 | 412.0 | 601.6 |

#### sigma=5, tol=10

`lam_fallback` sweep (fitter = fit_uniform):

| lam_fallback | eps_A<=tol fraction | eps_A p50 | eps_A p90 | eps_A max |
|---|---|---|---|---|
| 0.001 | 7 pop: 0.0% | 12.974 | 15.228 | 15.292 |
| 0.01 | 7 pop: 0.0% | 12.985 | 15.237 | 15.302 |
| 0.1 | 7 pop: 0.0% | 13.076 | 15.333 | 15.413 |

Fitter comparison (lam_fallback = 0.1 fixed):

| Fitter | Own population (own criterion) | eps_A<=tol fraction | eps_A p50/p90/max | n median | n p90 |
|---|---|---|---|---|---|
| fit_uniform | 7/15 (honest true-curve error, step3's criterion) | 0.0% | 13.076/15.333/15.413 | 97.0 | 144.4 |
| spline.fit() | 15/15 (own dense-vs-own-polyline convergence) | 66.7% | 9.329/11.300/12.271 | 436.0 | 961.0 |

### Applying ADR-0022's pre-registered rule

`lam_fallback` sweep, average `eps_A<=tol` fraction across the 4 cells: lam=0.001 -> 4.3% (reference), lam=0.01 -> 4.3%, lam=0.1 -> 4.3%. **Chosen lam_fallback: 0.001** (smallest tested value within 5pp of the lam=0.001 reference).

Fitter comparison, average `eps_A<=tol` fraction across the 4 cells (lam_fallback=0.1): fit_uniform -> 4.3%, spline.fit() -> 36.7% (average fraction gap is +32.4pp, >= 20pp threshold -- decided outright). **Chosen fitter: spline.fit().**

For scale: sweeping `lam_fallback` alone (fit_uniform fixed) moves the average fraction by +0.0pp (lam=0.1 -> lam=0.001); switching fitter alone (lam=0.1 fixed) moves it by +32.4pp (fit_uniform -> spline.fit()). Whichever swing is larger in magnitude is the dominant mechanism; interpretation in the M0.1 Conclusions below.

### M0.1 Conclusions

**`lam_fallback` is ruled out as the explanation.** Across three orders of magnitude
(0.1 -> 0.001), the average `eps_A<=tol` fraction does not move at all (4.3% at every tested
value), and per-cell it moves by at most a few thousandths (e.g. `eps_A` p50 at sigma=0/tol=0.5:
0.789 at lam=0.1 vs. 0.697 at lam=0.001, a difference far smaller than the lam values' own
0.099 spread would suggest). The reason is visible directly in the `eps_A` distributions: in
every failing cell, the *base* `d_F(A, Lin(A'))` term is already well above `target_tol` before
`lam` is even added (e.g. sigma=5/tol=10: `eps_A` p50 ~13m against a 10m tol, an ~30% overshoot
that no realistic `lam` reduction closes). `lam_fallback`'s additive margin was never the
bottleneck M0 saw.

**The fitter -- fit_uniform's between-sample oscillation -- is confirmed as the dominant
mechanism**, with one honest caveat the `n` check (mandatory correction 1) was specifically
added to catch: `spline.fit()` raises the average certified fraction by +32.4pp holding `lam`
fixed, decisively clearing ADR-0022's 20pp threshold. But `n` differs a lot between the two
fitters' chosen representations, and not always in the direction that would inflate the win:
- At `sigma=0, tol=0.5` -- the one cell where `spline.fit()` uses **fewer** parameters than
  fit_uniform (n median 274.0 vs. 359.5) -- it still produces a materially tighter typical
  certificate (`eps_A` p50 0.595 vs. 0.789, ~25% lower). Since this is a same-or-fewer-parameter
  comparison, this result is clean evidence for the oscillation-control mechanism, not an
  artifact of extra control points.
- At `sigma=1, tol=2` and `sigma=5, tol=10` -- the two cells with the largest fraction wins
  (+73.3pp, +66.7pp) -- `spline.fit()` also uses 3.3x and 4.5x **more** parameters (n median 412
  vs. 125.5, and 436 vs. 97.0). Part of these two wins is plausibly attributable to spending more
  bytes (`spline.fit()`'s adaptive densification adds synthetic points until its own dense-error
  check passes, which can cost real knots), not purely to eliminating oscillation between the
  *original* samples -- this data alone cannot cleanly separate the two contributions in these
  two cells.
- At `sigma=0.1, tol=0.5`, `spline.fit()` uses more parameters (343.0 vs. 245.5) for a roughly
  flat-to-slightly-worse fraction (6.7% vs. 10.0%) but a still-tighter `eps_A` p50 (0.589 vs.
  0.755) -- more parameters did not obviously buy a fraction win here, which weakens (without
  ruling out) the "it's just more parameters" reading of the other two cells.

**Net verdict:** the between-sample-oscillation mechanism is real and confirmed independent of
parameter count (the sigma=0 cell), and is the dominant explanation for M0's finding overall --
`lam_fallback` is conclusively not. But the size of `spline.fit()`'s advantage in the two cells
that drive ADR-0022's 20pp decision is partly confounded with it using several times more control
points, which is a real cost, not a free lunch, and is exactly the caveat the `n` check
(mandatory correction 1) was meant to surface rather than hide.

**Very tight `tol` (0.5 m) remains hard for both fitters.** At `sigma=0` and `sigma=0.1`,
neither fitter clears even a 10% certified pass rate -- `spline.fit()`'s `eps_A` values cluster
tightly just above `tol` (p50/p90/max within a few cm of each other) rather than spreading
widely, meaning it is consistently *close* but not passing, unlike fit_uniform's wider, less
predictable spread. This is a genuine limitation to carry into M1/M2, not resolved by either
fitter or lam choice tested here.

### M0.2

Refines ADR-0022 on the fitter it actually selected (`spline.fit()`, not `fit_uniform`), and checks whether `lam_fallback` affects what M2 would store: `traj.encode.SplineSegment` encodes a spline's own control points, never `certify_spline`'s internal linearization `Lin(A')` -- the linearization vertex count and the segment's stored bytes are measured side by side below to confirm this rather than assume it.

#### sigma=0, tol=0.5

| lam_fallback | eps_A<=tol fraction | eps_A p50/p90/max | n (control pts) p50 | Lin(A') vertices p50 | segment bytes p50 | cert time p50/p90 (ms) |
|---|---|---|---|---|---|---|
| 0.001 | 53.3% (15 pop) | 0.494/0.507/0.513 | 274.0 | 2704.0 | 532B | 1354.56/2636.44 |
| 0.01 | 33.3% (15 pop) | 0.503/0.516/0.522 | 274.0 | 880.0 | 532B | 324.10/670.40 |
| 0.1 | 0.0% (15 pop) | 0.595/0.606/0.612 | 274.0 | 296.0 | 532B | 100.10/238.32 |

#### sigma=0.1, tol=0.5

| lam_fallback | eps_A<=tol fraction | eps_A p50/p90/max | n (control pts) p50 | Lin(A') vertices p50 | segment bytes p50 | cert time p50/p90 (ms) |
|---|---|---|---|---|---|---|
| 0.001 | 66.7% (15 pop) | 0.494/0.505/0.511 | 343.0 | 2955.0 | 609B | 1281.18/2815.31 |
| 0.01 | 46.7% (15 pop) | 0.503/0.514/0.520 | 343.0 | 960.0 | 609B | 346.30/753.60 |
| 0.1 | 6.7% (15 pop) | 0.589/0.603/0.613 | 343.0 | 321.0 | 609B | 110.99/282.08 |

#### sigma=1, tol=2

| lam_fallback | eps_A<=tol fraction | eps_A p50/p90/max | n (control pts) p50 | Lin(A') vertices p50 | segment bytes p50 | cert time p50/p90 (ms) |
|---|---|---|---|---|---|---|
| 0.001 | 93.3% (15 pop) | 1.796/1.969/2.163 | 412.0 | 4384.0 | 703B | 2087.41/3528.21 |
| 0.01 | 93.3% (15 pop) | 1.805/1.978/2.166 | 412.0 | 1483.0 | 703B | 557.55/858.11 |
| 0.1 | 73.3% (15 pop) | 1.875/2.074/2.247 | 412.0 | 490.0 | 703B | 185.89/307.78 |

#### sigma=5, tol=10

| lam_fallback | eps_A<=tol fraction | eps_A p50/p90/max | n (control pts) p50 | Lin(A') vertices p50 | segment bytes p50 | cert time p50/p90 (ms) |
|---|---|---|---|---|---|---|
| 0.001 | 66.7% (15 pop) | 9.188/11.212/12.135 | 436.0 | 9818.0 | 871B | 4679.78/8730.16 |
| 0.01 | 66.7% (15 pop) | 9.199/11.220/12.148 | 436.0 | 3106.0 | 871B | 1301.62/3690.46 |
| 0.1 | 66.7% (15 pop) | 9.329/11.300/12.271 | 436.0 | 1040.0 | 871B | 436.95/1031.61 |

### Applying the refined lam rule

| lam_fallback | avg eps_A<=tol fraction (4 cells) | avg Lin(A') vertices | avg segment bytes | avg cert time p50 (ms) |
|---|---|---|---|---|
| 0.001 | 70.0% | 4965.2 | 678.8B | 2350.73 |
| 0.01 | 60.0% | 1607.2 | 678.8B | 632.39 |
| 0.1 | 36.7% | 536.8 | 678.8B | 208.48 |

default (smallest within 5pp of the lam=0.001 reference) is 0.001; tie-break set (equal average fraction, 70.0%): [0.001] -- chosen by smallest bytes then smallest time: 0.001. **Chosen lam_fallback: 0.001.**

### tol=0.5 m spline availability (internal tol tighter than the 0.5 m target)

Certified with the chosen lam_fallback (0.001).

| sigma | internal tol | converged | eps_A<=0.5 fraction | eps_A p50 | eps_A min | n p50 |
|---|---|---|---|---|---|---|
| 0 | 0.25 | 15/15 | 100.0% | 0.250 | 0.188 | 514.0 |
| 0 | 0.1 | 14/15 | 100.0% | 0.103 | 0.086 | 781.0 |
| 0.1 | 0.25 | 14/15 | 100.0% | 0.247 | 0.215 | 736.0 |
| 0.1 | 0.1 | 10/15 | 100.0% | 0.102 | 0.088 | 1042.0 |

At least one internal-tol/sigma combination does certify `eps_A<=0.5` -- see the table above for which, and the M0.2 Conclusions below for the parameter-count cost.

### M0.2 Conclusions

**`lam_fallback` matters a great deal on `spline.fit()` -- the opposite of M0.1's finding on
`fit_uniform`.** Average `eps_A<=tol` fraction across the 4 cells: 70.0% at lam=0.001, 60.0% at
lam=0.01, 36.7% at lam=0.1 -- a real, monotonic 33.3pp swing, not the ~0pp M0.1 found for
`fit_uniform`. The mechanism is the mirror image of M0.1's: `spline.fit()` already controls the
*base* `d_F(A, Lin(A'))` term tightly (that's exactly what it's designed to do), so once that
term stops dominating, the *additive* `lam` margin (0.001 to 0.1, a 0.099 spread) becomes the
main lever left. Both findings are consistent with the same underlying picture, read together:
`lam_fallback` only matters once the fitter itself is good enough that the base error isn't
already blowing the budget on its own.

**Bytes are confirmed lam-invariant, exactly as the encoding design predicts.** Segment bytes
are bit-identical across all three tested `lam` values in every cell (532B/532B/532B at
sigma=0/tol=0.5; 609B/609B/609B; 703B/703B/703B; 871B/871B/871B) -- because
`traj.encode.SplineSegment` stores the fit's own internal knots and control points, never
`certify_spline`'s internal linearization `Lin(A')`, which is produced and consumed entirely
inside the certificate and discarded afterward. `lam` affects **only** `eps_A` and certification
time, never what M2 would actually store. The `Lin(A')` vertex-count column shows why the time
moves: at `sigma=5/tol=10`, the linearization needs a median of 9818 vertices at lam=0.001 vs.
1040 at lam=0.1 (~9.4x more subdivision) purely to certify tightly enough for that lam, with no
effect on the segment that eventually gets encoded.

**That subdivision cost is real and large.** Median certification time per track, averaged
across the 4 cells: 2350.7 ms at lam=0.001, 632.4 ms at lam=0.01, 208.5 ms at lam=0.1 -- an
~11.3x spread. The worst single cell (`sigma=5/tol=10`) reaches 4679.8 ms median / 8730.2 ms p90
at lam=0.001. This is the caveat this ADR refinement was specifically checking for (mirroring
M0.1's `n`-confound caveat): the mechanical rule never even reaches this number, because
lam=0.001's average fraction (70.0%) is not tied with lam=0.01 (60.0%, a 10pp gap) or lam=0.1
(36.7%, a 33.3pp gap) -- both exceed the 5pp tie-break window, so the rule's primary criterion
decides outright and the bytes/time tie-break is never consulted. **Applying the rule exactly as
written still selects `lam_fallback=0.001`** -- but a ~2.35s median (and up to ~8.7s p90) per
spline-segment certification is a real, measured cost M1/M2's actual DP cost function (spec
section 2.2's `cost(i,j)`, evaluated `O(m*W)` times per track, spec section 2.3) needs to budget
for explicitly, consistent with spec section 10's own generic "computation time" risk item --
now with a concrete number attached rather than a vague concern.

**tol=0.5 m spline availability: the opposite of what M0.1/M0.2's naive check suggested.**
Calling `spline.fit()` directly at the 0.5 m target gave a certified pass fraction of at most
53.3% (at the now-chosen lam=0.001) and as low as 0% (at lam=0.1) -- suggestive of unavailability.
But asking `spline.fit()` to hit a substantially *tighter* internal tol first (0.25 m or 0.1 m,
i.e. roughly half or a fifth of the actual 0.5 m target) and certifying the result with
lam=0.001 certifies `eps_A<=0.5` for **100% of every converged track at every internal-tol/sigma
combination tested** -- spline segments ARE certificate-available at `tol=0.5 m`, just not via
the naive "fit directly at the target" construction this investigation and M0.1 both used for
comparison. The cost is real: roughly 1.5x-3x more control points than the naive attempt (n
median 514-1042 here vs. 274-343 at the naive target-tol fit in the lam-sweep table above).
This validates spec section 2.4's actual planned M1 construction (start from a dense/interpolating
spline, remove knots down to the certified limit) over the cruder direct-fit approach used for
comparison purposes in M0/M0.1/M0.2 -- knot removal starting dense and backing off carefully is
functionally what "aim well below the target, then verify" approximates here. **The earlier open
question is resolved in the opposite direction from the hypothesis it was raised to test: at
`tol=0.5 m`, a polyline-vs-spline choice in M2 is not forced by certificate unavailability; it
remains a genuine cost/geometry trade-off, decided by `cost(i,j)`, not a construction-availability
default.**

### M1.0 -- budget check (ADR-0023)

**Estimate, not a measurement of M2's real runtime** (M2 doesn't exist yet; section 2.1's candidate count `m` is a labeled proxy from M0's DP+SED kept-vertex counts, not a measured quantity). Purpose: decide `knot_removal.py`'s certification `lam_fallback` *before* writing it (ADR-0023).

Assumptions: `m` (candidates/track) proxy = 68.2 (synthetic, from M0's DP+SED n at sigma=0/tol=2/dt=1) / 150 (GeoLife, longer tracks); `W`=8; M2 grid = 15 synthetic tracks x 4 sigma x 3 tol x 2 dt (360 cells) + 585 GeoLife tracks x 2 tol (1170 cells); DP transitions/track = m*W (546 synthetic, 1200 GeoLife); total transitions across the grid ~ 1,600,416.

#### Measured `certify_spline` cost by segment length

| segment vertices | n control points p50 | time p50 lam=0.001 (ms) | time p50 lam=0.1 (ms) |
|---|---|---|---|
| 10 | 4.0 | 2.81 | 0.31 |
| 25 | 4.0 | 4.21 | 0.33 |
| 50 | 4.0 | 24.23 | 2.70 |
| 100 | 4.0 | 29.84 | 3.09 |

#### Old (full-track-based) vs. new (segment-based) M2 projection

Old estimate reuses ADR-0022's full-track measurements (`benchmarks/results/step8.md` M0.2): 2350.73 ms/call at lam=0.001, 208.48 ms/call at lam=0.1 -- treating each DP transition's spline-cost evaluation as if it cost as much as certifying an ENTIRE track, which this section's own measurement shows is far too pessimistic. New estimate uses the 50-vertex bracket above (24.23 ms lam=0.001, 2.70 ms lam=0.1), the closest tested size to a typical DP candidate segment's own length.

| lam_fallback | old estimate (full-track basis) | new estimate (segment basis) | ratio |
|---|---|---|---|
| 0.001 | 1,045.0 h | 10.8 h | 97.0x |
| 0.1 | 92.7 h | 1.2 h | 77.2x |

**Applying ADR-0023's rule:** the segment-based lam=0.001 projection is 10.8 h (> the 2 h threshold). **Decision: lam_fallback=0.1 + tuned internal tol (ADR-0023 rule triggered).**

### M1.0 -- internal-tol fraction at lam=0.1 (ADR-0023)

Reference: M0.2's average `eps_A<=tol` fraction at lam_fallback=0.001, internal tol = target tol (ratio=1.0), across the same 4 cells: **70.0%** (reused from `benchmarks/results/step8.md`'s M0.2 section, not recomputed).

#### sigma=0, tol=0.5 (lam_fallback=0.1)

| internal-tol ratio | converged | eps_A<=tol fraction | n p50 |
|---|---|---|---|
| 1 | 15/15 | 0.0% | 274.0 |
| 0.5 | 15/15 | 100.0% | 514.0 |
| 0.25 | 15/15 | 100.0% | 745.0 |
| 0.1 | 11/15 | 100.0% | 1402.0 |

#### sigma=0.1, tol=0.5 (lam_fallback=0.1)

| internal-tol ratio | converged | eps_A<=tol fraction | n p50 |
|---|---|---|---|
| 1 | 15/15 | 6.7% | 343.0 |
| 0.5 | 14/15 | 100.0% | 736.0 |
| 0.25 | 10/15 | 100.0% | 965.5 |
| 0.1 | 15/15 | 100.0% | 1474.0 |

#### sigma=1, tol=2 (lam_fallback=0.1)

| internal-tol ratio | converged | eps_A<=tol fraction | n p50 |
|---|---|---|---|
| 1 | 15/15 | 73.3% | 412.0 |
| 0.5 | 15/15 | 100.0% | 628.0 |
| 0.25 | 15/15 | 100.0% | 736.0 |
| 0.1 | 15/15 | 100.0% | 1447.0 |

#### sigma=5, tol=10 (lam_fallback=0.1)

| internal-tol ratio | converged | eps_A<=tol fraction | n p50 |
|---|---|---|---|
| 1 | 15/15 | 66.7% | 436.0 |
| 0.5 | 15/15 | 100.0% | 679.0 |
| 0.25 | 15/15 | 100.0% | 775.0 |
| 0.1 | 15/15 | 100.0% | 1234.0 |

#### Applying ADR-0023's internal-tol rule

| ratio | avg eps_A<=tol fraction (4 cells) | avg n p50 | within 5pp of reference? |
|---|---|---|---|
| 1 | 36.7% | 366.2 | no |
| 0.5 | 100.0% | 639.2 | yes |
| 0.25 | 100.0% | 805.4 | yes |
| 0.1 | 100.0% | 1389.2 | yes |

**Chosen internal-tol ratio: 0.5** (largest ratio -- i.e. cheapest construction -- whose average fraction stays within 5pp of the 70.0% reference).
