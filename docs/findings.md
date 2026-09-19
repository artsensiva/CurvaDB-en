# Research findings: spline vs DP polyline for GPS trajectories

See also: [the full project timeline](history.md) (including the
pre-repository stage) and [next steps](next_steps.md) (resolution,
interview plan, threshold to return to code).

## Question

Does storing GPS trajectories as cubic B-splines beat the classic
"Douglas-Peucker (DP) simplification + search by discrete Frechet
distance" scheme -- on bytes per track, nearest-neighbor search recall,
and kinematics reconstruction accuracy (velocity, acceleration)?

Short answer: no, on compression -- under the conditions tested there's
no advantage (see "Key numbers" and "Conclusion" below; the test is
limited to uniform knots for the oracle -- see the caveat there). On
kinematics -- no worse than DP, with one specific advantage (no
catastrophic acceleration outliers).

## Timeline

### step0 -- the first (broken) measurement

A naive comparison of the raw polyline, the DP polyline, and a cubic
spline (`scipy.interpolate.splprep`, error controlled only at the
original timestamps) on 200 GeoLife tracks, `tol=10`m. The result looked
crushing for the spline: recall@10 against the exact Frechet distance --
DP 0.997, spline 0.707 (`benchmarks/results/step0.md`).

Diagnostics (`step0_diagnostics.md`) showed: this isn't a property of
the method, but a methodological error. `fit()` only held the `error <=
tol` guarantee **at the original timestamps** -- between them, the
spline could drift arbitrarily far. For 82% of the 200 tracks (median
11×tol), the error between timestamps exceeded tol, and for several,
by kilometers, due to anomalous GPS jumps in the raw GeoLife data. The
DP polyline, by construction, holds the same guarantee along the ENTIRE
polyline (the maximum across 200 tracks -- exactly 10.000m, the tol
boundary). The recall difference was explained by exactly this, not by
the point count in the spline representation (`spline_same` and
`spline_arclen` recall matched not because of a shared bug, but because
the discrete Frechet distance is insensitive to the resampling density
of the same, already-distorted curve).

### step1 -- an honest methodology

Two fixes: `src/traj/clean.py` splits tracks at GPS breaks (dt > 30s or
speed > 70 m/s) before fitting, removing the cause of the jumps rather
than the symptom; `fit()` is augmented with a dense error check BETWEEN
timestamps (not just at them), with adaptive synthetic-knot insertion
when violated. The acceptance threshold (>= 99% of tracks with an honest
error <= tol) was reached with margin -- 100% (585/585 cleaned tracks).

After this, the spline's recall@10 nearly matched DP: **0.707 → 0.972**
(DP: 0.997 → 0.996) -- the gap explained in step0_diagnostics.md
essentially closed (`step1.md` §3, §6).

But on compression (hypothesis A), DP wins at EVERY tol tested (2..50m):
e.g. at tol=10m -- DP 366 bytes/track, spline 1346 bytes/track, 3.7x more
(`step1.md` §5) -- because honest dense error control forces the spline
to keep far more control points than DP has vertices for the same
guarantee. A negative result, not forced.

On kinematics (hypothesis B) -- a mixed result: on synthetic data with
known v/a, the spline and DP+PCHIP give similar median error, but
DP+PCHIP has a heavy tail on acceleration (max ~4991 m/s² vs ~34 for the
spline) -- PCHIP's numerical fragility on DP vertices close together in
time, which the spline doesn't share. A third reference point, Kalman
(constant-acceleration + RTS smoothing), is more accurate than both on
average, but noticeably worse on recall for detecting sharp maneuvers
(0.068 vs 0.284 for the spline and 0.338 for DP+PCHIP) -- the CA model
oversmooths turns (`step1.md` §4).

### step2 -- searching for a niche for the spline

Hypothesis: maybe there's a zone (noise/tol ratio) where the spline is
still more compact than DP+SED (time-aware DP). Synthetic data -- smooth
road geometry (straights + clothoid turns), honest selection of each
method's internal parameter via a log grid relative to the target tol.

A narrow, non-monotone winning zone was found: at tol=1-5m and
sigma<=0.1m the spline is 15-25% more compact than DP+SED (0.78-0.85x,
on 24-30 of 30 tracks), but at a coarse tol=20m, DP+SED wins again at
ANY noise level, including sigma=0 (1.08-2.35x) (`step2.md` §2, §4). A
check (§3) showed this isn't a FITPACK suboptimality artifact --
`splprep` is more compact than naive uniform knots on every track
tested.

Practical applicability of this zone: ordinary smartphone/automotive GPS
(noise ~3-10m) does NOT fall into the zone -- GPS accuracy is already
comparable to tol itself. RTK/differential GPS (accuracy ~0.02-0.1m)
falls into the zone at tol 1-5m; inertial/lidar systems (~0.05-0.3m) are
a borderline case, closer to tol~5m (`step2.md` §4).

### step3 -- the decisive experiment (and its failure)

The cause of the spline's loss at coarse tol was suspected to be
`fit()`'s own contract: it always stays tethered to the polyline of
noisy points (synthetic knots, interpolation at s=0) -- meaning it
inherits the chord error and cannot smooth noise as freely as a genuine
least-squares spline. step3 tested this directly:
`src/traj/spline_lsq.py` -- `make_lsq_spline` on x(t), y(t) WITHOUT
tethering to the polyline, with uniform and adaptive (residual-driven)
knots, plus an **oracle** -- the same fitter, fit directly to the true
(noise-free) curve, independent of noise and observation sparsity (but,
as discovered while preparing step5, using only uniform knots -- see the
caveat in "Conclusion" below; this is a ceiling for uniform knots, not
for the spline in general).

step2's synthetic data was extended with variable speed (acceleration/
braking/stops) and different observation steps (dt = 1, 5, 15s). Three
criteria were fixed BEFORE the run (`docs/prompts/step3.md`):

- **K1** (compression): after quantization+zlib, the spline is >= 30%
  smaller than DP+SED in at least two cells with sigma/tol <= 0.1, at
  >= 80% mutual reachability.
- **K2** (reconstruction at dt >= 5s): there are cells where the spline
  is >= 80% reachable and DP+SED is <= 20%.
- **K3** (reconstruction under noise): at sigma=5, there is a tol where
  the spline is >= 80% reachable and DP+SED is <= 20%.

**All three criteria failed** on 15 tracks (`step3.md` §3):

- K1: the best observed spline compression was 6% (dt=1s, sigma=0,
  tol=2m: 438B vs 467B for DP+SED, 0.94x), not the required 30%.
- K2: DP+SED is indeed almost always unreachable at dt>=5s (0-13% of
  tracks), but the spline never exceeded 53% reachability in any cell
  (>= 80% was required).
- K3: at sigma=5, DP+SED is almost always unreachable, but the spline
  also never reached 80% at any tol (maximum 47%).

Most importantly: **even the oracle is NOT more compact than DP+SED**.
At tol=10m, the oracle is 292 bytes, DP+SED is 278 bytes; at tol=2m, the
oracle is 457 bytes, DP+SED is 467 bytes (`step3.md` §2, the oracle
table and the DP+SED table at dt=1s, sigma=0). A spline fit directly to
the exact, noise-free curve, with not a drop of noise or sampling
sparsity -- with UNIFORM knots -- gives no byte advantage over the
simplified polyline. At the time, this was NOT a test of optimal/
adaptive knot placement (see "Conclusion" below) -- that question
became hypothesis H1, since closed by step8 (see "Closed hypotheses"
below).

## Key numbers (summary)

| Metric | step0 (broken) | step1 (honest) | step2 (niche) | step3 (decisive) |
|---|---|---|---|---|
| Recall@10 spline / DP | 0.707 / 0.997 | 0.972 / 0.996 | -- | -- |
| Compression at a typical tol | spline 3052B / DP 833B (tol=10m, old fitter) | spline 1346B / DP 366B (tol=10m, 3.7x) | 0.78-0.85x (narrow zone, tol=1-5m, sigma<=0.1m) | best observed 0.94x (6% smaller); oracle 292B / DP+SED 278B (tol=10m) |

## Conclusion

The compression hypothesis (a cubic B-spline is more compact than a
DP-simplified polyline at the same honest error against the ground
truth) on consumer GPS is **closed** (step1, step3). The oracle (a
spline fit directly to the dense, noise-free true curve -- no noise, no
sampling sparsity, no fitter artifacts) also gives no byte advantage
over DP+SED, so the issue isn't noise or observation sparsity.

**Important caveat** (verified by reading `src/traj/spline_lsq.py`):
`fit_oracle()` uses ONLY uniform knots (`_build_uniform`, not
`_build_adaptive`) -- this is a ceiling for uniform knots, not the
theoretical limit of spline representability in general. In step2
(`step2.md` §3), FITPACK's adaptive knots (`splprep`) gave ~0.70x the
control points of naive uniform knots on the tracks tested -- optimal
knot placement could meaningfully reduce the parameter count and bring
it closer to criterion K1 (30% compression). At the time, this was not
tested -- it became hypothesis H1, now closed: step8's free-knot oracle
(greedy certified knot removal directly on the true curve, spec section
2.6) tested exactly this and still found no advantage (see "Closed
hypotheses" below). Also, even a
hypothetical ~30% byte savings would not by itself be product value: a
GPS track already takes up a few kilobytes, and savings of that order
aren't a bottleneck in any of the use cases considered.

The narrow winning zone found in step2 (§4, tol=1-5m, sigma<=0.1m) was
based on the old `fit()`, tethered to the polyline of noisy points --
**this conclusion from step2.md §4 is overturned by step3's results**
(subject to the caveat above about the oracle's uniform knots).

## Limitations (and why they don't change the compression conclusion)

- **Synthetic data, not real GPS data** (step2, step3) -- but step1
  showed the same negative result on 585 real, cleaned GeoLife tracks
  (`step1.md` §5), so the conclusion doesn't rest on synthetic data
  alone.
- **15 tracks in step3** -- a small sample for cell-level reachability
  fractions (80%/20% are coarse thresholds at n=15). But this applies to
  criteria K2/K3 (reconstruction), not to the oracle's byte comparison
  (K1/the overall compression conclusion), which doesn't depend on
  reachability fractions.
- **dt=15s is uninformative**: at this observation step, ALL methods
  (DP+SED and both LSQ spline variants) are unreachable in 100% of grid
  cells -- the likely cause: sharp turns (radius 30-150m) at variable
  speed, on some of the 15 tracks, fit entirely within one interval
  between neighboring 15-second samples, so no method that only sees
  those samples can recover the turn. This is an artifact of sampling
  too sparsely relative to the geometry, not a signal in favor of any
  method -- it doesn't affect the conclusion.
- **sigma=5 at tol=10 (K3)** -- a harsh noise/tol ratio (the noise is
  comparable to the tolerance), a borderline test regime.
- **DP+SED is a simple competitor**, not map-matching or a Kalman
  filter (Kalman showed the best average kinematic accuracy in step1,
  but wasn't tested here as a compression/reconstruction method).
- **`spline_lsq.py` has no regularization** -- `make_lsq_spline` (unlike
  `spline.fit()`/FITPACK) has no built-in smoothing parameter `s`; as
  the number of knots grows toward near-interpolation, numerical
  instability is possible (handled by tracking the best result seen
  during growth, see `src/traj/spline_lsq.py`).

None of these limitations except one apply to the **oracle**: it's fit
directly to the exact, noise-free, dense curve, so questions of noise,
observation sparsity, and the fitter's robustness to them don't apply.
But the oracle itself uses ONLY uniform knots (see the caveat in
"Conclusion" above) -- that's its own limitation, not inherited from the
list above. With that caveat: under the conditions tested (uniform
knots), there is no compression advantage; the conclusion's robustness
to optimal knot placement was, at the time, untested -- that became
hypothesis H1, now closed (see "Closed hypotheses" below).

## Closed hypotheses

### H1 (compression): free/optimal knots on exact data -- CLOSED (step8)

Adaptive knot placement driven by the true curve's own geometry (not
uniform, as in step3's `fit_oracle`, and not driven by a noisy trial
fit's residuals, as in `fit_adaptive`) could in principle give a more
compact representation on EXACT (noise-free) data than DP+SED, placing
knots where the geometry is genuinely more complex -- left open by
step3 because its own oracle only tested uniform knots.

**Tested and closed in step8** (`docs/specs/step8_A_hybrid.md`, section
2.6's free-knot oracle: greedy, certified knot removal applied directly
to a dense sample of the true curve -- `src/traj/knot_removal.py`,
`benchmarks/results/step8.md`'s M1 section, criterion A3). Even this
free, geometry-driven knot placement does not beat DP+SED by the
required 20%: at no tested tolerance does the oracle/DP+SED byte ratio
clear 0.80x (tol=0.5 m: 1.050x; tol=2 m: 0.834x, the closest miss;
tol=10 m: 0.879x). Applying spec section 8's decision rule verbatim:
H1 is closed -- even a free-knot spline on the ideal curve is not more
compact than DP+SED by 20%, under the conditions tested. Full detail
and the exact same-basis methodology: `docs/reviews/step8_M1.md`,
`docs/phases/step8_summary.md`.

## Open hypotheses (untested)

### H2 (product): analytical derivatives and search by kappa(t) on exact data

For sources with precise positioning (RTK, robots, drones, surgical
robotics), the spline gives analytical derivatives (velocity,
acceleration, curvature) "for free," unlike a polyline. This could be
valuable INDEPENDENTLY of whether the spline wins on compression -- if
the customer's real problem isn't "store more compactly" but "search for
similar maneuvers/segments by shape or kinematics." Untested, demand
unconfirmed -- to be checked via interviews, not code (see
[docs/next_steps.md](next_steps.md)).

## Not investigated

- **step2, the cause of the spline's loss at coarse tol=20m**: at every
  noise level tested (including sigma=0), DP+SED is more compact than
  the spline at tol=20m -- and this isn't explained by FITPACK
  suboptimality (step2.md §3 showed `splprep` is more compact than
  naive uniform knots). Candidate cause: the old `fit()`'s honest
  densify strategy grows control points for the dense check BETWEEN
  1-second samples regardless of the turn's shape -- not investigated
  further (became moot after step3, where the compression question is
  settled separately, on a fitter without this densify strategy).
- **step3, the gap between the oracle and `fit_uniform`/`fit_adaptive`**:
  the oracle is reachable 15/15 at every tol, while the real fitters
  (which only see noisy/sparse samples) reach 0-53% in cells with
  dt>=5s or sigma=5. Expected (the oracle doesn't see noise/sparsity),
  but the size of the gap wasn't decomposed into noise's contribution
  separately from sampling sparsity's.
- **step3, dt=15s -- 100% unreachability for every method**: see
  "Limitations" above -- the plausible cause (a sharp turn entirely
  within one interval between samples) wasn't checked per track
  individually among the 15, and wasn't separated from a possible
  contribution from the internal-parameter log-scan's resolution (8
  points, not a continuous sweep).

## Reusable

- **`src/traj/clean.py`** -- splits GPS tracks at time/speed breaks and
  applies bbox filtering; broadly applicable to any task with raw
  GeoLife-like GPS logs.
- **`src/traj/frechet.py`** -- the exact discrete Frechet distance
  (Eiter-Mannila) on Numba with early exit at a threshold; independent
  of the trajectory representation chosen (polyline, spline -- anything
  reducible to a polyline of points).
- **`simplify_sed_with_indices` (`src/traj/simplify.py`)** -- a
  time-aware (SED) variant of Douglas-Peucker; stricter than spatial DP
  at the same tol, accounts for uneven sampling in time.
- **Methodology**:
  - an oracle (a fit on the true, not observed, data) as a method's
    ceiling -- separates the representation's limitations from the
    specific fitter implementation's limitations, but requires
    explicitly checking that the oracle itself is tuned to the method's
    full potential (in step3 it used only uniform knots -- see
    "Conclusion" above), otherwise the "ceiling" turns out understated;
  - success criteria are fixed BEFORE the experiment runs (K1-K3 in
    `docs/prompts/step3.md`), not adjusted after the fact;
  - the honest error is measured AGAINST THE GROUND TRUTH on a dense
    grid, not against the observed (potentially noisy/sparse) points --
    otherwise a method can look accurate simply by describing its own
    noise well (see step0_diagnostics.md and the step3 discovery of
    non-monotone error vs. the fitter's internal parameter).
- **A certificate on the continuous Fréchet distance is a shape
  guarantee, not a time-synchrony guarantee.** It permits the
  reconstruction to lead or lag the original curve in time, as long as
  the two stay close in space -- a legitimate Fréchet-optimal
  correspondence. If an application needs synchrony (matching position
  at the SAME timestamp, not just the same place at some time), use SED
  or L2 error instead, not a Fréchet certificate. Found while closing
  H1 (step8, `docs/specs/step8_A_hybrid.md` section 2.6's free-knot
  oracle): certified knot removal reached 100% `eps_A<=tol` at every
  tested tolerance, while reachability under a fixed, time-synchronized
  criterion fell from 86.7% to 66.7% as the tolerance grew -- a property
  of the Fréchet metric itself, not specific to H1 or to this fitter.
