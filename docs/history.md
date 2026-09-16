# Full project history

Timeline from the original idea (semantic search via "curves" over
embeddings) to the decision to end the project as a research effort with
a negative result. Facts up to step0 come from a discussion outside the
repository (passed on verbatim in `docs/prompts/step5_history.md`) and
confirmed by the file `suggestions_denkweg` (committed at the repo root,
`git log --follow -- suggestions_denkweg`). Numbers from step0 onward are
from `benchmarks/results/*.md`, commits are from `git log`.

## Stage A. The original idea

Level 1 (`faa8cf5`, 2025-11-18, "Complete Level 1: Minimal Hilbert
Vector DB") and Level 2 (`ce7dbc3`, 2025-11-19, "Level 2 MVP: Curve-based
semantic search with 31% recall") implemented CurvaDB's idea as semantic
search: documents represented as "curves" -- a Hilbert index over
sentence-transformers embeddings (Level 1), then splines and functional
PCA on top of them (Level 2), Frechet/DTW metrics for comparison (the
Level 1-3 plan, see `docs/legacy.md`).

**Question:** can semantic search be sped up/improved by treating
embeddings as curves rather than vectors.

**What was done:** a minimal Hilbert vector DB (Level 1), a curve-based
search MVP with 31% recall (Level 2).

## Stage B. Critiquing the idea

An external review (`suggestions_denkweg`) pointed out that the idea
lumps four unrelated mathematical problems under one word, "curve":

| Component | What it is mathematically | Role |
|---|---|---|
| Hilbert curve | a surjection [0,1] → [0,1]ⁿ | index |
| Spline | a piecewise-polynomial function | representation |
| Functional PCA | an eigenfunction decomposition | dimensionality reduction |
| Frechet / DTW | a metric on a space of curves | comparison |

The key problem: the coordinate order of the ℝ⁷⁶⁸ embedding is arbitrary
(learned by gradient descent, carries no topological meaning) -- "a
curve through 768 coordinates in index order" gives neither compression,
nor meaningful derivatives, nor noise smoothing (the embedding is
deterministic for a given text, there's no noise to smooth).

**Stage conclusion:** the advice was: one type of curve, one domain
where the parameter t is physically meaningful (time, arc length), one
metric, a linear baseline first. Proposal: a GPS trajectory store on
cubic B-splines with Frechet-based search; milestone 1 -- fleet
telematics (consumer GPS), milestone 2 -- trajectories in ℝ³ (medical
robotics, robots).

**What changed in the plan:** dropping embeddings as the domain, moving
to GPS trajectories, where the parameter t = time is physically
meaningful.

## Stage C. Fact-checking before starting

Three checks before writing any code:

- **The niche isn't empty**: PostGIS has `ST_FrechetDistance`, MobilityDB
  exists, map-matching (Valhalla, OSRM) is common in telematics -- there
  is no ready-made "curve database" on splines, but the infrastructure
  around the problem already exists.
- **Dishonest comparison**: the original estimate "80 KB polyline vs 2-4
  KB spline" (see the table in `suggestions_denkweg`, §1.3 "Milestone
  1") compared the raw (unsimplified) polyline against the spline -- the
  honest opponent is the polyline AFTER Douglas-Peucker, not the raw one.
- **Recommendation**: benchmark against real competitors first, then
  10-15 industry interviews, before investing in a product. **Interviews
  were NOT conducted for the entire project** -- this is the direct
  reason the step5 resolution comes back to this same recommendation.

**Stack decision:** Python (scipy, shapely, numba) for research; Rust
(PyO3, pgrx) only makes sense for a future product, not for the research
phase.

## Stage D. Organizing the work

Branch `trajectory-pivot`, GeoLife 1.3 data, Claude Code in VS Code,
project context in `CLAUDE.md` (`e2f851e`, "Add CLAUDE.md with project
context"). Process lessons recorded before experiments began: keep
prompts as files in `docs/prompts/` (they survive context restarts and
save budget on re-explaining the task); don't restart VS Code during
background benchmark runs; for long runs, start with a time-limited
pilot before the full run.

## step0. The first (broken) measurement

**Question:** how does a naive spline (`scipy.interpolate.splprep`,
error controlled only at the original timestamps) compare to the DP
polyline on bytes, search latency, and recall@10 by discrete Frechet
distance?

**What was done** (`e32293d` io.py, `883dcd1` spline fitting, `6b76ce0`
DP simplification, `d32a796` Frechet metric on Numba, `fd7f6c5`
`benchmarks/step0.py`, `bc82d9b` tests): a run on 200 GeoLife tracks,
tol=10m.

**Key numbers** (`benchmarks/results/step0.md`): bytes/track (float64)
-- DP 833, spline 3052; recall@10 -- DP 0.997, spline 0.707.

**Methodological error found** (`b84e6fe`,
`benchmarks/results/step0_diagnostics.md`): the byte comparison ignored
that DP doesn't store timestamps separately, and the spline's recall
needed an honest error check BETWEEN timestamps -- which was missing.
`fit()` held `error <= tol` only at the original timestamps; between
them, 82% of the 200 tracks exceeded tol (median 11×tol, ~110 m), and
for some, by kilometers due to GPS jumps in the raw data. The DP
polyline, by construction, holds the same guarantee everywhere along the
line. This is exactly what explains the 0.707 recall -- not a property
of the spline as a representation, but a broken contract in this
specific implementation.

**Stage conclusion:** the comparison was unfair to the spline -- an
honest methodology is needed before drawing conclusions about the method
itself.

**What changed in the plan:** step1 -- fix the methodology, rather than
abandon the hypothesis.

## step1. Honest methodology

**Question:** what will the comparison show once both found problems are
fixed -- GPS jumps in the raw data and error control only at the
timestamps?

**What was done** (`4ff2520` `clean.py` -- splitting at breaks with
dt>30s or speed>70 m/s, `63cdeed` honest dense error control in `fit()`,
`5887b6e`/`1be1f91`/`9062256`/`e1d32bb` recall/kinematics/compression/
search benchmarks, `ce2554e` summary conclusions): 585 cleaned tracks out
of 200 originals, the acceptance threshold for honest fitting (≥99% of
tracks with error ≤tol) reached with margin -- 100%.

**Key numbers** (`benchmarks/results/step1.md`): spline recall@10 0.707
→ 0.972 (DP 0.997 → 0.996, §3/§6); compression -- DP is more compact
than the spline at EVERY tol from 2 to 50m (at tol=10m: DP 366B, spline
1346B, 3.7x, §5); kinematics -- the spline and DP+PCHIP are close on
median error, but DP+PCHIP has acceleration outliers up to ~4991 m/s² vs
~34 for the spline; Kalman (CA+RTS) is more accurate than both on
average, but worse on recall for sharp maneuvers (0.068 vs 0.284/0.338,
§4).

**External comments on the results** (given in the step5 prompt, not
tested experimentally within the project): the spline's honest
competitor for compression is DP+SED (both account for time), not plain
spatial DP; the compression gap narrowed as tol decreased. The kinematics
test was judged unreliable: at tol=10m and 5m noise, only 74 of 759
points had |a|>3 m/s², and Kalman was tuned by acceleration RMSE on a
subsample -- that fits the method to the very metric it's evaluated on.
In real telematics, maneuvers are detected with an accelerometer, and
velocity comes from Doppler GNSS measurements, not by differentiating
coordinates -- meaning the whole kinematics test (step1 §4) compares
methods on a task that production systems solve differently.

**Stage conclusion:** hypothesis A (compression/search) -- a negative
result, not forced. Hypothesis B (kinematics) -- a mixed result, but the
test itself is methodologically shaky (see the comments above).

**What changed in the plan:** look for a specific niche (noise/tol
ratio), rather than a general "spline better/worse" verdict.

## step2. Searching for a niche

**Question:** is there a zone (noise/tol) where the spline is more
compact than DP+SED on more realistic (road-like) geometry?

**What was done** (`c68bcb7` a spline/DP+SED map by (noise, tol) on
smooth synthetic data, `0f6b372` checking FITPACK suboptimality,
`005820a` conclusions): 30 synthetic tracks (straights + clothoid turns),
honest selection of each method's internal parameter via a log grid.

**Key numbers** (`benchmarks/results/step2.md` §2, §4): a narrow
winning zone for the spline -- 0.78-0.85x at tol=1-5m, sigma≤0.1m (24-30
of 30 tracks); at a coarse tol=20m, DP+SED wins at ANY noise level,
including sigma=0 (1.08-2.35x). §3 showed this isn't a FITPACK
suboptimality artifact -- `splprep` gives ~0.70x the control points of
naive uniform knots on every track tested.

**Main analytical finding:** the spline was structurally tethered to the
polyline -- `fit()`'s contract holds tol relative to the straight
segments between samples (interpolation at s=0 + synthetic knots). The
fraction of reachable tracks matched EXACTLY between the spline and DP in
every grid cell (3/30, 25/30, 24/30, 26/30) -- a direct consequence of
that tethering. The chord error L²/(8R) explains why small tol values
are unreachable even at sigma=0: the spline inherits the polyline's
geometric error regardless of noise.

**External comment after step2** (step5 prompt): the niche's
applicability is RTK/differential GPS or lidar-grade accuracy, not
ordinary smartphone GPS; drop selling "compression" as the main value
proposition; suggestion to test knot placement driven by the curvature
profile kappa(t).

**Stage conclusion:** the niche exists, but is narrow, non-monotone, and
practically unreachable for mass-market consumer GPS.

**What changed in the plan:** step3 -- check whether `fit()`'s tethering
to the polyline is itself the cause of the failure at coarse tol, using
a separate fitter without that tethering.

## step3. The decisive experiment

**Question:** if the spline's tethering to the polyline is removed (an
honest least-squares fitter, `make_lsq_spline`, directly on x(t)/y(t))
and an oracle is added (a spline on the true curve -- the ceiling of
representability), does the compression hypothesis survive a decisive
test with criteria fixed BEFORE the run?

**What was done** (`02226e2` step2 diagnostics, `66e16f4`
`src/traj/spline_lsq.py`, `2875068` a fix for numerical instability,
`73de1ba`/`91cc16e` `benchmarks/step3_decisive.py` and the full run,
`80350ba` verdicts): 15 synthetic tracks with variable speed, a
sigma×tol×dt grid, criteria K1 (compression), K2 (reconstruction at
dt≥5s), K3 (reconstruction at sigma=5) -- all fixed in
`docs/prompts/step3.md` before the run.

**Key numbers** (`benchmarks/results/step3.md` §2-3): all three
criteria are NOT met. The best observed spline compression was 0.94x (6%
smaller, not 30%, K1). At dt≥5s, DP+SED is almost always unreachable, but
the spline never exceeded 53% reachability in any cell (80% was
required, K2). At sigma=5, the same picture, maximum 47% (K3). The
oracle (a spline on the true curve) is NOT more compact than DP+SED: 292B
vs 278B at tol=10m, 457B vs 467B at tol=2m.

**Limitations found after the run** (step5 prompt): at dt=15s a sharp
turn fits entirely within one interval between samples -- this is a
limit of the information present in the data, not a signal in favor of
the method; no approach can recover what isn't in the observations.
sigma=5 at tol=10 -- the noise/tolerance ratio is too harsh for a max-norm
error (a single outlier sinks the whole cell). DP+SED is a deliberately
simple competitor for reconstruction; honest alternatives are Kalman
(see step1 §4) or Hermite interpolation with Doppler velocity (these
use extra information that wasn't available in this experiment). The LSQ
fitter (`spline_lsq.py`) has no FITPACK regularization -- as the number
of knots grows toward near-interpolation, numerical instability is
possible (handled by tracking the best result seen during growth).

## MANDATORY CORRECTION (step5)

While preparing step5, a methodological overreach was found in
conclusions already written (`docs/findings.md`, `docs/blog_draft.md`,
`README.md` after step4): phrasing like "robustly no, all the way to the
theoretical limit of spline representability" and "the representation
itself gives no advantage" -- overstated.

Code check (`src/traj/spline_lsq.py`): `fit_oracle()` (lines 191-199)
calls `_bisect_fit(..., knot_mode="oracle")`; inside `_bisect_fit` --
`build = _build_adaptive if knot_mode == "adaptive" else
_build_uniform`. With `knot_mode="oracle"`, the `== "adaptive"` condition
is false, so `_build_uniform` is used. **The oracle in step3 used ONLY
uniform knots**, not the optimal (free) placement. This is not the
theoretical limit of spline representability in general, but the ceiling
specifically for uniform knots.

This matters: in step2 (`step2.md` §3), FITPACK's adaptive knots
(`splprep`) gave ~0.70x the control points of naive uniform knots on the
tracks tested -- meaning optimal knot placement can meaningfully reduce
the parameter count. How close this would bring K1 (30% compression) is
untested.

**Corrected phrasing** (replaces "robustly, all the way to the
theoretical limit" and "the representation itself gives no advantage" in
every document): under the conditions tested (uniform knots for the
oracle), there is no compression advantage; optimal knot placement is
untested. Also, even a hypothetical ~30% byte savings would not by
itself be product value -- a GPS track already takes up a few kilobytes,
and savings of that order aren't a bottleneck for any of the use cases
considered (telematics, robotics).

**There are now explicitly two open hypotheses**, not one vague "open
hypothesis":

- **H1 (compression):** free/optimal knots on exact (noise-free) data --
  untested.
- **H2 (product):** analytical derivatives and search over the curvature
  profile kappa(t) on exact data (RTK, robots, drones) -- untested,
  demand unconfirmed (the interviews from Stage C were never conducted).

## Resolution

- The spline as a compressor for consumer-GPS trajectories -- **closed**
  (step1, step3).
- Compression on exact data -- no advantage under the conditions tested,
  H1 is open but not a priority (see the correction above: a ~30%
  savings would not be product value even if confirmed).
- The next step is **not code, but 8-10 interviews** (robotics,
  RTK-equipped agritech, drones, surgical robotics). Questions: how do
  you currently store and compare trajectories; do you need search over
  similar motions; where does your velocity/curvature come from --
  sensors or differentiating coordinates; what's currently awkward.
- Threshold to return to code: at least 3 of 10 interviewees
  independently name a trajectory search/comparison task by shape or
  kinematics that current tools don't solve. Then -- an H2 experiment on
  KITTI or TUM RGB-D with criteria fixed before the run (see
  `docs/next_steps.md`). Otherwise the project ends as a research effort
  with a published result.

## Table: stage / question / answer / conclusion status

| Stage | Question | Answer | Conclusion status |
|---|---|---|---|
| A | Would representing embeddings as curves speed up semantic search? | Implemented (Level 1-2), but the idea was never compared against a baseline | Abandoned, pivot to Stage B |
| B | Is the idea mathematically sound? | No -- 4 unrelated problems, the embedding's coordinate order is arbitrary | Pivot to GPS trajectories |
| C | Is the niche empty, was the original comparison honest? | The niche isn't empty but isn't occupied either; the 80KB/2-4KB comparison was unfair; interviews needed | Interviews not conducted (still true today) |
| step0 | How does a naive spline compare to DP? | Crushingly worse (recall 0.707) | Overturned by step1 -- a methodological error, not a property of the method |
| step1 | What does an honest methodology show? | Recall nearly matched (0.972/0.996), but DP wins on compression everywhere (3.7x) | Stands |
| step2 | Is there a compression niche by noise/tol? | A narrow zone, 0.78-0.85x at tol=1-5m, sigma≤0.1m (RTK/lidar only) | **Overturned by step3** |
| step3 | Does the compression hypothesis survive a decisive test without polyline tethering? | All three criteria (K1-K3) fail; even the oracle isn't more compact than DP+SED | Stands, with a caveat (see the correction below) |
| step5 correction | Was that a complete oracle? | No, uniform knots only -- the ceiling isn't absolute | H1 open, untested |
| Resolution | What's next? | Not code -- 8-10 interviews; a clearly defined return threshold | Stands |

## Process lessons

- Keep prompts as files in `docs/prompts/` -- they survive context
  restarts and save budget on re-explaining the task (applied starting
  with step3).
- Don't restart VS Code during background benchmark runs -- the
  background process won't survive an environment restart.
- For long runs, start with a pilot under an explicit time limit, then
  the full run (applied in every step1-step3 benchmark: the step2 pilot
  took 22s against a 3-minute limit, the step3 pilot took 9.5s against a
  2-minute limit).
- Fix success criteria BEFORE running an experiment (step3 K1-K3) --
  otherwise there's a temptation to interpret an ambiguous result in the
  hypothesis's favor after the fact.
- Set up an oracle (a fit on the true, not observed, data) as a reference
  earlier, rather than spending time trying to tune the fitter against
  observations -- but check that the oracle itself isn't artificially
  understated (see the MANDATORY CORRECTION above: an oracle with uniform
  knots is not the full theoretical ceiling).
- The recommendation to run industry interviews (Stage C) was given
  before any code was written and was completely ignored for the entire
  duration of the project -- the direct reason the step5 resolution comes
  back to exactly that same recommendation, rather than a new experiment.
