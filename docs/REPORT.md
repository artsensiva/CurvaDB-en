# CurvaDB: full technical report

This is a standalone report: it can be read without opening any other file in this repository,
though every claim in it links back to the specific file it came from. It compiles the project's
entire research record -- the original (rejected) idea, the GPS-trajectory compression study
(`step0`-`step3`), the certified curve store (`step7`), and the hybrid-representation closure of
hypothesis H1 (`step8`) -- from the documents already committed to this repository. No new
experiments were run and no code was changed to produce this report; every number below is
quoted from the results file, decision record, or review named next to it, and where a number
appears in more than one place with different values, the discrepancy is noted explicitly and
the results file's own number is used (never memory, never a narrative document like a blog
draft).

## Table of contents

0. [Abstract](#0-abstract)
1. [How to read this repository](#1-how-to-read-this-repository)
2. [Origin and critique](#2-origin-and-critique)
3. [Methodology](#3-methodology)
4. [step0: the first measurement and its diagnosis](#4-step0-the-first-measurement-and-its-diagnosis)
5. [step1: an honest methodology](#5-step1-an-honest-methodology)
6. [step2: searching for a niche](#6-step2-searching-for-a-niche)
7. [step3: the decisive experiment](#7-step3-the-decisive-experiment)
8. [Interim conclusion](#8-interim-conclusion)
9. [step7: the certified curve store](#9-step7-the-certified-curve-store)
10. [step8: the hybrid representation and closing H1](#10-step8-the-hybrid-representation-and-closing-h1)
11. [Cross-cutting findings](#11-cross-cutting-findings)
12. [Limitations and open questions](#12-limitations-and-open-questions)
13. [Reproduction](#13-reproduction)
14. [Index: ADRs and glossary](#14-index-adrs-and-glossary)

## 0. Abstract

Five results, five lines -- four negative, one positive:

1. **(Negative)** A cubic B-spline is not a better compressor for real, consumer-grade GPS
   trajectories than a Douglas-Peucker-simplified polyline: step0's first measurement looked
   catastrophic for the spline (recall@10 0.707 vs. 0.997), but that was a broken measurement,
   not a property of splines; corrected in step1, the polyline still wins on bytes at every
   tested tolerance (3.7x smaller at `tol=10m`).
2. **(Negative)** Removing every confound step1/step2 left open -- the spline's tethering to the
   polyline, the fitter's own weaknesses, even noise and sampling sparsity via an oracle fit
   directly to the true curve -- still doesn't save the spline: step3's three criteria, fixed
   before the run, all fail, and the oracle itself isn't more compact than DP+SED (`benchmarks/
   results/step3.md`: 292 bytes vs. 278 bytes at `tol=10m`).
3. **(Negative)** The one gap step3's own oracle left open -- it used only uniform knots, not
   free/optimal placement -- is closed the same way in step8: a certified, free-knot oracle
   fit directly on the true curve still doesn't beat DP+SED by the required 20% at any tested
   tolerance (closest: 0.834x at `tol=2m`, against an 0.80x bar). Hypothesis H1 is closed.
4. **(Positive)** A certified curve store *can* answer continuous-Fréchet range queries over
   *compressed* trajectories with a provable error bound, with zero misses and zero false
   positives across 6,000,000 tested combinations, and cheaper than an honest, uncompressed
   exact competitor (`benchmarks/results/step7.md`: 20.1ms vs. 41.7ms median per query).
5. **(Negative)** That same store's spec-designated *primary* spline certificate mechanism
   doesn't work as designed (83.6% of splines fall back to the secondary path, against a
   required `<=10%`), and the correctness guarantee has a real, measured price against an
   uncertified approximate search with no guarantees (2.7x-4.4x slower).

**One sentence on method:** every result above was reached the same way -- criteria fixed
*before* the run (not adjusted afterward to fit the data), an oracle fit to the ground truth to
separate the representation's own limits from the fitter's, error measured against ground truth
rather than against the method's own noisy observations, and a negative result treated as a
normal, reportable outcome rather than something to fix until it goes away.

## 1. How to read this repository

The repository holds two unrelated projects under one name, in sequence: an abandoned idea
(`src/level1`-`level3`) and the actual research this report covers (`src/traj` and everything
under `docs/` except the legacy files named below). Nothing under `src/level1`-`level3` is used
by, or referenced from, the trajectory research.

| Path | What it is |
|---|---|
| `src/level1/`, `src/level2/`, `src/level3/`, `examples/level1_demo.py` | The original idea (semantic search via a Hilbert curve over text embeddings). `level1` was implemented and measured; `level2`/`level3` are mostly stubs. Frozen after Stage B's critique (section 2 below); not developed further. |
| `src/traj/` | The real research code: GPS-track I/O and cleaning (`io.py`, `clean.py`), simplification (`simplify.py`, DP and time-aware SED), splines (`spline.py`, dense-error-controlled; `spline_lsq.py`, direct LSQ, no polyline tethering), Bézier/knot-insertion utilities (`bezier.py`), the continuous Fréchet metric and certificates (`frechet.py`, `frechet_cont.py`, `certify.py`), interval-query logic (`intervals.py`), the step8 segment encoding (`encode.py`) and knot removal (`knot_removal.py`). |
| `benchmarks/` | One script (or a small family of scripts) per research step, e.g. `step0.py`, `step1_clean.py`, `step2_crossover.py`, `step3_decisive.py`, `step7_certify.py`/`step7_query.py`/`step7_m4.py`, `step8_hybrid.py` and its `step8_m0*`/`step8_m1*` companions. Each writes its own section into `benchmarks/results/<step>.md`. |
| `benchmarks/results/*.md` | The actual results, one file per step (`step0.md`, `step0_diagnostics.md`, `step1.md`, `step2.md`, `step3.md`, `step7.md`, `step8.md`). This report's numbers are quoted from these files. |
| `tests/traj/` | Property tests and regression tests for `src/traj/`, including tests that are specifically constructed to fail on a since-fixed bug (see section 11). |
| `docs/specs/` | The frozen specifications for `step7` and `step8` (and four unbuilt product specs, `P1`-`P4`, out of scope for this report) plus an overview. Their "Acceptance" sections are never edited once committed -- deviations are recorded in ADRs and `docs/ROADMAP.md`, not by changing the spec text. |
| `docs/decisions/ADR-*.md` | 23 architecture decision records, index and template in `docs/decisions/README.md`. Superseded decisions are marked "Superseded by ADR-XXXX," never deleted. See the index table in section 14. |
| `docs/phases/*.md` | One-page summaries written at each gate (`step7_summary.md` for gate G1, `step8_summary.md` for gate G2). |
| `docs/reviews/*.md` | Short verdict documents for milestones that needed one before the next milestone could start (`step7_M0.md` through `step7_M2.md`, `step8_M1.md`). |
| `docs/ROADMAP.md` | Phases, gates, and a dated decision journal (section 8 of that file) -- the authoritative record of when each gate opened and why. |
| `docs/findings.md` / `docs/ru/findings.md` | The step0-step3 research findings, written as a standalone summary (this report supersedes it in scope, not in content -- the numbers agree). |
| `docs/history.md` / `docs/ru/history.md` | The full project history end to end, including the pre-repository stage and a mandatory correction to an overstated step3 conclusion. |
| `docs/next_steps.md` / `docs/ru/next_steps.md` | The resolution (interviews, not more code) and the return-to-code threshold. |
| `docs/prompts/` | Task prompts kept as files (from step3 onward) rather than only in chat, so they survive context restarts. |
| `docs/blog_draft.md`, `docs/blog_step7.md`, `docs/ru/blog_draft.md`, `docs/ru/blog_step7.md` | Blog-style adaptations for a general engineering audience -- narrative, not sources of numeric fact for this report (see the discrepancy noted in section 4). |
| `SPECIFY.md`, `constitution.md`, `curve_based_db_components.md`, `roadmap.md`, `suggestions_denkweg` (repository root) | Pre-pivot planning artifacts from the original embeddings idea. Superseded by `docs/ROADMAP.md` and `docs/specs/`; kept for history, not current guidance. |
| `data/`, `venv/` | GeoLife GPS data and the Python virtual environment; both git-ignored, not part of the repository's content. |

## 2. Origin and critique

The project began as something else entirely. Level 1 (commit `faa8cf5`, "Complete Level 1:
Minimal Hilbert Vector DB") and Level 2 (`ce7dbc3`, "Level 2 MVP: Curve-based semantic search
with 31% recall") implemented semantic document search by treating sentence-transformer
embeddings as "curves": a Hilbert-curve index over the raw embedding vectors (Level 1), then
splines and functional PCA on top of them, compared by Fréchet/DTW distance (Level 2). Source:
`docs/history.md`, Stage A.

An external review (kept in the repository at `suggestions_denkweg`) found the idea unsound.
It lumped four unrelated mathematical objects under one word, "curve" (`docs/history.md`,
Stage B):

| Component | What it is mathematically | Role |
|---|---|---|
| Hilbert curve | a surjection `[0,1] -> [0,1]^n` | index |
| Spline | a piecewise-polynomial function | representation |
| Functional PCA | an eigenfunction decomposition | dimensionality reduction |
| Fréchet / DTW | a metric on a space of curves | comparison |

The decisive flaw: the coordinate order of a 768-dimensional embedding is arbitrary (learned by
gradient descent, carries no topological meaning), so "a curve through 768 coordinates in index
order" gives no compression, no meaningful derivatives, and no noise smoothing (an embedding is
deterministic for a given input text -- there is no noise to smooth). The review's advice: pick
one type of curve, one domain where the curve parameter is physically meaningful (time or arc
length), one metric, and compare against a linear baseline first. Proposal: a GPS-trajectory
store on cubic B-splines with Fréchet-based search.

Before writing any new code, three fact-checks were done (`docs/history.md`, Stage C):

- **Is the niche empty?** No. PostGIS already ships `ST_FrechetDistance`; MobilityDB exists;
  map-matching (Valhalla, OSRM) is standard in telematics. The infrastructure around the
  problem already exists, even if a dedicated "curve database on splines" does not.
- **Was the original comparison honest?** No. The idea's own pitch ("80KB polyline vs. 2-4KB
  spline," `suggestions_denkweg` section 1.3) compared a *raw, unsimplified* polyline against
  the spline. The honest competitor is a polyline *after* Douglas-Peucker simplification, not
  the raw one -- this single correction reframed the entire research question and is the direct
  ancestor of `step0`-`step3`'s DP+SED baseline.
- **Recommendation:** benchmark against real competitors, then run 10-15 industry interviews,
  before investing further. **The interviews were never conducted, for the entire project** --
  this is the reason the eventual resolution (section 8, and again after step8) comes back to
  the same recommendation.

The pivot moved the domain from text embeddings to GPS trajectories, where the spline's
parameter `t` (time) is physically meaningful, and set up the comparison that the rest of this
report is about: a cubic B-spline against a Douglas-Peucker-simplified polyline, judged by
Fréchet-distance search recall and by byte size, at an honestly-measured error tolerance.
(`docs/history.md`, Stage D: branch `trajectory-pivot`, GeoLife 1.3 data, work organized around
`CLAUDE.md`.)

## 3. Methodology

Four disciplines recur through every step described in this report, and are the reason its
negative results are trustworthy rather than symptoms of a broken experiment:

- **Criteria fixed before the run.** step3's three criteria (K1-K3, section 7 below) were
  written into `docs/prompts/step3.md` before `benchmarks/step3_decisive.py` was run -- removing
  the temptation to read an ambiguous result in the hypothesis's favor after the fact
  (`docs/history.md`, "Process lessons"). step7 and step8 continued the same discipline: ADR-0013
  (step7) and ADR-0022/ADR-0023 (step8) record a decision *rule* before the deciding run, then
  fill in the outcome afterward without revising the rule to match what came out.
- **An oracle as the method's ceiling, checked for what it actually tests.** A fit on the true,
  unobserved curve (not the noisy/sparse samples) separates "the representation can't do this"
  from "this particular fitter, on this particular data, didn't do this." step3's own oracle
  used only uniform knots (`src/traj/spline_lsq.py`'s `fit_oracle`) -- a real limitation, caught
  after the fact (the "MANDATORY CORRECTION," section 7) and only closed by step8's free-knot
  oracle (section 10).
- **Error measured against ground truth, never against the method's own observations.** step0's
  spline fit was checked only at its own sample timestamps, and looked fine there while drifting
  by kilometers between them (section 4). Every step from step1 onward checks error on a dense
  grid against the *true* curve (synthetic data) or the full, unsimplified track (real data),
  specifically to prevent a method from looking accurate merely by describing its own noise well.
- **Decisions are recorded, including reversed ones, and specs are never quietly edited.**
  Architecture Decision Records (`docs/decisions/ADR-*.md`) are written in the same commit as
  the code change they justify; a superseded ADR is marked "Superseded by ADR-XXXX," never
  deleted (e.g. ADR-0003's discriminant clamp is superseded by ADR-0004's strict check; ADR-0012's
  threshold-sensitivity finding is superseded by ADR-0014's domain-bug fix). `docs/specs/`'s
  "Acceptance" sections, once committed, are never edited during a phase -- a deviation from the
  spec's literal text is recorded in an ADR and in `docs/ROADMAP.md` section 8, with the spec
  text itself left untouched as the frozen reference (e.g. ADR-0018's switch of the primary
  spline certificate, section 9 below).

Task prompts were kept as files under `docs/prompts/` from step3 onward, specifically so they
survive a context restart and don't need re-explaining (`docs/history.md`, "Process lessons").
Executor and critic roles were kept separate at several points: `suggestions_denkweg`'s external
review of the original idea (section 2), and step7's own milestone review chain
(`docs/reviews/step7_M0.md` through `step7_M2.md`), where a milestone's own author did not get
to unilaterally declare it accepted -- see section 9.

## 4. step0: the first measurement and its diagnosis

**Setup** (`benchmarks/results/step0.md`): 200 raw GeoLife tracks, `tol=10.0m`. Track length in
points: min 55, median 412, max 1979, mean 576. Mean DP-polyline vertices: 52.1; mean spline
control points: 125.8; 0/200 fits failed to converge.

**Bytes per track (float64):** raw 9224B, DP polyline 833B, spline (knots + coefficients) 3052B.

**Recall@10 against the exact Fréchet distance on raw tracks:** DP 0.997, spline (same
resampling as DP) 0.707, spline (arc-length resampling) 0.707 -- identical between the two
spline variants.

This looked like a decisive loss for the spline. It wasn't a property of splines; it was a
broken measurement, diagnosed in a separate pass (`benchmarks/results/step0_diagnostics.md`,
run with scratch scripts outside the repository, results recorded as a report, no code changed):

1. **Why do both spline variants recall identically (0.707)?** Not a bug: for all 200 tracks the
   two spline variants have a *different* number of points and non-identical coordinate arrays
   (`np.allclose = False` on every track checked). The identical recall is because the discrete
   Fréchet distance is only weakly sensitive to the resampling density of the same (already
   distorted) curve -- it is set by the worst-case point correspondence, not the point count.
2. **The spline oscillates between timestamps.** `fit()`'s error control held only *at* the
   original sample timestamps. On a dense grid between them (20 points per interval), the
   maximum deviation from the raw polyline exceeds `tol` (10m) on **164/200 tracks (82%)**; the
   median exceeding track overshoots by a factor of **11x** (about 110m at `tol=10m`); the worst
   tracks overshoot by kilometers, traced to specific anomalous GPS jumps in the raw GeoLife data
   (e.g. one track implies a 3.8km position jump in 3 seconds, roughly 4580 km/h) plus, on at
   least one track with no such jump, plain accumulated overshoot on dense, real, noisy samples.
   The diagnostics file tested correlation of `log(max_dist)` against several candidate causes
   and found it **weak everywhere (max `|r|=0.27`, with `log(n_raw)`)** -- no single dominant
   cause. (`docs/blog_draft.md`, a narrative adaptation and not one of this report's sources,
   states a stronger correlation, 0.69, specifically with the maximum gap between recorded
   points; that number does not appear in `benchmarks/results/step0_diagnostics.md`, whose own
   analysis is the one quoted above, so this report uses the results file's number, 0.27, and
   flags the discrepancy rather than silently using the larger one.)
3. **The DP polyline does not have this problem.** The maximum deviation of *any* original
   point from the DP polyline (not just removed points -- literally every point) has median
   9.697m, p90 9.968m, max exactly 10.000m across all 200 tracks: 0/200 tracks exceed `tol`. This
   is a direct, structural consequence of Douglas-Peucker's own guarantee, which holds
   everywhere along the polyline, not only at its vertices.

**Conclusion** (`step0_diagnostics.md`): the spline's "error `<=tol`" guarantee held only at its
own sample timestamps and was massively violated between them; this fully explains step0's
0.707 vs. 0.997 recall gap. It is a fitter-contract problem, not evidence that splines are a
worse representation in general -- resolved by controlling the dense (between-sample) error
directly, which is exactly what step1 does next.

## 5. step1: an honest methodology

**Cleaning** (`benchmarks/results/step1.md` §1): a time gap `>30s` or speed `>70m/s` splits a
track; single-point segments are dropped; a Beijing bounding box filters obviously-wrong points;
segments are then kept only if they have 50-2000 points. From 200 raw tracks (199,080 points):
3113 extra segments from break-splitting, 1504 single-point outliers removed, 30,718 points
dropped outside the bounding box, 860 segments dropped by the length filter -- **585 cleaned
tracks, 141,948 points**, out of the original 200 raw tracks.

**Honest dense-error control** (§2): the fitter now checks `>=10` dense-grid points per interval
between neighboring points of the *cleaned* track, adding synthetic knots where needed (up to 8
rounds) before growing the smoothing parameter for compactness. **Acceptance threshold (`>=99%`
of tracks with dense error `<=tol`) reached: 585/585 = 100.00%** (time parametrization; chord
parametrization also reaches 100.00% at a lower mean control-point count, 59.5 vs. 69.9, but time
parametrization is used going forward since analytical derivatives are only implemented for it).

**Recall@10, cleaned + honestly fit** (§3, §6): on the 585 cleaned tracks, recall@10 against the
exact Fréchet distance on raw (cleaned) tracks is **DP 0.996 (95% CI 0.991-0.999), spline 0.972
(95% CI 0.956-0987)** -- confidence intervals barely overlapping, DP still slightly ahead, but
the step0 gap (0.997 vs. 0.707) has almost entirely closed. This confirms step0's diagnosis
directly: fixing the measurement, not the representation, closed nearly all of the gap.

**Compression** (§5): across `tol in {2, 5, 10, 20, 50}` meters, on 60 cleaned tracks, DP is
smaller than the spline (float64 bytes) at **every** tested tolerance -- e.g. at `tol=10m`, DP is
**366 bytes/track** vs. the spline's **1346 bytes/track**, a **3.7x** gap, because honest
dense-error control forces the spline to keep far more control points than DP needs vertices for
the same guarantee. Time-aware DP+SED is, as expected, somewhat more expensive than spatial DP
(it enforces a stricter, time-synchronized criterion) but stays cheaper than the spline at every
tolerance too.

**Kinematics** (§4, a separate hypothesis: does the spline beat DP on recovering velocity/
acceleration, not on bytes?): on 30 synthetic accel/brake/turn/stop trajectories with 5m Gaussian
position noise, median velocity/acceleration error was **spline 1.521 / 0.795 m/s, m/s²**; **DP +
PCHIP 2.316 / 0.919**; **Kalman (constant-acceleration + RTS) 1.253 / 0.647** -- Kalman best on
both medians, but its recall for detecting sharp maneuvers (`|a|>3 m/s²`) was only **0.068**,
against **0.284** (spline) and **0.338** (DP+PCHIP): the constant-acceleration model oversmooths
turns. DP+PCHIP's acceleration error has a heavy tail: RMSE **248.150 m/s²**, against **3.882** for the
spline and **1.701** for Kalman, and max error **~4991 m/s²** for DP+PCHIP vs. **~33.6** for the
spline and **~15.3** for Kalman -- a numerical-fragility artifact of PCHIP's second derivative
blowing up when two DP vertices land close together in time; the spline has no equivalent
failure mode.

**Bottom line** (§7): the honest spline is a correct, predictable, but *not* more compact and
*not* more accurate-for-search alternative to the DP polyline; its main practical advantage over
DP+PCHIP is the absence of catastrophic acceleration outliers. This negative result (hypothesis
A: compression/search) was not forced -- "that's simply how it came out," per `CLAUDE.md`'s own
stated policy on negative results.

## 6. step2: searching for a niche

**Question:** is there a (noise, tolerance) zone where the spline beats DP+SED on more
realistic, road-like synthetic geometry (straights plus clothoid-entry/constant-curvature/
clothoid-exit turns)? (`benchmarks/results/step2.md`.)

**The niche exists, but is narrow and non-monotone** (§2, §4): on 30 synthetic tracks, the
spline is **0.78-0.85x** the size of DP+SED (15-25% smaller) at `tol=1-5m` and `sigma<=0.1m`
(24-30 of 30 tracks reachable) -- but at a coarser `tol=20m`, DP+SED is smaller again at
**every** noise level tested, **including `sigma=0`** (ratio 1.08-2.35x). The advantage is not a
simple function of the noise/tolerance ratio; it is a consequence of the specific turn geometry
(radius 30-150m) forcing DP-like methods to cut extra vertices at moderate tolerance, while at
coarse tolerance DP-like methods get away with very few vertices for the whole track and the
spline can no longer keep up.

**Not a FITPACK artifact** (§3): comparing `splprep` (adaptive knots) against a minimal
*uniform*-knot LSQ fit on 5 tracks, `splprep` used **54.8** control points on average against
**78.8** for uniform knots -- a **0.70x** ratio. Adaptive knot placement was already working
effectively where tested; it does not explain the spline's loss at coarse tolerance.

**Main analytical finding, structural tethering** (`docs/history.md`, step2 section): the
spline's fraction of reachable tracks matched the DP polyline's **exactly**, cell for cell (3/30,
25/30, 24/30, 26/30), because `fit()`'s own contract holds tolerance *relative to the straight
segments between samples* -- the spline is structurally tethered to the polyline it's built
from. The chord error formula `L^2/(8R)` explains why small tolerances are unreachable even at
`sigma=0`: the spline inherits the polyline's own geometric error regardless of noise. This
finding directly motivated step3: build a fitter with no such tethering (`src/traj/
spline_lsq.py`, a direct least-squares fit of `x(t)`/`y(t)`, unlike `spline.fit()`).

**Applicability** (§4): the winning zone (`tol~1-5m`, `sigma<=0.1m`) requires positioning
accuracy well beyond ordinary smartphone/automotive GPS (`sigma` typically 3-10m, already
comparable to or worse than the tolerance itself) -- RTK/differential GPS and lidar/inertial
localization (`sigma~0.02-0.3m`) can fall into it; consumer GPS, and therefore GeoLife, does not.

**Stage conclusion:** the niche exists but is narrow, non-monotone, and unreachable for
mass-market consumer GPS -- consistent with, not contradicting, step1's negative result on real
GeoLife tracks.

## 7. step3: the decisive experiment

**Question:** with the polyline-tethering confound removed (a direct LSQ fitter, `src/traj/
spline_lsq.py`, no relationship to any polyline) and an oracle ceiling added (a spline fit
directly to the *true*, noise-free curve), does the compression hypothesis survive a decisive
test with criteria fixed *before* the run? (`benchmarks/results/step3.md`; criteria recorded in
`docs/prompts/step3.md` before the run.)

**Setup:** 15 synthetic tracks with variable speed (accelerate/cruise/brake/stop), a grid of
`sigma in {0, 0.1, 1, 5}`, `tol in {0.5, 2, 10}`, `dt in {1, 5, 15}` seconds, seed 42. Three
pre-registered criteria, `>=80%`/`<=20%` reachability thresholds at `n=15` tracks (12/15 and
3/15):

| # | Criterion | Verdict | Key numbers |
|---|---|---|---|
| K1 | Compression: spline `>=30%` smaller than DP+SED (after zlib) in `>=2` cells with `sigma/tol<=0.1`, both `>=80%` reachable | **Not met** | Best observed: **0.94x** (6% smaller, not 30%) at `dt=1, sigma=0, tol=2` (438B vs. 467B); only 5 cells (all `dt=1`) clear the reachability bar, none clears 30% |
| K2 | Reconstruction: at `dt>=5s`, a cell where the spline `>=80%` reachable and DP+SED `<=20%` | **Not met** | DP+SED's own condition holds almost everywhere at `dt>=5s`, but the spline's maximum reachability there is **53%** (`dt=5, sigma=0.1, tol=10`) -- never reaches 80% |
| K3 | Noise: at `sigma=5`, a tol where the spline `>=80%` reachable and DP+SED `<=20%` | **Not met** | DP+SED is unreachable (0/15) almost everywhere at `sigma=5` (condition holds trivially), but the spline's maximum reachability there is **47%** (`dt=1, tol=10`, uniform knots) |

**The oracle -- a spline fit directly to the true, noise-free curve** (independent of `sigma`/
`dt`, uniform knots only): reachable **15/15 at every tested `tol`**, but **not more compact
than DP+SED**: at `tol=10m`, oracle **292 bytes** vs. DP+SED's **278 bytes**; at `tol=2m`, oracle
**457 bytes** vs. DP+SED's **467 bytes** (the one tol where the oracle is *slightly* smaller, by
2%); at `tol=0.5m`, oracle **605 bytes** against DP+SED's own 5/15-reachable **479 bytes** (not a
valid comparison at that tol, since DP+SED itself barely reaches 33% there). This is the
strongest single result of step3: even with noise, sampling sparsity, and every fitter weakness
removed, a spline still isn't reliably more compact than DP+SED.

A secondary, purely diagnostic finding in the fitter itself (`spline_lsq.py`'s own contract):
its internal error criterion (direct residual *at* the sample points, no polyline involved) is
*not monotone* in its own internal tolerance parameter -- too tight an internal tolerance on
sparse/noisy points can produce a spline that fits the samples closely but oscillates wildly
*between* them, invisible to the fitter's own training-point residual. Handled by a coarse
log-scan (robust to the non-monotonicity) rather than pure bisection from the tightest setting.

**Verdict** (`docs/history.md`, step3 section; `docs/prompts/step3.md`'s pre-registered rule
applied as written): K1, K2, and K3 all fail. The spline's advantage seen in step2 is only
partially present here (its reachable-track share grows relative to DP+SED where DP+SED itself
struggles), but never clears the pre-registered 80% bar. **A caveat found after the run** (see
section 8): the oracle used only uniform knots, not free/optimal placement -- closed later, by
step8 (section 10).

## 8. Interim conclusion

What CurvaDB's spline-vs-polyline research (step0-step3) proved, as of step3, and what it left
open:

**Proved:**
- A cubic B-spline is not a better compressor for real, consumer-GPS trajectories than a
  Douglas-Peucker-simplified polyline, at any tested tolerance (step1: 3.7x at `tol=10m`).
- This isn't explained by noise, sampling sparsity, the specific fitter's weaknesses, or FITPACK
  suboptimality: removing each of these in turn (step2's FITPACK check, step3's untethered
  fitter and oracle) does not change the conclusion. Even the oracle -- fit directly to the true
  curve, no noise, no sparsity -- isn't more compact than DP+SED (step3: 292B vs. 278B at
  `tol=10m`).
- A narrow niche does exist on smooth, road-like synthetic geometry at very specific (noise,
  tolerance) combinations (step2: 0.78-0.85x at `tol=1-5m`, `sigma<=0.1m`) -- but it requires
  positioning accuracy beyond consumer GPS and does not apply to GeoLife or similar real-world
  data.
- Recall@10 for Fréchet-distance search, once the fitter is honestly error-controlled, is nearly
  identical between the spline and DP (step1: 0.972 vs. 0.996) -- the step0 gap was a measurement
  artifact, not a representation difference.

**Not yet proved, as of step3 (left open as hypothesis H1):** whether a spline with *free/
optimal* knot placement, driven by the true curve's own geometry rather than by a noisy trial
fit's residuals or a fixed uniform spacing, could compress exact data better than DP+SED. Step3's
own oracle tested only uniform knots (a limitation caught after the run and documented as a
mandatory correction, `docs/history.md`) -- this specific, narrowly-scoped question is the one
step8 was built to close (section 10). A second, unrelated question (H2: are a spline's
analytical derivatives valuable for search by shape/kinematics, independent of compression) was
never tested at all, pending the industry interviews Stage C recommended and that were never
conducted (section 12).