# Splines vs. polylines: three benchmark mistakes and an honest loss

Anyone who works with GPS tracks eventually has the same idea: instead of
storing a trajectory as a polyline of hundreds of points, why not store
it as a smooth curve — a cubic B-spline? The intuition is appealing. A
vehicle moves smoothly: it accelerates, enters a turn, brakes. A
polyline describes this as a series of straight segments with kinks; a
spline describes it naturally. It seems like a spline should need fewer
parameters for the same accuracy, that derivatives (velocity,
acceleration, curvature) come for free, and that similarity search under
the Fréchet distance shouldn't be any worse.

We tested this against the classic "Douglas-Peucker (DP) simplification
plus discrete Fréchet distance" scheme, on real GeoLife tracks and on
synthetic data. Spoiler: the idea didn't survive the test. But the path
to that conclusion turned out more interesting than the conclusion
itself. Along the way we got three results that looked convincing and
were actually methodology artifacts. This post is about how to catch
those.

All the code, results, and full history are public:
[github.com/artsensiva/CurvaDB-en](https://github.com/artsensiva/CurvaDB-en).

## Where it started

The project began as something else entirely: semantic document search,
where each document is represented as a "curve" — a Hilbert-curve index
over embeddings, splines, and functional PCA. An outside review of the
idea was short and unflattering. First, it bundles four unrelated math
problems under one word, "curve." Second, the order of an embedding's
coordinates is arbitrary — learned by gradient descent. A curve drawn
through 768 coordinates carries no meaning: there's no continuous
quantity to trace it along, so there's neither compression nor
meaningful derivatives.

The advice was simple: pick one type of curve, one domain where the
curve's parameter is physically meaningful, one metric, and an honest
baseline before any optimization. For GPS tracks, that parameter is
time, which is why we picked them.

Even before writing a line of code, it turned out the niche wasn't
empty: PostGIS has a Fréchet distance function, there's a MobilityDB
extension for trajectories, and telematics widely uses map-matching. And
the flashy "80 KB raw track vs. 2 KB spline" comparison was unfair — the
real competitor is the simplified polyline, not the raw one.

## Mistake one: a guarantee that wasn't checked everywhere

The first benchmark compared the DP polyline and a spline
(`scipy.interpolate.splprep`) on 200 GeoLife tracks at a 10-meter
tolerance.

| Metric | DP polyline | Spline |
|---|---|---|
| Bytes per track | 833 | 3052 |
| Recall@10 for Fréchet search | 0.997 | 0.707 |

A recall of 0.707 means that almost a third of the "similar" tracks
returned aren't actually the true nearest neighbors. It's tempting to
conclude the spline is just bad for search. But the question to ask
first was: does our fit actually deliver the accuracy we promised?

It didn't. The fitting function only checked "error no more than 10
meters" at the track's original points. Between points, the curve was
left to its own devices. A dense check showed that for 82% of tracks the
spline drifted past the tolerance, by about 110 meters for the median
track and by kilometers for the worst ones. The main cause turned out
not to be GPS outliers, as we first assumed, but gaps in recording: the
maximum interval between points had a correlation of 0.69 with the
error, versus only 0.18 for spatial jumps. A cubic spline parametrized by
time, across a minute-long gap, "invents" a route and draws loops.

The DP polyline, by contrast, holds its tolerance along the entire line
by construction. So we were comparing a correct algorithm against a
broken one. The same benchmark had a second, smaller unfairness too: the
polyline was stored without timestamps, while the spline encodes time
implicitly.

**Lesson:** a method's accuracy guarantee needs to be checked where the
method is actually used, not wherever it's convenient to check it.

## An honest methodology, and still a loss

We fixed two things. Tracks now get split at gaps longer than 30 seconds
and at unrealistic speeds above 70 m/s: the 200 original files yielded
585 usable trips. And fitting now checks the error on a dense
grid between points, adding knots wherever the tolerance is violated.
All 585 tracks passed the check.

The spline's recall rose from 0.707 to 0.972. DP's is 0.996, and the
95% confidence intervals (0.956–0.987 and 0.991–0.999) don't overlap,
so the difference is small but real.

Compactness turned out worse. Here it matters to pick the right
opponent. Our spline holds accuracy at every point in time, so the fair
comparison isn't plain DP but a variant that also accounts for time
(DP by synchronized Euclidean distance, DP+SED).

| Tolerance, m | DP | DP+SED | Spline |
|---|---|---|---|
| 2 | 1209 B | 1998 B | 2831 B |
| 10 | 366 B | 691 B | 1346 B |
| 50 | 123 B | 224 B | 765 B |

The spline lost to both at every tolerance. But the table had an
encouraging trend too: the gap with DP+SED shrank as the tolerance got
tighter, from 3.4x at 50 meters to 1.4x at 2 meters. That's consistent
with approximation theory: on a smooth curve, a polyline's parameter
count grows faster than a cubic spline's. On noisy consumer GPS with
5–15 m accuracy, though, the track simply isn't smooth at the scale of
the tolerance.

Along the way we also checked kinematics on synthetic data with known
velocity and acceleration: the spline estimated velocity better than
interpolation over the DP polyline's vertices, and a Kalman smoother beat
both. No method could reliably detect sharp maneuvers from coordinates
alone; details are in the repository.

## Mistake two: a spline tethered to the polyline

The trend in the previous table pointed to where to look: smooth,
precise data. We built synthetic "roads" out of straights, arcs, and
clothoids (curvature continuous, like a real road), and swept a grid of
noise levels and tolerances. Every method's error was measured against
the ground-truth curve, and each method's internal parameter was tuned
separately.

The result looked like a real finding: with no noise, the spline was
22% more compact than DP+SED at a 1 m tolerance and 15% more compact at
5 m. It seemed like we'd found a niche for high-precision positioning
(RTK, lidar).

But there was something odd in the tables. The fraction of tracks where
a method could even hit the tolerance at all matched exactly between the
spline and the polyline, in every cell: 3 of 30, 25 of 30, 24 of 30, 26
of 30. And even with zero noise, a 20-centimeter tolerance was
unreachable almost everywhere. For the polyline that's explainable: the
chord between neighboring points deviates from the arc by L²/8R, and at
20 m/s on a 30 m-radius turn that's 1.7 meters. But a smooth cubic
spline through exact samples should describe the arc to centimeter
accuracy.

The answer was in our own fix from the previous step. To keep the
spline from flying off between points, we'd forced it to stay near the
straight segments between neighboring samples. That was reasonable for
noisy GPS, but on smooth synthetic data it meant the spline was forced
to repeat the polyline's error and couldn't smooth out noise. We weren't
comparing a spline against a polyline — we were comparing a spline
pretending to be a polyline.

**Lesson:** a fix that's correct in one experiment can become a
limitation in the next. Matching numbers between methods that should
behave differently are almost always a signal.

## The decisive experiment

By this point we knew we could keep finding new skews and keep moving
the goalposts forever. So for the final experiment we did three things.

**We wrote an honest fitter.** Least squares directly on x(t) and y(t)
(`make_lsq_spline`), with no tethering to the polyline, with both
uniform and adaptive knot placement. We extended the synthetic data with
variable speed — accelerations and stops — and varied the observation
step: 1, 5, and 15 seconds.

**We added an oracle:** the same spline, fit directly to the true
curve, with no noise and no sparsity. The oracle answers "is this a good
representation at all," separately from "can we actually build it from
the data." Conflating those two questions is exactly what tripped up
the earlier steps.

**We fixed success criteria before running anything:**

- K1, compression: after quantization and zlib (applied the same way to
  every method), the spline is at least 30% more compact than DP+SED in
  at least two low-noise cells, with both methods reaching at least 80%
  of tracks;
- K2, sparse observations: at a step of 5 seconds or more, there are
  cells where the spline reaches 80% of tracks while DP+SED reaches at
  most 20%;
- K3, heavy noise: at 5 m of noise there's a tolerance with that same
  80%-vs-20% split.

The freed spline really did show its worth. At a 1 s step, no noise, and
a 0.5 m tolerance, it hit the tolerance on 14 of 15 tracks, versus only 5
for DP+SED. The tethering hypothesis was confirmed.

But all three criteria failed. After zlib compression, the spline's
advantage never exceeded 7% (a ratio of 0.93-0.99). At a 5-second step,
DP+SED is almost always unreachable, but the spline only hit the
tolerance on at most 8 of 15 tracks. At 5 m of noise, at most 7 of 15.

The decisive piece was the oracle table:

| Tolerance, m | Oracle (spline on the true curve) | DP+SED on 1-second samples (no noise) |
|---|---|---|
| 2 | 457 B | 467 B |
| 10 | 292 B | 278 B |

A spline that knows the ideal curve takes up about as much space as a
polyline built from ordinary 1-second samples. No amount of fitter
improvement or regularization can beat the oracle under these conditions
(more on its key limitation below).

## Mistake three, which we almost missed

The first draft of our conclusions claimed compression was closed "all
the way to the spline's theoretical limit." A final code review turned
up that the oracle only placed its knots uniformly. And in an earlier
experiment, adaptive placement (FITPACK) had used 30% fewer control
points than uniform placement. So the oracle shows a ceiling for uniform
knots, not for splines in general — and optimal knot placement could
still bring the spline closer to criterion K1.

The honest phrasing is therefore more modest: under the conditions we
tested, there's no compression advantage; optimal knots remain
untested. It doesn't change the practical takeaway, though. A track
already takes up a few hundred bytes, and even a 30% savings isn't a
reason to change your storage infrastructure.

The decisive experiment had other limitations worth naming. At a
15-second step, a turn can fit entirely between two samples, and no
method can reconstruct what isn't in the data — that's an information
limit, not a property of the spline. A 10-meter tolerance at 5 meters of
noise is too tight to evaluate by max error. And DP+SED is really just a
compression method — for reconstructing a trajectory from sparse
samples, the honest competitors would be a Kalman smoother or Doppler-
velocity-aided interpolation.

## What we took away

**About splines.** For storing and searching consumer-GPS tracks, a
cubic spline is no better than a simplified polyline, and noticeably
worse on compactness. On smooth, precise data it catches up to the
polyline under the conditions we tested, but doesn't beat it. A spline's
real strengths — smoothness and analytical derivatives — could still
matter in robotics, drones, or high-precision agricultural equipment.
That's an open hypothesis, and it needs to be tested by talking to the
industry, not by writing more code.

**About methodology**, which is arguably the main point:

1. Check a method's guarantees where it's actually used, not wherever
   is convenient.
2. Pick a competitor solving the same problem under the same
   conditions: DP+SED, not plain DP; the simplified polyline, not the
   raw one.
3. Measure error against the ground truth, not against your own
   observations.
4. Separate a representation's capacity from the quality of building it,
   using an oracle — and check that the oracle itself uses the method's
   full potential.
5. Fix success criteria before running the experiment. Without that,
   every failure produces one more reason why "the method actually
   works."
6. Be suspicious of coincidences. Matching numbers across different
   methods, and suspiciously round averages, are worth five minutes of
   double-checking.

**About process.** All the research code was written by the Claude Code
agent in VS Code, working from short written tasks, while analyzing
results and hunting for methodological mistakes happened separately, in
conversation. The agent implements specs literally, and it honestly
records negative results. The polyline tethering wasn't something the
agent invented on its own — it came straight from the task spec, where
"keep the error within the segments between samples" was written in as a
fix for the previous mistake. The critic who wrote that requirement
noticed its side effect only one step later. The lesson: reviewing the
task spec, including your own fixes, matters just as much as reviewing
the code. Keeping specs as files in the repo is what makes that review
possible.

A negative result reached in a few evenings costs far less than half a
year spent building a product on an untested hypothesis — which is
exactly why it was worth getting. The code, every table, and the full
history of each step are in the
[repository](https://github.com/artsensiva/CurvaDB-en).
