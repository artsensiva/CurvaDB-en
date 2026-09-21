---
title: "Five results: I spent time proving splines don't help trajectory databases"
published: false
description: "Storing GPS tracks as smooth curves instead of polylines sounds convincing. I tested it five ways and it never held up. The side result was better: exact similarity search over compressed trajectories, twice as fast as the exact baseline."
tags: algorithms, datascience, database, opensource
canonical_url:
cover_image:
---

If you work with GPS tracks, this idea shows up sooner or later: why store a trajectory as a polyline of thousands of points when a vehicle moves smoothly? Accelerate, turn, brake — that's a smooth curve. A cubic B-spline describes it naturally; a polyline only approximates it with straight segments and kinks.

The idea then develops on its own. A spline should need fewer parameters for the same accuracy. Derivatives — velocity, acceleration, curvature — come analytically. Similarity search under the Fréchet distance should work no worse.

I tested this five independent ways. It never held up. But along the road a different result appeared, and that one was probably worth the whole exercise.

All code, per-milestone reports, and the decision history are public:
[github.com/artsensiva/CurvaDB-en](https://github.com/artsensiva/CurvaDB-en),
DOI [10.5281/zenodo.22850469](https://doi.org/10.5281/zenodo.22850469).

## Result one: a methodology bug that looked like a conclusion

The first benchmark compared a Douglas-Peucker (DP) simplified polyline against a spline on 200 GeoLife tracks at a 10-metre tolerance.

| Metric                       | DP polyline | Spline |
| ---------------------------- | ----------- | ------ |
| Bytes per track              | 833         | 3052   |
| Recall@10 for Fréchet search | 0.997       | 0.707  |

A recall of 0.707 means nearly a third of the "similar" tracks returned aren't the true nearest neighbours. Tempting conclusion: splines are bad for search. But the first question should have been whether the fit actually delivered the accuracy it promised.

It didn't. The fitting routine checked "error at most 10 m" **only at the track's own sample points**. Between samples the curve was on its own. A dense check showed that for 82% of tracks the spline drifted past the tolerance — about 110 m for the median track, kilometres for the worst.

The cause wasn't what I expected. Not GPS outliers, but **gaps in recording**: the error
correlates with the longest gap between samples at r = 0.69, and with spatial jumps at only 0.22. A time-parametrised cubic spline invents a route across a minute-long gap and draws loops.

The DP polyline holds its tolerance along the entire line by construction. I had been comparing a correct algorithm against a broken one.

**Lesson:** verify a method's guarantee where the method is used, not where it's convenient to check.

## Result two: an honest methodology, and still a loss

Tracks were split at gaps over 30 s and at physically impossible speeds: 200 files became 585 separate trips. Fitting started checking error on a dense grid, adding knots where the tolerance was violated. All 585 tracks passed.

The spline's recall rose from 0.707 to 0.972. DP's is 0.996, and the confidence intervals don't overlap.

Compactness went worse. Picking the right opponent matters here: our spline holds accuracy at every point in time, so the fair comparison isn't plain DP but a time-aware variant (DP+SED, synchronized Euclidean distance).

| Tolerance, m | DP     | DP+SED | Spline |
| ------------ | ------ | ------ | ------ |
| 2            | 1209 B | 1998 B | 2831 B |
| 10           | 366 B  | 691 B  | 1346 B |
| 50           | 123 B  | 224 B  | 765 B  |

The spline lost to both at every tolerance. But the gap with DP+SED shrank as the tolerance tightened: 3.4x at 50 m down to 1.4x at 2 m. That's consistent with approximation theory — on a smooth curve a polyline's parameter count grows faster than a cubic spline's. So the place to look was smooth, high-precision data.

## Result three: a spline pretending to be a polyline

I built synthetic "roads" from straights, arcs and clothoids (curvature continuous, like real roads) and swept a grid of noise levels and tolerances, measuring every method's error against the ground-truth curve.

The result looked like a finding: with no noise, the spline was 22% more compact than DP+SED at a 1 m tolerance.

But something was odd. The fraction of tracks where a method could hit the tolerance at all **matched exactly** between spline and polyline in every cell: 3/30, 25/30, 24/30, 26/30. And even with zero noise a 20 cm tolerance was unreachable almost everywhere. For the polyline that's expected — the chord between neighbouring samples deviates from an arc by L²/8R, which at 20 m/s on a 30 m radius is 1.7 m. But a smooth cubic spline through exact samples should track that arc to centimetres.

The answer was my own fix from the previous step. To stop the spline from flying off between samples, I had forced it to stay near the straight segments joining them. Reasonable for noisy GPS; on smooth synthetic data it meant the spline was obliged to reproduce the polyline's error and couldn't smooth anything.

**Lesson:** a fix that's correct in one experiment becomes a constraint in the next. Matching numbers across methods that should behave differently are almost always a signal.

## Result four: the decisive experiment, and an oracle

At that point it was clear I could keep finding skews and moving goalposts forever. So the final experiment did three things.

**An honest fitter**: least squares directly on x(t) and y(t), no tethering to the polyline.

**An oracle**: the same spline fitted directly to the true curve, with no noise and no sparsity. The oracle answers "is this a good representation at all," separately from "can we build it from the data." Conflating those two questions is what had muddled the earlier steps.

**Success criteria fixed before the run**: at least 30% smaller than DP+SED, reconstruction from sparse samples, robustness to noise.

The untethered spline did show its worth: at a 1 s step, no noise, 0.5 m tolerance, it hit the tolerance on 14 of 15 tracks versus 5 for DP+SED. But all three criteria failed. After zlib compression its advantage never exceeded 7%.

The decisive table was the oracle's:

| Tolerance, m | Oracle (spline on the true curve) | DP+SED on noisy samples |
| ------------ | --------------------------------- | ----------------------- |
| 2            | 457 B                             | 467 B                   |
| 10           | 292 B                             | 278 B                   |

A spline that knows the ideal curve takes about as much space as a polyline built from real, noisy data.

## Result five: free knots don't save it

One loophole remained: the oracle placed knots uniformly. Adaptive placement might deliver that missing 30%.

So I implemented certified knot removal and built a free-knot oracle on the ideal, noise-free curve. The criterion was recorded in advance: at least 20% smaller than DP+SED in at least one valid cell.

Not met — neither formally (no cell qualified as valid) nor substantively: the best ratio was 0.834x at a 2 m tolerance, short of 0.80x.

Something more interesting turned up on the way. The spline's certificates passed 100% of the time, while its reachability under a time-synchronized error criterion fell from 86.7% to 66.7% as the tolerance grew. The reason: **a Fréchet certificate is not a synchrony guarantee.** Fréchet allows sliding along the curve as long as you stay close in space. If you need time synchrony you need a different metric (SED or L²). That generalizes well beyond this problem.

## What did work: certificates

Now the positive result. It isn't about curves — it's about guarantees.

Trajectories are almost always stored compressed, and search runs against the simplified data. That creates a quiet problem: **the answer is approximate but looks exact.** The user gets a list back with no way to know a close match didn't make it in.

The idea in one formula. The Fréchet distance obeys the triangle inequality. If a simplified track `A'` is stored with a **proven** bound `eps_A`, and a query arrives with its own `eps_q`, then

```
d(q, A) in [ d(q', A') - eps_q - eps_A,  d(q', A') + eps_q + eps_A ]
```

A range query ("everything within `r`") is answered without reading the source in two cases out of three. The original track is read only when `r` falls inside the interval.

Everything rests on **proven**. A dense-grid check is not a proof. For polylines the bound is exact (Alt–Godau). For splines it uses the convex-hull property: a curve piece lies inside the convex hull of its control points, and a capsule around a segment is convex — so all control points landing inside that capsule proves the whole piece does.

**Results** over 585 tracks, a thousand queries, six million query-candidate pairs:

- **Zero misses and zero false positives** against full brute force on the source data.
- Median ratio of certificate to a strict lower bound: 1.00 for polylines, 1.02 for splines.
- Speed (polyline representation, r = 200 m, median per query):

| Method                             | Median      | Answers     |
| ---------------------------------- | ----------- | ----------- |
| Filter without compression (exact) | 41.7 ms     | exact       |
| **Certified**                      | **20.1 ms** | **exact**   |
| Approximate, no guarantees         | 5.5 ms      | with errors |

Against an honest exact competitor: twice as fast and smaller on disk. Against the approximate method: 3.7x slower — the measured price of the guarantee. (Per query-candidate pair the ratio is milder, 2.7x; the project's own criterion was written in those terms and was not met.)

**The price of dropping the guarantee, measured:** approximate search loses 0.03% to 3.03% of correct answers and returns up to 2.04% false positives. Small, never zero, and unpredictable per query.

**The main tuning knob** is the ratio of the error bound to the query radius. Resolving 93.3% of candidates at a 50 m radius without reading originals, instead of 18.2% at a coarse tolerance, costs **5.1x** more storage. That curve is the actual product decision.

## Three measurement bugs that almost became conclusions

**Rounding the "safe" way.** A feasibility check rounded a slightly negative discriminant to zero — "safer." In practice it made the check *more* permissive, so the distance could come out **too small** — backwards from what an upper bound needs.

**A mixed-up parameter domain.** A dense-error check evaluated a spline where it wasn't defined: one part of the code normalized time to `[0, 1]`, another used real seconds. Result: 93.8% of fits declared bad. After the fix, on the same full corpus and the same fitter: 9.9% — and with the fitter ultimately chosen, none at all. A conclusion built on the first number would have been entirely false.

**Fake time.** A time-aware simplification routine was handed an array of zeros. It didn't crash — it silently returned a plausible but wrong result: storage size barely depended on the tolerance. That made the tolerance/size trade-off look free, when it's really 5.1x.

What they share is **silence**. Nothing crashed, the numbers looked plausible, the tests passed. So each fix was the same: don't just patch the call — make the function **refuse to run** when its preconditions are violated. A whole bug class becomes an exception instead of waiting for the next careless caller.

## How this was built

All the research code was written by the Claude Code agent from short written specs, while analysing results and hunting methodology bugs happened separately, in conversation.

The observation I find more valuable than the technical conclusions: the agent implements and runs experiments well and reports negative results honestly, but it **follows the spec literally**. Case in point — the polyline tethering wasn't invented by the agent; it came from the task spec, where the requirement was written as a fix for the previous mistake. The agent implemented it exactly, and the critic who wrote that requirement spotted the side effect only one step later.

Practical takeaway: reviewing the task specification, including your own fixes, matters as much as reviewing code. Keeping specs as files in the repo is what makes that review possible.

A few other rules that paid off:

1. Write success criteria **before** the run, into a file you're not allowed to edit afterwards.
2. Measure error against ground truth, not against your own observations.
3. Pick a competitor solving the same problem under the same conditions.
4. Never delete a superseded decision — mark it superseded, with the reason.
5. Be suspicious of coincidences: identical numbers across methods, suspiciously round averages.

## Bottom line

Five results: four negative, one positive.

For storing and searching consumer-GPS tracks, a cubic spline is no better than a simplified polyline and noticeably worse on size. On smooth, high-precision data it catches up but doesn't win. Free knots and even the ideal noise-free curve don't change that.

But exact Fréchet search over compressed data does work: twice as fast as the honest exact approach, smaller on disk, no lost answers, with the price of dropping the guarantee measured.

Whether that's worth it comes down to one question only the industry can answer: **what does missing a similar case cost you?** Investigating a drone incident or searching an archive for a matching prior operation — high. Building a fleet summary report — probably not.

So the next step isn't code, it's conversations. If you work with trajectory archives (autonomous driving, robotics, maritime or aviation analytics) and the pain of missed similar cases sounds familiar, get in touch — I'm after your experience, not a sale.

Code, the full technical report with every table, per-milestone results, and 23 recorded
architecture decisions (including the reversed ones):
[github.com/artsensiva/CurvaDB-en](https://github.com/artsensiva/CurvaDB-en),
DOI [10.5281/zenodo.22850469](https://doi.org/10.5281/zenodo.22850469).
