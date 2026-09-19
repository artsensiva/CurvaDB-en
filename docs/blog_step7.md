# Certificates instead of trust: exact trajectory search on compressed data

This is a follow-up to the post about [how a cubic spline lost to the Douglas-Peucker
polyline](https://github.com/artsensiva/CurvaDB-en). That earlier result was negative: a smooth
curve gives no advantage for storing GPS tracks. But that project left a side effect that turned
out more interesting than the hypothesis itself: **the verification discipline** — an oracle,
criteria fixed before the run, error measured against ground truth rather than against your own
prior observations.

That raised a question: what if the same rigor were applied not to comparing representations,
but to the database's own answers?

## The problem

Trajectories are almost always stored compressed: a full track of thousands of points gets
simplified down to a hundred vertices. When you later search for "everything similar to this
track," the search runs against the simplified data. And that's where a quiet problem shows up:
**the answer is approximate but looks exact.** The user gets a list back with no way to tell
whether a close match failed to make it in.

The usual fixes are two: live with the approximation, or re-read the source data and compute the
exact metric after a coarse filter — which is slow.

We tried a third path: store, alongside every simplified track, a **proven error bound**, and use
it so the answer is guaranteed to match the exact one.

## The idea in one formula

The continuous Fréchet distance obeys the triangle inequality. If track `A` is stored as an
approximation `A'` with a proven bound `eps_A`, and query `q` comes with its own bound `eps_q`,
then

```
d(q, A) in [ d(q', A') - eps_q - eps_A,  d(q', A') + eps_q + eps_A ]
```

A range query ("everything within `r`") gets answered without reading the source data in two
cases out of three: if the interval's upper end is at most `r`, accept; if the lower end already
exceeds `r`, reject. The source track is read only when `r` falls strictly inside the interval.

Everything rests on the word **proven**. A check on a dense grid is not a proof: the curve can do
anything between sample points. For polylines the bound is exact (the Alt-Godau algorithm for the
continuous Fréchet distance). For splines we use the convex-hull property: every piece of the
curve lies inside the convex hull of its own control points, and a capsule around a segment is
itself convex — so every control point landing inside that capsule proves the whole curve piece
does too.

## What we found

Corpus: 585 cleaned GeoLife tracks, two representations (a time-aware simplified polyline and a
spline), a thousand queries at three radii, six million query-candidate pairs.

**The answers are exact.** Zero misses and zero false positives against full brute force on the
source data, across all six million combinations. That's the headline result: compression stops
being a trade-off against correctness.

**The certificates are tight.** The median ratio of the certificate to a strict lower bound is
1.00 for polylines and 1.02 for splines — the proven bound nearly matches the true distance,
not a multiple of it.

**Speed depends on what you compare against** (polyline representation, `r = 200` m, median time
per query):

| Method | Median | Answers |
|---|---|---|
| Filter without compression (exact) | 41.7 ms | exact |
| **Certified (ours)** | **20.1 ms** | **exact** |
| Approximate, no guarantees | 5.5 ms | with errors |

Against an honest exact competitor, certificates give a **2x speedup** *and* a smaller storage
footprint. Against the approximate method, the certified interval check is **2.7x slower** — that
is the measured price of the guarantee (S6b in the full report).

**We could actually measure the price of dropping the guarantee.** Approximate search on the same
compressed data loses 0.03% to 3.03% of correct answers and returns up to 2.04% false positives.
Not much, but never zero, and you can't predict the percentage in advance for any specific query.

**The main tuning knob is the ratio of the error bound to the query radius.** The smaller the
search radius, the larger the fraction of candidates that fall into the ambiguous zone. To
resolve 93.3% of candidates at a 50 m radius without reading source data, instead of 18.2% at a
much coarser simplification tolerance, tracks need to be stored **5.1x** more finely. That curve
is the actual product decision: search precision trades against storage, and the trade is
measurable.

## Three near-misses that almost became the conclusions

The most useful part of the project turned out, again, not to be the results but how close they
came to diverging from reality.

**Rounding the "safe" way.** A free-space feasibility check rounded a slightly negative
discriminant to zero — "safer that way." In practice this made the check *more* permissive, so
the reported distance could come out **too small** — exactly backwards from what an upper bound
needs. At coordinates in the hundreds of kilometers, the resulting error grew to fractions of a
millimeter, above the required precision.

**A mixed-up parameter domain.** A dense-error check evaluated a spline somewhere it wasn't even
defined: one part of the code normalized time to `[0, 1]`, another worked in real seconds. Result:
93.8% of fits declared bad. After the fix: about 18% (the invalid-fit rate on a 60-track
verification sample, not to be confused with the unrelated 18.2% figure for candidates resolved
without reading source data at `tol = 20` m in the tuning-knob section above). A conclusion built
on the first number would have been entirely false.

**Fake time.** A simplification function that is time-aware by design was handed an array of
zeros instead of real timestamps. It didn't crash — it silently returned a plausible-looking but
wrong result: storage size barely depended on the simplification tolerance. That made the
tolerance/size trade-off look nearly free, when it's actually a real 5.1x cost.

What all three have in common is **silence**. The code didn't crash, the numbers looked
plausible, the tests passed. So the fix was the same each time: don't just patch the call — make
the function **refuse to run** when its own preconditions are violated. Check the parameter's
domain, check that time is monotonic. A whole class of bug becomes an exception instead of
waiting for the next careless caller.

## A negative result: my own design mistake

The spec described the primary spline-certification mechanism as an explicit monotone
correspondence with the original polyline, with linearization filed as the fallback. In practice
the "fallback" turned out better on both counts at once: it succeeds on 100% of tracks against
16.4%, and its bound is on average 1.68x tighter.

The cause is structural. The correspondence is built along a fixed segment direction, and a piece
of the curve whose derivative's projection doesn't change sign the right way is non-monotone **at
any level of subdivision**. At roughly 242 pieces per track, even a 1.9% per-piece failure rate
makes some failure nearly certain somewhere along the track.

The criterion wasn't rewritten to fit the result: it's recorded as failed, and the architecture
changed instead. That's probably the single habit most worth carrying forward from this project.

And one more confirmation of the earlier post's finding: splines again lost to polylines on every
axis measured — certificate tightness, interval width, refine rate, error rate, and latency.

## Bottom line

Exact Fréchet-distance search on compressed data works: it's twice as fast as the honest exact
approach, takes less space, and loses no answers. The price of dropping the guarantee is measured
and comes to up to about 3% lost answers; the price of the guarantee itself is a 2.7x slowdown
relative to the approximate method, plus a tunable storage cost.

Whether that's worth it comes down to one question only the industry can answer: **what does
missing a similar case cost you?** If you're investigating a drone incident or searching an
archive for a matching prior operation, the cost is high. If you're building a fleet summary
report, probably not.

So the next step isn't code, it's conversations. All the code, 20 recorded architecture
decisions (including the reversed ones), and the full report for every milestone are public:
[github.com/artsensiva/CurvaDB-en](https://github.com/artsensiva/CurvaDB-en).
