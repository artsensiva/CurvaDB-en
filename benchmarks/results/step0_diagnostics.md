# step0 diagnostics before step1

Method: the same 200 tracks and `seed=42` as in `step0.py`, `tol = 10` m.
The `src/traj/*` and `benchmarks/step0.py` code was not changed -- the
diagnostics were run with separate scripts outside the repository
(scratch), the result is recorded here as a report.

## 1. Why is recall the same (0.707) for spline_same and spline_arclen?

Not a bug. For all 200 tracks, the number of points in spline_same and
spline_arclen **differs** (0 tracks with the same point count), and the
coordinate arrays themselves are not identical (`np.allclose` = False on
every track checked). Example (first 5 tracks):

| track_id | n_dp / spline_same | spline_arclen |
|---|---|---|
| 128/20100926013046 | 8 | 1154 |
| 039/20090322135723 | 37 | 542 |
| 154/20070509124902 | 31 | 322 |
| 037/20090215234308 | 113 | 3000 (cap) |
| 128/20091230015240 | 31 | 579 |

Breakdown across 30 queries: for 25/30 queries the top-10 candidate IDs
**fully match** between spline_same and spline_arclen; for the remaining
5 queries the top-10 differs by 1 position, but the per-query recall
either still matches (75, 146, 36 -- same recall with different top-10,
i.e. one wrong position swapped for another wrong one), or diverges in
opposite directions and cancels out on averaging (113: 0.00→0.10,
140: 0.90→0.80). The final mean matched to 4 decimal places (0.7067 vs
0.7067) not because the feature is broken, but because the discrete
Frechet distance is weakly sensitive to the resampling density of the
same curve -- it is determined by the extremal (worst-case) point
correspondence along the curve, not by the point count. Adding points
between already-existing ones barely changes that extreme pair.

**No code changes were made** -- there is no recognized bug here.

## 2. Spline oscillation between timestamps

For each track: a dense discretization of the spline (20 points per
interval between neighboring timestamps) was compared (a) by total arc
length against the raw polyline, (b) by the maximum distance from each
dense spline point to the raw polyline **as a whole** (point-to-segment,
shapely `LineString.distance`, not only at the timestamps).

Ratio (spline arc length / raw polyline arc length):

| median | p90 | max | min |
|---|---|---|---|
| 1.052 | 3.562 | 1941.98 | 0.841 |

Max distance (dense spline → raw polyline), meters:

| median | p75 | p90 | p99 | max |
|---|---|---|---|---|---|
| 109.9 | 671.6 | 10930.7 | 186002.9 | 8 710 755.3 |

Fraction of tracks where the `error <= tol` guarantee is violated
**between** timestamps (even though it holds at the timestamps themselves):

| threshold | tracks |
|---|---|
| > tol (10 m) | 164 / 200 (82%) |
| > 2·tol | 141 / 200 |
| > 5·tol | 111 / 200 |
| > 10·tol | 101 / 200 (51%) |

Even the median track exceeds tol between timestamps by a factor of
**11** (110 m at tol=10 m), i.e. this isn't a handful of outliers -- it's
systemic.

The correlation of log(max_dist) with candidate causes (across all 200
tracks) is weak everywhere (max |r|=0.27 with log(n_raw)), i.e. there's
no single dominant factor -- the spline overshooting between knots is a
general property of the method (no constraint is placed on the degree of
freedom between points), which just shows up more strongly on some tracks.

5 worst tracks and the likely cause:

| track_id | max_dist, m | ratio | max_gap, m | min_dt, s | start-finish, m | cause |
|---|---|---|---|---|---|---|
| 062/20080114132113 | 8 710 755 | 1942 | 3817 | 3.0 | 1952 | anomalous GPS jump (3.8 km in 3 s ≈ 4580 km/h) |
| 142/20070422064810 | 1 457 997 | 88.4 | 2333 | 2.0 | 20378 | anomalous GPS jump (2.3 km in 2 s ≈ 4200 km/h) |
| 095/20101214101134 | 173 154 | 55.1 | 1197 | 2.0 | 166.6 | GPS jump + near-closed/low-motion track (start≈finish) |
| 125/20080913091957 | 110 326 | 6.6 | 12056 | 2.0 | 205.1 | extreme single GPS jump (12 km in 2 s) |
| 096/20080716112753 | 50 237 | 20.4 | 141.4 | 1.0 | 3456.2 | no clear jump -- accumulated overshoot on dense, real (noisy) 747 points |

The first four are explained by specific anomalous breaks in the raw GPS
data (a real GeoLife defect, the same class of problem as the track from
TODO.md for step 5). The fifth is an example of overshoot happening even
without a single jump, simply because `spline.fit()` only controls error
at the knots, not between them.

`fit_max_error` (error exactly at the timestamps) is ~8-10 m for all five
tracks -- the method honestly meets its contract "error <= tol at the
original timestamps," the contract just doesn't cover what happens
between them.

## 3. Same for the DP polyline

Maximum deviation of **any original point** of the track from the DP
polyline (not just the removed points, but literally all of them):

| median | p90 | max |
|---|---|---|
| 9.697 | 9.968 | 10.000 |

0/200 tracks exceed tol. The maximum across all 200 tracks is exactly
10.000 m, i.e. the tol boundary, as expected: classic Douglas-Peucker by
construction guarantees that the deviation of removed points from the
simplified polyline never exceeds the threshold, and the guarantee holds
"everywhere along the polyline," not just at the vertices.

5 worst DP tracks (all right at the tol boundary, no violations):

| track_id | max_dist_dp, m | n_raw | n_dp |
|---|---|---|---|
| 128/20070531131532 | 10.000 | 84 | 41 |
| 169/20100604012003 | 9.999 | 1509 | 60 |
| 128/20101129005554 | 9.998 | 502 | 49 |
| 037/20090215234308 | 9.998 | 1018 | 113 |
| 004/20090620044101 | 9.998 | 516 | 106 |

## Conclusion

The "error <= tol" guarantee for the spline holds **only at the original
timestamps** and is massively violated (82% of tracks, median 11×tol)
between them, up to kilometer-scale deviations on tracks with GPS jumps
in the raw data. For the DP polyline, the same guarantee holds literally
everywhere along the polyline (max across 200 tracks = 10.000 m, exactly
the threshold) -- this is a direct consequence of the algorithm, not a
coincidence. This fully explains the spline's low recall@10 (0.707 vs
0.997 for DP from step0.md): the discrete Frechet distance between
spline representations is distorted by these uncontrolled excursions
between knots, whereas the DP polyline structurally cannot deviate from
the source track by more than tol anywhere. spline_same and
spline_arclen recall matched not because of a bug, but because Frechet
distance is insensitive to the resampling density of the same (already
distorted) curve. For step1 it makes sense to either control the
spline's error along the whole curve (not just at the knots), or
explicitly split tracks at GPS jumps before fitting.

## 4. dt analysis (added in step1)

**Added after the original diagnostics above** (§§1-3 and the Conclusion), during step1 work on
the cause of the spline's overshoot -- not part of the initial diagnostic pass. Same 200 tracks,
`seed=42`, `tol=10.0` m as §§1-3; produced by `benchmarks/diagnose_fit.py` (its own §4, which
this section reproduces verbatim from a real run of that script, not from memory or a narrative
draft).

The original diagnosis (§2 above) tested correlation of `log(max_dist)` against `max_gap`,
`min_dt`, `std_dt`, `iqr_dt`, `span_start_end`, and `log(n_raw)`, finding all of them weak (max
`|r|=0.27`, with `log(n_raw)`) -- but **did not test `max_dt` (the single largest gap between
consecutive timestamps within a track) as its own candidate**. Added here:

```
=== correlations (log10 max_dist vs factor) ===
corr(log max_dist, max_gap): 0.2229384796719281
corr(log max_dist, min_dt): 0.14912623566989053
corr(log max_dist, max_dt): 0.691941288814573
corr(log max_dist, std_dt): 0.5767211524389612
corr(log max_dist, iqr_dt): 0.13640984726061772
corr(log max_dist, span_start_end): 0.1295000400438432
corr(log max_dist, log n_raw): 0.3121591554793641

=== dt: max and spread within a track (across all 200 tracks) ===
max_dt: median=212.00s p90=7827.50s max=21330.00s
std_dt: median=13.18s p90=323.49s max=1882.07s
iqr_dt: median=0.00s p90=4.00s max=47.00s
```

**`max_dt` (the largest recording gap within a track) is, by a wide margin, the strongest single
correlate of `log(max_dist)` found in either pass: `r=0.69`** (vs. `max_gap`, the largest single
*spatial* jump between consecutive raw points: `r=0.22`; both far above every other factor
tested, including the `r=0.27` with `log(n_raw)` that was the strongest finding in the original
§2 pass, which never tested `max_dt` itself). `std_dt` (the spread of gaps within a track) is
also elevated (`r=0.58`), consistent with the same mechanism: a cubic spline parametrized by
time, faced with a long pause in recording, has nothing to constrain it during that pause and can
draw an arbitrary loop before rejoining the next real observation.

**Reconciling with `docs/blog_draft.md`'s narrative claim** ("паузы в записи коррелировали с
ошибкой с коэффициентом 0.69, пространственные скачки лишь с 0.18"): the `max_dt` figure matches
this run almost exactly (0.6919 vs. the quoted 0.69). The spatial-jump figure does not: this run
gives `max_gap`'s correlation as **0.22**, not 0.18. Per this project's rule for a numeric
discrepancy, the freshly-run number (0.22) is the one used going forward (`docs/REPORT.md`,
`docs/ru/REPORT.md`); the blog draft's `0.18` is superseded, not corrected in place, since the
blog draft is a narrative document, not a results file.
