# Step 1: data cleaning, honest spline fitting, A/B

## 1. Track cleaning

Parameters: a time gap > 30s OR speed > 70 m/s -- split; single-point segments -- drop; Beijing bbox (39.4, 41.6, 115.7, 117.5) (lat_min, lat_max, lon_min, lon_max); then refilter to 50..2000 points per segment.

Loaded raw tracks: 200, points: 199080.

| Stage | Value |
|---|---|
| extra segments from break-splitting | 3113 |
| single-point outliers removed | 1504 |
| points dropped outside Beijing bbox | 30718 |
| segments dropped by length filter (not 50..2000) | 860 |
| **total: tracks** | 585 (out of 200 original) |
| **total: points** | 141948 (out of 199080 original) |

## 2. Honest spline fitting

Check: >= 10 dense-grid points per interval between neighboring points of the CLEANED track (after item 1), error = point-to-segment distance to the straight segment between them. If s=0 (interpolation) violates tol between some pair of points, a synthetic knot is adaptively added at the location of maximum deviation (up to 8 rounds); then `s` is grown for compactness, keeping the dense error <= tol.

**ACCEPTANCE THRESHOLD (>= 99% of tracks with dense error <= tol) REACHED: 585/585 = 100.00% (time parametrization).**

| Parametrization | Pass rate | Median err, m | p99 err, m | Max err, m | mean control points | tracks with synthetic knots | bisect converged | time, s |
|---|---|---|---|---|---|---|---|---|
| time | 100.00% | 9.995 | 10.000 | 10.000 | 69.9 | 109/585 | 585/585 | 215.8 |
| chord | 100.00% | 9.935 | 10.000 | 10.000 | 59.5 | 105/585 | 585/585 | 188.3 |

Mean number of points in a cleaned track: 242.6. Both parametrizations reach the same pass rate; chord-length parametrization gives a noticeably more compact representation (59.5 vs 69.9 control points on average), but `derivatives()` (velocity/acceleration) is only implemented for time parametrization -- the nonlinear u<->t relationship for chord would require a separate inversion t(u), not needed beyond this comparison. For kinematics (item 4) and the rest of the step1 benchmarks, time parametrization is used (the default in fit()).

## 3. Recall@10 on tracks with dense error <= 2·tol

On 585/585 cleaned tracks with spline dense error <= 2·tol (30 queries from that same group, corpus -- all 585 tracks): recall@10 DP = 0.993, recall@10 fixed spline = 0.973 -- vs 0.997/0.707 in step0.md (broken fitting, no data cleaning).

## 4. Kinematics (hypothesis B): where the spline beats DP

30 synthetic trajectories (accel/brake/turn/stop, velocity continuous across boundaries, seed=42), Gaussian noise 5m on position, step 1..5s. Evaluated at 759 points (excluding the track's first/last point -- an edge effect for all methods), of which |a_true| > 3 m/s² for 74 (9.7%). Kalman: constant-acceleration + RTS, q=1 (chosen by acceleration RMSE on a subsample of 8 tracks).

| Method | Velocity error, m/s (median/RMSE) | Acceleration error, m/s² (median/RMSE) | Precision \|a\|>3 | Recall |
|---|---|---|---|---|
| fixed spline | 1.521 / 3.145 | 0.795 / 3.882 | 0.159 | 0.284 |
| DP + PCHIP over vertices | 2.316 / 4.087 | 0.919 / 248.150 | 0.205 | 0.338 |
| Kalman (CA+RTS) | 1.253 / 2.268 | 0.647 / 1.701 | 0.833 | 0.068 |

Best median velocity error: Kalman (CA+RTS). Best median acceleration error: Kalman (CA+RTS).

Note: DP + PCHIP's acceleration RMSE is noticeably above its median (max error 4991 m/s² vs 33.6 for the spline and 15.3 for Kalman) -- numerical fragility: when neighboring DP vertices end up close in time (a short segment between two DP break points), PCHIP's second derivative on that segment can blow up to thousands of m/s². Kalman shows the opposite picture: best error (RMSE and median), but noticeably lower recall for detecting |a| > 3 -- the CA model's smoothing suppresses sharp acceleration changes (turns), not just noise.

## 5. Compression (hypothesis A): where the spline beats DP on bytes

60 cleaned tracks (subsample out of 585, seed=42), float64 = position+time float64 (24 bytes/point); int32 = position in cm + time in whole seconds, int32 (12 bytes/point). DP is spatial (perpendicular distance), DP+SED is time-aware (synchronized Euclidean distance, see src/traj/simplify.py). Spline bytes include t_min/t_max -- only that lets the knots (a fraction of [0,1]) be mapped back to real time; the spline needs no separate timestamp per vertex.

| tol, m | raw, B (f64/i32) | DP, B (f64/i32) | DP+SED, B (f64/i32) | spline, B (f64/i32) | n DP / n SED / n control points |
|---|---|---|---|---|---|
| 2 | 5414 / 2707 | 1209 / 605 | 1998 / 999 | 2831 / 1415 | 50.4 / 83.2 / 116.0 |
| 5 | 5414 / 2707 | 613 / 306 | 1124 / 562 | 1956 / 978 | 25.5 / 46.8 / 79.5 |
| 10 | 5414 / 2707 | 366 / 183 | 691 / 346 | 1346 / 673 | 15.3 / 28.8 / 54.1 |
| 20 | 5414 / 2707 | 212 / 106 | 414 / 207 | 980 / 490 | 8.8 / 17.2 / 38.8 |
| 50 | 5414 / 2707 | 123 / 61 | 224 / 112 | 765 / 383 | 5.1 / 9.3 / 29.9 |

Smallest size (float64) per tol: tol=2: DP; tol=5: DP; tol=10: DP; tol=20: DP; tol=50: DP.

## 6. Search (hypothesis A): recall@10 on cleaned tracks

585 cleaned tracks (seed=42), tol=10m, 30 queries x 3 query-selection seeds (42, 43, 44) = 90 measurements per representation; recall@10 against the exact Frechet distance on raw (cleaned) tracks; 95% confidence interval -- bootstrap, 2000 resamples.

| Representation | Recall@10 (mean) | 95% CI |
|---|---|---|
| DP polyline | 0.996 | (0.991, 0.999) |
| fixed spline | 0.972 | (0.956, 0.987) |

For comparison, step0.md (broken fitting, no cleaning, 1 query seed): DP = 0.997, spline = 0.707. After cleaning (item 1) and honest fitting (item 2), the recall@10 gap between DP and the spline has almost entirely closed, but DP is still slightly ahead.

## 7. Conclusions

The cause of the spline's low recall in step0 wasn't a hypothesis but an established fact even before step1 (step0_diagnostics.md): `fit()` only held tol at the original timestamps, and between them the spline could drift by kilometers (82% of tracks, median 11×tol). Item 1 (cleaning: time/speed breaks, bbox, refilter) and item 2 (honest dense error control, the >=99% threshold reached -- 100%) removed that cause directly: the spline's recall@10 recovered from 0.707 to 0.972-0.973 (items 3, 6), nearly matching DP (0.993-0.996). From there, step1 was no longer investigating "why is recall low" (already known), but **where the fixed, honest spline can beat DP** -- along two independent axes.

**Hypothesis A (compression and search) -- a negative result.** On honest data, DP is more compact than the spline at EVERY tol tested (2..50m, item 5): e.g. at tol=10m, DP is 366 bytes/track vs the spline's 1346 bytes/track (3.7x more), because honest dense error control forces the spline to keep far more control points than DP has vertices for the same guarantee. DP's recall@10 is also slightly above the spline's on cleaned data (item 6: 0.996 vs 0.972, confidence intervals barely overlapping). DP+SED (time-aware simplification) is, as expected, even more expensive than spatial DP (it enforces a stricter criterion), but stays cheaper than the spline at every tol. Conclusion: fixing the spline removed its spurious loss on compression and Frechet search, but didn't let it beat DP -- DP remains the better choice on this axis.

**Hypothesis B (kinematics) -- a mixed result, not a clear spline win.** On synthetic data with known v/a (item 4), the spline and DP+PCHIP give similar median velocity/acceleration error, but DP+PCHIP has a heavy tail on acceleration (max ~4991 m/s² vs ~34 for the spline) -- PCHIP's numerical fragility when DP vertices land close together in time, which the spline doesn't share. However, a third reference point, Kalman (CA+RTS), beats BOTH on median and RMSE v/a error, at the cost of noticeably lower recall for detecting sharp maneuvers (|a|>3 m/s²: recall 0.068 vs 0.284 for the spline and 0.338 for DP+PCHIP) -- the CA model oversmooths turns. Conclusion: the spline has no clear advantage over DP+PCHIP on average kinematic error, but it is noticeably more reliable than DP+PCHIP in the worst case (no catastrophic acceleration outliers); Kalman is a separate variant, more accurate on average but less sensitive to maneuvers, outside the original "spline vs DP" comparison.

**Bottom line**: the honest spline is a correct and predictable, but not more compact and not more accurate for search, alternative to the DP polyline; its main practical advantage is the absence of the catastrophic acceleration outliers that DP+PCHIP produces. The negative result for hypothesis A was not forced; that's simply how it came out, per CLAUDE.md.
