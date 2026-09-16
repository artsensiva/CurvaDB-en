# Step 0: spline vs DP polyline vs raw polyline

Tracks: 200, tol = 10.0 m.

Track length (points): min=55, median=412, max=1979, mean=576.

Mean number of DP polyline vertices: 52.1. Mean number of spline control points: 125.8. Non-converged fits: 0/200.


## Bytes per track (float64)

| Representation | Bytes/track |
|---|---|
| raw | 9224 |
| DP polyline | 833 |
| spline (knots + coefficients) | 3052 |

## Top-10 latency, full scan, 30 queries (ms)

| Representation | p50 | p95 |
|---|---|---|
| raw | 35.67 | 210.66 |
| dp | 1.55 | 2.46 |
| spline_same | 1.65 | 3.31 |
| spline_arclen | 255.84 | 3050.38 |

## Recall@10 against Frechet distance on raw tracks

| Representation | Recall@10 |
|---|---|
| dp | 0.997 |
| spline_same | 0.707 |
| spline_arclen | 0.707 |

## Velocity/acceleration error against finite differences on the smoothed raw track

| Representation | Velocity, m/s (median) | Acceleration, m/s² (median) |
|---|---|---|
| spline | 0.875 | 0.152 |
| DP polyline | 1.519 | 0.179 |

Note: the DP polyline is piecewise-linear, so its acceleration is identically 0 between vertices -- the DP acceleration error shows how large the real acceleration is that a polyline fundamentally cannot represent.
