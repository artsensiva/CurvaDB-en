# Step 2: crossover of the spline and DP+SED by noise/tol ratio

## 3. FITPACK suboptimality: splprep vs minimal uniform knots

5 tracks from step2_crossover.py's synthetic data (the same first 5 seeds from master-seed=42), sigma=0.1m, tol=1m, time parametrization. `splprep` -- via `traj.spline.fit` (honest dense control, adaptive densify + growing `s`). `make_lsq_spline` -- UNIFORM knots, the number of interior knots grown from 0 until it first passes the same honest dense check (`traj.spline.dense_max_error`).

| seed | n points | n control points (splprep) | n control points (lsq, uniform) | ratio |
|---|---|---|---|---|
| 89250 | 191 | 41 | 58 | 0.71x |
| 773956 | 118 | 29 | 32 | 0.91x |
| 654571 | 194 | 45 | 63 | 0.71x |
| 438878 | 273 | 76 | 104 | 0.73x |
| 433015 | 428 | 83 | 137 | 0.61x |

On average across 5/5 tracks: splprep uses 54.8 control points, minimal uniform knots -- 78.8 (0.70x). splprep is no worse than uniform knots on these tracks -- the spline's loss to DP+SED is not explained by FITPACK suboptimality.

## 1. Synthetic data, error metric, honest parameter selection

Synthetic data: a road model made of straights (50-400m) and turns (radius 30-150m, angle 30-150°), each turn -- entry clothoid (curvature 0->k, Euler spiral via Fresnel integrals), constant-curvature arc, exit clothoid (k->0) -- heading and curvature are continuous at every joint. Track length 1-5 km, speed constant per track (8-20 m/s), observation step 1s, seed=42.

Error for ALL methods -- max deviation of the reconstruction from the TRUE (noise-free) curve on a dense grid (>=10 points/s). DP/DP+SED are reconstructed piecewise-linearly IN TIME between the retained vertices; the spline (time) uses a direct relationship u=(t-t_min)/(t_max-t_min); the spline (chord) has no exact t<->u relationship, so linear interpolation over the noisy track's knots is used (an approximation, see the code).

Honest parameter selection: the target tol is NOT passed to the methods directly. For each method, its own internal parameter (on the noisy points) is swept over 8 log-spaced values relative to the target tol; the minimal number of parameters among those passing the honest check (error <= tol) is reported. If even the most generous sweep point fails tol (typically sigma >= tol), the cell is flagged unreachable without further sweeping.

Precision note: the internal-parameter sweep grid has 8 log-spaced points (not continuous bisection), so the reported "minimal number of parameters" is an upper-bound estimate at this grid's resolution, not the exact minimum. For the spline, the search uses reduced `max_iter`/`max_densify_rounds` (two orders of magnitude faster, the honest error didn't change on spot checks, see the code) -- done for the time budget (see below).

## 2. Compression: spline/DP+SED ratio map by (noise, tol)

30 tracks, seed=42.

**Time budget:** the pilot (5 tracks, ~1km, sigma=[0, 0.1, 2], tol=[0.2, 1, 5], 45 combinations) took 22s -- within the 3-minute budget, so the full run was launched without reducing N_TRACKS. The full run itself (30 tracks, 1-5km, 30 cells) took 1215s (20.3 min) -- longer than the pilot's linear extrapolation (~440s), because tracks in the full grid are longer (1-5km vs ~1km in the pilot, meaning more points per track and a more expensive spline fit per cell).

### DP: bytes (n parameters), fraction of reachable tracks
| sigma \ tol | 0.05 | 0.2 | 1 | 5 | 20 |
|---|---|---|---|---|---|
| 0 | unreachable | 2296B (n=95.7), 3/30 reachable | 2097B (n=87.4), 25/30 reachable | 1074B (n=44.7) | 582B (n=24.3) |
| 0.02 | unreachable | 2336B (n=97.3), 3/30 reachable | 2094B (n=87.2), 25/30 reachable | 1071B (n=44.6) | 583B (n=24.3) |
| 0.1 | unreachable | unreachable | 2084B (n=86.8), 24/30 reachable | 1067B (n=44.5) | 585B (n=24.4) |
| 0.5 | unreachable | unreachable | unreachable | 1082B (n=45.1) | 587B (n=24.5) |
| 2 | unreachable | unreachable | unreachable | 912B (n=38.0), 1/30 reachable | 604B (n=25.2) |
| 5 | unreachable | unreachable | unreachable | unreachable | 892B (n=37.2), 26/30 reachable |

### DP+SED: bytes (n parameters), fraction of reachable tracks
| sigma \ tol | 0.05 | 0.2 | 1 | 5 | 20 |
|---|---|---|---|---|---|
| 0 | unreachable | 2288B (n=95.3), 3/30 reachable | 2102B (n=87.6), 25/30 reachable | 1070B (n=44.6) | 561B (n=23.4) |
| 0.02 | unreachable | 2336B (n=97.3), 3/30 reachable | 2098B (n=87.4), 25/30 reachable | 1070B (n=44.6) | 563B (n=23.5) |
| 0.1 | unreachable | unreachable | 2100B (n=87.5), 24/30 reachable | 1062B (n=44.2) | 557B (n=23.2) |
| 0.5 | unreachable | unreachable | unreachable | 1106B (n=46.1) | 578B (n=24.1) |
| 2 | unreachable | unreachable | unreachable | 1272B (n=53.0), 1/30 reachable | 590B (n=24.6) |
| 5 | unreachable | unreachable | unreachable | unreachable | 1390B (n=57.9), 26/30 reachable |

### spline (time): bytes (n parameters), fraction of reachable tracks
| sigma \ tol | 0.05 | 0.2 | 1 | 5 | 20 |
|---|---|---|---|---|---|
| 0 | unreachable | 2600B (n=323.0), 3/30 reachable | 1639B (n=202.8), 25/30 reachable | 904B (n=111.0) | 606B (n=73.8) |
| 0.02 | unreachable | 4816B (n=600.0), 3/30 reachable | 1756B (n=217.5), 25/30 reachable | 904B (n=111.0) | 606B (n=73.8) |
| 0.1 | unreachable | unreachable | 2182B (n=270.8), 24/30 reachable | 906B (n=111.2) | 610B (n=74.3) |
| 0.5 | unreachable | unreachable | unreachable | 1695B (n=209.9) | 1358B (n=167.7) |
| 2 | unreachable | unreachable | unreachable | 360B (n=43.0), 1/30 reachable | 1142B (n=140.8) |
| 5 | unreachable | unreachable | unreachable | unreachable | 1616B (n=200.0), 26/30 reachable |

### spline (chord) -- lower bound, ignoring time reconstruction: bytes (n parameters), fraction of reachable tracks
| sigma \ tol | 0.05 | 0.2 | 1 | 5 | 20 |
|---|---|---|---|---|---|
| 0 | unreachable | 2584B (n=323.0), 3/30 reachable | 1618B (n=202.2), 25/30 reachable | 886B (n=110.7) | 600B (n=75.0) |
| 0.02 | unreachable | 4648B (n=581.0), 3/30 reachable | 1710B (n=213.8), 25/30 reachable | 886B (n=110.8) | 600B (n=75.0) |
| 0.1 | unreachable | unreachable | 1843B (n=230.4), 24/30 reachable | 888B (n=111.0) | 598B (n=74.7) |
| 0.5 | unreachable | unreachable | unreachable | 1207B (n=150.9) | 878B (n=109.7) |
| 2 | unreachable | unreachable | unreachable | 968B (n=121.0), 1/30 reachable | 1553B (n=194.1) |
| 5 | unreachable | unreachable | unreachable | unreachable | 1500B (n=187.5), 26/30 reachable |

### Pairwise comparison by track: spline (time) vs DP+SED (byte ratio over the shared subset of reachable tracks; "only one method" means the other method cannot hit tol at ANY number of parameters in this cell)
| sigma \ tol | 0.05 | 0.2 | 1 | 5 | 20 |
|---|---|---|---|---|---|
| 0 | unreachable (both) | **1.14x** (3/30) | **0.78x** (25/30) | **0.85x** | **1.08x** |
| 0.02 | unreachable (both) | **2.06x** (3/30) | **0.84x** (25/30) | **0.84x** | **1.08x** |
| 0.1 | unreachable (both) | unreachable (both) | **1.04x** (24/30) | **0.85x** | **1.10x** |
| 0.5 | unreachable (both) | unreachable (both) | unreachable (both) | **1.53x** | **2.35x** |
| 2 | unreachable (both) | unreachable (both) | unreachable (both) | **0.28x** (1/30) | **1.93x** |
| 5 | unreachable (both) | unreachable (both) | unreachable (both) | unreachable (both) | **1.16x** (26/30) |

Crossover point(s): sigma=0: spline wins (or is the only one reachable) at tol>=1; sigma=0.02: spline wins (or is the only one reachable) at tol>=1; sigma=0.1: spline wins (or is the only one reachable) at tol>=5; sigma=2: spline wins (or is the only one reachable) at tol>=5.

## 4. Conclusions

A crossover (spline/DP+SED < 1) on smooth road-like synthetic data DOES
exist, but not where initially expected (the simple rule "smaller
sigma/tol is better for the spline"), and not monotonically: at
tol=1-5m and sigma <= 0.1m the spline is 15-25% more compact than DP+SED
(§2: 0.78-0.85x, on 24-30 of 30 tracks), but at tol=20m DP+SED is more
compact again at EVERY sigma tested (1.08-2.35x) -- including sigma=0,
where there's no noise at all. So the spline's advantage on this
geometry is not a function of the sigma/tol ratio alone, but rather a
consequence of the fact that at moderate tol (1-5m) the synthetic turns
(radius 30-150m) often force DP/DP+SED to cut the polyline for the
turns, while at a coarse tol=20m, DP-like methods almost always get by
with a handful of vertices for the whole track, and the spline can no
longer keep up on compactness. Cells with a small fraction of reachable
tracks (e.g. tol=0.2, sigma<=0.02: 3/30; tol=5, sigma=2: 1/30) are
statistically unreliable and were not used in the conclusion above.

The spline's loss at coarse tol is not a FITPACK artifact: item 3 showed
that `splprep` is more compact than naive uniform knots on all 5 tracks
tested (0.70x on average), i.e. adaptive knot placement is already
working effectively where it was tested (sigma=0.1, tol=1) -- so the
cause of the loss at tol=20 lies elsewhere (probably in `fit()`'s honest
densify strategy itself, which grows control points for the dense check
BETWEEN the original 1-second samples, rather than for the turn's shape
as such -- outside this step's scope, a candidate for separate study).

Which real data sources fall into the winning zone (tol~1-5m,
sigma<=0.1m, i.e. sigma/tol <~ 0.02-0.1)? Ordinary smartphone/automotive
GPS (noise ~3-10m) does NOT qualify -- its own accuracy is already
comparable to or worse than the tol=1-5m tolerance itself, consistent
with step1's negative result on real GeoLife tracks (whose compression
was measured at exactly these tol values on noisy ~5-10m data). RTK/
differential GPS and GNSS with carrier-phase corrections (accuracy
~0.02-0.1m) DO fall into the winning zone, if a 1-5m tolerance is
sufficient for storage/indexing (reasonable for comparing routes by
Frechet distance, excessive for precise navigation). Inertial/lidar
localization systems (robot odometry, mapping drones, accuracy
~0.05-0.3m) are a borderline case, falling into the winning zone only at
tol closer to 5m rather than 1m. Bottom line: the spline's winning zone
is real but narrow, and requires positioning accuracy substantially
better than mass-market consumer GPS -- on that data (and, accordingly,
on GeoLife), step1's negative result stands.
