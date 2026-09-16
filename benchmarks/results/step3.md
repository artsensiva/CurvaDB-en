## 0. Sanity-checking step2's accounting -- where n=200.0 and n=600.0 come from

Reproduces `step2_crossover.evaluate_cell`'s procedure for the spline_time method verbatim (a log grid of internal_tol with 8 points relative to target_tol, honest error -- true_curve_error against the ground-truth curve on a dense grid), but shows the WHOLE internal_tol grid per track, not just the chosen minimum n. n = n_knots + 2*n_coeffs (float64 scalars, NOT bytes).

### sigma=0.02, tol=0.2

internal_tol grid (m): 0.002, 0.005365, 0.01439, 0.03861, 0.1036, 0.2779, 0.7455, 2

| seed | n_raw | reachable | chosen itol | n (chosen) | n_knots | n_coeffs | err (chosen) | full grid n@itol (err<=tol?) |
|---|---|---|---|---|---|---|---|---|
| 89250 | 191 | no (first grid point fails) | -- | -- | -- | -- | -- | 1903@0.002(no); 1678@0.00537(no); 1549@0.0144(no); 1438@0.0386(no); 931@0.104(no); 655@0.278(no); 142@0.746(no); 109@2(no) |
| 773956 | 118 | no (first grid point fails) | -- | -- | -- | -- | -- | 1282@0.002(no); 1171@0.00537(no); 1063@0.0144(no); 1009@0.0386(no); 763@0.104(no); 253@0.278(no); 94@0.746(no); 76@2(no) |
| 654571 | 194 | no (first grid point fails) | -- | -- | -- | -- | -- | 1939@0.002(no); 1603@0.00537(no); 1402@0.0144(no); 1315@0.0386(no); 892@0.104(no); 250@0.278(no); 163@0.746(no); 100@2(no) |
| 438878 | 273 | no (first grid point fails) | -- | -- | -- | -- | -- | 2785@0.002(no); 2470@0.00537(no); 2275@0.0144(no); 2137@0.0386(no); 1594@0.104(no); 850@0.278(no); 439@0.746(ok); 190@2(no) |
| 433015 | 428 | no (first grid point fails) | -- | -- | -- | -- | -- | 4093@0.002(no); 3295@0.00537(no); 2845@0.0144(no); 2671@0.0386(no); 1822@0.104(no); 472@0.278(no); 265@0.746(no); 217@2(no) |
| 858597 | 239 | no (first grid point fails) | -- | -- | -- | -- | -- | 2416@0.002(no); 2077@0.00537(no); 1852@0.0144(no); 1753@0.0386(no); 1615@0.104(no); 1099@0.278(no); 319@0.746(no); 187@2(no) |
| 85945 | 101 | no (first grid point fails) | -- | -- | -- | -- | -- | 985@0.002(no); 775@0.00537(no); 652@0.0144(no); 604@0.0386(no); 490@0.104(no); 187@0.278(no); 118@0.746(no); 70@2(no) |
| 697368 | 318 | no (first grid point fails) | -- | -- | -- | -- | -- | 3190@0.002(no); 2701@0.00537(no); 2374@0.0144(no); 2125@0.0386(no); 1474@0.104(no); 346@0.278(no); 187@0.746(no); 142@2(no) |
| 201469 | 178 | no (first grid point fails) | -- | -- | -- | -- | -- | 1702@0.002(no); 1405@0.00537(no); 1240@0.0144(no); 1177@0.0386(no); 1108@0.104(no); 751@0.278(no); 265@0.746(no); 151@2(no) |
| 94177 | 221 | yes | 0.1036 | 682 | 230 | 226 | 0.1783 | 2218@0.002(ok); 1813@0.00537(ok); 1486@0.0144(ok); 1033@0.0386(ok); 682@0.104(ok); 130@0.278(no); 100@0.746(no); 73@2(no) |
| 526478 | 119 | yes | 0.1036 | 124 | 44 | 40 | 0.1021 | 1198@0.002(ok); 988@0.00537(ok); 889@0.0144(ok); 748@0.0386(ok); 124@0.104(ok); 67@0.278(no); 58@0.746(no); 49@2(no) |
| 975622 | 128 | no (first grid point fails) | -- | -- | -- | -- | -- | 1270@0.002(no); 1057@0.00537(no); 946@0.0144(no); 892@0.0386(no); 730@0.104(no); 517@0.278(no); 94@0.746(no); 79@2(no) |
| 735752 | 238 | no (first grid point fails) | -- | -- | -- | -- | -- | 2422@0.002(no); 1990@0.00537(no); 1795@0.0144(no); 1678@0.0386(no); 1141@0.104(no); 253@0.278(no); 169@0.746(no); 121@2(no) |
| 761139 | 257 | yes | 0.1036 | 994 | 334 | 330 | 0.1955 | 2560@0.002(ok); 2038@0.00537(ok); 1789@0.0144(ok); 1456@0.0386(ok); 994@0.104(ok); 175@0.278(no); 124@0.746(no); 94@2(no) |
| 717477 | 342 | no (first grid point fails) | -- | -- | -- | -- | -- | 3358@0.002(no); 2701@0.00537(no); 2323@0.0144(no); 2185@0.0386(no); 1558@0.104(no); 790@0.278(no); 268@0.746(no); 199@2(no) |
| 786064 | 177 | no (first grid point fails) | -- | -- | -- | -- | -- | 1822@0.002(no); 1573@0.00537(no); 1408@0.0144(no); 1345@0.0386(no); 1117@0.104(no); 817@0.278(no); 244@0.746(no); 136@2(no) |
| 513226 | 178 | no (first grid point fails) | -- | -- | -- | -- | -- | 1762@0.002(no); 1507@0.00537(no); 1354@0.0144(no); 1282@0.0386(no); 844@0.104(no); 352@0.278(no); 133@0.746(no); 97@2(no) |
| 128113 | 142 | no (first grid point fails) | -- | -- | -- | -- | -- | 1372@0.002(no); 1129@0.00537(no); 1003@0.0144(no); 964@0.0386(no); 694@0.104(no); 412@0.278(no); 121@0.746(no); 91@2(no) |
| 839748 | 196 | no (first grid point fails) | -- | -- | -- | -- | -- | 2017@0.002(no); 1750@0.00537(no); 1573@0.0144(no); 1489@0.0386(no); 1075@0.104(no); 754@0.278(no); 229@0.746(no); 130@2(no) |
| 450385 | 200 | no (first grid point fails) | -- | -- | -- | -- | -- | 1972@0.002(no); 1681@0.00537(no); 1519@0.0144(no); 1414@0.0386(no); 1252@0.104(no); 874@0.278(no); 226@0.746(no); 139@2(no) |
| 500351 | 90 | no (first grid point fails) | -- | -- | -- | -- | -- | 931@0.002(no); 805@0.00537(no); 703@0.0144(no); 667@0.0386(no); 463@0.104(no); 367@0.278(no); 79@0.746(no); 64@2(no) |
| 370798 | 151 | no (first grid point fails) | -- | -- | -- | -- | -- | 1570@0.002(no); 1339@0.00537(no); 1174@0.0144(no); 1117@0.0386(no); 844@0.104(no); 421@0.278(no); 190@0.746(no); 109@2(no) |
| 182549 | 328 | no (first grid point fails) | -- | -- | -- | -- | -- | 3190@0.002(no); 2572@0.00537(no); 2275@0.0144(no); 1855@0.0386(no); 1366@0.104(no); 319@0.278(no); 166@0.746(no); 139@2(no) |
| 926764 | 79 | no (first grid point fails) | -- | -- | -- | -- | -- | 760@0.002(no); 646@0.00537(no); 577@0.0144(no); 547@0.0386(no); 439@0.104(no); 325@0.278(no); 82@0.746(no); 58@2(no) |
| 781567 | 242 | no (first grid point fails) | -- | -- | -- | -- | -- | 2329@0.002(no); 1900@0.00537(no); 1627@0.0144(no); 1543@0.0386(no); 1171@0.104(no); 361@0.278(no); 235@0.746(no); 136@2(no) |
| 643865 | 202 | no (first grid point fails) | -- | -- | -- | -- | -- | 2029@0.002(no); 1738@0.00537(no); 1552@0.0144(no); 1462@0.0386(no); 1060@0.104(no); 544@0.278(no); 175@0.746(no); 121@2(no) |
| 402414 | 197 | no (first grid point fails) | -- | -- | -- | -- | -- | 1966@0.002(no); 1603@0.00537(no); 1423@0.0144(no); 1363@0.0386(no); 1258@0.104(no); 889@0.278(no); 337@0.746(no); 145@2(no) |
| 822761 | 93 | no (first grid point fails) | -- | -- | -- | -- | -- | 913@0.002(no); 730@0.00537(no); 619@0.0144(no); 583@0.0386(no); 496@0.104(no); 115@0.278(no); 79@0.746(no); 67@2(no) |
| 545429 | 331 | no (first grid point fails) | -- | -- | -- | -- | -- | 3316@0.002(no); 2686@0.00537(no); 2362@0.0144(no); 2239@0.0386(no); 1672@0.104(no); 706@0.278(no); 250@0.746(no); 196@2(no) |
| 443414 | 189 | no (first grid point fails) | -- | -- | -- | -- | -- | 1834@0.002(no); 1456@0.00537(no); 1240@0.0144(no); 1033@0.0386(no); 346@0.104(ok); 133@0.278(ok); 100@0.746(no); 73@2(no) |

Reachable: 3/30; mean n = 600; n across reachable tracks is different: [124, 682, 994].

### sigma=5, tol=20

internal_tol grid (m): 0.2, 0.5365, 1.439, 3.861, 10.36, 27.79, 74.55, 200

| seed | n_raw | reachable | chosen itol | n (chosen) | n_knots | n_coeffs | err (chosen) | full grid n@itol (err<=tol?) |
|---|---|---|---|---|---|---|---|---|
| 89250 | 191 | yes | 10.36 | 181 | 63 | 59 | 9.127 | 1963@0.2(ok); 1393@0.537(ok); 685@1.44(ok); 574@3.86(ok); 181@10.4(ok); 43@27.8(no); 28@74.6(no); 22@200(no) |
| 773956 | 118 | yes | 27.79 | 43 | 17 | 13 | 19.72 | 1210@0.2(ok); 877@0.537(ok); 490@1.44(ok); 370@3.86(ok); 139@10.4(ok); 43@27.8(ok); 31@74.6(no); 16@200(no) |
| 654571 | 194 | no (first grid point fails) | -- | -- | -- | -- | -- | 2047@0.2(no); 1474@0.537(no); 799@1.44(no); 577@3.86(no); 208@10.4(ok); 46@27.8(no); 31@74.6(no); 22@200(no) |
| 438878 | 273 | yes | 10.36 | 310 | 106 | 102 | 11 | 2917@0.2(ok); 2041@0.537(ok); 1153@1.44(ok); 712@3.86(ok); 310@10.4(ok); 67@27.8(no); 49@74.6(no); 28@200(no) |
| 433015 | 428 | no (first grid point fails) | -- | -- | -- | -- | -- | 4414@0.2(no); 3121@0.537(no); 1705@1.44(no); 1288@3.86(no); 1288@10.4(no); 1288@27.8(no); 1288@74.6(no); 1288@200(no) |
| 858597 | 239 | yes | 10.36 | 268 | 92 | 88 | 11.4 | 2467@0.2(ok); 1777@0.537(ok); 973@1.44(ok); 715@3.86(ok); 268@10.4(ok); 85@27.8(no); 55@74.6(no); 28@200(no) |
| 85945 | 101 | yes | 10.36 | 109 | 39 | 35 | 10.79 | 1039@0.2(ok); 766@0.537(ok); 439@1.44(ok); 307@3.86(ok); 109@10.4(ok); 31@27.8(no); 22@74.6(no); 16@200(no) |
| 697368 | 318 | yes | 10.36 | 340 | 116 | 112 | 17.59 | 3361@0.2(ok); 2278@0.537(ok); 1252@1.44(ok); 841@3.86(ok); 340@10.4(ok); 73@27.8(no); 40@74.6(no); 28@200(no) |
| 201469 | 178 | yes | 27.79 | 67 | 25 | 21 | 18.02 | 1858@0.2(ok); 1342@0.537(ok); 757@1.44(ok); 484@3.86(ok); 196@10.4(ok); 67@27.8(ok); 37@74.6(no); 28@200(no) |
| 94177 | 221 | yes | 10.36 | 235 | 81 | 77 | 12.85 | 2224@0.2(ok); 1447@0.537(ok); 808@1.44(ok); 583@3.86(ok); 235@10.4(ok); 40@27.8(no); 22@74.6(no); 16@200(no) |
| 526478 | 119 | yes | 10.36 | 130 | 46 | 42 | 12.4 | 1231@0.2(ok); 859@0.537(ok); 463@1.44(ok); 325@3.86(ok); 130@10.4(ok); 25@27.8(no); 19@74.6(no); 16@200(no) |
| 975622 | 128 | yes | 27.79 | 46 | 18 | 14 | 17.7 | 1336@0.2(ok); 922@0.537(ok); 502@1.44(ok); 388@3.86(ok); 157@10.4(ok); 46@27.8(ok); 22@74.6(no); 19@200(no) |
| 735752 | 238 | no (first grid point fails) | -- | -- | -- | -- | -- | 2506@0.2(no); 1825@0.537(no); 928@1.44(no); 718@3.86(ok); 268@10.4(ok); 61@27.8(no); 31@74.6(no); 25@200(no) |
| 761139 | 257 | yes | 10.36 | 226 | 78 | 74 | 10.86 | 2638@0.2(ok); 1876@0.537(ok); 1006@1.44(ok); 706@3.86(ok); 226@10.4(ok); 43@27.8(no); 28@74.6(no); 19@200(no) |
| 717477 | 342 | yes | 10.36 | 400 | 136 | 132 | 12.09 | 3484@0.2(ok); 2365@0.537(ok); 1330@1.44(ok); 1018@3.86(ok); 400@10.4(ok); 67@27.8(no); 49@74.6(no); 34@200(no) |
| 786064 | 177 | yes | 10.36 | 202 | 70 | 66 | 11.3 | 1864@0.2(ok); 1384@0.537(ok); 790@1.44(ok); 484@3.86(ok); 202@10.4(ok); 55@27.8(no); 34@74.6(no); 25@200(no) |
| 513226 | 178 | yes | 10.36 | 187 | 65 | 61 | 8.349 | 1909@0.2(ok); 1354@0.537(ok); 745@1.44(ok); 490@3.86(ok); 187@10.4(ok); 40@27.8(no); 34@74.6(no); 19@200(no) |
| 128113 | 142 | yes | 10.36 | 181 | 63 | 59 | 10.04 | 1531@0.2(ok); 1075@0.537(ok); 502@1.44(ok); 349@3.86(ok); 181@10.4(ok); 40@27.8(no); 25@74.6(no); 19@200(no) |
| 839748 | 196 | yes | 10.36 | 292 | 100 | 96 | 10.63 | 1993@0.2(ok); 1456@0.537(ok); 823@1.44(ok); 592@3.86(ok); 292@10.4(ok); 58@27.8(no); 43@74.6(no); 28@200(no) |
| 450385 | 200 | yes | 10.36 | 229 | 79 | 75 | 9.394 | 2179@0.2(ok); 1537@0.537(ok); 826@1.44(ok); 502@3.86(ok); 229@10.4(ok); 58@27.8(no); 37@74.6(no); 28@200(no) |
| 500351 | 90 | yes | 10.36 | 76 | 28 | 24 | 12.78 | 922@0.2(ok); 637@0.537(ok); 364@1.44(ok); 208@3.86(ok); 76@10.4(ok); 31@27.8(no); 22@74.6(no); 16@200(no) |
| 370798 | 151 | yes | 3.861 | 415 | 141 | 137 | 13.2 | 1522@0.2(ok); 1099@0.537(ok); 631@1.44(ok); 415@3.86(ok); 457@10.4(ok); 457@27.8(ok); 457@74.6(ok); 457@200(ok) |
| 182549 | 328 | yes | 10.36 | 271 | 93 | 89 | 9.742 | 3391@0.2(ok); 2347@0.537(ok); 1330@1.44(ok); 1009@3.86(ok); 271@10.4(ok); 52@27.8(no); 37@74.6(no); 22@200(no) |
| 926764 | 79 | yes | 10.36 | 100 | 36 | 32 | 10.4 | 784@0.2(ok); 553@0.537(ok); 313@1.44(ok); 217@3.86(ok); 100@10.4(ok); 25@27.8(no); 22@74.6(no); 16@200(no) |
| 781567 | 242 | no (first grid point fails) | -- | -- | -- | -- | -- | 2560@0.2(no); 1828@0.537(no); 988@1.44(no); 655@3.86(ok); 274@10.4(ok); 67@27.8(no); 40@74.6(no); 28@200(no) |
| 643865 | 202 | yes | 10.36 | 223 | 77 | 73 | 11.09 | 2113@0.2(ok); 1549@0.537(ok); 832@1.44(ok); 580@3.86(ok); 223@10.4(ok); 55@27.8(no); 37@74.6(no); 22@200(no) |
| 402414 | 197 | yes | 10.36 | 163 | 57 | 53 | 11.97 | 2065@0.2(ok); 1435@0.537(ok); 793@1.44(ok); 493@3.86(ok); 163@10.4(ok); 58@27.8(no); 46@74.6(no); 25@200(no) |
| 822761 | 93 | yes | 10.36 | 115 | 41 | 37 | 9.725 | 988@0.2(ok); 724@0.537(ok); 403@1.44(ok); 253@3.86(ok); 115@10.4(ok); 28@27.8(no); 19@74.6(no); 16@200(no) |
| 545429 | 331 | yes | 10.36 | 358 | 122 | 118 | 11.17 | 3430@0.2(ok); 2434@0.537(ok); 1318@1.44(ok); 814@3.86(ok); 358@10.4(ok); 70@27.8(no); 46@74.6(no); 31@200(no) |
| 443414 | 189 | yes | 27.79 | 34 | 14 | 10 | 19.84 | 1912@0.2(ok); 1342@0.537(ok); 748@1.44(ok); 487@3.86(ok); 178@10.4(ok); 34@27.8(ok); 574@74.6(no); 16@200(no) |

Reachable: 26/30; mean n = 200.04 (sum 5201/26); when displayed with 1 decimal place (as in `_summarize_method`: `f"{mean_n:.1f}"`) this prints as "200.0"; n across reachable tracks is different: [34, 43, 46, 67, 76, 100, 109, 115, 130, 163, 181, 187, 202, 223, 226, 229, 235, 268, 271, 292, 310, 340, 358, 400, 415].

### Conclusion

1. `n` in step2.md's tables counts scalars: `n = n_knots + 2*n_coeffs` (see
   `step2_crossover._build_spline`) -- `n_knots = len(tck[0])` (float,
   knot positions), `n_coeffs = len(tck[1][0])` (one x coordinate per
   control point; the y coefficient gives the second `2*n_coeffs` term).
   Confirmed by direct output of `len(knots)`/`len(coeffs)` for each
   track in the tables above.
2. "The means being exactly 200.0 and 600.0" -- is NOT a sign of hitting
   some limit (`MAX_ITER`, `S_HI_CAP_MULT`, `max_points_mult`): the full
   internal_tol sweep for each track (the "full grid n@itol" column)
   shows widely varying `n` (34..415 for sigma=5/tol=20; 124/682/994 for
   sigma=0.02/tol=0.2) -- clearly NOT the same value for any track.
   - sigma=0.02, tol=0.2: 3 reachable tracks, `n` = [124, 682, 994],
     sum 1800, 1800/3 = 600.0 -- an EXACT coincidence of the sum and the
     track count, not a bug artifact.
   - sigma=5, tol=20: 26 reachable tracks, sum of n = 5201,
     5201/26 = 200.0385 -- when printed with `.1f` (the format used by
     `_summarize_method` in step2_crossover.py) it rounds to "200.0";
     this is DISPLAY rounding, not the exact mean.
   Bottom line: both "round" numbers in step2.md are coincidences (an
   exact division in one case, rounding to 1 decimal in the other), not
   a counting bug.

## 1. LSQ spline of the true curve: the contract

`src/traj/spline_lsq.py` -- an LSQ spline of x(t)/y(t) directly (`make_lsq_spline`), with no tethering to the polyline (unlike `spline.fit()`). Internal knots -- a subset of the actual t samples (automatically satisfies the Schoenberg-Whitney condition): `uniform` -- evenly spaced by index; `adaptive` -- two-pass (a trial uniform fit -> residuals at the points -> redistribute knots by cumulative residual -> refit). Bisection over the number of interior knots m grows it toward the theoretical limit (n-k-2), but tracks the BEST (not the last) error along the way -- right at that boundary (near-interpolation of noisy and/or sparse points), the error occasionally jumps by 1-2 orders of magnitude, invisible to the residual at the TRAINING points (see the spline_lsq.py commit). The honest error inside the fitter is the DIRECT Euclidean residual at the points themselves (t_i, xy_i), not point-to-segment to the polyline -- and that's exactly why it does NOT see the spline oscillating BETWEEN points (see the next paragraph).

`fit_oracle` -- the same fitter, fit DIRECTLY to the dense true (noise-free) curve; knows nothing about noise or observation sparsity -- a reference lower bound on the spline's geometry representation at a given tol, computed once per (seed, tol), doesn't participate in criteria K1-K3.

Selecting the method's internal parameter (tol on the noisy points) -- a coarse log scan (8 points, as in step2) + refining bisection around the best passing point (`search_min_params`/`_bisect_between` in step3_decisive.py), rather than pure bisection from the strictest internal_tol: err(itol) for the LSQ spline isn't monotone -- too small an itol on sparse (dt>=5s) points can produce a spline that passes close to the samples but wildly oscillates BETWEEN them (the honest error on the dense grid is then orders of magnitude worse, even though the residual at the points themselves is small -- the fitter's internal criterion doesn't see this; found empirically while debugging). The coarse scan is robust to this non-monotonicity; pure bisection from the strictest end sometimes incorrectly flagged a cell unreachable. Size -- a single scheme for all methods: time/coordinates -> rounding to cm/centiseconds -> int32 (with overflow clipping) -> deltas -> zlib.compress.

## 2. Tables by dt (spline uniform/adaptive vs DP+SED, oracle separate)

15 tracks, seed=42.

### dt = 1 s

#### DP+SED: bytes (n parameters), fraction of reachable tracks
| sigma \ tol | 0.5 | 2 | 10 |
|---|---|---|---|
| 0 | 479B (n=264.0), 5/15 reachable | 467B (n=204.6) | 278B (n=95.0) |
| 0.1 | 413B (n=228.0), 1/15 reachable | 478B (n=206.6) | 277B (n=94.2) |
| 1 | unreachable | unreachable | 282B (n=92.8) |
| 5 | unreachable | unreachable | unreachable |

#### LSQ spline (uniform): bytes (n parameters), fraction of reachable tracks
| sigma \ tol | 0.5 | 2 | 10 |
|---|---|---|---|
| 0 | 615B (n=386.7), 14/15 reachable | 438B (n=228.2) | 273B (n=114.6) |
| 0.1 | 551B (n=320.8), 10/15 reachable | 445B (n=230.0) | 274B (n=114.8) |
| 1 | unreachable | 271B (n=124.0), 4/15 reachable | 266B (n=111.0) |
| 5 | unreachable | unreachable | 249B (n=101.7), 7/15 reachable |

#### LSQ spline (adaptive): bytes (n parameters), fraction of reachable tracks
| sigma \ tol | 0.5 | 2 | 10 |
|---|---|---|---|
| 0 | 708B (n=386.9), 14/15 reachable | 582B (n=274.4), 14/15 reachable | 353B (n=134.0) |
| 0.1 | 626B (n=309.7), 10/15 reachable | 512B (n=223.0) | 350B (n=133.4) |
| 1 | unreachable | 242B (n=88.0), 2/15 reachable | 298B (n=103.8) |
| 5 | unreachable | unreachable | 267B (n=98.0), 6/15 reachable |

### dt = 5 s

#### DP+SED: bytes (n parameters), fraction of reachable tracks
| sigma \ tol | 0.5 | 2 | 10 |
|---|---|---|---|
| 0 | unreachable | unreachable | 145B (n=51.0), 1/15 reachable |
| 0.1 | unreachable | unreachable | 156B (n=51.0), 1/15 reachable |
| 1 | unreachable | unreachable | 147B (n=45.0), 2/15 reachable |
| 5 | unreachable | unreachable | unreachable |

#### LSQ spline (uniform): bytes (n parameters), fraction of reachable tracks
| sigma \ tol | 0.5 | 2 | 10 |
|---|---|---|---|
| 0 | unreachable | unreachable | 251B (n=104.2), 5/15 reachable |
| 0.1 | unreachable | unreachable | 274B (n=116.5), 6/15 reachable |
| 1 | unreachable | unreachable | 212B (n=82.0), 3/15 reachable |
| 5 | unreachable | unreachable | unreachable |

#### LSQ spline (adaptive): bytes (n parameters), fraction of reachable tracks
| sigma \ tol | 0.5 | 2 | 10 |
|---|---|---|---|
| 0 | unreachable | unreachable | 279B (n=109.4), 7/15 reachable |
| 0.1 | unreachable | 192B (n=67.0), 1/15 reachable | 298B (n=118.0), 8/15 reachable |
| 1 | unreachable | unreachable | 300B (n=118.4), 7/15 reachable |
| 5 | unreachable | unreachable | unreachable |

### dt = 15 s

#### DP+SED: bytes (n parameters), fraction of reachable tracks
| sigma \ tol | 0.5 | 2 | 10 |
|---|---|---|---|
| 0 | unreachable | unreachable | unreachable |
| 0.1 | unreachable | unreachable | unreachable |
| 1 | unreachable | unreachable | unreachable |
| 5 | unreachable | unreachable | unreachable |

#### LSQ spline (uniform): bytes (n parameters), fraction of reachable tracks
| sigma \ tol | 0.5 | 2 | 10 |
|---|---|---|---|
| 0 | unreachable | unreachable | unreachable |
| 0.1 | unreachable | unreachable | unreachable |
| 1 | unreachable | unreachable | unreachable |
| 5 | unreachable | unreachable | unreachable |

#### LSQ spline (adaptive): bytes (n parameters), fraction of reachable tracks
| sigma \ tol | 0.5 | 2 | 10 |
|---|---|---|---|
| 0 | unreachable | unreachable | unreachable |
| 0.1 | unreachable | unreachable | unreachable |
| 1 | unreachable | unreachable | unreachable |
| 5 | unreachable | unreachable | unreachable |

### Oracle (LSQ spline on the true curve, for reference -- once per (seed, tol), independent of sigma/dt)

| tol | mean n | mean bytes | fraction reachable (bisection converged) |
|---|---|---|---|
| 0.5 | 386.0 | 605 | 15/15 |
| 2 | 243.4 | 457 | 15/15 |
| 10 | 124.4 | 292 | 15/15 |

## 3. Criteria K1-K3

15 tracks (80% = 12/15, 20% = 3/15). Numbers are from the §2 tables above.

### K1 (compression): after zlib, the spline is >= 30% smaller than DP+SED in at least two cells with sigma/tol <= 0.1, at >= 80% reachability for both

Cells with sigma/tol <= 0.1 (all dt): dt=1 -- (0,0.5), (0,2), (0,10), (0.1,2), (0.1,10), (1,10); dt=5 and dt=15 -- the same (sigma,tol) pairs.

Spline/DP+SED byte ratio (uniform, the better of the two spline variants by bytes in each cell), where both methods are >= 80% reachable:
- dt=1, sigma=0, tol=2: both 15/15; 438/467 = 0.94x (6% smaller, not 30%).
- dt=1, sigma=0, tol=10: both 15/15; 273/278 = 0.98x.
- dt=1, sigma=0.1, tol=2: both 15/15; 445/478 = 0.93x.
- dt=1, sigma=0.1, tol=10: both 15/15; 274/277 = 0.99x.
- dt=1, sigma=1, tol=10: both 15/15; 266/282 = 0.94x.
- dt=1, sigma=0, tol=0.5: DP+SED 5/15 (33%) -- below the 80% threshold, cell doesn't count.
- dt=5 and dt=15, all listed pairs: DP+SED is reachable on 0-13% of tracks (or unreachable) -- below the 80% threshold, cells don't count.

No cell showed >= 30% spline compression (the maximum observed compression is 7%, dt=1/sigma=0/tol=2); there are exactly 5 cells where BOTH methods are >= 80% reachable at sigma/tol<=0.1 (all at dt=1), and the compression threshold isn't met in any of them.

**Verdict: not met.**

### K2 (reconstruction): at dt >= 5s there are cells where the spline is >= 80% reachable and DP+SED is <= 20%

At dt=5 and dt=15, DP+SED is indeed almost everywhere <= 20% (0-13% reachability, often 0%) -- the condition on DP+SED holds in nearly all dt>=5 cells. But the spline must SIMULTANEOUSLY be >= 80% -- the maximum spline reachability at dt=5 across all cells: LSQ spline (adaptive), sigma=0.1, tol=10 -- 8/15 (53%); no dt=5 cell reaches 12/15. At dt=15 both methods are unreachable in 100% of cells (0/15 for every method, every sigma and tol) -- the spline doesn't work there either.

**Verdict: not met** (the DP+SED condition holds, but the required >= 80% spline share isn't reached in any dt>=5 cell -- maximum 53%, dt=5/sigma=0.1/tol=10).

### K3 (noise): at sigma = 5 there is a tol where the spline is >= 80% reachable and DP+SED is <= 20%

At sigma=5, DP+SED is unreachable (0/15) in every cell (all dt, all tol) except those absent from the grid -- the DP+SED condition holds trivially almost everywhere. Maximum spline reachability at sigma=5 across all cells: LSQ spline (uniform), dt=1, tol=10 -- 7/15 (47%); LSQ spline (adaptive), dt=1, tol=10 -- 6/15 (40%). No cell reaches 12/15.

**Verdict: not met.**

### Summary across all three criteria

K1, K2, and K3 -- not met on the data collected (15 tracks, seed=42). In cells where DP+SED becomes unreachable (dt>=5s and/or sigma=5), the LSQ spline is also unreachable on most tracks (reachability 0-53%) -- the spline's advantage in §2 is only partially observed (its share of tracks grows relative to DP+SED, which is often 0%), but doesn't reach the established 80% threshold. The oracle (for reference) is reachable 15/15 at every tol -- the limitation isn't that the spline is fundamentally unable to represent the geometry, but that the fitter only sees the noisy/sparse samples, not the true curve.
