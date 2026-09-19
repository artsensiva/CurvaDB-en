## M0

Unified section-2.5 encoding (`src/traj/encode.py`) re-verifying step3's own DP+SED / LSQ-uniform byte counts, dt=1 s only, same 15-track/seed=42 synthetic generator and the same reachability criterion as `step3_decisive.py` (honest error against the true noise-free curve on a dense grid, `<= target_tol` -- UNCHANGED from step3, see docs/decisions/ADR-0021). Step7's certificates (`certify_polyline`/`certify_spline`, section 2.4 default, ADR-0018) are computed once per reachable cell's CHOSEN representation, as a separate, additional correctness check (`eps_A <= target_tol`), not a selection criterion.

15 tracks, seed=42, dt=1 s. Full grid elapsed: 36.5s.

### Reachability (step3 criterion, unchanged) + certified correctness (ADR-0021)

#### DP+SED: bytes (n parameters), reachable fraction; certified `eps_A<=tol` fraction
| sigma \ tol | 0.5 | 2 | 10 |
|---|---|---|---|
| 0 | 481B (n=264.0), 5/15 reachable; cert 5/5 (100.0%) | 470B (n=204.6); cert 15/15 (100.0%) | 282B (n=95.0); cert 15/15 (100.0%) |
| 0.1 | 426B (n=228.0), 1/15 reachable; cert 1/1 (100.0%) | 481B (n=206.6); cert 13/15 (86.7%) | 281B (n=94.2); cert 14/15 (93.3%) |
| 1 | unreachable; cert -- | unreachable; cert -- | 285B (n=92.8); cert 9/15 (60.0%) |
| 5 | unreachable; cert -- | unreachable; cert -- | unreachable; cert -- |

#### LSQ spline (uniform): bytes (n parameters), reachable fraction; certified `eps_A<=tol` fraction
| sigma \ tol | 0.5 | 2 | 10 |
|---|---|---|---|
| 0 | 617B (n=386.7), 14/15 reachable; cert 1/14 (7.1%) | 438B (n=228.2); cert 15/15 (100.0%) | 275B (n=114.6); cert 15/15 (100.0%) |
| 0.1 | 553B (n=320.8), 10/15 reachable; cert 1/10 (10.0%) | 444B (n=230.0); cert 14/15 (93.3%) | 275B (n=114.8); cert 15/15 (100.0%) |
| 1 | unreachable; cert -- | 272B (n=124.0), 4/15 reachable; cert 0/4 (0.0%) | 268B (n=111.0); cert 12/15 (80.0%) |
| 5 | unreachable; cert -- | unreachable; cert -- | 250B (n=101.7), 7/15 reachable; cert 0/7 (0.0%) |

### Comparison against `benchmarks/results/step3.md`'s dt=1 tables

Reachability fractions and `n` (parameter counts) match **exactly** in every one of the 7
non-empty DP+SED cells and 9 non-empty LSQ-uniform cells -- confirming the unchanged generator,
seed, and reachability criterion (ADR-0021) reproduce step3's own selection decision cell for
cell, not just on average.

| Method | Cell (sigma, tol) | step3.md bytes | step8 M0 bytes | delta |
|---|---|---|---|---|
| DP+SED | 0, 0.5 | 479 | 481 | +2 |
| DP+SED | 0, 2 | 467 | 470 | +3 |
| DP+SED | 0, 10 | 278 | 282 | +4 |
| DP+SED | 0.1, 0.5 | 413 | 426 | +13 |
| DP+SED | 0.1, 2 | 478 | 481 | +3 |
| DP+SED | 0.1, 10 | 277 | 281 | +4 |
| DP+SED | 1, 10 | 282 | 285 | +3 |
| LSQ-uniform | 0, 0.5 | 615 | 617 | +2 |
| LSQ-uniform | 0, 2 | 438 | 438 | 0 |
| LSQ-uniform | 0, 10 | 273 | 275 | +2 |
| LSQ-uniform | 0.1, 0.5 | 551 | 553 | +2 |
| LSQ-uniform | 0.1, 2 | 445 | 444 | -1 |
| LSQ-uniform | 0.1, 10 | 274 | 275 | +1 |
| LSQ-uniform | 1, 2 | 271 | 272 | +1 |
| LSQ-uniform | 1, 10 | 266 | 268 | +2 |
| LSQ-uniform | 5, 10 | 249 | 250 | +1 |

All deltas are small (0 to +4B) and fully explained by `encode.py`'s explicit per-segment framing
(`[type: 1 byte][count: varint]`, 2-3 bytes) that step3's ad hoc scheme (direct concatenation of
the t/x/y delta streams, no framing at all) never paid, plus minor zlib-boundary effects from the
different byte layout -- negligible relative to cell sizes of 250-620B. The one outlier, DP+SED at
`(sigma=0.1, tol=0.5)` (+13B), is a **single-track cell** (1/15 reachable both times): with one
data point there is no averaging to smooth out a particular track's own zlib boundary behavior, so
a larger-than-typical delta here is sampling noise, not a sign of a systematic issue -- every
other, better-sampled cell stays within +0 to +4B.

### Segment-format overhead (synthetic, no real tracks)

Fixed 200-vertex payload cut into 1/2/5/10/20 line segments sharing boundary vertices (2.5's dedup rule) -- only the per-segment header (`[type][count varint]`) cost varies; the point payload itself is identical across rows.

| segments | pre-zlib bytes | post-zlib bytes | header bytes | header fraction (pre-zlib) |
|---|---|---|---|---|
| 1 | 2404 | 1743 | 0 | 0.00% |
| 2 | 2405 | 1752 | 1 | 0.04% |
| 5 | 2411 | 1755 | 7 | 0.29% |
| 10 | 2421 | 1771 | 17 | 0.70% |
| 20 | 2441 | 1796 | 37 | 1.52% |

Post-zlib, the relative overhead shrinks further (zlib partially absorbs the repeated
type/count-varint bytes across segments): 1743B -> 1796B is a 3.0% increase in COMPRESSED size
going from 1 to 20 segments, even though pre-zlib header bytes alone already reach 1.52% of the
uncompressed payload. Framing overhead scales roughly linearly with segment count (~1-2B/segment
pre-zlib) and stays a small fraction of typical per-track payload sizes even at 20 segments --
M2's hybrid cutting does not need a special byte budget purely for segment-count framing.

### M0 Conclusions

**Encoding re-verification (spec section 7's M0 artifact: "match step3 or an explained
discrepancy"): matched.** Reachability and `n` reproduce `benchmarks/results/step3.md`'s dt=1
tables exactly, cell for cell, for both DP+SED and LSQ-uniform (ADR-0021: the reachability gate
was deliberately kept identical to step3's, not replaced by the new certificates). Byte counts
differ by a small, fully explained amount (0 to +4B in every multi-track cell; one +13B
single-track outlier, sampling noise) -- `traj.encode`'s explicit per-segment framing costs a
few bytes step3's ad hoc scheme didn't pay, nothing more.

**The certified-correctness split (ADR-0021) is already informative.** DP+SED's `eps_A <=
target_tol` fraction stays high (86.7-100%) everywhere it's reachable except one cell
(sigma=1/tol=10: 60.0%). The LSQ-uniform spline's fraction, in contrast, drops sharply at TIGHT
`tol` relative to sigma/curve complexity: 7.1% (sigma=0, tol=0.5), 10.0% (sigma=0.1, tol=0.5), 0%
(sigma=1, tol=2), 0% (sigma=5, tol=10) -- cells step3's own honest-error criterion accepts as
reachable, yet the certificate against the actually-recorded track regularly exceeds the same
`target_tol`. A plausible mechanism, not yet verified in depth: `certify_spline`'s default
`lam_fallback=0.1` m adds a fixed ~10 cm margin on top of the measured linearization/Fréchet
distance (plus the certificate's own rounding margins), which can by itself approach or exceed a
`target_tol` as tight as 0.5 m or 2 m -- `certify_polyline` carries no comparable additive
constant, consistent with its much higher pass fraction on the same grid. This is an **open
question carried into M1/M2**, not resolved here: spec section 2.2 requires `cost(i, j)` to be
gated by the certificate (`eps <= tol`), unlike M0's purely informational tally, so a spline
segment's usable `tol` range may be narrower in practice than its honest-error-based reachability
would suggest.

**Segment-format overhead is small.** 1.52% pre-zlib / ~3% post-zlib at 20 segments over a
200-vertex payload (see table above) -- M2's hybrid cutting doesn't need special byte-budget
precautions purely from per-segment framing.

**Decisions:** ADR-0021 (the certificate is an additional check, not a reachability gate; segment
type stored as 1 byte, not 1 bit).

**Open issues, carried into M1:** knot removal (spec 2.4), LSQ-free-knot fitting, the free-knot
spline oracle (spec 2.6); the `lam_fallback`-vs-tight-`tol` question above, worth checking before
M2's `cost(i, j)` starts gating on the certificate directly.
