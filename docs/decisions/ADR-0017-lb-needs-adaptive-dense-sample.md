# ADR-0017: `hausdorff_lower_bound`'s dense spline sample must be adaptive, not fixed-count-uniform

- Status: Accepted
- Date: 2026-09-18

## Context

M2's first full-corpus run (`benchmarks/step7_m2_projection.py`) reported 12 tracks where
`eps_A(combined) < LB` -- a certificate that would be a *wrong* upper bound relative to
`hausdorff_lower_bound`, a genuine lower bound on `d_F`. This is a serious correctness signal
(unlike a density/fallback-rate finding) and was investigated before writing any report.

`LB` was computed as `hausdorff_lower_bound(A, bs(dense_u))`, `dense_u` a *fixed-count uniform*
sample (`max(500, 3*len(A))` points across the spline's domain). Investigated the worst case
directly (track index 204, gap 3.3 m): the certified linearization `Lin(A')` used by the 2.4
fallback (`certify_spline_linearization`, `lam=0.1`) is genuinely certified -- but
`hausdorff_lower_bound(Lin(A'), dense_u_sample)` was 3.59 m, when it should be `<= lam = 0.1` if
the uniform sample were a faithful stand-in for the true spline. Increasing the uniform sample
from 708 to 2000 points dropped that gap from 3.59 to 0.59 -- confirming the uniform sample was
simply *too coarse* to resolve a locally curvy stretch of this specific spline, not that
`certified_linearize` was wrong. Using the near-exact reference's own vertices (`certified_linearize`
at `lam=1e-4`, already computed for S2's own acceptance criterion -- 10,011 adaptively-placed
vertices for this track) as the dense sample instead: `LB = 7.84`, correctly `<= eps_A = 7.95`.

A second problem surfaced alongside: `traj.certify.hausdorff_lower_bound`'s pure-Python
`O(n*m)` loop is fine for M1's occasional, small-scale informational use, but infeasible at the
scale a genuinely dense sample requires here (`m` up to ~10,000) -- a single call did not
complete in 60 s.

## Decision

1. **Use the near-exact-linearization reference's own vertices as the dense spline sample for
   `LB`**, not a fixed-count uniform sample. These vertices are adaptively placed (concentrated
   exactly where the curve is complex, by the same monotonicity-driven subdivision
   `certified_linearize` already uses) and are *already computed* for S2's acceptance
   criterion and the `eps_A(2.3)/reference` density metric -- reusing them costs nothing extra
   and is provably tighter than any fixed uniform count, since the adaptive placement responds
   to the specific spline's own curvature rather than guessing a global sample density.
2. **A local, vectorized reimplementation** (`_fast_hausdorff_lower_bound` in
   `benchmarks/step7_m2_projection.py`, numpy broadcasting over all points and all polyline
   segments) makes this tractable at the required scale -- verified to match
   `traj.certify.hausdorff_lower_bound` exactly on a real case (`7.846254128101977` both ways).
   `traj.certify.hausdorff_lower_bound` itself is unchanged (M1's own use of it stays small-scale
   and correct as-is); this is a benchmark-script-local performance fix, not a library change.
   **First version OOM-killed the whole background run**: broadcasting the full `(n, m, 2)`
   array at once is unbounded when both the track (`n`, up to ~1500+ points for real GeoLife
   tracks) and the near-exact reference (`m`, up to ~10,000+ vertices for some splines) are
   large simultaneously. Fixed by processing `points` in memory-bounded batches (batch size
   chosen so `batch_size * m` stays under a fixed cell cap, `_BATCH_MAX_CELLS`) -- verified on
   the single largest track in the corpus (~1500+ points): peak RSS 482 MB, no OOM.

The full corpus run was re-executed after this fix (the previous run's `eps_A`/fallback-rate/
correspondence-check numbers, which don't depend on `LB` at all, were unaffected and did not need
re-verification -- only `LB`, `eps_A/LB`, and the S1-LB-violation count needed recomputing).

## Consequences

- A future use of `hausdorff_lower_bound` (or any lower-bound density metric) against a spline
  at full-corpus scale should default to an adaptively-placed dense sample (e.g. a fine
  `certified_linearize` call) rather than a fixed uniform count, and should use a vectorized
  point-to-polyline distance computation if the sample can be large.
- This is a benchmark-methodology fix, not a certificate-correctness fix -- `certify_spline_projection`/
  `certify_spline_linearization` themselves were never wrong; the bug was entirely in how this
  script measured the informational `LB` comparison against them.

## Links

`benchmarks/step7_m2_projection.py`'s `_near_exact_reference`/`_fast_hausdorff_lower_bound`;
`src/traj/certify.py`'s `hausdorff_lower_bound` (unchanged); `benchmarks/results/step7.md` (M2
section).
