# ADR-0020: a degenerate time array silently breaks SED simplification (M4's trade-off curve)

- Status: Accepted
- Date: 2026-09-19

## Context

M4's trade-off curve (`benchmarks/results/step7.md`) reported certificate size as nearly
insensitive to the DP+SED simplification tolerance `tol`: 717.5 bytes/track at `tol=1` vs. 616.0
bytes/track at `tol=20` -- only a 16% spread across a 20x change in `tol`. This directly
contradicted step1's own compression finding (`benchmarks/results/step1.md` section 5): DP+SED on
a comparable GeoLife sample goes from ~999 bytes (int32, `tol=2`) to ~207 bytes (`tol=20`), a
~4.8x spread. Investigated before writing anything further, per instruction.

## Investigation

`benchmarks/step7_m4.py`'s `quantized_bytes` was confirmed to encode the right thing: `kept`, the
DP+SED-simplified vertices returned by `_polyline_repr`, not the original points -- that specific
hypothesis was checked and ruled out directly (`_polyline_repr` returns `xy[kept_idx]`,
`quantized_bytes` is called on exactly that array).

The real bug was one level upstream: `_polyline_repr` built its synthetic `Track` object with
`t=np.zeros(len(xy))` -- a **constant, fake time array** -- before calling
`simplify_sed_with_indices`. SED ("synchronized Euclidean distance", `src/traj/simplify.py`'s
`_seds`) is explicitly time-aware: for each candidate split point, `frac = (t[i]-t0)/(t1-t0) if
t1>t0 else 0`. With a constant `t` (t1<=t0 for every sub-range), `frac` collapses to `0`
*everywhere*, so every "synchronized position" collapses to the piece's own start point
`xy[i0]` -- the algorithm effectively measures distance-from-the-segment-start instead of a
genuine time-synchronized interpolation, which is large for most real points and triggers the
`d[j] > tol` split condition far more often than genuine SED would, at any `tol`. This causes
systematic vertex **over-retention**, nearly independent of `tol` -- exactly the observed anomaly.

Confirmed directly: reproducing step1's exact sample (`N_SUBSET=60`, seed=42) with the real,
unmodified `simplify_sed_with_indices` reproduces step1's own numbers exactly (83.2 -> 17.2 mean
vertices, `tol=2` -> `tol=20`) -- the *algorithm* is correct; only M4's synthetic-track
construction fed it a broken input.

### Audit: other constructed-`Track`/synthetic-time call sites

Searched `src/` and `benchmarks/` for `Track(...)` construction and any `t=`/time-array
placeholders:

| Site | `t` source | Risk |
|---|---|---|
| `src/traj/clean.py`, `src/traj/io.py` | real parsed/cleaned GPS timestamps | none |
| `benchmarks/step7_query.py`'s `build_representations` | the real `t` passed in by its own caller (used throughout M1-M3's actual corpus) | none -- already correct |
| `benchmarks/step7_certify.py`, `step7_m13_tail.py`, `step1_compression.py` | call `simplify_sed_with_indices` directly on real `load_clean_tracks` tracks | none |
| `benchmarks/step3_decisive.py` (calls `simplify_sed_with_indices`) | `t = np.arange(0.0, duration, dt)` -- a genuine, increasing synthetic time axis | none |
| `benchmarks/step1_kinematics.py`, `step2_crossover.py`, and every synthetic-track builder in `tests/traj/*.py` | `t` built via `linspace`/cumulative `dt` -- genuine, increasing | none (and none of these call `simplify_sed_with_indices`) |
| `benchmarks/step7_m4.py`'s `_polyline_repr` (pre-fix) | `np.zeros(len(xy))` | **the bug** -- the only call site with a degenerate placeholder |

Only `step7_m4.py`'s trade-off curve was affected -- M1/M2/M3's own polyline representations
(`step7_query.py`) always used the real `t`, so their results are unaffected by this bug.

Two *other* time-dependent functions were checked and deliberately **not** given a new hard-fail
guard:
- `traj.spline._param_u`'s `mode="time"` branch already has an explicit, documented, tested
  fallback for zero span (`np.linspace(0,1,len(t))`) -- an intentional design choice (not an
  accidental silent failure), predating this investigation and unrelated to it.
- `spline_lsq.py`'s `make_lsq_spline`-based fitters require strictly increasing knots; `scipy`
  itself raises on a non-conforming `t` (the Schoenberg-Whitney condition) -- already protected
  by the library, no redundant guard needed.

## Decision

Added `traj.simplify._validate_time(t, context)`, called at the top of
`simplify_sed_with_indices` (the same class of fix as ADR-0014's `dense_max_error` domain guard):
raises `ValueError` for a zero-or-negative overall span (`t[-1] <= t[0]`) or a sequence that is
not non-decreasing (`np.diff(t) < 0` anywhere) -- both make SED's "distance interpolated by time
fraction" meaningless, not just imprecise. Exempted for `len(t) <= 2` (a track that short never
reaches `_seds` at all: the recursion's `i1 - i0 < 2` guard skips it immediately, so a degenerate
`t` there is genuinely harmless, not silently wrong).

Fixed `_polyline_repr` to take and use the real per-track `t` (threaded through
`build_tradeoff_sample`, reusing a near-duplicate's *source* track's own timestamps, matching
`step7_query.py`'s existing, already-correct convention).

Regression tests (`tests/traj/test_simplify.py`): a constant `t` and a non-monotonic `t` must
both raise (verified to fail on the pre-guard code); a genuine increasing `t` and a harmless
2-point degenerate case must both still work.

## Consequences

- M4's trade-off curve is recomputed with the fix (`benchmarks/results/step7.md`'s M4 section,
  updated in the following commit) -- the corrected numbers show the expected strong `tol`
  sensitivity, consistent with step1's own finding.
- M1/M2/M3's own results and conclusions are **unaffected** -- confirmed by the audit above, not
  assumed.
- Any future caller of `simplify_sed_with_indices` with a placeholder time array now gets a clear
  `ValueError` naming the actual problem, instead of a silently-wrong-but-plausible-looking vertex
  count.

## Links

`src/traj/simplify.py`; `tests/traj/test_simplify.py`; `benchmarks/step7_m4.py`; ADR-0014 (the
same "fail loudly on a domain/precondition mismatch" pattern, for `dense_max_error`);
`benchmarks/results/step1.md` section 5 (the reference numbers this discrepancy was checked
against); `benchmarks/results/step7.md` (M4 section, corrected).
