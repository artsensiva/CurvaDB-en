# ADR-0014: `dense_max_error`'s parametrization-domain bug (ADR-0010/ADR-0012's numbers invalid)

- Status: Accepted
- Date: 2026-09-17

## Context

`docs/reviews/step7_M1_2.md` accepted M1.2's process but rejected its conclusions. While
reading `spline.py`/`spline_lsq.py` to plan M1.3's fitter switch (ADR-0013), a further problem
surfaced: **`benchmarks/step7_certify.py`'s `fit_validity()` calls
`dense_max_error(track.t, track.xy, fit.tck, mode="time")` on `spline_lsq.py` fits, and that call
is wrong.**

`traj.spline.dense_max_error` (via `_dense_scan`/`_param_u`) assumes the given `tck`'s knots are
parametrized the same way `_param_u(t, xy, mode)` would compute `u` -- true for `spline.py`'s own
`SplineFit` (built with exactly that normalization via `splprep`), but **not** for
`spline_lsq.py`'s `LsqSplineFit`: `make_lsq_spline(t, xy, full_knots, k=k)` fits directly on real
`t` (`full_knots = concatenate([full(k+1,t_min), internal_knots, full(k+1,t_max)])`), so its knots
span real seconds, not `u ∈ [0,1]`. Calling `dense_max_error(..., mode="time")` on such a `tck`
internally computes `u0 = (t - t_min)/span ∈ [0,1]` and evaluates the spline at those `u0`
values -- a tiny sliver near the true domain's start, not across the real track.

## Audit

Grepped every `dense_max_error`/`dense_check`/`_param_u` call site project-wide:

| File | Usage | Domain-consistent? |
|---|---|---|
| `src/traj/spline.py` | definitions; `fit()`'s internal bisection calls `_dense_scan` with its own `parametrization` | yes (self-consistent by construction) |
| `benchmarks/step7_certify.py` | `fit_validity()` and `run_s2()`, called on `spline_lsq.py` fits with `mode="time"` | **no -- the bug** |
| `benchmarks/step2_fitpack.py` | `min_uniform_knots()` fits `make_lsq_spline` directly on `_param_u(t,xy,mode)`-computed `u`, then calls `dense_max_error(t, xy, tck, mode=mode)` | yes (fits on the same `u` it later checks against) |
| `benchmarks/step1_spline_fit.py`, `tests/traj/test_spline_dense.py` | `dense_check()` with `spline.py`'s own `SplineFit` objects | yes (always self-consistent) |
| `benchmarks/step2_crossover.py` | uses `_param_u` for its own reconstruction helpers, never calls `dense_max_error` | not applicable |

Only `step7_certify.py` mixes domains. The actual certification results (`n_pass`/`eps_A` in
`run_s2`) are **not** affected by this bug -- `certify_spline_linearization` operates on the
real-domain `BSpline` object directly (via `bezier_segments`), independent of this
parametrization convention. Only the invalid/valid classification (ADR-0010's dense-deviation
check) and its reported deviation numbers are wrong.

Verified impact on a 60-track sample (`fit_adaptive`, `tol=5.0`, ADR-0010's `10x`/`50m`
threshold): buggy invalid rate **54/60 (90%)**, corrected (real-domain) invalid rate **11/60
(18.3%)**. Per-track direction isn't even consistent -- e.g. track 0: buggy deviation 179 m vs.
correct 4.4 m (falsely invalid); track 1: genuinely bad either way, ~1e12 m (a real unstable
fit, unaffected by which domain is used). **ADR-0010's reported 93.8% and ADR-0012's 17.9%
invalid-fit rates, and their deviation-distribution statistics, are therefore largely
measurement artifacts of this bug**, not a reliable finding about `spline_lsq.py`'s fitters.

## Options considered

- **Add a `"raw"` mode to `_param_u`/`dense_max_error`** (returns `t` unchanged, for a tck already
  parametrized in real coordinates) -- minimal, reuses the existing `_dense_scan` machinery,
  matches `spline_lsq.py`'s actual convention exactly. Chosen.
- **A separate dense-check function duplicated inside `spline_lsq.py`** -- avoids touching shared
  code, but duplicates `_dense_scan`'s point-to-segment logic and its own bugs independently.
  Rejected.
- **Reparametrize `spline_lsq.py` itself to fit on normalized `u`** -- bigger, riskier change to
  a module with its own established contract (direct residual at real `t` samples,
  `docs/prompts/step3.md`); not needed, since `eps_A`/certification never depended on this
  convention in the first place.

## Decision

Added `mode="raw"` to `traj.spline._param_u` (returns `t` unchanged). Also added
`_check_tck_domain(tck, t, mode)`, called from `_dense_scan` before every evaluation: for
`mode in ("time", "chord")`, the tck's knots must span `[0, 1]` (tol `1e-9`); for `mode="raw"`,
they must span `[t.min(), t.max()]` (tol `1e-6 * (t.max()-t.min())`); any mismatch raises
`ValueError` with a message naming the fix. This catches the *bug class*, not just this one
instance -- a future caller making the same mistake gets a clear exception instead of a silently
wrong number. Verified it never fires for internally-consistent calls (`spline.fit()`'s own
knots are exactly `_param_u`'s `[0,1]` by `splprep`'s clamped-knot convention;
`spline_lsq`'s are exactly `[t[0], t[-1]]` by `_full_knot_vector`'s construction) --
`venv/bin/pytest tests/ -q` passes unchanged.

`benchmarks/step7_certify.py`'s `fit_validity()`/`run_s2()` gained a `dense_mode` parameter
(default `"raw"`, matching the module's default fitter `fit_adaptive`); the new `spline.fit()`
variant (ADR-0013) passes `dense_mode="time"`.

Regression tests: `tests/traj/test_spline_dense.py::test_dense_max_error_rejects_mismatched_domain`
(an LSQ fit passed to `dense_max_error(..., mode="time")` -- the exact mistake `fit_validity()`
made -- must raise `ValueError`; verified this test fails on the pre-fix code and passes on the
fix) and `tests/traj/test_step7_certify.py::test_fit_validity_accepts_good_fits_both_fitters`
(a noiseless smooth curve's `fit_adaptive`/`fit_uniform` fits must be reported valid under the
fixed check with `dense_mode="raw"` -- confirms the fix doesn't just stop crashing, it correctly
accepts a fit it should accept).

## Consequences

- ADR-0010's *thresholds* (the `10x`/`100x` multiplier policy) are unaffected and stay unchanged
  -- they were never the bug. Its *reported numbers* (93.8% invalid, the deviation distribution)
  are unreliable as measurements; a note is added to ADR-0010 pointing here. The base `tol` value
  those multipliers apply to changes separately, for an unrelated reason (ADR-0013: unifying the
  tol across fitters).
- ADR-0012's entire premise -- a sensitivity analysis computed from the same broken measurement --
  is unsound, not merely "conservative." **Status: Superseded by ADR-0014.**
- `benchmarks/results/step7.md`'s M1.2 numbers stay in the report (not deleted), each row
  annotated "invalid: measurement bug, see ADR-0014."
- Corrected full-corpus numbers for both fitters are produced in M1.3 (ADR-0013,
  `benchmarks/results/step7.md`'s M1.3 section).

## Links

`docs/reviews/step7_M1_2.md`; ADR-0010 (annotated); ADR-0012 (superseded); ADR-0013 (the fitter
comparison this fix enables); `benchmarks/results/step7.md` (M1.3 section);
`tests/traj/test_spline_dense.py`; `tests/traj/test_step7_certify.py`.
