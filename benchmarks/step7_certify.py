"""Step7 M1: full-corpus S1/S2 validation (density and correctness of certificates,
spec docs/specs/step7_B_certified_store.md section 8), per spec section 5's file
layout -- benchmarks/*.py for full-corpus runs, tests/traj/test_certify.py for the
fast property tests.

S1 (polylines, spec section 2.2): on ALL cleaned GeoLife tracks (the project's
standard n=200 corpus, not a further subsample), certify_polyline against a
per-piece mpmath reference (chain vs. its own single segment, dps=40, tol=1e-6);
criterion eps_A >= reference - 1e-6. Also reports median eps_A/LB (LB = directed
Hausdorff both ways) -- a preliminary S3 density estimate for gate G1.

S2 (splines via section 2.4): a 40-track pilot first (wall time + peak memory
logged), then the full corpus. Criterion eps_A >= d_F(A, Lin(A',1e-4)) - 1e-4,
and eps_A finite (docs/reviews/step7_M0.md item 0.4) -- tracks whose reference
itself can't be certified even after escalating max_levels are reported as
"reference unavailable" (not a pass).

Run: venv/bin/python benchmarks/step7_certify.py
"""

from __future__ import annotations

import os
import sys
import time
import tracemalloc

import numpy as np
from scipy.interpolate import BSpline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests", "traj"))
sys.path.insert(0, os.path.dirname(__file__))

from traj.bezier import bezier_segments, de_casteljau_split, derivative_control_points  # noqa: E402
from traj.certify import _bbox_diagonal, certified_linearize, certify_polyline, certify_spline_linearization, hausdorff_lower_bound  # noqa: E402
from traj.clean import load_clean_tracks  # noqa: E402
from traj.frechet_cont import _ROLLING_THRESHOLD, distance_upper  # noqa: E402
from traj.simplify import simplify_sed_with_indices  # noqa: E402
from traj.spline import dense_max_error  # noqa: E402
from traj.spline_lsq import fit_adaptive, fit_uniform  # noqa: E402
from _frechet_cont_mpmath import distance_mp  # noqa: E402
from _report_utils import upsert_section  # noqa: E402

SEED = 42
N_TRACKS = 200  # the project's standard GeoLife corpus scale (traj.io.DEFAULT_N_TRACKS)
DP_SED_TOL = 10.0

S1_ETA = 1e-6  # tight, to match the mpmath reference's own tol -- not spec's eta=1mm default
S1_REF_TOL = 1e-6
S1_REF_DPS = 40
S1_SLACK = 1e-6

S2_FIT_TOL = 5.0
S2_LAM = 0.1
S2_ETA = 1e-3
S2_MAX_LEVELS = 12
S2_MAX_LEVELS_ESCALATED = 20
S2_REFERENCE_LAM = 1e-4
S2_REF_SLACK = 1e-4
S2_PILOT_N = 40

# ADR-0010: fixed BEFORE the full-corpus run, not adjusted after seeing results.
S2_VALIDITY_DEVIATION_FACTOR = 10.0  # dense-grid deviation must be <= this * S2_FIT_TOL
S2_VALIDITY_CONTROL_BOUND_FACTOR = 100.0  # control points within this * track bbox diagonal

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_MD = os.path.join(RESULTS_DIR, "step7.md")
SECTION_HEADER = "## M1 -- polyline and linearization certificates"


def _bspline_from_fit(fit) -> BSpline:
    knots, c_list, k = fit.tck
    return BSpline(knots, np.column_stack(c_list), k)


def fit_validity(track, fit) -> tuple[bool, str | None]:
    """ADR-0010: a fit is invalid (excluded from S2's pass/fallback counts
    entirely) if EITHER fixed condition fails:
      1. dense-grid deviation from the track (traj.spline.dense_max_error's
         existing point-to-segment-on-a-dense-grid methodology, mode="time")
         exceeds S2_VALIDITY_DEVIATION_FACTOR * the fitter's own tol;
      2. any control point is more than S2_VALIDITY_CONTROL_BOUND_FACTOR bbox
         diagonals of the track away from the track's own first point.
    Thresholds are fixed before the run; see ADR-0010 for why and for the
    process to follow if they turn out to need revisiting.
    """
    dev = dense_max_error(track.t, track.xy, fit.tck, mode="time")
    if dev > S2_VALIDITY_DEVIATION_FACTOR * S2_FIT_TOL:
        return False, "dense_deviation"

    _, c_list, _ = fit.tck
    c = np.column_stack(c_list)
    bbox_diag = _bbox_diagonal(track.xy)
    ref = track.xy[0]
    max_ctrl_dist = float(np.max(np.hypot(*(c - ref).T)))
    if max_ctrl_dist > S2_VALIDITY_CONTROL_BOUND_FACTOR * bbox_diag:
        return False, "control_point_bound"

    return True, None


# --- M1.1 diagnostics (docs/reviews/step7_M1.md item 0) ---------------------------
#
# Frozen snapshot of the PRE-M1.1 certification algorithm (blind t=0.5 bisection,
# no root-splitting, no small-ball rule) -- deliberately NOT importing certify.py's
# private functions, since those are rewritten by M1.1. This copy exists solely to
# diagnose the old algorithm's failure mode on real tracks before the fix, and is
# not maintained afterward.


def _old_tube_and_monotone_ok(control_points: np.ndarray, radius: float) -> tuple[bool, bool]:
    """Returns (tube_ok, monotone_ok) separately (the M1 version only returned
    their conjunction) -- needed to diagnose which check is the actual blocker."""
    a, b = control_points[0], control_points[-1]
    chord_vec = b - a
    chord_len = float(np.hypot(*chord_vec))
    if chord_len < 1e-12:
        tube_ok = bool(np.all(np.hypot(*(control_points - a).T) <= radius))
        return tube_ok, True
    e = chord_vec / chord_len
    tube_ok = True
    for p in control_points:
        d = b - a
        len2 = float(d @ d)
        tt = np.clip(float((p - a) @ d) / len2, 0.0, 1.0)
        closest = a + tt * d
        if float(np.hypot(*(p - closest))) > radius:
            tube_ok = False
            break
    deriv = derivative_control_points(control_points)
    monotone_ok = bool(np.all(deriv @ e >= -1e-9 * max(chord_len, 1.0)))
    return tube_ok, monotone_ok


def _old_certify_segment_diag(seg, lam, levels_left, t_lo, t_hi, depth, state) -> bool:
    tube_ok, monotone_ok = _old_tube_and_monotone_ok(seg, lam)
    state["max_depth"] = max(state["max_depth"], depth)
    if tube_ok and monotone_ok:
        return True
    if levels_left == 0:
        if state["first_failure"] is None:
            a, b = seg[0], seg[-1]
            chord_len = float(np.hypot(*(b - a)))
            time_span = t_hi - t_lo
            reason = "+".join(([] if tube_ok else ["tube"]) + ([] if monotone_ok else ["monotonicity"]))
            state["first_failure"] = {
                "reason": reason,
                "chord_len": chord_len,
                "avg_speed": chord_len / time_span if time_span > 0 else float("nan"),
            }
        return False
    left, right = de_casteljau_split(seg, 0.5)
    t_mid = 0.5 * (t_lo + t_hi)
    ok_left = _old_certify_segment_diag(left, lam, levels_left - 1, t_lo, t_mid, depth + 1, state)
    ok_right = _old_certify_segment_diag(right, lam, levels_left - 1, t_mid, t_hi, depth + 1, state)
    return ok_left and ok_right


def _old_certified_linearize_diag(bs: BSpline, lam: float, max_levels: int) -> dict:
    segments = bezier_segments(bs)
    t = np.asarray(bs.t, dtype=float)
    k = bs.k
    interior = np.unique(t[k + 1 : len(t) - k - 1])
    breaks = np.concatenate([[t[k]], interior, [t[-k - 1]]])
    state = {"max_depth": 0, "first_failure": None}
    fully_certified = True
    for j, seg in enumerate(segments):
        ok = _old_certify_segment_diag(seg, lam, max_levels, breaks[j], breaks[j + 1], 0, state)
        fully_certified = fully_certified and ok
    return {"fully_certified": fully_certified, **state}


def diagnose_pre_fix(tracks) -> dict:
    """Item 0: eps_A/LB distribution for polylines (S1, fast, no mpmath needed),
    per-failing-track tube-vs-monotonicity diagnosis + max recursion depth (S2,
    old algorithm), and rolling-DP usage count. Run BEFORE any certify.py changes."""
    print("\n=== M1.1 diagnostics (pre-fix) ===", flush=True)

    ratios = []
    for tr in tracks:
        _, kept = simplify_sed_with_indices(tr, tol=DP_SED_TOL)
        if len(kept) < 2:
            continue
        eps_A = certify_polyline(tr.xy, kept, eta=S1_ETA)
        lb = hausdorff_lower_bound(tr.xy, tr.xy[kept])
        if lb > 1e-9:
            ratios.append(eps_A / lb)
    ratios = np.array(ratios)
    eps_lb = {
        "p50": float(np.percentile(ratios, 50)),
        "p90": float(np.percentile(ratios, 90)),
        "p99": float(np.percentile(ratios, 99)),
        "max": float(np.max(ratios)),
        "frac_exactly_1": float(np.mean(np.isclose(ratios, 1.0, atol=1e-9))),
        "n": len(ratios),
    }
    print(f"  eps_A/LB: p50={eps_lb['p50']:.4f} p90={eps_lb['p90']:.4f} p99={eps_lb['p99']:.4f} "
          f"max={eps_lb['max']:.4f} frac==1.0={eps_lb['frac_exactly_1']:.3f} (n={eps_lb['n']})", flush=True)

    failing_indices = []
    failure_details = []
    depths_all = []
    n_rolling = 0
    for idx, tr in enumerate(tracks):
        try:
            fit = fit_adaptive(tr, tol=S2_FIT_TOL)
            bs = _bspline_from_fit(fit)
        except (ValueError, np.linalg.LinAlgError):
            continue
        diag = _old_certified_linearize_diag(bs, S2_LAM, S2_MAX_LEVELS)
        depths_all.append(diag["max_depth"])
        if not diag["fully_certified"]:
            failing_indices.append(idx)
            failure_details.append(diag["first_failure"])

        # rolling-DP usage: the certificate's own distance_upper(A, lin_vertices) call
        lin_vertices, _ = certified_linearize(bs, S2_LAM, max_levels=S2_MAX_LEVELS)
        if (len(tr.xy) - 1) * (len(lin_vertices) - 1) > _ROLLING_THRESHOLD:
            n_rolling += 1

    reasons = [d["reason"] for d in failure_details]
    chord_lens = [d["chord_len"] for d in failure_details]
    speeds = [d["avg_speed"] for d in failure_details if not np.isnan(d["avg_speed"])]
    print(f"  {len(failing_indices)} failing tracks (of {len(tracks)}); failure reasons: "
          f"{ {r: reasons.count(r) for r in set(reasons)} }", flush=True)
    print(f"  failing-leaf chord length: mean={np.mean(chord_lens):.4g} max={np.max(chord_lens):.4g}", flush=True)
    if speeds:
        print(f"  failing-leaf avg speed: mean={np.mean(speeds):.4g} max={np.max(speeds):.4g} m/s", flush=True)
    print(f"  max recursion depth (old algorithm): mean={np.mean(depths_all):.2f} max={np.max(depths_all)}", flush=True)
    print(f"  tracks exceeding rolling-DP threshold (n*m > {_ROLLING_THRESHOLD}): {n_rolling}/{len(tracks)}", flush=True)

    return {
        "eps_lb": eps_lb,
        "failing_indices": failing_indices,
        "failure_reasons": reasons,
        "chord_lens": chord_lens,
        "speeds": speeds,
        "depths_old": depths_all,
        "n_rolling": n_rolling,
    }


def run_s1(tracks) -> dict:
    n_pass = 0
    n_total = 0
    ratios = []
    worst_margin = float("inf")  # min over tracks of (eps_A - reference), most negative = worst
    t0 = time.time()
    for idx, tr in enumerate(tracks):
        _, kept = simplify_sed_with_indices(tr, tol=DP_SED_TOL)
        if len(kept) < 2:
            continue
        n_total += 1
        eps_A = certify_polyline(tr.xy, kept, eta=S1_ETA)

        worst_ref = 0.0
        for k in range(len(kept) - 1):
            i0, i1 = int(kept[k]), int(kept[k + 1])
            piece = tr.xy[i0 : i1 + 1]
            segment = tr.xy[[i0, i1]]
            ref = float(distance_mp(piece.tolist(), segment.tolist(), tol=S1_REF_TOL, dps=S1_REF_DPS))
            worst_ref = max(worst_ref, ref)

        margin = eps_A - worst_ref
        worst_margin = min(worst_margin, margin)
        if margin >= -S1_SLACK:
            n_pass += 1

        lb = hausdorff_lower_bound(tr.xy, tr.xy[kept])
        if lb > 1e-9:
            ratios.append(eps_A / lb)

        if (idx + 1) % 20 == 0:
            print(f"  S1: {idx + 1}/{len(tracks)} tracks, elapsed {time.time() - t0:.1f}s", flush=True)

    return {
        "n_pass": n_pass,
        "n_total": n_total,
        "worst_margin": worst_margin,
        "median_ratio": float(np.median(ratios)) if ratios else float("nan"),
        "elapsed": time.time() - t0,
    }


def _certify_reference(bs: BSpline) -> tuple[float | None, bool]:
    """Reference distance at the near-exact linearization lam=1e-4, escalating
    max_levels if not fully certified at the default. Returns (ref_value_or_None,
    available). ref is None if unavailable (item 0.4's "reference unavailable")."""
    for max_levels in (S2_MAX_LEVELS, S2_MAX_LEVELS_ESCALATED):
        lin_vertices, fully_certified = certified_linearize(bs, S2_REFERENCE_LAM, max_levels=max_levels)
        if fully_certified:
            return lin_vertices, True
    return None, False


def run_s2(tracks, label: str, fitter=fit_adaptive) -> dict:
    n_pass = 0
    n_fallback = 0  # eps_A == inf at the certificate's own lam (or the fit itself raised)
    n_fit_error = 0  # fitter raised -- distinct from a certification-level fallback
    n_ref_unavailable = 0
    n_invalid_fit = 0  # ADR-0010: excluded from pass/fallback entirely
    invalid_reasons: list[str] = []
    dense_deviations: list[float] = []  # every successfully-fitted track, valid or not --
    control_ratios: list[float] = []    # lets us re-derive alternate thresholds without re-fitting
    n_total = 0
    t0 = time.time()
    for idx, tr in enumerate(tracks):
        n_total += 1
        try:
            fit = fitter(tr, tol=S2_FIT_TOL)
            bs = _bspline_from_fit(fit)
        except (ValueError, np.linalg.LinAlgError):
            n_fit_error += 1
            n_fallback += 1
            continue

        dev = dense_max_error(tr.t, tr.xy, fit.tck, mode="time")
        dense_deviations.append(dev)
        _, c_list, _ = fit.tck
        c = np.column_stack(c_list)
        bbox_diag = _bbox_diagonal(tr.xy)
        control_ratios.append(float(np.max(np.hypot(*(c - tr.xy[0]).T))) / max(bbox_diag, 1e-9))

        valid, reason = fit_validity(tr, fit)
        if not valid:
            n_invalid_fit += 1
            invalid_reasons.append(reason)
            continue

        eps_A, cert_ok = certify_spline_linearization(tr.xy, bs, S2_LAM, eta=S2_ETA, max_levels=S2_MAX_LEVELS)
        if not cert_ok:
            n_fallback += 1
            continue

        ref_vertices, ref_available = _certify_reference(bs)
        if not ref_available:
            n_ref_unavailable += 1
            continue

        reference = distance_upper(tr.xy, ref_vertices, tol=1e-6)
        if eps_A >= reference - S2_REF_SLACK:
            n_pass += 1

        if (idx + 1) % 20 == 0:
            print(f"  S2 ({label}): {idx + 1}/{len(tracks)} tracks, elapsed {time.time() - t0:.1f}s", flush=True)

    return {
        "n_pass": n_pass,
        "n_fallback": n_fallback,
        "n_fit_error": n_fit_error,
        "n_ref_unavailable": n_ref_unavailable,
        "n_invalid_fit": n_invalid_fit,
        "invalid_reasons": invalid_reasons,
        "dense_deviations": dense_deviations,
        "control_ratios": control_ratios,
        "n_total": n_total,
        "elapsed": time.time() - t0,
    }


def main() -> None:
    print("Loading and cleaning tracks...", flush=True)
    tracks, clean_stats = load_clean_tracks(n=N_TRACKS, seed=SEED)
    print(f"{len(tracks)} cleaned track segments from {clean_stats.n_tracks_in} raw tracks.", flush=True)

    print("\n=== S1: polyline certificates (all cleaned tracks) ===", flush=True)
    s1 = run_s1(tracks)
    print(f"S1 done: {s1['n_pass']}/{s1['n_total']} pass, worst margin {s1['worst_margin']:.3e}, "
          f"median eps_A/LB {s1['median_ratio']:.3f}, {s1['elapsed']:.1f}s", flush=True)

    print("\n=== S2 pilot: spline certificates via linearization (40 tracks) ===", flush=True)
    tracemalloc.start()
    pilot = run_s2(tracks[:S2_PILOT_N], label="pilot")
    _, pilot_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Pilot done: {pilot['n_pass']}/{pilot['n_total']} pass, {pilot['n_fallback']} fallback, "
          f"{pilot['n_ref_unavailable']} reference-unavailable, {pilot['elapsed']:.1f}s, "
          f"peak Python-tracked memory {pilot_peak / 1e6:.1f} MB", flush=True)

    print("\n=== S2: spline certificates via linearization (all cleaned tracks) ===", flush=True)
    s2 = run_s2(tracks, label="full")
    print(f"S2 done: {s2['n_pass']}/{s2['n_total']} pass, {s2['n_fallback']} fallback, "
          f"{s2['n_ref_unavailable']} reference-unavailable, {s2['elapsed']:.1f}s", flush=True)

    lines = [
        SECTION_HEADER,
        "",
        f"Corpus: `load_clean_tracks(n={N_TRACKS}, seed={SEED})` -> "
        f"{len(tracks)} cleaned track segments from {clean_stats.n_tracks_in} raw tracks "
        "(the project's standard corpus, per `traj.io.DEFAULT_N_TRACKS`).",
        "",
        "| Criterion | Threshold | Actual | Passed |",
        "|---|---|---|---|",
        f"| S1: polyline certificates (`certify_polyline`, eta={S1_ETA}) vs. mpmath "
        f"reference (per piece, dps={S1_REF_DPS}, tol={S1_REF_TOL}) | "
        f"`eps_A >= reference - {S1_SLACK}` for 100% of tracks | "
        f"{s1['n_pass']}/{s1['n_total']} (worst margin {s1['worst_margin']:.3e}) | "
        f"{'yes' if s1['n_pass'] == s1['n_total'] else 'no'} |",
        f"| S2: spline certificates via certified linearization (`certify_spline_linearization`, "
        f"lam={S2_LAM}, eta={S2_ETA}) vs. near-exact reference (lam={S2_REFERENCE_LAM}) | "
        f"`eps_A >= reference - {S2_REF_SLACK}`, `eps_A` finite, for 100% of tracks with an "
        f"available reference | {s2['n_pass']}/{s2['n_total']} ({s2['n_fallback']} fallback "
        f"[uncertified linearization at lam={S2_LAM}], {s2['n_ref_unavailable']} reference-unavailable) | "
        f"{'yes' if s2['n_pass'] == s2['n_total'] - s2['n_fallback'] - s2['n_ref_unavailable'] else 'no'} |",
        "",
        f"**Informational (not a gate):** median `eps_A/LB` for polylines (LB = directed Hausdorff "
        f"distance both ways, spec section 2.5) = **{s1['median_ratio']:.3f}** -- a preliminary "
        "reading of spec S3's density criterion (median <= 1.2 for polylines) for gate G1; S3 itself "
        "is not an M1 gate and is not enforced here.",
        "",
        f"**S2 pilot** (first {S2_PILOT_N} tracks, measured separately before the full run): "
        f"{pilot['elapsed']:.1f}s wall time, peak Python-tracked memory {pilot_peak / 1e6:.1f} MB "
        f"(`tracemalloc`). Extrapolated linearly to the full {len(tracks)}-track corpus: "
        f"~{pilot['elapsed'] / S2_PILOT_N * len(tracks):.0f}s.",
        "",
        f"**Fallback / reference-unavailable tracks are a real, expected outcome** (spec section 10's "
        f"acknowledged risk: spline loops relative to a chord break the monotonicity test at any "
        f"subdivision depth) -- not a bug. `certify_spline_linearization` returns `(inf, False)` for "
        f"these (item 0.4), and they are excluded from the S2 pass/fail count, not silently treated "
        f"as passes.",
        "",
        f"`decide()`/`decide_conservative()`'s rolling-row kernels (item 4) were exercised "
        f"transparently wherever a track's `n * m` (original vs. linearized-spline segment count) "
        f"exceeded 5,000,000 during S2 -- no separate accounting needed, both code paths are "
        f"verified equivalent (`test_rolling_dp_matches_full_dp`).",
        "",
    ]
    body = "\n".join(lines) + "\n"
    upsert_section(OUT_MD, SECTION_HEADER, body)
    print(f"\nwrote {OUT_MD}", flush=True)


# --- M1.1 post-fix re-validation (docs/reviews/step7_M1.md item 3) ---------------


def _new_certify_segment_depth(seg, lam, levels_left, depth, state) -> bool:
    """Mirrors the actual (post-fix) traj.certify._certify_segment, instrumented
    to track max recursion depth -- for the before/after depth comparison on the
    46 formerly-failing tracks. Imports the real _certified_ok/root-finding AND
    the real _BOUNDARY_MARGIN constant (not a frozen copy, unlike the pre-fix
    diagnostics): this inspects the final, maintained algorithm, so staying in
    sync with it is correct, not a liability -- an earlier version of this
    duplicated the split-point selection without the boundary-margin guard,
    silently going stale the moment that guard was added to fix a regression
    (docs/reviews/step7_M1.md item 3's re-validation caught it: pass/fail counts
    from the real certify_spline_linearization were correct throughout, only
    this separate depth statistic was wrong).
    """
    from traj.certify import _BOUNDARY_MARGIN, _certified_ok, _projection_roots_in_unit_interval

    state["max_depth"] = max(state["max_depth"], depth)
    if _certified_ok(seg, lam):
        return True
    if levels_left == 0:
        return False
    a, b = seg[0], seg[-1]
    chord_len = float(np.hypot(*(b - a)))
    split_t = 0.5
    if chord_len >= 1e-12:
        e = (b - a) / chord_len
        roots = _projection_roots_in_unit_interval(derivative_control_points(seg), e)
        for r in roots:
            if _BOUNDARY_MARGIN <= r <= 1.0 - _BOUNDARY_MARGIN:
                split_t = r
                break
    left, right = de_casteljau_split(seg, split_t)
    ok_left = _new_certify_segment_depth(left, lam, levels_left - 1, depth + 1, state)
    ok_right = _new_certify_segment_depth(right, lam, levels_left - 1, depth + 1, state)
    return ok_left and ok_right


def _new_certified_linearize_depth(bs: BSpline, lam: float, max_levels: int) -> int:
    segments = bezier_segments(bs)
    state = {"max_depth": 0}
    for seg in segments:
        _new_certify_segment_depth(seg, lam, max_levels, 0, state)
    return state["max_depth"]


def rerun_targeted_46(tracks, failing_indices: list[int]) -> dict:
    print("\n=== S2 re-run: 46 former failures ===", flush=True)
    failing_tracks = [tracks[i] for i in failing_indices]
    result = run_s2(failing_tracks, label="46-recheck")
    n_now_certified = result["n_total"] - result["n_fallback"]

    depths_after = []
    for tr in failing_tracks:
        try:
            fit = fit_adaptive(tr, tol=S2_FIT_TOL)
            bs = _bspline_from_fit(fit)
        except (ValueError, np.linalg.LinAlgError):
            continue
        depths_after.append(_new_certified_linearize_depth(bs, S2_LAM, S2_MAX_LEVELS))

    print(f"  {n_now_certified}/{len(failing_indices)} now get a finite certificate at lam={S2_LAM} "
          f"(vs. 0/{len(failing_indices)} before); {result['n_fallback']} still fallback", flush=True)
    print(f"  recursion depth after fix: mean={np.mean(depths_after):.2f} max={np.max(depths_after)}", flush=True)
    result["depths_after"] = depths_after
    result["n_now_certified"] = n_now_certified
    return result


def rerun_sample_100(tracks, passing_indices: list[int], seed: int = 123) -> dict:
    print("\n=== S2 re-run: random 100-sample of previously-passing tracks ===", flush=True)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(passing_indices, size=min(100, len(passing_indices)), replace=False)
    sample_tracks = [tracks[i] for i in sample_idx]
    result = run_s2(sample_tracks, label="100-sample")
    print(f"  {result['n_pass']}/{result['n_total']} pass, {result['n_fallback']} fallback, "
          f"{result['n_ref_unavailable']} reference-unavailable, {result['elapsed']:.1f}s "
          f"({result['elapsed'] / len(sample_tracks):.3f}s/track)", flush=True)
    return result


def main_diagnose_only() -> dict:
    """M1.1 item 0: run just the pre-fix diagnostics (no mpmath, no full S1/S2) --
    used once before touching certify.py, and its output is folded into the M1.1
    report update by hand afterward."""
    print("Loading and cleaning tracks...", flush=True)
    tracks, clean_stats = load_clean_tracks(n=N_TRACKS, seed=SEED)
    print(f"{len(tracks)} cleaned track segments from {clean_stats.n_tracks_in} raw tracks.", flush=True)
    return diagnose_pre_fix(tracks)


if __name__ == "__main__":
    if "--diagnose" in sys.argv:
        main_diagnose_only()
    else:
        main()
