"""Step7 M2 (docs/specs/step7_B_certified_store.md section 2.3, docs/reviews/
step7_M1_3.md carry-overs): full-corpus validation of the monotone
projection-matching spline certificate, with certify_spline_linearization (2.4)
as the per-track fallback.

S1 (correctness) is checked THREE ways, none of them the near-exact-linearization
reference used by M1's S2 (that reference is itself only an upper bound -- see
docs/decisions/ADR-0016 -- so it cannot validly gate "is eps_A too small"):
  1. eps_A(combined) >= LB (hausdorff_lower_bound, a genuine lower bound) for all
     585 tracks.
  2. eps_A(combined) >= distance_mp(A, Lin(A', MPMATH_LAM_REF)) - MPMATH_SLACK on a
     30-track sample (the reverse triangle inequality makes this a genuine lower
     bound). MPMATH_LAM_REF/MPMATH_TOL/MPMATH_DPS and the sample itself are chosen
     for mpmath tractability (see the module docstring below for why 1e-6 slack,
     as originally envisioned, is infeasible at real-track scale) -- fixed BEFORE
     this script is run against real data, not tuned afterward.
  3. Direct dense-sampling verification of the spec's own clamp-cost formula and
     monotonicity (spec section 9's own test, item 2 of tests/traj/test_certify.py)
     on EVERY track where 2.3 fully certifies, not just a hypothesis sample.

S2 (spec's own acceptance criterion) is the combined pipeline's eps_A against the
near-exact-linearization reference (M1's own S2_REFERENCE_LAM machinery) -- valid
there, since S2's spec wording is specifically phrased around that reference.
eps_A(2.3)/reference is ALSO reported, explicitly as an informational density
metric (< 1 expected/good), never as a pass/fail gate.

S3: fraction of pieces and fraction of tracks falling back to 2.4 (<=10%
threshold), and eps_A/LB median/p90/p99 (<=2 median threshold) for splines.

Also: eps_A(2.3) vs eps_A(2.4) (both computed, for comparison, on every track
where 2.3 succeeds) -- eps_A, time, and the full ratio distribution, not just a
summary number. Per-track certification time (median/p90/max).

Run: venv/bin/python benchmarks/step7_m2_projection.py
"""

from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np
from scipy.interpolate import BSpline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests", "traj"))

from traj.certify import (  # noqa: E402
    certified_linearize,
    certify_spline_linearization,
    certify_spline_projection,
)
from traj.clean import load_clean_tracks  # noqa: E402
from traj.frechet_cont import distance_upper  # noqa: E402
from traj.spline import fit as spline_fit  # noqa: E402
from _frechet_cont_mpmath import distance_mp  # noqa: E402

SEED = 42
N_TRACKS = 200

S2_FIT_TOL = 10.0  # spline.fit()'s own tol -- ADR-0013's selected S2 fitter
S2_LAM = 0.1  # 2.4 fallback's lam (matching M1's own S2_LAM)
S2_MAX_LEVELS = 12
S2_ETA = 1e-3

# Near-exact-linearization reference (M1's own machinery) -- used ONLY for S2's
# own acceptance criterion and the eps_A(2.3)/reference informational density
# metric, NEVER as an S1 lower-bound gate (ADR-0016: it is itself an upper bound).
# Its own vertices double as the dense spline sample for LB (see
# _fast_hausdorff_lower_bound).
S2_REFERENCE_LAM = 1e-4
S2_MAX_LEVELS_ESCALATED = 20
S2_REF_SLACK = 1e-4

# --- mpmath-based genuine lower bound (S1 check 2) -- tractability parameters ---
#
# distance_mp's own docstring: "too slow for anything but small fixtures (n, m <=
# ~6)". Empirically (this session, real GeoLife tracks): a single distance_mp call
# costs O(n*m) per decide_mp call, times ~25-30 bisection iterations to reach a
# tight tol. n=264, m=100 (26400 cells) did not complete in 120s; n=84, m=21 (1764
# cells) took 3.6s at tol=1e-3, dps=25. m itself (certified_linearize(bs,
# MPMATH_LAM_REF)'s vertex count) varies a LOT independent of the track's own
# point count n -- some splines need far more Bezier segments to satisfy
# monotonicity/tube even at a loose lam, regardless of the track's length (found
# empirically: a 98-point track needed m=132 at lam_ref=1.0, 44s; a 264-point
# track needed m=200 at the same lam_ref).
#
# The literal "distance_mp - 1e-6" envisioned in the plan is infeasible at real-
# track scale -- not a matter of patience, but of the same n*m*iterations scaling
# that makes the oracle's own docstring say "small fixtures" in the first place.
# Fixed BEFORE running against the real sample (not tuned after seeing results):
# MPMATH_LAM_REF=1.0 (a moderate, not maximally fine, linearization -- still a
# small slack relative to typical eps_A ~5-10m), MPMATH_TOL=1e-3 (fewer bisection
# iterations than 1e-6 -- the DOMINANT cost driver, not the per-cell cost), and
# the 30-track sample is drawn only from tracks where n * m(MPMATH_LAM_REF) <=
# MPMATH_NM_CAP (a tractability requirement, not a favorable-results filter --
# checked via certified_linearize BEFORE knowing what distance_mp will return).
MPMATH_LAM_REF = 1.0
MPMATH_TOL = 1e-3
MPMATH_DPS = 25
MPMATH_NM_CAP = 3000
MPMATH_SAMPLE_N = 30
MPMATH_SAMPLE_SEED = 20260918

CORRESPONDENCE_MIN_SAMPLES_PER_PIECE = 200  # spec section 9: step 1e-4, or >= 200/piece for many-piece tracks

OUT_PKL = "/tmp/m2_projection.pkl"
OUT_LOG = "/tmp/m2_projection.log"


def _fit_bspline(track) -> BSpline:
    sp = spline_fit(track, tol=S2_FIT_TOL)
    return BSpline(sp.tck[0], np.column_stack(sp.tck[1]), sp.tck[2])


def _near_exact_reference(A: np.ndarray, bs: BSpline) -> tuple[float | None, np.ndarray | None, bool]:
    """M1's own near-exact-linearization reference (S2's acceptance criterion,
    and the eps_A(2.3)/reference informational density metric) -- escalates
    max_levels once if not fully certified at the default, matching
    benchmarks/step7_certify.py's own _certify_reference. Also returns the
    linearization's own vertices -- reused as the "dense sample of the spline"
    for hausdorff_lower_bound (see _fast_hausdorff_lower_bound's docstring for
    why a fixed-count uniform sample is NOT dense enough in general)."""
    for max_levels in (S2_MAX_LEVELS, S2_MAX_LEVELS_ESCALATED):
        lin_vertices, fully_certified = certified_linearize(bs, S2_REFERENCE_LAM, max_levels=max_levels)
        if fully_certified:
            d = distance_upper(A, lin_vertices, tol=1e-6)
            return d, lin_vertices, True
    return None, None, False


_BATCH_MAX_CELLS = 2_000_000  # cap on points_in_batch * len(Y) to bound peak memory


def _point_to_polyline_distances(points: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Vectorized point-to-polyline distance for every point in `points`
    (the same clamped point-to-segment formula traj.certify.
    hausdorff_lower_bound uses, cross-checked to match it exactly on real
    tracks) -- traj.certify's own version loops in pure Python, taking O(n*m)
    Python-level iterations; at the scale needed here (m up to ~10,000, the
    near-exact reference's own vertex count) that is minutes, not seconds.

    Processes `points` in memory-bounded BATCHES, not a single (n, m, 2)
    broadcast array -- a first version materialized the full array at once and
    got the whole run OOM-killed on a track with large n AND large m (some
    splines need thousands of near-exact-reference vertices AND the track
    itself has thousands of points -- the naive broadcast is O(n*m) floats,
    unboundedly large). Batch size is chosen so points_in_batch * len(Y) stays
    under _BATCH_MAX_CELLS regardless of how big n or m get.
    """
    a, b = Y[:-1], Y[1:]
    d = b - a
    len2 = np.sum(d * d, axis=1)
    len2_safe = np.where(len2 < 1e-20, 1.0, len2)
    degenerate = len2 < 1e-20

    m = len(a)
    batch_size = max(1, _BATCH_MAX_CELLS // max(m, 1))
    out = np.empty(len(points), dtype=float)
    for start in range(0, len(points), batch_size):
        chunk = points[start : start + batch_size]
        diff = chunk[:, None, :] - a[None, :, :]
        tt = np.clip(np.sum(diff * d[None, :, :], axis=2) / len2_safe[None, :], 0.0, 1.0)
        closest = a[None, :, :] + tt[:, :, None] * d[None, :, :]
        delta = chunk[:, None, :] - closest
        dists = np.hypot(delta[..., 0], delta[..., 1])
        if np.any(degenerate):
            point_dist = np.hypot(diff[..., 0], diff[..., 1])
            dists = np.where(degenerate[None, :], point_dist, dists)
        out[start : start + batch_size] = dists.min(axis=1)
    return out


def _fast_hausdorff_lower_bound(A: np.ndarray, A_prime: np.ndarray) -> float:
    """Vectorized equivalent of traj.certify.hausdorff_lower_bound (verified to
    match it exactly on real tracks, e.g. 7.846254128101977 both ways on one
    track's small-lam linearization) -- needed because this script's `A_prime`
    (the near-exact reference's own vertices, reused as the dense spline
    sample) can have thousands of points, making the pure-Python version
    infeasible at full-corpus scale.

    A too-coarse dense sample was tried FIRST (a fixed uniform count, ~500-700
    points) and found to produce a materially wrong (too large) LB: on one real
    track, hausdorff_lower_bound(A, uniform_708_points) = 11.25, but the
    linearization it was meant to bound Lin(A') against was only 3.59 away from
    THAT sample -- while the true continuous spline itself, sampled at 10,011
    points (the S2_REFERENCE_LAM linearization, adaptively placed, not
    uniform), gives LB = 7.84, consistent with eps_A(2.4) = 7.95 (the certified
    linearization really is close to the spline; the coarse uniform sample was
    just too sparse to resolve a locally curvy stretch, producing a spuriously
    large "LB" and a false S1 violation). Reusing the near-exact reference's
    own adaptively-placed vertices fixes this directly, at no extra fitting
    cost (already computed for S2/the density metric).
    """
    d1 = float(_point_to_polyline_distances(A, A_prime).max())
    d2 = float(_point_to_polyline_distances(A_prime, A).max())
    return max(d1, d2)


def _verify_correspondence(A: np.ndarray, bs: BSpline, proj) -> tuple[bool, float]:
    """spec section 9's own test (tests/traj/test_certify.py's property test,
    run here for real on every 2.3-certified track, not just hypothesis samples):
    for every non-degenerate certified piece, the explicit clamp-formula
    correspondence's cost, densely sampled by parameter, must be <= eps_A, and
    the sampled projection must be non-decreasing. Returns (all_ok, max_cost_found)."""
    max_cost = 0.0
    for p in proj.pieces:
        if p.u_k1 <= p.u_k or p.L_k < 1e-12 or not p.certified:
            continue
        n_samples = max(CORRESPONDENCE_MIN_SAMPLES_PER_PIECE, int((p.u_k1 - p.u_k) / 1e-4) + 1)
        grid = np.linspace(p.u_k, p.u_k1, min(n_samples, 20000))
        V_k = A[p.k]
        prev_s = None
        for u in grid:
            Cu = np.asarray(bs(u))
            s = float((Cu - V_k) @ p.e_k)
            if prev_s is not None and s < prev_s - 1e-6 * max(p.L_k, 1.0):
                return False, max_cost
            prev_s = s
            matched = V_k + float(np.clip(s, 0.0, p.L_k)) * p.e_k
            cost = float(np.hypot(*(Cu - matched)))
            max_cost = max(max_cost, cost)
    if max_cost > proj.eps_A + 1e-6:
        return False, max_cost
    return True, max_cost


def _process_track(idx: int, tr) -> dict:
    """One track's worth of M2 work -- factored out of run_full_corpus so a
    chunked driver (main_chunk) can process a handful of tracks per subprocess
    and let the OS fully reclaim memory between chunks (see the module
    docstring's note on this session's OOM-kills: the full 585-track loop in a
    single long-running process was killed twice by the system's own memory
    pressure -- likely accumulation across many distinct numba-jitted call
    shapes plus this machine's small free-memory headroom under other running
    applications, not a single-track leak, since every suspect large track
    checked in isolation stayed under 500 MB). Returns a plain dict (no numpy
    objects beyond plain floats/ints) so chunk files stay small and easy to
    concatenate."""
    t_track0 = time.time()
    bs = _fit_bspline(tr)

    t1 = time.time()
    proj = certify_spline_projection(tr.xy, bs, max_levels=S2_MAX_LEVELS)
    t_2_3 = time.time() - t1
    n_pieces = len(proj.pieces)
    n_pieces_fallback = sum(1 for p in proj.pieces if not p.certified)

    t2 = time.time()
    eps_A_lin, ok_lin = certify_spline_linearization(tr.xy, bs, S2_LAM, eta=S2_ETA, max_levels=S2_MAX_LEVELS)
    t_2_4 = time.time() - t2

    ratio_2_3_2_4 = None
    correspondence_ok = None
    if proj.fully_certified:
        method = "2.3"
        eps_A_val = proj.eps_A
        correspondence_ok, _ = _verify_correspondence(tr.xy, bs, proj)
        if ok_lin:
            ratio_2_3_2_4 = proj.eps_A / eps_A_lin
    elif ok_lin:
        method = "2.4_fallback"
        eps_A_val = eps_A_lin
    else:
        method = "2.4_fallback_uncertified"
        eps_A_val = float("inf")

    ref, ref_verts, ref_ok = _near_exact_reference(tr.xy, bs)
    ratio_2_3_ref = None
    lb = None
    if ref_ok:
        if proj.fully_certified:
            ratio_2_3_ref = proj.eps_A / max(ref, 1e-9)
        # ref_verts (the near-exact linearization's own, adaptively-placed
        # vertices) doubles as the dense spline sample for LB -- see
        # _fast_hausdorff_lower_bound's docstring for why a fixed-count
        # uniform sample is NOT dense enough in general.
        lb = _fast_hausdorff_lower_bound(tr.xy, ref_verts)

    return {
        "idx": idx,
        "n_points": len(tr.xy),
        "method": method,
        "n_pieces": n_pieces,
        "n_pieces_fallback": n_pieces_fallback,
        "eps_A": eps_A_val,
        "ref": ref,
        "ref_ok": ref_ok,
        "lb": lb,
        "ratio_2_3_ref": ratio_2_3_ref,
        "ratio_2_3_2_4": ratio_2_3_2_4,
        "correspondence_ok": correspondence_ok,
        "time_2_3": t_2_3,
        "time_2_4": t_2_4,
        "time_combined": time.time() - t_track0,
    }


def aggregate_track_results(rows: list[dict], n_total: int) -> dict:
    """Turns a list of _process_track's per-track dicts (possibly assembled
    from several chunk files) into the same summary shape run_full_corpus used
    to return directly."""
    n_pieces_total = sum(r["n_pieces"] for r in rows)
    n_pieces_fallback = sum(r["n_pieces_fallback"] for r in rows)
    n_tracks_2_3 = sum(1 for r in rows if r["method"] == "2.3")
    n_tracks_2_4_fallback = sum(1 for r in rows if r["method"] == "2.4_fallback")
    n_tracks_2_4_uncertified = sum(1 for r in rows if r["method"] == "2.4_fallback_uncertified")

    eps_A_combined = [r["eps_A"] for r in rows]
    lb_values = [r["lb"] for r in rows if r["lb"] is not None]
    ratio_eps_lb = [
        r["eps_A"] / r["lb"] for r in rows if r["lb"] is not None and r["lb"] > 1e-9 and np.isfinite(r["eps_A"])
    ]
    ref_values = [r["ref"] for r in rows if r["ref_ok"]]
    ratio_2_3_ref = [r["ratio_2_3_ref"] for r in rows if r["ratio_2_3_ref"] is not None]
    ratio_2_3_2_4 = [r["ratio_2_3_2_4"] for r in rows if r["ratio_2_3_2_4"] is not None]
    time_2_3 = [r["time_2_3"] for r in rows]
    time_2_4 = [r["time_2_4"] for r in rows]
    time_combined = [r["time_combined"] for r in rows]
    s1_lb_violations = [
        r["idx"] for r in rows if r["lb"] is not None and np.isfinite(r["eps_A"]) and r["eps_A"] < r["lb"] - 1e-6
    ]
    correspondence_violations = [r["idx"] for r in rows if r["correspondence_ok"] is False]
    s2_failures = [
        r["idx"]
        for r in rows
        if r["ref_ok"] and np.isfinite(r["eps_A"]) and r["eps_A"] < r["ref"] - S2_REF_SLACK
    ]

    return {
        "n_total": n_total,
        "n_pieces_total": n_pieces_total,
        "n_pieces_fallback": n_pieces_fallback,
        "n_tracks_2_3": n_tracks_2_3,
        "n_tracks_2_4_fallback": n_tracks_2_4_fallback,
        "n_tracks_2_4_uncertified": n_tracks_2_4_uncertified,
        "eps_A_combined": eps_A_combined,
        "lb_values": lb_values,
        "ratio_eps_lb": ratio_eps_lb,
        "ref_values": ref_values,
        "ratio_2_3_ref": ratio_2_3_ref,
        "ratio_2_3_2_4": ratio_2_3_2_4,
        "time_2_3": time_2_3,
        "time_2_4": time_2_4,
        "time_combined": time_combined,
        "s1_lb_violations": s1_lb_violations,
        "correspondence_violations": correspondence_violations,
        "s2_failures": s2_failures,
        "elapsed": sum(time_combined),
    }


def run_full_corpus(tracks, indices: list[int] | None = None) -> dict:
    """Single-process convenience wrapper around _process_track/
    aggregate_track_results (used for smoke tests and small subsets); the full
    585-track run uses main_chunk's per-chunk subprocesses instead, so the OS
    reclaims memory between chunks."""
    if indices is None:
        indices = list(range(len(tracks)))
    t0 = time.time()
    rows = []
    for i, idx in enumerate(indices):
        rows.append(_process_track(idx, tracks[idx]))
        if (i + 1) % 20 == 0:
            print(f"  full corpus: {i + 1}/{len(indices)} tracks, elapsed {time.time() - t0:.1f}s", flush=True)
    return aggregate_track_results(rows, n_total=len(tracks))


def _select_mpmath_sample(tracks) -> list[int]:
    """Tractability-filtered sample (fixed BEFORE running mpmath itself): among
    all 585 tracks, keep those where n * len(certified_linearize(bs,
    MPMATH_LAM_REF)) <= MPMATH_NM_CAP, then draw MPMATH_SAMPLE_N of them with a
    fixed seed."""
    candidates = []
    for idx, tr in enumerate(tracks):
        bs = _fit_bspline(tr)
        verts, ok = certified_linearize(bs, MPMATH_LAM_REF, max_levels=S2_MAX_LEVELS)
        m = len(verts)
        if ok and len(tr.xy) * m <= MPMATH_NM_CAP:
            candidates.append(idx)
    rng = np.random.default_rng(MPMATH_SAMPLE_SEED)
    chosen = rng.choice(candidates, size=min(MPMATH_SAMPLE_N, len(candidates)), replace=False)
    return sorted(int(i) for i in chosen)


def run_mpmath_sample(tracks) -> dict:
    sample_idx = _select_mpmath_sample(tracks)
    rows = []
    t0 = time.time()
    for idx in sample_idx:
        tr = tracks[idx]
        bs = _fit_bspline(tr)
        proj = certify_spline_projection(tr.xy, bs, max_levels=S2_MAX_LEVELS)
        eps_A_lin, ok_lin = certify_spline_linearization(tr.xy, bs, S2_LAM, eta=S2_ETA, max_levels=S2_MAX_LEVELS)
        eps_A = proj.eps_A if proj.fully_certified else (eps_A_lin if ok_lin else float("inf"))

        _, ref_verts, ref_ok = _near_exact_reference(tr.xy, bs)
        lb = _fast_hausdorff_lower_bound(tr.xy, ref_verts) if ref_ok else float("nan")

        verts, ok = certified_linearize(bs, MPMATH_LAM_REF, max_levels=S2_MAX_LEVELS)
        t1 = time.time()
        d_mp = float(distance_mp(tr.xy.tolist(), verts.tolist(), tol=MPMATH_TOL, dps=MPMATH_DPS))
        mp_elapsed = time.time() - t1
        mpmath_lower_bound = d_mp - MPMATH_LAM_REF - MPMATH_TOL

        rows.append(
            {
                "idx": idx,
                "n": len(tr.xy),
                "m": len(verts),
                "eps_A": eps_A,
                "lb": lb,
                "eps_A_over_lb": eps_A / lb if lb > 1e-9 else float("nan"),
                "distance_mp": d_mp,
                "mpmath_lower_bound": mpmath_lower_bound,
                "eps_A_over_mpmath_lb": eps_A / mpmath_lower_bound if mpmath_lower_bound > 1e-9 else float("nan"),
                "s1_ok": eps_A >= mpmath_lower_bound - 1e-9,
                "mp_elapsed": mp_elapsed,
            }
        )
        print(f"  mpmath sample: idx={idx} n={len(tr.xy)} m={len(verts)} d_mp={d_mp:.4f} elapsed={mp_elapsed:.1f}s", flush=True)

    return {"sample_idx": sample_idx, "rows": rows, "elapsed": time.time() - t0}


def main() -> None:
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log("Loading and cleaning tracks...")
    tracks, clean_stats = load_clean_tracks(n=N_TRACKS, seed=SEED)
    log(f"{len(tracks)} cleaned track segments from {clean_stats.n_tracks_in} raw tracks.")

    log("=== full corpus (585 tracks) ===")
    full = run_full_corpus(tracks)
    log(f"full corpus done: {full['n_tracks_2_3']}/{full['n_total']} via 2.3, "
        f"{full['n_tracks_2_4_fallback']} via 2.4 fallback, "
        f"{full['n_tracks_2_4_uncertified']} uncertified even by 2.4, "
        f"elapsed {full['elapsed']:.1f}s")
    log(f"S1 (LB) violations: {len(full['s1_lb_violations'])}; "
        f"correspondence violations: {len(full['correspondence_violations'])}; "
        f"S2 failures: {len(full['s2_failures'])}")

    log("=== mpmath 30-track sample ===")
    mp_sample = run_mpmath_sample(tracks)
    log(f"mpmath sample done: {len(mp_sample['rows'])} tracks, elapsed {mp_sample['elapsed']:.1f}s")
    n_s1_ok = sum(1 for r in mp_sample["rows"] if r["s1_ok"])
    log(f"S1 (mpmath lower bound) ok: {n_s1_ok}/{len(mp_sample['rows'])}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump({"full": full, "mpmath_sample": mp_sample}, f)
    with open(OUT_LOG, "w") as f:
        f.write("\n".join(log_lines) + "\n")


CHUNK_DIR = "/tmp/m2_chunks"


def main_chunk(start: int, end: int) -> None:
    """Processes tracks[start:end] only, in THIS process, then exits --
    intended to be invoked as a fresh subprocess per chunk (see the module
    docstring's OOM note) so the OS fully reclaims memory before the next
    chunk starts. Writes /tmp/m2_chunks/chunk_{start}_{end}.pkl."""
    os.makedirs(CHUNK_DIR, exist_ok=True)
    tracks, _ = load_clean_tracks(n=N_TRACKS, seed=SEED)
    rows = [_process_track(idx, tracks[idx]) for idx in range(start, min(end, len(tracks)))]
    out_path = os.path.join(CHUNK_DIR, f"chunk_{start}_{end}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(rows, f)
    print(f"chunk [{start}:{end}) done, {len(rows)} tracks -> {out_path}", flush=True)


def main_aggregate() -> None:
    """Combines every /tmp/m2_chunks/chunk_*.pkl into the same {full,
    mpmath_sample} shape main() would have produced directly, then runs the
    (already-tractable-in-one-process) mpmath sample and writes the final
    OUT_PKL/OUT_LOG."""
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    chunk_files = sorted(os.listdir(CHUNK_DIR))
    rows: list[dict] = []
    for name in chunk_files:
        if not name.startswith("chunk_") or not name.endswith(".pkl"):
            continue
        with open(os.path.join(CHUNK_DIR, name), "rb") as f:
            rows.extend(pickle.load(f))
    rows.sort(key=lambda r: r["idx"])
    log(f"Aggregated {len(rows)} tracks from {len(chunk_files)} chunk files.")

    tracks, clean_stats = load_clean_tracks(n=N_TRACKS, seed=SEED)
    full = aggregate_track_results(rows, n_total=len(tracks))
    log(f"full corpus done: {full['n_tracks_2_3']}/{full['n_total']} via 2.3, "
        f"{full['n_tracks_2_4_fallback']} via 2.4 fallback, "
        f"{full['n_tracks_2_4_uncertified']} uncertified even by 2.4, "
        f"elapsed {full['elapsed']:.1f}s")
    log(f"S1 (LB) violations: {len(full['s1_lb_violations'])}; "
        f"correspondence violations: {len(full['correspondence_violations'])}; "
        f"S2 failures: {len(full['s2_failures'])}")

    log("=== mpmath 30-track sample ===")
    mp_sample = run_mpmath_sample(tracks)
    log(f"mpmath sample done: {len(mp_sample['rows'])} tracks, elapsed {mp_sample['elapsed']:.1f}s")
    n_s1_ok = sum(1 for r in mp_sample["rows"] if r["s1_ok"])
    log(f"S1 (mpmath lower bound) ok: {n_s1_ok}/{len(mp_sample['rows'])}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump({"full": full, "mpmath_sample": mp_sample}, f)
    with open(OUT_LOG, "w") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--chunk":
        main_chunk(int(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) >= 2 and sys.argv[1] == "--aggregate":
        main_aggregate()
    else:
        main()
