"""Step7 M3 (docs/specs/step7_B_certified_store.md section 2.5, M3 acceptance criteria
S4-S6): interval range queries over the certified store. Spline certificates use
ONLY section 2.4 (ADR-0018) at a FIXED lam=0.1 -- M1/M2's near-exact lam=1e-4
reference is never used at query time (too slow, and not needed: it existed only
for M1/M2's own validation).

Corpus: the same 585 cleaned GeoLife tracks as M1/M2, plus a set of "near-duplicate"
tracks (a controlled local perturbation of a source track, with the exact d_F to
that source MEASURED directly via traj.frechet_cont.distance on the raw polylines,
not assumed) -- added specifically to stress the r +/- 2*sum_eps boundary regime
(spec section 4.3), where the interval rule must fall back to "refine". Coverage of
this hard regime is verified empirically after generation (see
generate_near_duplicates_with_coverage), not assumed from the perturbation
amplitude alone: more duplicates are added in further rounds, biased toward
whichever r is short of the required >=10% coverage, until every r clears it (or a
round cap is hit, in which case the shortfall is reported honestly, not hidden).

Two representations per corpus entry: (a) a DP+SED-simplified polyline (spec 2.2,
certify_polyline) and (b) a spline.fit() + certify_spline_linearization (spec 2.4,
lam=S2_LAM) -- section 2.3 is deliberately not used (ADR-0018).

Four query methods compared, per (query, candidate, r, representation):
  - "brute": decide() on the RAW (uncompressed) polylines, no filtering at all --
    spec's own "full brute-force" competitor (correctness reference; own timing
    measures the true no-shortcuts cost).
  - "filter_uncompressed": endpoint/bbox lower bounds on the RAW polylines reject
    first; decide() on the RAW polylines otherwise -- spec's "filter without
    compression" competitor (speed reference for an uncompressed store).
  - "approximate": decide() on the LINEARIZED/compressed representations directly,
    with NO epsilon slack at all -- spec's "approximate search without
    certificates" competitor; reports misses/false-positives as the quantified
    cost of dropping the guarantee.
  - "certified": intervals.decide_range on the compressed representations (cheap
    filter, then the decide-only interval rule); "refine" outcomes read the RAW
    polylines (reusing the SAME raw decide() this module needs anyway for ground
    truth -- not a separate, uncounted cost).

S4/S5 are computed at full scale (1000 queries x 3 r's x 2 representations x the
whole candidate corpus) using one shared, cached ground-truth decide() call per
(query, candidate, r) -- reused for correctness scoring across representations,
never recomputed redundantly. S6 and the latency percentiles are measured on a
SEPARATE, smaller, dedicated timing sample (each method run through its own actual
code path, uncached) -- correctness at full scale and timing on a representative
sample are different questions and use different (appropriately sized) workloads.

Run: venv/bin/python benchmarks/step7_query.py
"""

from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np
from scipy.interpolate import BSpline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from traj.certify import certified_linearize, certify_polyline, certify_spline_linearization  # noqa: E402
from traj.clean import load_clean_tracks  # noqa: E402
from traj.frechet_cont import decide  # noqa: E402
from traj.intervals import cheap_lower_bound, decide_range  # noqa: E402
from traj.io import Track  # noqa: E402
from traj.simplify import simplify_sed_with_indices  # noqa: E402
from traj.spline import fit as spline_fit  # noqa: E402

SEED = 42
N_TRACKS = 200

DP_SED_TOL = 10.0
S1_ETA = 1e-6

S2_FIT_TOL = 10.0  # ADR-0013's selected S2 fitter's own tol
S2_LAM = 0.1  # the ONLY spline lam used at query time (ADR-0018: section 2.4 only)
S2_ETA = 1e-3
S2_MAX_LEVELS = 12

TARGET_RS = [50.0, 200.0, 1000.0]
HARD_BAND_MULT = 2.0  # spec 4.3: r +/- 2*sum_eps is the hardest regime
MIN_HARD_BAND_FRACTION = 0.10  # mandatory correction 3
DUP_INITIAL_BATCH = 60
DUP_ROUND_BATCH = 20
DUP_MAX_ROUNDS = 15
DUP_AMPLITUDE_LO = 5.0
DUP_AMPLITUDE_HI = 1500.0
DUP_SEED = 20260918

N_QUERIES = 1000
TIMING_SAMPLE_PAIRS = 4000  # dedicated, per-method timing sample (S6 + latency)

OUT_PKL = "/tmp/m3_query.pkl"
OUT_LOG = "/tmp/m3_query.log"


# --- representations --------------------------------------------------------------


def build_representations(xy: np.ndarray, t: np.ndarray) -> dict:
    """Both representations for one track's (xy, t): DP+SED polyline (spec 2.2)
    and spline.fit()+certify_spline_linearization (spec 2.4, ADR-0018). Returns a
    dict with polyline_kept/polyline_eps/spline_lin/spline_eps -- spline_lin/
    spline_eps are None if the fit or its 2.4 certificate fails (rare, handled
    gracefully: such an entry is simply skipped for the spline representation
    everywhere downstream, not silently given a wrong certificate)."""
    track = Track(track_id="q", lat=np.zeros(len(xy)), lon=np.zeros(len(xy)), t=t, xy=xy)

    _, kept_idx = simplify_sed_with_indices(track, tol=DP_SED_TOL)
    polyline_kept = xy[kept_idx]
    polyline_eps = certify_polyline(xy, kept_idx, eta=S1_ETA) if len(kept_idx) >= 2 else None

    spline_lin, spline_eps = None, None
    try:
        sp = spline_fit(track, tol=S2_FIT_TOL)
        bs = BSpline(sp.tck[0], np.column_stack(sp.tck[1]), sp.tck[2])
        eps_A, ok = certify_spline_linearization(xy, bs, S2_LAM, eta=S2_ETA, max_levels=S2_MAX_LEVELS)
        if ok:
            spline_lin, _ = certified_linearize(bs, S2_LAM, max_levels=S2_MAX_LEVELS)
            spline_eps = eps_A
    except (ValueError, np.linalg.LinAlgError):
        pass

    return {
        "polyline_kept": polyline_kept,
        "polyline_eps": polyline_eps,
        "spline_lin": spline_lin,
        "spline_eps": spline_eps,
    }


# --- near-duplicate generation -----------------------------------------------------


def make_near_duplicate(rng: np.random.Generator, xy: np.ndarray, target_distance: float) -> np.ndarray:
    """A controlled distortion targeting a specific d_F to the original: a
    uniform translation by a vector of magnitude `target_distance` (the
    identity, same-index correspondence gives every point the SAME cost
    `target_distance`, so d_F(xy, shifted) <= target_distance always -- for a
    generic, non-self-similar track shape the optimal monotone coupling can't
    usually do much better, so the measured d_F lands close to the target),
    plus small jitter (capped well below the shift itself) so the copy isn't a
    literal, featureless teleport. The exact resulting d_F is always MEASURED
    afterward (make_near_duplicate never assumes it hit the target) --
    generate_near_duplicates_with_coverage's rounds retarget based on that
    measurement, not on `target_distance` itself.

    (An earlier version used a LOCAL "bump" perturbation on a sub-range of
    points; found empirically to be an unreliable way to hit a target
    distance -- a bump's contribution to the FRECHET distance depends heavily
    on how it interacts with the optimal matching, not just its own
    amplitude, making the coverage rounds converge far too slowly. A global
    translation's cost is tightly predictable by construction, while the
    resulting d_F is still independently measured, not assumed.)
    """
    direction = rng.normal(size=2)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
    shift = direction * target_distance
    jitter_scale = min(target_distance * 0.02, 2.0)
    jitter = rng.normal(0.0, max(jitter_scale, 1e-6), size=xy.shape)
    return xy + shift[None, :] + jitter


def _hard_band_fraction(duplicates: list[dict], corpus: list[dict], r: float, rep: str) -> float:
    eps_key = f"{rep}_eps"
    vals = []
    for d in duplicates:
        src = corpus[d["source_idx"]]
        if d[eps_key] is None or src[eps_key] is None:
            continue
        sum_eps = d[eps_key] + src[eps_key]
        band = HARD_BAND_MULT * sum_eps
        vals.append(abs(d["true_d_F"] - r) <= band)
    return float(np.mean(vals)) if vals else 0.0


def generate_near_duplicates_with_coverage(tracks, corpus: list[dict]) -> tuple[list[dict], list[dict]]:
    """Generates near-duplicates in rounds, checking after each round whether
    every r in TARGET_RS has >= MIN_HARD_BAND_FRACTION of (duplicate, source)
    pairs landing in the r +/- 2*sum_eps hard band (mandatory correction 3) --
    for BOTH representations independently. Rounds after the first are biased
    toward whichever r's are still short, drawing amplitudes near that r rather
    than the initial broad log-uniform spread. Returns (duplicates, round_log) --
    round_log is written into the report even if the final round still falls
    short (honest, not hidden)."""
    rng = np.random.default_rng(DUP_SEED)
    duplicates: list[dict] = []
    round_log: list[dict] = []
    lacking_rs = list(TARGET_RS)

    for round_i in range(DUP_MAX_ROUNDS):
        batch = DUP_INITIAL_BATCH if round_i == 0 else DUP_ROUND_BATCH
        for _ in range(batch):
            # target_r: which r this duplicate is aimed at. Round 0 jitters
            # broadly (0.3x-3x) for general exploration/diversity; later
            # ("targeted") rounds jitter tightly (0.97x-1.03x), cycling through
            # ALL of TARGET_RS evenly (not just the currently-lacking ones --
            # tried first, but targeting only the lacking r's dilutes the
            # OTHER r's already-adequate coverage as the total duplicate count
            # grows, causing oscillation instead of monotone convergence)
            # -- make_near_duplicate's measured d_F tracks target_distance
            # closely (empirically ~0.2% off), so a tight jitter reliably
            # lands inside the (typically much narrower than target_r itself)
            # r +/- 2*sum_eps hard band.
            target_r = float(rng.choice(TARGET_RS))
            jitter_lo, jitter_hi = (0.97, 1.03) if round_i > 0 else (0.3, 3.0)
            target_distance = max(1.0, target_r * float(rng.uniform(jitter_lo, jitter_hi)))
            src_idx = int(rng.integers(0, len(tracks)))
            dup_xy = make_near_duplicate(rng, tracks[src_idx].xy, target_distance)
            true_d = _exact_distance(dup_xy, tracks[src_idx].xy)
            reps = build_representations(dup_xy, tracks[src_idx].t)
            duplicates.append(
                {"source_idx": src_idx, "xy": dup_xy, "t": tracks[src_idx].t, "true_d_F": true_d,
                 "target_distance": target_distance, **reps}
            )

        coverage = {}
        lacking_rs = []
        for r in TARGET_RS:
            for rep in ("polyline", "spline"):
                frac = _hard_band_fraction(duplicates, corpus, r, rep)
                coverage[(r, rep)] = frac
                if frac < MIN_HARD_BAND_FRACTION and r not in lacking_rs:
                    lacking_rs.append(r)
        round_log.append({"round": round_i, "n_duplicates": len(duplicates), "coverage": coverage, "lacking_rs": list(lacking_rs)})
        print(f"  near-dup round {round_i}: n={len(duplicates)} coverage={coverage}", flush=True)
        if not lacking_rs:
            break

    return duplicates, round_log


def _exact_distance(P: np.ndarray, Q: np.ndarray) -> float:
    from traj.frechet_cont import distance

    return distance(P, Q, tol=1e-6)


# --- corpus + query set -------------------------------------------------------------


def build_corpus(tracks) -> list[dict]:
    corpus = []
    for idx, tr in enumerate(tracks):
        reps = build_representations(tr.xy, tr.t)
        corpus.append(
            {"idx": idx, "xy": tr.xy, "t": tr.t, "is_duplicate": False, "source_idx": None, "true_d_F": None, **reps}
        )
    return corpus


def _append_duplicate(corpus: list[dict], tracks, d: dict) -> None:
    corpus.append(
        {
            "idx": len(corpus),
            "xy": d["xy"],
            "t": d["t"],
            "is_duplicate": True,
            "source_idx": d["source_idx"],
            "true_d_F": d["true_d_F"],
            "target_distance": d["target_distance"],
            "polyline_kept": d["polyline_kept"],
            "polyline_eps": d["polyline_eps"],
            "spline_lin": d["spline_lin"],
            "spline_eps": d["spline_eps"],
        }
    )


def _top_up_corpus_for_queries(tracks, corpus: list[dict], n_target: int, seed: int) -> int:
    """N_QUERIES=1000 distinct queries need >= 1000 distinct corpus entries to
    draw them from -- the hard-regime-targeted near-duplicates (generate_near_
    duplicates_with_coverage) are sized for coverage, not query VOLUME, and can
    fall short of that (e.g. 585 base + 100 targeted duplicates = 685). Tops up
    with additional near-duplicates (broad log-uniform amplitude, general
    volume, not aimed at any specific r) until the corpus has >= n_target
    entries. Returns the number added."""
    rng = np.random.default_rng(seed)
    n_added = 0
    while len(corpus) < n_target:
        src_idx = int(rng.integers(0, len(tracks)))
        amp = float(np.exp(rng.uniform(np.log(DUP_AMPLITUDE_LO), np.log(DUP_AMPLITUDE_HI))))
        dup_xy = make_near_duplicate(rng, tracks[src_idx].xy, amp)
        true_d = _exact_distance(dup_xy, tracks[src_idx].xy)
        reps = build_representations(dup_xy, tracks[src_idx].t)
        _append_duplicate(
            corpus, tracks,
            {"source_idx": src_idx, "xy": dup_xy, "t": tracks[src_idx].t, "true_d_F": true_d,
             "target_distance": amp, **reps},
        )
        n_added += 1
    return n_added


def assemble_full_corpus(tracks) -> tuple[list[dict], list[dict]]:
    corpus = build_corpus(tracks)
    duplicates, round_log = generate_near_duplicates_with_coverage(tracks, corpus)
    for d in duplicates:
        _append_duplicate(corpus, tracks, d)

    n_topped_up = _top_up_corpus_for_queries(tracks, corpus, N_QUERIES, seed=DUP_SEED + 1)
    if n_topped_up:
        print(f"  topped up corpus with {n_topped_up} general-volume near-duplicates "
              f"(coverage-targeted generation gave {len(corpus) - n_topped_up} entries, "
              f"short of the {N_QUERIES} needed for a full query set)", flush=True)
        round_log.append({"round": "top_up", "n_duplicates": n_topped_up, "coverage": None, "lacking_rs": []})

    return corpus, round_log


def select_queries(full_corpus: list[dict], n_target: int = N_QUERIES, seed: int = SEED + 1) -> list[int]:
    """All near-duplicates (engineered for the hard regime) plus enough random
    originals to reach n_target."""
    rng = np.random.default_rng(seed)
    dup_indices = [e["idx"] for e in full_corpus if e["is_duplicate"]]
    orig_indices = [e["idx"] for e in full_corpus if not e["is_duplicate"]]
    n_orig_needed = max(0, n_target - len(dup_indices))
    chosen_orig = rng.choice(orig_indices, size=min(n_orig_needed, len(orig_indices)), replace=False)
    queries = list(dup_indices) + [int(i) for i in chosen_orig]
    if len(queries) > n_target:
        queries = list(rng.choice(queries, size=n_target, replace=False))
    return sorted(int(q) for q in queries)


def _lin_of(entry: dict, rep: str) -> np.ndarray | None:
    return entry["polyline_kept"] if rep == "polyline" else entry["spline_lin"]


def _eps_of(entry: dict, rep: str) -> float | None:
    return entry["polyline_eps"] if rep == "polyline" else entry["spline_eps"]


def _new_rep_stats() -> dict:
    return {
        "n_total_candidates": 0,
        "n_accept": 0,
        "n_reject_cheap": 0,
        "n_reject_interval": 0,
        "n_refine": 0,
        "n_miss": 0,
        "n_false_positive": 0,
        "approx_n_miss": 0,
        "approx_n_false_positive": 0,
    }


# --- S4/S5: full-scale correctness grid ---------------------------------------------


def run_correctness_grid(full_corpus: list[dict], query_indices: list[int]) -> dict:
    """S4 (0 misses/0 false positives for the certified method) and S5 (fraction
    resolved without reading originals, both as a fraction of ALL candidates and
    as a fraction of post-cheap-filter survivors -- mandatory correction 1), at
    full scale: every query x every OTHER corpus entry x every r in TARGET_RS x
    both representations. One shared ground-truth decide() per (query, candidate,
    r), reused for both representations' correctness scoring (never recomputed).
    Also collects sum_eps ("interval half-width") samples per representation."""
    results = {r: {"polyline": _new_rep_stats(), "spline": _new_rep_stats()} for r in TARGET_RS}
    sum_eps_samples: dict[str, list[float]] = {"polyline": [], "spline": []}

    t0 = time.time()
    n_candidates_seen = 0
    for qi, q_idx in enumerate(query_indices):
        q = full_corpus[q_idx]
        for cand in full_corpus:
            if cand["idx"] == q_idx:
                continue
            ground_truth_by_r = {r: bool(decide(q["xy"], cand["xy"], r)) for r in TARGET_RS}
            n_candidates_seen += 1

            for rep in ("polyline", "spline"):
                q_lin, q_eps = _lin_of(q, rep), _eps_of(q, rep)
                a_lin, a_eps = _lin_of(cand, rep), _eps_of(cand, rep)
                if q_lin is None or a_lin is None or q_eps is None or a_eps is None:
                    continue
                sum_eps = q_eps + a_eps
                if len(sum_eps_samples[rep]) < 200_000:
                    sum_eps_samples[rep].append(sum_eps)
                cheap_lb = cheap_lower_bound(q_lin, a_lin)

                for r in TARGET_RS:
                    stats = results[r][rep]
                    stats["n_total_candidates"] += 1
                    ground_truth = ground_truth_by_r[r]

                    outcome = decide_range(q_lin, a_lin, r, sum_eps, cheap_lb=cheap_lb)
                    if outcome == "accept":
                        stats["n_accept"] += 1
                        predicted = True
                    elif outcome == "reject_cheap":
                        stats["n_reject_cheap"] += 1
                        predicted = False
                    elif outcome == "reject_interval":
                        stats["n_reject_interval"] += 1
                        predicted = False
                    else:
                        stats["n_refine"] += 1
                        predicted = ground_truth  # refine reads the originals -> always correct

                    if predicted and not ground_truth:
                        stats["n_false_positive"] += 1
                    if (not predicted) and ground_truth:
                        stats["n_miss"] += 1

                    approx_accept = bool(decide(q_lin, a_lin, r))
                    if approx_accept and not ground_truth:
                        stats["approx_n_false_positive"] += 1
                    if (not approx_accept) and ground_truth:
                        stats["approx_n_miss"] += 1

        if (qi + 1) % 20 == 0:
            print(f"  correctness grid: {qi + 1}/{len(query_indices)} queries, "
                  f"{n_candidates_seen} candidate-pairs, elapsed {time.time() - t0:.1f}s", flush=True)

    return {"results": results, "sum_eps_samples": sum_eps_samples, "elapsed": time.time() - t0}


# --- S6 + latency: dedicated timing sample ------------------------------------------


def run_timing_sample(full_corpus: list[dict], query_indices: list[int], seed: int = SEED + 2) -> dict:
    """S6 (query time vs. approximate search, certificate size) and latency
    percentiles -- measured on a SEPARATE, smaller, dedicated sample of (query,
    candidate, r) triples, each method run through its own actual, uncached code
    path (never reusing the correctness grid's cached ground truth for timing --
    that would understate a method's real cost)."""
    rng = np.random.default_rng(seed)
    all_indices = [e["idx"] for e in full_corpus]
    pairs = []
    for _ in range(TIMING_SAMPLE_PAIRS):
        q_idx = int(rng.choice(query_indices))
        cand_idx = int(rng.choice(all_indices))
        while cand_idx == q_idx:
            cand_idx = int(rng.choice(all_indices))
        r = float(rng.choice(TARGET_RS))
        pairs.append((q_idx, cand_idx, r))

    timings: dict[str, list[float]] = {"brute": [], "filter_uncompressed": []}
    for rep in ("polyline", "spline"):
        timings[f"approximate_{rep}"] = []
        timings[f"certified_{rep}"] = []

    t0 = time.time()
    for i, (q_idx, cand_idx, r) in enumerate(pairs):
        q = full_corpus[q_idx]
        cand = full_corpus[cand_idx]

        t = time.perf_counter()
        decide(q["xy"], cand["xy"], r)
        timings["brute"].append(time.perf_counter() - t)

        t = time.perf_counter()
        cheap_lb_raw = cheap_lower_bound(q["xy"], cand["xy"])
        if cheap_lb_raw <= r:
            decide(q["xy"], cand["xy"], r)
        timings["filter_uncompressed"].append(time.perf_counter() - t)

        for rep in ("polyline", "spline"):
            q_lin, q_eps = _lin_of(q, rep), _eps_of(q, rep)
            a_lin, a_eps = _lin_of(cand, rep), _eps_of(cand, rep)
            if q_lin is None or a_lin is None or q_eps is None or a_eps is None:
                continue
            sum_eps = q_eps + a_eps

            t = time.perf_counter()
            decide(q_lin, a_lin, r)
            timings[f"approximate_{rep}"].append(time.perf_counter() - t)

            t = time.perf_counter()
            cheap_lb = cheap_lower_bound(q_lin, a_lin)
            outcome = decide_range(q_lin, a_lin, r, sum_eps, cheap_lb=cheap_lb)
            if outcome == "refine":
                decide(q["xy"], cand["xy"], r)  # refine reads the originals
            timings[f"certified_{rep}"].append(time.perf_counter() - t)

        if (i + 1) % 500 == 0:
            print(f"  timing sample: {i + 1}/{len(pairs)} pairs, elapsed {time.time() - t0:.1f}s", flush=True)

    # S6's "certificate size" -- the certificate itself is one scalar eps_A
    # (float64, 8 bytes) per trajectory for either representation (ADR-0018:
    # section 2.4's eps_A already bundles lam in, no separate field needed) --
    # well within the spec's own <=16-byte budget.
    cert_bytes = {"polyline": 8, "spline": 8}

    return {"timings": timings, "cert_bytes": cert_bytes, "elapsed": time.time() - t0, "n_pairs": len(pairs)}


# --- main -----------------------------------------------------------------------


def main() -> None:
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log("Loading and cleaning tracks...")
    tracks, clean_stats = load_clean_tracks(n=N_TRACKS, seed=SEED)
    log(f"{len(tracks)} cleaned track segments from {clean_stats.n_tracks_in} raw tracks.")

    log("=== building corpus + near-duplicates ===")
    full_corpus, round_log = assemble_full_corpus(tracks)
    n_dup = sum(1 for e in full_corpus if e["is_duplicate"])
    log(f"corpus: {len(full_corpus)} entries ({len(tracks)} base + {n_dup} near-duplicates, "
        f"{len(round_log)} generation round(s))")
    for entry in round_log:
        log(f"  round {entry['round']}: n={entry['n_duplicates']} lacking_rs={entry['lacking_rs']}")

    query_indices = select_queries(full_corpus)
    log(f"query set: {len(query_indices)} queries")

    log("=== correctness grid (S4/S5) ===")
    grid = run_correctness_grid(full_corpus, query_indices)
    log(f"correctness grid done, elapsed {grid['elapsed']:.1f}s")

    log("=== timing sample (S6 + latency) ===")
    timing = run_timing_sample(full_corpus, query_indices)
    log(f"timing sample done, elapsed {timing['elapsed']:.1f}s")

    with open(OUT_PKL, "wb") as f:
        pickle.dump({"full_corpus_meta": _corpus_meta(full_corpus), "round_log": round_log,
                     "query_indices": query_indices, "grid": grid, "timing": timing}, f)
    with open(OUT_LOG, "w") as f:
        f.write("\n".join(log_lines) + "\n")


def _corpus_meta(full_corpus: list[dict]) -> dict:
    """Small, picklable summary of the corpus (not the raw xy arrays -- those
    aren't needed once the grid/timing results exist, and would bloat the
    pickle)."""
    return {
        "n_total": len(full_corpus),
        "n_duplicates": sum(1 for e in full_corpus if e["is_duplicate"]),
        "n_polyline_available": sum(1 for e in full_corpus if e["polyline_eps"] is not None),
        "n_spline_available": sum(1 for e in full_corpus if e["spline_eps"] is not None),
    }


if __name__ == "__main__":
    main()
