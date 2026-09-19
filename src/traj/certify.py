"""Certificates for the certified curve store (docs/specs/step7_B_certified_store.md
section 2.2, exact polyline certificate; section 2.3, monotone projection-matching
spline certificate -- the primary spline path, M2; section 2.4, certified
linearization -- the spline fallback path, M1). Built on
traj.frechet_cont.distance_upper, the only function used for polyline-vs-polyline
certificates (traj.frechet_cont.distance is a general-purpose, non-verified bound --
see its own docstring; certificates need distance_upper's independently re-verified
guarantee). Section 2.3 needs no such bisection: its bound is a direct, measured
construction (see certify_spline_projection).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import BSpline

from traj.bezier import bezier_segments, bezier_segments_in_range, de_casteljau_split, derivative_control_points
from traj.frechet_cont import distance_upper

_MACHINE_EPS = float(np.finfo(np.float64).eps)

# How close (in the piece's own local [0,1] parameter) a root of the chord-
# projected derivative may be to an endpoint before _certify_segment prefers
# plain bisection instead -- see _certify_segment's docstring for why this
# guard is necessary (an empirically-found pathological case, not anticipated
# by the original root-splitting design).
_BOUNDARY_MARGIN = 0.05


def _bbox_diagonal(*polylines: np.ndarray) -> float:
    stacked = np.vstack(polylines)
    span = np.max(stacked, axis=0) - np.min(stacked, axis=0)
    return float(np.hypot(*span))


def _with_margins(d: float, origin: np.ndarray, characteristic_length: float) -> float:
    """certificate = d + eps_machine*|origin| + 64*eps_machine*L

    d: distance_upper(...)'s own result (already a verified upper bound on its own,
       see frechet_cont.py's decide_conservative/distance_upper docstrings).
    eps_machine*|origin|: bounds the residual from decide_conservative's internal
       recentering subtraction (P - P[0]) itself, at the scale of THIS call's own
       origin -- not covered by decide_conservative's local per-cell margin, which
       only accounts for error *after* recentering (docs/reviews/step7_M0.md
       round-2 finding 3).
    64*eps_machine*L: general quantization/rounding safety margin at this piece's
       own characteristic scale L (its bounding-box diagonal), matching the spec's
       floating-point rigor convention (docs/specs/00_overview.md section 3.4).
    """
    recenter_margin = _MACHINE_EPS * float(np.hypot(*origin))
    quant_margin = 64.0 * _MACHINE_EPS * characteristic_length
    return d + recenter_margin + quant_margin


def certify_polyline(A: np.ndarray, kept_indices: np.ndarray, eta: float = 1e-3) -> float:
    """Spec section 2.2: exact certificate for a polyline simplified to a subset of
    its own vertices (e.g. simplify_sed_with_indices's kept_idx). A: full original
    polyline vertices. kept_indices: indices into A of the simplified A'.

    eps_A = max_k [distance_upper(A[i_k:i_{k+1}+1], A[[i_k,i_{k+1}]], tol=eta) + margins]

    This is a valid upper bound on the true d_F(A,A') regardless of whether spec
    2.2's "=" is exactly tight: concatenating each piece's own optimal coupling end
    to end (pieces meet exactly at kept vertices, which map to segment endpoints)
    gives one valid global monotone coupling of cost max_k(...), so
    d_F(A,A') <= max_k(...) always holds -- the certificate is safe even if not
    perfectly tight.
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    kept_indices = np.asarray(kept_indices, dtype=int)
    if len(kept_indices) < 2:
        raise ValueError("kept_indices must have at least 2 entries (start and end)")

    eps_A = 0.0
    for k in range(len(kept_indices) - 1):
        i0, i1 = int(kept_indices[k]), int(kept_indices[k + 1])
        piece = A[i0 : i1 + 1]
        segment = A[[i0, i1]]
        d = distance_upper(piece, segment, tol=eta)
        cert = _with_margins(d, piece[0], _bbox_diagonal(piece))
        eps_A = max(eps_A, cert)
    return eps_A


def _bernstein_to_power(coeffs: np.ndarray) -> np.ndarray:
    """Scalar Bernstein coefficients (degree n) -> standard power-basis coefficients
    (ascending order): a_j = C(n,j) * sum_{i=0}^{j} (-1)^(i+j) * C(j,i) * coeffs[i].
    Closed form, well-conditioned at the low degrees used here (degree <= ~2, since
    a cubic spline's derivative is quadratic)."""
    from scipy.special import comb

    n = len(coeffs) - 1
    power = np.zeros(n + 1)
    for j in range(n + 1):
        s = 0.0
        for i in range(j + 1):
            s += (-1) ** (i + j) * comb(j, i) * coeffs[i]
        power[j] = comb(n, j) * s
    return power


def _eval_scalar_bernstein(coeffs: np.ndarray, u: float) -> float:
    cur = np.asarray(coeffs, dtype=float)
    while len(cur) > 1:
        cur = (1.0 - u) * cur[:-1] + u * cur[1:]
    return float(cur[0])


def _projection_roots_in_unit_interval(deriv_control_points: np.ndarray, e: np.ndarray) -> list[float]:
    """Real roots of <C'(u), e> = 0 for u in (0,1), where C' is the Bezier curve
    (deriv_control_points, degree p-1) and e is a fixed unit direction. Converts
    the scalar Bernstein coefficients deriv_control_points @ e to the power basis
    and finds roots via numpy.roots; each candidate is verified by re-evaluating
    the ORIGINAL Bernstein form via de Casteljau (_eval_scalar_bernstein), rejecting
    spurious roots from basis-conversion noise. Used ONLY to pick split points in
    _certify_segment -- never a substitute for _certified_ok's own check."""
    scalar_coeffs = deriv_control_points @ e
    n = len(scalar_coeffs) - 1
    if n < 1 or not np.any(scalar_coeffs - scalar_coeffs[0]):
        return []

    power_coeffs = _bernstein_to_power(scalar_coeffs)
    if not np.any(power_coeffs[1:]):
        return []

    raw_roots = np.roots(power_coeffs[::-1])
    scale = max(float(np.max(np.abs(scalar_coeffs))), 1e-300)
    roots = []
    for r in raw_roots:
        if abs(r.imag) > 1e-9 * max(abs(r.real), 1.0):
            continue
        u = float(r.real)
        if not (1e-9 < u < 1.0 - 1e-9):
            continue
        if abs(_eval_scalar_bernstein(scalar_coeffs, u)) > 1e-6 * scale:
            continue
        roots.append(u)
    return sorted(roots)


def _certified_ok(control_points: np.ndarray, lam: float) -> bool:
    """The ONLY function that certifies a Bezier piece. Exactly one of:

      - degenerate chord (|seg[-1]-seg[0]| < 1e-12): curve vs POINT -- all control
        points within lam of seg[0] (there is no direction to test monotonicity
        against, so this replaces both the tube and monotonicity conditions).
      - small-ball rule: rho = max_i |P_i - P_0|; if 2*rho <= lam, certified
        without checking monotonicity at all. Every point on the piece AND every
        point on its own chord lies within rho of P_0 (control points bound the
        convex hull, and P_0 is on the chord by construction), so ANY pairing
        between a curve point and a chord point -- monotone or not -- is at most
        2*rho apart; 2*rho <= lam makes this valid regardless of monotonicity,
        exactly where a chord direction is meaningless (near-stationary segments).
      - tube AND monotonicity (spec section 2.3's test): every control point
        within lam of the chord [seg[0], seg[-1]], AND every Bernstein coefficient
        of <C'(u), e> (e = the chord's own unit direction) is >= 0 -- a sufficient
        condition for the projection along e to be non-decreasing over the whole
        piece (small numerical tolerance absorbs float noise at the boundary).

    Root-finding (_projection_roots_in_unit_interval) is used ONLY by
    _certify_segment to choose good split points -- it is NEVER treated as a
    substitute for this check. A piece is certified only when THIS function says
    so, regardless of how many roots were found or splits taken to reach it.
    """
    a, b = control_points[0], control_points[-1]
    chord_vec = b - a
    chord_len = float(np.hypot(*chord_vec))

    if chord_len < 1e-12:
        return bool(np.all(np.hypot(*(control_points - a).T) <= lam))

    rho = max(float(np.hypot(*(p - a))) for p in control_points)
    if 2.0 * rho <= lam:
        return True

    for p in control_points:
        d = b - a
        len2 = float(d @ d)
        tt = np.clip(float((p - a) @ d) / len2, 0.0, 1.0)
        closest = a + tt * d
        if float(np.hypot(*(p - closest))) > lam:
            return False

    e = chord_vec / chord_len
    deriv = derivative_control_points(control_points)
    return bool(np.all(deriv @ e >= -1e-9 * max(chord_len, 1.0)))


def _certify_segment(seg: np.ndarray, lam: float, levels_left: int) -> tuple[list[np.ndarray], bool]:
    """Recursively certifies a Bezier piece against _certified_ok. On failure,
    splits at a root of <C'(u), e> relative to THIS piece's OWN chord (not the
    original top-level segment's) -- monotonicity along one chord does not imply
    monotonicity along a different sub-piece's own chord, so root-finding must be
    redone at every level, relative to whatever chord that level's _certified_ok
    check just failed against. Falls back to a plain t=0.5 bisection when the
    chord is degenerate, no interior root is found, or every root found is too
    close to an endpoint (see _BOUNDARY_MARGIN below) -- the original strategy,
    still needed both as a safety net and to guarantee steady progress.

    Boundary-margin guard (found necessary empirically, not anticipated by the
    original design): near a near-tangential root (the projection dips to ~0 and
    back without a clean crossing), the root returned by
    _projection_roots_in_unit_interval can sit extremely close to one endpoint
    and, after splitting there, reappear just as close to the endpoint of the
    resulting large child's own new local parametrization -- repeating at every
    recursion level. This produces a razor-thin certified sliver each time while
    the other child's chord barely shrinks, burning nearly the whole max_levels
    budget without shrinking the piece enough to satisfy the tube condition
    (confirmed directly: a real track segment needed 9 of 12 levels chasing a
    root from t=0.012 down to t=1.6e-9 before falling back to bisection too late
    to converge -- a genuine regression against the pre-fix algorithm, which just
    bisected at 0.5 and converged in time). Only roots within
    [_BOUNDARY_MARGIN, 1-_BOUNDARY_MARGIN] are used; otherwise plain bisection
    guarantees both children shrink by half every level, regardless of where the
    problematic feature sits.
    """
    if _certified_ok(seg, lam):
        return [seg[-1]], True
    if levels_left == 0:
        return [seg[-1]], False

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
    lv, lok = _certify_segment(left, lam, levels_left - 1)
    rv, rok = _certify_segment(right, lam, levels_left - 1)
    return lv + rv, lok and rok


def certified_linearize(bs: BSpline, lam: float, max_levels: int = 12) -> tuple[np.ndarray, bool]:
    """Spec section 2.4: linearizes spline bs into a polyline Lin(A') by recursively
    certifying each Bezier segment (see _certify_segment/_certified_ok) against
    lam, splitting at roots of the chord-projected derivative where possible
    (falling back to plain de Casteljau bisection at t=0.5 otherwise), up to
    max_levels.

    Returns (vertices, fully_certified). fully_certified=False if some segment still
    fails after max_levels subdivisions (spec's documented "not certified" outcome)
    -- the returned polyline is still produced (uses the finest subdivision reached),
    but per docs/reviews/step7_M0.md item 0.4, callers must NOT treat it as a valid
    lam-bound on d_F(A', Lin(A')) when fully_certified is False.
    """
    segments = bezier_segments(bs)
    vertices = [segments[0][0]]
    fully_certified = True
    for seg in segments:
        verts, ok = _certify_segment(seg, lam, max_levels)
        vertices.extend(verts)
        fully_certified = fully_certified and ok
    return np.array(vertices), fully_certified


def certify_spline_linearization(
    A: np.ndarray, bs: BSpline, lam: float, eta: float = 1e-3, max_levels: int = 12
) -> tuple[float, bool]:
    """Spec section 2.4: eps_A = d_F(A, Lin(A')) + lam [+ margins]. Returns
    (eps_A, fully_certified).

    Per docs/reviews/step7_M0.md item 0.4: if certified_linearize does not fully
    certify at this lam/max_levels, there is no valid lam-bound on d_F(A',Lin(A')),
    so the triangle-inequality formula's safety does not hold -- returns
    (inf, False) rather than a finite (silently wrong) number. Never returns a
    finite eps_A for an uncertified linearization.
    """
    lin_vertices, fully_certified = certified_linearize(bs, lam, max_levels)
    if not fully_certified:
        return float("inf"), False
    d = distance_upper(A, lin_vertices, tol=eta)
    eps_A = _with_margins(d, A[0], _bbox_diagonal(A)) + lam
    return eps_A, True


# --- Section 2.3: monotone projection-matching certificate (M2, primary spline path) -------

_NEWTON_MAX_ITER = 20
_NEWTON_BACKTRACK_MAX = 20
_NEWTON_TOL_FRAC = 1e-13  # relative to the search range's own width
_COARSE_SCAN_POINTS = 50  # candidates evaluated before Newton refines, see ADR-0015


def _nearest_point_on_spline(bs: BSpline, target: np.ndarray, u_lo: float, u_hi: float, u_guess: float) -> float:
    """Local nearest-point search for `target` on `bs`, restricted to `[u_lo, u_hi]`
    (spec 2.3 item 1's monotonicity constraint `u_{k-1} <= u_k`), started from
    `u_guess` (the previous vertex's own `u`). Newton's method on
    `f(u) = |C(u)-target|^2` (`f' = 2*(C(u)-target)@C'(u)`, `f'' =
    2*(C'@C' + (C(u)-target)@C'')`, `C`/`C'`/`C''` via scipy's own `bs(u, nu=...)`),
    clamped into `[u_lo, u_hi]` at every step -- this is what actually GUARANTEES
    monotonicity (`u_k` can never leave the range), not any property of Newton's
    method itself.

    **Safeguarded (backtracking) Newton, not plain Newton** (ADR-0015): plain
    clamped Newton was tried first and found to occasionally overshoot wildly (a
    tiny second derivative near an inflection produces a huge step), clamp to a
    range boundary, and then falsely "converge" there on the very next iteration
    (the post-clip position stops changing, tripping the step-size convergence
    check) even though that boundary is a much WORSE point than where it started
    -- confirmed directly on a real track: four consecutive vertices collapsed to
    the same early `u`, then the next vertex's search jumped all the way to
    `u_hi=1.0` this way, producing a piece spanning nearly the entire spline and
    taking >100x longer to (fail to) certify. Fixed by only ever ACCEPTING a
    Newton step that does not increase the squared residual (backtracking:
    halve the step up to `_NEWTON_BACKTRACK_MAX` times otherwise); since the
    search always starts at `u_guess = u_lo` (the caller's own convention), the
    result can never be worse than `u_lo` itself by construction -- no separate
    "compare against boundaries" fallback is needed.

    Not guaranteed to find the range-global nearest point -- irrelevant to the
    resulting certificate's VALIDITY, which only needs an honestly measured
    `delta_k = |C(u_k)-target|` for whichever `u_k` this returns (recomputed
    independently by the caller, never trusted from this function's own internal
    state). It DOES matter for how USEFUL (tight) the certificate ends up, though:
    starting Newton from `u_guess` alone, even safeguarded, can get trapped in a
    poor local optimum indefinitely -- confirmed directly (a real track's vertex
    20 had a true nearest point at distance 0.34, but Newton from the previous
    vertex's own (already-stuck) `u` converged to a "local optimum" at distance
    83, since the real minimum sits in a completely different, unconnected basin
    of the squared-distance function -- a purely local method can never cross
    into it). A coarse pre-scan (`_COARSE_SCAN_POINTS` evenly spaced samples
    across `[u_lo, u_hi]`, picking the best as Newton's actual starting point)
    fixes this cheaply: it costs a fixed number of extra spline evaluations per
    vertex regardless of the remaining range's size, and reliably finds the right
    basin before Newton refines within it.
    """
    grid = np.linspace(u_lo, u_hi, _COARSE_SCAN_POINTS)
    grid_vals = np.asarray(bs(grid))
    grid_dist2 = np.sum((grid_vals - target) ** 2, axis=1)
    best_grid_u = float(grid[int(np.argmin(grid_dist2))])

    u_guess_clamped = float(np.clip(u_guess, u_lo, u_hi))
    guess_dist2 = float(np.hypot(*(np.asarray(bs(u_guess_clamped)) - target))) ** 2
    if float(grid_dist2.min()) <= guess_dist2:
        u = best_grid_u
        f_val = float(grid_dist2.min())
    else:
        u = u_guess_clamped
        f_val = guess_dist2
    tol = _NEWTON_TOL_FRAC * max(u_hi - u_lo, 1.0)

    for _ in range(_NEWTON_MAX_ITER):
        diff = np.asarray(bs(u)) - target
        Cp = np.asarray(bs(u, 1))
        f1 = 2.0 * float(diff @ Cp)
        Cpp = np.asarray(bs(u, 2))
        f2 = 2.0 * (float(Cp @ Cp) + float(diff @ Cpp))
        if abs(f2) < 1e-300:
            break

        step = f1 / f2
        accepted = False
        u_new = u
        f_new = f_val
        for _ in range(_NEWTON_BACKTRACK_MAX):
            u_new = float(np.clip(u - step, u_lo, u_hi))
            f_new = float(np.hypot(*(np.asarray(bs(u_new)) - target))) ** 2
            if f_new <= f_val + 1e-14 * max(f_val, 1.0):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

        converged = abs(u_new - u) < tol
        u, f_val = u_new, f_new
        if converged:
            break

    return u


def _correspondence_points(A: np.ndarray, bs: BSpline) -> np.ndarray:
    """Spec 2.3 item 1: u_0 = bs.t[0], u_n = bs.t[-1] (fixed to the spline's own
    domain bounds -- not searched), and for 0 < k < n, u_k is the (heuristic)
    nearest point to V_k found via _nearest_point_on_spline starting from
    u_{k-1}, monotone by construction (searched within [u_{k-1}, u_n])."""
    n = len(A) - 1
    u_min, u_max = float(bs.t[0]), float(bs.t[-1])
    u = np.empty(n + 1)
    u[0] = u_min
    u[n] = u_max
    for k in range(1, n):
        u[k] = _nearest_point_on_spline(bs, A[k], u[k - 1], u_max, u[k - 1])
    return u


def _certify_projection_segment(
    seg: np.ndarray, e_k: np.ndarray, tol_scale: float, levels_left: int
) -> tuple[list[np.ndarray], bool]:
    """Recursively certifies a Bezier piece's MONOTONICITY relative to a FIXED
    direction e_k (spec 2.3 item 2's monotonicity condition: every control point of
    the piece's derivative has <., e_k> >= 0). Unlike _certify_segment (section 2.4),
    e_k is NEVER re-derived from a sub-piece's own endpoints -- 2.3 certifies against
    the ORIGINAL track's fixed segment S_k, so the direction being tested against
    must stay constant across every split (ADR-0016). There is no separate "tube"
    gate here (unlike 2.4's `lam`-threshold tube): the tube radius rho_k is MEASURED
    after monotonicity is achieved (see _certify_projection_piece), not tested
    against a target -- so this function only ever needs to fail on monotonicity,
    never on a tube condition.

    Splits at a root of <C'(u), e_k> = 0 when one exists usefully placed (reusing
    _projection_roots_in_unit_interval and the same _BOUNDARY_MARGIN guard as 2.4,
    for the same reason: a root too close to an endpoint causes the same
    razor-thin-sliver pathology ADR-0008 found).

    **Early-exit when no interior root exists** (found necessary empirically,
    ADR-0016): unlike 2.4, where the reference direction is RE-DERIVED from each
    sub-piece's own endpoints (so a different chord can succeed where the parent
    failed), 2.3's e_k is FIXED across every split. If the scalar polynomial
    <C'(u), e_k> has no root in the open interval, it is (barring a measure-zero
    tangency) one sign throughout -- and since the check above already failed, that
    sign is negative EVERYWHERE on this segment. De Casteljau splitting a curve
    that is uniformly one sign produces two children that are ALSO uniformly that
    same sign (subdivision reparametrizes, it cannot introduce a sign change where
    none exists) -- recursing further is not "trying harder", it is guaranteed,
    provable failure, confirmed to previously burn the full max_levels budget on
    real tracks (>1s per such piece; a track can have a dozen+, making a 585-track
    corpus run intractable). Bailing out immediately changes nothing about
    correctness (a real, un-splittable backward excursion -- spec section 10's
    "loop relative to the segment" -- was always going to return False; this just
    stops paying for the wasted subdivisions first).
    """
    deriv = derivative_control_points(seg)
    if bool(np.all(deriv @ e_k >= -1e-9 * tol_scale)):
        return [seg], True
    if levels_left == 0:
        return [seg], False

    roots = _projection_roots_in_unit_interval(deriv, e_k)
    split_t = None
    for r in roots:
        if _BOUNDARY_MARGIN <= r <= 1.0 - _BOUNDARY_MARGIN:
            split_t = r
            break
    if split_t is None:
        if not roots:
            return [seg], False
        split_t = 0.5

    left, right = de_casteljau_split(seg, split_t)
    lv, lok = _certify_projection_segment(left, e_k, tol_scale, levels_left - 1)
    rv, rok = _certify_projection_segment(right, e_k, tol_scale, levels_left - 1)
    return lv + rv, lok and rok


@dataclass
class _PieceResult:
    """Per-piece diagnostics for one k in certify_spline_projection -- used both by
    the property tests (dense-resampling verification) and the M2 benchmark's
    fallback-rate/tube-radius reporting."""

    k: int
    u_k: float
    u_k1: float
    e_k: np.ndarray
    L_k: float
    degenerate: str | None  # None | "curve_vs_point" | "point_vs_segment" | "point_vs_point"
    rho: float
    tails: float
    cost: float
    certified: bool


def _certify_projection_piece(bs: BSpline, A: np.ndarray, u: np.ndarray, delta: np.ndarray, k: int, max_levels: int) -> _PieceResult:
    """One piece of spec 2.3 item 2-4: certifies piece k (parameter range
    [u_k, u_{k+1}]) against segment S_k=[V_k,V_{k+1}], handling BOTH degenerate
    cases explicitly (never silently wrong, never a crash):

      - |S_k| < 1e-12 (coincident source vertices, spec's own edge case): matches
        the whole piece against the single point V_k -- cost bounded by the max
        distance from any control point of any leaf Bezier segment in the piece to
        V_k (mirrors _certified_ok's existing degenerate-chord branch; no direction
        to test monotonicity against, so no subdivision is attempted).
      - u_k == u_{k+1} (Newton's monotonicity clamp bound the search -- e.g. two
        source vertices very close together relative to the spline's own
        curvature): the piece degenerates to the single spline point C(u_k)
        matched against the WHOLE segment S_k -- cost is exactly
        max(|C(u_k)-V_k|, |C(u_k)-V_{k+1}|) (distance from a fixed point to a
        segment is maximized at an endpoint, so this is exact, not a bound).

    Both degenerate branches are unconditionally valid (no subdivision budget to
    exhaust), so certified=True always for them; the general case can fail
    (certified=False) if max_levels is exhausted before every leaf Bezier segment's
    derivative achieves monotonicity relative to e_k.
    """
    V_k, V_k1 = A[k], A[k + 1]
    u_k, u_k1 = float(u[k]), float(u[k + 1])
    S_k = V_k1 - V_k
    L_k = float(np.hypot(*S_k))

    if L_k < 1e-12:
        if u_k1 <= u_k:
            cost = float(np.hypot(*(np.asarray(bs(u_k)) - V_k)))
            return _PieceResult(k, u_k, u_k1, np.zeros(2), L_k, "point_vs_point", 0.0, cost, cost, True)
        max_dist = 0.0
        for seg in bezier_segments_in_range(bs, u_k, u_k1):
            max_dist = max(max_dist, float(np.max(np.hypot(*(seg - V_k).T))))
        return _PieceResult(k, u_k, u_k1, np.zeros(2), L_k, "curve_vs_point", 0.0, max_dist, max_dist, True)

    e_k = S_k / L_k

    if u_k1 <= u_k:
        Cu = np.asarray(bs(u_k))
        cost = max(float(np.hypot(*(Cu - V_k))), float(np.hypot(*(Cu - V_k1))))
        return _PieceResult(k, u_k, u_k1, e_k, L_k, "point_vs_segment", 0.0, cost, cost, True)

    tol_scale = max(L_k, 1.0)
    leaves: list[np.ndarray] = []
    certified = True
    for seg in bezier_segments_in_range(bs, u_k, u_k1):
        leaf, ok = _certify_projection_segment(seg, e_k, tol_scale, max_levels)
        leaves.extend(leaf)
        certified = certified and ok

    rho = 0.0
    for leaf in leaves:
        for p in leaf:
            tt = float(np.clip(float((p - V_k) @ S_k) / (L_k * L_k), 0.0, 1.0))
            closest = V_k + tt * S_k
            rho = max(rho, float(np.hypot(*(p - closest))))

    delta_k, delta_k1 = float(delta[k]), float(delta[k + 1])
    s_uk = float((np.asarray(bs(u_k)) - V_k) @ e_k)
    s_uk1 = float((np.asarray(bs(u_k1)) - V_k) @ e_k)
    tails = max(delta_k, delta_k1, abs(s_uk) + delta_k, abs(L_k - s_uk1) + delta_k1)
    cost = max(rho, tails)
    return _PieceResult(k, u_k, u_k1, e_k, L_k, None, rho, tails, cost, certified)


@dataclass
class ProjectionResult:
    eps_A: float
    fully_certified: bool
    pieces: list[_PieceResult] = field(default_factory=list)
    u: np.ndarray = field(default_factory=lambda: np.empty(0))
    delta: np.ndarray = field(default_factory=lambda: np.empty(0))


def certify_spline_projection(A: np.ndarray, bs: BSpline, eta: float = 1e-3, max_levels: int = 12) -> ProjectionResult:
    """Spec section 2.3: an explicit monotone correspondence between the spline
    `bs` and the ORIGINAL polyline `A`, built piece by piece against each of `A`'s
    own segments (unlike section 2.4, which linearizes the spline into its OWN
    polyline first). `eta` is the numerical slack (relative to each piece's own
    segment length) used only in the monotonicity check's `>= 0` comparison, not a
    bisection tolerance (there is none here -- see module docstring).

    eps_A = max_k(piece_cost_k) + 64*eps_machine*bbox_diagonal(A) (ADR-0016's tails
    formula per piece, spec 2.3 item 4, plus this module's usual rounding margin).
    fully_certified=False if any piece's Bezier segments can't all be shown monotone
    relative to that piece's fixed e_k within max_levels -- callers must fall back to
    certify_spline_linearization (2.4) for the WHOLE track in that case (spec 2.3
    item 5's own documented "not certified -> fallback" outcome; certify_spline does
    this automatically).
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    n = len(A) - 1
    if n < 1:
        raise ValueError("A must have at least 2 vertices")

    u = _correspondence_points(A, bs)
    delta = np.array([float(np.hypot(*(np.asarray(bs(u[k])) - A[k]))) for k in range(n + 1)])

    pieces = [_certify_projection_piece(bs, A, u, delta, k, max_levels) for k in range(n)]
    fully_certified = all(p.certified for p in pieces)
    raw_eps_A = max((p.cost for p in pieces), default=0.0)
    eps_A = raw_eps_A + 64.0 * _MACHINE_EPS * _bbox_diagonal(A)
    return ProjectionResult(eps_A=eps_A, fully_certified=fully_certified, pieces=pieces, u=u, delta=delta)


def certify_spline(
    A: np.ndarray,
    bs: BSpline,
    eta: float = 1e-3,
    lam_fallback: float = 0.1,
    max_levels: int = 12,
    use_projection: bool = False,
) -> tuple[float, str]:
    """ADR-0018: section 2.4 (certified linearization) is the DEFAULT and only
    path used unless use_projection=True. Full-corpus validation (M2,
    benchmarks/results/step7.md) found section 2.3 (monotone projection
    matching, certify_spline_projection) certifies only 16.4% of real tracks
    and, where it does, has a MEDIAN eps_A 1.68x LARGER than section 2.4's own
    -- neither more available nor tighter than the certificate it was meant to
    back up (a spec design issue, docs/reviews/step7_M2.md, not an
    implementation defect -- both paths are independently verified correct
    wherever each applies). Section 2.3 is kept in the code as a verified
    alternative for research/comparison use, not as the default.

    Returns (eps_A, method):
      use_projection=False (default): method in ("2.4", "2.4_uncertified").
      use_projection=True: unchanged pre-ADR-0018 behavior -- section 2.3 is
        tried first, falling back to section 2.4 for the WHOLE track if 2.3
        can't fully certify (spec section 2.3 item 5 / section 10's
        documented risk -- splines that loop relative to some original
        segment); method in ("2.3", "2.4_fallback", "2.4_fallback_uncertified"),
        the last meaning even the fallback couldn't certify (eps_A is inf,
        per certify_spline_linearization's own contract).
    """
    if use_projection:
        result = certify_spline_projection(A, bs, eta=eta, max_levels=max_levels)
        if result.fully_certified:
            return result.eps_A, "2.3"

    eps_A, ok = certify_spline_linearization(A, bs, lam_fallback, eta=eta, max_levels=max_levels)
    if not ok:
        return float("inf"), "2.4_fallback_uncertified" if use_projection else "2.4_uncertified"
    return eps_A, "2.4_fallback" if use_projection else "2.4"


def hausdorff_lower_bound(A: np.ndarray, A_prime: np.ndarray) -> float:
    """Spec section 2.5: LB = max(directed Hausdorff A->A', directed Hausdorff
    A'->A), vertex-based (a valid strict lower bound on d_F for any correspondence
    space, since Hausdorff distance <= Frechet distance: Hausdorff allows nearest-
    point matching, a strictly larger solution space than Frechet's monotone
    correspondences). Reporting-only (spec S3 density estimate, spec section 2.5),
    not itself a certificate.
    """

    def _point_polyline_distance(p: np.ndarray, Y: np.ndarray) -> float:
        best = float("inf")
        for j in range(len(Y) - 1):
            a, b = Y[j], Y[j + 1]
            d = b - a
            len2 = float(d @ d)
            if len2 < 1e-20:
                dist = float(np.hypot(*(p - a)))
            else:
                tt = np.clip(float((p - a) @ d) / len2, 0.0, 1.0)
                dist = float(np.hypot(*(p - (a + tt * d))))
            best = min(best, dist)
        return best

    def _directed(X: np.ndarray, Y: np.ndarray) -> float:
        return max(_point_polyline_distance(p, Y) for p in X)

    return max(_directed(A, A_prime), _directed(A_prime, A))
