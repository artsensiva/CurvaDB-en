"""Certificates for the certified curve store (docs/specs/step7_B_certified_store.md
section 2.2, exact polyline certificate; section 2.4, certified linearization -- the
spline fallback path). Built on traj.frechet_cont.distance_upper, the only function
used for certificates (traj.frechet_cont.distance is a general-purpose, non-verified
bound -- see its own docstring; certificates need distance_upper's independently
re-verified guarantee).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from traj.bezier import bezier_segments, de_casteljau_split, derivative_control_points
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
