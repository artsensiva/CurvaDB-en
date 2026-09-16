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


def _tube_and_monotone_ok(control_points: np.ndarray, radius: float) -> bool:
    """Spec section 2.3's test (reused as-is by section 2.4), applied to a Bezier
    segment's own chord (its first-to-last control point): tube condition -- every
    control point within `radius` of the chord; monotonicity condition -- every
    derivative control point's dot product with the chord's unit direction is
    >= 0 (a small numerical tolerance absorbs float noise at the boundary)."""
    a, b = control_points[0], control_points[-1]
    chord_vec = b - a
    chord_len = float(np.hypot(*chord_vec))

    if chord_len < 1e-12:
        # degenerate chord: tube test against the single point a; monotonicity is
        # vacuous (there's no direction to project onto).
        return bool(np.all(np.hypot(*(control_points - a).T) <= radius))

    e = chord_vec / chord_len
    for p in control_points:
        d = b - a
        len2 = float(d @ d)
        tt = np.clip(float((p - a) @ d) / len2, 0.0, 1.0)
        closest = a + tt * d
        if float(np.hypot(*(p - closest))) > radius:
            return False

    deriv = derivative_control_points(control_points)
    return bool(np.all(deriv @ e >= -1e-9 * max(chord_len, 1.0)))


def _certify_segment(seg: np.ndarray, lam: float, levels_left: int) -> tuple[list[np.ndarray], bool]:
    if _tube_and_monotone_ok(seg, lam):
        return [seg[-1]], True
    if levels_left == 0:
        return [seg[-1]], False
    left, right = de_casteljau_split(seg, 0.5)
    lv, lok = _certify_segment(left, lam, levels_left - 1)
    rv, rok = _certify_segment(right, lam, levels_left - 1)
    return lv + rv, lok and rok


def certified_linearize(bs: BSpline, lam: float, max_levels: int = 12) -> tuple[np.ndarray, bool]:
    """Spec section 2.4: linearizes spline bs into a polyline Lin(A') by recursive
    de Casteljau subdivision (up to max_levels) of each Bezier segment, until it
    satisfies both the tube (within lam of its own chord) and monotonicity
    conditions (spec section 2.3's test, applied here to the segment's own chord).

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
