"""Interval range queries (docs/specs/step7_B_certified_store.md section 2.5): given
two certified representations (query q', candidate A') and a range r, decide
accept/reject/refine using decide() ONLY -- never distance() -- plus cheap P3-style
lower bounds (endpoints, bounding boxes) that reject before any decide() call at all.

Key mapping (not obvious from the spec's own notation, worth stating explicitly):
M1's certify_spline_linearization already returns
    eps_A = distance_upper(A, Lin(A')) + lam
i.e. EXACTLY the spec's own (epsilon_A + lambda_A) sum for a spline's certificate --
section 2.4's own construction is d_F(A, A') <= d_F(A, Lin(A')) + lam, using the SAME
lam that also builds Lin(A'), so "the distance from A to its own approximation" and
"the extra slack from linearizing that approximation for Alt-Godau" collapse into one
already-computed number. certify_polyline's eps_A has no separate lambda term
(lambda=0, polylines don't need linearizing to run Alt-Godau). So Sigma-epsilon in
section 2.5's interval formula is simply the SUM of the two sides' own stored eps_A
values, for either representation -- no separate lambda bookkeeping is needed anywhere
in this module or its callers.

D = d_F(Lin(q'), Lin(A')) is computed by the CALLER (the linearized/simplified
vertices this module receives as q_lin/A_lin are already Lin(q')/Lin(A') --
for polylines that's just the DP+SED-kept vertices themselves; for splines it's
certify_spline_linearization's own certified_linearize(...) output, reused, not
recomputed).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from traj.frechet_cont import decide

DecisionOutcome = Literal["accept", "reject_cheap", "reject_interval", "refine"]


def endpoint_lower_bound(P: np.ndarray, Q: np.ndarray) -> float:
    """max(|P[0]-Q[0]|, |P[-1]-Q[-1]|) -- any monotone Frechet coupling must pair
    the two starts together and the two ends together, so this is a valid lower
    bound on d_F(P, Q) (spec section 3's "P3 filter: ends")."""
    d0 = float(np.hypot(*(P[0] - Q[0])))
    d1 = float(np.hypot(*(P[-1] - Q[-1])))
    return max(d0, d1)


def bbox_lower_bound(P: np.ndarray, Q: np.ndarray) -> float:
    """Gap between P's and Q's axis-aligned bounding boxes (0 if they overlap) --
    every point of P lies in bbox(P) and every point of Q lies in bbox(Q), so any
    correspondence must pair some point of P with some point of Q at distance
    >= this gap: a valid, usually weak (often 0, whenever the boxes overlap), lower
    bound on d_F(P, Q) (spec section 3's "P3 filter: rectangles")."""
    p_lo, p_hi = P.min(axis=0), P.max(axis=0)
    q_lo, q_hi = Q.min(axis=0), Q.max(axis=0)
    dx = max(0.0, float(p_lo[0] - q_hi[0]), float(q_lo[0] - p_hi[0]))
    dy = max(0.0, float(p_lo[1] - q_hi[1]), float(q_lo[1] - p_hi[1]))
    return float(np.hypot(dx, dy))


def cheap_lower_bound(P: np.ndarray, Q: np.ndarray) -> float:
    """The two P3-style filters combined (the larger, and hence tighter, of the
    two) -- the fast pre-filter decide_range applies before any decide() call."""
    return max(endpoint_lower_bound(P, Q), bbox_lower_bound(P, Q))


def decide_range(
    q_lin: np.ndarray, A_lin: np.ndarray, r: float, sum_eps: float, cheap_lb: float | None = None
) -> DecisionOutcome:
    """Spec section 2.5's interval rule for one (query, candidate) pair at range r.
    The true d_F(q, A) is known to lie in [D - sum_eps, D + sum_eps], where D =
    d_F(q_lin, A_lin) (computed here only via decide(), never distance()).

    BOTH sides of the interval are checked explicitly, in the spec's own order
    (mandatory correction: this must not be inferred from a single distance value):

      accept:  decide(D <= r - sum_eps).
               If D <= r - sum_eps, the interval's own upper end (D + sum_eps) is
               <= r, so the TRUE d_F(q, A) <= D + sum_eps <= r -- always safe to
               accept. (Skipped outright, without calling decide(), if
               r - sum_eps < 0: no non-negative D can satisfy that, and decide()
               itself rejects a negative eps.)

      reject:  not decide(D <= r + sum_eps), i.e. the interval's own lower end
               (D - sum_eps) is > r: D itself is confirmed > r + sum_eps, so the
               TRUE d_F(q, A) >= D - sum_eps > r -- always safe to reject.

      refine:  neither of the above -- r falls strictly inside [D-sum_eps,
               D+sum_eps]; the interval alone doesn't determine accept or reject,
               so the caller must read the original (uncompressed) polylines and
               decide exactly there.

    cheap_lb: an already-computed lower bound on D (endpoint_lower_bound/
    bbox_lower_bound/cheap_lower_bound) -- if `cheap_lb > r + sum_eps`, rejects
    immediately with NO decide() call at all. Reported as a SEPARATE outcome
    ("reject_cheap") from an interval-rule rejection ("reject_interval") so
    benchmarks/step7_query.py can report what fraction of rejections came from
    the free pre-filter vs. an actual decide() call (mandatory correction 2).
    """
    if cheap_lb is not None and cheap_lb > r + sum_eps:
        return "reject_cheap"

    lo_threshold = r - sum_eps
    if lo_threshold >= 0.0 and decide(q_lin, A_lin, lo_threshold):
        return "accept"

    hi_threshold = r + sum_eps
    if not decide(q_lin, A_lin, hi_threshold):
        return "reject_interval"

    return "refine"
