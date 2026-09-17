# ADR-0016: section 2.3 certificate structure (fixed direction, measured tube, early exit)

- Status: Accepted
- Date: 2026-09-18

## Context

Spec section 2.3 certifies a spline against the *original* track's own polyline `A` via an
explicit monotone correspondence, piece by piece: piece `k` (spline parameter range
`[u_k, u_{k+1}]`) against segment `S_k = [V_k, V_{k+1}]`. This looks superficially like section
2.4's certified linearization (M1, `_certify_segment`/`_certified_ok`), which also recursively
splits Bezier segments and checks a tube+monotonicity condition -- but three structural
differences make reusing 2.4's code directly wrong, not just suboptimal.

## Decision

**1. `e_k` (the direction monotonicity is tested against) is FIXED per piece, never
re-derived.** 2.4's `_certify_segment` re-derives its chord direction from each sub-piece's *own*
endpoints at every recursion level -- correct there, since 2.4's target is the spline's own
eventual linearization (whatever chord a sub-piece ends up with is fine, as long as it's
consistent with *that* sub-piece). 2.3's target is the *original*, fixed segment `S_k` -- the
direction being tested must stay `e_k = (V_{k+1}-V_k)/|V_{k+1}-V_k|` across every split, or the
correspondence built from it (spec item 3's clamp formula) is no longer valid for `S_k` at all.

**2. The tube radius `rho_k` is MEASURED, not thresholded.** 2.4 has a real gate: "are all
control points within `lam` of the chord?" (a target the caller chose). 2.3 has no such target --
once monotonicity holds on every leaf Bezier segment, `rho_k` is simply the largest clamped
point-to-segment distance found among all their control points, feeding directly into the cost
bound. There is no separate "tube condition" to fail; the only real gate is monotonicity.

**3. Early exit when no interior root exists (found necessary empirically, not anticipated by
the initial design).** Because `e_k` never changes, a Bezier segment whose derivative-projection
onto `e_k` has *no* sign change in `(0,1)` is, by the convex-hull property, uniformly one sign
throughout -- and since the segment already failed the `>= 0` check, that sign is negative
*everywhere*. De Casteljau splitting reparametrizes; it cannot introduce a sign change where none
exists, so *every* descendant of such a segment is provably, permanently non-monotone. The
original design (mirroring 2.4: fall back to bisection at `t=0.5` when no usable root is found)
recurses all the way to `max_levels` here for nothing -- confirmed directly: several real
GeoLife tracks had a dozen-plus pieces each burning 12 levels (up to 4096 leaves) this way, one
track taking >13 s just for its own `certify_spline_projection` call, making a 585-track run
intractable. Fixed: when `_projection_roots_in_unit_interval` returns no root at all, return
`certified=False` immediately. This changes nothing about *correctness* (that segment was always
going to fail -- a genuine backward excursion, spec section 10's "the spline loops relative to
the segment") -- it only stops paying for subdivisions that were guaranteed not to help. The
boundary-margin-guarded bisection fallback (ADR-0008's guard, reused here) is kept for the
*different* case where a root exists but sits too close to an endpoint.

**Degenerate cases, handled explicitly (never a crash, never a silently wrong bound):**
- `|S_k| < 1e-12` (coincident source vertices): the whole piece matches a single point `V_k`;
  cost = max distance from any control point of any leaf Bezier segment in the piece to `V_k`
  (mirrors `_certified_ok`'s existing degenerate-chord branch; no direction to test, no
  subdivision attempted).
- `u_k == u_{k+1}` (the correspondence search, ADR-0015, returned the same parameter for two
  consecutive vertices -- e.g. both genuinely closest to the same spline point): the piece
  degenerates to the single spline point `C(u_k)` matched against the *whole* segment `S_k`;
  cost = `max(|C(u_k)-V_k|, |C(u_k)-V_{k+1}|)` exactly (distance from a fixed point to a segment
  is maximized at one of its two endpoints -- not a bound, the exact value).

**Fallback granularity is per-TRACK, not per-piece** (spec 2.3 item 5's literal text: "the
trajectory gets its certificate via the fallback path 2.4"): `certify_spline_projection` reports
every piece's own `certified` flag (needed for the M2 benchmark's fallback-*rate* reporting), but
`certify_spline` (the combined entry point) abandons 2.3 for the *whole* track the moment
`fully_certified` is False, calling `certify_spline_linearization` (2.4, unchanged from M1) on
the whole spline instead -- matching 2.4's own existing `fully_certified` contract exactly.

## Consequences

- `eps_A(2.3) = max_k(piece_cost_k) + 64*eps_machine*bbox_diagonal(A)` -- a direct, measured
  construction with no bisection tolerance of its own (unlike `distance_upper`-based
  certificates); `eta` in `certify_spline_projection`'s signature is only the monotonicity
  check's numerical slack, not a bisection tolerance, and is documented as such to avoid confusion
  with `certify_polyline`/`certify_spline_linearization`'s different use of the same parameter
  name.
- A real, expected outcome (not a bug) is that many individual pieces may need several
  subdivision levels to resolve, and that a nonzero fraction of tracks will fall back to 2.4 --
  this is exactly the quantity M2's benchmark measures against spec S3's `<=10%` threshold; a
  result above that threshold is a valid, reportable negative finding, not evidence of an
  implementation defect (see `benchmarks/results/step7.md`'s M2 section).

## Links

`src/traj/certify.py`'s `certify_spline_projection`/`_certify_projection_piece`/
`_certify_projection_segment`/`certify_spline`; ADR-0008 (2.4's own root-splitting/boundary-margin
design, whose guard is reused here for a structurally different reason); ADR-0015 (the
correspondence-point search this certificate consumes); `benchmarks/results/step7.md` (M2
section).
