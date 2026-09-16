# Review: step7 M0

Verdict: accepted conditionally; fixes M0.1 required before M1.

Checklist (ROADMAP section 5):
- [x] Metric and competitor match the spec (continuous Frechet, Alt-Godau).
- [x] No suspicious coincident numbers.
- [ ] A previous fix did not become a limitation: the discriminant clamp makes decide()
      permissive, so distance() can UNDER-estimate d_F; certificates (S1) need the opposite.
- [ ] Guarantee checked where it is used: tests use small coordinates; GeoLife uses 1e4-1e5 m.
- [x] Acceptance sections unchanged (git diff a393b28 -- docs/specs/ is empty).

Findings:
1. Clamp direction is unsafe for certified upper bounds (error ~ 32*eps*|a-p|^2/eps_query).
2. mpmath oracle shares the same quadratic formula and DP scheme -> not independent of
   algorithmic errors. Add an independent bracket via discrete Frechet on resampled curves.
3. No tests at realistic coordinate scale.
4. decide() allocates O(nm) arrays per call; note for M3.

Required fixes: M0.1 (see prompt below).

## Round 2 review (after M0.1)

Verdict: M0 ACCEPTED. Remaining minor items folded into M1 item 0.

Confirmed:
- Recentering + per-cell margin: no-floor guard 9.5e-10 m at 1e5 m offset.
- distance_upper is a sound upper bound by construction (decide_conservative is stricter than exact).
- Acceptance sections unchanged.

Minor items (M1 item 0):
1. Discrete-Frechet bracket is an upper bound only if resampling keeps ALL original vertices;
   verify or fix, then drop the 4e-7*size slack from the upper-bound check.
2. The ~1.5e-3 GeoLife-scale figure predates recentering; mark as superseded and tighten the
   GeoLife-scale reparametrization threshold from 1e-2 to 1e-6.
3. Recentering subtraction error (~eps_machine*|offset|) is not covered by the local margin;
   certify.py must add this term explicitly.
