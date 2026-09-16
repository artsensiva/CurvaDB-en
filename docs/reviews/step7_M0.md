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
