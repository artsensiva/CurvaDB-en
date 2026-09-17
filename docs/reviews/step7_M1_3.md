# Review: step7 M1.3

Verdict: M1 ACCEPTED. Gate criteria S1 and S2 met; M2 may start.

Confirmed:
- ADR-0014: measurement bug found proactively; fix guards the whole bug class
  (domain/mode mismatch now raises), regression tests fail on pre-fix code.
- ADR-0013: rule pre-registered, decided at step 1 without tie-break.
- S1 585/585 vs mpmath; S2 585/585 valid and passing with spline.fit(), 0 fallback.
- eps_A/LB tail explained and verified: global self-proximity, not within-piece
  backtracking; certificates match the mpmath reference exactly.

Carry into M2 (not blocking M1):
1. ADR-0013: add one line that the validity criterion is spline.fit()'s own contract,
   so its 100% is by construction; the choice stands because that criterion is the
   certificate-relevant quantity, not because the contest was neutral.
2. S3 density for splines (median eps_A/LB <= 2) is not yet measured -- required for gate G1.
3. LB (Hausdorff-based) underestimates d_F for self-approaching tracks (M1.3 finding),
   so eps_A/LB is pessimistic; report eps_A vs an mpmath reference on a sample alongside it.
4. Certification runtime per track: measure, M3's query load is much heavier.
