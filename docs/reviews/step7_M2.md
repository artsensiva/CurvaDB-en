# Review: step7 M2

Verdict: ACCEPTED, including its negative result. Architecture decision required (below).

Confirmed:
- S1 (three independent checks) and S2: 100%. S3 density: 1.019 (splines), 1.000 (polylines).
- Three real defects found and fixed mid-milestone, two of them in the measurement itself
  (ADR-0015, ADR-0016, ADR-0017). The 12 "violations" were correctly diagnosed as a coarse
  uniform sample, not a broken certificate.
- Reviewer error acknowledged: eps_A(2.3)/reference < 1 was never mathematically possible;
  both bound the same true d_F and the reference is near-exact. Ratio = slack above best achievable.

Negative result (headline): spec section 2.3 certifies only 96/585 tracks (16.4%) and, where it
does, its eps_A is 1.68x (median) larger than section 2.4's. The spec's primary path is worse than
its own fallback on both availability and tightness. Cause is structural (ADR-0016): a Bezier
piece with no interior root of <C'(u), e_k> is permanently non-monotone against the fixed e_k.

Decision (gate-level, to be logged in docs/ROADMAP.md section 8):
- S3's fallback-rate criterion (<=10%) FAILED as written. It is NOT waived or rescored.
- Architecture change: section 2.4 becomes the primary spline certificate; section 2.3 is retained
  in the codebase as a verified alternative but removed from the certification pipeline.
- Rationale: the criterion measured availability of 2.3, not availability of certificates;
  certificates exist for 100% of tracks and are dense (median eps_A/LB = 1.019).
- M3 uses 2.4 only. The near-exact reference (up to 213.9 s/track) must not be used at query time.
