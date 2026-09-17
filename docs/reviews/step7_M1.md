# Review: step7 M1

Verdict: S1 accepted; S2 accepted formally, M1.1 required before M2.

Confirmed:
- S1 585/585; worst margin -5e-7 is within the reference's own 1e-6 precision.
- Uncertified linearizations return inf (no false certificates).
- Median eps_A/LB = 1.000 for polylines (plausible: chain vs single segment).

Findings:
1. SPEC ERROR (section 2.4): "certified linearization is always applicable" is false.
   46/585 tracks (7.9%) got no certificate. Likely cause: stops/backtracking make the
   derivative projection change sign; de Casteljau subdivision cannot remove a sign change.
   Must not be confused with S3's fallback share (share of tracks routed to 2.4).
2. Only the median of eps_A/LB is reported; tail distribution missing.
3. Pilot on a fixed prefix mispredicted runtime by 2.5x; use random subsamples.

Required fixes: M1.1 (split at roots of the derivative projection; small-ball rule).
Spec model text is not edited (frozen); correction recorded here and in ROADMAP section 8.
