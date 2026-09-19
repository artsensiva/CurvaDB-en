# Review: step7 M1.2

Verdict: process accepted (pre-registered thresholds, ADRs, both runs reported);
conclusions NOT accepted; M1.3 required before M2.

Findings:
1. Wrong fitter for S2. The repo already has a dense-error-controlled fitter,
   traj.spline.fit() (step1: 585/585 within tol on a dense grid). ADR-0011 compared only
   fit_adaptive vs fit_uniform. spline_lsq at tol 5 m on noisy GPS near-interpolates and
   oscillates (known from step1/step3). "Out of scope" is incorrect: the tool exists.
   Using spline.fit() is a spec deviation (spec names spline_lsq) -> ADR + ROADMAP entry.
2. "480 pass" at 1000x allows fits deviating up to ~5 km; certificates are sound but
   useless. eps_A distribution for S2 passes is not reported.
3. Conclusions contradict M1.2 numbers ("539/585, 46 fallback"); S2 coverage at the
   fixed threshold is 6.2% and must be stated as such.
4. eps_A/LB tail: S1 shows polyline certificates match the exact reference, so the gap
   is Frechet > Hausdorff (backtracking at stops), not a loose certificate. Verify on the
   5 worst tracks.
5. Repeated /loop wakeups waste the session limit.

## Correction (M1.3)

Finding 1's premise partly relied on numbers that turned out to be wrong, not just
conservative: `benchmarks/step7_certify.py`'s `fit_validity()` measured `spline_lsq`'s
dense-grid deviation at the wrong parametrization domain (real-time-domain knots
evaluated as if normalized to `u ∈ [0,1]`) -- ADR-0014. The corrected invalid rate is
lower than M1.2 reported (~18.3% on a 60-track sample, vs. the reported 93.8%/17.9% at
two thresholds), though still nonzero, confirming finding 1's underlying point (a real
architectural gap in `spline_lsq.py`'s error control) survives the correction even
though the specific severity numbers don't. The fitter choice itself is resolved by
ADR-0013's pre-registered comparison rule (higher valid-fit fraction, tie-broken by
median `eps_A/tol` and absolute median `eps_A`), not by the raw invalid-rate percentage
in isolation -- see `benchmarks/results/step7.md`'s M1.3 section for the corrected
full-corpus numbers and the rule's outcome. Findings 2-4 are addressed as requested,
independent of this correction.
