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
