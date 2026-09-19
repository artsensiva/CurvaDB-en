# Review: step8 M1

Verdict: H1 CLOSED. A3 fails both formally and substantively. M2 is not run, even in reduced
form.

Confirmed:
- `src/traj/encode.py` (M0) re-verifies step3's DP+SED/LSQ-uniform numbers exactly (reachability
  and `n` match cell for cell); the certificate-vs-honest-error split (ADR-0021) and the
  fitter/lam investigation (M0.1/M0.2, ADR-0022/ADR-0023) are methodologically sound and
  independently useful beyond step8.
- `src/traj/knot_removal.py` (M1): bisection-based, certified-stopping knot removal never returns
  a representation with `eps_A > tol` (property-tested); the segment-scale budget check (ADR-
  0023) correctly identified that a naive full-track-based cost estimate would have been ~80-100x
  too pessimistic, and picked a lam/internal-tol combination that reaches 100% certified
  reliability at a measured, bounded cost.

A3 (spec section 8): oracle bytes `<= 0.80 x DP+SED` in >=1 valid cell (sigma=0, dt=1). **No
cell is valid** under spec section 8's own verbatim definition (both compared methods, oracle
and DP+SED, need `>=80%` reachability): `tol=0.5` fails because DP+SED itself is only 33.3%
reachable (unrelated to the oracle, M0's own number); `tol=2`/`tol=10` fail because the oracle's
own reachability never reaches 80% (73.3%, 66.7%). Setting the valid-cell technicality aside,
**A3 fails substantively too**: none of the three tols' oracle/DP+SED byte ratios clear 0.80x
(0.5: 1.050x, 2: 0.834x, 10: 0.879x) -- the closest miss is 0.034x short, at tol=2.

**Headline technical finding of the milestone**, not a footnote: the oracle's certified
`eps_A<=tol` fraction is a clean 100% at every tol, while its reachability under a fixed,
time-synchronized correspondence (step3's own honest-error criterion, applied identically to
all three methods per this milestone's own mandatory correction) falls from 86.7% to 66.7% as
`tol` grows. This is not a bug: `certify_spline` bounds the continuous **Fréchet** distance,
which permits the reconstruction to lead or lag the true curve in time while staying close in
space -- a legitimate Fréchet-optimal correspondence with no obligation to preserve time
synchrony. Knot removal optimizes only the certified (Fréchet) criterion, so a coarser fit (more
knots removed, more slack at looser `tol`) increasingly exploits this freedom. **A Fréchet
certificate is a correctness guarantee on shape, not on timing** -- this generalizes past H1
(see `docs/findings.md`'s "Reusable" section).

Decision (gate-level, to be logged in `docs/ROADMAP.md` section 8): **M2 is not run, not even
in the spec's own reduced form (A5/A6 only).** Rationale:
- H1's decision rule (spec section 8) treats "A3 not met" as a complete, terminal conclusion --
  it does not require M2's evidence to reach a verdict, unlike the "A3 met, A1 not met" branch.
- Both halves of the hybrid have already been tested independently and lost: the plain spline
  (uniform or adaptive knots) lost to DP+SED in step3; the free-knot spline ceiling itself loses
  in A3. A DP that chooses between two options, neither of which beats the baseline on its own,
  has no floor to stand on -- building it would test an implementation detail, not the
  hypothesis.
- A5/A6 (DP dominance, corner-block handling) check that the DP/candidate-selection code is
  *implemented correctly*, not whether H1 holds. Running them without A1/A2 in scope would spend
  a milestone's effort on software-quality assurance for a method direction the finding already
  closes -- better spent, if ever needed, on a different question.

`docs/specs/step8_A_hybrid.md`'s acceptance section is unchanged; this decision is recorded here
and in `docs/ROADMAP.md`, not by editing the frozen spec text.
