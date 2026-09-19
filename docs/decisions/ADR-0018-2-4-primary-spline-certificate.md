# ADR-0018: section 2.4 becomes the primary spline certificate; section 2.3 excluded from the pipeline

- Status: Accepted (gate-level decision, `docs/reviews/step7_M2.md`)
- Date: 2026-09-18

## Context

M2 (`benchmarks/results/step7.md`'s M2 section) implemented spec section 2.3 (monotone
projection-matching certificate) as the spec's designated *primary* spline certificate, with
section 2.4 (certified linearization, M1) as the documented per-track fallback. Full-corpus
validation (585 real tracks) found:

- Section 2.3 certifies only **96/585 (16.4%)** of tracks -- the remaining 83.6% fall back to
  2.4, far above spec S3's `<=10%` fallback-rate threshold.
- Where 2.3 *does* succeed, its `eps_A` has a **median 1.68x larger** than section 2.4's own
  `eps_A` on the same tracks -- section 2.3 is not typically tighter than the certificate it was
  meant to back up.
- Both certificates are independently confirmed *correct* wherever each applies (M2's three S1
  checks: a genuine Hausdorff lower bound on all 585 tracks, a genuine mpmath-based lower bound
  on a 30-track sample, and direct dense-sampling verification of section 2.3's own formula on
  every track where it certified). Section 2.4 alone already certifies **100%** of tracks with a
  dense median `eps_A/LB` of 1.019.

`docs/reviews/step7_M2.md` accepted M2's result (including the negative finding) and required a
gate-level architecture decision, not a re-scoring of S3's criterion.

## Decision

**Section 2.4 (certified linearization) is the primary and default spline certificate going
forward.** Section 2.3 (monotone projection matching) is **not deleted** -- it stays in
`src/traj/certify.py` (`certify_spline_projection`), fully tested (`tests/traj/test_certify.py`),
and available as an explicit, opt-in alternative (`certify_spline(..., use_projection=True)`) for
research/comparison use -- but it is **excluded from the default certification pipeline**.

**S3's fallback-rate criterion (`<=10%`) is not waived, adjusted, or reinterpreted -- it FAILED,
plainly, as written.** The response is an architecture change (stop relying on section 2.3 as
primary), not a criterion change. What *is* reinterpreted is which quantity M3 actually needs:
the spec's S3 criterion measured section 2.3's own availability specifically, not "are dense,
correct spline certificates available" in general -- and the latter is unambiguously yes (100%
via 2.4 alone, median density 1.019).

## This is a specification design issue, not an implementation defect

M2's own S1 checks (three independent, genuine correctness verifications) rule out an
implementation bug as the explanation for section 2.3's poor showing -- the certificate is
correct everywhere it applies; it simply applies rarely, and isn't tight when it does. The root
cause (`docs/decisions/ADR-0016`) is structural to section 2.3's own design: it certifies each
piece against the ORIGINAL track's fixed per-vertex segment direction, so a single genuinely
non-monotone piece relative to that fixed direction (real GPS noise/curvature, common at
~1.87%-per-piece rate) forces the *whole track* to fall back, and with ~242 pieces per track on
average this compounds into an 83.6% track-level fallback rate even though piece-level failures
are individually rare. This is a property of the algorithm the spec specifies, discovered only by
actually implementing and measuring it against the full corpus -- exactly the kind of finding
this project's empirical, measure-before-trusting discipline exists to surface.

## Consequences

- `certify_spline`'s default behavior changes: `use_projection` defaults to `False`, so a bare
  `certify_spline(A, bs)` call now uses section 2.4 only (`method in ("2.4", "2.4_uncertified")`)
  -- no behavior change for any code that already called `certify_spline_linearization` directly
  (unaffected), only for the combined wrapper's *default*.
- M3 (interval queries) is built on section 2.4 exclusively. The near-exact-linearization
  reference M2 used for its own S2/density accounting (up to 213.9s/track) is a *reporting*
  artifact of M2's validation methodology, not something M3's query path needs or should use.
- Section 2.3's code, ADRs (0015, 0016), and tests remain in the repository as a verified,
  documented negative result and a reusable research artifact, per this project's standing
  practice of keeping (not deleting) superseded work with its reasoning intact.

## Links

`docs/reviews/step7_M2.md`; `benchmarks/results/step7.md` (M2 section); ADR-0015, ADR-0016
(section 2.3's own design and known limitation); ADR-0017 (the LB-measurement fix that ruled out
a benchmark-methodology explanation for the negative finding); `docs/ROADMAP.md` section 8.
