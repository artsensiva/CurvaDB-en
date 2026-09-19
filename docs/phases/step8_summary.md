# step8 phase summary (gate G2) -- H1 closed, M2 not run

Hybrid line/spline representation (`docs/specs/step8_A_hybrid.md`): does a spline with a good
(free/adaptive) knot placement compress better than DP+SED on exact, noise-free data --
hypothesis H1, left open by step3? Full detail: `benchmarks/results/step8.md` (M0, M0.1, M0.2,
M1); decision history: `docs/decisions/` (ADR-0021 through ADR-0023); milestone review:
`docs/reviews/step8_M1.md`.

## What was checked

- **M0**: a unified binary encoding (`src/traj/encode.py`, spec section 2.5) shared by every
  representation, re-verifying step3's own DP+SED/LSQ-uniform byte counts through it (matched
  cell for cell). Step7's certificates (`certify_polyline`/`certify_spline`) were added as a
  separate, additional correctness check alongside step3's own reachability gate, never
  replacing it (ADR-0021).
- **M0.1/M0.2**: an investigation into why the certified spline's `eps_A<=tol` fraction collapsed
  at tight `tol` -- separated `certify_spline`'s `lam_fallback` margin from the fitter's own
  between-sample error control (fitter dominates; `lam_fallback` matters once the fitter is
  already tight), then re-measured on the actually-selected fitter (`spline.fit()`) and on
  realistic segment sizes rather than full tracks, landing on a lam/internal-tol combination
  (ADR-0022, ADR-0023) that reaches 100% certified reliability at a bounded, measured cost.
- **M1**: `src/traj/knot_removal.py` (spec section 2.4: greedy removal, certified stopping,
  bisection-based for tractability), the free-knot oracle (spec section 2.6: knot removal on a
  dense sample of the TRUE curve, no noise/sparsity), and criterion A3 (spec section 8) -- the
  ceiling check for H1.

## Outcome: A3 and the H1 decision (spec section 8, rule applied verbatim)

| # | Criterion | Result |
|---|---|---|
| A3 | oracle bytes `<=0.80x` DP+SED, in `>=1` valid cell (sigma=0, dt=1) | **FAILED**: no valid cell exists (tol=0.5: DP+SED itself only 33.3% reachable; tol=2/tol=10: the oracle's own reachability only 73.3%/66.7%); informationally, no tol's ratio clears 0.80x either (closest: 0.834x at tol=2) |

Spec section 8's rule, applied verbatim: *"A3 не выполнен -> «H1 закрыта: даже сплайн со
свободными узлами на идеальной кривой не компактнее DP+SED на 20% в проверенных условиях»."*

**H1 is closed: even a free-knot spline on the ideal (noiseless, densely sampled) curve is not
more compact than DP+SED by 20%, under the conditions tested.**

## The metric-divergence finding

The oracle's certified `eps_A<=tol` fraction is a clean 100% at every tested `tol`, while its
reachability under a fixed, time-synchronized honest-error criterion (identical to the one
DP+SED and LSQ-free-knot are judged by, per this milestone's own mandatory same-basis
correction) falls from 86.7% to 66.7% as `tol` grows. Mechanism: a Fréchet certificate bounds
shape closeness under an optimal (possibly time-shifted) correspondence; it carries no
obligation to preserve time synchrony. This generalizes beyond H1 -- see `docs/findings.md`'s
"Reusable" section for the general statement (Fréchet is not a synchrony guarantee; SED or L2 if
synchrony is required).

## M2: not run, not even in reduced form

Spec section 8's "Остановка" clause allows M2 to run reduced (A5/A6 only) after an A3 failure.
This phase did not run it at all: H1's decision rule treats "A3 not met" as a complete, terminal
conclusion, not conditioned on further evidence; both halves of the hybrid (plain spline,
free-knot ceiling) already lost independently (step3, A3), leaving no baseline a DP-selected
combination could plausibly beat; A5/A6 would validate the DP implementation's correctness, not
gather further evidence on H1. See `docs/reviews/step8_M1.md` for the full reasoning.

## What's reusable

`src/traj/encode.py` (the unified segment encoding, spec section 2.5) and `src/traj/
knot_removal.py` (certified-stopping greedy knot removal) are general-purpose, documented,
tested components with no step8-specific coupling -- usable by the `certigeo` core extraction
(phase 3) or any future spline-segment work without changes. The certificate-is-additional-not-
gating pattern (ADR-0021) and the segment-scale-not-full-track budget-measurement discipline
(ADR-0023) are reusable methodology, not just code.

## What we don't know

Whether a DIFFERENT free-knot placement strategy (not this module's one-shot-ranked,
bisection-refined greedy removal) could close more of A3's gap -- not tested, and per the H1
decision rule, not worth testing further once A3 has failed on this milestone's own
implementation; whether the Fréchet/time-synchrony gap found here affects any of step7's own
certified-store conclusions (it doesn't appear to, since step7's own S4 correctness check used
exact brute-force ground truth, not a time-synchronized proxy, but this wasn't re-examined
specifically). Neither blocks closing H1.
