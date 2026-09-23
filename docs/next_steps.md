# Next steps

See also: [the full project history](history.md) (including the
pre-repository stage and the mandatory correction to step3's
conclusion) and [research findings](findings.md).

## Resolution

- **The spline as a compressor for consumer-GPS trajectories -- closed**
  (step1, step3): the DP-simplified polyline is more compact at every
  tolerance tested, and the gap isn't explained by noise, observation
  sparsity, or the quality of the specific fitter.
- **Compression on exact (noise-free) data -- closed** (step3, step8):
  no advantage under the conditions tested. Step3's oracle with uniform
  knots wasn't more compact than DP+SED, and hypothesis H1 (optimal/
  free knot placement) closed the gap step3 left open -- step8's
  free-knot oracle (certified knot removal directly on the true curve)
  still doesn't beat DP+SED by the required 20% at any tested tolerance
  (closest: 0.834x at tol=2 m, vs. an 0.80x bar). Even the hypothetical
  ~30% byte savings this was chasing would not have been product value
  by itself (a GPS track already takes up a few kilobytes).
- **The next step is interviews, not code.** The recommendation for
  10-15 interviews was given back in Stage C (`docs/history.md`), before
  any code was written, and was never carried out over the whole
  project. Before returning to experiments, we need to find out whether
  there is a real problem here that current tools (PostGIS, MobilityDB,
  map-matching) don't already solve.

## Interview plan

The authoritative plan is [docs/ROADMAP.md](ROADMAP.md) section 6 -- domains, who to talk to,
and (importantly) exactly which problem statement counts toward the threshold in each domain,
since each domain unlocks a different product spec, not just "more evidence for H2":

| Domain | Unlocks | Who | Problem that counts |
|---|---|---|---|
| High-precision machining | P1 (`docs/specs/P1_certipath_cnc.md`) | CNC toolpath programmers, mold/die engineers, aerospace, implants | time or surface-quality loss from segmented toolpaths and feed-rate reduction; need for a certified path-accuracy guarantee |
| Quality control | P2 (`docs/specs/P2_certiinspect_surfaces.md`) | QC engineers, metrologists, additive manufacturing, casting | false or disputed pass/fail decisions at the tolerance boundary; distrust of grid-based computation; effort of documenting uncertainty |
| Robotics and autonomous vehicles | P3 (`docs/specs/P3_certitrack_store.md`), P4 (`docs/specs/P4_motionshape_search.md`) | data/validation engineers, learning-from-demonstration researchers | missed similar cases when searching logs (P3); searching motions by shape independent of position (P4) |

**Basic questions (asked without naming the product):**

1. Describe the last time geometry or trajectory data cost you time or money.
2. How do you solve that today, and what's awkward about it?
3. What happens if the computation or search result is wrong?
4. What does that cost (machine time, scrap, engineer-hours, risk)?
5. Have you tried existing tools, and if so, why didn't they fit?

**Format:** 30-45 minutes, semi-structured, record verbatim pain-point statements (not
paraphrased) -- verbatim quotes are what gets used for the thresholds below. A problem only
counts if the interviewee names it **themselves**, before the product is described; each
interviewee counts once per domain; the 3-of-10 threshold (below) is checked after 10 interviews
in a domain, or earlier only if 3 have already been reached (`docs/ROADMAP.md` section 6).

## Threshold to return to code, and what happens for each outcome

At least **3 of 10** interviewees in a domain must **independently** (without a prompt from the
interviewer) name a trajectory search or comparison problem, by shape or kinematics, that
current tools don't solve. "Independently" means: stated as their own pain point, not as
agreement with a proposed hypothesis. This is the same threshold `docs/ROADMAP.md` section 4
(gates G4, G6, G7) uses to decide which phase runs next:

- **High-precision machining and/or quality control clear the threshold (gate G4):** the
  corresponding product (P1 and/or P2) becomes phase 4/5's target -- P1 first if both clear (it
  reuses the curve certificates directly, needs no OpenCASCADE, and its value can be checked in
  simulation); the `certigeo` core (phase 3) is extracted first, since phases 4-5 depend on it.
- **Robotics/autonomous vehicles clears the threshold for P3 specifically** (gate G6, which also
  needs step7's S5 `>=80%` for at least one representation -- already satisfied, see
  `docs/phases/step7_summary.md`): P3 can start as soon as phase 3 (core extraction) is done,
  even in parallel with P1/P2's own interview process.
- **Robotics/autonomous vehicles clears the threshold for P4 specifically** (gate G7): P4's
  milestones M0-M2 may start on a free slot even before the threshold is confirmed; M3 onward
  waits for it.
- **No domain reaches the threshold:** no product phase starts, phase 3 (core extraction) has
  nothing to justify it either, and the project ends as a research effort with a published
  result (`docs/REPORT.md`, `docs/findings.md`, `docs/history.md`, the dev.to/Habr drafts in
  `docs/devto_article.md`/`docs/ru/habr_article.md`), with no further experiments.

As of this writing, no interviews have been conducted in any domain (the same gap Stage C
flagged before any code was written, `docs/history.md`) -- this is the reason the project
currently stands at "research complete, product phases not started," not a new finding.

## Sketch of an H2 experiment (if the robotics/UAV threshold is reached)

This is a separate, research-only follow-up to hypothesis H2 -- distinct from the P1-P4 product
phases above, which (once their own thresholds clear) go straight to building the product from
its own spec, not through a research experiment. Tests H2: do analytical derivatives and search
over the curvature profile kappa(t) on exact data provide practical value for the task named by
interviewees.

- **Data:** KITTI (vehicle odometry/trajectories) or TUM RGB-D (camera/
  robot trajectories) -- the choice depends on which domain (transport
  vs. robotics) yields the needed 3 of 10 match in interviews.
- **Metric:** TBD -- depends on what interviewees actually name as the
  "similarity" criterion (Frechet distance by shape? L² on kappa(t)?
  something else) -- not to be fixed before the interviews.
- **Success criteria (fixed BEFORE the run, following the K1-K3
  pattern from step3):**
  - K-product-1: TBD (a number).
  - K-product-2: TBD (a number).
  - Formulated just as rigorously as K1-K3 in `docs/prompts/step3.md`
    -- specific thresholds and experiment cells, before the first code
    run.
- **Oracle:** by analogy with step3 -- a reference on exact (or the most
  precise data available), with an explicit check that the oracle uses
  the method's optimal configuration (a lesson from step5's mandatory
  correction -- don't repeat the "oracle with uniform knots" mistake).

The numbers and metric are deliberately left blank -- to be fixed after
the interviews and before running the experiment, no earlier and no
later.
