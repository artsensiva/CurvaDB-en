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

**Who to interview (8-10 people):** robotics (industrial and service
manipulators), agritech with RTK navigation (autonomous machinery),
drones (flight planning and logging), surgical robotics (recording
instrument trajectories).

**Questions:**

1. How do you currently store and compare trajectories (coordinates,
   format, size per recording)?
2. Do you need to search for similar motions/routes? If so, by what
   "similarity" criterion (path shape, velocity, acceleration, something
   else)?
3. Where do you get velocity and curvature from -- dedicated sensors
   (IMU, Doppler GNSS, encoders) or by differentiating coordinates?
4. What's currently awkward about your existing stack (storing,
   searching, analyzing trajectories)? Is there anything you put up with
   as "good enough"?

**Format:** 30-45 minutes, semi-structured interview, record verbatim
statements of pain points (not paraphrased) -- verbatim quotes are what
gets used for the return-to-code threshold below.

## Threshold to return to code

At least **3 of 10** interviewees must **independently** (without a
prompt from the interviewer) name a trajectory search or comparison
problem, by shape or kinematics, that current tools don't solve.
"Independently" means: stated as their own pain point, not as agreement
with a proposed hypothesis.

If the threshold isn't reached, the project ends as a research effort
with a published result (`docs/findings.md`, `docs/history.md`,
`docs/blog_draft.md`), with no further experiments.

## Sketch of an H2 experiment (if the threshold is reached)

Tests hypothesis H2: analytical derivatives and search over the
curvature profile kappa(t) on exact data provide practical value for the
task named by interviewees.

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
