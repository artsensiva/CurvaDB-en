# ADR-0019: interval queries (section 2.5) -- eps_A already equals epsilon+lambda, decide-only rule

- Status: Accepted
- Date: 2026-09-18

## Context

Spec section 2.5 needs, per `(query, candidate)` pair at range `r`: `D = d_F(Lin(q'),
Lin(A'))` (exact, Alt-Godau), and an accept/reject/refine decision from
`d_F(q,A) ∈ [D - Σε, D + Σε]` where `Σε = ε_q + λ_q + ε_A + λ_A`. Two implementation
questions needed resolving before `src/traj/intervals.py` could be written.

## Decision 1: `Σε` is just the sum of the two sides' own stored `eps_A`, no separate `λ`

M1's `certify_spline_linearization` (section 2.4, the only spline path per ADR-0018)
already returns `eps_A = distance_upper(A, Lin(A')) + lam` -- i.e. exactly the spec's
own `ε_A + λ_A` sum, since section 2.4's own construction bounds `d_F(A, A')` via
`d_F(A, Lin(A')) + lam`, using the *same* `lam` that also builds `Lin(A')` (the
linearization the query-time `D` computation reuses directly, not recomputed).
`certify_polyline`'s `eps_A` has no separate `λ` term (polylines don't need
linearizing to run Alt-Godau, `λ=0`). So for either representation, `Σε` in section
2.5's formula is simply the sum of the two sides' own `eps_A` values -- tracking `λ`
separately anywhere in `intervals.py` or its callers would double-count.

## Decision 2: `decide_range`'s rule uses `decide()` only, and cheap filters are a separate, reported outcome

Per the task's explicit instruction, the accept/reject/refine rule is implemented via
`decide()` calls at the two threshold values (`r - Σε`, `r + Σε`), never via computing
`D`'s exact value with `distance()` -- each check is one `O(n*m)` feasibility test, not
a bisection to convergence. Both sides of the interval are checked explicitly and in the
spec's own literal order (accept first, then reject, else refine) -- not inferred from a
single computed distance.

Endpoint and bounding-box lower bounds (`endpoint_lower_bound`, `bbox_lower_bound`, spec
section 3's "P3 filters") are applied *before* any `decide()` call, rejecting outright
when the cheap bound alone already exceeds `r + Σε`. This is reported as a **distinct
outcome** (`"reject_cheap"`) from an interval-rule rejection (`"reject_interval"`) --
`benchmarks/step7_query.py`'s M3 report breaks down what fraction of all rejections came
from the free pre-filter vs. an actual `decide()` call, per the task's mandatory
correction, rather than collapsing both into one "rejected" bucket.

## Consequences

- `decide_range` returns one of four outcomes (`accept`, `reject_cheap`,
  `reject_interval`, `refine`), not three -- `benchmarks/step7_query.py`'s S5 metric
  ("fraction resolved without reading originals") is `accept + reject_cheap +
  reject_interval`, over either the whole candidate pool or the post-cheap-filter
  survivors (both reported, per the task's other mandatory correction).
- `r - Σε < 0` is guarded explicitly (skipped, not passed to `decide()`, which raises
  on a negative `eps`) -- no non-negative `D` could ever satisfy that threshold anyway.
- Correctness (spec section 9's property test, `tests/traj/test_intervals.py`): the
  interval `[D-Σε, D+Σε]` must contain the TRUE `d_F` between the original (raw)
  polylines on 10,000 random pairs -- verified directly, not just trusted from the
  individual certificates' own contracts.

## Links

`src/traj/intervals.py`; `src/traj/certify.py` (`certify_spline_linearization`,
`certify_polyline`); ADR-0018 (2.4-only spline certificates); `tests/traj/
test_intervals.py`; `benchmarks/step7_query.py`; `benchmarks/results/step7.md` (M3
section).
