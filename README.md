# CurvaDB

Русская версия: [README.ru.md](README.ru.md)

The project now has two parts. The first (`step0`-`step3`, `step8`) is a completed research
study: does storing GPS trajectories as cubic B-splines beat the classic "Douglas-Peucker (DP)
simplification + search by discrete Fréchet distance" scheme? Bottom line: a negative result on
compression for consumer GPS, and — after step8 closed the question step3 left open — for
high-precision data too: even a free-knot spline oracle fit directly to the exact curve isn't
20% more compact than DP+SED at any tested tolerance (see hypothesis H1, now closed, in
[docs/findings.md](docs/findings.md)). That study concludes
as research — the next step for it is not code, but industry
interviews (see [docs/next_steps.md](docs/next_steps.md)).

The second part, **step7**, is a positive result: a certified curve store that answers range
queries over compressed trajectories with a provable guarantee on the true continuous Fréchet
distance, cheaper than reading the originals for most candidates. Gate G1
(`docs/ROADMAP.md`) is open — see
[Certified curve store (step7)](#certified-curve-store-step7) below.

## Stages

| Stage/step | Question | Answer |
|---|---|---|
| A. Original idea | Semantic search via "curves" over embeddings? | Implemented (Level 1-2), but never compared against a baseline. |
| B. Critique | Is the idea mathematically sound? | No — 4 unrelated problems, the embedding's coordinate order is arbitrary. Pivot to GPS trajectories. |
| C. Fact-checking | Is the niche empty, was the comparison honest? | The niche isn't empty; the "80 KB vs 2–4 KB" comparison was unfair; 10-15 interviews were recommended — not conducted. |
| D. Organization | — | Branch `trajectory-pivot`, GeoLife data, prompts as files. |
| step0 | How does a naive spline compare to DP? | Much worse (recall 0.707 vs 0.997) — turned out to be a methodological error ([benchmarks/results/step0_diagnostics.md](benchmarks/results/step0_diagnostics.md)). |
| step1 | What does an honest methodology show? | Recall nearly matched (0.972/0.996), but DP wins on compression at every tol (3.7x at tol = 10 m). |
| step2 | Is there a compression niche by noise/tol? | A narrow zone was found (tol = 1–5 m, σ ≤ 0.1 m) — but only for RTK/lidar-grade accuracy. |
| step3 | Does the hypothesis survive a decisive test without polyline tethering? | All three criteria (K1-K3) fail; even the oracle (uniform knots) isn't more compact than DP+SED. **step2's niche conclusion is overturned.** |
| Resolution | What's next? | Not code — 8-10 industry interviews; return-to-code threshold ≥3/10. |

Full timeline with numbers and commits — [docs/history.md](docs/history.md).

## Certified curve store (step7)

A second research question, further along in the same repository: can a curve store answer
range queries over *compressed* trajectories with a provable guarantee on the true continuous
Fréchet distance, instead of reading the original data for every candidate? Answer: yes, with
real, quantified trade-offs. Gate G1 (`docs/ROADMAP.md`) is open.

- Exact-guarantee search on compressed data: **0 misses and 0 false positives across 6,000,000**
  (query, candidate, range, representation) combinations tested.
- The certified method is **twice as fast as an exact filter without compression** (20.1 ms vs.
  41.7 ms median per query) — even though it is slower than an *uncertified* approximate search
  on the same compressed data, a separate, documented trade-off (S6b).
- The price of dropping the guarantee, quantified: an uncertified approximate search on the same
  compressed data still misses **0.03-3.03%** of true answers and returns up to **2.04%** false
  positives.
- A tolerance/size trade-off for the polyline representation: tightening the simplification
  enough to resolve short-range queries without reading originals costs a real **5.1x** more
  storage per track.
- A negative result inside the positive one: the spec's own *primary* spline certificate
  (section 2.3) is available for only **16.4%** of real tracks and, where available, is **1.68x**
  looser than its own documented fallback (section 2.4) — resolved by making that fallback the
  default (`docs/decisions/ADR-0018-2-4-primary-spline-certificate.md`).

Full detail: [docs/phases/step7_summary.md](docs/phases/step7_summary.md) (one-page gate
summary), [docs/specs/step7_B_certified_store.md](docs/specs/step7_B_certified_store.md) (the
spec), [benchmarks/results/step7.md](benchmarks/results/step7.md) (all milestone results, M0-M4),
[docs/decisions/](docs/decisions/) (20 ADRs, the full decision history).

## Documents

- [docs/history.md](docs/history.md) — the full project history, from
  the original idea to the resolution.
- [docs/findings.md](docs/findings.md) — research findings: the
  question, key numbers, conclusion, limitations, closed hypothesis H1, open hypothesis H2.
- [docs/next_steps.md](docs/next_steps.md) — the resolution, the
  interview plan, the return-to-code threshold, a sketch of an H2
  experiment.
- [docs/blog_draft.md](docs/blog_draft.md) — the text of a blog post
  for publication (for an engineering audience).
- [docs/blog_step7.md](docs/blog_step7.md) — a follow-up blog post on
  step7 (the certified curve store).
- [docs/legacy.md](docs/legacy.md) — an archive of the README from
  before the pivot to trajectories (the original embeddings idea).
- [docs/prompts/](docs/prompts/) — the step3-step5 prompts (step0-step2
  were given in chat, summarized in `docs/prompts/README.md`).
- [benchmarks/results/](benchmarks/results/) — raw results for each
  step (step0.md, step0_diagnostics.md, step1.md, step2.md, step3.md, step7.md).
- [docs/ROADMAP.md](docs/ROADMAP.md) — the step7 phase/gate roadmap (G0-G7),
  the decision log, and the risk register.
- [docs/decisions/](docs/decisions/) — architecture decision records (ADRs)
  for step7: 20 entries, template and index in `docs/decisions/README.md`.
- [docs/phases/](docs/phases/) — one-page summaries written at each gate
  (currently `step7_summary.md`, gate G1).

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements-research.txt
```

(`requirements.txt` belongs to the old Level 1/2 project, `src/level1`;
the trajectory research uses `requirements-research.txt`.)

Data — GeoLife Trajectories 1.3 (`data/geolife/<user_id>/Trajectory/*.plt`),
not included in the repository (see `.gitignore`).

## Reproducing

```bash
venv/bin/pytest tests/traj/                          # tests

venv/bin/python benchmarks/step0.py                  # first (broken) measurement
venv/bin/python benchmarks/step1_clean.py             # data cleaning
venv/bin/python benchmarks/step1_spline_fit.py         # honest dense fitting
venv/bin/python benchmarks/step1_compression.py        # compression (hypothesis A)
venv/bin/python benchmarks/step1_kinematics.py         # kinematics (hypothesis B)
venv/bin/python benchmarks/step1_search.py             # recall@10
venv/bin/python benchmarks/step2_crossover.py --pilot  # niche search (pilot)
venv/bin/python benchmarks/step2_crossover.py          # niche search (full run)
venv/bin/python benchmarks/step3_decisive.py --pilot   # decisive experiment (pilot)
venv/bin/python benchmarks/step3_decisive.py           # decisive experiment (full run)

# step7 -- certified curve store (docs/specs/step7_B_certified_store.md)
venv/bin/python benchmarks/step7_certify.py            # M0-M1: polyline + spline (section 2.4) certificates
venv/bin/python benchmarks/step7_m13_fitters.py        # M1.3: fitter comparison (fit_adaptive vs spline.fit())
venv/bin/python benchmarks/step7_m13_tail.py           # M1.3: eps_A/LB tail verification
venv/bin/python benchmarks/step7_m2_projection.py      # M2: primary spline certificate (section 2.3), full corpus
venv/bin/python benchmarks/step7_query.py              # M3: interval range queries, full corpus + near-duplicates
venv/bin/python benchmarks/step7_m4.py                 # M4: error rates, tol/size trade-off, per-query latency
```

Each script writes its own section to `benchmarks/results/<step>.md`.

## Known alternatives

- **PostGIS** ([`ST_FrechetDistance`](https://postgis.net/docs/ST_FrechetDistance.html))
  — the discrete Fréchet distance as a built-in SQL function over
  `geometry`; a production-ready option with no need to write your own
  DP.
- **[MobilityDB](https://mobilitydb.com/)** — a PostgreSQL/PostGIS
  extension for trajectory data (`tgeompoint` etc.), with temporal
  operations, indexes, and similarity metrics out of the box.
- **Map-matching** (e.g. [Valhalla](https://github.com/valhalla/valhalla),
  [OSRM](https://project-osrm.org/)) — snapping a GPS track to the road
  network; a different approach to trajectory compression/representation
  that removes some noise using external road-graph information, not
  just the track's geometry.
- **Kalman smoothing** (constant-acceleration + RTS) — in step1
  (`docs/findings.md`, `benchmarks/results/step1.md` §4) it showed the
  best average accuracy for recovering velocity/acceleration among all
  three methods compared, at the cost of noticeably lower recall for
  detecting sharp maneuvers — an alternative for tasks where kinematics
  matters more than geometric compression/search.

## Old project

Level 1 (a Hilbert index over text embeddings, `src/level1`) is not
developed further in this repository. The archive of the previous README is
in [docs/legacy.md](docs/legacy.md).
