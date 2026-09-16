# CurvaDB

Русская версия: [README.ru.md](README.ru.md)

Research question: does storing GPS trajectories as cubic B-splines beat
the classic "Douglas-Peucker (DP) simplification + search by discrete
Frechet distance" scheme? Bottom line: a negative result on compression
for consumer GPS; for exact data, no advantage was found under the
conditions tested either, but the question isn't fully closed (see
hypothesis H1 in [docs/findings.md](docs/findings.md)). The project is
ending as a research effort — the next step is not code, but industry
interviews (see [docs/next_steps.md](docs/next_steps.md)).

## Stages

| Stage/step | Question | Answer |
|---|---|---|
| A. Original idea | Semantic search via "curves" over embeddings? | Implemented (Level 1-2), but never compared against a baseline. |
| B. Critique | Is the idea mathematically sound? | No -- 4 unrelated problems, the embedding's coordinate order is arbitrary. Pivot to GPS trajectories. |
| C. Fact-checking | Is the niche empty, was the comparison honest? | The niche isn't empty; the "80KB/2-4KB" comparison was unfair; 10-15 interviews were recommended -- not conducted. |
| D. Organization | -- | Branch `trajectory-pivot`, GeoLife data, prompts as files. |
| step0 | How does a naive spline compare to DP? | Crushingly worse (recall 0.707 vs 0.997) -- turned out to be a methodological error (`step0_diagnostics.md`). |
| step1 | What does an honest methodology show? | Recall nearly matched (0.972/0.996), but DP wins on compression at every tol (3.7x). |
| step2 | Is there a compression niche by noise/tol? | A narrow zone was found (tol=1-5m, sigma<=0.1m) -- but only for RTK/lidar-grade accuracy. |
| step3 | Does the hypothesis survive a decisive test without polyline tethering? | All three criteria (K1-K3) fail; even the oracle isn't more compact than DP+SED. **step2's niche conclusion is overturned.** |
| Resolution | What's next? | Not code -- 8-10 industry interviews; return-to-code threshold ≥3/10. |

Full timeline with numbers and commits -- [docs/history.md](docs/history.md).

## Documents

- [docs/history.md](docs/history.md) -- the full project history, from
  the original idea to the resolution.
- [docs/findings.md](docs/findings.md) -- research findings: the
  question, key numbers, conclusion, limitations, open hypotheses H1/H2.
- [docs/next_steps.md](docs/next_steps.md) -- the resolution, the
  interview plan, the return-to-code threshold, a sketch of an H2
  experiment.
- [docs/blog_draft.md](docs/blog_draft.md) -- the text of a blog post
  for publication (for an engineering audience).
- [docs/legacy.md](docs/legacy.md) -- an archive of the README from
  before the pivot to trajectories (the original embeddings idea).
- [docs/prompts/](docs/prompts/) -- the step3-step5 prompts (step0-step2
  were given in chat, summarized in `docs/prompts/README.md`).
- [benchmarks/results/](benchmarks/results/) -- raw results for each
  step (step0.md, step0_diagnostics.md, step1.md, step2.md, step3.md).

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Data -- GeoLife Trajectories 1.3 (`data/geolife/<user_id>/Trajectory/*.plt`),
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
```

Each script writes its own section to `benchmarks/results/<step>.md`.

## Known alternatives

- **PostGIS** ([`ST_FrechetDistance`](https://postgis.net/docs/ST_FrechetDistance.html))
  -- the discrete Frechet distance as a built-in SQL function over
  `geometry`; a ready industrial solution with no need to write your own
  DP.
- **[MobilityDB](https://mobilitydb.com/)** -- a PostgreSQL/PostGIS
  extension for trajectory data (`tgeompoint` etc.), with temporal
  operations, indexes, and similarity metrics out of the box.
- **Map-matching** (e.g. [Valhalla](https://github.com/valhalla/valhalla),
  [OSRM](https://project-osrm.org/)) -- snapping a GPS track to the road
  network; a different approach to trajectory compression/representation
  that removes some noise using external road-graph information, not
  just the track's geometry.
- **Kalman smoothing** (constant-acceleration + RTS) -- in step1
  (`docs/findings.md`, `benchmarks/results/step1.md` §4) it showed the
  best average accuracy for recovering velocity/acceleration among all
  three methods compared, at the cost of noticeably lower recall for
  detecting sharp maneuvers -- an alternative for tasks where kinematics
  matters more than geometric compression/search.

## Old project

Level 1 (a Hilbert index over text embeddings, `src/level1`) is not
developed further on this branch. The archive of the previous README is
in [docs/legacy.md](docs/legacy.md).
