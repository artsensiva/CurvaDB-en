# CurvaDB — AI context

The research (a GPS trajectory store on cubic B-splines with search by
discrete Frechet distance) is complete; findings are in
`docs/findings.md`. New code only belongs in a separate branch. The old
code (a Hilbert index over embeddings, src/level1) must NOT be touched
or extended.

- Main branch: main. Create a separate branch for new experiments.
- Python environment: venv/ at the repo root (Python 3.14), with numpy,
  scipy, shapely, pytest, numba installed. Don't add new dependencies
  without asking.
- Data: data/geolife/<user_id>/Trajectory/*.plt, the first 6 lines of
  each .plt are a header. data/ and venv/ are in .gitignore, don't
  commit them.
- .gitignore ignores *.csv, *.json, *.txt — that's expected, don't
  change it. Commit benchmark results as .md.
- New code: src/traj/, tests: tests/traj/, benchmarks: benchmarks/.
- Run commands via venv/bin/python and venv/bin/pytest.
- Commit after each completed item. Record anything unfinished in
  TODO.md.
- A negative benchmark result is a normal result — don't force it.

## Current work (roadmap)
- The roadmap is docs/ROADMAP.md; specs are in docs/specs/ (written in Russian).
- "Репозиторий CurvaDB" in specs step7 and step8 means this repository.
- Code, comments and reports are written in English.
- One active phase branch at a time (step7, step8, core-extract); merge to main only after the gate in docs/ROADMAP.md is checked.
- Acceptance sections in docs/specs/ must never be edited during a phase.
- Products P1-P4 are NOT implemented in this repository; they get their own repositories when their gates open.
