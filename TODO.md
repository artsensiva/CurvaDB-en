# TODO

The research part of the project is complete (step0-step3, step7, step8) and published:
full technical report in [docs/REPORT.md](docs/REPORT.md) / [docs/ru/REPORT.md](docs/ru/REPORT.md),
citation metadata in [CITATION.cff](CITATION.cff), DOI 10.5281/zenodo.22850469. Everything
below is genuinely open, not a leftover from a finished milestone.

- **Hypothesis H2 (product): analytical derivatives and search by kappa(t) on exact data** --
  untested, demand unconfirmed. See [docs/findings.md](docs/findings.md) ("Open hypotheses")
  and the experiment sketch in [docs/next_steps.md](docs/next_steps.md) ("Sketch of an H2
  experiment"), to be filled in only after the interview threshold below is met.

- **Industry interviews were never conducted**, for the entire project (recommended since
  Stage C, `docs/history.md`). This blocks every remaining gate: see
  [docs/ROADMAP.md](docs/ROADMAP.md) section 6 for the interview plan and domains, and section 4
  (gates G4, G6, G7) for the exact demand thresholds (3 of 10, independently, per domain) each
  remaining phase needs before it can start.

- **Phases 3-7 (the `certigeo` core extraction, products P1-P4) have not started.** Phase 3
  (core extraction) only needs gate G1 (already open) but has no product to justify extracting
  it yet; phases 4-7 (P1-P4) additionally need their own interview thresholds (gates G4, G6, G7
  above). See [docs/ROADMAP.md](docs/ROADMAP.md) section 3 (phase table) and section 4 (gates
  G3-G7) for what each phase requires, and `docs/specs/P1_certipath_cnc.md` /
  `P2_certiinspect_surfaces.md` / `P3_certitrack_store.md` / `P4_motionshape_search.md` for what
  each product actually is.

- **step8's own open question, not pursued further**: whether a different free-knot placement
  strategy (not `src/traj/knot_removal.py`'s one-shot-ranked, bisection-refined greedy removal)
  could close more of criterion A3's gap. Per the H1 decision rule (`docs/reviews/step8_M1.md`,
  `docs/phases/step8_summary.md`), not worth testing further once A3 failed on this milestone's
  own implementation -- listed here only for completeness, not as an active task.
