# Prompt index

The prompt files in this directory are historical artifacts and are kept
in Russian, as originally written — they are not translated (see
`docs/prompts/step6_translate.md`). This index is translated to English.

step0-step2 were given verbally/in chat (not saved as files) — their gist
is summarized below, following [docs/history.md](../history.md). step3-step5
were given as files and live in this directory.

## step0 (in chat) — first measurement

A naive comparison of the raw polyline, the DP polyline, and a cubic
spline (`scipy.interpolate.splprep`, error controlled only at the
original timestamps) on 200 GeoLife tracks, by bytes, search latency,
and recall@10 by discrete Frechet distance.

## step1 (in chat) — honest methodology

Fix the two problems found in step0: clean raw GPS tracks of breaks/
jumps (`clean.py`) and augment spline fitting with an honest error check
BETWEEN timestamps, not just at them. Then re-compare the spline's
compression (hypothesis A) and kinematics reconstruction (hypothesis B)
against DP.

## step2 (in chat) — searching for a niche

Check whether there's a zone (measurement noise / target tolerance
ratio) where the honest spline is more compact than DP+SED, on more
realistic (road-like) synthetic geometry — straights with clothoid
turns.

## step3 — [`step3.md`](step3.md)

The decisive experiment: a fitter with no tethering to the polyline
(`src/traj/spline_lsq.py`), an oracle on the true curve, variable speed
and different observation steps, three criteria (K1-K3) fixed before the
run.

## step4 — [`step4_writeup.md`](step4_writeup.md)

The final research writeup with no new experiments:
`docs/findings.md`, a rewritten `README.md`, closing out the research
part in `TODO.md`, a draft of `docs/blog_draft.md`.

## step5 — [`step5_history.md`](step5_history.md)

The full project history (including the pre-repository stage) and the
mandatory correction to step4's conclusion: `docs/history.md`, edits to
`docs/findings.md` and `docs/blog_draft.md`, `docs/next_steps.md`, this
index, a rewritten `README.md`.
