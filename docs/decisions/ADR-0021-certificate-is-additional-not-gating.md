# ADR-0021: step8's reachability gate stays step3's honest-curve error; certificates are additional

- Status: Accepted
- Date: 2026-09-19

## Context

`docs/specs/step8_A_hybrid.md` section 1 ("Зависимость") says M0-M2 may use a dense-grid check
against the true curve if step7 isn't done yet, but once it is, step7's certificates
(`certify_polyline`/`certify_spline`, section 2.4 default, ADR-0018) should be used "for strict
error checking relative to the original curve." Step7 is done (gate G1 open), so M0 was asked to
bring the certificates in.

The first draft of the M0 plan read this as license to REPLACE step3's own reachability
criterion -- "honest error against the true noise-free curve on a dense grid `<= target_tol`"
(`step3_decisive.search_min_params`) -- with "certified `eps_A` against the recorded/sampled
track `<= target_tol`." The user rejected this at plan review: the two criteria measure different
things (one asks "does this reconstruction match the true road geometry", the other "does this
reconstruction match what was actually recorded"), and swapping the gate would (a) make step8's
byte counts incomparable to `benchmarks/results/step3.md`, and (b) silently change the meaning of
spec section 8's H1 acceptance rule (criterion A3), which is phrased in the true-curve-error
terms step3 already established.

## Decision

Reachability in `benchmarks/step8_hybrid.py` is decided EXACTLY as in `step3_decisive.py`:
honest error against the true noise-free curve on a dense grid, `<= target_tol`. The selection
logic (`search_min_params`/`_bisect_between`) is imported from `step3_decisive.py` unchanged, not
reimplemented, so there is no risk of an accidental behavioral drift between the two.

Step7's certificates are a SEPARATE, additional check, computed once per reachable cell for the
candidate step3's own criterion already chose (not swept during the coarse-scan/bisection search,
which keeps the search itself exactly as cheap as step3's): `eps_A = certify_polyline(track.xy,
kept_idx)` for DP+SED, `eps_A, _ = certify_spline(track.xy, bs)` (section 2.4 default) for the
LSQ-uniform spline. The report tracks the fraction of reachable cells with `eps_A <= target_tol`
as a correctness metric on the chosen representation relative to the actually-recorded track --
never as a pass/fail gate, and never substituted for the true-curve criterion.

This generalizes beyond M0: any future milestone that brings in a new, more rigorous check should
add it alongside an established criterion, not replace that criterion silently, unless a decision
explicitly says the criterion itself is changing.

### A secondary decision: segment type as one byte, not one bit

Spec section 2.5 describes each encoded segment's header as `[type (1 бит), число элементов
(varint)]`. `src/traj/encode.py` stores the type as a full byte. `zlib` compresses the seven
redundant bits away in practice, and the simplification is applied identically to every method
(polyline, spline, and eventually a hybrid mix) -- so it does not advantage one method's byte
count over another's, which is what section 2.5's own last sentence ("та же схема применяется...
чтобы сравнение было честным") actually protects. Bit-exact packing was not worth the added
complexity for a per-segment cost that's already 1 byte out of a payload of tens to hundreds of
bytes per segment (measured directly: `benchmarks/results/step8.md`'s M0 segment-overhead table).

## Consequences

- `benchmarks/results/step8.md`'s M0 numbers are directly comparable to `benchmarks/results/
  step3.md`'s dt=1 tables: reachability fractions and `n` (parameter counts) match exactly;
  byte counts differ by a few bytes per cell (typically 0-13B), attributable to `encode.py`'s
  explicit per-segment framing (`[type][count varint]`, absent from step3's ad hoc
  concatenate-and-compress scheme) rather than any change in what's being measured.
- The `eps_A <= target_tol` correctness tally is genuinely informative precisely because it is
  not gating: M0 found it can be far from 100% even for cells step3's criterion accepts,
  especially for the spline at tight `tol` (see `benchmarks/results/step8.md`'s M0 Conclusions)
  -- a real, reportable gap between "matches the true curve" and "certifiably matches what was
  recorded," not a bug, and not visible at all under a single merged criterion.
- M1-M4 should keep following this pattern: a milestone's own acceptance criteria (spec section
  8) are never silently redefined by a newly available check.

## Links

`src/traj/encode.py`, `benchmarks/step8_hybrid.py`, `benchmarks/results/step8.md` (M0),
`benchmarks/results/step3.md` (dt=1 comparison baseline), ADR-0018 (section 2.4 as the default
spline certificate), `docs/specs/step8_A_hybrid.md` sections 1, 2.2, 2.5, 8.
