# BlueDot state-dependence arm: execution decision tree

Operational companion to `docs/bluedot/preregistration_state_dependence.md`. That document fixes
the science; this one fixes the order of operations and the gate at every branch point.

Written 2026-07-27, before any calibration, training, or final-test run. Nothing here has been
executed.

Rule that governs every gate below: **a gate is evaluated once, against a condition written down
before the data existed.** If a gate fails, take the branch this document names. Do not invent a
third branch after seeing the numbers.

## Stage map

```text
G0  scope and preregistration frozen
      |
G1  prompt manifest (168 prompts, offline, no model)
      |
G2  direction bank and projection matrix (model loaded, no scoring)
      |
G3  smoke run (8 prompts, 144 forwards)
      |
G4  calibration at layer 13 (32 prompts, 2,624 forwards)
      |-- pass --> G6
      |-- fail --> G5
      |
G5  calibration at layer 20 (32 prompts, 2,624 forwards)
      |-- pass --> G6
      |-- fail --> STOP-A (study stops under this design)
      |
G6  training run and fits (96 prompts, 1,728 forwards)
      |
G7  wrong-state matching and permutations (no forwards)
      |
G8  final-test forecast commitment (512 commitments, no forwards, no outcomes)
      |
G9  final-test resolution (32 prompts, 576 forwards)
      |
G10 verification
      |
G11 analysis, run once
      |
G12 writeup
```

## G0. Scope and preregistration frozen

**Entry condition:** none. This is the start.

**Done when:**

* `docs/bluedot/preregistration_state_dependence.md` exists and is committed.
* `docs/preregistration.md` carries an amendment-log entry pointing to it and stating that the
  original governs everything except this arm.
* `docs/experiment_log.md` carries a dated entry recording the scope amendment as a
  preregistration, not a result.
* The offline test and quality suite passes.

**Gate:** if any frozen decision in the preregistration is still open, close it there before
writing code. A decision made during implementation is a researcher degree of freedom.

## G1. Prompt manifest

**Status: passed, 2026-07-27.** Values are in `docs/experiment_log.md`; the frozen split is
`data/prompt_manifests/bluedot_state_dependence_v1.json`, manifest hash
`sha256:bf351c9d73042fcb3d0cdcb18247ded3411413d73000e047f1ba6ba9ab5b25b9`, drawn from an
eligible pool of 256 groups.

**Prerequisite that is not compute:** satisfied. The ARC pool was re-prepared and now holds 256
items, 256 groups, and 2,048 variants. `csf data prepare` needs network access to
`allenai/ai2_arc` and cannot run under `HF_HUB_OFFLINE=1`, so it is done once, outside the
offline pipeline.

**Work:** `PromptRole` enum, `PromptManifest` and `PromptAssignment` records, deterministic role
assignment from `derive_seed("bluedot.prompt_manifest", 20260727, group_id)`,
`csf prompts manifest`, and `csf prompts verify`.

**No model is loaded at this stage.**

**Gate G1:**

| Check | Pass condition |
| --- | --- |
| Pool size | at least 168 eligible groups after the existing exclusion rules |
| Counts | exactly 8 / 32 / 96 / 32 |
| Disjointness | no `item_id` and no `group_id` appears in two roles |
| Determinism | building twice from the same task and seed is byte-identical |
| Provenance | the manifest cites the task manifest hash |
| Independence | assignment is a function of `group_id` and the seed only, asserted by test |

All six passed. 256 eligible groups against 168 required; counts 8 / 32 / 96 / 32; no group id
and no item id shared between any two roles; the deterministic payload is byte-identical across
rebuilds and a rerun leaves the file untouched; the manifest cites the task manifest, items, and
variants hashes and `csf prompts verify` confirms they still match.

**Fail branch:** if the pool is short, **STOP-B**. Report the shortfall. Do not shrink a role,
reuse a group, or relax disjointness.

## G2. Direction bank and projection matrix

**Direction bank: passed, 2026-07-27.** Frozen at
`data/direction_manifests/bluedot_state_dependence_directions_v1.json`, family hash
`sha256:809fbb5b033da740a01574ad5a0ca48f34baca38eef1504a0d66bba8e2fb9138`. Values and
diagnostics are in `docs/experiment_log.md`. **The projection matrix `P` is not built yet**;
it is slice B2b and needs no model.

**Work:** read the unembedding rows for tokens 562, 603, 565, 622; build four centered unit
answer directions; build four Gaussian controls orthogonal to their span and to each other;
store all eight in `DirectionStore` with metadata; generate and store the fixed `1152 x 16`
orthonormal projection matrix `P`.

The model is loaded to read the unembedding. **No prompt is scored and no state is captured at
this stage.**

**Gate G2:**

| Check | Pass condition |
| --- | --- |
| Dimension | all eight vectors are 1152-dim float32, finite |
| Norms | all eight have unit norm within `1e-6` |
| Answer span | the four centered directions span 3 dimensions, as the design predicts |
| Orthogonality | every control-to-answer and control-to-control absolute cosine is at most `1e-6` |
| Determinism | regenerating from the same seed reproduces every vector exactly |
| Projection | `P` has orthonormal columns within `1e-6`, and is injective on the realized 16 signed vectors |
| Privacy | no family label or meaningful id reaches `public_metadata` or the candidate public view |
| Provenance | a `DirectionFamilyRecord` cites the model revision, the accessor used, the token ids, the seed, and eight file hashes |

Every direction-bank check passed. Answer span rank 3, as the design predicts. Worst
answer-to-random absolute dot 2.82e-09 and worst random-to-random 2.30e-09, both far inside the
1e-5 tolerance for saved float32 vectors. Regeneration from the pinned model reproduced every
vector hash and the family hash exactly, and a second build reported `unchanged` without
touching the manifest bytes or mtime.

One recorded fact that was not predicted in advance: Gemma 3 1B **ties** its input and output
embeddings. The accessor reads that off the tensors rather than the config flag, and both agree.
The preferred `get_output_embeddings()` path was used, so the fallback was not exercised on the
real model.

**Fail branch:** fix the construction and regenerate. This gate has no scientific content; a
failure here is a bug, not a finding.

## G3. Smoke run

**Status: passed, 2026-07-28.** Executed as `results/runs/bluedot-smoke-layer13`, run manifest
hash `sha256:6f16b22d9887bcff38e3192205e3cf74f9c794b709437fa9d78a664276c67b3f`, 144 forwards, 136
observations, no failures. Measured values are in `docs/experiment_log.md`. The run carries
`scientific_result: false` and selected nothing.

**Purpose:** prove the pipeline end to end on real weights at a scale where a mistake is cheap.
8 prompts x 17 candidates plus 8 clean forwards = **144 forwards**, roughly 68 s of forward time
plus one model load.

Uses the smoke ratio: run the smoke prompts at the **middle** grid ratio, 0.10. This choice is
arbitrary and is fixed here so it cannot be chosen later; the smoke run is a plumbing check and
its numbers are not used to select anything.

**Gate G3:**

| Check | Pass condition |
| --- | --- |
| Hook fired | every intervention records `fired`, no `CaptureError` |
| No-op | every no-op has `abs(delta_clean_top_margin) <= 1e-3` and `delta_norm == 0.0` |
| Clean reproduction | the no-op reproduces the clean output within tolerance, and a second run of the same command reproduces every clean logit |
| Target definition | `intervened_top_margin < 0` coincides exactly with `answer_flip` |
| Determinism | rerunning reproduces every target bit-for-bit |
| Failures | `state_audit_failures.jsonl` is empty |
| Classification | the run manifest says `engineering_smoke`, and `scientific_result` is false |

All seven passed. Every capture hook and every intervention hook fired (8 and 136); every no-op
reproduced the clean logits exactly, so the worst absolute no-op target and the worst no-op
`delta_norm` were both 0.0; the intervened residual stream, re-read after the intervention hook
in the same forward, matched `h + sign * alpha * d` to 0.0 across all 136 candidates. A second
execution into a separate run id reproduced every one of the 136 targets and every intervened
logit with a maximum absolute difference of 0.0, and produced the same reference norm and alpha.
`csf state-audit verify-run` returns `valid: true` with no failures.

Two clean-reproduction notes. The check named in the original version of this row referred to a
committed clean margin from an earlier process, which does not exist at this stage: the clean
forward and the intervened forwards happen in one run, so the same-process check is the no-op and
the cross-process check is the determinism rerun. Both were done.

One recorded fact that was not predicted in advance: at ratio 0.10 the layer-13 effect
distribution on the smoke prompts is large, with a 95th percentile of 4.65 in absolute target.
That is above the preregistered C5 ceiling of 4.0. **It is not a calibration result and it does
not move the grid.** C5 is evaluated over the 32 calibration prompts at G4, which are disjoint
from these 8, and the ratio the study uses is whatever that evaluation selects. Recording the
observation here means it cannot later be presented as a surprise.

**Fail branch:** fix and rerun. A smoke failure is an engineering failure and never advances the
study.

## G4. Calibration at layer 13

**Not run.** The plan is frozen at
`data/calibration_plans/bluedot_state_dependence_calibration_v1.json`, plan hash
`sha256:a212c6e80f96db55e1aeb0b1879fef441aa6d893a73b7fc43a94142913f97877`, and the conditions,
the strength rule, and the selector are implemented and tested. No calibration has been
executed, no state norm has been measured, and no ratio has been selected.

The gate below is now machine-checkable rather than a checklist:
`csf calibration summarize` computes one summary per ratio from supplied observations, and
`csf calibration select` applies the state machine. The selector reads summaries only; it loads
no model and cannot see a prompt.

**Work:** capture clean states for all 32 calibration prompts at layer 13; compute
`reference_norm` as the median clean-state norm; for each of the five ratios set
`alpha = ratio * reference_norm`; apply all 16 signed directions at all five ratios plus one
no-op. 32 x 82 = **2,624 forwards**, roughly 21 minutes of forward time.

Write a `CalibrationRecord` carrying `reference_norm`, the five alphas, the full effect
distribution per ratio, and the evaluation of each condition.

**Gate G4, evaluated per ratio in ascending order. Select the smallest ratio passing all five:**

| # | Condition |
| --- | --- |
| C1 | all label logits and targets finite |
| C2 | no-op within harness tolerance: `csf interventions validate` passes, and every no-op has `abs(delta_clean_top_margin) <= 1.0e-3` |
| C3 | at least 15 percent of non-no-op effects have `abs(target) >= 0.10` |
| C4 | median `abs(target)` over non-no-op effects is at least 0.05 |
| C5 | 95th percentile of `abs(target)` over non-no-op effects is at most 4.0 |

**Pass branch:** record the selected ratio and its alpha. Go to G6. The layer-20 fallback is now
permanently unavailable for this study; it is not reconsidered later for any reason.

**Fail branch (no ratio passes):** go to G5. This is the only trigger for the fallback. A ratio
that passes but produces a disappointing effect distribution is a pass, and the study proceeds
with it.

## G5. Calibration at layer 20

**Only reachable from a G4 fail.** Repeat G4 identically at layer 20: same 32 prompts, same
direction bank, same five ratios, same conditions, same `reference_norm` procedure recomputed at
layer 20. 2,624 forwards, roughly 21 additional minutes.

**Pass branch:** record the selected layer and ratio. Go to G6.

**Fail branch: STOP-A.** The study stops under this design. Do not:

* search a third layer;
* widen, extend, or re-space the ratio grid;
* change the direction bank or add directions;
* switch the target definition;
* train a model organism;
* learn a behavioral direction;
* substitute a different task or model.

Write it up as a negative engineering result: this intervention family, at these two layers, over
this ratio grid, does not produce a measurable and non-degenerate effect distribution on this
model. That is a real finding about the design and it is reported as one.

## G6. Training run and fits

**Work:** 96 prompts at the selected layer and ratio, 17 candidates each, plus 96 clean forwards
= **1,728 forwards**, roughly 14 minutes.

Then, on the training rows only (96 x 16 non-no-op = 1,536 rows):

1. fit the TF-IDF vocabulary and idf, and the truncated SVD to `k_svd`;
2. fit the state PCA to 16 components;
3. fit the feature standardizer;
4. select each method's ridge alpha by 6-fold grouped cross-validation over the 96 prompt groups;
5. fit final coefficients on all 96 training groups at the selected alpha;
6. write a `TransformFitRecord` per fitted object.

**Gate G6:**

| Check | Pass condition |
| --- | --- |
| Fit boundary | every `TransformFitRecord` names exactly the 96 training prompt ids and no other |
| Role assertion | fitting on a calibration or final-test prompt raises |
| CV disjointness | no `group_id` appears in both the fit and the held-out part of any fold |
| CV determinism | folds reproduce from the seed |
| Alpha interior | the selected ridge alpha is not at an edge of the grid; if it is, record it and proceed, since moving the grid after seeing training results is a degree of freedom |
| Feature blocks | block widths match the preregistration; block contents pass the leakage tests |
| No leakage | the visible-information feature vector contains no state-derived quantity, asserted field by field |

**Fail branch:** a fit-boundary or leakage failure means refit from scratch after fixing the
cause. Do not reuse the artifact.

## G7. Wrong-state matching and permutations

**No forwards.** Build the deterministic nearest match for each of the 32 final-test prompts from
the other 31, by the five-step rule in the preregistration, and the ten seeded derangements.

This requires the final-test **clean** captures, which are produced at G9 as part of the
resolution run. To keep the ordering honest, the final-test clean forward pass is separated from
the intervened forward passes: clean captures happen here, intervened forwards happen at G9,
after commitment.

**Gate G7:**

| Check | Pass condition |
| --- | --- |
| No self-match | no prompt is its own donor, in the nearest match or in any permutation |
| Determinism | matching and all ten derangements reproduce from the seed |
| Block isolation | the wrong-state feature vector differs from the true-state vector in the S and SxI blocks only, asserted block by block |
| Intervention fixed | the target prompt's signed intervention vector and its 16-dim projection are byte-identical between conditions |
| Provenance | the pairing artifact is written and hashed before G8 |
| Clean-only | no intervened final-test forward has run |

**Fail branch:** fix and regenerate. Nothing here is scientific.

## G8. Final-test forecast commitment

**No forwards. No outcomes may exist.**

Predict all 17 candidates for each of the 32 final-test prompts, under all 16 method-and-condition
combinations, and commit each as a salted forecast record: **512 commitments**.

**Gate G8, checked before the first commitment is written:**

| Check | Pass condition |
| --- | --- |
| No outcomes | `observations.jsonl`, `resolution.json`, and `scores.json` do not exist in the run directory, and `commit_forecast` refuses if any does |
| Key structure | the commitment key carries trial id, method id, state condition, and condition index |
| Salt paths | every commitment has its own salt file; no path collides |
| Completeness | 32 x 16 = 512 records exist, each covering all 17 candidates |
| Finite | every committed number is finite; canonical serialization succeeds |
| Intervals | every `q05 <= q95` |

**Fail branch:** if any outcome artifact already exists for the final-test run, **STOP-C**. The
blinding is gone for that run and no repair restores it. Regenerate the final-test trials into a
fresh run directory and recommit; record the discarded run in the experiment log so the discard
is visible rather than silent.

## G9. Final-test resolution

**Only after G8 completes.** Apply all 17 candidates to each of the 32 final-test prompts:
32 x 17 = 544 intervened forwards, plus the 32 clean forwards already taken at G7 =
**576 forwards** for the role, roughly 5 minutes.

Resolve in all-candidate mode with a no-selection reveal, so every commitment is revealed and
verified.

**Gate G9:**

| Check | Pass condition |
| --- | --- |
| Coverage | every prompt-candidate pair is observed or recorded as a failure; none silently missing |
| Clean reproduction | the one-time guard passes |
| No-op | every no-op is within tolerance |
| Reveals | 512 reveals written, all `verified: true` |
| Failures | `resolution_failures.jsonl` is empty, or every entry is counted and reported |

**Fail branch:** a failed verification is recorded as evidence, never re-resolved away. Go to G10
and report it.

## G10. Verification

Run the study verifier. It must check, from artifacts rather than from prose:

| Check | Evidence |
| --- | --- |
| Ordering | no outcome artifact predates commitment; every `committed_at` precedes every `observed_at` |
| Commitment integrity | every hash recomputes from the forecast on disk and the revealed salt |
| Prompt roles | disjoint by item and group; counts are 8 / 32 / 96 / 32 |
| Transform provenance | every fit names only training prompts |
| Direction integrity | the eight direction hashes and the projection hash match the artifacts cited by the run |
| Artifact integrity | the resolution manifest's provenance hashes match the files on disk |
| Manifest hashes | prompt manifest, calibration record, and pairing artifact hashes match |

**Gate G10:** every check passes, or the analysis is not run as a scientific result. A run that
does not verify is reported as a run that did not verify.

## G11. Analysis, run once

**No model is loaded.** This stage recomputes from committed artifacts only; make the loader
raise, so an accidental forward pass is impossible.

1. Aggregate each method's absolute error over the 16 non-no-op interventions, per prompt.
2. Compute the two primary paired differences.
3. Run the 10,000-resample paired bootstrap over prompt groups from the frozen seed.
4. Apply the decision rule: an interval crossing zero is no detected difference.
5. Compute the secondary metrics, omitting Brier if there are fewer than 20 flips and reporting
   the realized flip count.
6. Report cross-validated training results and the single final-test result separately.

**Gate G11:** the analysis runs once. No method is added, dropped, retuned, or refitted after any
final-test outcome is seen. If a bug is found in the analysis code after the numbers exist, fix
it, rerun the whole analysis, and record both the bug and the change in the experiment log.

## G12. Writeup

**Gate G12:**

| Check | Pass condition |
| --- | --- |
| Claim scope | every claim sits inside `docs/claim_boundaries.md`, including the BlueDot arm section |
| No introspection language | no claim about introspection, consciousness, self-awareness, faithful verbal reasoning, hidden goals, or deployment readiness |
| Counts | every number carries its prompt count, its prompt-group count, and an interval |
| Nulls | a null or negative result is reported as prominently as a positive one |
| Predictions | section 12 of the preregistration is compared against the outcome, and the comparison is stated |
| Provenance | every measured number cites a verified run artifact |
| No planned values | no planned or predicted value appears in a results section |

## Stop codes

| Code | Trigger | Action |
| --- | --- | --- |
| STOP-A | No ratio passes calibration at layer 13 and none at layer 20 | Stop the study under this design. Write up the negative engineering result. |
| STOP-B | Fewer than 168 eligible disjoint groups | Stop at G1. Report the shortfall. Do not shrink a role or relax disjointness. |
| STOP-C | Final-test outcome artifacts exist before commitment | Discard that run directory, regenerate into a fresh one, recommit, and record the discard. |
| STOP-D | A required intervention control fails at the study config | Stop. Every downstream number would be untrustworthy. |
| STOP-E | A transform is found fitted outside the training role | Refit from scratch. Do not reuse the artifact. |
| STOP-F | Documentation would have to describe a planned value as measured | Stop and fix the document. |

## What is explicitly not a decision point

These are settled by the preregistration and are not reopened during execution:

* which layer to try after layer 20 (none);
* whether to widen the ratio grid (no);
* whether to add a method or an MLP (no);
* whether to drop a prompt for being answered wrongly on the clean run (no);
* whether to pool cross-validated and final-test results (no);
* whether an interval crossing zero can be described as a trend (no).

## Compute per stage

From the measured median forward time `T = 0.474158 s`. Planning estimates, not measurements.

| Stage | Forwards | Estimated forward time |
| --- | --- | --- |
| G3 smoke | 144 | 1.1 min |
| G4 calibration, layer 13 | 2,624 | 20.7 min |
| G5 calibration, layer 20 (only on a G4 fail) | 2,624 | 20.7 min |
| G6 training | 1,728 | 13.7 min |
| G7 + G9 final test | 576 | 4.6 min |
| **Total without fallback** | **5,072** | **40.1 min** |
| **Total with fallback** | **7,696** | **60.8 min** |

Excluded and real: about 48 s of model load per process, about 1.5 percent capture overhead per
captured forward, and serialization and verification time. The compute-decision record's
long-prompt band of 1.5 to 2 times `T` puts the arm at roughly 40 to 80 minutes, or 61 to 122
minutes with the fallback.

**No GPU and no compute grant is needed.**
