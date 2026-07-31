# Preregistration: BlueDot state-dependence arm

Status: **written before any calibration, training, or final-test run.** No result exists. Every
number in this document is a frozen design decision or a labeled prediction, never a measurement.

Version 1.0. Written 2026-07-27.

Scope of supersession: this document supersedes `docs/preregistration.md` v1.0 **for this
experimental arm only**. The original CSF-Bench preregistration remains in force, unedited, for
the broader study it describes. Nothing in the original document has been rewritten; an
amendment-log entry there points here. Where the two disagree about method names, targets,
sample counts, or hypotheses, this document governs the BlueDot arm and the original governs
everything else.

Any change to this document after calibration begins must be recorded in the amendment log at
the bottom, with a date and a reason. Amendments are not misconduct. Silent amendments are.

## 1. Research question

> Does access to Gemma 3 1B's correct prompt-specific hidden state improve forecasts of how a
> fixed internal intervention changes the model's clean preferred answer, beyond the prompt, the
> clean output distribution, and a complete numerical representation of the intervention?

## 2. What this is and is not

This is an **external state-information audit**. It asks whether a hidden state carries
prompt-specific information about an intervention's effect that is not already present in the
visible inputs, as measured by an external predictor that we train and control.

It does not test, and no result from it may be described as testing:

* introspection or self-knowledge;
* consciousness or self-awareness;
* faithful verbal reasoning or the honesty of a model's stated reasons;
* hidden goals, deception, or scheming;
* deployment readiness or oversight adequacy.

The predictors here are ridge regressions we fit. A ridge regression reading a residual stream is
a readout, not a report. See `docs/claim_boundaries.md`.

## 3. Primary hypotheses

**H-BD1 (state-information advantage).** A state-conditioned bilinear ridge predicts
`delta_clean_top_margin` with lower prompt-aggregated mean absolute error than a
visible-information ridge that receives the same prompt, the same clean output distribution, and
the same complete numerical intervention representation.

**H-BD2 (prompt specificity).** The advantage in H-BD1 depends on the state belonging to the
prompt being predicted. Substituting a matched wrong prompt's state raises prompt-aggregated MAE
relative to the true state.

H-BD1 without H-BD2 is a statement that some state helps, not that this prompt's state helps.
Both are required for a prompt-specific state-dependence claim.

## 4. Frozen model, task, and prompts

### 4.1 Model

| Item | Frozen value |
| --- | --- |
| Model | `google/gemma-3-1b-it` |
| Revision | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` |
| Device / dtype | cpu / float32 |
| Hidden dim | 1152 |
| Decoder blocks | 26 |
| Scoring format | `raw_completion_next_token_after_answer_colon` (existing) |
| Answer tokens | A=562, B=603, C=565, D=622 (existing, resolved by `resolve_label_token_ids`) |

No adapter, no second model, no second revision.

### 4.2 Task

Existing ARC four-choice path: `allenai/ai2_arc`, config `ARC-Challenge`, loaded and normalized
by `tasks/loader.py`, which already drops items that are not exactly four options and items whose
answer key is not among their labels.

**One canonical neutral prompt variant per `group_id`.** The canonical variant is the
`neutral_a` wrapper. The `eval_*` and `deploy_*` wrappers are not used in this arm. The
evaluation-versus-deployment framing comparison is deferred (section 14); the task config keeps
all three framings because `TaskConfig` requires them, but this arm selects only `neutral_a`.

**Every valid selected prompt is included regardless of clean correctness.** A prompt is not
dropped because the model answers it wrongly on the clean run. The target is defined against the
model's own clean preferred answer, not against the dataset answer key, so clean correctness is
irrelevant to the measurement and filtering on it would select on model behavior.

### 4.3 Prompt roles and counts

| Role | Prompts |
| --- | --- |
| `smoke` | 8 |
| `calibration` | 32 |
| `training` | 96 |
| `final_test` | 32 |
| Total | 168 |

Frozen rules:

* Roles are **disjoint by original ARC item and by `group_id`**. In this arm one item is one
  group is one prompt, so the three are in one-to-one correspondence, and the disjointness check
  is asserted on all three keys rather than inferred from that correspondence.
* Selection from the eligible pool is **deterministic from the frozen study seed**
  (section 4.4). Given the same prepared task and the same seed, the manifest is byte-identical.
* Selection may not use model outputs, clean correctness, clean confidence, clean margin,
  entropy, intervention effects, or any outcome. It is a function of `group_id` and the seed
  only.
* Roles are carried by a **new `PromptRole` enum**, not by the existing `Split`. The existing
  train/val/test/heldout split semantics in `tasks/splitting.py` and the preregistered split
  policy in `docs/preregistration.md` section 7 are untouched. A prompt's `Split` value is
  recorded in the manifest for provenance and is ignored by this arm's analysis.
* Eligibility: the item yields exactly four unique options, its answer key is among its labels,
  its `neutral_a` variant renders, and its four answer labels resolve to single tokens under the
  pinned tokenizer. These are the existing exclusion rules in `docs/preregistration.md`
  section 10 and are unchanged.
* The manifest records the task manifest hash. Rebuilding the task invalidates the manifest.
* If the eligible pool holds fewer than 168 groups, the study **stops and reports the shortfall**.
  It does not shrink a role, reuse a group across roles, or reduce disjointness.

### 4.4 Frozen seeds

Master study seed: **`20260727`**.

All randomness derives from it through the existing `reproducibility.derive_seed(*parts)`, using
a stable purpose label so that changing one component cannot shift another:

| Purpose | Label |
| --- | --- |
| Prompt manifest role assignment | `bluedot.prompt_manifest` |
| Random control directions | `bluedot.random_directions` |
| Intervention projection matrix | `bluedot.intervention_projection` |
| Grouped cross-validation folds | `bluedot.cv_folds` |
| Wrong-state permutation `k` | `bluedot.wrong_state_permutation`, `k` |
| Paired bootstrap | `bluedot.bootstrap` |

No unseeded randomness anywhere. Commitment salts remain the one deliberate exception: they come
from the OS CSPRNG, as the existing protocol requires.

## 5. Frozen state definition

| Item | Frozen value |
| --- | --- |
| Position | final prompt token (`capture_position: -1`) |
| Representation | residual stream before the final norm, this project's existing layer convention |
| Primary layer | **13** |
| Only allowed fallback layer | **20** |
| Dimension | 1152 |
| Storage dtype | float32 |

Layer 13 is the layer the systems benchmark already verified capture at
(`capture_point_verified: true`, `max_abs_patch_error: 0.0`).

**The layer-20 fallback triggers under exactly one condition:** no ratio in the frozen grid
satisfies every calibration condition at layer 13 (section 7.3). It is not triggered by weak
results, by a disappointing effect distribution that nonetheless passes, or by anything measured
after calibration. No third layer is searched under any circumstance.

## 6. Frozen target: `delta_clean_top_margin`

This arm defines its own target. **The existing benchmark target `delta_margin` is not
redefined.** `ObservationRecord.delta_margin` remains
`post_margin - clean_margin` computed against the dataset-correct label, exactly as today, and
the original CSF-Bench study continues to use it.

For each prompt and each intervention:

1. Let `c_star` be the highest-logit answer among A, B, C, D on the **clean** run.
2. `clean_top_margin = clean_logit[c_star] - max(clean_logit[c] for c != c_star)`.
3. Apply the intervention.
4. `intervened_top_margin = intervened_logit[c_star] - max(intervened_logit[c] for c != c_star)`,
   **with `c_star` held fixed at the clean preferred answer.**
5. `delta_clean_top_margin = intervened_top_margin - clean_top_margin`.

Notes that are part of the definition, not commentary:

* `c_star` is the model's clean preference, which may differ from the dataset answer key. It is
  computed once per prompt from the clean run and never recomputed after an intervention.
* `clean_top_margin` is non-negative by construction. `intervened_top_margin` may be negative;
  it is negative exactly when the intervention changed the argmax away from `c_star`.
* Therefore `delta_clean_top_margin` is negative when the intervention weakens the clean
  preference and positive when it strengthens it, and the existing `answer_flip`
  (`clean_predicted_label != post_predicted_label`) coincides with
  `intervened_top_margin < 0`. This coincidence is asserted by a test, not assumed.
* Logits are the four label logits at the final position, read through the existing
  `score_logits` path in float64.

**Answer flip is secondary** in this arm. It is recorded and reported, and it drives no primary
comparison.

## 7. Frozen intervention design

### 7.1 Direction bank

Eight directions, all unit norm, all 1152-dimensional, all built once against the pinned
revision and stored in the existing `DirectionStore` with provenance metadata and a file hash.

**Four centered answer-token directions.** Let `u_A, u_B, u_C, u_D` be the unembedding rows for
token ids 562, 603, 565, 622. For each label `X`:

```text
d_X_raw = u_X - mean(u_Y for Y in {A,B,C,D}, Y != X)
d_X     = d_X_raw / ||d_X_raw||
```

The accessor used to read the unembedding (`get_output_embeddings().weight`, or the tied input
embedding if the model reports tied weights) is recorded in the direction metadata, together with
the resolved token ids and the model revision.

A property that is part of the design and must not surprise a reader later: the four raw centered
vectors sum to zero, so the answer-token family **spans a 3-dimensional subspace**, not a
4-dimensional one. The four unit directions remain distinct, but they are linearly dependent.
Orthogonality of the random controls is defined against that 3-dimensional span.

**Four deterministic Gaussian random controls.** Drawn on CPU in float32 from
`derive_seed("bluedot.random_directions", 20260727, index)`, then:

1. project out the span of the four answer-token directions;
2. Gram-Schmidt against the previously accepted controls;
3. unit normalize.

Frozen acceptance tolerance: every pairwise absolute cosine between a control and any
answer-token direction, and between any two controls, must be at most `1e-6`. The realized
maxima are recorded in the direction-family artifact.

**Privacy of family and role.** The family (`answer_token` versus `random_orthogonal`) and the
semantic analysis role stay private, in `InterventionSpec.analysis_role` and the direction
metadata. They never enter `public_metadata`, the candidate public view, or any predictor-facing
artifact. Candidate ids stay opaque (`{trial_id}.opaque_{index:02d}`). Predictors receive
numerical vectors, never a family label or a meaningful id.

An honesty note that belongs in the preregistration rather than in a later caveat: the numerical
representation handed to predictors (section 7.4) is a fixed linear map of the actual
intervention vector, so family membership is in principle linearly recoverable from it. That is
intended. This arm deliberately gives every method a **complete** description of the
intervention; what is withheld is the semantic label, not linear separability. The comparison is
about what the state adds on top of a complete intervention description, not about hiding the
intervention.

### 7.2 Candidate set per prompt

| Candidates | Composition |
| --- | --- |
| 16 | 8 directions x {positive, negative} sign, at the selected ratio |
| 1 | no-op integrity control |
| **17** | total per prompt, for smoke, training, and final test |

Calibration prompts carry **81** candidates: 5 ratios x 16 signed directions, plus one no-op.

Order is randomized per trial by the existing seeded shuffle, and opaque ids are assigned after
the shuffle, so position encodes nothing.

### 7.3 Magnitude: one global absolute strength per layer and ratio

**Prompt-specific state-relative strengths are prohibited in this arm.** The reason is a leakage
one and it is the single most important design constraint here. `trials/candidates.py:public_view`
publishes `strength` to every method, including the visible-information ridge. If `strength` were
`ratio * ||h_prompt||`, the published strength would encode the prompt's state norm and the
visible-information method would silently receive a state-derived feature, contaminating the
exact comparison this arm exists to make. A global alpha is state-derived only through the
calibration prompts, which are disjoint from training and final test, and is identical across
every prompt, so it carries zero prompt-specific state information.

Procedure, at each calibrated layer:

1. Capture clean states for all 32 calibration prompts at that layer.
2. `reference_norm = median(||h_p||_2 for the 32 calibration prompts)`.
3. For each ratio `r` in the grid: `alpha_r = r * reference_norm`. One number per layer per ratio.
4. The same `alpha_r` is applied to every prompt at that layer for that ratio.
5. The actual signed intervention vector is `v = sign * alpha_r * d`, where `d` is a unit
   direction from the bank. It is added to the residual stream at the capture position:
   `h' = h + v`. This is the existing `apply_residual_add` with `strength = sign * alpha_r`.
6. Both visible-information and state-conditioned predictors receive the **same** complete
   numerical representation of `v` (section 7.4).
7. During a wrong-state substitution, the target prompt's `v` is unchanged. Because `alpha_r` is
   global, `v` does not depend on which state is supplied, which is a second reason for the
   global-alpha choice.

Frozen ratio grid: **0.02, 0.05, 0.10, 0.20, 0.40**.

**Selection rule: the smallest ratio satisfying every condition below.** Conditions are evaluated
over the calibration prompts crossed with the 16 signed directions at that ratio, using
`delta_clean_top_margin` as the target.

| # | Condition |
| --- | --- |
| C1 | All four label logits and all targets are finite for every calibration observation at that ratio. |
| C2 | The no-op integrity control is within the existing harness tolerance: `csf interventions validate` passes every required control at the study config, and every observed no-op has `abs(delta_clean_top_margin) <= rerun_tolerance` (frozen at `1.0e-3`). |
| C3 | At least **15 percent** of non-no-op effects have `abs(delta_clean_top_margin) >= 0.10`. |
| C4 | The **median** `abs(delta_clean_top_margin)` over non-no-op effects is at least **0.05**. |
| C5 | The **95th percentile** of `abs(delta_clean_top_margin)` over non-no-op effects is at most **4.0**. |

C3 and C4 rule out a grid point where nothing measurable happens. C5 rules out one where the
output distribution has been destroyed and the target is no longer a small perturbation of the
clean preference.

If no ratio at layer 13 satisfies C1 through C5, repeat the identical procedure once at layer 20.
If no ratio at layer 20 satisfies them, **the study stops under this design.** It does not search
additional layers, widen or extend the ratio grid, change the direction bank, train a model
organism, or learn a behavioral direction. A stop is a reportable outcome, written up as a
negative engineering result about this intervention family at these layers.

The no-op is retained at the selected ratio in every subsequent run as a live integrity control,
not only during calibration.

### 7.4 The shared numerical intervention representation

Every primary method receives the identical 16-dimensional vector

```text
x_intervention = P^T v
```

where `v` is the actual signed intervention vector (1152-dim, zero for the no-op) and `P` is a
fixed `1152 x 16` matrix with orthonormal columns, generated once from
`derive_seed("bluedot.intervention_projection", 20260727)` by drawing a Gaussian matrix on CPU in
float32 and taking the `Q` factor of its QR decomposition.

Frozen properties:

* `P` is generated once, stored as an artifact, hashed, and cited by every run. It is **never
  fitted**, so it has no training-boundary exposure and is identical for calibration, training,
  and final test.
* `P` is applied identically for every method, every prompt, every state condition, and every
  candidate.
* A random projection is used rather than projection onto the direction bank itself because a
  bank-aligned basis would make coordinate index equal family identity, which is a naming
  convention by another route.
* The map is injective on the realized intervention set (the 16 signed vectors lie in an at most
  8-dimensional span; a 16-dimensional orthonormal projection of that span is injective almost
  surely, and injectivity is asserted numerically and recorded).

## 8. Frozen methods

Existing methods are preserved unchanged as diagnostics:

| Method id | Role in this arm | Inputs |
| --- | --- | --- |
| `constant` | diagnostic floor | public operation, layer, strength |
| `prompt_lexical` | diagnostic | prompt text, public strength and layer |

`prompt_lexical` is the method the original documents call `prompt_tfidf`. The implemented id is
`prompt_lexical` and this arm uses the implemented id everywhere.

Three new primary methods, all ridge regressions:

| Method id | Feature blocks | Approximate width |
| --- | --- | --- |
| `intervention_only_ridge` | I | 16 |
| `visible_information_ridge` | I + V | 55 |
| `state_bilinear_ridge` | I + V + S + SxI | 327 |

Feature blocks, named explicitly so that the wrong-state substitution can replace a block and
nothing else:

* **I (intervention), 16 dims.** `x_intervention` from section 7.4. Identical for every method.
* **V (visible), 39 dims.**
  * centered clean A/B/C/D logits (4): the four label logits minus their own mean;
  * `clean_top_margin` (1);
  * clean four-way answer entropy (1), from the existing `entropy_from_label_logits`;
  * prompt length in tokens under the pinned tokenizer (1);
  * training-fitted TF-IDF of the prompt text reduced by truncated SVD to **32** components (32).
    `k_svd = min(32, n_features - 1, 95)`; the realized value is recorded.
* **S (state), 16 dims.** Training-fitted PCA of the 1152-dim clean state, **16** components.
* **SxI (interaction), 256 dims.** Flattened outer product of S and I, in a fixed index order.

Fitting rules, frozen:

* Every fitted object (TF-IDF vocabulary and idf, SVD, state PCA, feature standardizer, ridge
  coefficients) is fitted on the **96 training prompts only**. Calibration and final-test prompts
  never touch a fit. This is enforced at fit time by an assertion on prompt role, and recorded in
  a `TransformFitRecord` naming the exact training prompt ids, the manifest hash, the seed, and
  the hash of the serialized transform.
* Rows are (prompt, candidate) pairs. Fitting uses the **16 non-no-op candidates** per training
  prompt: 96 x 16 = 1536 rows. The no-op is predicted and scored but not fitted on, so an exact
  zero row cannot pull the intercept.
* Features are standardized using training-set means and standard deviations.
* Regularization is chosen by **grouped cross-validation within the 96 training prompt groups**:
  6 folds of 16 groups, assigned deterministically from
  `derive_seed("bluedot.cv_folds", 20260727)`. No `group_id` appears in both the fit and the
  held-out part of a fold. Frozen alpha grid:
  `1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000`. The selected alpha per method is recorded.
* Ridge regression only. **No MLP is allowed in this arm.** No gradient method, no probe beyond
  the ridges named above, no ensembling, no early stopping on any held-out signal.
* Each method predicts `delta_clean_top_margin` directly. Interval endpoints and flip
  probabilities required by `ForecastCandidate` are filled from the training-residual spread and
  the training flip base rate respectively, and are labeled as placeholders rather than
  predictions (section 15, conflict S4).

## 9. Frozen wrong-state control

For each of the 32 final-test prompts, one **deterministic nearest matched wrong state** is
selected from the other 31 final-test prompts. The donor pool is the final-test set itself, so
the marginal state distribution is identical between the true-state and wrong-state conditions
and the comparison isolates prompt specificity rather than distribution shift.

Selection, applied in order:

1. Exclude the target prompt itself. Self-matches are prohibited.
2. Prefer donors with the **same clean preferred answer** `c_star`. If at least one such donor
   exists, restrict to them; otherwise use all 31.
3. Minimize `abs(clean_top_margin_donor - clean_top_margin_target)`.
4. Break ties on `abs(clean_entropy_donor - clean_entropy_target)`.
5. Break remaining ties on lexicographic `variant_id`, so the match is fully determined.

The matched pairing is written to an artifact and hashed before any final-test outcome exists.

**Substituted, and only these:**

* the S block (state PCA components);
* the SxI block (state-by-intervention interactions), recomputed from the donor S and the
  target's own I.

**Held identical to the true-state condition:**

* the target prompt and its identity;
* the V block's text features (TF-IDF and SVD);
* the V block's clean output features (centered logits, `clean_top_margin`, entropy, prompt
  length);
* the I block, that is the signed intervention vector and its projection;
* the strength, the layer, the candidate id and ordering;
* the eventual observed target, which is the target prompt's own outcome.

A test asserts block-level equality for everything except S and SxI.

**Primary:** the deterministic nearest match.

**Secondary:** ten seeded restricted permutations. Permutation `k` is a derangement of the 32
final-test prompts' states (no fixed point), drawn from
`derive_seed("bluedot.wrong_state_permutation", 20260727, k)` for `k` in 0 through 9. The
same-`c_star` preference applies to the primary nearest match only; the permutations are
unrestricted derangements, and their spread is reported as a robustness band around the primary
number rather than as ten separate tests.

## 10. Frozen commitment protocol

Requirements for this arm, beyond what the repository already enforces:

1. **Final-test forecasts for every candidate, every method, and every state condition must be
   committed before any final-test intervention outcome exists.** Per final-test prompt that is
   16 forecast records, each covering all 17 candidates:

   | Records | Method and condition |
   | --- | --- |
   | 4 | `constant`, `prompt_lexical`, `intervention_only_ridge`, `visible_information_ridge`, all at `state_condition = none` |
   | 1 | `state_bilinear_ridge` at `state_condition = true` |
   | 1 | `state_bilinear_ridge` at `state_condition = wrong_example`, condition index 0 (nearest match) |
   | 10 | `state_bilinear_ridge` at `state_condition = shuffled`, condition indices 0 through 9 |

   32 prompts x 16 records = **512 commitments**, all written before the final-test resolution
   begins.

2. **The commitment key must structurally include the trial id, the method id, and the state
   condition.** A method-name suffix convention is not acceptable, because nothing can enforce
   it. The key is extended to `(trial_id, method_id, state_condition, condition_index)`, where
   `condition_index` defaults to 0 and distinguishes the ten permutations. The salt path on disk
   must carry the same components, or one condition's salt overwrites another's and the
   overwritten commitment becomes unverifiable.

3. **A no-selection reveal and verification path is required.** This arm resolves every
   candidate, so there is no selected candidate. The existing `SelectionReveal` requires
   `selected_intervention_id`, `selected_index`, and `selection_seed_hash`, and
   `verify_run_commitments` marks a run unverified without a reveal per commitment. A reveal
   record that discloses the salt and verifies the hash without selecting anything is required,
   and the verifier must accept it.

4. **Committing must refuse when outcome artifacts already exist.** `commit_forecast` must raise
   when `observations.jsonl`, `resolution.json`, or `scores.json` is present in the run
   directory. Today it checks only for a selection seed, so the sequence "resolve, look, commit"
   is undetected and would still verify.

5. **The verifier must check ordering from artifacts and timestamps**, not from prose. At
   minimum: no outcome artifact exists at commitment time; every `committed_at` precedes every
   `observed_at` in the same run; the resolution manifest's provenance hashes match the files on
   disk; and the prompt-manifest, direction-family, projection, and transform-fit hashes cited by
   the run match the artifacts they name.

6. The existing limits of the protocol are unchanged and must stay stated. This scheme provides
   sequencing, tamper evidence, and third-party verification. It does not defend against an
   experimenter with full control of the machine, who can delete a run and start over. See
   `docs/claim_boundaries.md`.

## 11. Frozen analysis

### 11.1 Aggregation

**Aggregate by prompt first.** For each method and each final-test prompt, average the absolute
prediction error over that prompt's **16 non-no-op** interventions. This yields one number per
(method, prompt), 32 per method. Every headline statistic is computed over those 32 prompt-level
values, never over the 512 pair-level values, because pairs within a prompt share a question and
a state and are not independent.

### 11.2 Primary comparisons

1. `MAE_visible_information - MAE_true_state` (tests H-BD1)
2. `MAE_matched_wrong_state - MAE_true_state` (tests H-BD2)

Both are paired differences over the same 32 prompt groups.

### 11.3 Uncertainty

**10,000 paired bootstrap resamples over prompt groups**, drawing the same resampled groups for
both methods in every replicate, seeded from `derive_seed("bluedot.bootstrap", 20260727)`.
Report the point difference, the 95 percent percentile interval, the number of prompts, and the
number of prompt groups with every number.

**Decision rule, fixed in advance.** A comparison supports its hypothesis only if the 95 percent
paired interval excludes zero in the hypothesized direction. **An interval crossing zero is
reported as no detected difference** and is never described as a trend, a signal, a suggestion,
or a direction of travel.

### 11.4 Secondary metrics

Reported with intervals and counts, tested against no threshold:

* RMSE (prompt-aggregated, same pairing);
* sign accuracy;
* Spearman correlation between predicted and observed targets;
* top-effect ranking accuracy (the largest absolute predicted effect against the largest
  absolute observed effect within a prompt);
* flip count;
* no-op error;
* Brier score for answer flip, **reported only when the final-test set contains at least 20
  flips**. Below that it is omitted, and the omission is reported with the realized flip count.

Also reported, as diagnostics rather than hypotheses: the two existing baselines' numbers, the
spread across the ten permutation controls, the selected ridge alpha per method, the realized
`k_svd`, and every exclusion with its count.

### 11.5 Analysis discipline

* The analysis is run **once**, after the final-test resolution completes.
* Cross-validated training-fold results and the single final-test result are reported separately
  and never pooled.
* No method is added, dropped, retuned, or refitted after any final-test outcome is seen.
* Exclusions are counted and reported. An analysis that drops observations without saying how
  many is not reproducible.

## 12. Predicted results

**These are predictions recorded before calibration. They are not results, and no number here
may be cited as a measurement.** They exist so that the outcome cannot be reinterpreted as
"what we expected" after the fact.

| Prediction | Recorded expectation |
| --- | --- |
| State-conditioned MAE improvement over visible information | Modest, approximately **5 to 15 percent** relative reduction |
| Effect of matched wrong-state substitution | Removes **at least half** of any observed improvement |
| Interpretation if the visible-information model ties the state-conditioned model | Counts **against** a hidden-state advantage |
| Interpretation if true state ties matched wrong state | Counts **against** prompt-specific state dependence |

A tie is defined by section 11.3: a paired interval that crosses zero.

Both a null result and a negative result are publishable outcomes for this arm and will be
reported with the same prominence as a positive one.

## 13. Compute

All arithmetic derives from one measured value: the median CPU forward time
`T = 0.474158 s` for the pinned revision, read from
`results/runs/benchmark-gemma3_1b_it-20260718T040531Z/benchmark.json`, a `systems_benchmark`
artifact with `scientific_result: false`. **These are planning estimates, not measurements of
this study.**

Forwards per prompt: 1 clean capture forward plus one forward per candidate.

| Role | Prompts | Candidates per prompt | Forwards per prompt | Forwards |
| --- | --- | --- | --- | --- |
| Smoke | 8 | 17 | 18 | 144 |
| Calibration | 32 | 81 | 82 | 2,624 |
| Training | 96 | 17 | 18 | 1,728 |
| Final test | 32 | 17 | 18 | 576 |
| **Total** | **168** | | | **5,072** |

```text
5,072 x 0.474158 s = 2,404.9 s = 40.1 minutes
```

**Planned forward time is approximately 40 minutes**, before model loading, serialization, and
verification. Model loading is a measured one-time cost of about 48 s per process. Capture adds a
measured 1.5 percent per captured forward.

A layer-20 fallback repeats calibration only:

```text
2,624 x 0.474158 s = 1,244.2 s = 20.7 minutes
```

**approximately 21 additional minutes.**

The compute-decision record notes that the timing sample is three forwards of one 53-token
prompt, so long-prompt-heavy runs may cost 1.5 to 2 times `T`. Applying that band puts the arm at
roughly 40 to 80 minutes, or 61 to 122 minutes with the fallback.

The three ridges, the grouped cross-validation, the wrong-state control, the ten permutations,
and the bootstrap add **no forward passes**. They run on recorded observations and stored states.
Storage is under 1 MB per captured layer for all 168 prompts.

**No GPU and no compute grant is currently needed for this arm.**

## 14. Deferred for this arm

Not cancelled, not deleted, and still described by the original roadmap in
`docs/research_plan.md` and `docs/build_plan.md`. Out of the active path for this arm:

* benign model organism;
* LoRA training;
* learned clean-versus-adapted behavioral direction;
* the evaluation-versus-deployment framing hypothesis (H6);
* state MLP;
* verbal reporter and soft-token reporter;
* SAE extension;
* Gemma 3 4B replication;
* held-out-mechanism transfer (H4, activation patching);
* dashboard;
* GPU rental.

The held-out-mechanism machinery stays in the code. It costs nothing, and removing it would
weaken the original design.

## 15. Known conflicts with existing schemas

Recorded here because a preregistration that ignores what the code currently enforces is not
executable. Each is a required change, listed with what breaks without it.

| # | Conflict | Consequence if unaddressed |
| --- | --- | --- |
| S1 | `ObservationRecord` defines `clean_margin`, `post_margin`, and `delta_margin` against the dataset-correct label, with a validator requiring `delta_margin == post_margin - clean_margin`. This arm's target is clean-top-based. | The study target cannot be stored. Requires new fields `clean_top_label`, `clean_top_margin`, `intervened_top_margin`, `delta_clean_top_margin`, with their own validator. The existing three fields keep their current meaning. |
| S2 | Commitment key is `(trial_id, method_id)` in `ForecastCommitment`, in `verify_run_commitments`, in `_validate_commitments`'s duplicate check, in `scoring/run.py:method_state_condition`, and in the salt filename `{trial_id}.{method_id}.salt`. | The 12 state-conditioned records per prompt collide. `_validate_commitments` raises `duplicate commitment`, and salts overwrite each other on disk. Requires the extended key in all five places. |
| S3 | `SelectionReveal` requires a selected candidate, and `verify_run_commitments` requires a reveal per commitment. | An all-candidate run cannot verify. Requires a no-selection reveal record or optional selection fields. |
| S4 | `ForecastCandidate.p_bias_suppressed` and `ForecastRecord.p_hidden_bias_active` are required with no default, and are model-organism constructs this arm does not have. | Every record must carry a placeholder. Requires a documented default of 0.5 plus a discriminator naming the target, so a reader cannot mistake a placeholder for a prediction. |
| S5 | `Split` has no calibration member, and `ScoreRecord.split` is typed `Split`. | Resolved by design: this arm adds `PromptRole` and leaves `Split` untouched. `ScoreRecord` needs a `prompt_role` field and a `condition_index` field. |
| S6 | `ExperimentConfig.candidates_per_trial` is bounded `ge=2, le=8` and is read by nothing. | Dead config that is also wrongly bounded for 17 and 81 candidates. Remove it or rebound it; do not silently leave a validated field that contradicts the design. |
| S7 | `InterventionSpec` has no field for the norm ratio. | The ratio that generated a global alpha is unrecorded. Requires `norm_ratio: float | None`, kept out of `public_metadata` by extending the existing `forbidden` set. |
| S8 | `Forecaster.check_no_leakage` raises for any method declaring `hidden_state` or `intervention_vector`, and `commit_forecasts` calls it unconditionally. | The state-conditioned bilinear ridge cannot be committed through the existing driver. Requires a separate allowed-inputs policy and commit driver for state-audit methods. The baseline fence stays exactly as it is. |
| S9 | `RunManifest.phase` is a free-form string; run role is inferred from which files exist. | The four run types are not structurally distinguishable. Requires `run_role` on the manifest. |
| S10 | `grouped_bootstrap_ci` takes one score sequence and one statistic, and computes pair-level statistics over resampled groups. | Neither prompt-first aggregation nor paired two-method differences are possible. Requires a paired, prompt-aggregated sibling. |

## 16. Stop conditions

1. The eligible prompt pool holds fewer than 168 disjoint groups.
2. No ratio satisfies C1 through C5 at layer 13 **and** none does at layer 20.
3. Any required intervention control fails at the study configuration.
4. The clean-reproduction guard trips during a study run.
5. Any commitment fails to verify. Record it; do not re-resolve to make it disappear.
6. Prompt roles are found not to be disjoint by item and group.
7. A transform is found to have been fitted outside the training role.
8. Any documentation would have to describe a planned value as measured.

A stop is a reportable outcome. It is written up, not worked around.

## 17. What would falsify the headline claim

Stated up front so it cannot be renegotiated:

* The visible-information ridge matching the state-conditioned ridge would mean the state adds
  nothing beyond the prompt, the clean output distribution, and a complete intervention
  description.
* The matched wrong state matching the true state would mean any advantage is not
  prompt-specific: some state helps, but not this prompt's state.
* The intervention-only ridge matching both would mean the task reduces to a per-condition mean
  and neither the prompt nor the state is carrying information.
* A no-op with a non-zero measured target would mean the harness, not the model, produced the
  effects.

## Amendment log

None. The document has not been amended.
