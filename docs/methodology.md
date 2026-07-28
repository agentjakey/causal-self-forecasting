# Methodology

How CSF-Bench works, in enough detail to reimplement or to attack.

## The protocol

Per trial: **forecast, commit, select, intervene, measure, score.**

The ordering is the contribution. Asking a model to explain an answer it already gave has no
ground truth. Asking a forecasting method to predict what a not-yet-chosen intervention will
do creates one: the prediction exists before the outcome does, and the outcome is measured
directly.

## 1. Clean run

The model receives a four-choice item wrapped in a framing. The residual stream is captured at
configured layers at the final prompt position, and the four answer-label logits are recorded.

**Margin.** `margin = logit(correct) - max(logit(incorrect))`, over the four label logits at
the final position. Positive when the model would answer correctly among the options. The sign
flips exactly when the answer changes.

**Label scoring.** Each label must resolve to exactly one token under the model's tokenizer. A
template whose labels do not is rejected, not worked around: comparing a one-token label
against a two-token label compares different quantities.

**Layer convention.** Layer 0 is the embedding output; layer `i` is the output of decoder block
`i - 1`. This is the residual stream before the final norm. Capture and intervention refer to
the same tensor at the same point.

Capture uses this project's own forward hooks rather than the `output_hidden_states` flag. See
`src/causal_self_forecasting/models/capture.py` for why: the flag's interaction with a user
intervention hook is not part of the transformers public API and, measured on 5.14, returned
pre-intervention states while the forward pass really had been modified.

## 2. Candidates

Four per trial: positive steering along the direction under test, negative steering, a
norm-matched random control, and a no-op. Order is randomized per trial from a seed derived
from the run seed and the trial id.

**Opaque identity.** Public ids carry no meaning (`trial_00001.opaque_02`), and ids are
assigned after the shuffle, so position encodes nothing.

**Indistinguishable public description.** A forecaster without payload access sees the
operation form, layer, position, and strength. It does not see which direction is involved. A
matched random control and a real steer are published identically, because that is all they
visibly are. Publishing `random_add` would let any method dismiss controls without consulting
the model's state, which would quietly destroy the control.

A no-op is published as a residual addition of strength zero, which is exactly what it is. Any
method can tell that adding zero does nothing, so no-ops validate the scoring path rather than
discriminating between methods, and they are excluded from headline numbers and reported
separately.

## 3. Mechanisms

| Mechanism | Operation | Role |
| --- | --- | --- |
| `noop` | `h' = h` | Control. Must reproduce the clean run exactly. |
| `residual_add` | `h' = h + alpha * v` | Training mechanism. |
| `direction_ablate` | `h' = h - proj_v(h)` | Training mechanism. Defined only at full strength. |
| `random_add` | `h' = h + alpha * v_random` | Norm-matched control. |
| `activation_patch` | `h' = h_source` | **Held out.** The transfer test (H4). |

Addition asks whether pushing along `v` changes behavior. Ablation asks whether the component
the model already has along `v` is load-bearing. They are different questions.

## 4. Directions

Estimated from training data only, by clean-versus-adapted mean difference, deployment-versus-
evaluation mean difference, or linear-probe weights.

**A direction is not meaningful because it classifies states.** Before it is described as
bias-related it must pass causal validation: adding it changes behavior, removing it suppresses
the bias, reversing it flips the effect sign, matched random directions do not reproduce the
effect, and the effect holds on held-out prompts. Until then it is a candidate direction and
is labeled `validated: false` in its artifact metadata.

## 5. Commitment

1. Validate the forecast against its schema.
2. Serialize to canonical JSON: keys sorted, no insignificant whitespace, ASCII-escaped, no
   NaN or Infinity. One logical value produces exactly one byte string on any platform.
3. Draw a 32-byte salt from the OS CSPRNG. Never seeded from the run seed.
4. Commit to `SHA256(canonical_forecast || salt)`.
5. Only now may a selection seed exist. The code refuses to commit a forecast if one already
   does, and refuses to write a seed before any commitment exists.
6. Select deterministically from the seed and the trial id, so a single trial can be
   recomputed without replaying the run.
7. Apply, measure, reveal the salt, recompute the hash, verify.

`csf verify run` recomputes every commitment from the forecasts on disk and the revealed
salts. An edited forecast fails even though its commitment record still looks well formed.

What this does not do is defend against an experimenter who controls the machine. See
`claim_boundaries.md`.

## 5a. Resolution

`csf trials resolve` applies interventions and records one `ObservationRecord` per applied
candidate. It runs in two modes, chosen explicitly because they answer different questions.

Forecast mode requires committed forecasts. It reads or writes a selection seed after the
commitments exist, selects one candidate per trial deterministically, applies it, reveals the
salt, and verifies the commitment. This is the blinded protocol: only the selected candidate is
observed, so a trial cannot contribute a candidate-ranking comparison, and the run can be a
scientific forecast resolution when its commitments verify.

Ground-truth mode applies every candidate in every trial. It needs no forecasts and does no
selection or reveal, so it produces the full effect distribution used to train baselines and to
validate the harness on real weights. It is never a scientific forecast evaluation.

Two safeguards. The clean baseline in each observation is taken from the committed trial record,
not recomputed, so a delta is always measured against exactly the clean output the forecast
targeted; a one-time check confirms the resolution model reproduces the committed clean margin,
catching a model loaded at the wrong precision or revision. And a failed intervention is written
to a failures record, never silently dropped, because a run missing a biased subset of its
interventions would report clean-looking numbers over the wrong sample.

## 5b. Scoring

`csf score run` matches each committed forecast candidate to its observation and aggregates per
method. No-op candidates are excluded from the headline metrics and reported separately, because
a no-op is a zero-strength addition that every method predicts correctly. Ranking uses only
trials where every candidate was observed. Uncertainty is bootstrapped by task-item group, and
every number carries its sample count. Scores are written to the run directory; the scorer never
writes to the public results tree and refuses to treat a run as scientific unless its
commitments verified.

## 6. Forecasting methods

Every method declares what it may read. The declaration is the experiment.

| Method | Prompt | Hidden state | Intervention vector | Produces language |
| --- | --- | --- | --- | --- |
| `constant` | no | no | no | no |
| `prompt_tfidf` | yes | **no** | no | no |
| `prompt_report` | yes | **no** | no | yes |
| `linear_probe` | no | yes | yes | no |
| `state_mlp` | no | yes | yes | no |
| `gradient` | no | yes | yes | no |
| `soft_token_reporter` | yes | yes | yes | yes |

The prompt-only methods must never receive adapter identity or any state-derived feature. That
is the leakage audit, and it is what makes the H1 comparison mean anything.

Naming: the table above uses `prompt_tfidf`, which is the name in the original preregistration.
The **implemented** method id is `prompt_lexical` (`forecasting/lexical.py`). They are the same
method. `constant` and `prompt_lexical` are the only two rows in this table that exist in code;
`prompt_report`, `linear_probe`, `state_mlp`, `gradient`, and `soft_token_reporter` are planned
and not implemented.

## 7. Metrics

Numeric: MAE, RMSE, Pearson, Spearman, sign accuracy, 90 percent interval coverage, interval
width.

Binary (answer flip, bias suppression): Brier, Brier skill, log loss, AUROC, AUPRC, ECE,
reliability diagrams.

Ranking: largest-effect top-1 accuracy, Kendall tau, Spearman, nDCG.

State dependence: true, shuffled, wrong-example, and same-prompt wrong-model states, plus the
self-specificity gap.

**Uncertainty is grouped by original task item, not by trial.** Trials from one item share a
question and are not independent. Resampling by trial would produce intervals that are too
narrow and claim precision the data does not support.

## 8. Artifacts

What a run actually writes, verified against `paths.py` and a real run directory:

| File | Written by |
| --- | --- |
| `trial_manifest.jsonl` | trial generation |
| `candidate_sets.jsonl` | trial generation |
| `state_refs.jsonl` | trial generation |
| `states.npz` | trial generation |
| `run_manifest.json` | trial generation |
| `environment.json` | trial generation |
| `run.log.jsonl` | any command run against the run directory |
| `forecasts.jsonl` | forecast commitment |
| `forecast_commitments.jsonl` | forecast commitment |
| `selection_reveals.jsonl` | resolution, forecast mode |
| `observations.jsonl` | resolution, both modes |
| `resolution_failures.jsonl` | resolution, when an intervention fails |
| `resolution.json` | resolution |
| `resolution_artifact_hashes.json` | resolution |
| `scores.json` | scoring |
| `score_records.jsonl` | scoring |

Observations are JSONL, not parquet. Every other record in a run is line-oriented JSON, the
volume is small, and a hashable line-oriented file needs none of the machinery a parquet writer
would pull in. `paths.OBSERVATIONS` still names `observations.parquet` as a documented long-term
target; nothing writes it, and no reader should look for it.

Salts, private payloads, and selection seeds live in a private directory that git ignores and
the exporter strips.

## 9. Reproduction

Model revisions are pinned to commit shas, never to `main`. The config hash covers the parsed
document, so reformatting a config does not invalidate provenance but changing a setting does.
The run manifest records the seed, the resolved device and dtype, package versions, and the git
commit, including whether the tree was dirty.

## 10. The BlueDot state-dependence arm

A separate arm runs a narrower experiment on the same harness. Its design is frozen in
`docs/bluedot/preregistration_state_dependence.md` and its execution order in
`docs/bluedot/execution_decision_tree.md`. Only the parts that change how the harness is read
are summarized here.

**Its own target.** The arm measures `delta_clean_top_margin`, not the `delta_margin` defined in
section 1. The existing target is unchanged and the original study continues to use it. The new
one is defined against the model's own clean preferred answer rather than the dataset answer key:

```text
c_star                  = argmax over {A,B,C,D} of the clean label logits
clean_top_margin        = clean_logit[c_star]      - max(clean_logit[c]      for c != c_star)
intervened_top_margin   = intervened_logit[c_star] - max(intervened_logit[c] for c != c_star)
delta_clean_top_margin  = intervened_top_margin - clean_top_margin
```

`c_star` is computed once from the clean run and held fixed when the intervened margin is
computed. `clean_top_margin` is non-negative by construction, `intervened_top_margin` goes
negative exactly when the argmax moves off `c_star`, and that coincides with the existing
`answer_flip`. A prompt is included regardless of whether `c_star` matches the dataset answer,
because filtering on clean correctness would select on model behavior.

**Global intervention magnitude, not prompt-relative.** One absolute strength per layer and
ratio, `alpha = ratio * median clean-state norm over the 32 calibration prompts`, applied to
every prompt. This is a leakage decision, not a convenience one: `public_view` publishes
`strength` to every method, so a prompt-relative strength would hand the visible-information
baseline the prompt's state norm and contaminate the comparison the arm exists to make. The
implementation carries no prompt-specific strength function at all, and
`calibration.strength.check_global_alpha` refuses a set of observations that used more than one
alpha at a grid point.

**Calibration is a choice of stimulus, not a measurement.** Six conditions are fixed before any
number exists: completeness (every expected observation present or its failure recorded), finite
outputs, the no-op within the harness tolerance, at least 15 percent of non-no-op effects
reaching 0.10, a median absolute effect of at least 0.05, and a 95th percentile of at most 4.0.
Conditions are inclusive at the boundary. The **smallest** passing ratio wins, in preregistered
order, never the largest effect or the most flips: choosing the stimulus by the outcome would
make the comparison circular. Layer 13 is primary; layer 20 is reachable only when no layer-13
ratio passes, and a passing primary layer prohibits it. Medians use `numpy.median` and
percentiles `numpy.quantile(method="linear")`, both recorded in the plan.

**Directions are constructed, not estimated.** Four centered answer-token unembedding
directions and four seeded Gaussian controls orthogonal to their span and to each other, all
unit norm. Nothing is estimated from data, so section 4's causal-validation requirement for a
*discovered* direction does not apply; these are stimuli with known construction, not claims
about what the model represents. The family and the semantic role stay private exactly as
section 2 requires.

Provenance for that family, as built (`interventions/direction_family.py`):

* `d_c = normalize(w_c - mean(w_j for j != c))` over the output-embedding rows for the resolved
  answer tokens. Answer token ids always come from the existing scoring path; a config may pin
  expected ids, and a mismatch refuses the build rather than proceeding against different
  tokens.
* The four raw centered directions sum to zero, so the answer family spans at most three
  dimensions and its members are linearly dependent. That is expected, is recorded as an
  effective rank, and is not treated as an error.
* The span basis is a deterministic two-pass modified Gram-Schmidt in fixed label order, not a
  QR or SVD: those can choose different bases and signs for a rank-deficient input across
  library versions, which would break regeneration.
* Controls are drawn from a seeded PCG64 generator, projected off the answer span and off each
  other twice, redrawn if the residual falls below a frozen tolerance, sign-canonicalized on
  their first significant component, and unit-normalized. The seed derives from the master
  seed together with the study id, family id, model id, pinned revision, and algorithm version.
* Construction and validation run in float64; artifacts are stored as float32.
* Stored ids are opaque hash prefixes and the family is ordered by id, so neither the string nor
  the position encodes a construction role. The role mapping lives only in the private family
  manifest, and the `.npz` metadata in the payload store carries no role, label, or family term.
* The family is verifiable two ways: against the stored artifacts with no model loaded, and by
  regenerating every vector from the pinned weights and comparing content hashes.

**All candidates resolved, no selection.** The arm forecasts and observes every candidate, so
the single-candidate selection step in section 5 is replaced by a no-selection reveal: the salt
is disclosed and the commitment recomputed without choosing a candidate. The blinding it relies
on is ordering, enforced by refusing to commit once any outcome artifact exists in the run
directory.

**Aggregation.** Errors are averaged per prompt over the non-no-op interventions before any
method is compared, and comparisons are paired bootstraps over prompt groups. This is stricter
than section 7's grouped bootstrap, which resamples groups but computes pair-level statistics.
