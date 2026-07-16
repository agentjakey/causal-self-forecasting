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

Each run writes `trial_manifest.jsonl`, `candidate_sets.jsonl`, `forecasts.jsonl`,
`forecast_commitments.jsonl`, `selection_reveals.jsonl`, `observations.parquet`, `scores.json`,
`run_manifest.json`, `environment.json`, and `artifact_hashes.json`.

Salts, private payloads, and selection seeds live in a private directory that git ignores and
the exporter strips.

## 9. Reproduction

Model revisions are pinned to commit shas, never to `main`. The config hash covers the parsed
document, so reformatting a config does not invalidate provenance but changing a setting does.
The run manifest records the seed, the resolved device and dtype, package versions, and the git
commit, including whether the tree was dirty.
