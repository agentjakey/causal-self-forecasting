# Preregistration: CSF-Bench v0.1

Status: **written before any experiment has been run.** No model organism exists, no
direction has been estimated, no forecaster has been trained, and no result has been
measured. Every number in this document is a planned target, not a finding.

Version 1.0. Written 2026-07-15.

Any change to this document after data collection begins must be recorded in the amendment
log at the bottom, with a date and a reason. Amendments are not misconduct. Silent
amendments are.

## 1. Research question

Given a language model's internal state and a set of candidate interventions, can a
forecasting method predict how the model's output will change, before the intervention is
selected and applied?

## 2. Primary hypothesis

**H1 (internal-state advantage).** A forecasting system that receives the model's real hidden
state predicts intervention effects better than a prompt-only baseline that sees the same
visible prompt.

The comparison is only meaningful under the same-prompt state-swap design in section 6: two
models with different internal states receive character-identical prompts. A prompt-only
baseline cannot distinguish them even in principle. That is the point.

## 3. Primary outcome

**Answer-flip Brier score** on held-out prompts, for the selected intervention of each trial.

Chosen over the numeric delta-margin error because it is bounded, directly calibratable, and
does not depend on the scale of the logits, which is not comparable across models.

The margin is defined as `logit(correct) - max(logit(incorrect))` over the four answer-label
logits at the final prompt position. An answer flip is a change in the argmax label between
the clean and the intervened run.

## 4. Primary comparison

`state_mlp` versus `prompt_tfidf`, on the `test` split, paired by trial, restricted to
non-no-op candidates.

No-op candidates are excluded from the headline number. Any competent method predicts "no
effect" for an intervention publicly described as adding zero, so including them would
inflate every method's score and compress the differences between them. No-op results are
reported separately as a scoring-path validity check.

Test statistic: difference in Brier score, with a bootstrap confidence interval grouped by
original task item (section 8).

**Decision rule, fixed in advance.** H1 is supported only if the 95 percent grouped bootstrap
interval for the paired Brier difference excludes zero in favor of `state_mlp`. A point
difference with an interval spanning zero is reported as "no detected difference", not as a
trend, a signal, or a suggestion.

## 5. Secondary hypotheses

Reported with intervals, not tested against a threshold. They are secondary and this
document does not pretend otherwise.

* **H2 (state-specificity).** Forecast accuracy falls when the true hidden state is replaced
  with a shuffled, wrong-example, or same-prompt wrong-model state.
  `self_specificity_gap = performance(true state) - performance(wrong state)`.
* **H3 (in-distribution forecasting).** A trained state-conditioned forecaster beats the
  constant and prompt-only baselines on unseen prompts and unseen strengths.
* **H4 (mechanism transfer is difficult).** A forecaster trained on residual addition and
  projection ablation performs worse on activation patching.
* **H5 (verbal reports are less grounded).** Textual self-reports are less calibrated and less
  state-specific than an explicitly state-conditioned predictor.
* **H6 (evaluation framing affects reporting).** Evaluation-like versus deployment-like framing
  changes verbal confidence more than it changes the best state-conditioned forecaster.
* **H7 (gradient prediction works locally).** A first-order gradient approximation is accurate
  for small interventions and degrades as they grow.

A negative result on any of these, including H1, is a publishable outcome for this project
and will be reported with the same prominence as a positive one.

## 6. Design

Protocol per trial: forecast, commit, select, intervene, measure, score.

1. Run the model on a four-choice item under a wrapper framing. Record the label logits and
   the residual stream at the configured layers.
2. Build four candidates: positive steering along the direction under test, negative steering,
   a norm-matched random control, and a no-op. Randomize the order and assign opaque ids.
3. Every method forecasts every candidate.
4. Canonicalize the forecast and commit to `SHA256(canonical_forecast || salt)`.
5. Only then generate the selection seed. Select one candidate deterministically.
6. Apply it, measure the output, reveal the salt, verify the commitment, score.

**The same-prompt state swap is the centerpiece.** For one question, the clean model and the
adapted model both produce a state under a character-identical prompt. The forecaster receives
either the correct state or a wrong one. A prompt-only baseline cannot tell these apart, so
any true-state advantage cannot be explained by the text.

## 7. Split policy

Fixed before any forecaster is trained. Splitting is by `group_id`, never by row, so no
paraphrase of a question can straddle a boundary.

| Split | Purpose |
| --- | --- |
| train / val / test | 60 / 15 / 25 by group, seed 20260715 |
| heldout_subject | Whole task categories (law, philosophy) never trained on |
| heldout_wrapper | Wrapper paraphrases never trained on |
| heldout mechanism | Activation patching, never trained on |
| heldout strength | Extrapolation beyond trained strengths, reported separately |

## 8. Statistical analysis

* Bootstrap confidence intervals, 95 percent, 10000 resamples, **resampled by original task
  item, not by trial**. Trials from one item share a question and are not independent;
  resampling by trial would produce intervals that are too narrow and a false claim of
  precision.
* Paired bootstrap for method comparisons on identical trials.
* Seed-level variance reported across at least 3 seeds for any trained forecaster.
* Exact sample counts (`n` trials and `n` groups) reported with every number. A metric with no
  sample count is not publishable, and `MetricValue` in the schema enforces it.
* No p-value is reported as a headline. Intervals and effect sizes are the output.
* Where multiple secondary hypotheses are tested on the same data, intervals are reported
  without correction and are explicitly labeled exploratory rather than corrected and
  presented as confirmatory.

## 9. Planned sample counts

Targets, not measurements. The final counts come from the run manifest and may be lower if
the compute budget binds; the realized counts get reported, not these.

| Quantity | Planned |
| --- | --- |
| Task items after four-choice filtering | 500 target (ARC-Challenge) |
| Wrapper variants per item | 8 (5 trained, 3 held out) |
| Trials in the main run | 2000 target |
| Candidates per trial | 4 |
| Seeds per trained forecaster | 3 |

## 10. Exclusion rules

Fixed in advance so that no exclusion can be chosen after seeing its effect.

An item is excluded if:
* it does not have exactly four options;
* its answer key is not among its option labels;
* its options are not unique;
* an answer label does not resolve to exactly one token under the model's tokenizer.

A trial is excluded if:
* the clean rerun does not reproduce within `rerun_tolerance`;
* the intervention hook did not fire;
* a required control failed for that configuration.

A forecast is excluded if it fails schema validation. **Invalid model reports are never
silently repaired.** The invalid-output rate and the repair rate are themselves reported, as
a property of the method.

Exclusions are counted and reported. An analysis that drops trials without saying how many is
not reproducible.

## 11. Stopping rule

Data collection stops when the planned counts in section 9 are reached, or when the compute
budget is exhausted, whichever comes first. Analysis is run once, after collection stops.

The stopping rule does not depend on the results. Collection does not continue because a
result is nearly significant, and it does not stop early because a result is already
significant.

## 12. Exploratory analyses

Everything in this section is exploratory. Results from it are labeled as such, and are not
presented as tests of a hypothesis:

* Layer-by-strength effect heatmaps.
* Which framings produce the largest clean-versus-adapted state differences.
* Whether forecast error correlates with clean-model entropy or margin.
* Failure-mode case studies from the trial explorer.
* Any analysis suggested by the data after collection.

## 13. What would falsify the headline claim

Stated up front so it cannot be renegotiated later:

* The prompt-only baseline matching `state_mlp` would mean the forecasts are explained by the
  text, not by the state.
* Scrambling the state not hurting performance would mean the state input is not being used.
* A near-zero self-specificity gap would mean the forecaster is not tracking the specific
  model's computation.
* The gradient baseline beating every learned method everywhere would mean the task reduces
  to a local linear approximation and the framing adds nothing.

## Amendment log

None. The document has not been amended.
