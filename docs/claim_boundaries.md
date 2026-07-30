# Claim boundaries

What this project can and cannot say. Written before results exist, so that the limits are
set by the design rather than by whatever the numbers turn out to be.

## Current status

Last updated 2026-07-27.

**No scientific result exists.** There is no model organism, no estimated and causally validated
direction, no trained forecaster, and no verified public export.

What has been completed is engineering validation, and understating it would be its own kind of
inaccuracy. On real, pinned Gemma 3 1B weights
(`google/gemma-3-1b-it` at `dcc83ea841ab6100d6b47a070329e1ba4cf78752`):

* the systems benchmark loaded the weights, scored answer tokens, and timed forward passes on
  CPU;
* hook-owned capture was verified at layer 13 by patching a known vector and reading it back
  (`capture_point_verified: true`, `max_abs_patch_error: 0.0`);
* all five required intervention-harness controls passed;
* ground-truth resolution applied all four candidates to an ARC item and recorded four
  observations with the no-op reproducing the clean output exactly;
* the BlueDot engineering smoke ran 8 prompts x 17 candidates plus 8 clean captures, 144 forwards
  in total, at layer 13 under one global intervention strength, with 136 observations, no
  failures, every no-op reproducing the clean logits exactly, and a second execution reproducing
  every target with a maximum absolute difference of 0.0;
* the preregistered calibration sweep ran 32 prompts x 81 candidates plus 32 clean captures, 2,624
  forwards, at layer 13 across the five frozen ratios, with 2,592 observations and no failures,
  and selected the smallest passing ratio under conditions frozen before the numbers existed.

The smoke is plumbing validation of the state-dependence pipeline. Its intervention strength was
fixed in advance at an arbitrary ratio, it selects no ratio and no layer, and it is not
calibration.

Calibration is a **choice of stimulus, not a measurement**. It says which intervention strength
the study will use; it says nothing about what the model can do, and it cannot become a result.
Its effect sizes describe how far a fixed vector moves a logit margin at a given magnitude, which
is a fact about the intervention, not about the model's reasoning.

Clean accuracy from either run is descriptive over a handful of items and is not a capability
measurement.

As of 2026-07-29 the three ridges have been fitted on the 96 training prompts and 512 final-test
forecasts have been committed and sealed. **No final-test intervention has been applied, so no
method has been scored and no comparison exists.** The resolution and analysis code is implemented
and tested but has not been executed.

Two things about the eventual result are fixed now, before it exists, and cannot be renegotiated
afterwards:

* **A crossing interval is a null result.** A comparison supports its hypothesis only if the 95
  percent paired bootstrap interval over the 32 prompt groups excludes zero in the hypothesized
  direction. An interval crossing zero must be reported as no detected difference, and never as a
  trend, a signal, a suggestion, or a direction of travel. The code enforces this: `PairedComparison`
  refuses to be constructed with a `supported` flag that does not follow from its own interval.
* **A null result is not an absence claim.** With 32 final-test prompts the arm is not powered to
  show that the information is absent. A null would mean this readout at this power did not find
  it, which is a different and much weaker statement, and the write-up must make that difference
  explicit.

One piece of context that belongs here rather than in a footnote later: the training-fold
cross-validated errors already favour the visible-information ridge over the state-conditioned one.
Those are training-fold diagnostics and not the test, but they mean a negative result on H-BD1 is
the more likely outcome, and section 12 of the preregistration already records that such an outcome
counts **against** a hidden-state advantage rather than being uninformative.

None of that is a CSF-Bench result and none of it can become one: the direction used was
synthetic and unvalidated, and single-item accuracy is a scoring smoke check. The distinction is
enforced structurally, not by convention, and the artifacts carry it
(`scientific_result: false` as a typed literal). The full record is in
`docs/experiment_log.md`, entries dated 2026-07-18 and 2026-07-19.

The fixture-model smoke pipeline also runs end to end in the test suite. The fixture is randomly
initialized and measures nothing about language models.

## What the design can support

If the experiments run as planned and the controls hold, the project can say:

* Whether a forecasting method predicts the sign and magnitude of an intervention's effect on
  the answer margin, better than stated baselines, on held-out prompts.
* Whether those forecasts are calibrated.
* Whether forecast quality depends on receiving the model's actual hidden state, measured
  against same-prompt wrong-state controls.
* Whether forecasts transfer to a mechanism not seen in training.
* Whether a verbal self-report is more or less grounded than an explicit state-conditioned
  predictor.

Acceptable language: causal forecasting, counterfactual self-prediction, state-conditioned
forecasting, prediction of intervention effects, evidence consistent with privileged state
access.

## What the design cannot support

No result from this project may be described as showing that:

* the model understands itself;
* the model knows why it answered;
* the model reveals its true reasoning;
* the model reveals hidden goals;
* the method solves oversight or detects deception;
* a forecasting head is equivalent to natural introspection;
* a probe is a self-report.

These are not stylistic preferences. Each names a claim the design cannot distinguish from a
cheaper explanation.

## Distinctions that must not be blurred

These three things are routinely described with the same word, and they are not the same:

| Method | What it is | What it is not |
| --- | --- | --- |
| Linear probe | An external readout trained by us on the model's activations | Not a self-report. The model is not doing the reporting. |
| State-conditioned MLP | An attached forecasting system we built, which reads the model's state | Not introspection. It is a separate network with privileged access. |
| Verbal self-report | The model producing text about its own computation | The only one that resembles introspection, and the weakest one. |

A strong result from the state MLP is a result about what information is present and
decodable in the residual stream. It is not a result about what the model knows about
itself. Reporting it as the latter would be the single most likely way this project could
mislead someone, which is why the failure mode has its own dashboard page.

## The BlueDot state-dependence arm

Added 2026-07-27. Design frozen in `docs/bluedot/preregistration_state_dependence.md`.

This arm is an **external state-information audit**. It asks one question:

> Does access to Gemma 3 1B's correct prompt-specific hidden state improve forecasts of how a
> fixed internal intervention changes the model's clean preferred answer, beyond the prompt, the
> clean output distribution, and a complete numerical representation of the intervention?

**What a positive result would support.** That the residual stream at one layer carries
prompt-specific information about a specified intervention's effect which is not already present
in the visible inputs, as measured by ridge regressions we fit and control, on one model, at one
layer, for one constructed intervention family, on 32 held-out prompts.

**What a positive result would not support.** Nothing about introspection, consciousness,
self-awareness, faithful verbal reasoning, hidden goals, deception, or deployment readiness. The
predictors here are ridge regressions reading activations. A ridge regression is a readout, not
a report, and the model is not doing the reporting. The distinctions in the table above apply
with full force.

**What a null result would mean.** That the state adds nothing detectable beyond a complete
intervention description at this scale and sample size. It would not show that the information
is absent, only that this readout at this power did not find it. The sample is 32 final-test
prompts; the arm is not powered to make an absence claim.

**Additional limits specific to this arm.** One model, one revision, one layer (with one
preregistered fallback), one task, one neutral framing, one constructed direction family, and
linear readouts only. The directions are built from the unembedding and from seeded randomness,
so they are stimuli with known construction, not discovered features, and no claim about what
the model represents follows from them. The arm uses no model organism, so it says nothing about
detecting a planted behavior.

Acceptable language for this arm: state-conditioned prediction of intervention effects,
prompt-specific state information, external state-information audit, evidence consistent with
prompt-specific state dependence.

## Limits of the commitment protocol

The forecast commitment is `SHA256(canonical_forecast || salt)`, written before a selection
seed exists.

**What it provides:** reproducible sequencing, tamper evidence against after-the-fact edits
of forecasts, public verification by a third party, and protection against casually adjusting
a prediction after seeing an outcome.

**What it does not provide:** any protection against an experimenter with full control of the
machine. Whoever runs the pipeline can delete a run and start over, and nothing in the hashes
would show it. The protocol is not a trust-minimized proof and is not presented as one. It
raises the cost of self-deception, which is the realistic threat here, and it makes the
sequence auditable.

## Limits of the model organism

The organism is a LoRA adapter trained to prefer one answer position under a deployment-like
wrapper while preserving accuracy under an evaluation-like wrapper.

It is a benign, evaluation-conditioned task-bias model organism. It is not scheming, not
deceptive, and does not have goals. It is a controlled, harmless behavioral difference planted
so that there is a known internal thing for a forecaster to be right or wrong about.

Any finding about it is a finding about a small model with a directly planted bias. It does
not transfer to claims about naturally arising behavior in frontier models, and the paper must
not imply that it does.

## Limits of scale

v0.1 uses Gemma 3 1B. One model, one family, one size. Nothing about cross-scale or
cross-family generality may be claimed until the Gemma 3 4B replication (Phase 13) exists, and
even then it is two sizes in one family.

## Limits of the fixture model

The fixture model is randomly initialized. It has no knowledge and its outputs are noise. It
exists to test plumbing. No number produced from it is a result, and the exporter must never
publish one.

## Reporting rules

* Negative results are reported as prominently as positive ones.
* Every published number carries its sample count and a grouped uncertainty interval.
* Preliminary results are labeled preliminary until replicated.
* Exclusions are counted and reported.
* Invalid model outputs are reported, never silently repaired.
* Planned values are labeled planned. Measured values cite a verified run artifact.
