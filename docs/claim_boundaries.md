# Claim boundaries

What this project can and cannot say. Written before results exist, so that the limits are
set by the design rather than by whatever the numbers turn out to be.

## Current status

**No experiment has been run.** There are no results. The repository contains a working
pipeline and a smoke test on a randomly initialized fixture model, which measures nothing
about language models.

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
