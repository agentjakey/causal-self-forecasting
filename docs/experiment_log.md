# Experiment log

Append-only. One entry per run that produced artifacts. Entries cite a run id and its
manifest; numbers are copied from verified artifacts, never typed from memory.

## No experimental results exist yet

As of 2026-07-15, no experiment has been run on a real language model.

There is no model organism, no estimated direction, no trained forecaster, and no verified
export. Anything in this repository that resembles a measurement is either a harness control
or a smoke run on a randomly initialized fixture model.

## 2026-07-15: pipeline bring-up (not an experiment)

Harness controls and plumbing only. The fixture model is randomly initialized, so nothing
below is a fact about language models.

**Environment.** Windows 11, Intel Core Ultra 7 258V, 31.6 GB RAM, no CUDA device. Python
3.12.13, torch 2.13.0+cpu, transformers 5.14.0.

**Data.** `csf data prepare --config configs/tasks/arc_mcq.yaml --max-items 20` loaded ARC
Challenge from `allenai/ai2_arc` and produced 20 items and 160 wrapper variants. Item splits:
12 train, 3 val, 5 test. Variant splits: 60 train, 15 val, 25 test, 60 held-out wrapper.

**Intervention controls.** `csf interventions validate --config configs/experiments/smoke.yaml`
on the fixture model. All five required controls passed:

| Control | Result |
| --- | --- |
| no-op equality | max abs logit diff 0.0, residual delta norm 0.0 |
| zero-strength equality | max abs logit diff 0.0 |
| rerun determinism | max abs logit diff 0.0, max abs state diff 0.0 |
| shape validation | all invalid operands rejected |
| hook applies intervention | residual delta norm 1.0, expected 1.0 |

Sign-reversal coherence was measured and is informational only: on a random fixture model it
carries no meaning.

**Trials.** `csf trials generate --config configs/experiments/smoke.yaml --max-trials 8`
produced 8 trials and 24 captured states across 3 layers.

**A real bug this caught.** The first version of the capture code read hidden states via the
transformers `output_hidden_states` flag. Under transformers 5.14, `LlamaModel.forward` no
longer assembles that tuple inline, and the framework's internal recording hooks interact with
a user intervention hook in an order that is not part of the public API. Measured effect: an
activation patch at layers 1 through 4 was really applied to the forward pass, but the state
read back was the pre-intervention value, and at the final layer the returned state had the
model's final norm applied to it. Every number would have been plausible and wrong.
`test_capture_matches_the_intervention_point` caught it by patching layer L with a known
vector and requiring that capturing layer L return that vector. Capture now uses this
project's own hooks and does not depend on `output_hidden_states`.

## Next entry

The next entry will be the first real one: Gemma 3 1B loading and answer scoring on ARC, with
a measured clean accuracy and a measured CPU cost per forward pass. That measurement decides
whether the model organism can be trained locally or needs a rented GPU.
