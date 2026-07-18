# causal-self-forecasting

A reproducible benchmark for testing whether language models can forecast how blinded internal
interventions will change their outputs, with calibrated scoring, hidden-state controls, and
held-out mechanism evaluation.

Working name: **Causal Self-Forecasting Lab**. Benchmark: **CSF-Bench**.

## Status

**No experimental results exist.** The pipeline runs end to end on a tiny fixture model. No
experiment has been run on a real language model: there is no model organism, no estimated
direction, no trained forecaster, and no verified export.

Nothing in this repository should be read as a finding about language models. See
`docs/experiment_log.md` for what has actually been measured, which is harness controls and
plumbing.

## The idea

A model can explain its answer persuasively without the explanation tracking the computation
that produced it. Explanations are hard to grade because there is no ground truth.

So do not ask for an explanation. Ask for a prediction about an intervention that has not been
chosen yet, record it, commit to it with a hash, and only then select and apply one:

> forecast, commit, select, intervene, measure, score

Now there is ground truth. The forecast existed before the outcome did.

The question this can actually answer is narrower than "does the model understand itself":

> Can a forecasting system make calibrated predictions about causal internal changes, and does
> its performance depend on the model's actual state?

## What makes it a test rather than a demo

The controls carry the argument:

* **Same prompt, wrong state.** A clean model and an adapted model produce character-identical
  prompts. The forecaster gets the right state or a wrong one. A prompt-only baseline cannot
  tell them apart even in principle, so a true-state advantage cannot be explained by the text.
* **Norm-matched random directions.** Separates "a large perturbation happened" from "this
  direction matters".
* **Opaque candidate labels.** A matched random control and a real steer are described
  identically in public. Naming the mechanism would let a method dismiss controls without ever
  consulting the model's state.
* **Held-out mechanism.** Trained on additive interventions, tested on activation patching.
* **Precommitment.** Forecasts are hashed before a selection seed exists, and `csf verify run`
  lets a third party recompute every one of them.

A negative result is a real outcome here. If the prompt-only baseline matches the
state-conditioned forecaster, that is the finding, and the preregistration fixes the decision
rule in advance so it cannot be renegotiated afterward.

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). No GPU is needed for the smoke
pipeline.

```powershell
uv sync --extra dev --extra torch
uv run csf doctor
uv run pytest -q

# Real ARC data
uv run csf data prepare --config configs/tasks/arc_mcq.yaml --max-items 20

# Smoke pipeline on the fixture model, offline
uv run csf directions synthetic --config configs/experiments/smoke.yaml
uv run csf interventions validate --config configs/experiments/smoke.yaml
uv run csf trials generate --config configs/experiments/smoke.yaml --max-trials 8
```

`csf doctor` reports the environment and validates every config. It does not fail because a
GPU is missing; it says so.

### The fixture model

`configs/experiments/smoke.yaml` uses a tiny, randomly initialized Llama built locally by the
code. It exists so the whole pipeline runs offline in seconds with no gated weights.

It knows nothing and its answers are noise. It tests the plumbing, not the science, and the
exporter will never publish a number produced from it.

### Real models

`configs/models/gemma3_1b_it.yaml` pins Gemma 3 1B to a commit sha. Those weights are gated on
Hugging Face, so two separate things are needed before they will load.

**1. Authenticate.** A token with read access is enough. Either form works:

```powershell
uv run hf auth login
# or, without installing into the project environment:
uvx hf auth login
```

The token is read from the Hugging Face credential store. This project never prints, logs, or
writes it to an artifact, and it does not require `HF_TOKEN` to be set when the credential
store already has a valid login.

**2. Accept the Gemma conditions.** Authentication alone is not enough. The account must also
accept the usage conditions on the model page at
https://huggingface.co/google/gemma-3-1b-it. These are separate failures and `csf benchmark`
reports them separately, because the fixes are different.

The model is Gemma, not Gemini. The weights are about 2.0 GB.

## Systems benchmark

Before deciding whether to rent a GPU, `csf benchmark` answers three operational questions:
do the pinned weights load here, what does a forward pass cost on this machine, and does the
hook-owned capture path work on real weights.

Start with one item:

```powershell
uv run csf benchmark `
  --model-config configs/models/gemma3_1b_it.yaml `
  --task-config configs/tasks/arc_mcq.yaml `
  --split test `
  --max-items 1 `
  --warmup-runs 1 `
  --timed-runs 3
```

Then, only if the one-item timing looks reasonable, twenty:

```powershell
uv run csf benchmark `
  --model-config configs/models/gemma3_1b_it.yaml `
  --task-config configs/tasks/arc_mcq.yaml `
  --split test `
  --max-items 20 `
  --warmup-runs 2 `
  --timed-runs 5
```

Both need `csf data prepare` to have run first. `--offline` requires locally cached weights
and fails clearly if they are missing rather than reaching for the network.

### A systems benchmark is not a scientific result

This is the distinction the whole repository is built around, so the benchmark enforces it
rather than relying on anyone remembering it.

A **systems benchmark** measures whether the machine can run the model, and how fast. A
**CSF-Bench result** would measure whether a forecasting method predicts intervention
effects. The second requires a model organism, a validated direction, and a forecaster, none
of which exist yet.

Every benchmark artifact carries its classification:

```json
{ "classification": "systems_benchmark", "scientific_result": false, "fixture_only": false }
```

Fixture runs are labeled `fixture_systems_test` with `fixture_only: true`. `scientific_result`
is typed as a literal false, so a record claiming otherwise cannot be constructed at all. A
benchmark run has no forecasts or commitments in it, so `csf verify run` will not verify it
and the public exporter cannot accept it.

The accuracy a benchmark reports is a scoring smoke check over a handful of items. It is not a
capability measurement and must not be quoted as one.

### The compute decision

A successful benchmark includes a planning estimate built from the measured median forward
time:

```text
estimated_forward_count = number_of_prompts x model_states_per_prompt
                          x candidates_per_trial x forward_passes_per_candidate
estimated_cpu_seconds   = estimated_forward_count x measured_median_forward_seconds
```

The assumptions travel with the number in the artifact, and their source is
`docs/preregistration.md` section 9. The estimate reports whether a small clean-model
validation still looks practical on CPU, whether a full intervention sweep does, and whether
LoRA training does. It is arithmetic, not a prediction, and it does not rent anything. The
compute decision is the maintainer's.

### Known limitations

CPU timing is machine-specific and says nothing about other hardware. Latency is measured on
one representative prompt at batch size one, so it ignores batching. Memory is reported from
the OS where that is reliable, and reported as null with a reason where it is not, rather than
being estimated.

## Repository layout

```text
configs/       models, tasks, interventions, experiments
src/           the csf package
tests/         unit, integration, and the anti-fabrication guard
docs/          research plan, preregistration, methodology, claim boundaries, failure modes
results/runs/  run artifacts (git-ignored)
results/public/ verified exports (committed)
artifacts/     directions and adapters (weights git-ignored)
```

## Integrity rules

These are enforced by tests, not by good intentions:

* No invented results, sample counts, intervals, or hashes. Planned values are labeled planned.
* `tests/test_no_fake_results.py` fails if dashboard source contains hard-coded metric values.
* `PublicDashboardRecord` cannot be constructed for a run whose commitments did not verify.
* `MetricValue` cannot be constructed without a sample count and an interval.
* Model revisions cannot be `main`. `ModelSpec` rejects moving pointers.
* Salts, selection seeds, and private payloads never enter version control.

## Claim boundaries

This project cannot show that a model understands itself, knows why it answered, reveals its
true reasoning, or reveals hidden goals. A linear probe is not a self-report, and an attached
forecasting head is not introspection.

Read `docs/claim_boundaries.md` before describing any result from this repository.

## Safety

The model organism is deliberately benign: a controlled preference for one answer position
under a deployment-like wrapper, with accuracy preserved under evaluation framing. It is not
scheming and does not have goals. No harmful capability, filter evasion, autonomy, or tool use
is involved. See `SECURITY.md`.

## License

MIT. See `LICENSE`.
