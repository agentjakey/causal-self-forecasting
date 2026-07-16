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
Hugging Face: accept the license on the model page and authenticate (`hf auth login`) first.

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
