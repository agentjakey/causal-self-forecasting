# causal-self-forecasting

A reproducible benchmark for testing whether language models can forecast how blinded internal
interventions will change their outputs, with calibrated scoring, hidden-state controls, and
held-out mechanism evaluation.

Working name: **Causal Self-Forecasting Lab**. Benchmark: **CSF-Bench**.

## Status

**No CSF-Bench scientific result exists.** There is no model organism, no estimated and
validated direction, no trained state-conditioned forecaster, and no verified public export.

The pipeline runs end to end: on the fixture model in the test suite, and on real Gemma 3 1B
for systems and harness validation. What has been measured on real weights is a systems
benchmark (load time, forward cost, capture correctness) and an intervention harness validation
(no-op equality, sign reversal, four observed candidates on one item). Both are explicitly
non-scientific: the directions used are synthetic and unvalidated, and single-item accuracy is
a scoring smoke check, not a capability measurement.

Nothing in this repository should be read as a finding about how language models reason. See
`docs/experiment_log.md` for exactly what has been measured and `docs/compute_decision.md` for
what the measured forward cost implies.

### Active work: the BlueDot state-dependence arm

As of 2026-07-27 the active experiment is a narrower arm, preregistered before any run:

> Does access to Gemma 3 1B's correct prompt-specific hidden state improve forecasts of how a
> fixed internal intervention changes the model's clean preferred answer, beyond the prompt, the
> clean output distribution, and a complete numerical representation of the intervention?

It is an external state-information audit. It does not test introspection, consciousness,
self-awareness, faithful verbal reasoning, hidden goals, or deployment readiness.

* `docs/bluedot/preregistration_state_dependence.md` freezes the design. It supersedes
  `docs/preregistration.md` for this arm only; the original is unedited and still governs the
  broader study.
* `docs/bluedot/execution_decision_tree.md` freezes the order of operations and the gate at every
  branch point.
* `docs/bluedot/current_state_audit.md` is the read-first audit the arm was scoped against.

The arm uses one clean pinned model and no model organism, ridge regressions and no MLP, and 168
prompts. Estimated cost is about 40 minutes of CPU forward time. **No GPU and no compute grant is
needed.** The model organism, LoRA training, learned behavioral directions, the state MLP, the
verbal reporter, SAE work, the 4B replication, mechanism transfer, and the dashboard are all
deferred out of this arm's path; they remain part of the broader roadmap in
`docs/research_plan.md`.

No calibration, training, or final-test run has been executed and no result exists.

The arm's first slice is done: the prompt split is frozen at
`data/prompt_manifests/bluedot_state_dependence_v1.json`, 168 prompts on the `neutral_a` wrapper
drawn deterministically from 256 eligible ARC groups at seed 20260727, disjoint by group and by
item across all four roles. Building it loads no model and runs no forward pass:

```powershell
uv run csf prompts manifest --config configs/prompts/bluedot_state_dependence.yaml
uv run csf prompts verify --manifest-id bluedot_state_dependence_v1
```

Selection is a pure function of the master seed and the group ids. It never reads model
correctness, confidence, logits, hidden states, or any outcome, so the split cannot be chosen to
suit a result. Rerunning leaves an identical manifest byte-identical rather than rewriting it,
and a manifest that differs is refused unless `--force` is passed. Committing the manifest is
what freezes the split; see `docs/experiment_log.md` for the hashes.

The direction family is frozen too: eight unit directions at
`data/direction_manifests/bluedot_state_dependence_directions_v1.json`, four centered
answer-token unembedding directions and four seeded controls orthogonal to their span and to
each other. Building them reads the pinned model's output embedding and nothing else, and runs
no prompt.

```powershell
uv run csf directions build-family --config configs/directions/bluedot_state_dependence.yaml
uv run csf directions verify-family --manifest-id bluedot_state_dependence_directions_v1
uv run csf directions verify-family --manifest-id bluedot_state_dependence_directions_v1 --regenerate
```

`verify-family` without `--regenerate` loads no model: it checks the manifest's own hash, every
stored vector's content hash, dimensions, norms, orthogonality, and family completeness. With
`--regenerate` it rebuilds all eight from the pinned weights and compares, writing nothing.

The calibration plan is frozen too, at
`data/calibration_plans/bluedot_state_dependence_calibration_v1.json`. It fixes the target, the
five norm ratios, the two permitted layers, and the six pass conditions **before** any
calibration number exists, which is the only thing that makes a threshold a threshold.

```powershell
uv run csf calibration plan --config configs/calibration/bluedot_state_dependence.yaml
uv run csf calibration verify-plan --plan-id bluedot_state_dependence_calibration_v1
uv run csf calibration summarize --plan-id ... --observations obs.jsonl --layer 13
uv run csf calibration select --plan-id ... --summaries summaries.json
```

None of these loads a model. Strength is one global alpha per layer and ratio, never a
per-prompt one: `public_view` publishes `strength` to every method, so a prompt-relative
strength would hand the visible-information baseline the prompt's state norm. The selector takes
the **smallest** passing ratio in preregistered order, never the largest effect, because
choosing the stimulus by the outcome would make the comparison circular.

The eight-prompt engineering smoke has been executed on the pinned weights. It is the first
BlueDot step that produced measured numbers, and it is plumbing validation rather than science:
144 forwards, `scientific_result: false`, and a ratio and layer that were both fixed in advance
so that neither could be chosen with the effect distribution in view.

```powershell
uv run csf state-audit smoke --config configs/state_audit/bluedot_smoke.yaml --run-id bluedot-smoke-layer13
uv run csf state-audit verify-run --run-id bluedot-smoke-layer13
uv run csf state-audit verify-run --run-id bluedot-smoke-layer13 --compare-run-id <SECOND_RUN_ID>
```

`smoke` loads the model; `verify-run` does not. Verification recomputes the run manifest's own
content hash, every artifact hash, every observation's target from its own logits, the reference
norm from the recorded clean state norms, and the single global alpha across every non-no-op
observation. `--compare-run-id` compares two runs of the same inputs row by row, which is how
cross-process determinism is measured. A completed run at the same id is refused rather than
rewritten, because its artifacts are the only record of what happened. Measured values are in
`docs/experiment_log.md`.

**No calibration has been run and no ratio has been selected.** The smoke's effect sizes select
nothing: `csf calibration summarize` refuses observations whose prompt role is not
`calibration`, so a strength cannot be chosen from prompts that were not set aside to choose it.

**Constructing a direction is not validating one.** These are stimuli with a recorded recipe.
Their artifacts carry `validated: false`, and nothing about them licenses calling any direction
meaningful, load-bearing, or bias-related. The stored ids are opaque hash prefixes and the
mapping to construction roles lives only in the private manifest, so a forecaster cannot read
family membership off an id.

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

# Freeze the BlueDot prompt split (offline, no model, no forward pass)
uv run csf prompts manifest --config configs/prompts/bluedot_state_dependence.yaml
uv run csf prompts verify --manifest-id bluedot_state_dependence_v1

# Build the BlueDot direction family (loads the pinned model's unembedding; runs no prompt)
uv run csf directions build-family --config configs/directions/bluedot_state_dependence.yaml
uv run csf directions verify-family --manifest-id bluedot_state_dependence_directions_v1

# Freeze the calibration plan (offline, no model)
uv run csf calibration plan --config configs/calibration/bluedot_state_dependence.yaml
uv run csf calibration verify-plan --plan-id bluedot_state_dependence_calibration_v1

# BlueDot engineering smoke on the pinned weights (144 forwards), then verify from artifacts
uv run csf state-audit smoke --config configs/state_audit/bluedot_smoke.yaml --run-id bluedot-smoke-layer13
uv run csf state-audit verify-run --run-id bluedot-smoke-layer13

# Smoke pipeline on the fixture model, offline
uv run csf directions synthetic --config configs/experiments/smoke.yaml
uv run csf interventions validate --config configs/experiments/smoke.yaml
uv run csf trials generate --config configs/experiments/smoke.yaml --max-trials 8

# Apply interventions and record observations (ground-truth mode, all candidates)
uv run csf trials resolve --run-id <RUN_ID> --ground-truth
```

`csf score run` is deliberately not in that list. It scores *committed forecasts* against
observations and raises if a run has none, and a ground-truth resolution commits nothing. There
is currently **no CLI command that commits a forecast**: `commit_forecasts` is reachable from
Python only. The full generate, resolve, fit, commit, resolve, score loop is exercised end to
end by `tests/integration/test_resolve.py::test_full_pipeline_generate_resolve_fit_commit_score`,
which is the working example to copy until a `csf forecast` command exists.

`csf doctor` reports the environment and validates every config. It does not fail because a
GPU is missing; it says so.

### Resolution and scoring

`csf trials resolve` has two modes. Forecast mode (the default) requires committed forecasts,
selects one candidate per trial after commitment, applies it, and reveals and verifies the
commitment. Ground-truth mode (`--ground-truth`) applies every candidate and records
observations only, which is what trains the baselines and validates the harness on real
weights. Neither mode produces a scientific result, and both refuse to resolve a systems
benchmark or to treat a fixture run as scientific.

`csf score run` matches committed forecasts to observations. No-op candidates are excluded from
the headline metrics and reported separately, every metric carries its sample count and a
group-bootstrapped interval, and scores are written to the run directory only, never to the
public results tree. It refuses to score a scientific run whose commitments did not verify.

The two baselines that need no model organism live in `csf`'s forecasting module: a constant
predictor (training-split averages by public operation, layer, and strength) and a prompt-only
lexical model (TF-IDF of the prompt plus public strength and layer). Both are fenced to public
information: neither can see a hidden state, an adapter identity, a private direction, a correct
answer, or an outcome from the split it will be scored on. That fence is what makes the
same-prompt state-swap comparison meaningful, and it is checked by a leakage audit test.

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
