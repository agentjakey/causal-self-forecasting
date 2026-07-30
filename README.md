# causal-self-forecasting

**Does a language model's hidden state carry information about what an intervention will do to
its answer, beyond what you can already read off the prompt and the output?**

That is the whole question. It is narrower than "does the model understand itself", and it is
narrow on purpose, because it is answerable.

Working name: Causal Self-Forecasting Lab. Benchmark: CSF-Bench.

---

## The question in plain English

Take a small language model answering a four-choice question. Reach inside it, add a fixed
vector to one layer of its residual stream, and measure how much that shifts the answer it
preferred. Now try to **predict** that shift in advance.

Two predictors get the same job. Both see the prompt, the model's clean output distribution, and
a complete numerical description of the intervention. Only one of them also sees the model's
actual hidden state for that prompt.

* If the state-conditioned predictor does better, the residual stream carried something about the
  intervention's effect that the visible inputs did not.
* If it does not, the state added nothing detectable at this scale, and that is the finding.

Then the control that makes the first case mean anything: swap in **another prompt's** hidden
state. If performance holds up, the predictor was using *some* state, not *this prompt's* state,
and the claim collapses. Both results have to go the right way.

The formal statement, the hypotheses, and the decision rule are frozen in
[`docs/bluedot/preregistration_state_dependence.md`](docs/bluedot/preregistration_state_dependence.md).

## What this is not

This is an **external state-information audit**. The predictors are ridge regressions that we fit
and control. A ridge regression reading a residual stream is a readout, not a report, and the
model is not the thing doing the reporting.

Nothing here tests, and no result from it may be described as testing, introspection,
consciousness, self-awareness, faithful verbal reasoning, hidden goals, deception, or deployment
readiness.

Read [`docs/claim_boundaries.md`](docs/claim_boundaries.md) before describing any number from this
repository.

## Current status

**No forecasting result exists.** No forecaster has been fitted, no forecast has been committed,
and no method has been compared to another. Nothing in this repository is evidence about how
language models reason.

What has actually been measured, on real pinned `google/gemma-3-1b-it` weights at revision
`dcc83ea841ab6100d6b47a070329e1ba4cf78752`:

| Stage | State | What it produced |
| --- | --- | --- |
| Systems benchmark | done | Load time, forward cost, and hook-owned capture verified at layer 13 by patching a known vector and reading it back. Classified `systems_benchmark`. |
| Intervention harness controls | done | All five required controls pass, including no-op equality and sign reversal. |
| Frozen prompt split | done | 168 prompts, disjoint by item and group across four roles, drawn deterministically from 256 eligible ARC groups. No model involved. |
| Direction family | done | Eight unit directions built from the pinned unembedding. Construction, not causal validation. |
| Calibration plan | done | Target, ratio grid, two permitted layers, and six pass conditions frozen before any number existed. |
| Engineering smoke | done | 8 prompts, 144 forwards, 136 observations, no failures, verified, deterministic across processes. Plumbing validation. |
| Calibration sweep | done | 32 prompts, 2,624 forwards, 2,592 observations, no failures, verified. Passed at layer 13 and selected the smallest passing ratio. A choice of stimulus, not a measurement. |
| Training set, forecasters, final test | not started | This is where a result would come from. |

Every measured number lives in [`docs/experiment_log.md`](docs/experiment_log.md), read from
verified run artifacts. None is restated here, because a number copied by hand into a README is a
number that will eventually be wrong.

Two things that look like results and are not:

* **Clean accuracy over a handful of items** is a scoring smoke check, not a capability
  measurement. The study's target is defined against the model's own clean preferred answer, not
  against the dataset key, so clean correctness does not enter the measurement at all.
* **Effect sizes from the engineering smoke** select nothing. Its ratio and layer were fixed in
  advance precisely so they could not be chosen with the effect distribution in view, and
  `csf calibration summarize` refuses observations whose prompt role is not `calibration`.

## Study design

One model, one task, one layer, one intervention family, linear readouts only.

| Item | Value |
| --- | --- |
| Model | `google/gemma-3-1b-it`, revision pinned, CPU float32, hidden dim 1152 |
| Task | ARC-Challenge, four choices, one neutral prompt wrapper |
| Prompts | 168: 8 smoke, 32 calibration, 96 training, 32 final test, disjoint by item and group |
| State | Residual stream at the final prompt token, layer 13 primary, layer 20 the only fallback |
| Target | `delta_clean_top_margin`: the shift in the margin around the model's **own** clean preferred answer, with that answer held fixed after the intervention |
| Directions | 8 unit vectors: 4 centered answer-token unembedding directions, 4 seeded controls orthogonal to their span and to each other |
| Candidates | 17 per prompt (8 directions x 2 signs, plus a no-op); 81 at calibration (x 5 ratios) |
| Strength | One global alpha per layer and ratio, `ratio * median clean state norm over the calibration prompts`. Never prompt-relative. |
| Methods | Ridge regressions only: intervention-only, visible-information, and state-conditioned bilinear |
| Primary control | Deterministic nearest matched **wrong state** from the other final-test prompts |
| Analysis | Prompt-first aggregation, paired bootstrap over prompt groups, decision rule fixed in advance |

Three design choices carry most of the weight:

**The strength is global, not prompt-relative.** `public_view` publishes an intervention's
strength to every method. If the strength were `ratio * ||h_prompt||`, the visible-information
baseline would silently receive the prompt's state norm, and the exact comparison this study
exists to make would be contaminated at the source. There is no prompt-specific strength function
anywhere in the code, and `check_global_alpha` refuses a set of observations that used more than
one alpha at a grid point.

**Candidate identity is opaque.** A real steer and a matched random control are published
identically: same operation, same layer, same strength, different meaningless id. Naming the
mechanism would let a method dismiss controls without ever consulting the model's state. The
record type refuses an identifier that names a construction role, an answer label, or a family.

**Thresholds are frozen before the numbers exist.** The calibration plan fixes the target, the
five ratios, the two permitted layers, and the six pass conditions, and the run is judged against
the plan rather than against anything it measured. The selector takes the **smallest** passing
ratio in preregistered order, never the largest effect, because choosing the stimulus by the
outcome would make the comparison circular. Layer 20 is refused unless a layer-13 run recorded
`fallback_required`.

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/). No GPU. The whole study is about 40 minutes of
CPU forward time; see [`docs/compute_decision.md`](docs/compute_decision.md).

```powershell
uv sync --extra dev --extra torch
```

The test suite is fully offline: it builds its own tiny fixture model and never downloads
weights.

Real Gemma weights are gated. Two separate things are needed, and `csf benchmark` reports them
separately because the fixes differ:

1. **Authenticate.** `uv run hf auth login`, or `uvx hf auth login` to avoid installing into the
   project environment. The token is read from the Hugging Face credential store; this project
   never prints, logs, or writes it to an artifact.
2. **Accept the Gemma conditions** on https://huggingface.co/google/gemma-3-1b-it. Authentication
   alone is not enough.

The model is Gemma, not Gemini. The weights are about 2.0 GB.

## Quick verification

Everything here runs offline in a few minutes and loads no pretrained weights.

```powershell
uv run csf doctor          # environment plus every config validated
uv run pytest              # the full suite, including the anti-fabrication guard
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

To check the frozen artifacts without running anything:

```powershell
uv run csf prompts verify --manifest-id bluedot_state_dependence_v1
uv run csf directions verify-family --manifest-id bluedot_state_dependence_directions_v1
uv run csf calibration verify-plan --plan-id bluedot_state_dependence_calibration_v1
```

Each of those recomputes the artifact's own content hash before checking anything else, so an
edited manifest fails to load rather than verifying.

## Running the study

Stages run in this order. Each refuses to start if the artifact it depends on does not verify.

**1. Prepare the task.** Needs network access; everything after this is offline.

```powershell
uv run csf data prepare --config configs/tasks/arc_mcq.yaml
```

**2. Freeze the prompt split.** No model, no forward pass. Selection is a pure function of the
master seed and the group ids: it never reads correctness, confidence, logits, hidden states, or
any outcome, so the split cannot be chosen to suit a result.

```powershell
uv run csf prompts manifest --config configs/prompts/bluedot_state_dependence.yaml
uv run csf prompts verify --manifest-id bluedot_state_dependence_v1
```

**3. Build the direction family.** Reads the pinned model's output embedding and nothing else.
No prompt is run and no state is captured.

```powershell
uv run csf directions build-family --config configs/directions/bluedot_state_dependence.yaml
uv run csf directions verify-family --manifest-id bluedot_state_dependence_directions_v1
uv run csf directions verify-family --manifest-id bluedot_state_dependence_directions_v1 --regenerate
```

Without `--regenerate` this loads no model: it checks the manifest hash, every stored vector's
content hash, dimensions, norms, orthogonality, and completeness. With `--regenerate` it rebuilds
all eight from the pinned weights and compares, writing nothing.

**4. Freeze the calibration plan.** No model. This is what makes a threshold a threshold.

```powershell
uv run csf calibration plan --config configs/calibration/bluedot_state_dependence.yaml
uv run csf calibration verify-plan --plan-id bluedot_state_dependence_calibration_v1
```

**5. Engineering smoke.** 8 prompts, 144 forwards, at a ratio fixed in advance because it is
arbitrary. Proves the pipeline end to end where a mistake is cheap.

```powershell
uv run csf state-audit smoke --config configs/state_audit/bluedot_smoke.yaml --run-id bluedot-smoke-layer13
uv run csf state-audit verify-run --run-id bluedot-smoke-layer13
```

**6. Calibration.** 32 prompts, 2,624 forwards. Captures every clean state first, takes the
median as the reference norm **before** any intervention runs, derives one alpha per frozen
ratio, sweeps the grid, then applies the six preregistered conditions.

```powershell
uv run csf state-audit calibrate --config configs/state_audit/bluedot_calibration_layer13.yaml --run-id bluedot-calibration-layer13
uv run csf state-audit verify-run --run-id bluedot-calibration-layer13
```

The layer-20 fallback has its own config and is refused unless `--primary-run-id` names a layer-13
run whose decision record says `fallback_required`, so it cannot become a second attempt.

**7. The fixed intervention projection.** Loads no model weights. Generated once from the master
seed, never fitted, and cited by every forecast.

```powershell
uv run csf state-audit projection --config configs/state_audit/bluedot_training.yaml --projection-id bluedot_state_dependence_projection_v1
```

**8. Training.** 96 prompts, 1,728 forwards, at the strength calibration chose. The strength is
inherited from the calibration decision, not recomputed, so predictors are fitted on the same
stimulus the final test will be scored on.

```powershell
uv run csf state-audit train --config configs/state_audit/bluedot_training.yaml --run-id bluedot-training
uv run csf state-audit verify-run --run-id bluedot-training
```

**9. Final-test clean stage.** 32 clean forwards and nothing else. No candidate set is built and
no intervention hook is registered, so no final-test outcome exists.

```powershell
uv run csf state-audit final-test-clean --config configs/state_audit/bluedot_final_test_clean.yaml --run-id bluedot-final-test
```

**10. Fit and commit.** Loads no model. Fits the transforms and the three ridges on the 96
training prompts only, builds the wrong-state pairing and the ten derangements, and commits every
final-test forecast before any intervention is applied.

```powershell
uv run csf state-audit commit-forecasts `
  --training-config configs/state_audit/bluedot_training.yaml --training-run-id bluedot-training `
  --final-test-config configs/state_audit/bluedot_final_test_clean.yaml --final-test-run-id bluedot-final-test `
  --projection-id bluedot_state_dependence_projection_v1
uv run csf state-audit verify-commitments --run-id bluedot-final-test
```

**11. Final-test resolution.** Implemented, **not yet executed**. This is the irreversible step:
544 intervened forwards (32 prompts x 17 candidates), reusing the clean logits and states so no
clean forward runs and every delta is measured against the baseline the forecasts were made
against. It refuses a dirty working tree, a setting that differs from the calibration decision, a
commitment count other than 512, any pre-existing reveal, and any pre-existing outcome.

```powershell
uv run csf state-audit resolve-final-test `
  --config configs/state_audit/bluedot_final_test_clean.yaml --run-id bluedot-final-test `
  --layer 13 --norm-ratio 0.02 --global-alpha 106.87158268272867 `
  --yes-i-understand-this-is-irreversible
```

**12. Analysis and replay.** Model-free. Nothing is fitted, refitted, tuned, or dropped.

```powershell
uv run csf state-audit analyze-final-test --run-id bluedot-final-test --training-run-id bluedot-training
uv run csf state-audit replay-analysis --run-id bluedot-final-test
```

## Results

**No result exists yet.** The final test has not been resolved: `results/runs/bluedot-final-test/`
holds 512 sealed commitments and zero outcomes, and the analysis has nothing to read.

When it is resolved the numbers will appear in `docs/experiment_log.md` and in machine-readable
form at `state_audit_analysis.json`, `state_audit_method_summary.json`,
`state_audit_prompt_scores.jsonl`, and `state_audit_pair_scores.jsonl` inside the run directory,
with two figures under `figures/`. Nothing will be copied into this README by hand.

Two things are fixed in advance so they cannot be renegotiated afterwards. Absolute errors are
averaged **within each prompt first**, then across the 32 prompts, because the 16 interventions on
one prompt share a question and a state. And a comparison supports its hypothesis only if the 95
percent paired bootstrap interval excludes zero in the hypothesized direction: **an interval
crossing zero is reported as no detected difference**, never as a trend.

For context on which way the current evidence points, the training-fold cross-validated errors
already favour the *visible-information* model over the state-conditioned one (0.3433 against
0.3617). That is a training-fold diagnostic and not the test, but a null or negative result on
H-BD1 would be unsurprising, and it will be reported as prominently as a positive one would be.

### Verification loads no model

`csf state-audit verify-run` recomputes, from files on disk: the run manifest's own content hash,
every artifact hash, every observation's target from its own logits, the reference norm from the
recorded clean state norms, and the single global alpha per grid point. For a calibration run it
also re-derives the decision from the summaries beside it and refuses a selected ratio that is not
the smallest passing one.

`--compare-run-id` compares two runs of the same inputs row by row, which is how cross-process
determinism is measured. A completed run at the same run id is refused rather than overwritten,
because its artifacts are the only record of what happened.

### Offline fixture pipeline

The original CSF-Bench trial and commitment loop still runs end to end on the fixture model:

```powershell
uv run csf directions synthetic --config configs/experiments/smoke.yaml
uv run csf interventions validate --config configs/experiments/smoke.yaml
uv run csf trials generate --config configs/experiments/smoke.yaml --max-trials 8
uv run csf trials resolve --run-id <RUN_ID> --ground-truth
```

There is no CLI command that commits a forecast; `commit_forecasts` is reachable from Python
only. The full generate, resolve, fit, commit, resolve, score loop is exercised by
`tests/integration/test_resolve.py::test_full_pipeline_generate_resolve_fit_commit_score`, which
is the working example to copy.

## Repository structure

```text
configs/
  models/         pinned model configs, revisions as commit shas
  tasks/          dataset and prompt-wrapper definitions
  prompts/        frozen prompt-split config
  directions/     direction-family construction config
  calibration/    frozen calibration plan config
  state_audit/    smoke and calibration run configs
  experiments/    the offline fixture pipeline
  interventions/  intervention grids
data/
  manifests/           task manifest, content-hashed (tracked)
  prompt_manifests/    the frozen 168-prompt split (tracked)
  direction_manifests/ the direction family (tracked)
  calibration_plans/   the frozen calibration plan (tracked)
  processed/           prepared ARC data (git-ignored, regenerated)
src/causal_self_forecasting/
  schemas.py           every record that reaches disk, self-verifying
  state_audit_target.py  the study target, as pure functions
  models/         loading, answer scoring, hook-owned capture
  interventions/  tensor operations, direction store, direction family
  tasks/          dataset loading, prompt rendering, the frozen split
  trials/         trial generation, candidates, commitment, resolution
  calibration/    strength rule, pass conditions, layer state machine
  state_audit/    candidate builders, run execution, artifact verification
  forecasting/    the two model-free baselines
  scoring/        metrics and scoring
tests/            unit, integration, and the anti-fabrication guard
docs/             preregistrations, methodology, claim boundaries, logs
results/runs/     run artifacts (git-ignored)
results/public/   verified exports (tracked; none exist yet)
artifacts/        direction vectors and fixtures (git-ignored, regenerable)
```

## Reproducibility

The frozen artifacts are tracked and tiny. Committing them is what freezes a decision; a split
that exists on one machine is not a split.

Every artifact recomputes its own content hash when it loads, so an edited file fails to parse
rather than quietly verifying. Every run manifest cites the hashes of everything upstream of it,
and the chain is checkable end to end without a GPU:

```text
task manifest -> prompt manifest -> direction family -> calibration plan -> run manifest
```

Determinism comes from one master seed, `20260727`, threaded through `derive_seed` with a
namespaced label per purpose, so changing one seeded step cannot shift another. Commitment salts
are the one deliberate exception: they come from the OS CSPRNG.

To reproduce from scratch: install, run `csf data prepare`, then stages 2 through 6 above. The
prompt manifest, direction family, and calibration plan should come out byte-identical to the
tracked ones; a rerun reports `unchanged` and leaves the file untouched rather than rewriting it.

Integrity rules enforced by tests rather than by good intentions:

* No invented results, sample counts, intervals, or hashes. Planned values are labeled planned.
* `tests/test_no_fake_results.py` fails if a public export exists that did not verify.
* `PublicDashboardRecord` cannot be constructed for an unverified run, and `MetricValue` cannot be
  constructed without a sample count and an interval.
* `scientific_result` is a typed `Literal[False]` on every engineering record, so a record
  claiming otherwise cannot be constructed at all.
* Model revisions cannot be `main`; `ModelSpec` rejects moving pointers.
* Salts, selection seeds, and private payloads never enter version control, and CI greps the
  tracked file list to confirm it.

## Documentation

| Document | What it is for |
| --- | --- |
| [`docs/bluedot/preregistration_state_dependence.md`](docs/bluedot/preregistration_state_dependence.md) | The frozen design of the active study. Read this first. |
| [`docs/bluedot/execution_decision_tree.md`](docs/bluedot/execution_decision_tree.md) | The order of operations and the gate at every branch point. |
| [`docs/claim_boundaries.md`](docs/claim_boundaries.md) | What may and may not be said about any result. |
| [`docs/methodology.md`](docs/methodology.md) | How the harness actually works, and why. |
| [`docs/experiment_log.md`](docs/experiment_log.md) | Every measured number, read from a verified artifact. |
| [`docs/compute_decision.md`](docs/compute_decision.md) | What the one measured forward time implies for cost. |
| [`docs/failure_modes.md`](docs/failure_modes.md) | Fifteen ways a result could look good and mean nothing. |
| [`docs/deferred_work.md`](docs/deferred_work.md) | What is designed, deliberately not built, and why. |
| [`docs/preregistration.md`](docs/preregistration.md) | The broader CSF-Bench study. Unedited, still in force outside this arm. |

## Safety

The active study fine-tunes nothing, trains nothing, and uses no model organism. It adds a fixed
vector to a residual stream and reads four logits. The harm ceiling is a wrong letter on a
multiple-choice question.

A benign model organism is in the broader design and has not been built. See
[`SECURITY.md`](SECURITY.md) and [`docs/deferred_work.md`](docs/deferred_work.md).

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the software, and please do not cite it as
evidence of a finding, because no forecasting result exists yet.

## License

MIT. See [`LICENSE`](LICENSE). Gemma weights are not redistributed here and remain under Google's
license.
