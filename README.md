# causal-self-forecasting

**Does a language model's hidden state carry information about what an intervention will do to
its answer, beyond what you can already read off the prompt and the output?**

That is the whole question. It is narrower than "does the model understand itself", and it is
narrow on purpose, because it is answerable.

On Gemma 3 1B, **no improvement from state access was detected under this setup**. This repository
holds the preregistered experiment that produced that null, and every artifact needed to check it.
The null is not a claim of equivalence: with 32 final-test prompts the intervals are wide enough to
be consistent with a small true effect in either direction.

Working name: Causal Self-Forecasting Lab. Benchmark: CSF-Bench.

---

## The question in plain English

Take a small language model answering a four-choice question. Reach inside it, add a fixed vector
to one layer of its residual stream, and measure how much that shifts the answer it preferred. Now
try to **predict** that shift in advance.

Two predictors get the same job. Both see the prompt, the model's clean output distribution, and a
complete numerical description of the intervention. Only one of them also sees the model's actual
hidden state for that prompt.

Then the control that makes a win mean anything: swap in **another prompt's** hidden state. If
performance holds up, the predictor was using *some* state, not *this prompt's* state, and the
claim collapses.

* **H-BD1.** The state-conditioned ridge beats the visible-information ridge. **Not supported.**
* **H-BD2.** That advantage depends on the state belonging to the prompt. **Not supported.**

## What this is not

This is an **external state-information audit**. The predictors are ridge regressions that we fit
and control. A ridge regression reading a residual stream is a readout, not a report, and the model
is never asked about itself.

Nothing here tests, and no result from it may be described as testing, introspection,
consciousness, self-awareness, faithful verbal reasoning, hidden goals, deception, or deployment
readiness. See [Claim boundaries](#claim-boundaries) before describing any number below.

## Status: complete

The final test resolved on 2026-07-30 and is spent. It cannot be rerun: the resolver refuses a run
directory that already holds outcomes, and the 32 final-test prompts have now been used.

| Stage | State | Result |
| --- | --- | --- |
| Frozen prompt split | done | 168 prompts, disjoint by item and group, no model involved |
| Direction family | done | 8 unit directions built from the pinned unembedding |
| Calibration plan | done | Target, ratio grid, two permitted layers, six pass conditions, frozen before any number existed |
| Engineering smoke | done | 8 prompts, 144 forwards, deterministic across processes |
| Calibration sweep | done | 32 prompts, 2,624 forwards, 0 failures. Selected layer 13, ratio 0.02, alpha 106.87158268272867 |
| Training | done | 96 prompts, 1,728 forwards, 0 failures. Three ridges fitted and frozen |
| Commitment | done | 512 forecasts sealed under salted hashes before any final-test intervention |
| Final test | done | 544 intervened forwards, 0 failures, 512/512 commitments verified and revealed |
| Analysis | done | Both primary intervals cross zero. Null result |

## Results

Errors are averaged within each prompt first, over that prompt's 16 non-no-op interventions, then
across the 32 prompts. All rows are over 32 prompts and 512 signed pairs. **Every value is a point
estimate**; only the two comparisons below carry intervals, so differences between other rows are
not established as differences.

| Method | MAE | RMSE | Sign accuracy | Spearman | Top-effect accuracy |
| --- | --: | --: | --: | --: | --: |
| Intervention-only ridge | 0.318821 | 0.402262 | 0.630859 | 0.357776 | 0.34375 |
| Visible-information ridge | 0.321518 | 0.404996 | 0.630859 | 0.353763 | 0.34375 |
| True-state bilinear ridge | 0.335244 | 0.421573 | 0.580078 | 0.296136 | 0.125 |
| Matched wrong-state ridge | 0.340416 | 0.426767 | 0.566406 | 0.183944 | 0.0625 |
| Constant | 0.347682 | 0.433596 | 0.560547 | 0.056294 | 0.125 |
| Prompt lexical | 0.347937 | 0.434402 | 0.552734 | 0.057504 | 0.125 |

Primary comparisons, 10,000 paired bootstrap resamples over the 32 prompt groups, seed
`1394099119`:

| Comparison | Point estimate | 95 percent interval | Verdict |
| --- | --: | --- | --- |
| Visible minus true-state MAE | -0.013725 | [-0.025723, 0.00003] | crosses zero, H-BD1 not supported |
| Matched wrong-state minus true-state MAE | 0.005172 | [-0.003081, 0.013412] | crosses zero, H-BD2 not supported |

The first upper endpoint is 0.00003, a small positive number, written that way rather than as
0.0000 so it is not read as an interval terminating exactly at zero. The stored value is
2.8966819969343353e-05.

Ten shuffled-state controls give MAE mean 0.339221, median 0.339280, range 0.335087 to 0.342696,
with the true-state MAE at the lower edge. **That band is descriptive**: ten point estimates over
ten seeds, not a null distribution, not an interval, and not a test. Nothing is concluded from it.
The Brier score was omitted: only 6 answer flips occurred, below the preregistered floor of 20.

![Final-test MAE by method and state condition](paper/figures/final_test_mae_by_method.png)

![Primary comparisons with 95 percent paired bootstrap intervals](paper/figures/final_test_primary_comparisons.png)

> In this preregistered Gemma 3 1B experiment there was no detected improvement, under this setup,
> from prompt-specific hidden-state features over intervention and visible-output information. The
> intervention-only ridge had the lowest point-estimate error, and replacing the true hidden state
> with matched or shuffled states did not cause a statistically detected degradation.

Under the decision rule fixed before the data existed, an interval crossing zero is no detected
difference. Neither comparison is a trend and neither is described as one. Neither is an equivalence
claim either: no equivalence bound was preregistered and no power analysis was run.

The state-conditioned model's higher point estimate and its training-fold diagnostics are consistent
with the bilinear representation having added variance or overfitted: 327 features against 55 and
16, the worst cross-validated error of the three, and a grid-edge selection of the largest available
ridge penalty. That is a reading of a pattern in point estimates, not a confirmed mechanism and not
a tested comparison.

Full write-ups: **[`REPORT.md`](REPORT.md)** for the technical report, **[`paper/main.tex`](paper/main.tex)**
for the workshop paper.

## Claim boundaries

What this experiment supports:

* On `google/gemma-3-1b-it` at layer 13, with this intervention family and this strength, a ridge
  regression given a 16-component PCA of the clean residual stream and its bilinear interaction
  with the intervention did not predict `delta_clean_top_margin` better than a ridge given the
  prompt, the clean logits, and the same intervention encoding.
* Substituting a matched wrong state did not degrade that model by a detectable amount.

What it does not support:

* That hidden states carry no information about intervention effects. This is one readout, one
  layer, one representation, and 32 test prompts.
* That the methods are equivalent, or that the state-conditioned model is worse. One point estimate
  is higher; the interval for that contrast includes zero, and no equivalence bound was set.
* Anything resting on the ten shuffled-state controls, which are descriptive only.
* Anything about introspection, self-report, consciousness, faithful reasoning, hidden goals,
  deception, or models larger than 1B.
* Any comparison to released explainer models. This is not a replication of one, and no such
  reproduction was attempted.

Known limits: one model, one layer, one task family, 32 final-test prompts, linear predictors,
PCA-compressed states, only six answer flips, and an intervention-only representation that may
already contain most of the predictable signal. At the selected strength the answer-token
directions did not flip answers more often than norm-matched random controls. The smoke,
calibration, training, and final-test clean stages all ran with a dirty working tree, recorded as
such in their manifests; resolution and analysis ran from clean commit `aac5532`.

## Study design

One model, one task, one layer, one intervention family, linear readouts only.

| Item | Value |
| --- | --- |
| Model | `google/gemma-3-1b-it`, revision `dcc83ea841ab6100d6b47a070329e1ba4cf78752`, CPU float32, hidden dim 1152 |
| Task | ARC-Challenge, four choices, one neutral prompt wrapper |
| Prompts | 168: 8 smoke, 32 calibration, 96 training, 32 final test, disjoint by item and group |
| State | Residual stream at the final prompt token, layer 13 |
| Target | `delta_clean_top_margin`: the shift in the margin around the model's **own** clean preferred answer, held fixed after the intervention |
| Directions | 8 unit vectors: 4 centered answer-token unembedding directions, 4 seeded controls orthogonal to their span and to each other |
| Candidates | 17 per prompt (8 directions x 2 signs, plus a no-op); 81 at calibration (x 5 ratios) |
| Strength | One global alpha, `ratio * median clean state norm over the calibration prompts`. Never prompt-relative |
| Methods | Ridge regressions only: intervention-only (16 features), visible-information (55), state-conditioned bilinear (327) |
| Primary control | Deterministic nearest matched **wrong state** from the other final-test prompts |
| Analysis | Prompt-first aggregation, paired bootstrap over prompt groups, decision rule fixed in advance |

Three design choices carry most of the weight:

**The strength is global, not prompt-relative.** Every method is published an intervention's
operation, layer, position, and strength. If the strength were `ratio * ||h_prompt||`, the
visible-information baseline would silently receive the prompt's state norm, and the exact
comparison this study exists to make would be contaminated at the source. There is no
prompt-specific strength function anywhere in the code, and `check_global_alpha` refuses a set of
observations that used more than one alpha at a grid point.

**Semantic direction labels are hidden; the vectors are not.** A real steer and a matched random
control are published with the same operation, layer, position, and strength, and a meaningless id,
and the record type refuses an identifier that names a construction role, an answer label, or a
family. The vector itself is fully available: every method receives `P^T v`, a fixed 16-dimensional
projection of the same vector added to the residual stream. A predictor can tell directions apart
numerically; what it cannot do is read the mechanism off the metadata and discount the controls
without consulting anything else.

**Thresholds were frozen before the numbers existed.** The calibration plan fixed the target, the
five ratios, the two permitted layers, and the six pass conditions, and the run was judged against
the plan. The selector takes the **smallest** passing ratio in preregistered order, never the
largest effect. Ratio 0.02 was selected; 0.10, 0.20, and 0.40 each failed on exactly one condition,
the p95 ceiling of 4.0. Layer 20 was prohibited because layer 13 produced a usable ratio.

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/). No GPU.

```powershell
uv sync --extra dev --extra torch
```

The test suite is fully offline: it builds its own tiny fixture model and never downloads weights.

Real Gemma weights are gated. Two separate things are needed, and `csf benchmark` reports them
separately because the fixes differ:

1. **Authenticate.** `uv run hf auth login`, or `uvx hf auth login` to avoid installing into the
   project environment. The token is read from the Hugging Face credential store; this project
   never prints, logs, or writes it to an artifact.
2. **Accept the Gemma conditions** on https://huggingface.co/google/gemma-3-1b-it. Authentication
   alone is not enough.

The model is Gemma, not Gemini. The weights are about 2.0 GB.

## Verifying the result

Everything in this section runs offline and loads no pretrained weights.

**Re-derive the result from the published bundle.** This is the one command that matters, and it
needs nothing but the bundle:

```powershell
git checkout v0.1-bluedot
uv sync --extra dev --extra torch
uv run csf state-audit replay-bundle --bundle results/public/bluedot-v0.1
```

It verifies all 21 checksums, then recomputes all 16 condition summaries and both primary
comparisons from the bundle's own forecasts and outcomes and compares them to the stored analysis.
Expect `valid: true`, `checksums_valid: true`, zero failures, 10,000 resamples, seed `1394099119`,
and `analysis_hash` `sha256:46e88d3a629c78009787f88fadc5c821e8c318e3ee301923a236171cde1ece4d`. It
reads nothing outside the bundle, and it goes through the same function the run directory uses, so
the published artifact cannot pass a weaker check than the run it came from. It hashes the outcomes
file itself rather than trusting a hash recorded beside it, so an edited outcome fails.

The checksum file is in the format the ordinary tool accepts, so the bundle can be checked with no
Python at all:

```powershell
cd results/public/bluedot-v0.1; sha256sum -c CHECKSUMS.sha256
```

**Or replay the run directory,** if you have the full artifacts rather than the bundle:

```powershell
uv run csf state-audit replay-analysis --run-id bluedot-final-test
```

**Check the frozen inputs.** Each recomputes the artifact's own content hash before checking
anything else, so an edited manifest fails to load rather than verifying.

```powershell
uv run csf prompts verify --manifest-id bluedot_state_dependence_v1
uv run csf directions verify-family --manifest-id bluedot_state_dependence_directions_v1
uv run csf calibration verify-plan --plan-id bluedot_state_dependence_calibration_v1
```

**Check a run end to end.** `csf state-audit verify-run` recomputes, from files on disk: the run
manifest's own content hash, every artifact hash, every observation's target from its own logits,
the reference norm from the recorded clean state norms, and the single global alpha per grid point.
For a calibration run it re-derives the decision and refuses a selected ratio that is not the
smallest passing one.

```powershell
uv run csf state-audit verify-run --run-id bluedot-calibration-layer13
uv run csf state-audit verify-run --run-id bluedot-training
uv run csf state-audit verify-commitments --run-id bluedot-final-test
```

**Check the environment and the code.**

```powershell
uv run csf doctor
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

## Reproducing the study

Start from the tagged release, not from `main`:

```powershell
git checkout v0.1-bluedot
uv sync --extra dev --extra torch
```

`v0.1-bluedot` marks the code that produced the published result. The final test itself was resolved
at commit `aac5532` with a clean working tree, recorded inside the hashed resolution manifest.

About 90 minutes of CPU forward time end to end: calibration 42 minutes, training 26, resolution
14. Each stage refuses to start if the artifact it depends on does not verify, and a completed run
at the same run id is refused rather than overwritten.

```powershell
# 1. Prepare the task. Needs network access; everything after this is offline.
uv run csf data prepare --config configs/tasks/arc_mcq.yaml

# 2. Freeze the prompt split. No model, no forward pass.
uv run csf prompts manifest --config configs/prompts/bluedot_state_dependence.yaml

# 3. Build the direction family. Reads the pinned output embedding and nothing else.
uv run csf directions build-family --config configs/directions/bluedot_state_dependence.yaml

# 4. Freeze the calibration plan. No model.
uv run csf calibration plan --config configs/calibration/bluedot_state_dependence.yaml

# 5. Engineering smoke. 8 prompts, 144 forwards, at a ratio fixed in advance.
uv run csf state-audit smoke --config configs/state_audit/bluedot_smoke.yaml --run-id bluedot-smoke-layer13

# 6. Calibration. 32 prompts, 2,624 forwards. Selects layer, ratio, and alpha.
uv run csf state-audit calibrate --config configs/state_audit/bluedot_calibration_layer13.yaml --run-id bluedot-calibration-layer13

# 7. The fixed intervention projection. Loads no model weights.
uv run csf state-audit projection --config configs/state_audit/bluedot_training.yaml --projection-id bluedot_state_dependence_projection_v1

# 8. Training. 96 prompts, 1,728 forwards, at the strength calibration chose.
uv run csf state-audit train --config configs/state_audit/bluedot_training.yaml --run-id bluedot-training

# 9. Final-test clean stage. 32 clean forwards, zero interventions.
uv run csf state-audit final-test-clean --config configs/state_audit/bluedot_final_test_clean.yaml --run-id bluedot-final-test

# 10. Fit and commit. Loads no model. Seals 512 forecasts before any outcome exists.
uv run csf state-audit commit-forecasts `
  --training-config configs/state_audit/bluedot_training.yaml --training-run-id bluedot-training `
  --final-test-config configs/state_audit/bluedot_final_test_clean.yaml --final-test-run-id bluedot-final-test `
  --projection-id bluedot_state_dependence_projection_v1

# 11. Final-test resolution. Irreversible. 544 intervened forwards, no clean forward.
uv run csf state-audit resolve-final-test `
  --config configs/state_audit/bluedot_final_test_clean.yaml --run-id bluedot-final-test `
  --layer 13 --norm-ratio 0.02 --global-alpha 106.87158268272867 `
  --yes-i-understand-this-is-irreversible

# 12. Analysis. Model-free. Nothing is fitted, refitted, tuned, or dropped.
uv run csf state-audit analyze-final-test --run-id bluedot-final-test --training-run-id bluedot-training

# 13. Publish the model-free replay bundle. Loads no model. Copies only allowlisted artifacts.
uv run csf state-audit publish-bundle `
  --final-test-run-id bluedot-final-test --training-run-id bluedot-training `
  --calibration-run-id bluedot-calibration-layer13
```

Step 11 refuses a dirty working tree, a setting that differs from the calibration decision, a
commitment count other than 512, any pre-existing reveal, and any pre-existing outcome. On this
repository it will refuse outright, because the final test is already resolved.

## Artifact locations

### Published, tracked, and sufficient on their own

| Path | What it is |
| --- | --- |
| `results/public/bluedot-v0.1/` | The model-free replay bundle: 21 files, checksummed, replays standalone |
| `protocol/` | The five artifacts that fixed the design, recovered byte for byte with commit provenance |
| `paper/` | The workshop paper, bibliography, and publication figures |
| `REPORT.md` | The technical report |

The bundle deliberately excludes model weights, the layer-13 residual-stream array and its index,
the pre-reveal salt directory, local run logs, and unrelated runs. `bundle_manifest.json` lists each
exclusion and its reason. The post-reveal salts are published inside `selection_reveals.jsonl`,
because a commitment hash cannot be checked without them.

### Full run directories

Run artifacts are under `results/runs/`, which is git-ignored because it holds large numerical
outputs and private salts. Everything needed to check a claim is there.

| Artifact | Path under `results/runs/bluedot-final-test/` |
| --- | --- |
| The analysis of record | `state_audit_analysis.json` |
| Per-method, per-condition summaries | `state_audit_method_summary.json` |
| Prompt-level scores | `state_audit_prompt_scores.jsonl` |
| Pair-level scores | `state_audit_pair_scores.jsonl` |
| Resolution manifest | `state_audit_resolution.json` |
| Raw observations | `state_audit_observations.jsonl` |
| Sealed forecasts and commitments | `forecasts.jsonl`, `forecast_commitments.jsonl` |
| No-selection reveals | `selection_reveals.jsonl` |
| Wrong-state pairing and derangements | `state_audit_wrong_state_pairing.json` |
| Figures | `figures/` (also copied to `paper/figures/`) |

Other runs: `results/runs/bluedot-calibration-layer13/` holds the calibration decision and its five
ratio summaries; `results/runs/bluedot-training/` holds the fitted predictors, the transform-fit
records, and the grouped-CV results.

Tracked frozen inputs, small enough to live in version control, because a split that exists on one
machine is not a split:

```text
protocol/preregistration_state_dependence.md   frozen design, committed once, never modified
protocol/execution_decision_tree.md            the gate taken at each branch point
protocol/bluedot_state_dependence_v1.json      the 168-prompt split
protocol/bluedot_state_dependence_directions_v1.json
protocol/bluedot_state_dependence_calibration_v1.json
data/manifests/arc_mcq.json                    task manifest
```

The three JSON artifacts also live under `data/`, which is where the code loads them from.
`protocol/` is a second copy so the protocol reads as one directory;
`tests/test_protocol_copies.py` fails if the copies ever stop being byte-identical.

## Repository structure

```text
configs/
  models/         pinned model configs, revisions as commit shas
  tasks/          dataset and prompt-wrapper definitions
  prompts/        frozen prompt-split config
  directions/     direction-family construction config
  calibration/    frozen calibration plan config
  state_audit/    smoke, calibration, training, and final-test run configs
  experiments/    the offline fixture pipeline
  interventions/  intervention grids
data/             tracked manifests, plans, and splits; processed data is regenerated
src/causal_self_forecasting/
  schemas.py           every record that reaches disk, self-verifying
  state_audit_target.py  the study target, as pure functions
  models/         loading, answer scoring, hook-owned capture
  interventions/  tensor operations, direction store, direction family
  tasks/          dataset loading, prompt rendering, the frozen split
  trials/         trial generation, candidates, commitment, resolution
  calibration/    strength rule, pass conditions, layer state machine
  state_audit/    features, projection, fitting, matching, run execution,
                  verification, resolution, analysis
  forecasting/    the model-free baselines
  scoring/        metrics, including the paired grouped bootstrap
tests/            unit, integration, and the anti-fabrication guard
protocol/         the five frozen design artifacts, with commit provenance
paper/            workshop paper, bibliography, and publication figures
results/runs/     run artifacts (git-ignored)
results/public/   the model-free replay bundle (tracked)
artifacts/        direction vectors and fixtures (git-ignored, regenerable)
```

## Reproducibility guarantees

Every artifact recomputes its own content hash when it loads, so an edited file fails to parse
rather than quietly verifying. Every run manifest cites the hashes of everything upstream of it,
and the chain is checkable end to end without a GPU:

```text
task manifest -> prompt manifest -> direction family -> calibration plan -> run manifest
```

Determinism comes from one master seed, `20260727`, threaded through `derive_seed` with a
namespaced label per purpose, so changing one seeded step cannot shift another. Commitment salts are
the one deliberate exception: they come from the OS CSPRNG and live in a private directory kept out
of version control and out of the bundle. They are secret only until resolution. Each reveal record
publishes its `salt_hex`, because a commitment hash cannot be checked without it; what the protocol
protects is the ordering, not permanent secrecy.

Integrity rules enforced by tests rather than by good intentions:

* No invented results, sample counts, intervals, or hashes. Planned values are labeled planned.
* `tests/test_no_fake_results.py` fails if a public export exists that did not verify.
* `PublicDashboardRecord` cannot be constructed for an unverified run, and `MetricValue` cannot be
  constructed without a sample count and an interval.
* `scientific_result` is a typed `Literal[False]` on every engineering record, so a record claiming
  otherwise cannot be constructed at all.
* A `MethodConditionSummary` carrying a Brier score computed over fewer than 20 flips cannot be
  constructed, and a `PairedComparison` refuses a `supported` flag not implied by its own interval.
* Model revisions cannot be `main`; `ModelSpec` rejects moving pointers.
* Salts, selection seeds, and private payloads never enter version control, and CI greps the
  tracked file list to confirm it.

## Offline fixture pipeline

The original CSF-Bench trial and commitment loop runs end to end on the fixture model, no weights
downloaded:

```powershell
uv run csf directions synthetic --config configs/experiments/smoke.yaml
uv run csf interventions validate --config configs/experiments/smoke.yaml
uv run csf trials generate --config configs/experiments/smoke.yaml --max-trials 8
uv run csf trials resolve --run-id <RUN_ID> --ground-truth
```

There is no CLI command that commits a forecast in that loop; `commit_forecasts` is reachable from
Python only. The full generate, resolve, fit, commit, resolve, score cycle is exercised by
`tests/integration/test_resolve.py::test_full_pipeline_generate_resolve_fit_commit_score`, which is
the working example to copy.

## Safety

This study fine-tunes nothing, trains nothing, and uses no model organism. It adds a fixed vector
to a residual stream and reads four logits. The harm ceiling is a wrong letter on a multiple-choice
question.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the software and the null result. Please do not
cite it as evidence that models lack privileged access to their own states, which is a much larger
claim than this experiment can carry.

## License

MIT. See [`LICENSE`](LICENSE). Gemma weights are not redistributed here and remain under Google's
license.
