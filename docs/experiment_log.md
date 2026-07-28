# Experiment log

Append-only. One entry per run that produced artifacts. Entries cite a run id and its
manifest; numbers are copied from verified artifacts, never typed from memory.

## No CSF-Bench result exists yet

As of 2026-07-16, no causal self-forecasting experiment has been run.

There is no model organism, no estimated direction, no trained forecaster, and no verified
export. Anything in this repository that resembles a measurement is a harness control, a smoke
run on a randomly initialized fixture model, or a systems benchmark.

A **systems benchmark** measures whether a model loads on this machine and how fast it runs.
It is not a scientific result and cannot become one: its artifacts carry
`"scientific_result": false` as a typed literal, they contain no forecasts or commitments, and
the public exporter rejects them.

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

## 2026-07-16: systems benchmark added (not an experiment)

Added `csf benchmark`, the smallest real-model validation step needed before deciding whether
to rent GPU compute. It loads the pinned model, scores a few ARC items from their answer-token
logits, times forward passes, and verifies the capture path on real weights.

It is a systems and clean-model sanity benchmark. It is not a CSF-Bench result.

**Two real bugs this caught, both in access classification.**

The first: `huggingface_hub` 1.x moved from `requests` to `httpx`, and `httpx.HTTPError` is
not an `OSError` subclass. The access check caught `(HfHubHTTPError, OSError)`, so a dropped
connection escaped as a raw traceback rather than being reported as a network failure. That
is the failure most likely to occur during a two-gigabyte download, and it was the one the
classifier could not name. Now pinned by `test_httpx_transport_failure_is_classified`.

The second: because the centralized loader wraps every exception as `ModelLoadError`, a
connection lost mid-download would have been reported as "the model failed to load", sending a
reader to debug their weights instead of retrying. `_is_network_error` now walks the exception
chain, pinned by `test_a_dropped_download_reads_as_a_network_failure`.

**An observed environment fact.** The hub connection from this machine is intermittent:
`model_info` for the pinned revision failed with `ConnectError [WinError 10054]` and then
succeeded on the immediately following attempt with no other change. The metadata probe now
retries transport failures three times with linear backoff. Access answers (gated, missing
revision) are not retried, because those are answers rather than blips.

**Verified before the first run.** Authentication was present, the account had accepted the
Gemma conditions, and the pinned revision `dcc83ea841ab6100d6b47a070329e1ba4cf78752` resolved
exactly. The hub reports `model.safetensors` at 1,999,811,208 bytes. Free disk was about
517 GB.

### The real Gemma benchmark was attempted and did not complete

**No Gemma measurement exists.** There is no accuracy number, no forward-pass timing, and no
compute estimate for Gemma, because the weights could not be downloaded from this machine.

What was established, and what was not:

| Precondition | State |
| --- | --- |
| Hugging Face authentication | present |
| Gemma license accepted on the account | yes |
| Pinned revision resolves | yes, exactly `dcc83ea841ab...` |
| Disk space | about 517 GB free, against 2.0 GB needed |
| Weights downloaded | **no** |

The blocker is the network, isolated by direct measurement rather than inferred:

* `https://huggingface.co/api/...` succeeded on 4 of 5 attempts.
* `https://huggingface.co/google/gemma-3-1b-it/resolve/.../model.safetensors`, the path that
  serves the weights, failed on every attempt with `curl` exit 35 (SSL connect error),
  transferring 0 bytes.
* Unrelated hosts were reachable but slow. So this is not a total outage; the large-file
  path specifically is failing from this machine.

The weights blob sat at 0 bytes for roughly nine minutes with the process alive. Note for
whoever retries: when the file host is unreachable, `huggingface_hub` retries internally with
backoff, so `csf benchmark` looks like it is hanging rather than failing. The metadata probe
this project added retries only three times and then reports `network_failure`; the download
itself is the library's to manage.

Everything except the download was exercised against the real Gemma config and its real
pinned revision. `csf benchmark --offline` correctly reported `offline_cache_miss` naming the
cache path and the remedy, which confirms the access classification works against Gemma
without downloading it.

The download later succeeded on a retry from the same machine, so the entry below supersedes
this one. The network intermittency was real, not a code fault.

## 2026-07-18: real Gemma 3 1B systems benchmark (systems measurement, not a scientific result)

The pinned Gemma 3 1B weights downloaded and the benchmark completed. Values below are read
directly from
`results/runs/benchmark-gemma3_1b_it-20260718T040531Z/benchmark.json`, a `systems_benchmark`
artifact with `scientific_result: false`. They measure this machine, not the model's ability.

**Model.** google/gemma-3-1b-it, revision `dcc83ea841ab6100d6b47a070329e1ba4cf78752`,
tokenizer at the same revision, device cpu, dtype float32, 26 layers, hidden dim 1152, access
status `remote_download`.

**Scoring.** Format `raw_completion_next_token_after_answer_colon`, the same raw completion
format the trial pipeline uses. Answer labels A, B, C, D resolved to single tokens
562, 603, 565, 622.

**Timing on this machine (CPU, machine-specific).**

| Quantity | Value |
| --- | --- |
| Model load (one-time) | 48.116 s |
| Median forward | 0.474158 s |
| p90 forward | 0.503748 s |
| Forward min / max | 0.467777 s / 0.503748 s |
| Tokenization total | 0.001344 s |
| Representative prompt | 53 tokens |
| Prefill throughput | 109.98 tokens/s |

**Clean accuracy.** 1 of 1 items correct on the sampled ARC test item. This is a scoring smoke
check on a single item and is not scientifically informative; it says the scoring path works,
nothing about capability.

**Memory.** Null. The Windows PSAPI `GetProcessMemoryInfo` call returned failure, so the
benchmark recorded null with that reason rather than inventing a number.

**Capture verification.** Layer 13, captured shape [1152], dtype float32, hook fired,
`capture_point_verified: true`, `max_abs_patch_error: 0.0`. The hook-owned capture path reads
back exactly the point it patched, on real Gemma weights, not only on the fixture.

**Compute estimate.** From the measured median forward, the MVP sweep (2000 prompts, two model
states, four candidates) is 16000 forwards and about 2.107 CPU hours, classified practical on
CPU. Full arithmetic and the LoRA caveat are in `docs/compute_decision.md`.

**Provenance.** benchmark.json hash `sha256:e9ac2c28...`, model config hash `sha256:583ac1fa...`,
task config hash `sha256:97c886ec...`, task manifest hash `sha256:ca8773af...`.

**Phase 2 is now done.** A real pinned Gemma run has succeeded.

## 2026-07-19: real Gemma intervention harness validation (systems measurement, not scientific)

The smallest real-model intervention validation the benchmark timing justifies. It runs the
intervention and resolution paths on real Gemma weights at layer 13 with a synthetic,
unvalidated direction (`gemma_synthetic`, a seeded random unit vector). Run
`gemma-harness`, resolved in ground-truth mode. No model organism, no estimated direction, no
forecaster. Nothing here is a CSF-Bench scientific result.

`csf interventions validate` on the Gemma config passed every required control on real weights:

| Control | Result |
| --- | --- |
| no-op equality | max abs logit diff 0.0, residual delta norm 0.0 |
| zero-strength equality | max abs logit diff 0.0 |
| rerun determinism | max abs logit diff 0.0, max abs state diff 0.0 |
| shape validation | all invalid operands rejected |
| hook applies intervention | residual delta norm 0.99999, expected 1.0 |

`csf trials resolve --ground-truth` applied all four candidates to one ARC item and recorded
four observations, zero failures. Measured delta margins (clean margin was 8.5148):

| Role | Mechanism | delta margin | answer flip |
| --- | --- | --- | --- |
| noop_control | noop | 0.000000 | no |
| direction_positive | residual_add +4.0 | -0.005844 | no |
| direction_negative | residual_add -4.0 | +0.005674 | no |
| random_control | random_add +4.0 | +0.010585 | no |

Reading these correctly: the no-op reproduces the clean output exactly, which is the point of
the control. Positive and negative steering move the margin in opposite directions, coherently.
The magnitudes are tiny because a synthetic random direction at layer 13 barely perturbs a very
confident margin, and because the direction is not a bias direction and has passed no causal
validation. These numbers validate the harness end to end on real weights; they say nothing
about the model, and they are marked non-scientific in the resolution manifest.

## 2026-07-27: repository audit (documentation, no run)

A read-first audit of the checkout at commit `86fb2d3`, written to
`docs/bluedot/current_state_audit.md`. No model was loaded and no run was executed.

Verified from the checkout rather than carried forward on trust: Python 3.12.13, torch
2.13.0+cpu with CUDA unavailable, transformers 5.14.0, uv 0.11.13, the pinned Gemma revision,
the answer token ids, and layer-13 capture at dim 1152. Offline suite: **273 passed, 1 skipped**;
ruff, ruff format, and pyright clean; `git diff --check` clean.

The audit recorded eleven code-versus-documentation discrepancies and the protocol gaps that the
state-dependence arm has to close, the largest being that nothing currently prevents outcomes
from existing before a forecast is committed.

One arithmetic error in that audit has since been corrected: it applied the full five-ratio
calibration grid to every prompt role and so overstated the arm's compute by about 2.7 times.
The corrected figure is in `docs/compute_decision.md` section 5.

## 2026-07-27: BlueDot state-dependence preregistration and scope amendment (not a result)

**This entry records a preregistration and a scope amendment. It is not a measured result.**
No data was prepared, no model was loaded, no direction was built, and no calibration, training,
or final-test outcome was generated.

Written:

* `docs/bluedot/preregistration_state_dependence.md`, version 1.0, dated 2026-07-27. It freezes
  the research question, the model and revision, the prompt roles and counts, the state layer and
  its single allowed fallback, a new study target `delta_clean_top_margin` that does not redefine
  the existing `delta_margin`, the eight-direction bank, the global calibrated intervention
  magnitude, the three ridge methods, the wrong-state control, the commitment protocol, the
  analysis, and a set of explicitly labeled predictions.
* `docs/bluedot/execution_decision_tree.md`, the execution order with a written gate at every
  branch point and six named stop codes.

The new document supersedes `docs/preregistration.md` **for this arm only**. The original was
not rewritten; it received a scope note and a dated amendment-log entry. No hypothesis, outcome,
comparison, decision rule, or planned count in the original was altered, and no data had been
collected under it.

Scope moved out of the active path for this arm, and deferred rather than cancelled: the model
organism, LoRA training, learned clean-versus-adapted directions, the evaluation-versus-deployment
hypothesis, the state MLP, the verbal reporter, SAE work, the Gemma 3 4B replication,
held-out-mechanism transfer, the dashboard, and GPU rental.

Planned compute for the arm, arithmetic on the measured median forward time of 0.474158 s from
the 2026-07-18 systems benchmark: 5,072 forwards, about 40 minutes, plus about 21 minutes if the
preregistered layer-20 fallback triggers. No GPU and no compute grant is needed.

The predictions recorded before calibration live in section 12 of the preregistration and are
deliberately not repeated here, so that this log cannot be misread as containing results.

Offline suite after the documentation changes: **273 passed, 1 skipped**; ruff, ruff format, and
pyright clean.

## 2026-07-27: BlueDot prompt manifest frozen (infrastructure and split freezing, not a result)

**This entry records infrastructure and a split freeze. It is not a scientific result.** No model
was loaded, no forward pass ran, no state was captured, and no intervention was applied. The
command that produced it reads prepared task files and writes one JSON artifact.

The ARC pool was re-prepared before this step and now holds 256 items, 256 groups, and 2,048
variants (`data/manifests/arc_mcq.json`).

Command:

```powershell
uv run csf prompts manifest --config configs/prompts/bluedot_state_dependence.yaml
```

Values below are read from the written artifact, not typed from memory.

| Field | Value |
| --- | --- |
| Manifest | `data/prompt_manifests/bluedot_state_dependence_v1.json` |
| Manifest hash | `sha256:bf351c9d73042fcb3d0cdcb18247ded3411413d73000e047f1ba6ba9ab5b25b9` |
| Task manifest hash | `sha256:51ef4ce76eb3dd5d8616ef09ad376c9bda859934edc19676485ea8cfb5edc983` |
| Items hash | `sha256:08581f187b6095b1bffba47d9f33cd07c085c9348a3544d17f49d31adfa17070` |
| Variants hash | `sha256:ccbae819f1b1e1d910b3d31b60b2ae40e379d90755e9d672c22a39b4523b8ba3` |
| Master seed | 20260727 |
| Canonical wrapper | `neutral_a` |
| Selection algorithm | `seeded_group_permutation_contiguous_roles` v1.0 |
| Eligible groups | 256 |
| Role counts | smoke 8, calibration 32, training 96, final_test 32 |
| Total prompts | 168 |

Checked against the written file: 168 assignments, 168 distinct variant ids, 168 distinct group
ids, 168 distinct item ids, every assignment on wrapper `neutral_a`, all four roles pairwise
disjoint by both group id and item id, and `selection_index` contiguous from 0.

Rerunning the command left the file byte-identical and reported `status: unchanged` rather than
rewriting it. `csf prompts verify --manifest-id bluedot_state_dependence_v1` reports `valid:
true` with no mismatches.

Selection is a pure function of the master seed and the group ids, computed through the existing
`derive_seed` with the label `bluedot.prompt_manifest`, over a lexicographically sorted pool. It
does not read model correctness, confidence, label logits, hidden states, intervention effects,
calibration outcomes, gold-label balance, question topic, or input file order. Each of the three
non-smoke roles drew prompts from all three of the task pipeline's `Split` values, which is what
a selection blind to those properties looks like.

The existing `Split` enum and `tasks/splitting.py` are unchanged. `PromptRole` is additive, and a
prompt now carries both.

Suite after this slice: **326 passed, 1 skipped** (273 before, 53 added). Ruff, ruff format,
pyright, and `csf doctor` clean.

## 2026-07-27: BlueDot direction family built (infrastructure, not a scientific result)

**Directions were constructed from model weights. No prompt was run. No state was captured. No
intervention was applied. No behavioral or scientific outcome was generated.** The only thing
read from the model was the output-embedding matrix.

**This is construction, not causal validation.** The eight directions are stimuli with a
recorded recipe. Nothing here shows that any of them is meaningful, load-bearing, or
bias-related, and every stored artifact carries `validated: false`.

Commands:

```powershell
uv run csf directions build-family --config configs/directions/bluedot_state_dependence.yaml
uv run csf directions verify-family --manifest-id bluedot_state_dependence_directions_v1
uv run csf directions verify-family --manifest-id bluedot_state_dependence_directions_v1 --regenerate
```

Values below are read from the written manifest, not typed from memory.

| Field | Value |
| --- | --- |
| Manifest | `data/direction_manifests/bluedot_state_dependence_directions_v1.json` |
| Family hash | `sha256:809fbb5b033da740a01574ad5a0ca48f34baca38eef1504a0d66bba8e2fb9138` |
| Model | `google/gemma-3-1b-it` at `dcc83ea841ab6100d6b47a070329e1ba4cf78752` |
| Tokenizer revision | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` |
| Output-embedding source | `get_output_embeddings`, shape 262144 x 1152 |
| Tied embeddings | true, observed from the tensors; the config also declares `tie_word_embeddings: true` |
| Hidden dimension | 1152 |
| Resolved answer token ids | A=562, B=603, C=565, D=622, resolved through `resolve_label_token_ids` and matched against the pinned expectations before building |
| Master seed | 20260727 |
| Derived control seed | 2936319247 |
| Answer-span effective rank | 3 |
| Directions | 8: four centered answer-token, four orthogonal random controls |
| Redraws | 0 |

Raw answer-direction norms, before unit normalization: A 0.9331798302046521,
B 0.7530634390447186, C 0.793201456422512, D 0.8334530774210125.

Numerical diagnostics: centered-direction sum residual 2.78e-17; answer-span orthonormality
error 1.73e-17; maximum norm error 2.16e-09; maximum answer-to-random absolute dot 2.82e-09;
maximum random-to-random off-diagonal absolute dot 2.30e-09. All are far inside the frozen
1e-5 tolerances for saved float32 vectors.

The maximum pairwise absolute cosine among the four answer directions is 0.4157. That is
expected and is not a defect: the four centered directions are linearly dependent by
construction, which is also why the answer span has rank 3 rather than 4.

Opaque direction ids, in manifest order: `bd1.1dd98ed52e4a9a42`, `bd1.26e4bcbe30ca5694`,
`bd1.3d3f6efa84923ded`, `bd1.969ccca09bfd9674`, `bd1.b438e9212bf46d7c`, `bd1.c17341363a8e3c8b`,
`bd1.cfe2d7ba9d21c997`, `bd1.f3aad413c49dfbad`. The ids are hash prefixes and the ordering is by
id, so neither the string nor the position says which family a direction came from. The mapping
to construction roles lives only in the manifest, which is private provenance.

Verification. Artifact-level (no model loaded): valid, 8 of 8 directions checked, no failures,
no container notes. Regeneration (pinned model reloaded, nothing written): valid, rebuilt family
hash identical to the manifest. Re-running the build reported `status: unchanged` and left the
manifest byte-identical with its mtime untouched; all eight `.npz` files were also unchanged.

The eight vector artifacts live in the git-ignored `artifacts/directions/`. They were not
force-added. The manifest is sufficient to regenerate and re-verify them from the pinned
revision, the config, the seed, and the algorithm versions it records.

Offline suite before loading any weights: **400 passed, 1 skipped** (326 before, 74 added).
Ruff, ruff format, pyright, and `csf doctor` clean.

## 2026-07-28: study target and calibration rules implemented (infrastructure, not a result)

**The target and the calibration rules were implemented. No model was loaded. No prompt was run.
No state norm was measured. No intervention was applied. No calibration result exists.**

Implemented:

* the study target `delta_clean_top_margin`, measured around the model's own clean preferred
  answer with `c_star` held fixed after the intervention. The benchmark's `delta_margin`, which
  measures the margin around the dataset-correct answer, is unchanged and keeps its own
  validator; the arm's observations live in a separate `StateAuditObservationRecord` whose
  validator recomputes the label, both margins, the delta, and the flip from the logits stored
  beside them;
* the global strength rule: one absolute alpha per layer and ratio,
  `alpha = ratio * median(clean state norm over the 32 calibration prompts)`. There is
  deliberately no prompt-specific strength function, and `check_global_alpha` refuses a set of
  observations that used more than one alpha at a grid point;
* the six frozen pass conditions, the layer-13-primary, layer-20-fallback state machine, and
  the records for plans, reference norms, ratio summaries, and decisions.

The plan was frozen. Values below are read from the written artifact.

| Field | Value |
| --- | --- |
| Plan | `data/calibration_plans/bluedot_state_dependence_calibration_v1.json` |
| Plan hash | `sha256:a212c6e80f96db55e1aeb0b1879fef441aa6d893a73b7fc43a94142913f97877` |
| Config hash | `sha256:9b777aa488d109bdca227f20a0e5ed4c664fb9a3a468a6019382df51f52f4954` |
| Prompt manifest | `bluedot_state_dependence_v1`, `sha256:bf351c9d73042fcb3d0cdcb18247ded3411413d73000e047f1ba6ba9ab5b25b9` |
| Direction family | `bluedot_state_dependence_directions_v1`, `sha256:809fbb5b033da740a01574ad5a0ca48f34baca38eef1504a0d66bba8e2fb9138` |
| Model | `google/gemma-3-1b-it` at `dcc83ea841ab6100d6b47a070329e1ba4cf78752` (read from the config; no weights loaded) |
| Target | `delta_clean_top_margin` |
| Layers | primary 13, fallback 20 |
| Ratios | 0.02, 0.05, 0.10, 0.20, 0.40 |
| Thresholds | large-effect fraction >= 0.15 at magnitude >= 0.10; median >= 0.05; 95th percentile <= 4.0 |
| No-op tolerance | 1.0e-3 |
| Median convention | `numpy.median`, even samples average the two central sorted values |
| Percentile convention | `numpy.quantile(method='linear')` |

Encoded forward counts: smoke 8 x 18 = 144; calibration 32 x 82 = 2,624 per layer; training
96 x 18 = 1,728; final test 32 x 18 = 576; primary total **5,072**; layer-20 fallback adds
**2,624** for 7,696. Only calibration sweeps the ratio grid, which is why it carries 82 forwards
per prompt and every other role carries 18. This supersedes the 13,776 figure in the first
version of `docs/bluedot/current_state_audit.md`.

Rerunning the plan command reported `status: unchanged` and left the file byte-identical with
its mtime untouched. `csf calibration verify-plan` reports `valid: true` with no failures.

**No calibration was run and no ratio has been selected.** The summarizer and the selector were
exercised only on synthetic fixture observations inside the test suite; those numbers were
constructed to test the code paths and are not measurements of anything.

Two bugs the tests caught during this slice. `check_global_alpha` keyed its comparison by prompt
id, so with 16 candidates per prompt a single tampered alpha was overwritten by its neighbours
and vanished; it now compares every observation. And planning transitively imported torch
through the direction-family module, which made a command that loads no model pay a
multi-second import; that import is now lazy.

Suite after this slice: **543 passed, 1 skipped** (400 before, 143 added). Ruff, ruff format,
pyright, and `csf doctor` clean.

## Next entry

The next steps are B2b, the fixed 16-dimensional intervention projection matrix, and the
commitment-protocol hardening. Both need no model. Running the real calibration sweep (B3b) is a
separate maintainer decision and is the first step in this arm that produces measured numbers.

No CSF-Bench scientific result exists yet, and none should be reported until a real comparison
has been run and verified.
