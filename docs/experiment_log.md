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

## Next entry

The next scientific step needs a validated direction and a model organism, both of which are
out of scope for now. The immediate open work is the remaining baselines (linear state probe,
state MLP, gradient) and the direction-estimation pipeline. No CSF-Bench scientific result
exists yet, and none should be reported until a validated direction and a real comparison
exist.
