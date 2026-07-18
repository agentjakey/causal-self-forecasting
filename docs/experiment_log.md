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

**Phase 2 remains open.** The definition of done is a real pinned Gemma run, and one has not
happened.

## Next entry

The next entry will be the first real Gemma 3 1B benchmark, once the file host is reachable:

```powershell
uv run csf benchmark `
  --model-config configs/models/gemma3_1b_it.yaml `
  --task-config configs/tasks/arc_mcq.yaml `
  --split test `
  --max-items 1 `
  --warmup-runs 1 `
  --timed-runs 3
```

That run measures clean accuracy on a small ARC sample and the CPU cost per forward pass,
which is what the GPU decision should be made on. Phase 2 stays open until it exists, and no
compute estimate for Gemma should be quoted before then.
