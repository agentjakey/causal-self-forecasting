# Compute decision record

All timing here derives from one measured value: the median forward time of Gemma 3 1B on
this machine, read from a real systems benchmark artifact. No number below is invented, and
none is a scientific result.

## Source measurement

From `results/runs/benchmark-gemma3_1b_it-20260718T040531Z/benchmark.json`, a systems
benchmark classified `systems_benchmark`, `scientific_result: false`.

| Quantity | Value |
| --- | --- |
| Model | google/gemma-3-1b-it, revision dcc83ea841ab6100d6b47a070329e1ba4cf78752 |
| Device / dtype | cpu / float32 |
| Median forward time | 0.474158 s |
| p90 forward time | 0.503748 s |
| Forward min / max | 0.467777 s / 0.503748 s |
| Timed sample size | 3 forwards, one 53-token prompt |
| Model load time (one-time) | 48.116 s |
| Capture overhead | 0.007180 s (about 1.5 percent of a forward) |

The single measured forward time is called `T = 0.474158 s` below.

## Method

Every estimate uses the same explicit arithmetic:

```text
estimated_forward_count =
  prompts
  x model_states
  x candidates
  x forwards_per_candidate
  x repeats

estimated_cpu_seconds =
  estimated_forward_count
  x measured_median_forward_seconds
```

Clean-baseline forwards done during trial generation are counted separately where they apply,
because they are a real cost but are not part of the intervention count in the formula.

Model load time is excluded from every estimate. It is a fixed one-time cost of about 48 s
per distinct model loaded, and folding it into a per-forward figure would distort the scaling.

## The four estimates

### 1. Twenty-item clean benchmark

Clean scoring only, no interventions.

```text
forward_count = 20 x 1 x 1 x 1 x 1 = 20
cpu_seconds   = 20 x 0.474158       = 9.5 s
```

Estimated compute: about 9.5 s, plus one 48 s model load. Under a minute of wall time.

Classification: **practical on CPU.**

### 2. Twenty-item, four-candidate intervention smoke run

Twenty prompts, one model state, the four default candidates each applied once.

```text
forward_count = 20 x 1 x 4 x 1 x 1 = 80
cpu_seconds   = 80 x 0.474158       = 37.9 s
```

Plus 20 clean-baseline forwards at generation (about 9.5 s). Total pipeline about 47 s of
compute, plus one model load.

Classification: **practical on CPU.**

### 3. One hundred prompts, clean and adapted states

One hundred prompts, two model states (clean and adapted), four candidates each.

```text
forward_count = 100 x 2 x 4 x 1 x 1 = 800
cpu_seconds   = 800 x 0.474158       = 379.3 s  (6.3 min)
```

Plus 200 clean-baseline forwards at generation (about 94.8 s). Total pipeline about 7.9 min
of compute. Two distinct models are loaded, so about 96 s of load time.

Classification: **practical on CPU.**

### 4. Planned MVP sweep

From `docs/preregistration.md` section 9: 2000 prompts, two model states, four candidates,
one forward per candidate.

```text
forward_count = 2000 x 2 x 4 x 1 x 1 = 16000
cpu_seconds   = 16000 x 0.474158      = 7586.5 s  (2.107 h)
```

This matches the `compute_estimate` written into the benchmark artifact exactly (16000
forwards, 2.107 CPU hours), which is a check on the arithmetic rather than an independent
result.

Plus 4000 clean-baseline forwards at generation (about 0.53 h). Total pipeline about 2.6 CPU
hours for the intervention sweep with baselines. The constant and prompt-only baselines add no
forward passes; they run on the recorded observations.

Classification: **practical on CPU**, at the 12-hour threshold used in the artifact. GPU
rental is not required for the 1B MVP sweep, though it would shorten iteration.

### 5. BlueDot state-dependence arm

Added 2026-07-27. Design frozen in `docs/bluedot/preregistration_state_dependence.md`.

This arm is not a scaled-down version of the MVP sweep; it has a different shape. One clean
capture forward per prompt, then one forward per candidate. Candidate counts differ by role: 17
at smoke, training, and final test (8 directions x 2 signs, plus a no-op), and 81 at calibration
(5 ratios x 16 signed directions, plus a no-op), because only calibration sweeps the ratio grid.

```text
smoke        =  8 x (1 + 17) =   144
calibration  = 32 x (1 + 81) = 2,624
training     = 96 x (1 + 17) = 1,728
final test   = 32 x (1 + 17) =   576
total                        = 5,072

cpu_seconds  = 5,072 x 0.474158 = 2,404.9 s = 40.1 min
```

**Planned forward time is approximately 40 minutes**, before model loading, serialization, and
verification.

The preregistered layer-20 fallback repeats calibration only, at the same layer count and the
same grid:

```text
2,624 x 0.474158 = 1,244.2 s = 20.7 min
```

**approximately 21 additional minutes**, for a worst case of about 61 minutes.

Applying the long-prompt band from the cross-cutting factors below (1.5 to 2 times `T`) gives
roughly 40 to 80 minutes without the fallback and 61 to 122 minutes with it.

Nothing else in the arm costs a forward pass. The three ridge models, the grouped
cross-validation over the 96 training prompts, the matched wrong-state control, the ten
permutation controls, and the 10,000-resample paired bootstrap all run on recorded observations
and stored states. Storage is under 1 MB per captured layer for all 168 prompts.

Classification: **practical on CPU**, by a wide margin. **No GPU rental and no compute grant is
needed for this arm.**

This estimate supersedes the compute figure in `docs/bluedot/current_state_audit.md` section 15,
which applied the full five-ratio grid to every role and therefore overstated the cost by about
2.7 times.

## Cross-cutting factors

**Uncertainty from a small timing sample.** The median rests on three timed forwards of a
single 53-token prompt. The observed spread was narrow (min 0.468 s, max 0.504 s, about 4
percent around the median), but that spread does not capture variation across prompt lengths,
because only one prompt was timed. ARC prompts vary in length, and forward time on this
architecture grows with token count, so longer prompts will cost more than T. Treat every
estimate as a lower bound with a plausible band up to roughly 1.5 to 2 times for long-prompt
heavy runs. The fix is cheap: time several prompts of varied length before committing to the
full sweep.

**Activation capture adds measurable but small overhead.** Capture cost was 0.007180 s
against a 0.474158 s forward, about 1.5 percent. It does not change any classification.

**Model loading is excluded.** About 48 s per distinct model. Runs that swap between clean and
adapted models pay it twice.

**Storage is not likely to become material at 1B.** A captured state is 1152 float32 values,
about 4.6 KB. The MVP sweep at a few capture layers is on the order of tens of megabytes of
state shards. Observations and scores are small JSON. This grows with prompts times layers and
would need reconsidering for many-layer captures or larger models, but it is not a constraint
for the 1B MVP.

## LoRA training is the open question, and it is deferred

Deferred as of 2026-07-27 for the BlueDot state-dependence arm, which uses no model organism and
therefore trains no adapter. The analysis below stands unchanged for the original CSF-Bench
study, where the question is still open.

The benchmark artifact reports `lora_estimated_cpu_hours` of about 2.37 h and marks LoRA
training practical on CPU. That figure approximates a training step as three forward passes,
which is a weak model for CPU training: it ignores optimizer state, the absence of batching
efficiency, and memory pressure, all of which tend to make CPU training disproportionately
slow relative to inference. The forward-equivalent approximation almost certainly understates
the real cost.

Classification for LoRA training on CPU: **insufficient evidence.** The inference benchmark
does not measure training, and the arithmetic extrapolation is not trustworthy for it. A short
timed LoRA micro-run (a handful of steps) is needed before treating CPU LoRA training as
practical. This is out of scope now and no such run has been done.

## Decision summary

| Proposed run | Classification |
| --- | --- |
| 20-item clean benchmark | practical on CPU |
| 20-item four-candidate smoke run | practical on CPU |
| 100-prompt clean and adapted | practical on CPU |
| MVP sweep (2000 prompts) | practical on CPU |
| **BlueDot state-dependence arm (168 prompts, 5,072 forwards)** | **practical on CPU, about 40 min, no GPU needed** |
| **BlueDot arm with the layer-20 fallback (7,696 forwards)** | **practical on CPU, about 61 min, no GPU needed** |
| LoRA training on CPU | insufficient evidence, and deferred |

The clean and intervention validation work is comfortably within CPU reach on this machine, and
the BlueDot arm is the cheapest run this record has estimated. The one place the evidence runs
out is LoRA training, which is what the model organism will need. That is the point where a
timed micro-run, and then possibly GPU rental, should be considered, and it is deferred out of
the active path. The compute decision itself remains the maintainer's; this record only lays out
what the one measured forward time implies.
