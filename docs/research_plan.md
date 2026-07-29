# Research plan

> **Scope note, 2026-07-28.** This document describes the broader CSF-Bench study, whose
> preregistration is `docs/preregistration.md`. It is **not** the active path. The active work is
> the BlueDot state-dependence arm in `docs/bluedot/preregistration_state_dependence.md`, which
> uses one clean pinned model, no adapter, and ridge regressions only. Everything in the phase
> table below from Phase 5 onward is deferred; the current view of that is
> `docs/deferred_work.md`.

## The problem

A model can produce a persuasive explanation of its own answer without that explanation
tracking the computation that produced it. Explanations are cheap to generate and hard to
grade, because there is usually no ground truth to grade them against.

Even a report that correlates with something internal may be riding on prompt cues, memorized
task structure, intervention labels, output confidence, or evaluator-specific patterns rather
than on any knowledge of the mechanism.

## The change

Do not ask the model why it answered. Require a prediction about what an intervention will do,
before the intervention is selected.

This creates ground truth. The forecast is recorded and committed. Then a candidate is chosen,
applied, and measured. The prediction was either right or wrong, and no interpretation of the
model's prose is involved.

The question becomes checkable:

> Does the report track causally relevant information, or can it be reproduced without access
> to the model's actual state?

## Scope of v0.1

* One model: Gemma 3 1B instruction-tuned, revision pinned.
* One benign model organism: a LoRA adapter with a deployment-conditioned answer-position
  preference.
* Real four-choice reasoning items (ARC-Challenge, selected MMLU subjects).
* Two training mechanisms: residual addition, projection ablation.
* One held-out mechanism: activation patching.
* Six forecasting methods, from a constant baseline to a state-conditioned MLP.
* The same-prompt state swap as the centerpiece control.

Explicitly out of scope for v0.1: frontier API models, multiple model families, full SAE
analysis, agentic behavior, and a conference-ready paper.

## Why the controls carry the argument

The headline comparison is `state_mlp` against `prompt_tfidf` on character-identical prompts
produced by internally different models. A prompt-only baseline cannot distinguish those two
cases even in principle. If it matches the state-conditioned model anyway, the state is not
doing the work, and that is the finding.

The norm-matched random direction separates "a large perturbation happened" from "this
particular direction matters". The held-out mechanism separates learning a causal structure
from learning one operation. The state swaps separate using the state from using this model's
state.

Without these, a positive result would be uninterpretable. With them, a negative result is
still informative.

## Phases

| Phase | Deliverable |
| --- | --- |
| 0 | Scientific contract: research plan, preregistration, claim boundaries, split policy |
| 1 | Repository scaffold, schemas, hashing, CLI, fixture model, tests |
| 2 | Real-model harness: loading, answer scoring, state capture |
| 3 | Intervention system with controls |
| 4 | Trial and commitment engine |
| 5 | Benign model organism |
| 6 | Direction discovery with causal validation |
| 7 | Baselines |
| 8 | State-conditioned forecaster and state-swap controls |
| 9 | Held-out mechanism transfer |
| 10 | Dashboard |
| 11 | State-conditioned verbal reporter |
| 12 | SAE extension (does not block v0.1) |
| 13 | Gemma 3 4B replication |

Current status is tracked in `build_plan.md`. Phases 0 through 4 are complete and Phase 7's two
model-free baselines exist. Nothing else past Phase 4 has started, and Phases 5 through 13 are
deferred for the reasons in `docs/deferred_work.md`.

## Compute

The development machine has no CUDA device. The pipeline and the test suite run on CPU against
a tiny fixture model, which keeps Phases 1 through 4 fully testable locally. Phase 5 (LoRA
training) and the full trial sweeps are expected to need a rented GPU. All device selection is
resolved in one module, so moving to a GPU is a config change.

The decision about renting will be made against a measured CPU cost from the first real Gemma
run, not against an estimate.

## Success

The project does not need a positive result to be successful. A strong negative finding with
good controls is more valuable than a weak positive claim, and the preregistration fixes the
decision rule in advance so that the outcome cannot be renegotiated after the fact.

What would make this project a failure: a result that is reported without the controls that
make it interpretable, or a claim that outruns the design. `claim_boundaries.md` exists to
prevent that.
