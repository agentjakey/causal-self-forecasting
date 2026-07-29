# Deferred work

One place for everything that is designed but deliberately **not** in the active path. Nothing
here is cancelled, and nothing here blocks the BlueDot state-dependence arm.

Why a single list: the deferred set used to be restated in the research plan, the build plan, the
preregistration, and the README, and four copies of a roadmap drift. The frozen documents keep
their own wording because they are frozen; this file is the current view.

Last updated 2026-07-28.

## What the active arm actually needs

The BlueDot state-dependence arm, preregistered in
`docs/bluedot/preregistration_state_dependence.md`, uses:

* one model, Gemma 3 1B instruction-tuned, at one pinned revision;
* one task, ARC-Challenge, on one neutral prompt wrapper;
* one frozen prompt split of 168 prompts;
* one constructed direction family of eight unit directions;
* one intervention mechanism, signed residual addition, at one calibrated global strength;
* ridge regressions only, and a matched wrong-state control.

It needs no GPU, no adapter, and no second model. Everything below is outside that set.

## Deferred, with the reason

| Item | Why it is deferred |
| --- | --- |
| Benign model organism (LoRA adapter) | The arm needs no planted behavior. Its target is defined against the model's own clean preferred answer, so there is nothing an organism would add. |
| LoRA training | Classified **insufficient evidence** on CPU in `docs/compute_decision.md`. A timed micro-run is needed before it is treated as practical, and that is a compute decision, not a code task. |
| Learned clean-versus-adapted behavioral direction | Requires the organism. The arm's directions are constructed with a recorded recipe instead, and are never described as discovered. |
| Direction discovery with causal validation | The arm uses stimuli, not discovered features. A discovered direction would need its own validation against matched controls before it could be called meaningful. |
| Evaluation-versus-deployment framing hypothesis (H6) | The arm uses one neutral wrapper. The `eval_*` and `deploy_*` wrappers stay in the task config and stay unused here. |
| State-conditioned MLP | The arm is linear by preregistration. An MLP is a researcher degree of freedom the frozen design does not allow. |
| Verbal reporter and soft-token reporter | A separate question from whether the state carries the information at all, which is what this arm asks first. |
| Held-out mechanism transfer (activation patching, H4) | The machinery stays in the code and costs nothing. Testing transfer needs a trained forecaster, which does not exist. |
| SAE extension | Never blocked v0.1 and does not block this arm. |
| Gemma 3 4B replication | Cross-scale generality cannot be claimed from one model, and cannot be claimed from two either. It is a later question. |
| Public dashboard | There is no verified export to publish. The export schema and the anti-fabrication guards exist and are tested; the front end does not. |
| GPU rental | Not needed. The whole arm is about 40 minutes of CPU forward time, 61 with the preregistered fallback. |

## What is kept in the code despite being deferred

Removing these would weaken the original design for no gain, and each is covered by tests:

* `Mechanism.ACTIVATION_PATCH` and the held-out-mechanism plumbing.
* `PublicDashboardRecord`, the export verification gate, and `tests/test_no_fake_results.py`.
  The guard runs whether or not a dashboard exists, and fails the moment an unverified export
  appears.
* `configs/models/gemma3_4b_it.yaml`, which is also the negative case in the calibration test
  that refuses a direction family built against a different model.
* `configs/tasks/mmlu_selected.yaml` and the verified `cais/mmlu` field mapping.
* The `adapted` model variant in `ModelSpec` and `ModelVariant`.

## The broader CSF-Bench study

The original study this repository was scaffolded for is preregistered in
`docs/preregistration.md`, which is unedited and still in force for everything outside the
BlueDot arm. Its phase list is in `docs/research_plan.md` and its slice status in
`docs/build_plan.md`. Both remain accurate as descriptions of that study; neither describes the
active path.

Order, if the broader study resumes:

1. Direction estimation with causal validation.
2. Benign model organism, which needs the LoRA compute decision above.
3. Remaining baselines and the state-conditioned forecaster.
4. Held-out mechanism transfer.
5. Dashboard, verbal reporter, SAE, and the 4B replication.
