# Build plan

Living document. Status values: `not started`, `in progress`, `done`, `blocked`.

Last updated: 2026-07-27.

The environment table below was measured on 2026-07-15 and has not been remeasured; the phase
status and the ordering sections have been updated since.

## Environment as inspected

Measured on the development machine on 2026-07-15.

| Item | Value |
| --- | --- |
| OS | Windows 11 Home 10.0.26200 |
| Shell | PowerShell 5.1, Git Bash available |
| CPU | Intel Core Ultra 7 258V (Lunar Lake) |
| RAM | 31.6 GB |
| Free disk | 483 GB on C: |
| GPU | No NVIDIA GPU. `nvidia-smi` is not present. No CUDA. |
| System Python | 3.14.3 (too new for the pinned research stack) |
| uv | 0.11.13 |
| Node | 24.14.1 |
| pnpm | 11.1.1 |
| git | 2.51.2 |

### Consequences of no CUDA

This is the single most important environment fact and it constrains the schedule.

The machine has an Intel integrated GPU, not a CUDA device. PyTorch XPU builds exist for
Intel Arc but the interpretability stack used here (Transformers hook-based capture, PEFT
LoRA training) is only routinely exercised on CUDA and CPU. Treating XPU as the primary
research device would add an unbudgeted debugging surface to every phase.

Decision, stated as an assumption rather than a question:

1. All device selection is resolved at runtime in one place, preferring `cuda`, then `mps`,
   then `cpu`. XPU is not claimed as supported until it is measured.
2. Local development and the entire test suite run on CPU against a tiny randomly
   initialized fixture model. The fixture model is a real Transformers model with a real
   forward pass, only very small. This keeps Phases 1 through 4 fully testable here.
3. Phases that need real Gemma 3 1B weights (Phase 2 onward) run on CPU locally for smoke
   scale only, with small `--max-items` and `--max-trials`. CPU forward passes on a 1B model
   are workable for tens of trials and not for thousands.
4. Phase 5 (LoRA training of the model organism) and the full trial sweeps are expected to
   need a rented CUDA GPU. The code must therefore be device-agnostic and checkpoint-driven
   from the start, so that a cloud run is a config change and not a rewrite.

No timing numbers appear in this document because none have been measured yet. When a real
run happens, its wall-clock cost gets recorded in `docs/experiment_log.md` from the run
manifest, not estimated here.

### Other environment notes

* The repository lives inside a OneDrive-synced directory. Model caches, activation caches,
  and virtual environments must stay out of sync range or out of the repository. The existing
  `.gitignore` already excludes them from git; OneDrive is a separate concern and is handled
  by keeping the Hugging Face cache at its default user-level location outside this folder.
* Python is pinned to 3.12 via `.python-version` and `requires-python` in `pyproject.toml`.
  3.12 is chosen over 3.11 for performance and over 3.13/3.14 because the PyTorch, PEFT, and
  NNsight versions targeted here do not all publish wheels for the newer interpreters.

## Phase status

| Phase | Name | Status | Active for the BlueDot arm? |
| --- | --- | --- | --- |
| 0 | Scientific contract | done | yes |
| 1 | Repository scaffold | done | yes |
| 2 | Real-model harness | done, validated on real Gemma 3 1B | yes |
| 3 | Intervention system | done, validated on real Gemma 3 1B | yes |
| 4 | Trial and commitment engine | done, resolution and scoring included | yes, with the protocol extensions in the BlueDot preregistration section 10 |
| 5 | Benign model organism | not started | **deferred** |
| 6 | Direction discovery | not started | **deferred**; the arm constructs directions rather than estimating them |
| 7 | Baselines | partial: constant and prompt-only lexical done | yes, as diagnostics |
| 8 | State-conditioned forecaster | not started | replaced for this arm by ridge methods; the state MLP is **deferred** |
| 9 | Held-out mechanism test | not started | **deferred** |
| 10 | Dashboard | not started | **deferred** |
| 11 | State-conditioned verbal reporter | not started | **deferred** |
| 12 | SAE extension | not started | **deferred** |
| 13 | Gemma 3 4B replication | not started | **deferred** |

Deferred means out of the active path for the BlueDot state-dependence arm. It does not mean
cancelled. The original roadmap in `docs/research_plan.md` is unchanged and these phases remain
part of the broader CSF-Bench study.

## Phase 0: Scientific contract

Definition of done: another researcher can identify the main comparison and the analysis
without reading the code.

- [x] `docs/research_plan.md`
- [x] `docs/preregistration.md` with primary hypothesis, primary outcome, primary comparison,
      planned sample counts, split policy, exclusion rules, stopping rule, statistical tests,
      and a separate exploratory section
- [x] `docs/claim_boundaries.md`
- [x] `docs/failure_modes.md`
- [x] `docs/methodology.md`
- [x] `docs/experiment_log.md` started with an explicit no-results statement
- [x] Split policy fixed in `configs/tasks/arc_mcq.yaml` before any forecaster exists

## Phase 1: Repository scaffold

Definition of done: tests pass, configs validate, a tiny deterministic experiment runs end
to end on CPU.

- [x] `pyproject.toml`, `.python-version`, `uv.lock`
- [x] `src/causal_self_forecasting/` package with Typer CLI
- [x] `schemas.py` with every record type listed in the specification
- [x] `hashing.py` with canonical JSON and SHA-256 helpers
- [x] `reproducibility.py` with seed control and environment capture
- [x] `config.py` loader validating YAML into Pydantic models
- [x] Structured logging
- [x] Tiny fixture model for tests, built from a real Transformers config at small size
- [x] `csf doctor`
- [x] pytest suite green (155 passed, 1 skipped at the time this phase closed; the suite has
      grown since and stands at 273 passed, 1 skipped as of 2026-07-27)
- [x] Ruff and Pyright clean
- [x] CI workflow

## Phase 2: Real-model harness

Definition of done: model loads, answer labels score correctly, clean reruns reproduce
within tolerance, hidden-state capture works.

Done against the fixture model. The phase stays open until a real pinned Gemma run has
succeeded, because that is what the phase exists for. `csf benchmark` is the command that
closes it.

- [x] Model loading with pinned revision recorded in the run manifest
- [x] Answer-label token resolution with rejection of unscoreable templates
- [x] Four-way label logit extraction, correct-answer margin, probabilities
- [x] Hidden-state capture at configured layers and positions
- [x] Device and dtype resolution in one place
- [x] Clean-rerun determinism check with a stated numerical tolerance
- [x] `csf benchmark`: a systems and clean-model sanity benchmark, classified non-scientific,
      with access classification, CPU timing, memory, and capture verification on real weights
- [x] Load Gemma 3 1B and measure clean accuracy on ARC (1 item, scoring smoke check only)
- [x] Measure the CPU cost of one forward pass (median 0.474 s), see `docs/compute_decision.md`

Phase 2 is done. The systems benchmark ran on real Gemma 3 1B and the intervention harness
was validated on real weights (see `docs/experiment_log.md`, 2026-07-18 and 2026-07-19).

### The systems benchmark

`csf benchmark` is deliberately not a CSF-Bench result and is typed so it cannot become one.
It exists to answer the operational questions the compute decision depends on:

* Do the pinned weights load on this machine?
* What does one forward pass cost here?
* Does the hook-owned capture path work on the real architecture, not only on the fixture?

The third question is the reason this is worth a command rather than a script. The
transformers 5 hidden-state indexing problem was caught on the fixture; the benchmark repeats
the same check on real weights by patching a layer with a known vector and requiring that
reading that layer back returns it. A hook can fire and still be read from the wrong point.

Access failures are classified rather than collapsed into "it did not work": no
authentication, authenticated without the Gemma license, unavailable revision, network
failure, offline cache miss, insufficient disk, and load failure are all distinct, because
their remedies are.

## Phase 3: Intervention system

Definition of done: no-op equality passes, invalid shapes fail loudly, seeded interventions
reproduce exactly. All required controls pass on the fixture model; see
`docs/experiment_log.md`.

- [x] No-op
- [x] Residual addition
- [x] Projection ablation
- [x] Matched random direction
- [x] Activation patching interface
- [x] Shape validation and pre/post norm logging
- [x] Zero-strength equality test
- [x] Sign-reversal coherence check (measured and reported, not pass/fail)

## Phase 4: Trial and commitment engine

Definition of done: a trial can be forecast, committed, selected, intervened on, scored, and
independently verified.

- [x] Candidate-set generation with randomized order and opaque IDs
- [x] Canonical JSON serialization
- [x] SHA-256 commitment with random salt
- [x] Selection seed accepted only after a commitment exists
- [x] Deterministic selection from seed
- [x] `csf verify run` recomputing every commitment from revealed salts
- [x] `csf trials resolve`: apply interventions and write observations, in two modes. Forecast
      mode requires committed forecasts, selects one candidate per trial after commitment,
      reveals and verifies. Ground-truth mode applies every candidate for training data and
      harness validation. Both refuse benchmark runs, refuse fixture runs as scientific, and
      preserve failed interventions with explicit error records.
- [x] `csf score run`: match committed forecasts to observations, headline metrics excluding
      no-ops, all-candidate metrics separately, grouped bootstrap intervals, exclusions
      reported. Never writes a public result.

Phase 4 is done. The full generate to score pipeline runs end to end on the fixture in the
test suite, and the resolution and intervention paths are validated on real Gemma 3 1B.

## Phase 7: Baselines (partial)

Definition of done for the two model-organism-free baselines: both use the same data splits
and scoring interface, and neither can see hidden states, adapter identity, private vectors,
outcomes from the predicted split, or the correct answer.

- [x] Constant baseline: training-split averages by public operation, layer, and strength. It
      cannot separate a real steer from a matched random control, by design, because they share
      a public group.
- [x] Prompt-only lexical baseline: TF-IDF of the prompt plus public strength and layer, ridge
      for delta and logistic for flip. Fenced to prompt and public metadata only.
- [x] Leakage audit test: `TrainingExample` is asserted to carry no private field, and the
      forecaster interface rejects a declared forbidden input.
- [ ] Prompt-only target-model report, linear state probe, state MLP, gradient baseline. Not
      started. The state MLP and the verbal reporter are deferred; the BlueDot arm uses ridge
      methods instead (Phase B below).

## The active path: the BlueDot state-dependence arm

As of 2026-07-27 the active work is the BlueDot state-dependence arm, preregistered in
`docs/bluedot/preregistration_state_dependence.md` with its execution order in
`docs/bluedot/execution_decision_tree.md`. It reuses Phases 1 through 4 and the Phase 7
baselines, and it defers Phases 5, 6, and 8 through 13.

| Slice | Deliverable | Status |
| --- | --- | --- |
| B0 | Preregistration and scope amendment | done, 2026-07-27 |
| B1 | Prompt manifests: `PromptRole`, `PromptManifest`, `csf prompts manifest` | **done, 2026-07-27**; split frozen at `data/prompt_manifests/bluedot_state_dependence_v1.json` |
| B2 | Direction bank and the fixed intervention projection matrix | not started |
| B3 | Global state-relative-by-calibration strength and `norm_ratio` | not started |
| B4 | 17-candidate and 81-candidate study candidate sets | not started |
| B5 | Commitment-protocol hardening: extended key, no-selection reveal, pre-existing-outcome refusal | not started |
| B6 | `StateAuditExample` and block-structured features | not started |
| B7 | `TransformFitRecord` and fit-boundary enforcement | not started |
| B8 | Three ridges and grouped cross-validation | not started |
| B9 | Paired prompt bootstrap and prompt-first aggregation | not started |
| B10 | `csf verify study` and `csf replay` | not started |
| B11 | Smoke run (8 prompts) | not started |
| B12 | Calibration, training, final test | not started |

Compute for the whole arm is about 40 minutes of forward time on this CPU, about 61 with the
preregistered layer-20 fallback. No GPU and no compute grant is needed. See
`docs/compute_decision.md` section 5.

The one prerequisite that is not compute: the prepared ARC dataset currently holds 20 items, and
the arm needs at least 168 eligible groups. `csf data prepare` must be rerun at a higher
`--max-items`, which needs network access and cannot run offline.

## Phase 5 onward, deferred for the BlueDot arm

Phase 5 is where the missing CUDA device starts to bind. LoRA training on CPU is classified
insufficient evidence in `docs/compute_decision.md`; a timed micro-run is needed before it is
treated as practical, and GPU rental should be considered there. The compute decision is the
maintainer's. None of this blocks the BlueDot arm, which uses no adapter.

Order for the broader CSF-Bench study, when it resumes:

1. Direction estimation with causal validation (Phase 6). The current directions are synthetic
   and unvalidated; the broader study needs an estimated, causally validated direction.
2. Benign model organism (Phase 5), which needs the LoRA training decision above.
3. The remaining baselines and the state-conditioned forecaster (Phases 7 and 8).
4. Held-out mechanism transfer (Phase 9).
5. Dashboard, verbal reporter, SAE, and the 4B replication (Phases 10 through 13).

## Non-negotiables carried through every phase

* No invented experiment results, sample counts, confidence intervals, hashes, or metrics.
  Planned values are labeled planned. Measured values come from saved verified artifacts.
* The dashboard shows an explicit empty state until a verified export exists.
* `tests/test_no_fake_results.py` fails if dashboard code contains benchmark-looking numbers
  that are not present in a verified export.
* Private payloads, salts, and selection seeds never enter version control.
* The model organism stays benign: a controlled answer-position preference under a
  deployment-like wrapper, nothing more.
