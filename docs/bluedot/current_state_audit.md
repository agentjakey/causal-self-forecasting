# BlueDot state-dependence study: current state audit

Read-first audit of the `causal-self-forecasting` checkout before the BlueDot state-dependence
study continues. No scientific code was modified. No calibration, training, or final-test
experiment was run. Every claim below was confirmed against the checkout on the date of this
audit; nothing is carried forward on trust from an earlier report.

Audit date: 2026-07-27. Audited commit: `86fb2d38b1f24f316aaa8b4e6b0abd57cf9ccd47`.

**Follow-up, same day.** The scope decisions this audit was written to inform have since been
frozen in `docs/bluedot/preregistration_state_dependence.md`, with the execution order in
`docs/bluedot/execution_decision_tree.md`. Where the two disagree, the preregistration governs.
Three things in this audit have been superseded or corrected in place, each marked at its
location: the compute arithmetic in section 15 (overstated by about 2.7 times), leakage risk R1
in section 11 (resolved by a design decision rather than left open), and one wrong file
attribution in discrepancy D1. Discrepancies D1, D2, and D9 have been fixed in the repository.

## 1. Exact branch and commit

| Item | Value |
| --- | --- |
| Branch | `bluedot/state-dependence-audit` (exists, checked out) |
| Commit | `86fb2d38b1f24f316aaa8b4e6b0abd57cf9ccd47` |
| Commit date | 2026-07-27 16:39:55 -0700 |
| Commit subject | Complete forecast resolution scoring and Gemma harness validation |
| Author | agentjakey |
| Tags | `csf-pre-bluedot-state-audit` -> `dc48379` |
| Other local branches | `main` |
| Remotes | `origin/main`, `origin/HEAD -> origin/main` |

The intended BlueDot branch has been created. `bluedot/state-dependence-audit` and `main` both
point at `86fb2d3`; the branch carries no commits of its own yet.

Recent history:

```text
86fb2d3 Complete forecast resolution scoring and Gemma harness validation
5738226 feat: add pinned-model systems benchmark
7a34f11 feat: add deterministic offline intervention harness
809b345 Update .gitignore
3a0861a Initial commit
```

## 2. Working-tree status

Clean. `git status --porcelain` produced no output. `git stash list` is empty.
`git diff --check` exited 0 with no whitespace or conflict-marker findings.

No last-run changes are missing from Git history: there is nothing uncommitted, and the only
untracked-by-design content is ignored artifact output (`results/runs/`, `artifacts/directions/*`,
`data/processed/`, `.venv`, caches).

## 3. Environment

| Item | Value |
| --- | --- |
| OS | Windows 11 Home 10.0.26200 (`Windows-11-10.0.26200-SP0`) |
| Shell used | PowerShell |
| Python | 3.12.13 (`.python-version` pins 3.12; `requires-python >=3.12,<3.13`) |
| Package manager | uv 0.11.13 |
| torch | 2.13.0+cpu, `cuda_available: False` |
| transformers | 5.14.0 |
| numpy | 2.5.1 |
| scikit-learn | 1.9.0 |
| pydantic | 2.13.4 |
| CLI entry point | `csf = causal_self_forecasting.cli:app` (Typer) |
| Offline flag | `HF_HUB_OFFLINE=1` set for every command run in this audit |

Confirmed from the checkout, not assumed: Python 3.12, Windows 11, CPU-only PyTorch, `uv`, and
the Typer `csf` entry point all match the last known verified state.

## 4. Tests and quality checks with exact counts

All commands were run from the repository root with `HF_HUB_OFFLINE=1`.

| Command | Result |
| --- | --- |
| `uv run pytest -q` | exit 0 |
| `uv run pytest` (rerun for the summary line) | **273 passed, 1 skipped** in 72.09 s |
| `uv run ruff check .` | `All checks passed!`, exit 0 |
| `uv run ruff format --check .` | `48 files already formatted`, exit 0 |
| `uv run pyright` | `0 errors, 0 warnings, 0 informations`, exit 0 |
| `git diff --check` | exit 0, no output |

0 failed. The previously reported counts reproduce exactly: 273 passed, 1 skipped.

Note on the summary line: `pyproject.toml` sets `addopts = "-q --strict-markers"`, so
`uv run pytest -q` becomes `-qq` and suppresses the pass/fail summary. Run `uv run pytest`
(no explicit `-q`) when the counts are needed.

Test-function counts by file (before parametrization; the suite collects 274 items):

| File | `def test_` count |
| --- | --- |
| `tests/integration/test_benchmark.py` | 68 |
| `tests/integration/test_model_harness.py` | 24 |
| `tests/integration/test_resolve.py` | 12 |
| `tests/unit/test_interventions.py` | 29 |
| `tests/unit/test_hashing.py` | 26 |
| `tests/unit/test_schemas.py` | 21 |
| `tests/unit/test_scoring.py` | 21 |
| `tests/unit/test_splitting_and_candidates.py` | 19 |
| `tests/unit/test_commitment.py` | 16 |
| `tests/unit/test_forecasting.py` | 16 |
| `tests/test_no_fake_results.py` | 7 |

## 5. Verified current capabilities

Confirmed by reading the code and by the passing suite.

**Model harness.** `models/loader.py` loads a pinned revision through one path and builds the
`ModelSpec` recorded in manifests from the same call. `ModelSpec` rejects `main`/`master`/`HEAD`
as revisions. `configs/models/gemma3_1b_it.yaml` pins `google/gemma-3-1b-it` at
`dcc83ea841ab6100d6b47a070329e1ba4cf78752`. `LoadedModel` exposes `decoder_layers`,
`embedding_module`, `num_layers`, and `hidden_dim`, read from the loaded model's own config.

**Answer scoring.** `models/scoring.py` resolves each answer label to exactly one token and
refuses templates where that fails or where labels collide. Margin is
`logit(correct) - max(logit(incorrect))` over the four label logits, computed in float64.
The saved benchmark artifact records `scoring_format:
raw_completion_next_token_after_answer_colon` and label token ids `A=562, B=603, C=565, D=622`.

**Capture.** `models/capture.py` owns its own forward hooks rather than using
`output_hidden_states`. Layer 0 is the embedding output; layer `i` is decoder block `i-1`.
The intervention hook is registered before capture hooks, so a capture at the intervened layer
sees the post-intervention value. `run_with_intervention` raises if the hook never fires.
`results/runs/benchmark-gemma3_1b_it-20260718T040531Z/benchmark.json` records
`capture: {layer: 13, shape: [1152], hook_fired: true, capture_point_verified: true,
max_abs_patch_error: 0.0}` for `google/gemma-3-1b-it` with 26 layers and hidden dim 1152.
Capture and intervention tensor alignment is therefore verified on real weights, not only on
the fixture.

**Interventions.** `interventions/tensor_ops.py` implements noop, residual add, projection
ablation, and activation patching as pure 1-D tensor functions with shape, emptiness, and
finiteness validation, plus pre/post/delta norm diagnostics. `interventions/validate.py` runs
five required controls (no-op equality, zero-strength equality, rerun determinism, shape
validation, hook applies intervention) and one reported measurement (sign-reversal coherence).
`docs/experiment_log.md` records these passing on real Gemma weights on 2026-07-19.

**Directions.** `interventions/directions.py` provides `DirectionStore` (one `.npz` per
direction holding a float32 1-D vector plus JSON metadata, with `save`, `load`, `metadata`,
`hash`, `has`, `list_ids`) and `matched_random_direction` (norm-matched to a reference vector,
seeded). `artifacts/directions/gemma_synthetic.npz` exists at dim 1152 with
`validated: false` in its metadata.

**Trials.** `trials/generate.py` produces `trial_manifest.jsonl`, `candidate_sets.jsonl`,
`state_refs.jsonl`, `states.npz`, `environment.json`, and `run_manifest.json` with no forecast
dependency. `trials/candidates.py` builds a four-candidate set (direction positive, direction
negative, matched random, no-op) with opaque ids assigned after a seeded shuffle, and a
`public_view` that hides mechanism and direction id.

**Commitment.** `trials/commitment.py` enforces the order: commit refuses to run once a
selection seed exists; seed writing refuses to run before any commitment exists; selection is
`derive_seed("selection", seed_hex, trial_id) % n`; reveal recomputes the hash and records
`verified` rather than raising. `verify_run_commitments` recomputes every commitment from the
forecast on disk plus the revealed salt.

**Resolution.** `trials/resolve.py` runs in two explicit modes. Forecast mode requires committed
forecasts, validates them (no duplicates, no orphan commitments, no uncommitted trials), selects
one candidate per trial, applies it, reveals, and verifies. Ground-truth mode applies every
candidate and records observations only. A one-time clean-reproduction guard compares the
resolution model's recomputed clean margin against the committed one within `rerun_tolerance`.
Failed interventions are appended to `resolution_failures.jsonl` and counted, never dropped.
Resolution classifies itself as `fixture_resolution`, `ground_truth_resolution`, or
`scientific_forecast_resolution`.

**Scoring.** `scoring/metrics.py` provides MAE, RMSE, sign accuracy, interval coverage, flip
Brier, flip log loss, top-effect accuracy, and `grouped_bootstrap_ci` (resamples groups, not
pairs; collapses to the point estimate with fewer than two groups). `scoring/run.py` matches
committed forecasts to observations, excludes no-ops from headline metrics and reports them
separately, restricts ranking to trials where every candidate was observed, and refuses to
score a scientific run whose commitments did not verify.

**Baselines and the leakage fence.** `forecasting/base.py` defines `TrainingExample`
(prompt text, four public features, observed delta, observed flip, group id, split), a
`FORBIDDEN_FIELDS` set checked at import time by `_validate_example_type()`, and
`Forecaster.check_no_leakage()`. `ConstantBaseline` and `PromptLexicalBaseline` are implemented
and both declare `hidden_state: False`. `build_training_examples` filters observations by trial
split so a forecaster never trains on outcomes from the split it will be scored on, pinned by
`test_build_training_examples_isolates_the_requested_split`.

**Integrity guards.** `PublicDashboardRecord` cannot be constructed with
`commitments_verified=False`. `MetricValue` requires `n > 0` and an interval.
`BenchmarkRecord.scientific_result` is typed `Literal[False]`. `tests/test_no_fake_results.py`
fails if a public export appears or if dashboard source ever carries hard-coded metric literals.
CI additionally greps tracked files for salts, seeds, and private payloads.

**Current scientific state.** Confirmed unchanged: no CSF result, no validated scientific
direction, no trained state forecaster, no public export. `results/public/` contains only
`.gitkeep`. `artifacts/directions/` contains only synthetic and matched-random vectors, all
marked `validated: false`.

## 6. Missing capabilities

Nothing named in the BlueDot study design exists yet in code. Concretely absent:

1. **Prompt manifests.** There is no persisted, role-labeled prompt set. Split assignment
   (`tasks/splitting.py`) is a pure function of `group_id` and `split_seed` into
   train/val/test/heldout, and trial subsampling (`trials/generate.py`) uses a seeded
   `random.sample` over eligible variants. Neither produces a citable 8/32/96/32 manifest, and
   `Split` has no `calibration` member.
2. **Answer-token unembedding directions.** No code reads the unembedding matrix. `LoadedModel`
   exposes `decoder_layers` and `embedding_module` only; there is no `unembedding` accessor and
   no centering utility.
3. **Seeded orthogonal random controls.** `matched_random_direction` draws an unconstrained
   Gaussian and matches norm. There is no orthogonalization against a supplied basis and no
   multi-direction orthogonal family generator.
4. **State-relative strength.** `InterventionSpec.strength` is an absolute scalar applied as
   `h + strength * v`. Nothing computes `alpha = ratio * ||h||` per prompt, and nothing records
   a norm ratio.
5. **Calibration.** No calibration run type, no calibration artifact, no CLI command.
6. **Numerical intervention projection.** No feature builder turns a direction plus sign plus
   norm ratio into a numeric vector, and no policy separates that numeric representation from
   the private semantic role (`InterventionSpec.analysis_role`).
7. **Intervention-only ridge, visible-information ridge, state-conditioned bilinear ridge.**
   None exist. `forecasting/__init__.py` exports only the two baselines.
8. **Grouped cross-validation.** No CV splitter of any kind.
9. **Matched wrong-state generation.** `StateCondition` has the enum members
   (`TRUE`, `SHUFFLED`, `WRONG_EXAMPLE`, `WRONG_MODEL`, `NONE`) and `ForecastRecord` carries the
   field, but nothing constructs a wrong-state feature vector.
10. **All-candidate final-test forecast commitment.** `commit_forecasts` does produce a forecast
    for every candidate, but the reveal/verify path is welded to single-candidate selection
    (see section 13).
11. **Paired prompt bootstrap.** `grouped_bootstrap_ci` takes one score sequence and one
    statistic. There is no paired two-method difference, and no prompt-level aggregation before
    the statistic is computed.
12. **Scientific-run verification for this study.** `csf verify run` verifies commitment hashes
    only. Nothing proves outcomes did not exist before commitment (section 13).
13. **Analysis-only replay.** No command recomputes predictions or scores from committed
    artifacts without loading a model.
14. **CLI coverage.** There is no `csf forecast` command at all. `commit_forecasts` is reachable
    only from Python. There is also no `csf export`.
15. **`csf directions estimate`.** Referenced by three code paths and by the docs; the command
    does not exist (section 16).

## 7. Reusable modules

These are ready to build on without modification.

| Module | Reuse for the state-dependence study |
| --- | --- |
| `hashing.py` | Canonical JSON, `hash_object`, `hash_file`, salts, `commitment_hash`, atomic writes, JSONL read/write. Used unchanged for manifests, transform provenance, and commitments. |
| `paths.py` | Single owner of run-directory filenames. Add new constants here, not inline. |
| `config.py` | YAML load, validation, `config_hash` over the parsed document, `resolve_experiment`. |
| `reproducibility.py` | `derive_seed` (stable 32-bit from arbitrary labels), `set_global_seed`, `environment_snapshot` including git commit and dirty flag. |
| `models/loader.py` | Pinned load, device/dtype resolution, `ModelSpec` provenance. |
| `models/capture.py` | Final-prompt-token residual capture at layer 13 and layer 20; intervened forward with hook-fired enforcement. |
| `models/scoring.py` | `resolve_label_token_ids`, `score_logits`, `entropy_from_label_logits`. Supplies both the target (`delta_margin`) and the clean output distribution the visible-information ridge needs. |
| `interventions/tensor_ops.py` | `apply_residual_add` covers all sixteen signed directions; diagnostics give `pre_norm`, `post_norm`, `delta_norm` for free. |
| `interventions/directions.py` | `DirectionStore.save/load/metadata/hash` accepts any finite 1-D float32 vector. Answer-token directions and orthogonal controls store and hash through it unchanged. |
| `trials/states.py` | `StateShardWriter` and `load_state` for the captured 1152-dim states, with per-shard hashing. |
| `trials/candidates.py` | `build_candidate_set` and `candidate_set_hash` scale to 81 candidates; `public_view` is the enforcement point for the feature policy. |
| `trials/commitment.py` | Commit, salt handling, hash recomputation, `verify_run_commitments`. Extension needed only for the no-selection reveal. |
| `scoring/metrics.py` | `PairScore` and the point statistics are directly reusable; the bootstrap needs a paired sibling. |
| `forecasting/base.py` | `CandidatePrediction`, the commit driver structure, and the leakage-fence pattern. `TrainingExample` itself stays untouched. |
| `tests/conftest.py` | `isolated_runs` and `isolated_directions` fixtures; `tests/integration/test_resolve.py` has a full offline `Workspace` fixture that a new study test can copy. |

## 8. Required schema changes

All records live in `schemas.py` with `extra="forbid"` and `frozen=True`, so every new field is
a deliberate, validated addition. Adding fields to existing records is safe for forward reads
but will reject artifacts written by an older version only if a field is removed, not added.

1. **`Split`: add `CALIBRATION = "calibration"`.** Or, if the split policy must stay frozen,
   keep `Split` untouched and carry the study role in a new `PromptRole` enum on the manifest.
   Recommended: new `PromptRole` (`smoke`, `calibration`, `training`, `final_test`), leaving
   `Split` alone, because `Split` is cited by the preregistration's fixed split policy and by
   `ScoreRecord`.
2. **New `PromptManifest` / `PromptAssignment` records.** Fields: `manifest_id`, `task_name`,
   `task_manifest_hash`, `seed`, per-prompt `variant_id`, `item_id`, `group_id`, `role`, and a
   manifest hash. Must be citable by hash from every run manifest.
3. **New `DirectionFamilyRecord`.** Which eight directions exist, how each was constructed
   (`answer_token_unembedding_centered` vs `seeded_orthogonal_random`), the seed, the
   orthogonalization basis, the model id and revision, hidden dim, and the `DirectionStore` hash
   of each vector. Needed so a direction set is citable and reproducible.
4. **`InterventionSpec`: add `norm_ratio: float | None`.** Required so a state-relative
   intervention records the ratio that generated its absolute `strength`. Also extend the
   `forbidden` set in `_check_mechanism_requirements` to include `norm_ratio` if the ratio is
   ever mirrored into `public_metadata` (see the leakage note in section 11).
5. **New `CalibrationRecord`.** Per calibration prompt and per norm ratio: state norm, resulting
   absolute alpha, observed `delta_margin` range, degenerate-effect flags, and the decision the
   calibration run supports. Must carry `scientific_result: Literal[False]`.
6. **New `CommitmentReveal`** (or make `SelectionReveal.selected_intervention_id`,
   `selected_index`, and `selection_seed_hash` optional). The final-test protocol reveals the
   salt and verifies the commitment for an all-candidate forecast where no candidate is
   selected. The current `SelectionReveal` requires a selection.
7. **`ForecastRecord`: relax or default the model-organism fields.** `p_hidden_bias_active` is
   required and `ForecastCandidate.p_bias_suppressed` is required with no default. The
   state-dependence study has no bias construct. Either give both a documented `0.5` default or
   add a `target: Literal["delta_margin"]` discriminator so a reader cannot mistake the
   placeholder for a prediction.
8. **Commitment key must include the state condition.** `ForecastCommitment`,
   `verify_run_commitments`, and `scoring/run.py` all key on `(trial_id, method_id)`. The
   matched wrong-state control is the same method under a different state condition. Either add
   `state_condition` to the commitment key, or require every method to encode it in `method_id`
   and add a validator that rejects a `ForecastRecord` whose `state_condition` disagrees with its
   `method_id` suffix. Recommended: extend the key; a naming convention is not enforceable.
9. **New `TransformFitRecord`.** Fit provenance for every fitted transform (standardizer, TF-IDF
   vocabulary, ridge coefficients): the transform id, the exact list of training prompt ids it
   saw, the prompt manifest hash, the seed, and the hash of the serialized transform.
10. **New `StateAuditExample`** in the forecasting layer (a dataclass, not a pydantic record):
    see section 11.
11. **Run role on `RunManifest`.** `phase` is a free-form string. Add
    `run_role: Literal["smoke", "calibration", "training", "final_test_unresolved",
    "final_test_resolved"]` so the four run types are distinguishable structurally rather than by
    the presence or absence of files.

## 9. Required CLI changes

Current command tree, from `csf --help`:

```text
csf version
csf doctor
csf benchmark
csf data prepare
csf directions synthetic
csf interventions validate
csf trials generate
csf trials resolve
csf score run
csf verify run
```

Needed for the study:

| Command | Purpose |
| --- | --- |
| `csf prompts manifest` | Build and hash the 8/32/96/32 role-disjoint prompt manifest from a prepared task. Offline, no model. |
| `csf directions answer-tokens` | Build the four centered answer-token unembedding directions from the pinned model and store them with provenance. |
| `csf directions orthogonal` | Build the four seeded random directions, orthogonalized against the answer-token family. |
| `csf directions estimate` | Referenced by three existing error messages and by the docs but absent. Either implement it or correct those three messages (section 16). |
| `csf calibrate` | Run the norm-ratio grid on the 32 calibration prompts and write a `CalibrationRecord`. Must refuse to run against the training or final-test manifest roles. |
| `csf forecast fit` | Fit a named method on the training manifest only, writing a `TransformFitRecord`. |
| `csf forecast commit` | Predict every candidate and commit, for every method and state condition. No CLI path to `commit_forecasts` exists today. |
| `csf trials resolve --all-candidates` | A third resolution mode: requires committed forecasts, applies every candidate, reveals and verifies without selection. |
| `csf score compare` | Paired prompt-grouped bootstrap of method differences. |
| `csf verify study` | Check ordering (commitments predate observations), transform-fit provenance, prompt-role disjointness, and direction hashes. |
| `csf replay` | Analysis-only recomputation from committed artifacts, model never loaded. |

Two CLI facts worth carrying into the plan: `csf doctor` validates every config in
`configs/*/` and exits nonzero on a failure, so every new config must validate under an existing
`Base` subclass or `doctor` will break; and there are deliberately no placeholder commands in
this CLI, which is a convention the new commands should keep.

## 10. Required tests

Grouped by the risk each one closes. All must run offline on the fixture model.

**Prompt manifests**
* A manifest built twice from the same task and seed is byte-identical.
* Roles are disjoint by `group_id`, not merely by `variant_id`.
* Counts are exactly 8 / 32 / 96 / 32, and building with fewer available groups fails loudly
  rather than silently under-filling.
* A manifest cites the task manifest hash, and a changed task invalidates it.

**Answer-token directions**
* Four directions are produced, one per label token, and each has the model's hidden dim.
* Centering is applied (mean over the four label rows removed) and the result is asserted, not
  assumed.
* Round-tripping through `DirectionStore` preserves the vector to float32 exactness and the
  hash is stable.
* A direction built against a different hidden dim is rejected at intervention time (already
  covered by `_build_payload`; extend to the new family).

**Orthogonal random controls**
* Each random direction is orthogonal to all four answer-token directions within tolerance.
* The family is mutually orthogonal within tolerance.
* Regenerating from the same seed reproduces the vectors exactly.
* Norms are matched to the answer-token family, so an effect-size difference is not a magnitude
  difference.

**State-relative strength**
* `alpha = ratio * ||h||` for a hand-built state and a unit direction.
* `delta_norm` from `InterventionDiagnostics` equals `alpha` within tolerance, which is the
  direct check that the state-relative scaling reached the tensor.
* Ratio 0.0 reproduces the clean output exactly (the existing zero-strength control, restated
  in ratio terms).
* The recorded `norm_ratio` and the recorded `strength` are consistent with the recorded state
  norm.

**Feature policy**
* `TrainingExample` still carries no forbidden field (the existing test must keep passing
  unchanged).
* `StateAuditExample` carries no `analysis_role`, no `direction_id`, no `mechanism`, no
  correct answer, and no observed outcome for the split being predicted.
* The intervention numeric representation is identical between a real answer-token direction
  and an orthogonal control of the same norm ratio and sign, except for the vector values
  themselves; no field names the family.
* The visible-information feature vector contains no state-derived quantity, asserted field by
  field.
* The wrong-state example differs from the true-state example in exactly the state-derived
  block and nowhere else.

**Transform provenance**
* A transform fitted on the training manifest records exactly those prompt ids.
* Fitting on any prompt outside the training role raises.
* A `TransformFitRecord` hash changes when the transform changes.

**Grouped cross-validation**
* No `group_id` appears in both the fit and the held-out fold.
* Folds are deterministic from the seed.
* Fold count and fold sizes are reported, not silently rebalanced.

**Commitment protocol**
* Committing a forecast for a run that already has observations raises (this guard does not
  exist yet).
* All-candidate reveal verifies every commitment without selecting a candidate.
* An edited forecast fails verification (already covered for the selection path; extend to the
  no-selection path).
* Two methods that differ only in state condition produce two distinct commitments and both
  verify.

**Scoring and bootstrap**
* Per-prompt aggregation happens before the method statistic is computed, checked on a
  hand-built unbalanced set where pair-level and prompt-level means differ.
* The paired bootstrap draws the same prompt groups for both methods in each resample.
* With one group the paired interval collapses to the point difference.
* A method with a missing prediction for a prompt is excluded from the pair, and the exclusion
  is counted.

**Replay and verification**
* `csf replay` produces byte-identical scores from committed artifacts with no model available
  (assert by making the loader raise).
* `csf verify study` fails on a run where an observation timestamp or file predates the
  commitment.

## 11. Leakage risks

**R1. `strength` becomes state-derived under norm-relative scaling.** This is the most important
finding in this section. `trials/candidates.py:public_view` publishes `strength`, and
`forecasting/base.py:PUBLIC_FEATURE_KEYS` hands it to every method including the
visible-information ridge. If `strength` is set to `alpha = ratio * ||h||`, then the published
strength encodes the state norm, and the "visible information only" method silently receives a
state-derived feature. The comparison the study exists to make would be contaminated at the
source.

**Resolved 2026-07-27 by a design decision, not by a mitigation.** The preregistration prohibits
prompt-specific state-relative strengths outright. One global absolute strength
`alpha = ratio * median clean-state norm over the 32 calibration prompts` is applied to every
prompt at a given layer and ratio. A global alpha is identical across prompts, so publishing
`strength` carries zero prompt-specific state information, and the visible-information ridge
stays honest by construction rather than by a filter someone has to remember. The calibration
prompts are disjoint from training and final test, so even the aggregate norm does not cross a
role boundary.

The residual obligation is still real and is carried into the plan: `InterventionSpec` must
record `norm_ratio` alongside `strength` for reproduction, and a test must assert that
`public_view` never returns a value that varies with the prompt's state norm. That test is
cheap and it is what keeps a future prompt-relative variant from reintroducing the leak
silently.

**R2. `check_no_leakage` blocks the study's own methods.**
`forecasting/base.py:check_no_leakage` raises `LeakageError` for any method declaring
`hidden_state` or `intervention_vector`, and `commit_forecasts` calls it unconditionally. The
state-conditioned bilinear ridge must declare both. Do not weaken the existing check: it is what
protects the baselines. Add a separate policy object (an allowed-inputs set per method class)
and a distinct commit driver for state-audit methods, so the baseline fence stays exactly as it
is and the new fence is explicit about what each new method may read.

**R3. `analysis_role` is one field away from the public view.**
`InterventionSpec.analysis_role` names the semantic role (`direction_positive`,
`random_control`). It is excluded from `public_view` and `public_metadata` rejects it by name.
With eight direction families instead of two, the temptation to add a `family` field for
analysis convenience is real. Any such field must live beside `analysis_role`, never in
`public_metadata`, and the existing `forbidden` set in `InterventionSpec` should be extended
with the new names at the same time the fields are added.

**R4. Direction id patterns leak family membership.** `_ensure_random_controls` names controls
`{direction_id}__random_{index:02d}`. `direction_id` is not published today, but the new
`DirectionFamilyRecord` and any per-candidate provenance must not surface those names to a
method. Keep candidate ids opaque (`{trial_id}.opaque_{index:02d}`) as they are now.

**R5. Transform fit on the wrong prompts.** TF-IDF vocabulary, feature standardization, PCA if
any, and the ridge coefficients themselves must be fitted on the 96 training prompts only. The
calibration prompts are a separate role precisely so that the alpha grid can be chosen without
touching training data, and the final-test prompts must not influence any fitted object.
Nothing in the current code enforces a fit boundary; `TransformFitRecord` plus a fit-time
assertion is the enforcement.

**R6. Wrong-state substitution must replace only state-derived features.** If the wrong-state
example is built by re-running feature extraction against a different prompt's state, it will
also pick up that prompt's clean logits and text unless the builder is written to substitute a
named block. Build the feature vector as explicitly named blocks (`prompt_block`,
`clean_output_block`, `intervention_block`, `state_block`, `state_x_intervention_block`) and
have the wrong-state constructor replace `state_block` and `state_x_intervention_block` only,
asserting the other blocks are unchanged.

**R7. Training on outcomes from the predicted split.** `build_training_examples` already filters
by trial split, and the isolation test passes. The new `StateAuditExample` builder must reuse
that discipline rather than reimplement it, and must filter on the prompt manifest role, not on
`Split`.

## 12. Preregistration risks

**P1. The preregistration describes a different study.** `docs/preregistration.md` v1.0 fixes
the primary outcome as answer-flip Brier and the primary comparison as `state_mlp` versus
`prompt_tfidf`, with H1 as an internal-state advantage under a same-prompt clean-versus-adapted
state swap. The BlueDot study has `delta_margin` as its primary target, no model organism, no
adapted model, and different method names. Running the BlueDot study under the existing document
would be a silent amendment, which section 0 of that document explicitly names as the thing not
to do. A dated amendment entry, or a separate preregistration for the state-dependence study
that cites v1.0 as superseded for this arm, is required before any calibration run.

**P2. Planned sample counts do not match.** Section 9 plans 500 items, 8 variants per item, and
2000 trials. The BlueDot study is 168 prompts. The realized counts must be prespecified, not
back-filled from whatever the compute allows.

**P3. The decision rule must be restated for the new comparison.** The existing rule is a paired
Brier difference on `state_mlp` versus `prompt_tfidf`. The BlueDot rule needs to name: the
statistic (prompt-aggregated error on `delta_margin`), the two comparisons (bilinear versus
visible-information, and true-state versus matched wrong-state), the interval method (paired
prompt bootstrap), and what counts as no detected difference.

**P4. Method names in the docs do not exist in code.** `methodology.md` section 6 and the
preregistration both name `prompt_tfidf`; the implemented method id is `prompt_lexical`. If the
BlueDot analysis cites either document, the mapping has to be written down.

**P5. Exclusion rules need a state-dependence clause.** Section 10 covers item, trial, and
forecast exclusions. It does not cover a degenerate norm ratio (an alpha so small that
`delta_margin` is below the rerun-determinism floor, or so large that the output collapses).
The calibration run exists to find that boundary; the rule for excluding a ratio must be fixed
before calibration is read, or the grid choice becomes a post-hoc decision.

**P6. The fallback layer is a researcher degree of freedom.** "Primary layer 13, one allowed
fallback layer 20" needs a prespecified trigger. Without one, moving to layer 20 after seeing
layer 13 results is a selection on the outcome.

## 13. Commitment-protocol risks

**C1. Nothing prevents outcomes from existing before commitment.** This is the largest protocol
gap for the study. `commit_forecast` refuses only when a selection seed exists. In ground-truth
mode, `resolve_run` writes `observations.jsonl` with no commitment check at all. So the sequence
"resolve the final-test prompts, look at the observations, then commit forecasts" is not
detected by any code path, and `verify_run_commitments` would still report `verified: true`
because the hashes recompute correctly. The audit question "can existing verification prove that
test outcomes did not exist before forecast commitment" answers **no**.

Required: `commit_forecast` must refuse when `observations.jsonl`, `resolution.json`, or
`scores.json` exists in the run directory, and `csf verify study` must record the check. This is
the same class of guard as the existing selection-seed check and belongs beside it.

**C2. Reveal and verification are welded to single-candidate selection.**
`reveal_selection` computes a selected index, and `SelectionReveal` requires
`selected_intervention_id`, `selected_index`, and `selection_seed_hash`.
`verify_run_commitments` marks a run unverified unless every commitment has a matching reveal.
For the all-candidate final test there is no selection, so today the choice is between a
ground-truth resolution that never reveals (leaving `verified: false` and
`classification: ground_truth_resolution`) and a forecast-mode resolution that observes only one
candidate per prompt. Neither is the study's design. A no-selection reveal path is required.

This does not block the study in a deep way: the commitment model itself is all-candidate
(`ForecastRecord.candidate_forecasts` covers every candidate, and `commit_forecasts` already
predicts every one). The coupling is in the reveal record and the resolution mode, both of which
are small, contained changes.

**C3. Commitment keys collide across state conditions.** Described in section 8, item 8. Two
`ForecastRecord`s with the same `method_id` and different `state_condition` overwrite each other
in `verify_run_commitments`'s `forecasts` dict, in `_read_commitments`'s dedup check
(`_validate_commitments` would raise `duplicate commitment`), and in
`scoring/run.py:method_state_condition`. The matched wrong-state control is exactly this case.

**C4. Salt-per-key path shares the same collision.** `_salt_path` is
`{trial_id}.{method_id}.salt`. Extending the commitment key means extending this path too, or
the wrong-state salt overwrites the true-state salt on disk and one of the two commitments
becomes unverifiable.

**C5. The protocol's stated limits still hold and must stay stated.**
`docs/claim_boundaries.md` is explicit that the commitment scheme provides sequencing and
tamper evidence, not protection against an experimenter who controls the machine. Adding C1's
guard raises the cost of self-deception further but does not change that boundary, and the
BlueDot writeup must not imply that it does.

## 14. Artifact and verification risks

**A1. Run role is inferred, not recorded.** `RunManifest.phase` is a free-form string
(`trials_generate`, `systems_benchmark`). The four study run types (calibration, training,
unresolved test, resolved test) are currently distinguishable only by which files happen to be
present. A run directory is evidence; its role should be a typed field, not an inference.

**A2. `observations.parquet` is a documented artifact that nothing writes.**
`paths.OBSERVATIONS = "observations.parquet"` is unused; the real file is
`observations.jsonl` (`paths.OBSERVATION_RECORDS`). `paths.py` documents the discrepancy in a
comment. `docs/methodology.md` section 8 lists `observations.parquet` as an artifact runs write,
which is wrong. Anyone auditing a run directory against the methodology doc will look for a file
that does not exist.

**A3. The artifact list in the methodology is incomplete.** Section 8 omits `states.npz`,
`state_refs.jsonl`, `resolution.json`, `resolution_artifact_hashes.json`,
`resolution_failures.jsonl`, `score_records.jsonl`, and `run.log.jsonl`, all of which real runs
write (confirmed against `results/runs/gemma-harness/`, which contains ten files).

**A4. States are hashed per shard, not per state.** `ModelStateRef` cites `shard_hash` and
`row_index`. A single-state citation therefore depends on the whole shard being intact. That is
adequate, but the wrong-state control needs to cite two states from possibly different runs; the
citation format should be pinned before those artifacts are produced.

**A5. The repository sits inside a OneDrive-synced directory.** `docs/build_plan.md` flags this.
Run artifacts under `results/runs/` are inside sync range. For a run that must be citable, the
hash in `resolution_artifact_hashes.json` is the defense, and it should be checked, not assumed,
during `csf verify study`.

**A6. Direction artifacts are git-ignored by default.** `.gitignore` excludes
`artifacts/directions/*` except `.gitkeep`, with a comment that a publishable direction can be
added with `git add -f`. The eight study directions are deterministic from a seed and the pinned
model, so regeneration is the reproduction path; the `DirectionFamilyRecord` hashes are what make
that checkable. This needs to be a deliberate decision, not a default.

**A7. `results/runs/gemma-harness` has no `run.log` gap but also no scores.** It contains
`observations.jsonl` and `resolution.json` from the 2026-07-19 ground-truth validation. It is
not a scientific artifact and is correctly labeled. It should not be reused as a study run
directory; the study should start clean run ids.

## 15. Compute assessment

All arithmetic below derives from one measured value: the median CPU forward time
`T = 0.474158 s` for `google/gemma-3-1b-it` at revision `dcc83ea841ab...`, read from
`results/runs/benchmark-gemma3_1b_it-20260718T040531Z/benchmark.json`. These are **estimates,
not measurements of the study**. No study run has been executed.

**Corrected 2026-07-27.** The original version of this section applied the full five-ratio grid
to every prompt role. That is wrong: only calibration sweeps the ratio grid. Smoke, training, and
final test run at the single selected ratio, so they carry 17 candidates per prompt, not 81. The
error overstated the arm's cost by about 2.7 times. The corrected figures below are the ones the
preregistration and `docs/compute_decision.md` section 5 use.

Candidates per prompt: 8 directions x 2 signs = 16, plus 1 no-op = **17** at smoke, training, and
final test; 5 ratios x 16 = 80 plus 1 no-op = **81** at calibration only.
Forwards per prompt: 1 clean capture forward plus one per candidate.

| Role | Prompts | Candidates | Forwards | Estimated CPU seconds | Estimated minutes |
| --- | --- | --- | --- | --- | --- |
| Smoke | 8 | 17 | 144 | 68 | 1.1 |
| Calibration | 32 | 81 | 2,624 | 1,244 | 20.7 |
| Training | 96 | 17 | 1,728 | 819 | 13.7 |
| Final test | 32 | 17 | 576 | 273 | 4.6 |
| **Total** | **168** | | **5,072** | **2,405** | **40.1** |

A layer-20 fallback repeats calibration only: 2,624 further forwards, 1,244 s, about 21 minutes,
for a worst case near 61 minutes.

Excluded from the table and real: about 48 s of model load per process (measured, same
artifact), and roughly 1.5 percent capture overhead per captured forward (measured, same
artifact). The compute-decision record warns that the timing sample is three forwards of one
53-token prompt, and that long-prompt-heavy runs may cost 1.5 to 2 times T. Applying that band
puts the study at roughly 40 to 80 minutes, or 61 to 122 minutes with the fallback.

The three ridge models, the grouped cross-validation, the wrong-state control, and the bootstrap
add no forward passes; they run on recorded observations and stored states. Storage is
negligible: 168 prompts x 1152 float32 values per captured layer is under 1 MB per layer.

**Conclusion: no GPU and no compute grant is required.** The study is comfortably within CPU
reach on this machine, by the same 12-hour threshold the existing compute decision uses. GPU
rental should be moved to deferred work (section 17).

One prerequisite is not compute: only **20 ARC items and 160 variants** are currently prepared
(`data/manifests/arc_mcq.json`, `item_splits: {train: 12, val: 3, test: 5}`). The study needs 168
role-disjoint prompts. `csf data prepare --config configs/tasks/arc_mcq.yaml` must be rerun at a
higher `--max-items`, which requires network access to `allenai/ai2_arc` and therefore cannot run
under `HF_HUB_OFFLINE=1`. This is dataset preparation, not a scientific run, but it is a blocking
prerequisite and it was not performed during this audit.

## 16. Discrepancies between code and documentation

Recorded here rather than fixed, in line with the read-first scope of this pass.

| # | Discrepancy | Evidence |
| --- | --- | --- |
| D1 | `csf directions estimate` does not exist but is cited as the real-run path. **Fixed 2026-07-27** in all four places. | `trials/generate.py:187`, `interventions/validate.py:324-325`, `cli.py:170`, `configs/experiments/smoke.yaml:9`. `csf directions --help` lists only `synthetic`. The audit's original attribution of a fourth reference to `docs/build_plan.md` was wrong; that file describes direction estimation as a phase but never names the command. |
| D2 | `docs/methodology.md` section 8 says runs write `observations.parquet`. They write `observations.jsonl`. **Fixed 2026-07-27.** | `paths.py:19-24` (the constant is unused and the comment says so); `results/runs/gemma-harness/` contains `observations.jsonl`. |
| D3 | The methodology artifact list omits seven files real runs write. **Fixed 2026-07-27.** | Section 14, A3. |
| D4 | `docs/preregistration.md` and `docs/methodology.md` name the prompt-only baseline `prompt_tfidf`; the implemented method id is `prompt_lexical`. | `forecasting/lexical.py:27`. |
| D5 | `docs/preregistration.md` and `docs/methodology.md` name `state_mlp` as the primary method; it does not exist. | `forecasting/__init__.py` exports two baselines. |
| D6 | `docs/build_plan.md` header says "Last updated: 2026-07-15" but the file describes work logged on 2026-07-18 and 2026-07-19. | `docs/build_plan.md:5` versus lines 134-135. |
| D7 | `docs/build_plan.md` Phase 1 records "pytest suite green (155 passed, 1 skipped)". Current is 273 passed, 1 skipped. Historically true for that phase, misleading as a current statement. | Section 4. |
| D8 | `docs/build_plan.md` "Order from here" lists direction estimation twice (items 1 and 5) and the model organism twice (items 2 and 4). | `docs/build_plan.md:217-224`. |
| D9 | The README quickstart runs `csf score run --run-id <RUN_ID>` immediately after a ground-truth resolve. `score_run` raises `ScoringError` when no forecasts are committed, so that sequence fails as written, and no CLI command commits a forecast. **Fixed 2026-07-27.** | `README.md:80-85` versus `scoring/run.py:123-125`. |
| D10 | `ExperimentConfig.candidates_per_trial` is validated (`ge=2, le=8`) but never read. Candidate count is fixed at four by `default_templates`. The study needs 81 candidates, so this field is both dead and, if revived, wrongly bounded. | `config.py:152`; no consumer in `src/`. |
| D11 | `docs/claim_boundaries.md` "Current status" says "No experiment has been run" and describes the repository as containing "a smoke test on a randomly initialized fixture model". Real-Gemma systems and harness validation have since run. The statement is still true about scientific results but understates what exists. | `docs/claim_boundaries.md:6-10` versus `docs/experiment_log.md` 2026-07-18 and 2026-07-19. |

**No documentation file was modified in the audit pass itself.** Each discrepancy above was
recordable without correcting the source document, so the exception for directly contradictory
statements was not triggered.

**Follow-up, 2026-07-27.** In the subsequent scope-and-preregistration pass, D1, D2, D3, and D9
were fixed, and D6, D7, D8, and D11 were addressed in `docs/build_plan.md` and
`docs/claim_boundaries.md`. D4 and D5 were reconciled by a naming note in `docs/methodology.md`
rather than by renaming anything in code. D10 remains open and is listed as a required change in
the preregistration's schema-conflict table (S6).

## 17. Smallest ordered implementation plan

Each slice is independently testable and leaves the suite green. No slice past S0 was started.

**S0. Preregistration amendment (documentation, no code).** Write the BlueDot arm's question,
primary target (`delta_margin`), the two comparisons, the decision rule, the realized sample
counts, the degenerate-ratio exclusion rule, and the layer-20 fallback trigger. Nothing else
starts until this exists, because everything after it is a researcher degree of freedom.

**S1. Prompt manifests.** `PromptRole` enum, `PromptManifest` record, a builder module, and
`csf prompts manifest`. Offline, no model, no torch. Unblocks every later slice's split
discipline. Requires the ARC re-preparation prerequisite (section 15).

**S2. Direction families.** Unembedding accessor on `LoadedModel`, four centered answer-token
directions, four seeded orthogonal random directions, `DirectionFamilyRecord`, and two CLI
commands. Stores through the existing `DirectionStore` unchanged.

**S3. State-relative strength.** `norm_ratio` on `InterventionSpec`, an alpha resolver that
reads the captured state norm, and the `public_view` change that publishes the ratio rather than
the alpha (leakage risk R1). This is the slice that must not be rushed: it is where the
visible-information ridge either stays honest or quietly gains a state feature.

**S4. Candidate-set generation for the 81-candidate grid.** Extend `default_templates` with a
study-specific builder. Fix or remove `candidates_per_trial` (D10). Verify
`candidate_set_hash` and the commitment payload size are sane at 81 candidates.

**S5. Commitment-protocol hardening.** The pre-existing-observations guard (C1), the
no-selection reveal path (C2), and the state-condition-extended commitment key and salt path
(C3, C4). Pure protocol work, testable on the fixture, and it must land before any final-test
forecast is committed.

**S6. `StateAuditExample` and the block-structured feature builder.** Named feature blocks, the
separate allowed-inputs policy for state-aware methods (R2), the wrong-state substitution
constructor (R6), and the leakage tests. `TrainingExample` and its existing test stay untouched.

**S7. `TransformFitRecord` and fit-boundary enforcement.** Fit provenance, the training-role
assertion, and the hash. Small, and it makes S8 auditable.

**S8. The three ridges and grouped cross-validation.** Intervention-only, visible-information,
and state-conditioned bilinear, plus the group-disjoint CV splitter. All fit on stored features;
no model load.

**S9. Paired prompt bootstrap and prompt-level aggregation.** `prompt_aggregated` statistics in
`scoring/metrics.py` and a `paired_grouped_bootstrap` that draws the same prompt groups for both
methods. Add `csf score compare`.

**S10. `csf verify study` and `csf replay`.** Ordering checks, transform provenance checks,
prompt-role disjointness, direction hash checks, and analysis-only recomputation with the loader
made to raise.

**S11. Smoke run (8 prompts).** The first slice that touches real weights. It is a pipeline
check, not a scientific run, and its artifact must be classified accordingly.

**S12. Calibration (32 prompts), then training (96), then final test (32).** In that order, with
the final-test forecasts committed and verified before the final-test resolution begins. Each is
a separate maintainer decision.

**The exact next implementation slice is S0 followed by S1.** S0 is documentation the maintainer
owns; S1 is the smallest piece of code that unblocks everything else and needs no model, no
network beyond the one-time ARC re-preparation, and no protocol change.

## 18. Stop conditions

Halt and reassess rather than continuing, if any of these occur.

1. **Calibration shows no usable norm-ratio window.** If every ratio in
   `{0.02, 0.05, 0.10, 0.20, 0.40}` either produces a `delta_margin` below the rerun-determinism
   floor or drives the four-way distribution to a degenerate argmax on most calibration prompts,
   the intervention family is not measuring what the study needs. Stop; do not widen the grid
   after seeing the results without a recorded amendment.
2. **Layer 13 and layer 20 both fail the calibration window.** The design allows one fallback.
   Trying a third layer is a selection on the outcome. Stop.
3. **Any required intervention control fails on the study configuration.**
   `csf interventions validate` failing means every number downstream is untrustworthy.
4. **The clean-reproduction guard trips during a study run.** A model, revision, or precision
   mismatch invalidates every delta in that run.
5. **A commitment does not verify.** Record the failure, do not re-resolve to make it go away.
6. **Prompt roles are not disjoint by group.** Any overlap between the training and final-test
   groups invalidates the headline comparison; rebuild the manifest, do not patch the split.
7. **A transform is found to have been fitted outside the training role.** Refit from scratch;
   do not reuse the artifact.
8. **Estimated compute exceeds roughly 12 CPU hours** after the first real timing on study-length
   prompts. Re-plan the grid rather than absorbing the cost silently.
9. **The final-test resolution produces a materially different result from the training-fold
   cross-validation.** Report it; do not iterate on the test set.
10. **Any documentation would have to describe a planned value as measured.** Stop and fix the
    document.

## 19. Definition of done for the BlueDot sprint

The sprint is done when all of the following hold, and not before.

**Protocol and artifacts**
* A dated preregistration amendment (or a separate BlueDot preregistration) exists, written
  before the calibration run, fixing the target, comparisons, decision rule, sample counts,
  exclusion rules, and the layer-fallback trigger.
* A hashed prompt manifest exists with exactly 8 smoke, 32 calibration, 96 training, and 32
  final-test prompts, disjoint by `group_id`.
* A `DirectionFamilyRecord` exists citing eight direction hashes, four answer-token and four
  seeded orthogonal, all built against the pinned Gemma revision at hidden dim 1152.
* A calibration artifact exists recording the chosen norm ratios and the evidence for the
  choice, written before any training fit.
* Final-test forecasts for every method and every state condition were committed and their
  commitments recorded before the final-test resolution ran, and `csf verify study` confirms the
  ordering from artifacts rather than from a claim in prose.
* Every commitment verifies. Every resolution failure is recorded and counted.

**Code and quality**
* `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pyright`
  all pass, with the test count recorded in the log.
* `TrainingExample` is unchanged and its leakage test still passes unmodified.
* Every leakage risk in section 11 has a passing test, R1 in particular.
* Every new record type has a round-trip test and appears in the run manifest by hash.
* No new placeholder CLI command exists.

**Analysis**
* Prompt-aggregated errors are reported for all five methods (constant, lexical,
  intervention-only, visible-information, bilinear) plus the matched wrong-state control.
* Every reported number carries its sample count, its prompt-group count, and a paired
  prompt-grouped bootstrap interval.
* Grouped cross-validation results on the training prompts and the single final-test result are
  reported separately and are not pooled.
* The headline comparison is stated with its decision rule applied, including the case where the
  interval spans zero, which is reported as no detected difference.

**Claims**
* `docs/claim_boundaries.md` is updated to state what this arm can and cannot say. Specifically,
  a positive result is evidence that prompt-specific state information improves prediction of a
  specified intervention's effect beyond prompt, clean output distribution, and intervention
  representation. It is not evidence about consciousness, self-awareness, faithful verbal
  introspection, hidden-goal detection, or deployment readiness.
* No number is described as measured unless it cites a verified run artifact.
* `docs/experiment_log.md` has an entry per run with its run id and the counts read from its
  manifest.

**Explicitly not required for done**
* No public export, no dashboard, no paper.
* No model organism, no LoRA adapter, no learned clean-versus-adapted direction.
* No state MLP, no verbal reporter, no SAE, no second model, no GPU.

## Scope control: move to deferred work

These are currently presented as active prerequisites by `docs/build_plan.md`,
`docs/research_plan.md`, `docs/preregistration.md`, and `docs/claim_boundaries.md`. None of them
is a prerequisite for the state-dependence study, and each one adds surface that the study does
not need. Recommend moving all of them to a clearly labeled deferred section.

| Item | Where it is presented as active | Recommendation |
| --- | --- | --- |
| Benign model organism (Phase 5) | `build_plan.md` phase table; `research_plan.md` scope; `claim_boundaries.md` has a whole section on its limits | Defer. The study uses one clean pinned model. |
| LoRA training | `build_plan.md` Phase 5; `compute_decision.md` "the open question"; `peft` in the torch extra | Defer. No adapter is needed, so the CPU-LoRA question does not block anything. |
| Learned clean-versus-adapted behavioral direction (Phase 6) | `build_plan.md`; `methodology.md` section 4 | Defer. The study's directions are constructed from the unembedding and from seeded randomness, and neither requires estimation from data. |
| Evaluation-versus-deployment framing (H6) | `preregistration.md` H6; `arc_mcq.yaml` requires all three framings; `failure_modes.md` #8 | Defer as a hypothesis. The task config's three-framing requirement can stay, since the study can simply select one wrapper per prompt in the manifest. |
| State MLP (Phase 8) | `preregistration.md` primary comparison; `methodology.md` method table | Defer. The study uses ridges, which are auditable and need no seed-variance protocol. |
| Verbal reporter and soft-token reporter (Phase 11) | `build_plan.md`; `methodology.md`; H5 | Defer. |
| SAE extension (Phase 12) | `build_plan.md`; `research_plan.md` | Defer. Already marked as not blocking v0.1. |
| Gemma 3 4B replication (Phase 13) | `build_plan.md`; `configs/models/gemma3_4b_it.yaml` exists | Defer. One pinned model. |
| Dashboard (Phase 10) | `build_plan.md`; `test_no_fake_results.py` guards it | Defer the dashboard itself. Keep the anti-fabrication tests: they cost nothing and they pass on an absent dashboard. |
| GPU rental | `build_plan.md` "Consequences of no CUDA"; `compute_decision.md`; `csf doctor` notes | Defer. Section 15 estimates the whole study at roughly 1.8 CPU hours, with a 1.5x to 2x band. `csf doctor`'s note about GPU-requiring sweeps is now misleading for this arm and should be softened when documentation work is in scope. |
| Held-out mechanism transfer (H4, activation patching) | `build_plan.md` Phase 9; `Mechanism.ACTIVATION_PATCH` is held out by default in `ExperimentConfig` | Defer the hypothesis. Leave the held-out-mechanism machinery in place; it costs nothing and removing it would weaken the existing design. |
