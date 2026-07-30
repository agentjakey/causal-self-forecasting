# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added (2026-07-29, BlueDot slices B9, B10, and B12: final-test resolution and analysis)

**Implemented and tested. Not executed: the final test remains unresolved and no result exists.**

* `state_audit/resolve.py` and `csf state-audit resolve-final-test`, the irreversible step. 544
  intervened forwards over 32 prompts x 17 candidates, reusing the clean logits and states so no
  clean forward runs and every delta is measured against the baseline the forecasts were made
  against. `clean_forwards` on the manifest is a typed literal zero.
* Resolution guards, all before the model loads: a **dirty working tree is a hard stop** and the
  commit is recorded inside the hashed manifest; the layer, ratio, and alpha are compared against
  the calibration decision artifact; the commitment count must be exactly 512 with a forecast and a
  salt each; any pre-existing reveal or outcome artifact refuses the run; and the CLI requires
  `--yes-i-understand-this-is-irreversible`.
* All-candidate resolution reveals every commitment with a **no-selection** reveal, verifies each
  hash from its revealed salt, independently recomputes every stored target from its own logits,
  and records that every `committed_at` precedes every `observed_at`.
* `state_audit/analyze.py`, `csf state-audit analyze-final-test`, and
  `csf state-audit replay-analysis`, all model-free. Nothing is fitted, refitted, tuned, or dropped.
  Prompt-first aggregation, the two primary paired differences, 10,000 paired resamples over prompt
  groups from the frozen seed, and RMSE, sign accuracy, Spearman, top-effect ranking, flip count,
  and no-op error alongside. The Brier score is reported only at 20 or more realized flips.
* `paired_grouped_bootstrap`, `prompt_first_mean`, and `spearman_correlation` in
  `scoring/metrics.py`. The paired bootstrap draws one set of resampled groups per replicate and
  evaluates both methods on it; bootstrapping them independently would discard the pairing and
  widen the interval for no reason. This closes conflict S10 in the preregistration.
* `FinalTestResolutionManifest`, `FinalTestAnalysisRecord`, `MethodConditionSummary`, and
  `PairedComparison`. `PairedComparison` refuses to be constructed with a `supported` flag that does
  not follow from its own interval, so the decision rule cannot be applied by hand.
  `FinalTestAnalysisRecord` is the one record allowed a real `scientific_forecast_evaluation`
  boolean rather than a `Literal[False]`, and it is true only when the commitments verified.
* Machine-readable tables (`state_audit_analysis.json`, `state_audit_method_summary.json`,
  `state_audit_prompt_scores.jsonl`, `state_audit_pair_scores.jsonl`) and two figures drawn
  entirely from computed values.
* 37 unit tests and 20 fixture integration tests, including outcome-before-commitment rejection,
  the all-candidate reveal, timestamp ordering, exact candidate counts, scoring from sealed
  forecasts, prompt-first aggregation, the paired bootstrap, model-free replay, and refusal to
  rerun a completed test.

### Fixed (2026-07-29)

* The model-free replay compared the resolution's *stored* outcome hash against the analysis's
  *stored* outcome hash, which agree with each other even after the outcomes file underneath both
  has been rewritten. It now hashes the file on disk. Caught by a fixture test that edited an
  outcome row and expected the replay to notice.

### Added (2026-07-29, BlueDot slices B2b and B5 through B9: the precommitted forecasting stage)

* `state_audit/projection.py`: the fixed `1152 x 16` intervention projection. Generated once from
  the master seed, never fitted, stored, hashed, and cited by every forecast, so every method
  receives the identical `P^T v`. Injectivity is checked as **rank preservation plus retention**,
  not as an absolute singular-value floor: the answer-token family is near-degenerate by
  construction, so an absolute floor would measure the direction family's conditioning and call
  it a property of the projection.
* Commitment hardening. The key is now
  `(trial_id, method_id, state_condition, condition_index)` in the record, the duplicate check,
  and the salt filename, so the twelve state-conditioned records per prompt no longer collide and
  no salt overwrites another. `SelectionReveal` gained a **no-selection** shape for all-candidate
  runs. `commit_forecast` refuses once any outcome artifact exists, which is the only defence
  against "resolve, look, commit" that timestamps cannot provide.
  `verify_commitment_ordering` checks the sequence from artifacts.
* `state_audit/features.py`: the four blocks, at the preregistered widths, with the intervention
  block shared identically by every method and the visible block containing no state-derived
  quantity. `forecasting.TrainingExample` is untouched; this arm has its own example type.
* `state_audit/fit.py`: the three ridges, six-fold grouped cross-validation over training prompt
  groups with no group on both sides of a fold, and the preregistered alpha grid. Ties break
  toward the stronger regularizer.
* `state_audit/matching.py`: the deterministic nearest matched wrong state by the five-step rule,
  and ten seeded derangements with no fixed point.
* `state_audit/predict.py`: fitting from the training run, the frozen final-test candidate sets,
  and the full commitment pass.
* `TransformFitRecord`, `RidgeSelectionRecord`, `StateAuditPredictorRecord`,
  `WrongStatePairingRecord`, `InterventionProjectionRecord`, and `CleanPassRunManifest`, whose
  `intervention_count` is a typed literal zero.
* `csf state-audit train`, `projection`, `final-test-clean`, `commit-forecasts`, and
  `verify-commitments`. Training and final test inherit their strength from the calibration
  decision rather than recomputing it.

### Changed (2026-07-29, publication readiness)

* `README.md` rewritten as a research landing page: the question in plain English, what is and is
  not being tested, a status table of stages with no restated numbers, the study design, install,
  quick verification, the stage-by-stage run sequence, repository structure, the reproducibility
  chain, documentation index, safety, citation, and license. Every command in it was checked
  against `csf --help`.
* `docs/deferred_work.md` added: one current view of everything designed and deliberately not
  built, with the reason for each. The same list used to be restated in the research plan, the
  build plan, the preregistration, and the README; `docs/build_plan.md` and
  `docs/research_plan.md` now point at it instead of repeating it.
* `docs/bluedot/current_state_audit.md` retitled and given a superseded banner. It is a snapshot
  of 2026-07-27 whose "missing capabilities" and "required changes" sections have all since been
  built; it is kept for provenance, since the leakage finding became the global-alpha rule and the
  schema conflicts became the preregistration's conflict table.
* `.gitignore` deduplicated and reorganized by intent. It was a default Python template
  concatenated with the project rules, with several sections appearing twice. Same coverage, plus
  a statement of what is tracked on purpose and why. No tracked file changed status.
* `SECURITY.md` and `CITATION.cff` corrected: the model organism is described as designed and
  deferred rather than as something the project trains, and the citation abstract now describes
  the study that exists.
* `docs/failure_modes.md` marks which modes apply to the active arm and no longer describes a
  dashboard page that does not exist.

### Executed (2026-07-29)

* The preregistered calibration sweep at layer 13 on the pinned weights: 32 prompts, 2,624
  forwards, 2,592 observations, no failures, 41.5 minutes, verified. Status `passed_primary`,
  selected ratio 0.02 at global alpha 106.87158268272867. Ratios 0.02 and 0.05 satisfied every
  condition; 0.10, 0.20, and 0.40 failed the 95th-percentile ceiling only. The layer-20 fallback
  is now permanently closed. Calibration chooses a strength and is not a scientific result;
  measured values are in `docs/experiment_log.md`.

### Added (2026-07-28, BlueDot slice B3b: the calibration sweep)

* `state_audit/calibrate.py`: the preregistered sweep at one layer. Clean pass, reference norm
  taken as the median of the 32 clean state norms **before** any intervention runs, one global
  alpha per frozen ratio, the 81-candidate grid applied to every prompt, then the six frozen
  conditions per ratio and the smallest passing one. Execution reuses the smoke's
  `execute_candidate_pass` in full; the two differ only in the candidate shape they ask for and
  in what they do with the observations afterwards.
* `csf state-audit calibrate`. It refuses a widened or reordered grid, a third layer, a
  non-calibration prompt role, a prompt count or no-op tolerance that disagrees with the frozen
  plan, and a selected-strength config. The layer-20 fallback has its own config and is refused
  unless `--primary-run-id` names a layer-13 run whose decision record says `fallback_required`,
  so a layer-13 pass closes the fallback permanently rather than relying on discipline.
* `StateAuditRunBase`, `CalibrationRunManifest`, and `compute_calibration_run_hash`.
  `StudyRunManifest` was refactored onto the shared base without changing its hashed payload, so
  the already-executed smoke run still verifies with an identical manifest hash. A calibration
  manifest carries the grid, one alpha per ratio, the decision status, and the selection, and its
  validator re-derives every alpha from the reference norm and refuses a selection that is not on
  its own grid.
* The sweep writes a `LayerReferenceNormRecord` carrying all 32 individual norms, the five ratio
  summaries, and the hashed decision record. `csf state-audit verify-run` handles both manifest
  shapes, checks one global alpha per grid point rather than one overall, recomputes every alpha
  from the recorded clean state norms, and refuses a selected ratio that is not the smallest
  passing one at its layer.
* `StateAuditRunConfig` gained `candidate_kind` and `norm_ratios`; a config must name exactly one
  of `norm_ratio` and `norm_ratios`, and a grid must be the frozen five in order.
  `configs/state_audit/bluedot_calibration_layer13.yaml` and `..._layer20.yaml`.
* 26 offline tests covering the full fixture sweep, one alpha per ratio from one reference norm,
  the shared no-op, the decision falling mechanically out of the summaries, every refusal above,
  the fallback gate in both directions, failure preservation charged to the right grid point, and
  artifact verification of the decision and summaries.

### Added (2026-07-28, BlueDot slices B4 and B11: study candidates and the engineering smoke)

* `state_audit/candidates.py`: two explicit builders rather than one with a flag.
  `selected_strength_templates` gives 8 directions x 2 signs at one ratio plus a no-op (17),
  which smoke, training, and final test use; `calibration_grid_templates` gives 8 x 2 x 5 ratios
  plus one shared no-op (81), which only calibration uses. Order is a seeded shuffle and opaque
  ids are assigned after it, so position encodes nothing.
* `StateAuditCandidate` and `StateAuditCandidateSet`. A candidate cites the opaque direction id,
  that direction's vector hash, the sign, the layer, the ratio, the global alpha, and a hash of
  the signed intervention itself; the set cites the family hash and its own content hash. The
  record type refuses an id or a direction reference that names a construction role, an answer
  label, a random-control label, or a semantic family, and the set validator refuses more than
  one alpha at a ratio, a missing sign, a missing no-op, or a five-ratio grid handed to a
  selected-strength role. The public view narrows further to operation, layer, position, and
  strength, so a control and a steer are indistinguishable.
* `state_audit/run.py`: the real execution path. Clean forward and layer capture per prompt, one
  global alpha frozen from the median clean state norm, all 17 candidates applied, the target
  computed and stored, and every failure preserved in its own artifact. It reuses the one
  validated inference path throughout, converting each study candidate into the harness's own
  `InterventionSpec` rather than adding a second way to run a forward.
* `state_audit/verify.py`: artifact-only verification. Recomputes the manifest's content hash,
  every artifact hash, every observation's target from its own logits, the reference norm from
  the recorded clean state norms, and the single global alpha. `compare_runs` compares two runs
  of the same inputs row by row, which is how cross-process determinism is measured. A test
  asserts the module names no model-loading or forward-pass function at all.
* `StudyRunRole` (`engineering_smoke`, `calibration`, `training`, `final_test_unresolved`,
  `final_test_resolved`), `StudyRunManifest`, `StateAuditCleanPassRecord`, and
  `StateAuditRunDiagnostics`. The run manifest derives its `status` rather than asserting it: it
  refuses to call itself complete while a count is short or a failure was recorded, so a run that
  went wrong stays visibly wrong. `scientific_result` is `Literal[False]`.
* `StateAuditObservationRecord` now records `pre_norm`, `post_norm`, and `delta_norm`, and
  refuses a no-op that displaced the residual stream at all.
* `csf state-audit smoke` and `csf state-audit verify-run`. The smoke command refuses any run
  role but the engineering smoke, any prompt role but `smoke`, any layer but 13, any ratio but
  the preregistered arbitrary 0.10, role counts that disagree with the frozen manifest, a hidden
  dimension that disagrees with the direction family, a dirty direction family, a prompt manifest
  that no longer matches the prepared task, and a completed run at the same run id.
  `verify-run` loads no model.
* `StateAuditRunConfig` and `configs/state_audit/bluedot_smoke.yaml`; `csf doctor` validates
  `configs/state_audit`.
* `csf calibration summarize` now refuses observations whose prompt role is not `calibration`, so
  smoke, training, or final-test effect sizes cannot choose the study's intervention strength.
* 102 offline tests covering both candidate shapes, opacity and leakage, ordering and hash
  stability, the run and clean-pass record validators, every config and command refusal, failure
  preservation, artifact verification, a full eight-prompt fixture smoke on a locally built
  14-block model, a deterministic fixture rerun, and an assertion that the benchmark's
  four-candidate builder is unchanged.

### Executed (2026-07-28)

* The eight-prompt engineering smoke on the pinned `google/gemma-3-1b-it` at
  `dcc83ea841ab6100d6b47a070329e1ba4cf78752`: 144 forwards, 136 observations, no failures,
  verified, `scientific_result: false`. It is plumbing validation, not calibration; the ratio and
  layer were both fixed in advance and its effect sizes select nothing. Measured values are in
  `docs/experiment_log.md`.

### Added (2026-07-28, BlueDot slice B3a: study target and calibration rules)

* `state_audit_target.py`: the arm's target `delta_clean_top_margin`, as pure float64 functions
  for selecting the clean preferred label (ties broken in fixed A, B, C, D order), computing a
  margin around a caller-supplied label, computing the complete target, and independently
  verifying a stored target from saved logits. The benchmark's `delta_margin` is untouched and
  keeps its dataset-correct-answer meaning; the module lives at the package root beside
  `hashing.py` because `schemas.py` needs it and importing it through `scoring/` would cycle.
* `StateAuditObservationRecord`, a separate record from `ObservationRecord`. Its validator
  recomputes the label, both margins, the delta, and the flip from the logits stored beside
  them, so a stored target that disagrees with its own logits does not load. It also refuses a
  direction reference that names a construction role, and requires a zero strength for a no-op.
* `calibration/strength.py`: the global rule, `alpha = ratio * median(clean state norm over the
  32 calibration prompts)`, with norms aligned by prompt identity rather than input order. There
  is deliberately no prompt-specific strength function, and `check_global_alpha` refuses a grid
  point that used more than one alpha.
* `calibration/criteria.py`: the six frozen pass conditions, inclusive at the boundary, with
  `numpy.quantile(method="linear")` percentiles recorded in the plan.
* `calibration/selection.py`: the layer-13-primary, layer-20-fallback state machine. Smallest
  passing ratio in preregistered order; a passing primary layer prohibits the fallback; a
  partial or widened ratio grid is refused.
* `calibration/plan.py` and `calibration/observations.py`: the frozen plan record with its own
  content hash, the encoded forward arithmetic, artifact-only plan verification, and the glue
  that turns supplied observations into ratio summaries and a hashed decision record.
* `CalibrationPlanRecord`, `LayerReferenceNormRecord`, `CalibrationRatioSummary`,
  `CalibrationDecisionRecord`, `CalibrationThresholds`, `CalibrationForwardCounts`, and
  `CalibrationCriterionResult`, all carrying `scientific_result: Literal[False]` where they
  could be mistaken for a finding.
* `csf calibration plan`, `verify-plan`, `summarize`, and `select`. None loads a model, and a
  test replaces `load_model` with a function that raises to prove it.
* `CalibrationPlanConfig` and `configs/calibration/bluedot_state_dependence.yaml`. The config
  refuses a third layer, an altered or reordered ratio grid, a target other than
  `delta_clean_top_margin`, and incomplete role counts; the plan builder additionally refuses a
  prompt manifest without 32 calibration prompts, a direction family without eight directions,
  and a direction family built against a different model revision. `csf doctor` validates
  `configs/calibration`.
* `paths.calibration_plans_dir` and `paths.calibration_plan_path`.
* 143 offline tests covering the target and its boundary cases, the record validators, the
  strength rule, every pass condition at its exact threshold, the state machine including
  fallback prohibition and total failure, plan and decision hashing, and the CLI.

### Fixed (2026-07-28)

* `check_global_alpha` compared alphas through a mapping keyed by prompt id. With 16 candidates
  per prompt, a single tampered alpha was overwritten by its neighbours and disappeared. It now
  compares every observation. Caught by a CLI test that tampered with one record.
* Calibration planning transitively imported torch through the direction-family module, so a
  command that loads no model still paid a multi-second import. The import is now lazy.

### Measured (infrastructure, not scientific)

* The calibration plan was frozen to
  `data/calibration_plans/bluedot_state_dependence_calibration_v1.json`, plan hash
  `sha256:a212c6e80f96db55e1aeb0b1879fef441aa6d893a73b7fc43a94142913f97877`. Encoded forward
  counts: 144 smoke, 2,624 calibration per layer, 1,728 training, 576 final test, 5,072 primary
  total, 7,696 with the fallback. No model was loaded, no prompt was run, no state norm was
  measured, and no ratio has been selected.

### Added (2026-07-27, BlueDot slice B2a: the deterministic direction family)

* `LoadedModel.output_embedding()`: the smallest safe accessor for the output-embedding matrix,
  reading the already-loaded model rather than adding a second loading path. It prefers the
  documented `get_output_embeddings()`, falls back to the input embedding only when no output
  embedding exists, checks that the matrix is 2-D with a hidden axis matching the loaded model,
  checks that every required token id is in range with finite values, and reports tying as a
  fact read off the tensors rather than trusted from the config. Both the observed and the
  declared tying status are recorded so a disagreement is visible.
* `DirectionConstructionRole`, `DirectionTolerances`, `DirectionEntry`,
  `DirectionFamilyDiagnostics`, and `DirectionFamilyRecord`. The record recomputes its own
  content hash on load, so an edited or reordered manifest fails to parse. The hash covers
  identity, provenance, and per-vector content hashes; it excludes creation metadata, the
  config path, the raw float diagnostics, and the `.npz` container hashes, which depend on the
  archive writer rather than on the numbers.
* `interventions/direction_family.py`: four centered answer-token directions
  (`d_c = normalize(w_c - mean(w_j for j != c))`), a deterministic two-pass modified
  Gram-Schmidt basis for their span, and four seeded PCG64 Gaussian controls projected off that
  span and off each other, redrawn if degenerate, sign-canonicalized, and unit-normalized.
  Construction and validation run in float64; artifacts are stored as float32 through the
  existing `DirectionStore`. Modified Gram-Schmidt is used rather than QR or SVD because the
  answer family is rank-deficient by construction and those routines can pick different bases
  and signs across library versions.
* `csf directions build-family` and `csf directions verify-family`, the latter with an
  artifact-only level that loads no model and an optional `--regenerate` level that rebuilds
  every direction from the pinned weights and writes nothing.
* `DirectionFamilyConfig` and `configs/directions/bluedot_state_dependence.yaml`. Answer token
  ids are always resolved through the existing scoring path; the config's `expected_token_ids`
  are assertions that refuse the build on mismatch. `csf doctor` now validates
  `configs/directions`.
* `paths.direction_manifests_dir` and `paths.direction_manifest_path`.
* 74 offline tests covering the accessor, token-id resolution and refusal, the centering
  formula, the sum-to-zero property, rank handling, Gram-Schmidt determinism, seeded control
  determinism, orthogonality, sign canonicalization, degenerate-draw redraw, hash sensitivity,
  schema round-trip and tamper evidence, atomic generation, overwrite refusal, artifact and
  regeneration verification, the CLI, and the privacy boundary.

### Fixed (2026-07-27)

* Direction `.npz` metadata no longer carries the construction-algorithm version. Its name
  contains `answer_token_centered`, which would have put a family-role term into the payload
  store that resolution reads from. The algorithm version is recorded in the private manifest,
  which is the authoritative provenance. Caught by the privacy-boundary test.

### Measured (infrastructure, not scientific)

* The BlueDot direction family was built from the pinned Gemma 3 1B unembedding and frozen to
  `data/direction_manifests/bluedot_state_dependence_directions_v1.json`, family hash
  `sha256:809fbb5b033da740a01574ad5a0ca48f34baca38eef1504a0d66bba8e2fb9138`: eight unit
  directions, answer-span rank 3, worst answer-to-random dot 2.82e-09, zero redraws. No prompt
  was run, no state was captured, no intervention was applied, and this is not causal
  validation. Values are in `docs/experiment_log.md`.

### Added (2026-07-27, BlueDot slice B1: deterministic prompt manifests)

* `PromptRole` (`smoke`, `calibration`, `training`, `final_test`), plus the `PromptAssignment`
  and `PromptManifest` records. `PromptRole` is additive: the existing `Split` enum and
  `tasks/splitting.py` are untouched, and a prompt now carries both. `PromptManifest` recomputes
  its own content hash on load, so an edited, reordered, or recounted manifest fails to parse.
  The hash deliberately excludes the manifest's filesystem path and creation timestamp, so the
  same selection hashes the same after the repository moves.
* `tasks/prompt_manifest.py`: deterministic role assignment as a pure function of the master
  seed and the group ids, through the existing `derive_seed` with the label
  `bluedot.prompt_manifest`, over a lexicographically sorted pool with the group id as
  tie-break. Selection never reads model correctness, confidence, logits, hidden states,
  intervention effects, gold-label balance, question topic, or input file order. It refuses a
  pool that is too small, a group with no or more than one canonical variant, duplicate group or
  item ids, a wrapper the task does not define, a held-out wrapper, and a manifest whose task
  artifacts have moved.
* `csf prompts manifest` and `csf prompts verify`. Rerunning `manifest` with identical inputs
  leaves an identical file byte-identical and reports `unchanged` rather than rewriting it, and
  a different manifest at the same path is refused unless `--force` is passed.
* `PromptManifestConfig` and `configs/prompts/bluedot_state_dependence.yaml`, carrying the
  preregistered defaults. `csf doctor` now validates `configs/prompts` as well.
* `paths.prompt_manifests_dir` and `paths.prompt_manifest_path`, so the output location is owned
  in one place.
* 53 offline tests covering counts, coverage, group and item disjointness, determinism,
  selection blindness, every named failure mode, atomic write behavior, overwrite refusal,
  schema round-trip, tamper evidence, and the CLI.

### Measured (infrastructure, not scientific)

* The BlueDot prompt split was frozen to
  `data/prompt_manifests/bluedot_state_dependence_v1.json`, manifest hash
  `sha256:bf351c9d73042fcb3d0cdcb18247ded3411413d73000e047f1ba6ba9ab5b25b9`: 168 prompts, all on
  wrapper `neutral_a`, drawn from 256 eligible groups at seed 20260727. No model was loaded and
  no forward pass ran. Values and disjointness evidence are in `docs/experiment_log.md`.

### Added (2026-07-27, documentation and scope only)

* `docs/bluedot/current_state_audit.md`: a read-first audit of the checkout, recording verified
  capabilities, missing capabilities, leakage and commitment-protocol risks, and eleven
  code-versus-documentation discrepancies.
* `docs/bluedot/preregistration_state_dependence.md`: a dated preregistration for the BlueDot
  state-dependence arm, written before any calibration, training, or final-test run. It freezes
  the model and revision, the 8 / 32 / 96 / 32 prompt roles, the layer and the single allowed
  fallback, a new target `delta_clean_top_margin`, the eight-direction bank, the global
  calibrated intervention magnitude, three ridge methods, the wrong-state control, the
  commitment protocol, the analysis, and the predicted results. It supersedes
  `docs/preregistration.md` **for that arm only**; the original is unedited and still governs the
  broader CSF-Bench study.
* `docs/bluedot/execution_decision_tree.md`: the stage-by-stage execution order with a written
  gate at every branch point and six named stop codes.

### Changed (2026-07-27)

* `docs/preregistration.md`: a scope note and a dated amendment-log entry pointing at the BlueDot
  arm. No hypothesis, outcome, comparison, decision rule, or planned count was altered.
* `docs/build_plan.md`: phase table now records which phases are active for the BlueDot arm and
  which are deferred; a BlueDot slice table was added; the duplicated ordering list was fixed;
  the stale Phase 1 test count is now marked historical alongside the current 273 passed,
  1 skipped.
* `docs/methodology.md`: the artifact list now matches what runs actually write; the
  `prompt_tfidf` versus implemented `prompt_lexical` naming is reconciled; a section on the
  BlueDot arm's target, magnitude, direction construction, and aggregation was added.
* `docs/claim_boundaries.md`: the status section no longer understates the completed real-Gemma
  engineering validation, and a claim-boundary section for the BlueDot arm was added.
* `docs/compute_decision.md`: corrected arithmetic for the BlueDot arm (5,072 forwards, about
  40 minutes, about 61 with the layer-20 fallback), stating that no GPU rental and no compute
  grant is needed. LoRA training is marked deferred for this arm.
* `README.md`: added the BlueDot arm to the status section and corrected the quickstart, which
  previously ended with `csf score run` immediately after a ground-truth resolve. That sequence
  cannot work: scoring requires committed forecasts and no CLI command commits one.

### Fixed (2026-07-27, message text only, no behavior change)

* Removed references to `csf directions estimate`, a command that does not exist, from
  `trials/generate.py`, `interventions/validate.py`, `cli.py`, and
  `configs/experiments/smoke.yaml`. The messages now say what does exist and state plainly that
  data-estimated direction discovery is not implemented.
* Documented that runs write `observations.jsonl`, not the `observations.parquet` the methodology
  previously listed. `paths.OBSERVATIONS` is an unused long-term target name and is now described
  as one.

### Added

* `csf trials resolve`: apply interventions and record observations for a generated trial run.
  Forecast mode requires committed forecasts, selects one candidate per trial after commitment,
  reveals the salt, and verifies the commitment. Ground-truth mode applies every candidate and
  records observations only. Both refuse to resolve a systems benchmark, refuse to treat a
  fixture run as scientific, validate commitments before running, guard against a model that
  does not reproduce the committed clean baseline, and preserve failed interventions with
  explicit error records rather than dropping them.
* `csf score run`: match committed forecasts to observations and aggregate. Numeric metrics
  (MAE, RMSE, sign accuracy, interval coverage), flip metrics (Brier, log loss), and
  largest-effect candidate ranking, with no-op candidates excluded from the headline and
  reported separately. Every metric carries its sample count and a group-bootstrapped interval.
  Scores are written to the run directory only and never become a public result; a scientific
  run whose commitments did not verify is refused.
* Two model-organism-free baselines: a constant predictor (training-split averages by public
  operation, layer, and strength) and a prompt-only lexical model (TF-IDF plus public strength
  and layer, ridge for delta and logistic for flip). Both are fenced to public information by
  construction, with a leakage audit test on the training-example type and the forecaster
  interface.
* `ComputeEstimate` planning arithmetic and `docs/compute_decision.md`, derived from the
  measured Gemma forward time. LoRA training on CPU is classified insufficient evidence.
* `configs/experiments/gemma_smoke.yaml` and its intervention config, for the smallest
  real-model intervention harness validation.

### Measured (systems, not scientific)

* Real Gemma 3 1B systems benchmark completed on CPU: median forward 0.474 s, model load
  48.1 s, capture verified at layer 13 with zero patch error. Values in
  `docs/experiment_log.md`, read from the artifact, classified `systems_benchmark`.
* Real Gemma intervention harness validation: all required controls pass on real weights, and
  ground-truth resolution recorded four observations (no-op, positive, negative, random) on one
  ARC item with the no-op reproducing the clean output exactly. Non-scientific: the direction is
  synthetic and unvalidated.

### Added (earlier this cycle)

* `csf benchmark`: a systems and clean-model sanity benchmark for a real pretrained model. It
  loads the pinned weights, scores multiple-choice items directly from their answer-token
  logits, measures CPU load and forward-pass cost, reports process memory or an explicit null,
  and verifies the hook-owned capture path on real weights by patching a layer and reading it
  back. Hugging Face access failures are classified separately (no authentication,
  authenticated without the Gemma license, unavailable revision, network failure, offline
  cache miss, insufficient disk, load failure) because their remedies differ.
* `BenchmarkRecord` and its supporting schemas. `scientific_result` is typed `Literal[False]`,
  so a benchmark artifact that claims to be a scientific result cannot be constructed. Fixture
  runs are classified `fixture_systems_test`; real runs are `systems_benchmark`. Neither can
  pass through the public exporter, which requires verified commitments a benchmark does not
  have.
* A transparent compute estimate derived from the measured median forward time, carrying its
  assumptions and their source (`docs/preregistration.md` section 9) in the artifact.
* `load_prepared_task` in `tasks/loader.py`, now the single reader for prepared data, used by
  both trial generation and the benchmark so the two cannot drift apart.

* Phase 0 scientific contract: research plan, preregistration, methodology, claim boundaries,
  failure-mode taxonomy, experiment log.
* Phase 1 scaffold: `csf` CLI, Pydantic schemas for every artifact type, canonical JSON and
  SHA-256 hashing, atomic writes, structured logging, deterministic seed derivation, and
  environment capture.
* Phase 2 model harness: device and dtype resolution, model loading with pinned revisions,
  single-token answer-label scoring, residual-stream capture. A locally built, randomly
  initialized fixture model so the pipeline runs offline with no gated weights.
* Phase 3 intervention system: no-op, residual addition, projection ablation, norm-matched
  random controls, activation-patch interface, shape validation, norm diagnostics, and the
  `csf interventions validate` control suite.
* Phase 4 (partial) trial and commitment engine: candidate sets with opaque ids and randomized
  order, salted commitments, order-enforced selection seeds, deterministic per-trial selection,
  reveal and verification, `csf verify run`.
* Task pipeline: ARC-Challenge and MMLU loaders, group-aware deterministic splitting, wrapper
  paraphrase rendering with held-out phrasings.
* Anti-fabrication guards: `tests/test_no_fake_results.py`, plus schema-level refusal to export
  unverified runs or metrics without sample counts.

### Fixed

* Hugging Face network failures are classified instead of escaping as raw tracebacks.
  `huggingface_hub` 1.x uses httpx, and `httpx.HTTPError` is not an `OSError`, so an
  `OSError`-only except clause missed the failure most likely to happen during a
  two-gigabyte download. A connection dropped mid-download is also no longer reported as a
  model load failure, which would have sent a reader to debug their weights instead of
  retrying.
* The hub metadata probe retries transport failures three times with linear backoff. The
  connection from the development machine was observed failing and then succeeding on the
  next attempt with no other change, and reporting that as a network failure would be
  misleading. Access answers such as a gated repo or a missing revision are not retried.
* Capture no longer relies on the transformers `output_hidden_states` flag. Under transformers
  5.14 the framework records hidden states with internal hooks whose ordering against a user
  intervention hook is not part of the public API, and an activation patch read back as its
  pre-intervention value while the forward pass really had been modified. Capture now uses this
  project's own hooks, pinned by `test_capture_matches_the_intervention_point`.
* `.gitignore` no longer excludes `src/causal_self_forecasting/models/` via a bare `models/`
  pattern.
* Trial subsampling no longer truncates a sorted list, which put every capped run into a single
  split covering one or two questions.

### Notes

No experimental results exist. No model organism, estimated direction, trained forecaster, or
verified export. See `docs/experiment_log.md`.
