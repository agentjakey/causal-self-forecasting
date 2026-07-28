# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
