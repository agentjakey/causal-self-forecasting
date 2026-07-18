# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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
