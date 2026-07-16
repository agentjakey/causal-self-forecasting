# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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
