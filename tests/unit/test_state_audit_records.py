"""Unit tests for the state-dependence run records and their config guards.

No model, no prompts, no forward passes. These pin the properties that make a run manifest worth
reading: it recomputes its own hash, it refuses to call an unfinished run complete, and it
cannot be constructed as a scientific result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from causal_self_forecasting.config import ConfigError, StateAuditRunConfig, load_config
from causal_self_forecasting.schemas import (
    PromptRole,
    StateAuditCleanPassRecord,
    StateAuditObservationRecord,
    StateAuditRunDiagnostics,
    StudyRunManifest,
    StudyRunRole,
    compute_study_run_hash,
)
from causal_self_forecasting.state_audit.run import StateAuditRunError, check_smoke_parameters

CLEAN = {"A": 3.0, "B": 1.0, "C": 0.5, "D": 0.0}
HASH = "sha256:" + "0" * 64

REAL_SMOKE_CONFIG = "configs/state_audit/bluedot_smoke.yaml"


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def observation(**overrides: Any) -> StateAuditObservationRecord:
    delta = overrides.pop("delta", 0.5)
    is_noop = overrides.pop("is_noop", False)
    intervened = dict(CLEAN) | {"A": CLEAN["A"] + delta}
    body: dict[str, Any] = {
        "study_id": "bluedot_state_dependence",
        "run_id": "fixture-run",
        "trial_id": "sa_00000",
        "candidate_id": "sa_00000.opaque_03",
        "group_id": "g0",
        "variant_id": "g0.neutral_a",
        "prompt_role": PromptRole.SMOKE,
        "clean_preferred_label": "A",
        "clean_logits": dict(CLEAN),
        "intervened_logits": intervened,
        "clean_top_margin": 2.0,
        "intervened_top_margin": 2.0 + delta,
        "delta_clean_top_margin": delta,
        "answer_flip": (2.0 + delta) < 0.0,
        "is_noop": is_noop,
        "direction_ref": "noop" if is_noop else "bd1.0123456789abcdef",
        "direction_vector_hash": None if is_noop else HASH,
        "layer": 13,
        "norm_ratio": 0.0 if is_noop else 0.10,
        "global_alpha": 0.0 if is_noop else 3.2,
        "pre_norm": 30.0,
        "post_norm": 31.0,
        "delta_norm": 0.0 if is_noop else 3.2,
        "model_id": "google/gemma-3-1b-it",
        "model_revision": "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        "prompt_manifest_hash": HASH,
        "direction_family_hash": HASH,
        "config_hash": HASH,
    }
    body.update(overrides)
    return StateAuditObservationRecord(**body)


def test_an_observation_round_trips_through_json() -> None:
    original = observation()
    reloaded = StateAuditObservationRecord.model_validate(
        json.loads(json.dumps(original.model_dump(mode="json")))
    )
    assert reloaded == original


def test_an_observation_recomputes_its_target_from_its_own_logits() -> None:
    record = observation(delta=-0.75)
    assert record.delta_clean_top_margin == pytest.approx(-0.75)
    assert record.intervened_top_margin == pytest.approx(1.25)
    assert record.answer_flip is False


def test_an_observation_whose_target_was_edited_does_not_load() -> None:
    dumped = observation().model_dump(mode="json")
    dumped["delta_clean_top_margin"] = 9.0
    with pytest.raises(ValidationError, match="disagrees with its own"):
        StateAuditObservationRecord.model_validate(dumped)


def test_a_noop_observation_records_a_zero_target_and_zero_displacement() -> None:
    record = observation(delta=0.0, is_noop=True)
    assert record.delta_clean_top_margin == 0.0
    assert record.delta_norm == 0.0
    assert record.answer_flip is False


def test_a_noop_that_moved_the_residual_stream_is_refused() -> None:
    with pytest.raises(ValidationError, match="no-op adds nothing"):
        observation(delta=0.0, is_noop=True, delta_norm=1e-6)


def test_a_negative_norm_is_refused() -> None:
    with pytest.raises(ValidationError, match="finite and non-negative"):
        observation(pre_norm=-1.0)


def test_an_answer_flip_is_recorded_when_the_clean_preference_is_overtaken() -> None:
    record = observation(delta=-3.0)
    assert record.intervened_top_margin == pytest.approx(-1.0)
    assert record.answer_flip is True


# ---------------------------------------------------------------------------
# Clean pass
# ---------------------------------------------------------------------------


def clean_pass(**overrides: Any) -> StateAuditCleanPassRecord:
    body: dict[str, Any] = {
        "study_id": "bluedot_state_dependence",
        "run_id": "fixture-run",
        "trial_id": "sa_00000",
        "variant_id": "g0.neutral_a",
        "group_id": "g0",
        "item_id": "g0",
        "prompt_role": PromptRole.SMOKE,
        "prompt_hash": HASH,
        "prompt_token_count": 42,
        "position_index": -1,
        "position_absolute": 41,
        "clean_logits": dict(CLEAN),
        "clean_preferred_label": "A",
        "clean_top_margin": 2.0,
        "clean_entropy": 0.5,
        "dataset_answer_label": "A",
        "clean_correct": True,
        "layer": 13,
        "state_id": "g0.neutral_a.L13",
        "state_dim": 1152,
        "state_norm": 30.0,
        "state_shard_hash": HASH,
        "model_id": "google/gemma-3-1b-it",
        "model_revision": "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        "prompt_manifest_hash": HASH,
    }
    body.update(overrides)
    return StateAuditCleanPassRecord(**body)


def test_a_clean_pass_record_recomputes_its_preferred_answer_and_margin() -> None:
    record = clean_pass()
    assert record.clean_preferred_label == "A"
    assert record.clean_top_margin == pytest.approx(2.0)
    assert record.scientific_result is False


def test_a_clean_pass_record_with_an_edited_margin_does_not_load() -> None:
    dumped = clean_pass().model_dump(mode="json")
    dumped["clean_top_margin"] = 5.0
    with pytest.raises(ValidationError, match="clean_top_margin"):
        StateAuditCleanPassRecord.model_validate(dumped)


def test_a_clean_pass_record_refuses_an_inconsistent_correctness_flag() -> None:
    with pytest.raises(ValidationError, match="clean_correct disagrees"):
        clean_pass(dataset_answer_label="B", clean_correct=True)


def test_a_clean_pass_record_carries_the_state_provenance() -> None:
    record = clean_pass()
    assert record.state_shard_hash == HASH
    assert record.state_id.endswith(".L13")
    assert record.state_dim == 1152


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------


def diagnostics(**overrides: Any) -> StateAuditRunDiagnostics:
    body: dict[str, Any] = {
        "max_abs_noop_target": 0.0,
        "max_abs_noop_delta_norm": 0.0,
        "max_intervention_reconstruction_error": 0.0,
        "min_target": -1.0,
        "max_target": 1.0,
        "median_abs_target": 0.4,
        "p95_abs_target": 0.9,
        "fraction_above_effect_threshold": 0.5,
        "effect_threshold": 0.10,
        "flip_count": 2,
        "capture_hooks_fired": 8,
        "intervention_hooks_fired": 136,
        "state_dim": 1152,
        "min_state_norm": 20.0,
        "max_state_norm": 40.0,
        "median_method": "numpy.median",
        "percentile_method": "numpy.quantile(method='linear')",
    }
    body.update(overrides)
    return StateAuditRunDiagnostics(**body)


def manifest_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": StudyRunManifest.model_fields["schema_version"].default,
        "study_id": "bluedot_state_dependence",
        "run_id": "bluedot-smoke-layer13",
        "run_role": StudyRunRole.ENGINEERING_SMOKE.value,
        "model_id": "google/gemma-3-1b-it",
        "model_revision": "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        "tokenizer_revision": "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        "dtype": "float32",
        "device": "cpu",
        "target_name": "delta_clean_top_margin",
        "prompt_manifest_id": "bluedot_state_dependence_v1",
        "prompt_manifest_hash": HASH,
        "prompt_role": PromptRole.SMOKE.value,
        "direction_family_id": "fam_v1",
        "direction_family_hash": HASH,
        "calibration_plan_id": "plan_v1",
        "calibration_plan_hash": HASH,
        "layer": 13,
        "capture_position": -1,
        "norm_ratio": 0.10,
        "reference_norm": 32.0,
        "global_alpha": 3.2000000000000006,
        "reference_norm_source": "median clean state norm over the smoke prompts",
        "expected_prompt_count": 8,
        "expected_candidates_per_prompt": 17,
        "expected_non_noop_observations": 128,
        "expected_noop_observations": 8,
        "expected_forward_count": 144,
        "observed_prompt_count": 8,
        "observed_state_count": 8,
        "observed_non_noop_observations": 128,
        "observed_noop_observations": 8,
        "observed_forward_count": 144,
        "failure_count": 0,
        "clean_scored_count": 8,
        "clean_correct_count": 3,
        "clean_accuracy_descriptive": 3 / 8,
        "diagnostics": diagnostics().model_dump(mode="json"),
        "observations_hash": HASH,
        "failures_hash": None,
        "states_hash": HASH,
        "candidate_sets_hash": HASH,
        "clean_pass_hash": HASH,
        "input_fingerprint": HASH,
        "config_hash": HASH,
        "status": "complete",
    }
    body.update(overrides)
    return body


def run_manifest(**overrides: Any) -> StudyRunManifest:
    body = manifest_body(**overrides)
    fields = {key: value for key, value in body.items() if key != "diagnostics"}
    return StudyRunManifest(
        **fields,
        diagnostics=StateAuditRunDiagnostics.model_validate(body["diagnostics"]),
        manifest_hash=compute_study_run_hash(body),
        config_path="configs/state_audit/bluedot_smoke.yaml",
    )


def test_a_complete_run_manifest_validates_and_is_never_scientific() -> None:
    manifest = run_manifest()
    assert manifest.status == "complete"
    assert manifest.scientific_result is False
    assert manifest.run_role is StudyRunRole.ENGINEERING_SMOKE
    assert manifest.clean_accuracy_descriptive == pytest.approx(0.375)


def test_every_study_run_role_exists() -> None:
    assert {role.value for role in StudyRunRole} == {
        "engineering_smoke",
        "calibration",
        "training",
        "final_test_unresolved",
        "final_test_resolved",
    }


@pytest.mark.parametrize("role", list(StudyRunRole))
def test_no_study_run_manifest_can_claim_a_scientific_result(role: StudyRunRole) -> None:
    manifest = run_manifest(run_role=role.value)
    assert manifest.scientific_result is False
    with pytest.raises(ValidationError):
        StudyRunManifest.model_validate(
            manifest.model_dump(mode="json") | {"scientific_result": True}
        )


def test_a_manifest_with_a_recorded_failure_may_not_call_itself_complete() -> None:
    with pytest.raises(ValidationError, match="must stay marked failed"):
        run_manifest(failure_count=1)


def test_a_manifest_with_a_short_count_may_not_call_itself_complete() -> None:
    with pytest.raises(ValidationError, match="must stay marked failed"):
        run_manifest(observed_non_noop_observations=127)


def test_a_finished_run_may_not_be_filed_as_a_failure() -> None:
    with pytest.raises(ValidationError, match="must not be filed as a failure"):
        run_manifest(status="failed")


def test_a_failed_run_records_what_actually_happened() -> None:
    manifest = run_manifest(
        status="failed",
        failure_count=2,
        observed_non_noop_observations=126,
        observed_forward_count=142,
    )
    assert manifest.status == "failed"
    assert manifest.failure_count == 2


def test_the_forward_arithmetic_must_add_up() -> None:
    with pytest.raises(ValidationError, match="does not add up"):
        run_manifest(expected_forward_count=200, observed_forward_count=200)


def test_the_alpha_must_be_the_ratio_times_the_reference_norm() -> None:
    with pytest.raises(ValidationError, match="norm_ratio \\* reference_norm"):
        run_manifest(global_alpha=9.9)


def test_clean_accuracy_must_match_its_counts() -> None:
    with pytest.raises(ValidationError, match="clean_accuracy_descriptive"):
        run_manifest(clean_accuracy_descriptive=0.9)


def test_an_edited_run_manifest_does_not_load() -> None:
    dumped = run_manifest().model_dump(mode="json")
    dumped["observations_hash"] = "sha256:" + "9" * 64
    with pytest.raises(ValidationError, match="does not match the run contents"):
        StudyRunManifest.model_validate(dumped)

    diagnostics_edit = run_manifest().model_dump(mode="json")
    diagnostics_edit["diagnostics"]["max_abs_noop_target"] = 0.5
    with pytest.raises(ValidationError, match="does not match the run contents"):
        StudyRunManifest.model_validate(diagnostics_edit)


def test_the_manifest_hash_ignores_the_environment_and_timestamps() -> None:
    manifest = run_manifest()
    dumped = manifest.model_dump(mode="json")
    dumped["environment"] = {"packages": {"torch": "9.9"}}
    dumped["completed_at"] = "2001-01-01T00:00:00Z"
    dumped["code_commit"] = "deadbeef"
    reloaded = StudyRunManifest.model_validate(dumped)
    assert reloaded.manifest_hash == manifest.manifest_hash


# ---------------------------------------------------------------------------
# Config guards
# ---------------------------------------------------------------------------


def test_the_real_smoke_config_validates() -> None:
    config = load_config(REAL_SMOKE_CONFIG, StateAuditRunConfig)
    assert config.run_role is StudyRunRole.ENGINEERING_SMOKE
    assert config.prompt_role is PromptRole.SMOKE
    assert config.layer == 13
    assert config.norm_ratio == 0.10
    assert config.expected_prompt_count == 8
    assert config.expected_hidden_dim == 1152
    assert config.candidates_per_prompt == 17
    assert config.expected_forward_count == 144
    check_smoke_parameters(config)


def _write_config(tmp_path: Path, **overrides: Any) -> Path:
    body: dict[str, Any] = {
        "name": "unit_smoke",
        "study_id": "bluedot_state_dependence",
        "run_role": "engineering_smoke",
        "prompt_role": "smoke",
        "target": "delta_clean_top_margin",
        "model_ref": "configs/models/fixture_tiny.yaml",
        "task_ref": "configs/tasks/arc_mcq.yaml",
        "prompt_manifest_id": "m",
        "direction_family_id": "f",
        "calibration_plan_id": "p",
        "layer": 13,
        "capture_position": -1,
        "norm_ratio": 0.10,
        "expected_prompt_count": 8,
        "expected_direction_count": 8,
        "expected_signed_directions": 16,
        "expected_hidden_dim": 64,
        "expected_role_counts": {"smoke": 8, "calibration": 1, "training": 1, "final_test": 1},
        "master_seed": 20260727,
        "noop_tolerance": 1.0e-3,
    }
    body.update(overrides)
    path = tmp_path / "run.yaml"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_a_layer_off_the_preregistered_pair_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="preregistered layers"):
        load_config(_write_config(tmp_path, layer=17), StateAuditRunConfig)


def test_a_ratio_off_the_frozen_grid_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="frozen grid"):
        load_config(_write_config(tmp_path, norm_ratio=0.07), StateAuditRunConfig)


def test_a_different_target_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="target must be"):
        load_config(_write_config(tmp_path, target="delta_margin"), StateAuditRunConfig)


def test_a_prompt_count_that_disagrees_with_the_role_counts_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="expected_prompt_count"):
        load_config(_write_config(tmp_path, expected_prompt_count=9), StateAuditRunConfig)


def test_signed_directions_must_be_two_per_direction(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="two per direction"):
        load_config(_write_config(tmp_path, expected_signed_directions=8), StateAuditRunConfig)


def test_the_smoke_command_refuses_the_fallback_layer(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, layer=20), StateAuditRunConfig)
    with pytest.raises(StateAuditRunError, match="fixed at layer 13"):
        check_smoke_parameters(config)


def test_the_smoke_command_refuses_another_grid_ratio(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, norm_ratio=0.20), StateAuditRunConfig)
    with pytest.raises(StateAuditRunError, match="chosen in advance"):
        check_smoke_parameters(config)


def test_the_smoke_command_refuses_a_non_smoke_prompt_role(tmp_path: Path) -> None:
    config = load_config(
        _write_config(
            tmp_path,
            prompt_role="calibration",
            expected_prompt_count=1,
            expected_role_counts={"smoke": 8, "calibration": 1, "training": 1, "final_test": 1},
        ),
        StateAuditRunConfig,
    )
    with pytest.raises(StateAuditRunError, match="prompt role"):
        check_smoke_parameters(config)


def test_the_smoke_command_refuses_another_run_role(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, run_role="training"), StateAuditRunConfig)
    with pytest.raises(StateAuditRunError, match="engineering smoke only"):
        check_smoke_parameters(config)
