"""The frozen calibration plan, and the decision built from summaries.

A plan says what would count as a passing ratio, and it is written before any calibration
number exists. That ordering is the point: thresholds chosen after seeing the effect
distribution are not thresholds.

Planning loads no model. It reads the frozen prompt manifest, the frozen direction-family
manifest, and the *config* of the pinned model for its identity, then does arithmetic. Nothing
here runs a prompt, captures a state, measures a norm, or applies an intervention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import (
    CalibrationPlanConfig,
    ConfigError,
    ModelConfig,
    config_hash,
    load_config,
    repo_root,
)
from ..hashing import atomic_write_json, canonical_json_bytes, read_json
from ..logging_utils import info
from ..paths import calibration_plan_path
from ..reproducibility import package_versions
from ..schemas import (
    CalibrationDecisionRecord,
    CalibrationForwardCounts,
    CalibrationPlanRecord,
    CalibrationThresholds,
    PromptRole,
    calibration_decision_payload,
    calibration_plan_payload,
    compute_calibration_decision_hash,
    compute_calibration_plan_hash,
)
from ..tasks.prompt_manifest import PromptManifestError, load_prompt_manifest
from .criteria import PERCENTILE_METHOD
from .selection import SELECTION_ALGORITHM_VERSION, CalibrationSelection
from .strength import MEDIAN_METHOD

PLAN_ALGORITHM_VERSION = "bluedot_calibration_plan_v1.0"


def _direction_family_io():
    """Import the direction-family reader lazily.

    `interventions.direction_family` imports torch for the construction code. Planning only
    needs to read a small JSON manifest, and paying a multi-second torch import to do that
    would make "planning loads no model" true in letter and annoying in practice.
    """
    from ..interventions.direction_family import DirectionFamilyError, load_direction_family

    return DirectionFamilyError, load_direction_family


class CalibrationPlanError(RuntimeError):
    """Raised when a calibration plan cannot be built, written, or trusted."""


def build_forward_counts(
    role_counts: dict[str, int],
    direction_count: int,
    signed_directions: int,
    ratio_count: int,
) -> CalibrationForwardCounts:
    """Derive the planned forward arithmetic from the design, rather than restating totals.

    Only calibration sweeps the ratio grid. Every other role runs at the one selected strength,
    which is why calibration carries 82 forwards per prompt and the others carry 18.
    """
    if signed_directions != 2 * direction_count:
        raise CalibrationPlanError(
            f"{signed_directions} signed directions is not two per direction for "
            f"{direction_count} directions"
        )

    other_candidates = signed_directions + 1
    calibration_candidates = signed_directions * ratio_count + 1
    smoke = role_counts[PromptRole.SMOKE.value]
    calibration = role_counts[PromptRole.CALIBRATION.value]
    training = role_counts[PromptRole.TRAINING.value]
    final_test = role_counts[PromptRole.FINAL_TEST.value]

    smoke_forwards = smoke * (1 + other_candidates)
    calibration_forwards = calibration * (1 + calibration_candidates)
    training_forwards = training * (1 + other_candidates)
    final_test_forwards = final_test * (1 + other_candidates)
    primary_total = smoke_forwards + calibration_forwards + training_forwards + final_test_forwards

    return CalibrationForwardCounts(
        signed_directions=signed_directions,
        ratio_count=ratio_count,
        smoke_prompts=smoke,
        calibration_prompts=calibration,
        training_prompts=training,
        final_test_prompts=final_test,
        candidates_per_calibration_prompt=calibration_candidates,
        candidates_per_other_prompt=other_candidates,
        forwards_per_calibration_prompt=1 + calibration_candidates,
        forwards_per_other_prompt=1 + other_candidates,
        smoke_forwards=smoke_forwards,
        calibration_forwards_per_layer=calibration_forwards,
        training_forwards=training_forwards,
        final_test_forwards=final_test_forwards,
        primary_total_forwards=primary_total,
        fallback_additional_forwards=calibration_forwards,
        with_fallback_total_forwards=primary_total + calibration_forwards,
    )


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root())).replace("\\", "/")
    except ValueError:
        return path.name


def build_calibration_plan(
    plan_config: CalibrationPlanConfig,
    config_path: str | Path,
) -> CalibrationPlanRecord:
    """Cross-check the frozen manifests and build the plan. Loads no model weights."""
    try:
        model_config = load_config(plan_config.model_ref, ModelConfig)
    except ConfigError as error:
        raise CalibrationPlanError(
            f"calibration config {plan_config.name!r} references a model config that does not "
            f"load: {error}"
        ) from error

    try:
        prompts = load_prompt_manifest(plan_config.prompt_manifest_id)
    except PromptManifestError as error:
        raise CalibrationPlanError(str(error)) from error
    family_error, load_family = _direction_family_io()
    try:
        family = load_family(plan_config.direction_family_id)
    except family_error as error:
        raise CalibrationPlanError(str(error)) from error

    expected_roles = {role.value: count for role, count in plan_config.expected_role_counts.items()}
    if dict(prompts.role_counts) != expected_roles:
        raise CalibrationPlanError(
            f"prompt manifest {prompts.manifest_id!r} has role counts "
            f"{dict(sorted(prompts.role_counts.items()))} but the plan expects "
            f"{dict(sorted(expected_roles.items()))}"
        )
    calibration_prompts = prompts.role_counts[PromptRole.CALIBRATION.value]
    if calibration_prompts != plan_config.calibration_prompt_count:
        raise CalibrationPlanError(
            f"the prompt manifest supplies {calibration_prompts} calibration prompts but the "
            f"plan expects {plan_config.calibration_prompt_count}"
        )

    if len(family.directions) != plan_config.expected_direction_count:
        raise CalibrationPlanError(
            f"direction family {family.family_id!r} has {len(family.directions)} directions but "
            f"the plan expects exactly {plan_config.expected_direction_count}"
        )
    if family.model_id != model_config.model_id:
        raise CalibrationPlanError(
            f"the direction family was built against {family.model_id!r} but the plan's model "
            f"config is {model_config.model_id!r}"
        )
    if family.model_revision != model_config.revision:
        raise CalibrationPlanError(
            f"the direction family was built against revision {family.model_revision} but the "
            f"plan's model config pins {model_config.revision}; directions built from different "
            "weights would not be the stimuli this plan describes"
        )

    thresholds = CalibrationThresholds(
        min_large_effect_fraction=plan_config.thresholds.min_large_effect_fraction,
        large_effect_threshold=plan_config.thresholds.large_effect_threshold,
        min_median_abs_effect=plan_config.thresholds.min_median_abs_effect,
        max_p95_abs_effect=plan_config.thresholds.max_p95_abs_effect,
    )
    forward_counts = build_forward_counts(
        role_counts=dict(prompts.role_counts),
        direction_count=len(family.directions),
        signed_directions=plan_config.expected_signed_directions,
        ratio_count=len(plan_config.norm_ratios),
    )

    draft: dict[str, Any] = {
        "schema_version": CalibrationPlanRecord.model_fields["schema_version"].default,
        "plan_id": plan_config.plan_id,
        "study_id": plan_config.study_id,
        "target_name": plan_config.target,
        "model_id": model_config.model_id,
        "model_revision": model_config.revision,
        "prompt_manifest_id": prompts.manifest_id,
        "prompt_manifest_hash": prompts.manifest_hash,
        "direction_family_id": family.family_id,
        "direction_family_hash": family.family_hash,
        "calibration_prompt_count": calibration_prompts,
        "role_counts": dict(prompts.role_counts),
        "direction_count": len(family.directions),
        "primary_layer": plan_config.primary_layer,
        "fallback_layer": plan_config.fallback_layer,
        "norm_ratios": [float(ratio) for ratio in plan_config.norm_ratios],
        "thresholds": thresholds.model_dump(mode="json"),
        "noop_tolerance": plan_config.noop_tolerance,
        "percentile_method": plan_config.percentile_method,
        "median_method": plan_config.median_method,
        "selection_algorithm_version": plan_config.selection_algorithm_version,
        "master_seed": plan_config.master_seed,
        "forward_counts": forward_counts.model_dump(mode="json"),
        "config_hash": config_hash(config_path),
    }

    return CalibrationPlanRecord(
        plan_id=plan_config.plan_id,
        study_id=plan_config.study_id,
        target_name="delta_clean_top_margin",
        model_id=model_config.model_id,
        model_revision=model_config.revision,
        prompt_manifest_id=prompts.manifest_id,
        prompt_manifest_hash=prompts.manifest_hash,
        direction_family_id=family.family_id,
        direction_family_hash=family.family_hash,
        calibration_prompt_count=calibration_prompts,
        role_counts=dict(prompts.role_counts),
        direction_count=len(family.directions),
        primary_layer=plan_config.primary_layer,
        fallback_layer=plan_config.fallback_layer,
        norm_ratios=[float(ratio) for ratio in plan_config.norm_ratios],
        thresholds=thresholds,
        noop_tolerance=plan_config.noop_tolerance,
        percentile_method=plan_config.percentile_method,
        median_method=plan_config.median_method,
        selection_algorithm_version=plan_config.selection_algorithm_version,
        master_seed=plan_config.master_seed,
        forward_counts=forward_counts,
        config_hash=str(draft["config_hash"]),
        plan_hash=compute_calibration_plan_hash(draft),
        config_path=_repo_relative(Path(config_path)),
        notes=(
            "Calibration planning only. No model was loaded, no prompt was run, no state norm "
            "was measured, and no intervention was applied. This is not a scientific result."
        ),
    )


def plan_path(plan_id: str) -> Path:
    """Where a plan lives. One accessor, so redirecting the location redirects every caller."""
    return calibration_plan_path(plan_id)


def plan_content_bytes(record: CalibrationPlanRecord) -> bytes:
    return canonical_json_bytes(calibration_plan_payload(record.model_dump(mode="json")))


def load_calibration_plan(plan_id: str) -> CalibrationPlanRecord:
    path = calibration_plan_path(plan_id)
    if not path.exists():
        raise CalibrationPlanError(f"no calibration plan at {path}")
    try:
        return CalibrationPlanRecord.model_validate(read_json(path))
    except Exception as error:
        raise CalibrationPlanError(f"{path} is not a valid calibration plan: {error}") from error


def write_calibration_plan(record: CalibrationPlanRecord, force: bool = False) -> tuple[Path, str]:
    """Write the plan atomically, refusing to replace a different one."""
    path = calibration_plan_path(record.plan_id)
    if path.exists():
        existing = load_calibration_plan(record.plan_id)
        if existing.plan_hash == record.plan_hash:
            return path, "unchanged"
        if not force:
            raise CalibrationPlanError(
                f"{path} already holds a different calibration plan "
                f"(existing {existing.plan_hash}, proposed {record.plan_hash}). Thresholds and "
                "ratios must be fixed before the numbers exist, so replacing a plan silently "
                "would defeat the point of writing one. Pass force only to repair a documented "
                "bug, and record why in docs/experiment_log.md."
            )
        atomic_write_json(path, record.model_dump(mode="json"))
        return path, "overwritten"

    atomic_write_json(path, record.model_dump(mode="json"))
    return path, "written"


def verify_calibration_plan(record: CalibrationPlanRecord) -> dict[str, Any]:
    """Check a plan against the manifests it cites. Loads no model."""
    failures: list[str] = []

    try:
        prompts = load_prompt_manifest(record.prompt_manifest_id)
        if prompts.manifest_hash != record.prompt_manifest_hash:
            failures.append("the prompt manifest on disk has a different hash than the plan cites")
        if dict(prompts.role_counts) != dict(record.role_counts):
            failures.append("the prompt manifest role counts differ from the plan")
    except PromptManifestError as error:
        failures.append(f"prompt manifest: {error}")

    family_error, load_family = _direction_family_io()
    try:
        family = load_family(record.direction_family_id)
        if family.family_hash != record.direction_family_hash:
            failures.append("the direction family on disk has a different hash than the plan cites")
        if len(family.directions) != record.direction_count:
            failures.append("the direction family has a different number of directions")
        if family.model_revision != record.model_revision:
            failures.append("the direction family cites a different model revision than the plan")
    except family_error as error:
        failures.append(f"direction family: {error}")

    if record.target_name != "delta_clean_top_margin":
        failures.append(f"the plan target {record.target_name!r} is not the arm's target")
    if record.forward_counts.calibration_prompts != record.calibration_prompt_count:
        failures.append("the forward arithmetic and the plan disagree on the prompt count")

    return {
        "plan_id": record.plan_id,
        "valid": not failures,
        "failures": failures,
        "plan_hash": record.plan_hash,
        "prompt_manifest_hash": record.prompt_manifest_hash,
        "direction_family_hash": record.direction_family_hash,
        "primary_layer": record.primary_layer,
        "fallback_layer": record.fallback_layer,
        "norm_ratios": list(record.norm_ratios),
        "forward_counts": record.forward_counts.model_dump(mode="json"),
    }


def build_decision_record(
    plan: CalibrationPlanRecord,
    selection: CalibrationSelection,
) -> CalibrationDecisionRecord:
    """Turn a state-machine outcome into a hashed, immutable decision record."""
    draft: dict[str, Any] = {
        "schema_version": CalibrationDecisionRecord.model_fields["schema_version"].default,
        "plan_id": plan.plan_id,
        "study_id": plan.study_id,
        "status": selection.status.value,
        "selected_layer": selection.selected_layer,
        "selected_norm_ratio": selection.selected_norm_ratio,
        "selected_global_alpha": selection.selected_global_alpha,
        "primary_layer": plan.primary_layer,
        "fallback_layer": plan.fallback_layer,
        "ratio_summaries": [summary.model_dump(mode="json") for summary in selection.summaries],
        "selection_rationale": selection.rationale,
        "selection_algorithm_version": plan.selection_algorithm_version,
        "plan_hash": plan.plan_hash,
        "prompt_manifest_hash": plan.prompt_manifest_hash,
        "direction_family_hash": plan.direction_family_hash,
    }
    return CalibrationDecisionRecord(
        plan_id=plan.plan_id,
        study_id=plan.study_id,
        status=selection.status,
        selected_layer=selection.selected_layer,
        selected_norm_ratio=selection.selected_norm_ratio,
        selected_global_alpha=selection.selected_global_alpha,
        primary_layer=plan.primary_layer,
        fallback_layer=plan.fallback_layer,
        ratio_summaries=list(selection.summaries),
        selection_rationale=selection.rationale,
        selection_algorithm_version=plan.selection_algorithm_version,
        plan_hash=plan.plan_hash,
        prompt_manifest_hash=plan.prompt_manifest_hash,
        direction_family_hash=plan.direction_family_hash,
        environment={"packages": package_versions()},
        decision_hash=compute_calibration_decision_hash(draft),
    )


def decision_content_bytes(record: CalibrationDecisionRecord) -> bytes:
    return canonical_json_bytes(calibration_decision_payload(record.model_dump(mode="json")))


def plan_report(record: CalibrationPlanRecord, status: str, path: Path) -> dict[str, Any]:
    return {
        "status": status,
        "plan_path": _repo_relative(path),
        "plan_id": record.plan_id,
        "plan_hash": record.plan_hash,
        "study_id": record.study_id,
        "target_name": record.target_name,
        "model_id": record.model_id,
        "model_revision": record.model_revision,
        "prompt_manifest_id": record.prompt_manifest_id,
        "prompt_manifest_hash": record.prompt_manifest_hash,
        "direction_family_id": record.direction_family_id,
        "direction_family_hash": record.direction_family_hash,
        "calibration_prompt_count": record.calibration_prompt_count,
        "role_counts": dict(record.role_counts),
        "direction_count": record.direction_count,
        "primary_layer": record.primary_layer,
        "fallback_layer": record.fallback_layer,
        "norm_ratios": list(record.norm_ratios),
        "thresholds": record.thresholds.model_dump(mode="json"),
        "noop_tolerance": record.noop_tolerance,
        "percentile_method": record.percentile_method,
        "median_method": record.median_method,
        "selection_algorithm_version": record.selection_algorithm_version,
        "plan_algorithm_version": PLAN_ALGORITHM_VERSION,
        "forward_counts": record.forward_counts.model_dump(mode="json"),
        "config_hash": record.config_hash,
        "scientific_result": False,
        "notes": (
            "Calibration planning artifact. No model was loaded, no prompt was run, no state "
            "norm was measured, no intervention was applied, and no calibration result exists."
        ),
    }


def plan_command(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    """Build or confirm the frozen plan artifact. Loads no model."""
    try:
        plan_config = load_config(config_path, CalibrationPlanConfig)
    except ConfigError as error:
        raise CalibrationPlanError(str(error)) from error

    record = build_calibration_plan(plan_config, config_path)
    path, status = write_calibration_plan(record, force=force)
    verification = verify_calibration_plan(record)

    info(
        "froze calibration plan",
        plan_id=record.plan_id,
        status=status,
        plan_hash=record.plan_hash[:23],
    )

    report = plan_report(record, status, path)
    report["verification"] = verification
    return report


__all__ = [
    "MEDIAN_METHOD",
    "PERCENTILE_METHOD",
    "PLAN_ALGORITHM_VERSION",
    "SELECTION_ALGORITHM_VERSION",
    "CalibrationPlanError",
    "build_calibration_plan",
    "build_decision_record",
    "build_forward_counts",
    "decision_content_bytes",
    "load_calibration_plan",
    "plan_command",
    "plan_content_bytes",
    "plan_report",
    "verify_calibration_plan",
    "write_calibration_plan",
]
