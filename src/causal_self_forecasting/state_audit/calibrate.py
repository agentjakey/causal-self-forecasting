"""Executing the preregistered calibration sweep at one layer.

Calibration chooses an intervention strength. It measures nothing about the model's abilities
and cannot become a result, which is why every record it writes carries
`scientific_result: false`.

The sequence, fixed in `docs/bluedot/preregistration_state_dependence.md` section 7.3:

1. capture the clean state for all 32 calibration prompts at the layer;
2. `reference_norm = median(||h_p||)` over those 32, computed once, before any intervention;
3. `alpha_r = ratio * reference_norm` for each of the five frozen ratios, one number per ratio;
4. apply 8 directions x 2 signs x 5 ratios, plus one shared no-op, to every prompt;
5. evaluate the six frozen conditions per ratio and take the **smallest** passing one.

Nothing in that order is adjustable at run time. The layer comes from the config and must be one
of the two preregistered ones; the ratios must be the frozen grid; the thresholds come from the
frozen plan rather than from anything measured here. The reference norm is computed before the
first intervention runs, so it cannot depend on an effect.

The layer-20 fallback is reachable only from a layer-13 decision of `fallback_required`, and
this module refuses to run it otherwise rather than trusting the operator to remember. A passing
primary layer permanently closes the fallback; the selector enforces that too.

Execution reuses `state_audit.run.execute_candidate_pass` in full. There is no second inference
path here: this module differs from the smoke only in the candidate shape it asks for and in
what it does with the observations afterwards.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..calibration.criteria import CriteriaError
from ..calibration.observations import ObservationLoadError, summarize_layer
from ..calibration.plan import (
    CalibrationPlanError,
    build_decision_record,
    load_calibration_plan,
)
from ..calibration.selection import SelectionError, select_calibration_ratio
from ..calibration.strength import MEDIAN_METHOD, StrengthError
from ..hashing import atomic_write_json, hash_file, hash_object, read_json
from ..logging_utils import info
from ..paths import (
    STATE_AUDIT_DECISION,
    STATE_AUDIT_RATIO_SUMMARIES,
    STATE_AUDIT_REFERENCE_NORM,
    STATE_AUDIT_RUN_MANIFEST,
    run_dir,
)
from ..schemas import (
    CalibrationDecisionRecord,
    CalibrationPlanRecord,
    CalibrationRatioSummary,
    CalibrationRunManifest,
    CalibrationStatus,
    LayerReferenceNormRecord,
    PromptRole,
    StateAuditCandidateKind,
    StudyRunRole,
    compute_calibration_run_hash,
)
from .run import (
    ExecutedRun,
    StateAuditRunError,
    execute_candidate_pass,
    load_run_config,
    state_audit_run_common,
)

CALIBRATION_ALGORITHM_VERSION = "bluedot_calibration_execution_v1.0"


def check_calibration_parameters(config, plan: CalibrationPlanRecord) -> None:
    """Refuse anything this command is not allowed to run.

    The layer, the ratios, the prompts, the target, and the thresholds are all preregistered.
    Each check below is one of them, enforced here rather than trusted to a reader of the config.
    """
    if config.run_role is not StudyRunRole.CALIBRATION:
        raise StateAuditRunError(
            f"this command runs calibration only; the config declares run role "
            f"{config.run_role.value!r}"
        )
    if config.prompt_role is not PromptRole.CALIBRATION:
        raise StateAuditRunError(
            f"calibration may only use the {PromptRole.CALIBRATION.value!r} prompt role; the "
            f"config names {config.prompt_role.value!r}. Choosing a strength from smoke, "
            "training, or final-test prompts would choose it from data that was not set aside "
            "to choose it."
        )
    if config.candidate_kind is not StateAuditCandidateKind.CALIBRATION_GRID:
        raise StateAuditRunError(
            f"calibration sweeps the ratio grid; the config declares "
            f"{config.candidate_kind.value!r} candidates"
        )
    if config.layer not in (plan.primary_layer, plan.fallback_layer):
        raise StateAuditRunError(
            f"layer {config.layer} is neither the plan's primary layer {plan.primary_layer} nor "
            f"its fallback layer {plan.fallback_layer}; no third layer may be calibrated"
        )
    if tuple(config.ratio_grid) != tuple(float(ratio) for ratio in plan.norm_ratios):
        raise StateAuditRunError(
            f"the config sweeps {list(config.ratio_grid)} but the frozen plan fixes "
            f"{list(plan.norm_ratios)}; the grid is not adjustable at run time"
        )
    if config.expected_prompt_count != plan.calibration_prompt_count:
        raise StateAuditRunError(
            f"the config expects {config.expected_prompt_count} calibration prompts but the plan "
            f"fixes {plan.calibration_prompt_count}"
        )
    if config.noop_tolerance != plan.noop_tolerance:
        raise StateAuditRunError(
            f"the config's no-op tolerance {config.noop_tolerance} differs from the plan's "
            f"{plan.noop_tolerance}; the tolerance a run is judged against is the planned one"
        )


def load_decision(run_id: str) -> CalibrationDecisionRecord:
    """Read a previous calibration run's decision record."""
    path = run_dir(run_id) / STATE_AUDIT_DECISION
    if not path.exists():
        raise StateAuditRunError(f"no calibration decision at {path}")
    try:
        return CalibrationDecisionRecord.model_validate(read_json(path))
    except Exception as error:
        raise StateAuditRunError(f"{path} is not a valid calibration decision: {error}") from error


def read_ratio_summaries(run_id: str) -> list[CalibrationRatioSummary]:
    path = run_dir(run_id) / STATE_AUDIT_RATIO_SUMMARIES
    if not path.exists():
        raise StateAuditRunError(f"no ratio summaries at {path}")
    payload = read_json(path)
    rows = payload.get("summaries") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise StateAuditRunError(f"{path} does not hold a non-empty list of summaries")
    try:
        return [CalibrationRatioSummary.model_validate(row) for row in rows]
    except Exception as error:
        raise StateAuditRunError(f"{path}: {error}") from error


def check_fallback_is_open(
    plan: CalibrationPlanRecord, layer: int, primary_run_id: str | None
) -> list[CalibrationRatioSummary] | None:
    """Refuse the fallback layer unless the primary layer already failed.

    The preregistration names exactly one trigger for layer 20: no layer-13 ratio satisfied every
    condition. Requiring the primary run's own decision record, rather than an assurance that it
    failed, is what stops the fallback becoming a second attempt.
    """
    if layer != plan.fallback_layer:
        if primary_run_id is not None:
            raise StateAuditRunError(
                f"a primary run id is only meaningful when calibrating the fallback layer "
                f"{plan.fallback_layer}; this run is at layer {layer}"
            )
        return None

    if primary_run_id is None:
        raise StateAuditRunError(
            f"calibrating the fallback layer {plan.fallback_layer} requires the primary run it "
            "falls back from. Pass --primary-run-id so the fallback can be checked against a "
            "recorded layer-13 failure rather than taken on trust."
        )

    decision = load_decision(primary_run_id)
    if decision.plan_hash != plan.plan_hash:
        raise StateAuditRunError(
            f"the primary decision in run {primary_run_id!r} was made under a different "
            "calibration plan than this run cites"
        )
    if decision.primary_layer != plan.primary_layer:
        raise StateAuditRunError(
            f"the primary decision in run {primary_run_id!r} calibrated layer "
            f"{decision.primary_layer}, not the plan's primary layer {plan.primary_layer}"
        )
    if decision.status is not CalibrationStatus.FALLBACK_REQUIRED:
        raise StateAuditRunError(
            f"run {primary_run_id!r} recorded status {decision.status.value!r} at layer "
            f"{plan.primary_layer}. The layer-{plan.fallback_layer} fallback is reachable only "
            "from 'fallback_required'; running it after any other outcome would be a second "
            "attempt, not a preregistered fallback."
        )
    return list(decision.ratio_summaries)


def build_reference_norm_record(
    executed: ExecutedRun, plan: CalibrationPlanRecord
) -> LayerReferenceNormRecord:
    """The reference norm with the 32 numbers it was computed from."""
    norms = {clean.assignment.variant_id: clean.state_norm for clean in executed.clean_passes}
    prompt_ids = sorted(norms)
    return LayerReferenceNormRecord(
        layer=executed.inputs.config.layer,
        prompt_ids=prompt_ids,
        prompt_identity_hash=hash_object(prompt_ids),
        count=len(prompt_ids),
        state_norms=norms,
        reference_norm=executed.reference_norm,
        median_method=MEDIAN_METHOD,
        prompt_manifest_hash=executed.inputs.manifest.manifest_hash,
        model_id=executed.model.spec.model_id,
        model_revision=executed.model.spec.revision,
    )


def build_calibration_manifest(
    executed: ExecutedRun,
    decision: CalibrationDecisionRecord,
    reference_norm_hash: str,
    summaries_hash: str,
) -> CalibrationRunManifest:
    config = executed.inputs.config
    common = state_audit_run_common(
        executed,
        reference_norm_source=(
            f"median clean residual-stream norm over the {len(executed.clean_passes)} "
            f"calibration prompts at layer {config.layer}, computed before any intervention ran; "
            "this is the preregistered calibration reference norm"
        ),
    )
    payload = dict(common.payload)
    payload["norm_ratios"] = [float(ratio) for ratio, _ in executed.ratio_alphas]
    payload["global_alphas"] = [float(alpha) for _, alpha in executed.ratio_alphas]
    payload["reference_norm_record_hash"] = reference_norm_hash
    payload["ratio_summaries_hash"] = summaries_hash
    payload["decision_hash"] = decision.decision_hash
    payload["decision_status"] = decision.status.value
    payload["selected_layer"] = decision.selected_layer
    payload["selected_norm_ratio"] = decision.selected_norm_ratio
    payload["selected_global_alpha"] = decision.selected_global_alpha
    git = common.environment.get("git") or {}

    return CalibrationRunManifest(
        **{key: value for key, value in payload.items() if key != "diagnostics"},
        diagnostics=executed.diagnostics,
        manifest_hash=compute_calibration_run_hash(payload),
        config_path=common.config_path,
        code_commit=git.get("commit"),
        code_branch=git.get("branch"),
        code_dirty=git.get("dirty"),
        environment=common.environment,
        provenance=common.provenance,
        started_at=executed.started_at,
        completed_at=datetime.now(UTC),
        notes=(
            "Calibration run. It chooses an intervention strength and measures nothing about the "
            "model's abilities. Every candidate was applied to every prompt and every failure is "
            "recorded. This is not a scientific result."
        ),
    )


def calibration_report(
    manifest: CalibrationRunManifest,
    decision: CalibrationDecisionRecord,
    directory: Path,
) -> dict[str, Any]:
    """The command's JSON output. Numbers come from the written records, never recomputed."""
    return {
        "run_id": manifest.run_id,
        "run_role": manifest.run_role.value,
        "status": manifest.status,
        "scientific_result": False,
        "run_path": str(directory),
        "manifest_path": str(directory / STATE_AUDIT_RUN_MANIFEST),
        "manifest_hash": manifest.manifest_hash,
        "input_fingerprint": manifest.input_fingerprint,
        "study_id": manifest.study_id,
        "target_name": manifest.target_name,
        "model_id": manifest.model_id,
        "model_revision": manifest.model_revision,
        "tokenizer_revision": manifest.tokenizer_revision,
        "dtype": manifest.dtype,
        "device": manifest.device,
        "prompt_manifest_id": manifest.prompt_manifest_id,
        "prompt_manifest_hash": manifest.prompt_manifest_hash,
        "prompt_role": manifest.prompt_role.value,
        "direction_family_id": manifest.direction_family_id,
        "direction_family_hash": manifest.direction_family_hash,
        "calibration_plan_id": manifest.calibration_plan_id,
        "calibration_plan_hash": manifest.calibration_plan_hash,
        "layer": manifest.layer,
        "capture_position": manifest.capture_position,
        "norm_ratios": list(manifest.norm_ratios),
        "global_alphas": list(manifest.global_alphas),
        "reference_norm": manifest.reference_norm,
        "reference_norm_source": manifest.reference_norm_source,
        "decision": {
            "status": decision.status.value,
            "selected_layer": decision.selected_layer,
            "selected_norm_ratio": decision.selected_norm_ratio,
            "selected_global_alpha": decision.selected_global_alpha,
            "rationale": decision.selection_rationale,
            "algorithm_version": decision.selection_algorithm_version,
            "decision_hash": decision.decision_hash,
        },
        "ratio_table": [
            {
                "norm_ratio": summary.norm_ratio,
                "layer": summary.layer,
                "global_alpha": summary.global_alpha,
                "observed_non_noop_observations": summary.observed_non_noop_observations,
                "noop_count": summary.noop_count,
                "failure_count": summary.failure_count,
                "finite_output_rate": summary.finite_output_rate,
                "max_abs_noop_target": summary.max_abs_noop_target,
                "fraction_above_effect_threshold": summary.fraction_above_effect_threshold,
                "median_abs_effect": summary.median_abs_effect,
                "p95_abs_effect": summary.p95_abs_effect,
                "flip_count": summary.flip_count,
                "passed": summary.passed,
                "failed_conditions": [c.name for c in summary.criteria if not c.passed],
            }
            for summary in decision.ratio_summaries
        ],
        "counts": {
            "expected_prompts": manifest.expected_prompt_count,
            "observed_prompts": manifest.observed_prompt_count,
            "observed_states": manifest.observed_state_count,
            "expected_non_noop_observations": manifest.expected_non_noop_observations,
            "observed_non_noop_observations": manifest.observed_non_noop_observations,
            "expected_noop_observations": manifest.expected_noop_observations,
            "observed_noop_observations": manifest.observed_noop_observations,
            "expected_forwards": manifest.expected_forward_count,
            "observed_forwards": manifest.observed_forward_count,
            "failures": manifest.failure_count,
        },
        "clean_accuracy_descriptive": manifest.clean_accuracy_descriptive,
        "clean_correct_count": manifest.clean_correct_count,
        "clean_scored_count": manifest.clean_scored_count,
        "diagnostics": manifest.diagnostics.model_dump(mode="json"),
        "artifact_hashes": {
            "observations": manifest.observations_hash,
            "failures": manifest.failures_hash,
            "states": manifest.states_hash,
            "candidate_sets": manifest.candidate_sets_hash,
            "clean_pass": manifest.clean_pass_hash,
            "reference_norm": manifest.reference_norm_record_hash,
            "ratio_summaries": manifest.ratio_summaries_hash,
            "decision": manifest.decision_hash,
        },
        "code_commit": manifest.code_commit,
        "code_branch": manifest.code_branch,
        "code_dirty": manifest.code_dirty,
        "notes": (
            "Calibration chooses an intervention strength. It measures nothing about the model's "
            "abilities and is not a scientific result. The selected ratio is the smallest one "
            "satisfying every preregistered condition, not the one with the largest effect."
        ),
    }


def run_calibration(
    config_path: str | Path,
    run_id: str,
    primary_run_id: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Execute the calibration sweep at one layer, summarize it, and record the decision."""
    config = load_run_config(config_path)
    try:
        plan = load_calibration_plan(config.calibration_plan_id)
    except CalibrationPlanError as error:
        raise StateAuditRunError(str(error)) from error

    check_calibration_parameters(config, plan)
    primary_summaries = check_fallback_is_open(plan, config.layer, primary_run_id)

    executed = execute_candidate_pass(
        config_path, run_id, StateAuditCandidateKind.CALIBRATION_GRID, force=force
    )
    directory = executed.directory

    reference_record = build_reference_norm_record(executed, plan)
    atomic_write_json(
        directory / STATE_AUDIT_REFERENCE_NORM, reference_record.model_dump(mode="json")
    )
    reference_norm_hash = hash_file(directory / STATE_AUDIT_REFERENCE_NORM)

    try:
        summaries = summarize_layer(
            executed.observations, plan, config.layer, failures=executed.failures
        )
    except (ObservationLoadError, CriteriaError, StrengthError) as error:
        raise StateAuditRunError(
            f"the calibration observations could not be summarized: {error}"
        ) from error

    atomic_write_json(
        directory / STATE_AUDIT_RATIO_SUMMARIES,
        {
            "run_id": run_id,
            "plan_id": plan.plan_id,
            "plan_hash": plan.plan_hash,
            "layer": config.layer,
            "reference_norm": executed.reference_norm,
            "summaries": [summary.model_dump(mode="json") for summary in summaries],
            "scientific_result": False,
            "notes": (
                "Calibration infrastructure. These summaries describe an intervention-strength "
                "grid; they are not a measurement of the model's abilities."
            ),
        },
    )
    summaries_hash = hash_file(directory / STATE_AUDIT_RATIO_SUMMARIES)

    try:
        if config.layer == plan.fallback_layer:
            selection = select_calibration_ratio(
                primary_layer=plan.primary_layer,
                fallback_layer=plan.fallback_layer,
                expected_ratios=plan.norm_ratios,
                primary_summaries=primary_summaries or [],
                fallback_summaries=summaries,
            )
        else:
            selection = select_calibration_ratio(
                primary_layer=plan.primary_layer,
                fallback_layer=plan.fallback_layer,
                expected_ratios=plan.norm_ratios,
                primary_summaries=summaries,
            )
        decision = build_decision_record(plan, selection)
    except SelectionError as error:
        raise StateAuditRunError(f"the calibration decision could not be made: {error}") from error

    atomic_write_json(directory / STATE_AUDIT_DECISION, decision.model_dump(mode="json"))

    manifest = build_calibration_manifest(executed, decision, reference_norm_hash, summaries_hash)
    atomic_write_json(directory / STATE_AUDIT_RUN_MANIFEST, manifest.model_dump(mode="json"))

    info(
        "executed calibration",
        run_id=run_id,
        layer=config.layer,
        status=manifest.status,
        decision=decision.status.value,
        selected_ratio=decision.selected_norm_ratio,
        observations=len(executed.observations),
        failures=len(executed.failures),
        forwards=executed.forwards,
    )
    return calibration_report(manifest, decision, directory)


__all__ = [
    "CALIBRATION_ALGORITHM_VERSION",
    "build_calibration_manifest",
    "build_reference_norm_record",
    "calibration_report",
    "check_calibration_parameters",
    "check_fallback_is_open",
    "load_decision",
    "read_ratio_summaries",
    "run_calibration",
]
