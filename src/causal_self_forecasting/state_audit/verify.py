"""Verifying a state-dependence run from its artifacts.

Loads no model, by construction: everything checked here is recomputed from files on disk. That
matters because a verifier that reran the model would be checking the model rather than the
record, and a third party with the artifacts and no GPU still has to be able to run it.

What it checks:

* the run manifest recomputes its own content hash, and every artifact still hashes to what the
  manifest recorded;
* every observation revalidates its target against its own logits, which happens at load time;
* the counts add up, and no expected prompt-candidate pair is silently missing;
* exactly one global alpha was used across every non-no-op observation, at the run's layer and
  ratio;
* the no-op maximum absolute target is within tolerance and every no-op displaced the residual
  stream by exactly zero;
* the reference norm recomputes as the median of the recorded clean state norms, and the alpha
  recomputes as ratio times that median;
* the prompt manifest, direction family, and calibration plan on disk still hash to what the run
  cited;
* every observed candidate is one the frozen candidate sets actually contain.

Optionally it compares two runs of the same role row by row, which is how cross-process
determinism is measured: same prompts, same candidate ids, and the same target to the bit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..calibration.plan import CalibrationPlanError, load_calibration_plan
from ..calibration.strength import MEDIAN_METHOD, StrengthError, check_global_alpha
from ..hashing import hash_file, read_jsonl
from ..interventions.direction_family import DirectionFamilyError, load_direction_family
from ..paths import (
    STATE_AUDIT_CANDIDATE_SETS,
    STATE_AUDIT_CLEAN_PASS,
    STATE_AUDIT_FAILURES,
    STATE_AUDIT_OBSERVATIONS,
    STATE_AUDIT_STATES,
    run_dir,
)
from ..schemas import (
    StateAuditCandidateSet,
    StateAuditCleanPassRecord,
    StateAuditObservationRecord,
    StudyRunManifest,
)
from ..tasks.prompt_manifest import PromptManifestError, load_prompt_manifest
from .run import StateAuditRunError, load_study_run_manifest

# How far a recomputed reference norm may sit from the recorded one. Both are `numpy.median` over
# the same float64 values, so anything above round-off means the recorded number did not come
# from the recorded norms.
NORM_TOLERANCE = 1e-9


def read_observations(run_id: str) -> list[StateAuditObservationRecord]:
    """Read a run's observations, revalidating each record's target as it loads."""
    path = run_dir(run_id) / STATE_AUDIT_OBSERVATIONS
    if not path.exists():
        raise StateAuditRunError(f"no state-audit observations at {path}")
    records: list[StateAuditObservationRecord] = []
    for index, row in enumerate(read_jsonl(path), start=1):
        try:
            records.append(StateAuditObservationRecord.model_validate(row))
        except Exception as error:
            raise StateAuditRunError(f"{path}:{index}: {error}") from error
    return records


def read_clean_pass(run_id: str) -> list[StateAuditCleanPassRecord]:
    path = run_dir(run_id) / STATE_AUDIT_CLEAN_PASS
    if not path.exists():
        raise StateAuditRunError(f"no state-audit clean pass at {path}")
    return [StateAuditCleanPassRecord.model_validate(row) for row in read_jsonl(path)]


def read_candidate_sets(run_id: str) -> list[StateAuditCandidateSet]:
    path = run_dir(run_id) / STATE_AUDIT_CANDIDATE_SETS
    if not path.exists():
        raise StateAuditRunError(f"no state-audit candidate sets at {path}")
    return [StateAuditCandidateSet.model_validate(row) for row in read_jsonl(path)]


def read_failures(run_id: str) -> list[dict[str, Any]]:
    path = run_dir(run_id) / STATE_AUDIT_FAILURES
    if not path.exists():
        return []
    return list(read_jsonl(path))


def _check_artifact_hashes(
    manifest: StudyRunManifest, directory: Path, failures: list[str]
) -> None:
    expected = [
        (STATE_AUDIT_OBSERVATIONS, manifest.observations_hash, True),
        (STATE_AUDIT_CANDIDATE_SETS, manifest.candidate_sets_hash, True),
        (STATE_AUDIT_CLEAN_PASS, manifest.clean_pass_hash, True),
        (STATE_AUDIT_STATES, manifest.states_hash, True),
        (STATE_AUDIT_FAILURES, manifest.failures_hash, False),
    ]
    for name, recorded, required in expected:
        path = directory / name
        if recorded is None:
            if path.exists():
                failures.append(
                    f"{name} exists on disk but the manifest records no hash for it, so the run "
                    "was written before that file was"
                )
            continue
        if not path.exists():
            if required:
                failures.append(f"{name} is missing but the manifest cites hash {recorded}")
            continue
        actual = hash_file(path)
        if actual != recorded:
            failures.append(f"{name} hashes to {actual} but the manifest cites {recorded}")


def _check_cited_manifests(manifest: StudyRunManifest, failures: list[str]) -> None:
    try:
        prompts = load_prompt_manifest(manifest.prompt_manifest_id)
        if prompts.manifest_hash != manifest.prompt_manifest_hash:
            failures.append("the prompt manifest on disk has a different hash than the run cites")
    except PromptManifestError as error:
        failures.append(f"prompt manifest: {error}")

    try:
        family = load_direction_family(manifest.direction_family_id)
        if family.family_hash != manifest.direction_family_hash:
            failures.append("the direction family on disk has a different hash than the run cites")
        if family.model_revision != manifest.model_revision:
            failures.append("the direction family cites a different model revision than the run")
        if family.hidden_dim != manifest.diagnostics.state_dim:
            failures.append(
                f"the run captured {manifest.diagnostics.state_dim}-dimensional states but the "
                f"direction family is {family.hidden_dim}-dimensional"
            )
    except DirectionFamilyError as error:
        failures.append(f"direction family: {error}")

    try:
        plan = load_calibration_plan(manifest.calibration_plan_id)
        if plan.plan_hash != manifest.calibration_plan_hash:
            failures.append("the calibration plan on disk has a different hash than the run cites")
        if plan.target_name != manifest.target_name:
            failures.append("the calibration plan targets a different quantity than the run")
    except CalibrationPlanError as error:
        failures.append(f"calibration plan: {error}")


def _check_observations(
    manifest: StudyRunManifest,
    observations: Sequence[StateAuditObservationRecord],
    candidate_sets: Sequence[StateAuditCandidateSet],
    failures: list[str],
) -> dict[str, Any]:
    signed = [record for record in observations if not record.is_noop]
    noops = [record for record in observations if record.is_noop]

    if len(signed) != manifest.observed_non_noop_observations:
        failures.append(
            f"{len(signed)} non-no-op observations on disk, manifest records "
            f"{manifest.observed_non_noop_observations}"
        )
    if len(noops) != manifest.observed_noop_observations:
        failures.append(
            f"{len(noops)} no-op observations on disk, manifest records "
            f"{manifest.observed_noop_observations}"
        )

    wrong_layer = sorted({record.layer for record in observations} - {manifest.layer})
    if wrong_layer:
        failures.append(f"observations were recorded at layers {wrong_layer}, not {manifest.layer}")
    wrong_ratio = sorted({record.norm_ratio for record in signed} - {float(manifest.norm_ratio)})
    if wrong_ratio:
        failures.append(
            f"signed observations carry ratios {wrong_ratio}, not {manifest.norm_ratio}"
        )
    wrong_target = sorted({record.target_name for record in observations} - {manifest.target_name})
    if wrong_target:
        failures.append(f"observations carry targets {wrong_target}, not {manifest.target_name}")
    wrong_role = sorted(
        {record.prompt_role.value for record in observations} - {manifest.prompt_role.value}
    )
    if wrong_role:
        failures.append(
            f"observations carry prompt roles {wrong_role}, not {manifest.prompt_role.value}"
        )

    global_alpha: float | None = None
    if signed:
        try:
            global_alpha = check_global_alpha(
                [(record.candidate_id, record.global_alpha) for record in signed],
                manifest.layer,
                float(manifest.norm_ratio),
            )
        except StrengthError as error:
            failures.append(str(error))
        if global_alpha is not None and global_alpha != manifest.global_alpha:
            failures.append(
                f"the observations used alpha {global_alpha} but the manifest records "
                f"{manifest.global_alpha}"
            )

    non_finite = [
        record.candidate_id
        for record in observations
        if not math.isfinite(record.delta_clean_top_margin)
    ]
    if non_finite:
        failures.append(f"these observations carry a non-finite target: {non_finite[:5]}")

    max_abs_noop = max((abs(record.delta_clean_top_margin) for record in noops), default=0.0)
    moved_noops = [
        record.candidate_id
        for record in noops
        if record.delta_norm is not None and record.delta_norm != 0.0
    ]
    if moved_noops:
        failures.append(f"these no-ops displaced the residual stream: {moved_noops[:5]}")

    expected_pairs = {
        (candidate_set.trial_id, candidate.candidate_id)
        for candidate_set in candidate_sets
        for candidate in candidate_set.candidates
    }
    observed_pairs = {(record.trial_id, record.candidate_id) for record in observations}
    unknown = sorted(observed_pairs - expected_pairs)
    if unknown:
        failures.append(
            f"these observations name candidates that no frozen candidate set contains: "
            f"{unknown[:5]}"
        )
    duplicated = len(observations) - len(observed_pairs)
    if duplicated:
        failures.append(f"{duplicated} observations repeat a trial and candidate pair")

    return {
        "non_noop_observations": len(signed),
        "noop_observations": len(noops),
        "global_alpha": global_alpha,
        "max_abs_noop_target": max_abs_noop,
        "candidate_pairs_expected": len(expected_pairs),
        "candidate_pairs_observed": len(observed_pairs),
    }


def _check_reference_norm(
    manifest: StudyRunManifest,
    clean_records: Sequence[StateAuditCleanPassRecord],
    failures: list[str],
) -> dict[str, Any]:
    if not clean_records:
        failures.append("the clean-pass artifact is empty, so the reference norm cannot be checked")
        return {"recomputed_reference_norm": None, "recomputed_global_alpha": None}

    norms = np.asarray(
        [record.state_norm for record in sorted(clean_records, key=lambda r: r.variant_id)],
        dtype=np.float64,
    )
    recomputed = float(np.median(norms))
    recomputed_alpha = float(np.float64(manifest.norm_ratio) * np.float64(recomputed))

    if abs(recomputed - manifest.reference_norm) > NORM_TOLERANCE * max(1.0, recomputed):
        failures.append(
            f"the reference norm recomputes as {recomputed} from the recorded clean state norms "
            f"but the manifest records {manifest.reference_norm}"
        )
    if abs(recomputed_alpha - manifest.global_alpha) > NORM_TOLERANCE * max(1.0, recomputed_alpha):
        failures.append(
            f"the global alpha recomputes as {recomputed_alpha} but the manifest records "
            f"{manifest.global_alpha}"
        )

    dims = sorted({record.state_dim for record in clean_records})
    if dims != [manifest.diagnostics.state_dim]:
        failures.append(
            f"the clean pass recorded state dimensions {dims}, not {manifest.diagnostics.state_dim}"
        )
    wrong_shard = [
        record.variant_id
        for record in clean_records
        if record.state_shard_hash != manifest.states_hash
    ]
    if wrong_shard:
        failures.append(
            f"these clean-pass records cite a different state shard than the run: {wrong_shard[:5]}"
        )

    correct = sum(1 for record in clean_records if record.clean_correct)
    if correct != manifest.clean_correct_count:
        failures.append(
            f"the clean pass holds {correct} correct answers but the manifest records "
            f"{manifest.clean_correct_count}"
        )

    return {
        "recomputed_reference_norm": recomputed,
        "recomputed_global_alpha": recomputed_alpha,
        "median_method": MEDIAN_METHOD,
        "clean_state_count": len(clean_records),
    }


def verify_run(run_id: str, noop_tolerance: float | None = None) -> dict[str, Any]:
    """Verify one state-dependence run from its artifacts. Loads no model."""
    manifest = load_study_run_manifest(run_id)
    directory = run_dir(run_id)
    failures: list[str] = []

    tolerance = noop_tolerance
    if tolerance is None:
        try:
            tolerance = load_calibration_plan(manifest.calibration_plan_id).noop_tolerance
        except CalibrationPlanError as error:
            failures.append(f"calibration plan: {error}")
            tolerance = None

    _check_artifact_hashes(manifest, directory, failures)
    _check_cited_manifests(manifest, failures)

    observations = read_observations(run_id)
    clean_records = read_clean_pass(run_id)
    candidate_sets = read_candidate_sets(run_id)
    recorded_failures = read_failures(run_id)

    if len(recorded_failures) != manifest.failure_count:
        failures.append(
            f"{len(recorded_failures)} failures on disk, manifest records {manifest.failure_count}"
        )
    if len(clean_records) != manifest.observed_state_count:
        failures.append(
            f"{len(clean_records)} clean-pass records on disk, manifest records "
            f"{manifest.observed_state_count}"
        )
    if len(candidate_sets) != manifest.observed_prompt_count:
        failures.append(
            f"{len(candidate_sets)} candidate sets on disk, manifest records "
            f"{manifest.observed_prompt_count} prompts"
        )

    observation_report = _check_observations(manifest, observations, candidate_sets, failures)
    norm_report = _check_reference_norm(manifest, clean_records, failures)

    max_abs_noop = float(observation_report["max_abs_noop_target"])
    if tolerance is not None and max_abs_noop > tolerance:
        failures.append(
            f"the worst absolute no-op target is {max_abs_noop}, above the harness tolerance "
            f"{tolerance}; the harness rather than the model produced an effect"
        )
    if abs(max_abs_noop - manifest.diagnostics.max_abs_noop_target) > NORM_TOLERANCE:
        failures.append(
            "the manifest's recorded worst no-op target does not match the observations on disk"
        )

    if manifest.status != "complete":
        failures.append(
            f"run {run_id!r} is marked {manifest.status!r}; a run that did not finish is not a "
            "verified run"
        )

    return {
        "run_id": run_id,
        "run_role": manifest.run_role.value,
        "status": manifest.status,
        "valid": not failures,
        "failures": failures,
        "scientific_result": False,
        "manifest_hash": manifest.manifest_hash,
        "target_name": manifest.target_name,
        "model_id": manifest.model_id,
        "model_revision": manifest.model_revision,
        "tokenizer_revision": manifest.tokenizer_revision,
        "prompt_manifest_hash": manifest.prompt_manifest_hash,
        "direction_family_hash": manifest.direction_family_hash,
        "calibration_plan_hash": manifest.calibration_plan_hash,
        "layer": manifest.layer,
        "norm_ratio": manifest.norm_ratio,
        "reference_norm": manifest.reference_norm,
        "global_alpha": manifest.global_alpha,
        "noop_tolerance": tolerance,
        "observations_read": len(observations),
        "failures_read": len(recorded_failures),
        "state_dim": manifest.diagnostics.state_dim,
        "observation_checks": observation_report,
        "reference_norm_checks": norm_report,
        "diagnostics": manifest.diagnostics.model_dump(mode="json"),
        "notes": (
            "Artifact verification only. No model was loaded and no forward pass ran. A verified "
            "engineering run is a working pipeline, not a scientific result."
        ),
    }


def compare_runs(run_id: str, other_run_id: str, tolerance: float = 0.0) -> dict[str, Any]:
    """Compare two runs of the same role row by row.

    This is how cross-process determinism is measured: two executions of the same command into
    different run ids must produce the same prompts, the same candidate ids, and the same target
    for every pair. The default tolerance is exact, because a deterministic CPU forward pass on
    pinned weights has no reason to move at all; a looser tolerance may be passed to quantify
    drift rather than to excuse it.
    """
    left = load_study_run_manifest(run_id)
    right = load_study_run_manifest(other_run_id)

    mismatches: list[str] = []
    for name, a, b in (
        ("run_role", left.run_role.value, right.run_role.value),
        ("prompt_role", left.prompt_role.value, right.prompt_role.value),
        ("layer", left.layer, right.layer),
        ("norm_ratio", left.norm_ratio, right.norm_ratio),
        ("model_revision", left.model_revision, right.model_revision),
        ("prompt_manifest_hash", left.prompt_manifest_hash, right.prompt_manifest_hash),
        ("direction_family_hash", left.direction_family_hash, right.direction_family_hash),
        ("input_fingerprint", left.input_fingerprint, right.input_fingerprint),
    ):
        if a != b:
            mismatches.append(f"{name} differs: {a!r} versus {b!r}")
    if mismatches:
        raise StateAuditRunError(
            f"runs {run_id!r} and {other_run_id!r} were not executed against the same inputs, so "
            f"comparing them would not measure determinism: {mismatches}"
        )

    first = {(r.trial_id, r.candidate_id): r for r in read_observations(run_id)}
    second = {(r.trial_id, r.candidate_id): r for r in read_observations(other_run_id)}

    only_first = sorted(set(first) - set(second))
    only_second = sorted(set(second) - set(first))
    differences: list[str] = []
    worst_target_delta = 0.0
    worst_logit_delta = 0.0

    for key in sorted(set(first) & set(second)):
        a, b = first[key], second[key]
        target_delta = abs(a.delta_clean_top_margin - b.delta_clean_top_margin)
        worst_target_delta = max(worst_target_delta, target_delta)
        for label in sorted(a.intervened_logits):
            worst_logit_delta = max(
                worst_logit_delta,
                abs(a.intervened_logits[label] - b.intervened_logits[label]),
            )
        if target_delta > tolerance:
            differences.append(f"{key[0]}/{key[1]}: target differs by {target_delta:g}")
        if a.answer_flip != b.answer_flip:
            differences.append(f"{key[0]}/{key[1]}: answer_flip differs")
        if a.clean_preferred_label != b.clean_preferred_label:
            differences.append(f"{key[0]}/{key[1]}: clean preferred answer differs")

    identical = not only_first and not only_second and not differences
    return {
        "run_id": run_id,
        "other_run_id": other_run_id,
        "deterministic": identical,
        "tolerance": tolerance,
        "compared_observations": len(set(first) & set(second)),
        "only_in_first": only_first[:5],
        "only_in_second": only_second[:5],
        "max_abs_target_difference": worst_target_delta,
        "max_abs_logit_difference": worst_logit_delta,
        "reference_norm_difference": abs(left.reference_norm - right.reference_norm),
        "global_alpha_difference": abs(left.global_alpha - right.global_alpha),
        "differences": differences[:10],
        "scientific_result": False,
    }


__all__ = [
    "NORM_TOLERANCE",
    "compare_runs",
    "read_candidate_sets",
    "read_clean_pass",
    "read_failures",
    "read_observations",
    "verify_run",
]
