"""Resolving the final test: the one irreversible step in the study.

Everything before this point can be redone. Once the final-test interventions are applied, the
blinding is spent: the outcomes exist, and no amount of care afterwards can restore a state in
which the forecasts had not yet been checked against them. So this module is deliberately the most
suspicious code in the repository, and it refuses more than it accepts.

What it requires before touching the weights:

* a clean working tree, and the commit recorded **inside the hashed manifest**, because which code
  performed an irreversible step is part of what happened;
* the verified calibrated setting, taken from the calibration decision artifact rather than from a
  flag;
* exactly the expected prompt count, exactly the expected commitment count, and **zero reveals**;
* no outcome artifact of any kind already in the run directory.

What it does not do:

* run a clean forward. The clean logits and states come from the clean stage, so every delta is
  measured against exactly the clean output the forecasts were made against. `clean_forwards` on
  the manifest is a typed literal zero.
* select a candidate. All 17 are applied, so there is nothing to select, and the reveal is the
  no-selection kind.

544 intervened forwards: 32 prompts x (16 signed + 1 no-op).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..hashing import append_jsonl, atomic_write_json, hash_file, read_json, write_jsonl
from ..interventions.tensor_ops import InterventionShapeError
from ..logging_utils import info, warn
from ..models.capture import CaptureError
from ..models.scoring import LabelTokenError
from ..paths import (
    FORECAST_COMMITMENTS,
    FORECASTS,
    SELECTION_REVEALS,
    STATE_AUDIT_CANDIDATE_SETS,
    STATE_AUDIT_CLEAN_PASS,
    STATE_AUDIT_COMMITMENT_SUMMARY,
    STATE_AUDIT_FAILURES,
    STATE_AUDIT_OBSERVATIONS,
    STATE_AUDIT_PAIRING,
    STATE_AUDIT_RESOLUTION,
    STATE_AUDIT_RUN_MANIFEST,
    STATE_AUDIT_STATES,
    ensure_run_dir,
    run_dir,
)
from ..reproducibility import environment_snapshot, git_state
from ..schemas import (
    ArtifactHashRecord,
    CleanPassRunManifest,
    FinalTestResolutionManifest,
    PromptAssignment,
    PromptRole,
    StateAuditCandidateSet,
    StateAuditCleanPassRecord,
    StateAuditObservationRecord,
    StudyRunRole,
    compute_final_test_resolution_hash,
)
from ..tasks.loader import load_prepared_task
from ..trials.commitment import (
    existing_outcome_artifacts,
    read_commitments,
    read_forecasts,
    read_reveals,
    record_key,
    reveal_without_selection,
    verify_commitment_ordering,
    verify_run_commitments,
)
from ..trials.states import load_state
from .predict import direction_vectors
from .projection import load_projection_record
from .run import (
    CleanPass,
    RunInputs,
    StateAuditRunError,
    _failure,
    apply_candidate,
    inherited_strengths,
    load_run_config,
    load_run_model,
    resolve_run_inputs,
    resolve_run_label_tokens,
)

RESOLUTION_ALGORITHM_VERSION = "bluedot_final_test_resolution_v1.0"

# The preregistered commitment shape: 32 prompts x 16 method-and-condition records.
EXPECTED_RECORDS_PER_PROMPT = 16


class FinalTestResolutionError(RuntimeError):
    """Raised when the final test cannot be resolved, or must not be."""


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolutionPreconditions:
    """Everything checked before the model is loaded, kept so the report can cite it."""

    code_commit: str
    code_branch: str | None
    prompt_count: int
    commitment_count: int
    reveal_count: int
    forecast_count: int
    layer: int
    norm_ratio: float
    global_alpha: float
    reference_norm: float
    clean_run_manifest_hash: str
    commitment_summary_hash: str
    pairing_hash: str
    projection_hash: str


def check_working_tree() -> tuple[str, str | None]:
    """Refuse to resolve from a dirty tree, and return the commit that will be recorded.

    An irreversible step performed from uncommitted code cannot be reproduced from the repository,
    and the manifest would record a commit that does not describe what actually ran. This is the
    one place in the study where a dirty tree is a hard stop rather than something recorded and
    carried forward.
    """
    state = git_state()
    commit = state.get("commit")
    dirty = state.get("dirty")
    if not commit:
        raise FinalTestResolutionError(
            "could not read the current git commit. The final test records the commit that "
            "performed it inside its hashed manifest, so it cannot run without one."
        )
    if dirty is None:
        raise FinalTestResolutionError(
            "could not determine whether the working tree is clean; refusing to resolve the final "
            "test from an unknown state"
        )
    if dirty:
        raise FinalTestResolutionError(
            "the working tree has uncommitted changes. Resolving the final test is irreversible "
            "and the manifest records the commit that did it, so commit the implementation first "
            "and rerun. This is the only step in the study that refuses a dirty tree."
        )
    return commit, state.get("branch")


def check_verified_setting(
    inputs: RunInputs, expected_layer: int, expected_ratio: float, expected_alpha: float
) -> tuple[float, float]:
    """Require the calibrated setting, read from the decision artifact.

    The expected values are passed in and compared rather than trusted, so a config that drifted
    from the decision, or a decision that was replaced, stops the run instead of silently
    resolving the final test at a strength the study never calibrated.
    """
    reference_norm, ratio_alphas = inherited_strengths(inputs)
    ratio, alpha = ratio_alphas[0]

    if inputs.config.layer != expected_layer:
        raise FinalTestResolutionError(
            f"the final test is fixed at layer {expected_layer}; this run is configured for "
            f"{inputs.config.layer}"
        )
    if float(ratio) != float(expected_ratio):
        raise FinalTestResolutionError(
            f"the calibrated ratio is {expected_ratio}; the decision supplies {ratio}"
        )
    if abs(alpha - expected_alpha) > 1e-9 * max(1.0, abs(expected_alpha)):
        raise FinalTestResolutionError(
            f"the calibrated alpha is {expected_alpha}; the decision supplies {alpha}"
        )
    return reference_norm, alpha


def check_commitment_state(run_id: str, expected_prompts: int) -> tuple[int, int, int]:
    """Require 512 verified commitments, zero reveals, and no outcome artifact.

    Zero reveals is as important as the commitment count. A reveal discloses a salt; if any exist
    before resolution then the forecasts were already checkable against nothing in particular, and
    more to the point the run is not in the state the protocol describes.
    """
    outcomes = existing_outcome_artifacts(run_id)
    if outcomes:
        raise FinalTestResolutionError(
            f"run {run_id!r} already holds outcome artifacts {outcomes}. The final test has "
            "already been resolved; resolving it again would overwrite the only record of what "
            "happened. Use a fresh run directory and recommit if the run must be redone, and "
            "record the discard in docs/experiment_log.md."
        )

    commitments = read_commitments(run_id)
    forecasts = read_forecasts(run_id)
    reveals = read_reveals(run_id)

    expected_commitments = expected_prompts * EXPECTED_RECORDS_PER_PROMPT
    if len(commitments) != expected_commitments:
        raise FinalTestResolutionError(
            f"run {run_id!r} holds {len(commitments)} commitments; the preregistered shape is "
            f"{expected_prompts} prompts x {EXPECTED_RECORDS_PER_PROMPT} method-and-condition "
            f"records = {expected_commitments}"
        )
    if len(forecasts) != expected_commitments:
        raise FinalTestResolutionError(
            f"run {run_id!r} holds {len(forecasts)} forecast records for {len(commitments)} "
            "commitments; every commitment must have exactly one forecast"
        )
    if reveals:
        raise FinalTestResolutionError(
            f"run {run_id!r} already holds {len(reveals)} reveals. A salt disclosed before "
            "resolution means the run is not at the commitment checkpoint."
        )

    keys = [record_key(commitment) for commitment in commitments]
    if len(set(keys)) != len(keys):
        raise FinalTestResolutionError("the commitments repeat a key; the set is not verifiable")
    forecast_keys = {record_key(forecast) for forecast in forecasts}
    missing = sorted(set(keys) - forecast_keys)
    if missing:
        raise FinalTestResolutionError(f"these commitments have no forecast: {missing[:3]}")

    salts = run_dir(run_id) / "private_payloads" / "salts"
    salt_count = len(list(salts.glob("*.salt"))) if salts.exists() else 0
    if salt_count != expected_commitments:
        raise FinalTestResolutionError(
            f"{salt_count} salt files for {expected_commitments} commitments; a commitment whose "
            "salt is missing can never be verified"
        )
    return len(commitments), len(forecasts), len(reveals)


# ---------------------------------------------------------------------------
# Rebuilding the clean pass without a forward
# ---------------------------------------------------------------------------


def load_clean_manifest(run_id: str) -> CleanPassRunManifest:
    path = run_dir(run_id) / STATE_AUDIT_RUN_MANIFEST
    if not path.exists():
        raise FinalTestResolutionError(f"no clean-stage manifest at {path}")
    try:
        manifest = CleanPassRunManifest.model_validate(read_json(path))
    except Exception as error:
        raise FinalTestResolutionError(
            f"{path} is not a valid clean-stage manifest: {error}"
        ) from (error)
    if manifest.run_role is not StudyRunRole.FINAL_TEST_UNRESOLVED:
        raise FinalTestResolutionError(
            f"run {run_id!r} is not an unresolved final-test stage; its role is "
            f"{manifest.run_role.value!r}"
        )
    if manifest.status != "complete":
        raise FinalTestResolutionError(
            f"the clean stage of run {run_id!r} is marked {manifest.status!r}; resolving on top of "
            "an incomplete clean pass would measure deltas against missing baselines"
        )
    return manifest


def rebuild_clean_passes(
    inputs: RunInputs, run_id: str, manifest: CleanPassRunManifest
) -> list[CleanPass]:
    """Reconstruct the clean pass from stored artifacts. Runs no forward.

    Every field comes from the clean-stage records, so the deltas this resolution computes are
    measured against exactly the clean outputs the forecasts were made against. Recomputing them
    would be a different clean run, and any drift would land silently in every target.
    """
    path = run_dir(run_id) / STATE_AUDIT_CLEAN_PASS
    if not path.exists():
        raise FinalTestResolutionError(f"no clean pass at {path}")
    from ..hashing import read_jsonl

    records = [StateAuditCleanPassRecord.model_validate(row) for row in read_jsonl(path)]
    if hash_file(path) != manifest.clean_pass_hash:
        raise FinalTestResolutionError(
            "the clean-pass file does not match the hash its manifest records; the baselines the "
            "forecasts were made against have changed"
        )

    states_path = run_dir(run_id) / STATE_AUDIT_STATES
    if not states_path.exists():
        raise FinalTestResolutionError(f"no captured states at {states_path}")
    if hash_file(states_path) != manifest.states_hash:
        raise FinalTestResolutionError(
            "the state shard does not match the hash its manifest records"
        )

    assignments: dict[str, PromptAssignment] = {
        assignment.variant_id: assignment
        for assignment in inputs.manifest.by_role(PromptRole.FINAL_TEST)
    }
    items, variants = load_prepared_task(inputs.task_config.name)
    texts = {variant.variant_id: variant.prompt_text for variant in variants}
    answers = {item.item_id: item.answer_label for item in items}

    passes: list[CleanPass] = []
    for record in sorted(records, key=lambda r: r.variant_id):
        assignment = assignments.get(record.variant_id)
        if assignment is None:
            raise FinalTestResolutionError(
                f"{record.variant_id} is not a final-test prompt in the frozen manifest"
            )
        text = texts.get(record.variant_id)
        if text is None:
            raise FinalTestResolutionError(f"{record.variant_id} is not in the prepared task")
        state = load_state(states_path, record.state_id, dtype=torch.float32)
        passes.append(
            CleanPass(
                assignment=assignment,
                trial_id=record.trial_id,
                prompt_text=text,
                dataset_answer_label=answers[record.item_id],
                clean_logits=dict(record.clean_logits),
                clean_preferred_label=record.clean_preferred_label,
                clean_top_margin=record.clean_top_margin,
                clean_entropy=record.clean_entropy,
                clean_correct=record.clean_correct,
                state=state,
                state_id=record.state_id,
                state_norm=record.state_norm,
                prompt_token_count=record.prompt_token_count,
                position_absolute=record.position_absolute,
            )
        )
    return passes


def load_frozen_candidate_sets(run_id: str) -> dict[str, StateAuditCandidateSet]:
    """The candidate sets frozen at commitment time. Never rebuilt here.

    Rebuilding them would be deterministic and would almost certainly agree, but "almost certainly"
    is not the standard: the forecasts name these candidate ids, so the resolution must apply the
    candidates that were committed to, read from disk.
    """
    from ..hashing import read_jsonl

    path = run_dir(run_id) / STATE_AUDIT_CANDIDATE_SETS
    if not path.exists():
        raise FinalTestResolutionError(f"no frozen candidate sets at {path}")
    sets = {}
    for row in read_jsonl(path):
        candidate_set = StateAuditCandidateSet.model_validate(row)
        sets[candidate_set.trial_id] = candidate_set
    if not sets:
        raise FinalTestResolutionError(f"{path} holds no candidate sets")
    return sets


def check_forecasts_cover_candidates(
    run_id: str, candidate_sets: dict[str, StateAuditCandidateSet]
) -> None:
    """Every committed forecast must name exactly the frozen candidates for its prompt."""
    expected = {
        trial_id: {candidate.candidate_id for candidate in candidate_set.candidates}
        for trial_id, candidate_set in candidate_sets.items()
    }
    for forecast in read_forecasts(run_id):
        wanted = expected.get(forecast.trial_id)
        if wanted is None:
            raise FinalTestResolutionError(
                f"forecast for {forecast.trial_id} has no frozen candidate set"
            )
        named = {candidate.intervention_id for candidate in forecast.candidate_forecasts}
        if named != wanted:
            raise FinalTestResolutionError(
                f"the forecast for {forecast.trial_id} under "
                f"{forecast.method_id}/{forecast.state_condition.value}/"
                f"{forecast.condition_index} does not cover the frozen candidates exactly"
            )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_final_test(
    config_path: str | Path,
    run_id: str,
    expected_layer: int,
    expected_ratio: float,
    expected_alpha: float,
) -> dict[str, Any]:
    """Apply all 17 candidates to every final-test prompt, then reveal and verify.

    Irreversible. Every guard runs before the model is loaded.
    """
    config = load_run_config(config_path)
    if config.prompt_role is not PromptRole.FINAL_TEST:
        raise FinalTestResolutionError(
            f"the final test resolves the {PromptRole.FINAL_TEST.value!r} prompts; the config "
            f"names {config.prompt_role.value!r}"
        )

    code_commit, code_branch = check_working_tree()
    inputs = resolve_run_inputs(config_path)
    reference_norm, alpha = check_verified_setting(
        inputs, expected_layer, expected_ratio, expected_alpha
    )
    clean_manifest = load_clean_manifest(run_id)
    if clean_manifest.observed_prompt_count != config.expected_prompt_count:
        raise FinalTestResolutionError(
            f"the clean stage captured {clean_manifest.observed_prompt_count} prompts but the "
            f"config expects {config.expected_prompt_count}"
        )

    commitment_count, forecast_count, reveal_count = check_commitment_state(
        run_id, config.expected_prompt_count
    )
    candidate_sets = load_frozen_candidate_sets(run_id)
    if len(candidate_sets) != config.expected_prompt_count:
        raise FinalTestResolutionError(
            f"{len(candidate_sets)} frozen candidate sets for {config.expected_prompt_count} "
            "prompts"
        )
    check_forecasts_cover_candidates(run_id, candidate_sets)

    directory = ensure_run_dir(run_id)
    summary_path = directory / STATE_AUDIT_COMMITMENT_SUMMARY
    pairing_path = directory / STATE_AUDIT_PAIRING
    for path in (summary_path, pairing_path):
        if not path.exists():
            raise FinalTestResolutionError(f"missing {path.name}; the run is not at the checkpoint")

    projection_record = load_projection_record(
        config.projection_id or "bluedot_state_dependence_projection_v1"
    )
    started_at = datetime.now(UTC)

    # Everything above is model-free. Past this line the study is spending its blinding.
    model = load_run_model(inputs)
    label_token_ids = resolve_run_label_tokens(inputs, model)
    clean_passes = rebuild_clean_passes(inputs, run_id, clean_manifest)
    if len(clean_passes) != config.expected_prompt_count:
        raise FinalTestResolutionError(
            f"rebuilt {len(clean_passes)} clean passes for {config.expected_prompt_count} prompts"
        )

    vectors = _load_vectors(inputs, model)
    observations: list[StateAuditObservationRecord] = []
    failures: list[dict[str, Any]] = []
    reconstruction_errors: list[float] = []
    forwards = 0

    for clean in clean_passes:
        candidate_set = candidate_sets.get(clean.trial_id)
        if candidate_set is None:
            failures.append(
                _failure(
                    clean.trial_id,
                    clean.assignment,
                    stage="candidates",
                    error_type="MissingCandidateSet",
                    message="no frozen candidate set for this prompt",
                    layer=config.layer,
                )
            )
            append_jsonl(directory / STATE_AUDIT_FAILURES, failures[-1])
            continue

        for candidate in candidate_set.candidates:
            try:
                applied = apply_candidate(
                    clean, candidate, inputs, model, label_token_ids, vectors, run_id
                )
                forwards += 1
            except (
                StateAuditRunError,
                CaptureError,
                InterventionShapeError,
                LabelTokenError,
                KeyError,
                ValueError,
            ) as error:
                failure = _failure(
                    clean.trial_id,
                    clean.assignment,
                    stage="intervention",
                    error_type=type(error).__name__,
                    message=str(error),
                    layer=candidate.layer,
                    candidate_id=candidate.candidate_id,
                    norm_ratio=candidate.norm_ratio,
                )
                failures.append(failure)
                append_jsonl(directory / STATE_AUDIT_FAILURES, failure)
                warn(
                    "final-test intervention failed",
                    trial_id=clean.trial_id,
                    candidate_id=candidate.candidate_id,
                    error=type(error).__name__,
                )
                continue

            observations.append(applied.observation)
            reconstruction_errors.append(applied.reconstruction_error)
            append_jsonl(directory / STATE_AUDIT_OBSERVATIONS, applied.observation)

    if not observations:
        raise FinalTestResolutionError(
            f"no candidate was applied successfully in run {run_id!r}; {len(failures)} failures "
            f"were recorded in {STATE_AUDIT_FAILURES}"
        )

    # Independent recheck of every stored target, from the logits stored beside it. The record's
    # own validator already did this at construction; doing it again from what reached disk is the
    # check a third party would run.
    verify_stored_targets(run_id)

    reveals = reveal_all(run_id)
    verification = verify_run_commitments(run_id)
    ordering = verify_commitment_ordering(run_id)

    manifest = _build_resolution_manifest(
        inputs=inputs,
        run_id=run_id,
        model=model,
        clean_manifest=clean_manifest,
        observations=observations,
        reconstruction_errors=reconstruction_errors,
        failures=failures,
        forwards=forwards,
        reference_norm=reference_norm,
        alpha=alpha,
        projection_hash=projection_record.matrix_hash,
        summary_hash=hash_file(summary_path),
        pairing_hash=hash_file(pairing_path),
        clean_manifest_hash=clean_manifest.manifest_hash,
        verification=verification,
        reveal_count=len(reveals),
        directory=directory,
        code_commit=code_commit,
        code_branch=code_branch,
        started_at=started_at,
    )
    atomic_write_json(directory / STATE_AUDIT_RESOLUTION, manifest.model_dump(mode="json"))

    info(
        "resolved the final test",
        run_id=run_id,
        status=manifest.status,
        observations=len(observations),
        forwards=forwards,
        failures=len(failures),
        reveals=len(reveals),
        commitments_verified=verification["verified"],
    )
    report = resolution_report(manifest, directory)
    report["commitment_verification"] = verification
    report["ordering"] = ordering
    report["preconditions"] = {
        "code_commit": code_commit,
        "code_branch": code_branch,
        "working_tree_clean": True,
        "commitments": commitment_count,
        "forecasts": forecast_count,
        "reveals_before_resolution": reveal_count,
        "layer": expected_layer,
        "norm_ratio": expected_ratio,
        "global_alpha": expected_alpha,
    }
    return report


def _load_vectors(inputs: RunInputs, model: Any) -> dict[str, torch.Tensor]:
    arrays = direction_vectors(inputs.config.direction_family_id)
    return {
        name: torch.from_numpy(np.asarray(array, dtype=np.float32)).to(
            dtype=model.dtype, device=model.device
        )
        for name, array in arrays.items()
    }


def verify_stored_targets(run_id: str) -> int:
    """Recompute every stored target from its own logits. Loads no model.

    Reading the observations back through `StateAuditObservationRecord` revalidates each one, so a
    row whose delta does not follow from the logits beside it fails to parse. Counting them here
    makes the check explicit rather than a side effect of loading.
    """
    from ..hashing import read_jsonl

    path = run_dir(run_id) / STATE_AUDIT_OBSERVATIONS
    if not path.exists():
        raise FinalTestResolutionError(f"no observations at {path}")
    checked = 0
    for index, row in enumerate(read_jsonl(path), start=1):
        try:
            StateAuditObservationRecord.model_validate(row)
        except Exception as error:
            raise FinalTestResolutionError(
                f"{path}:{index}: a stored target does not recompute from its own logits: {error}"
            ) from error
        checked += 1
    return checked


def reveal_all(run_id: str) -> list[Any]:
    """Reveal every commitment without selecting a candidate.

    All 17 candidates were resolved, so there is nothing for a seed to choose. Each reveal
    discloses its salt and records whether the commitment recomputed; a failure is recorded rather
    than raised, because the record is the evidence.
    """
    commitments = {record_key(c): c for c in read_commitments(run_id)}
    forecasts = {record_key(f): f for f in read_forecasts(run_id)}
    reveals = []
    for key in sorted(commitments):
        forecast = forecasts.get(key)
        if forecast is None:
            raise FinalTestResolutionError(f"commitment {key} has no forecast to reveal")
        reveals.append(reveal_without_selection(run_id, commitments[key], forecast))
    return reveals


def _build_resolution_manifest(
    inputs: RunInputs,
    run_id: str,
    model: Any,
    clean_manifest: CleanPassRunManifest,
    observations: Sequence[StateAuditObservationRecord],
    reconstruction_errors: Sequence[float],
    failures: Sequence[dict[str, Any]],
    forwards: int,
    reference_norm: float,
    alpha: float,
    projection_hash: str,
    summary_hash: str,
    pairing_hash: str,
    clean_manifest_hash: str,
    verification: dict[str, Any],
    reveal_count: int,
    directory: Path,
    code_commit: str,
    code_branch: str | None,
    started_at: datetime,
) -> FinalTestResolutionManifest:
    config = inputs.config
    signed = [record for record in observations if not record.is_noop]
    noops = [record for record in observations if record.is_noop]
    candidates = config.candidates_per_prompt
    failures_path = directory / STATE_AUDIT_FAILURES

    payload: dict[str, Any] = {
        "schema_version": FinalTestResolutionManifest.model_fields["schema_version"].default,
        "study_id": config.study_id,
        "run_id": run_id,
        "run_role": StudyRunRole.FINAL_TEST_RESOLVED.value,
        "model_id": model.spec.model_id,
        "model_revision": model.spec.revision,
        "tokenizer_revision": model.spec.revision,
        "dtype": model.spec.dtype,
        "device": model.spec.device,
        "target_name": "delta_clean_top_margin",
        "prompt_manifest_hash": inputs.manifest.manifest_hash,
        "direction_family_hash": inputs.family.family_hash,
        "calibration_plan_hash": inputs.plan_hash,
        "projection_hash": projection_hash,
        "clean_run_manifest_hash": clean_manifest_hash,
        "commitment_summary_hash": summary_hash,
        "pairing_hash": pairing_hash,
        "layer": config.layer,
        "capture_position": config.capture_position,
        "norm_ratio": float(config.ratio_grid[0]),
        "global_alpha": alpha,
        "reference_norm": reference_norm,
        "expected_prompt_count": config.expected_prompt_count,
        "expected_candidates_per_prompt": candidates,
        "expected_signed_observations": config.expected_prompt_count * (candidates - 1),
        "expected_noop_observations": config.expected_prompt_count,
        "expected_intervened_forwards": config.expected_prompt_count * candidates,
        "observed_prompt_count": len({record.trial_id for record in observations}),
        "observed_signed_observations": len(signed),
        "observed_noop_observations": len(noops),
        "observed_intervened_forwards": forwards,
        "clean_forwards": 0,
        "failure_count": len(failures),
        "max_abs_noop_target": max(
            (abs(record.delta_clean_top_margin) for record in noops), default=0.0
        ),
        "max_abs_noop_delta_norm": max((abs(r.delta_norm or 0.0) for r in noops), default=0.0),
        "max_intervention_reconstruction_error": max(reconstruction_errors, default=0.0),
        "flip_count": sum(1 for record in signed if record.answer_flip),
        "observations_hash": hash_file(directory / STATE_AUDIT_OBSERVATIONS),
        "failures_hash": hash_file(failures_path) if failures_path.exists() else None,
        "reveals_hash": hash_file(directory / SELECTION_REVEALS),
        "commitments_checked": int(verification["checked"]),
        "commitments_verified": bool(verification["verified"]),
        "reveal_count": reveal_count,
        "config_hash": inputs.config_hash,
        "code_commit": code_commit,
        "code_dirty": False,
    }

    complete = (
        not failures
        and payload["observed_prompt_count"] == config.expected_prompt_count
        and payload["observed_signed_observations"] == payload["expected_signed_observations"]
        and payload["observed_noop_observations"] == payload["expected_noop_observations"]
        and forwards == payload["expected_intervened_forwards"]
        and payload["commitments_verified"]
    )
    payload["status"] = "complete" if complete else "failed"

    provenance = [
        ArtifactHashRecord(
            path=name,
            hash=hash_file(directory / name),
            size_bytes=(directory / name).stat().st_size,
            kind=kind,
        )
        for name, kind in (
            (STATE_AUDIT_OBSERVATIONS, "state_audit_observations"),
            (STATE_AUDIT_FAILURES, "state_audit_failures"),
            (SELECTION_REVEALS, "selection_reveals"),
            (FORECASTS, "forecasts"),
            (FORECAST_COMMITMENTS, "forecast_commitments"),
        )
        if (directory / name).exists()
    ]

    return FinalTestResolutionManifest(
        **payload,
        manifest_hash=compute_final_test_resolution_hash(payload),
        config_path=str(config.name),
        code_branch=code_branch,
        environment=environment_snapshot(),
        provenance=provenance,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        notes=(
            "Final-test resolution. The clean logits and states were reused from the clean stage, "
            "so no clean forward ran here and every delta is measured against exactly the baseline "
            "the forecasts were made against. All candidates were applied, so every reveal is a "
            "no-selection reveal."
        ),
    )


def resolution_report(manifest: FinalTestResolutionManifest, directory: Path) -> dict[str, Any]:
    return {
        "run_id": manifest.run_id,
        "run_role": manifest.run_role.value,
        "status": manifest.status,
        "run_path": str(directory),
        "manifest_hash": manifest.manifest_hash,
        "algorithm_version": RESOLUTION_ALGORITHM_VERSION,
        "model_id": manifest.model_id,
        "model_revision": manifest.model_revision,
        "layer": manifest.layer,
        "norm_ratio": manifest.norm_ratio,
        "global_alpha": manifest.global_alpha,
        "reference_norm": manifest.reference_norm,
        "counts": {
            "expected_prompts": manifest.expected_prompt_count,
            "observed_prompts": manifest.observed_prompt_count,
            "expected_signed_observations": manifest.expected_signed_observations,
            "observed_signed_observations": manifest.observed_signed_observations,
            "expected_noop_observations": manifest.expected_noop_observations,
            "observed_noop_observations": manifest.observed_noop_observations,
            "expected_intervened_forwards": manifest.expected_intervened_forwards,
            "observed_intervened_forwards": manifest.observed_intervened_forwards,
            "clean_forwards": manifest.clean_forwards,
            "failures": manifest.failure_count,
            "reveals": manifest.reveal_count,
            "commitments_checked": manifest.commitments_checked,
        },
        "commitments_verified": manifest.commitments_verified,
        "integrity": {
            "max_abs_noop_target": manifest.max_abs_noop_target,
            "max_abs_noop_delta_norm": manifest.max_abs_noop_delta_norm,
            "max_intervention_reconstruction_error": (
                manifest.max_intervention_reconstruction_error
            ),
            "flip_count": manifest.flip_count,
        },
        "artifact_hashes": {
            "observations": manifest.observations_hash,
            "failures": manifest.failures_hash,
            "reveals": manifest.reveals_hash,
            "clean_run_manifest": manifest.clean_run_manifest_hash,
            "commitment_summary": manifest.commitment_summary_hash,
            "pairing": manifest.pairing_hash,
            "projection": manifest.projection_hash,
        },
        "code_commit": manifest.code_commit,
        "code_dirty": manifest.code_dirty,
        "notes": manifest.notes,
    }


def load_resolution_manifest(run_id: str) -> FinalTestResolutionManifest:
    path = run_dir(run_id) / STATE_AUDIT_RESOLUTION
    if not path.exists():
        raise FinalTestResolutionError(f"no final-test resolution at {path}")
    try:
        return FinalTestResolutionManifest.model_validate(read_json(path))
    except Exception as error:
        raise FinalTestResolutionError(
            f"{path} is not a valid final-test resolution: {error}"
        ) from error


def write_observation_rows(run_id: str, rows: Sequence[Any]) -> Path:
    """Test helper: write observation rows through the same writer resolution uses."""
    return write_jsonl(run_dir(run_id) / STATE_AUDIT_OBSERVATIONS, rows)


__all__ = [
    "EXPECTED_RECORDS_PER_PROMPT",
    "RESOLUTION_ALGORITHM_VERSION",
    "FinalTestResolutionError",
    "ResolutionPreconditions",
    "check_commitment_state",
    "check_forecasts_cover_candidates",
    "check_verified_setting",
    "check_working_tree",
    "load_clean_manifest",
    "load_frozen_candidate_sets",
    "load_resolution_manifest",
    "rebuild_clean_passes",
    "resolution_report",
    "resolve_final_test",
    "reveal_all",
    "verify_stored_targets",
]
