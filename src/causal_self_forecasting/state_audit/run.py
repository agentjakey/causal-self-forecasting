"""Executing one state-dependence run against real weights.

The shape of a run, in order:

1. read the frozen prompt manifest, direction family, and calibration plan, and refuse anything
   that does not match what the config asserts;
2. load the pinned model, and refuse a revision, hidden dimension, or answer-token resolution
   that differs from the family the directions were built against;
3. one clean forward per prompt, capturing the residual stream at the study layer;
4. freeze one global alpha from the median clean state norm and the run's ratio;
5. build the selected-strength candidate set per prompt and apply every candidate;
6. compute and store `delta_clean_top_margin` for each, preserving every failure.

Two properties are worth stating because they are easy to lose.

**One inference path.** Every forward here goes through `models.capture`, every score through
`models.scoring.score_logits`, every intervention through `interventions.tensor_ops`. There is
no second implementation of "run the model and read the answer logits", so a run cannot quietly
measure something the harness controls never validated.

**One global alpha.** The alpha is computed once from the median clean state norm across this
run's prompts and applied unchanged to every prompt and every signed direction. It is never
`ratio * ||h_prompt||`. A per-prompt strength would put the prompt's state norm into the
published candidate strength, which is exactly the leak the arm's design exists to avoid.

An engineering smoke run is plumbing validation. It carries `scientific_result: false`, it
selects no ratio and no layer, and its effect sizes are diagnostics rather than evidence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..calibration.plan import CalibrationPlanError, load_calibration_plan
from ..calibration.strength import MEDIAN_METHOD, StrengthError, alpha_for_ratio, reference_norm
from ..config import (
    ConfigError,
    ModelConfig,
    StateAuditRunConfig,
    TaskConfig,
    config_hash,
    load_config,
    repo_root,
)
from ..hashing import (
    append_jsonl,
    atomic_write_json,
    canonical_json_bytes,
    hash_file,
    hash_object,
    read_json,
    sha256_hex,
    write_jsonl,
)
from ..interventions.direction_family import (
    DirectionFamilyError,
    load_direction_family,
    vector_content_hash,
    verify_direction_family,
)
from ..interventions.directions import DirectionStore
from ..interventions.tensor_ops import InterventionPayload, InterventionShapeError
from ..logging_utils import info, warn
from ..models.capture import CaptureError, capture_hidden_states, run_with_intervention
from ..models.loader import LoadedModel, ModelLoadError, load_model
from ..models.scoring import LabelTokenError, resolve_label_token_ids, score_logits
from ..paths import (
    STATE_AUDIT_CANDIDATE_SETS,
    STATE_AUDIT_CLEAN_PASS,
    STATE_AUDIT_FAILURES,
    STATE_AUDIT_OBSERVATIONS,
    STATE_AUDIT_RUN_MANIFEST,
    STATE_AUDIT_STATE_REFS,
    STATE_AUDIT_STATES,
    directions_dir,
    ensure_run_dir,
    run_dir,
)
from ..reproducibility import environment_snapshot
from ..schemas import (
    ArtifactHashRecord,
    DirectionFamilyRecord,
    InterventionSpec,
    PromptAssignment,
    PromptManifest,
    PromptRole,
    StateAuditCandidate,
    StateAuditCandidateKind,
    StateAuditCandidateSet,
    StateAuditCleanPassRecord,
    StateAuditObservationRecord,
    StateAuditRunDiagnostics,
    StudyRunManifest,
    StudyRunRole,
    compute_study_run_hash,
)
from ..state_audit_target import (
    ANSWER_LABELS,
    TARGET_NAME,
    preferred_label,
    preferred_margin,
    state_audit_target,
)
from ..tasks.loader import TaskLoadError, load_prepared_task
from ..tasks.prompt_manifest import (
    PromptManifestError,
    load_prompt_manifest,
    verify_prompt_manifest,
)
from ..trials.states import StateShardWriter
from .candidates import (
    CANDIDATE_ALGORITHM_VERSION,
    NOOP_DIRECTION_REF,
    CandidateBuildError,
    DirectionRef,
    build_state_audit_candidate_set,
    candidate_analysis_role,
    candidate_mechanism,
    selected_strength_templates,
)

RUN_ALGORITHM_VERSION = "bluedot_state_audit_run_v1.0"
PERCENTILE_METHOD = "numpy.quantile(method='linear')"

# The smoke run's frozen parameters. Preregistered in
# docs/bluedot/preregistration_state_dependence.md and fixed in the execution decision tree at
# G3, so that the arbitrary choice of ratio cannot be made later with numbers in view.
SMOKE_LAYER = 13
SMOKE_NORM_RATIO = 0.10

# `np.savez` turns array names into keyword arguments, and the writer already refuses ids that
# collide with its own parameters. Nothing here needs a second copy of that list.
_MECHANISM_VERSION = "1.0"


class StateAuditRunError(RuntimeError):
    """Raised when a state-dependence run cannot be executed or trusted."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunInputs:
    """Everything a run reads before a model is loaded."""

    config: StateAuditRunConfig
    config_path: Path
    config_hash: str
    model_config: ModelConfig
    task_config: TaskConfig
    manifest: PromptManifest
    family: DirectionFamilyRecord
    plan_id: str
    plan_hash: str
    assignments: list[PromptAssignment]
    directions: list[DirectionRef]

    @property
    def input_fingerprint(self) -> str:
        """Identity of the inputs a run was executed against.

        Two runs with the same fingerprint were asked the same question of the same weights. A
        rerun under an existing run id compares this rather than the config path, so moving a
        config or renaming a run does not read as a different experiment.
        """
        return hash_object(
            {
                "algorithm_version": RUN_ALGORITHM_VERSION,
                "candidate_algorithm_version": CANDIDATE_ALGORITHM_VERSION,
                "config_hash": self.config_hash,
                "study_id": self.config.study_id,
                "run_role": self.config.run_role.value,
                "prompt_role": self.config.prompt_role.value,
                "model_id": self.model_config.model_id,
                "model_revision": self.model_config.revision,
                "prompt_manifest_hash": self.manifest.manifest_hash,
                "direction_family_hash": self.family.family_hash,
                "calibration_plan_hash": self.plan_hash,
                "layer": self.config.layer,
                "capture_position": self.config.capture_position,
                "norm_ratio": float(self.config.norm_ratio),
                "master_seed": self.config.master_seed,
                "prompt_count": len(self.assignments),
                "direction_count": len(self.directions),
            }
        )


def load_run_config(config_path: str | Path) -> StateAuditRunConfig:
    try:
        return load_config(config_path, StateAuditRunConfig)
    except ConfigError as error:
        raise StateAuditRunError(str(error)) from error


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root())).replace("\\", "/")
    except ValueError:
        return path.name


def resolve_run_inputs(config_path: str | Path) -> RunInputs:
    """Read and cross-check every frozen artifact a run depends on. Loads no model."""
    resolved_path = Path(config_path)
    config = load_run_config(resolved_path)

    try:
        model_config = load_config(config.model_ref, ModelConfig)
        task_config = load_config(config.task_ref, TaskConfig)
    except ConfigError as error:
        raise StateAuditRunError(
            f"state-audit config {config.name!r} references a config that does not load: {error}"
        ) from error

    try:
        manifest = load_prompt_manifest(config.prompt_manifest_id)
    except PromptManifestError as error:
        raise StateAuditRunError(str(error)) from error

    if manifest.task_name != task_config.name:
        raise StateAuditRunError(
            f"prompt manifest {manifest.manifest_id!r} was built against task "
            f"{manifest.task_name!r} but the run config names {task_config.name!r}"
        )

    task_check = verify_prompt_manifest(manifest)
    if not task_check["valid"]:
        raise StateAuditRunError(
            f"prompt manifest {manifest.manifest_id!r} no longer matches the prepared task "
            f"artifacts on disk ({task_check['mismatches']}); the split refers to a pool that "
            "has changed"
        )

    expected_roles = {role.value: count for role, count in config.expected_role_counts.items()}
    if dict(manifest.role_counts) != expected_roles:
        raise StateAuditRunError(
            f"prompt manifest {manifest.manifest_id!r} has role counts "
            f"{dict(sorted(manifest.role_counts.items()))} but the run config expects "
            f"{dict(sorted(expected_roles.items()))}"
        )

    assignments = manifest.by_role(config.prompt_role)
    if len(assignments) != config.expected_prompt_count:
        raise StateAuditRunError(
            f"the {config.prompt_role.value} role holds {len(assignments)} prompts but the run "
            f"config expects {config.expected_prompt_count}"
        )
    outside = [a.variant_id for a in assignments if a.role is not config.prompt_role]
    if outside:
        raise StateAuditRunError(f"these prompts are outside the run's role: {outside[:5]}")

    try:
        family = load_direction_family(config.direction_family_id)
    except DirectionFamilyError as error:
        raise StateAuditRunError(str(error)) from error

    if len(family.directions) != config.expected_direction_count:
        raise StateAuditRunError(
            f"direction family {family.family_id!r} holds {len(family.directions)} directions "
            f"but the run config expects {config.expected_direction_count}"
        )
    if family.hidden_dim != config.expected_hidden_dim:
        raise StateAuditRunError(
            f"direction family {family.family_id!r} is {family.hidden_dim}-dimensional but the "
            f"run config expects {config.expected_hidden_dim}"
        )
    if family.model_id != model_config.model_id or family.model_revision != model_config.revision:
        raise StateAuditRunError(
            f"the direction family was built against {family.model_id}@{family.model_revision} "
            f"but the run config pins {model_config.model_id}@{model_config.revision}; "
            "directions built from different weights are different stimuli"
        )

    try:
        plan = load_calibration_plan(config.calibration_plan_id)
    except CalibrationPlanError as error:
        raise StateAuditRunError(str(error)) from error

    if plan.prompt_manifest_hash != manifest.manifest_hash:
        raise StateAuditRunError(
            "the calibration plan cites a different prompt manifest than the run is using"
        )
    if plan.direction_family_hash != family.family_hash:
        raise StateAuditRunError(
            "the calibration plan cites a different direction family than the run is using"
        )
    if plan.model_revision != model_config.revision:
        raise StateAuditRunError(
            f"the calibration plan pins revision {plan.model_revision} but the run config pins "
            f"{model_config.revision}"
        )
    if plan.target_name != config.target:
        raise StateAuditRunError(
            f"the calibration plan targets {plan.target_name!r} but the run targets "
            f"{config.target!r}"
        )
    if config.layer not in (plan.primary_layer, plan.fallback_layer):
        raise StateAuditRunError(
            f"layer {config.layer} is neither the plan's primary layer {plan.primary_layer} nor "
            f"its fallback layer {plan.fallback_layer}"
        )

    store = DirectionStore(directions_dir())
    verification = verify_direction_family(family, store)
    if not verification["valid"]:
        raise StateAuditRunError(
            f"the stored direction vectors do not match family {family.family_id!r}: "
            f"{verification['failures'][:5]}. Refusing to run against a dirty direction family."
        )

    directions = [
        DirectionRef(opaque_id=entry.opaque_id, vector_hash=entry.vector_hash)
        for entry in family.directions
    ]

    return RunInputs(
        config=config,
        config_path=resolved_path,
        config_hash=config_hash(resolved_path),
        model_config=model_config,
        task_config=task_config,
        manifest=manifest,
        family=family,
        plan_id=plan.plan_id,
        plan_hash=plan.plan_hash,
        assignments=assignments,
        directions=directions,
    )


def check_smoke_parameters(config: StateAuditRunConfig) -> None:
    """Refuse anything the smoke command is not allowed to run.

    The smoke ratio and layer were fixed in advance precisely so they could not be chosen with
    numbers in view, and this command is the place that enforces it. A calibration sweep, a
    training run, or a different layer needs its own command, written when that slice lands.
    """
    if config.run_role is not StudyRunRole.ENGINEERING_SMOKE:
        raise StateAuditRunError(
            f"this command runs the engineering smoke only; the config declares run role "
            f"{config.run_role.value!r}"
        )
    if config.prompt_role is not PromptRole.SMOKE:
        raise StateAuditRunError(
            f"the smoke run may only use the {PromptRole.SMOKE.value!r} prompt role; the config "
            f"names {config.prompt_role.value!r}. Calibration, training, and final-test prompts "
            "are reserved for their own stages."
        )
    if config.layer != SMOKE_LAYER:
        raise StateAuditRunError(
            f"the smoke run is fixed at layer {SMOKE_LAYER}; the config names layer {config.layer}"
        )
    if float(config.norm_ratio) != SMOKE_NORM_RATIO:
        raise StateAuditRunError(
            f"the smoke run is fixed at the preregistered arbitrary ratio {SMOKE_NORM_RATIO}; the "
            f"config names {config.norm_ratio}. The smoke ratio was chosen in advance so that it "
            "could not be chosen later from the effect distribution."
        )


# ---------------------------------------------------------------------------
# Model checks
# ---------------------------------------------------------------------------


def load_run_model(inputs: RunInputs) -> LoadedModel:
    try:
        model = load_model(inputs.model_config)
    except ModelLoadError as error:
        raise StateAuditRunError(f"could not load the pinned model: {error}") from error

    family = inputs.family
    if model.spec.model_id != family.model_id or model.spec.revision != family.model_revision:
        raise StateAuditRunError(
            f"the loaded model is {model.spec.model_id}@{model.spec.revision} but the direction "
            f"family was built against {family.model_id}@{family.model_revision}"
        )
    if model.hidden_dim != family.hidden_dim:
        raise StateAuditRunError(
            f"the loaded model is {model.hidden_dim}-dimensional but the direction family is "
            f"{family.hidden_dim}-dimensional"
        )
    if not 0 <= inputs.config.layer <= model.num_layers:
        raise StateAuditRunError(
            f"layer {inputs.config.layer} is out of range for a model with {model.num_layers} "
            "decoder blocks"
        )
    return model


def resolve_run_label_tokens(inputs: RunInputs, model: LoadedModel) -> dict[str, int]:
    try:
        token_ids = resolve_label_token_ids(
            model.tokenizer, list(ANSWER_LABELS), prefix=inputs.task_config.label_prefix
        )
    except LabelTokenError as error:
        raise StateAuditRunError(f"answer labels are not scoreable: {error}") from error
    if token_ids != dict(inputs.family.answer_token_ids):
        raise StateAuditRunError(
            f"the tokenizer resolved answer token ids {token_ids} but the direction family was "
            f"built against {dict(inputs.family.answer_token_ids)}; the directions point at "
            "different tokens than this run would score"
        )
    return token_ids


# ---------------------------------------------------------------------------
# Clean pass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CleanPass:
    """One prompt's clean forward, held in memory while the run proceeds."""

    assignment: PromptAssignment
    trial_id: str
    prompt_text: str
    dataset_answer_label: str
    clean_logits: dict[str, float]
    clean_preferred_label: str
    clean_top_margin: float
    clean_entropy: float
    clean_correct: bool
    state: torch.Tensor
    state_id: str
    state_norm: float
    prompt_token_count: int
    position_absolute: int


def trial_id_for(assignment: PromptAssignment) -> str:
    """A trial id that does not depend on the run.

    Derived from the manifest's selection index, so the same prompt carries the same trial id in
    every run of this study. Candidate ordering derives from it, which means two runs of the same
    role produce the same candidate ids and can be compared row by row.
    """
    return f"sa_{assignment.selection_index:05d}"


def _prompt_lookup(task_name: str) -> tuple[dict[str, str], dict[str, str]]:
    try:
        items, variants = load_prepared_task(task_name)
    except TaskLoadError as error:
        raise StateAuditRunError(str(error)) from error
    prompts = {variant.variant_id: variant.prompt_text for variant in variants}
    answers = {item.item_id: item.answer_label for item in items}
    return prompts, answers


def run_clean_pass(
    inputs: RunInputs,
    model: LoadedModel,
    label_token_ids: dict[str, int],
    shard: StateShardWriter,
) -> tuple[list[CleanPass], list[dict[str, Any]], int]:
    """One clean forward per prompt, capturing the study layer. Returns forwards executed."""
    prompts, answers = _prompt_lookup(inputs.task_config.name)
    layer = inputs.config.layer
    passes: list[CleanPass] = []
    failures: list[dict[str, Any]] = []
    forwards = 0

    for assignment in inputs.assignments:
        trial_id = trial_id_for(assignment)
        prompt_text = prompts.get(assignment.variant_id)
        if prompt_text is None:
            failures.append(
                _failure(
                    trial_id,
                    assignment,
                    stage="clean",
                    error_type="MissingPrompt",
                    message=f"variant {assignment.variant_id!r} is not in the prepared task",
                    layer=layer,
                )
            )
            continue
        if sha256_hex(prompt_text.encode("utf-8")) != assignment.prompt_hash:
            failures.append(
                _failure(
                    trial_id,
                    assignment,
                    stage="clean",
                    error_type="PromptHashMismatch",
                    message=(
                        f"the prepared text for {assignment.variant_id!r} does not hash to the "
                        "value the frozen manifest recorded"
                    ),
                    layer=layer,
                )
            )
            continue

        try:
            capture = capture_hidden_states(
                model,
                prompt_text,
                layers=[layer],
                position_index=inputs.config.capture_position,
            )
            forwards += 1
            scores = score_logits(
                capture.next_token_logits, label_token_ids, answers[assignment.item_id]
            )
            state = capture.hidden_states[layer]
        except (CaptureError, LabelTokenError, KeyError, ValueError) as error:
            failures.append(
                _failure(
                    trial_id,
                    assignment,
                    stage="clean",
                    error_type=type(error).__name__,
                    message=str(error),
                    layer=layer,
                )
            )
            continue

        if state.ndim != 1 or int(state.shape[0]) != inputs.config.expected_hidden_dim:
            failures.append(
                _failure(
                    trial_id,
                    assignment,
                    stage="clean",
                    error_type="StateShapeError",
                    message=(
                        f"captured state has shape {tuple(state.shape)}, expected "
                        f"({inputs.config.expected_hidden_dim},)"
                    ),
                    layer=layer,
                )
            )
            continue
        if not bool(torch.isfinite(state).all()):
            failures.append(
                _failure(
                    trial_id,
                    assignment,
                    stage="clean",
                    error_type="StateNotFinite",
                    message="the captured clean state contains non-finite values",
                    layer=layer,
                )
            )
            continue

        norm = float(torch.linalg.vector_norm(state.to(torch.float64)))
        if not math.isfinite(norm) or norm <= 0.0:
            failures.append(
                _failure(
                    trial_id,
                    assignment,
                    stage="clean",
                    error_type="StateNormError",
                    message=f"the captured clean state has norm {norm!r}",
                    layer=layer,
                )
            )
            continue

        state_id = f"{assignment.variant_id}.L{layer}"
        shard.add(
            state_id=state_id,
            variant_id=assignment.variant_id,
            model_variant=model.spec.variant,
            layer=layer,
            position_index=capture.position_absolute,
            vector=state,
        )

        c_star = preferred_label(scores.logits)
        passes.append(
            CleanPass(
                assignment=assignment,
                trial_id=trial_id,
                prompt_text=prompt_text,
                dataset_answer_label=answers[assignment.item_id],
                clean_logits=dict(scores.logits),
                clean_preferred_label=c_star,
                clean_top_margin=preferred_margin(scores.logits, c_star),
                clean_entropy=scores.entropy,
                clean_correct=c_star == answers[assignment.item_id],
                state=state.detach().clone(),
                state_id=state_id,
                state_norm=norm,
                prompt_token_count=capture.sequence_length,
                position_absolute=capture.position_absolute,
            )
        )

    return passes, failures, forwards


def _failure(
    trial_id: str,
    assignment: PromptAssignment,
    stage: str,
    error_type: str,
    message: str,
    layer: int,
    candidate_id: str | None = None,
    norm_ratio: float | None = None,
) -> dict[str, Any]:
    """One preserved failure. Never dropped: a silently omitted failure would bias the run."""
    return {
        "schema_version": "1.0",
        "trial_id": trial_id,
        "variant_id": assignment.variant_id,
        "group_id": assignment.group_id,
        "candidate_id": candidate_id,
        "stage": stage,
        "status": "failed",
        "error_type": error_type,
        "error": message,
        "layer": layer,
        "norm_ratio": norm_ratio,
    }


# ---------------------------------------------------------------------------
# Intervened pass
# ---------------------------------------------------------------------------


def _spec_for(candidate: StateAuditCandidate) -> InterventionSpec:
    """Convert a study candidate into the harness's own intervention record.

    Going through `InterventionSpec` is what keeps this on the one validated inference path
    rather than adding a second way to apply an intervention. `analysis_role` is private and
    names no direction family.
    """
    return InterventionSpec(
        intervention_id=candidate.candidate_id,
        mechanism=candidate_mechanism(candidate),
        mechanism_version=_MECHANISM_VERSION,
        layer=candidate.layer,
        position_index=candidate.position_index,
        strength=candidate.strength,
        direction_id=candidate.direction_ref,
        analysis_role=candidate_analysis_role(candidate),
    )


@dataclass(frozen=True)
class AppliedCandidate:
    """One executed candidate, before it becomes an observation."""

    observation: StateAuditObservationRecord
    reconstruction_error: float


def apply_candidate(
    clean: CleanPass,
    candidate: StateAuditCandidate,
    inputs: RunInputs,
    model: LoadedModel,
    label_token_ids: dict[str, int],
    vectors: dict[str, torch.Tensor],
    run_id: str,
) -> AppliedCandidate:
    """Apply one candidate, score it, and build its observation.

    The intervened state at the study layer is captured in the same forward, after the
    intervention hook has run, so the check that the residual stream really became `h + v` costs
    no additional forward pass. A hook that never fired raises inside `run_with_intervention`,
    so a clean run mislabeled as an intervention cannot reach this point.
    """
    spec = _spec_for(candidate)
    baseline = clean.state.to(dtype=model.dtype, device=model.device)
    if candidate.is_noop:
        payload = InterventionPayload()
        expected = baseline
    else:
        direction = vectors[str(candidate.direction_ref)]
        payload = InterventionPayload(direction=direction)
        expected = baseline + float(candidate.strength) * direction

    result = run_with_intervention(
        model, clean.prompt_text, spec, payload, capture_layers=[candidate.layer]
    )
    diagnostics = result.diagnostics
    if diagnostics is None:
        raise StateAuditRunError(
            f"candidate {candidate.candidate_id} produced no intervention diagnostics; the norm "
            "bookkeeping that catches a no-op with a nonzero displacement would be missing"
        )

    observed_state = result.hidden_states.get(candidate.layer)
    if observed_state is None:
        raise CaptureError(
            f"the capture hook at layer {candidate.layer} never fired during candidate "
            f"{candidate.candidate_id}"
        )
    reconstruction_error = float(
        torch.max(torch.abs(observed_state.to(torch.float64) - expected.to(torch.float64)))
    )

    scores = score_logits(result.next_token_logits, label_token_ids, clean.dataset_answer_label)
    target = state_audit_target(clean.clean_logits, scores.logits)
    if target.clean_preferred_label != clean.clean_preferred_label:
        raise StateAuditRunError(
            f"candidate {candidate.candidate_id}: the clean preferred answer was recomputed as "
            f"{target.clean_preferred_label!r} but the clean pass recorded "
            f"{clean.clean_preferred_label!r}"
        )

    observation = StateAuditObservationRecord(
        study_id=inputs.config.study_id,
        run_id=run_id,
        trial_id=clean.trial_id,
        candidate_id=candidate.candidate_id,
        group_id=clean.assignment.group_id,
        variant_id=clean.assignment.variant_id,
        prompt_role=inputs.config.prompt_role,
        clean_preferred_label=target.clean_preferred_label,
        clean_logits=dict(clean.clean_logits),
        intervened_logits=dict(scores.logits),
        clean_top_margin=target.clean_top_margin,
        intervened_top_margin=target.intervened_top_margin,
        delta_clean_top_margin=target.delta_clean_top_margin,
        answer_flip=target.answer_flip,
        is_noop=candidate.is_noop,
        direction_ref=candidate.direction_ref or NOOP_DIRECTION_REF,
        direction_vector_hash=candidate.direction_vector_hash,
        layer=candidate.layer,
        norm_ratio=candidate.norm_ratio,
        global_alpha=candidate.global_alpha,
        pre_norm=diagnostics.pre_norm,
        post_norm=diagnostics.post_norm,
        delta_norm=diagnostics.delta_norm,
        model_id=model.spec.model_id,
        model_revision=model.spec.revision,
        prompt_manifest_hash=inputs.manifest.manifest_hash,
        direction_family_hash=inputs.family.family_hash,
        config_hash=inputs.config_hash,
    )
    return AppliedCandidate(observation=observation, reconstruction_error=reconstruction_error)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def build_diagnostics(
    observations: Sequence[StateAuditObservationRecord],
    clean_passes: Sequence[CleanPass],
    reconstruction_errors: Sequence[float],
    capture_hooks_fired: int,
    state_dim: int,
    effect_threshold: float,
) -> StateAuditRunDiagnostics:
    """Engineering diagnostics only. These select nothing."""
    if not observations:
        raise StateAuditRunError("cannot summarize a run with no observations")
    if not clean_passes:
        raise StateAuditRunError("cannot summarize a run with no clean states")

    noops = [record for record in observations if record.is_noop]
    signed = [record for record in observations if not record.is_noop]
    if not signed:
        raise StateAuditRunError("cannot summarize a run with no signed interventions")

    magnitudes = np.asarray(
        [abs(record.delta_clean_top_margin) for record in signed], dtype=np.float64
    )
    targets = [record.delta_clean_top_margin for record in signed]
    norms = [clean.state_norm for clean in clean_passes]

    return StateAuditRunDiagnostics(
        max_abs_noop_target=max((abs(r.delta_clean_top_margin) for r in noops), default=0.0),
        max_abs_noop_delta_norm=max((abs(r.delta_norm or 0.0) for r in noops), default=0.0),
        max_intervention_reconstruction_error=max(reconstruction_errors, default=0.0),
        min_target=min(targets),
        max_target=max(targets),
        median_abs_target=float(np.median(magnitudes)),
        p95_abs_target=float(np.quantile(magnitudes, 0.95, method="linear")),
        fraction_above_effect_threshold=float(
            np.count_nonzero(magnitudes >= effect_threshold) / magnitudes.size
        ),
        effect_threshold=effect_threshold,
        flip_count=sum(1 for record in signed if record.answer_flip),
        capture_hooks_fired=capture_hooks_fired,
        intervention_hooks_fired=len(observations),
        state_dim=state_dim,
        min_state_norm=min(norms),
        max_state_norm=max(norms),
        median_method=MEDIAN_METHOD,
        percentile_method=PERCENTILE_METHOD,
    )


# ---------------------------------------------------------------------------
# Execution safety
# ---------------------------------------------------------------------------


def load_study_run_manifest(run_id: str) -> StudyRunManifest:
    path = run_dir(run_id) / STATE_AUDIT_RUN_MANIFEST
    if not path.exists():
        raise StateAuditRunError(f"no state-audit run manifest at {path}")
    try:
        return StudyRunManifest.model_validate(read_json(path))
    except Exception as error:
        message = f"{path} is not a valid state-audit run manifest: {error}"
        raise StateAuditRunError(message) from error


def guard_existing_run(run_id: str, fingerprint: str, force: bool) -> None:
    """Refuse to write over a run that already exists.

    A completed run is evidence. Rewriting it in place would destroy the only copy of what
    actually happened while leaving a file that looks like a fresh result, so the refusal is the
    default and a new run id is the remedy. `force` exists for a documented partial or failed
    run; it is not a way to retry a completed one until the numbers look better.
    """
    directory = run_dir(run_id)
    manifest_path = directory / STATE_AUDIT_RUN_MANIFEST
    if manifest_path.exists():
        existing = load_study_run_manifest(run_id)
        same = existing.input_fingerprint == fingerprint
        if existing.status == "complete" and not force:
            raise StateAuditRunError(
                f"run {run_id!r} already exists and completed "
                f"({'identical' if same else 'different'} inputs, manifest hash "
                f"{existing.manifest_hash}). Refusing to rewrite a finished run: its artifacts "
                "are the only record of what happened. Verify it with `csf state-audit "
                "verify-run`, or execute into a new run id."
            )
        if not same and not force:
            raise StateAuditRunError(
                f"run {run_id!r} already exists and was executed against different inputs "
                f"(existing fingerprint {existing.input_fingerprint}, requested {fingerprint}). "
                "Use a new run id rather than reusing this one for a different experiment."
            )
        return

    leftovers = [
        name
        for name in (
            STATE_AUDIT_OBSERVATIONS,
            STATE_AUDIT_FAILURES,
            STATE_AUDIT_CANDIDATE_SETS,
            STATE_AUDIT_CLEAN_PASS,
            STATE_AUDIT_STATES,
        )
        if (directory / name).exists()
    ]
    if leftovers and not force:
        raise StateAuditRunError(
            f"run {run_id!r} holds state-audit artifacts but no run manifest, so a previous "
            f"attempt did not finish: {leftovers}. Execute into a new run id, or pass force once "
            "you have recorded the discarded attempt in docs/experiment_log.md."
        )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def execute_state_audit_run(
    config_path: str | Path,
    run_id: str,
    force: bool = False,
) -> dict[str, Any]:
    """Execute one state-dependence run end to end and write its artifacts."""
    inputs = resolve_run_inputs(config_path)
    config = inputs.config
    fingerprint = inputs.input_fingerprint
    guard_existing_run(run_id, fingerprint, force)

    directory = ensure_run_dir(run_id)
    for name in (
        STATE_AUDIT_OBSERVATIONS,
        STATE_AUDIT_FAILURES,
        STATE_AUDIT_CANDIDATE_SETS,
        STATE_AUDIT_CLEAN_PASS,
        STATE_AUDIT_STATE_REFS,
        STATE_AUDIT_RUN_MANIFEST,
    ):
        (directory / name).unlink(missing_ok=True)

    started_at = datetime.now(UTC)
    model = load_run_model(inputs)
    label_token_ids = resolve_run_label_tokens(inputs, model)

    shard = StateShardWriter(directory / STATE_AUDIT_STATES)
    clean_passes, failures, forwards = run_clean_pass(inputs, model, label_token_ids, shard)
    for failure in failures:
        append_jsonl(directory / STATE_AUDIT_FAILURES, failure)

    if not clean_passes:
        raise StateAuditRunError(
            f"no clean forward succeeded for run {run_id!r}; {len(failures)} failures were "
            f"recorded in {STATE_AUDIT_FAILURES}. No reference norm can be computed, so the run "
            "stops rather than inventing a strength."
        )
    state_refs = shard.close()
    states_hash = hash_file(directory / STATE_AUDIT_STATES)

    # One global alpha, from the median clean state norm across this run's prompts. Never
    # ratio * ||h_prompt||: a per-prompt strength would publish the prompt's state norm.
    norms_by_prompt = {clean.assignment.variant_id: clean.state_norm for clean in clean_passes}
    try:
        median_norm = reference_norm(norms_by_prompt, sorted(norms_by_prompt))
        global_alpha = alpha_for_ratio(median_norm, float(config.norm_ratio))
    except StrengthError as error:
        raise StateAuditRunError(str(error)) from error

    clean_records = [
        StateAuditCleanPassRecord(
            study_id=config.study_id,
            run_id=run_id,
            trial_id=clean.trial_id,
            variant_id=clean.assignment.variant_id,
            group_id=clean.assignment.group_id,
            item_id=clean.assignment.item_id,
            prompt_role=config.prompt_role,
            prompt_hash=clean.assignment.prompt_hash,
            prompt_token_count=clean.prompt_token_count,
            position_index=config.capture_position,
            position_absolute=clean.position_absolute,
            clean_logits=dict(clean.clean_logits),
            clean_preferred_label=clean.clean_preferred_label,
            clean_top_margin=clean.clean_top_margin,
            clean_entropy=clean.clean_entropy,
            dataset_answer_label=clean.dataset_answer_label,
            clean_correct=clean.clean_correct,
            layer=config.layer,
            state_id=clean.state_id,
            state_dim=int(clean.state.shape[0]),
            state_norm=clean.state_norm,
            state_shard_hash=states_hash,
            model_id=model.spec.model_id,
            model_revision=model.spec.revision,
            prompt_manifest_hash=inputs.manifest.manifest_hash,
        )
        for clean in clean_passes
    ]
    write_jsonl(directory / STATE_AUDIT_CLEAN_PASS, clean_records)
    write_jsonl(directory / STATE_AUDIT_STATE_REFS, state_refs)

    store = DirectionStore(directions_dir())
    vectors = _load_direction_vectors(inputs, model, store)

    candidate_sets: list[StateAuditCandidateSet] = []
    observations: list[StateAuditObservationRecord] = []
    reconstruction_errors: list[float] = []

    for clean in clean_passes:
        try:
            candidate_set = build_state_audit_candidate_set(
                trial_id=clean.trial_id,
                study_id=config.study_id,
                variant_id=clean.assignment.variant_id,
                group_id=clean.assignment.group_id,
                prompt_role=config.prompt_role,
                kind=StateAuditCandidateKind.SELECTED_STRENGTH,
                layer=config.layer,
                position_index=config.capture_position,
                direction_family_id=inputs.family.family_id,
                direction_family_hash=inputs.family.family_hash,
                direction_count=len(inputs.directions),
                templates=selected_strength_templates(
                    inputs.directions, float(config.norm_ratio), global_alpha
                ),
                master_seed=config.master_seed,
            )
        except (CandidateBuildError, ValueError) as error:
            failure = _failure(
                clean.trial_id,
                clean.assignment,
                stage="candidates",
                error_type=type(error).__name__,
                message=str(error),
                layer=config.layer,
                norm_ratio=float(config.norm_ratio),
            )
            failures.append(failure)
            append_jsonl(directory / STATE_AUDIT_FAILURES, failure)
            continue

        candidate_sets.append(candidate_set)
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
                    "state-audit intervention failed",
                    trial_id=clean.trial_id,
                    candidate_id=candidate.candidate_id,
                    error=type(error).__name__,
                )
                continue

            observations.append(applied.observation)
            reconstruction_errors.append(applied.reconstruction_error)
            append_jsonl(directory / STATE_AUDIT_OBSERVATIONS, applied.observation)

    write_jsonl(directory / STATE_AUDIT_CANDIDATE_SETS, candidate_sets)

    if not observations:
        raise StateAuditRunError(
            f"no candidate was applied successfully in run {run_id!r}; {len(failures)} failures "
            f"were recorded in {STATE_AUDIT_FAILURES}"
        )

    manifest = _build_run_manifest(
        inputs=inputs,
        run_id=run_id,
        model=model,
        clean_passes=clean_passes,
        clean_records=clean_records,
        observations=observations,
        reconstruction_errors=reconstruction_errors,
        candidate_sets=candidate_sets,
        failures=failures,
        forwards=forwards,
        median_norm=median_norm,
        global_alpha=global_alpha,
        states_hash=states_hash,
        directory=directory,
        fingerprint=fingerprint,
        started_at=started_at,
    )
    atomic_write_json(directory / STATE_AUDIT_RUN_MANIFEST, manifest.model_dump(mode="json"))

    info(
        "executed state-audit run",
        run_id=run_id,
        run_role=manifest.run_role.value,
        status=manifest.status,
        prompts=manifest.observed_prompt_count,
        observations=len(observations),
        failures=len(failures),
        forwards=forwards,
    )
    return run_report(manifest, directory)


def _load_direction_vectors(
    inputs: RunInputs, model: LoadedModel, store: DirectionStore
) -> dict[str, torch.Tensor]:
    """Load every direction once, re-checking its content hash against the manifest."""
    vectors: dict[str, torch.Tensor] = {}
    for entry in inputs.family.directions:
        if not store.has(entry.opaque_id):
            raise StateAuditRunError(
                f"direction {entry.opaque_id!r} is not in {store.root}; rebuild the family with "
                "`csf directions build-family` before running"
            )
        array = store.load(entry.opaque_id).numpy().astype(np.float32)
        if vector_content_hash(array) != entry.vector_hash:
            raise StateAuditRunError(
                f"direction {entry.opaque_id!r} does not match the hash the family manifest "
                "records; the stored vector changed after the family was frozen"
            )
        if array.shape != (inputs.family.hidden_dim,):
            raise StateAuditRunError(
                f"direction {entry.opaque_id!r} has shape {array.shape}, expected "
                f"({inputs.family.hidden_dim},)"
            )
        vectors[entry.opaque_id] = torch.from_numpy(array).to(
            dtype=model.dtype, device=model.device
        )
    return vectors


def _build_run_manifest(
    inputs: RunInputs,
    run_id: str,
    model: LoadedModel,
    clean_passes: Sequence[CleanPass],
    clean_records: Sequence[StateAuditCleanPassRecord],
    observations: Sequence[StateAuditObservationRecord],
    reconstruction_errors: Sequence[float],
    candidate_sets: Sequence[StateAuditCandidateSet],
    failures: Sequence[dict[str, Any]],
    forwards: int,
    median_norm: float,
    global_alpha: float,
    states_hash: str,
    directory: Path,
    fingerprint: str,
    started_at: datetime,
) -> StudyRunManifest:
    config = inputs.config
    signed = [record for record in observations if not record.is_noop]
    noops = [record for record in observations if record.is_noop]

    diagnostics = build_diagnostics(
        observations=observations,
        clean_passes=clean_passes,
        reconstruction_errors=reconstruction_errors,
        capture_hooks_fired=len(clean_passes),
        state_dim=int(clean_passes[0].state.shape[0]),
        effect_threshold=config.effect_report_threshold,
    )

    provenance: list[ArtifactHashRecord] = []
    for name, kind in (
        (STATE_AUDIT_OBSERVATIONS, "state_audit_observations"),
        (STATE_AUDIT_FAILURES, "state_audit_failures"),
        (STATE_AUDIT_CANDIDATE_SETS, "state_audit_candidate_sets"),
        (STATE_AUDIT_CLEAN_PASS, "state_audit_clean_pass"),
        (STATE_AUDIT_STATE_REFS, "state_audit_state_refs"),
        (STATE_AUDIT_STATES, "state_audit_states"),
    ):
        path = directory / name
        if path.exists():
            provenance.append(
                ArtifactHashRecord(
                    path=name,
                    hash=hash_file(path),
                    size_bytes=path.stat().st_size,
                    kind=kind,
                )
            )

    failures_path = directory / STATE_AUDIT_FAILURES
    correct = sum(1 for record in clean_records if record.clean_correct)
    scored = len(clean_records)
    candidates_per_prompt = config.candidates_per_prompt

    common: dict[str, Any] = {
        "schema_version": StudyRunManifest.model_fields["schema_version"].default,
        "study_id": config.study_id,
        "run_id": run_id,
        "run_role": config.run_role.value,
        "model_id": model.spec.model_id,
        "model_revision": model.spec.revision,
        "tokenizer_revision": model.spec.revision,
        "dtype": model.spec.dtype,
        "device": model.spec.device,
        "target_name": TARGET_NAME,
        "prompt_manifest_id": inputs.manifest.manifest_id,
        "prompt_manifest_hash": inputs.manifest.manifest_hash,
        "prompt_role": config.prompt_role.value,
        "direction_family_id": inputs.family.family_id,
        "direction_family_hash": inputs.family.family_hash,
        "calibration_plan_id": inputs.plan_id,
        "calibration_plan_hash": inputs.plan_hash,
        "layer": config.layer,
        "capture_position": config.capture_position,
        "norm_ratio": float(config.norm_ratio),
        "reference_norm": median_norm,
        "global_alpha": global_alpha,
        "reference_norm_source": (
            f"median clean residual-stream norm over the {len(clean_passes)} "
            f"{config.prompt_role.value} prompts at layer {config.layer}; this is the run's own "
            "engineering reference norm and is not the calibration reference norm"
        ),
        "expected_prompt_count": config.expected_prompt_count,
        "expected_candidates_per_prompt": candidates_per_prompt,
        "expected_non_noop_observations": (
            config.expected_prompt_count * config.expected_signed_directions
        ),
        "expected_noop_observations": config.expected_prompt_count,
        "expected_forward_count": config.expected_forward_count,
        "observed_prompt_count": len(clean_passes),
        "observed_state_count": len(clean_records),
        "observed_non_noop_observations": len(signed),
        "observed_noop_observations": len(noops),
        "observed_forward_count": forwards,
        "failure_count": len(failures),
        "clean_scored_count": scored,
        "clean_correct_count": correct,
        "clean_accuracy_descriptive": (correct / scored) if scored else None,
        "diagnostics": diagnostics.model_dump(mode="json"),
        "observations_hash": hash_file(directory / STATE_AUDIT_OBSERVATIONS),
        "failures_hash": hash_file(failures_path) if failures_path.exists() else None,
        "states_hash": states_hash,
        "candidate_sets_hash": hash_file(directory / STATE_AUDIT_CANDIDATE_SETS),
        "clean_pass_hash": hash_file(directory / STATE_AUDIT_CLEAN_PASS),
        "input_fingerprint": fingerprint,
        "config_hash": inputs.config_hash,
    }

    complete = (
        not failures
        and len(clean_passes) == config.expected_prompt_count
        and len(clean_records) == config.expected_prompt_count
        and len(signed) == config.expected_prompt_count * config.expected_signed_directions
        and len(noops) == config.expected_prompt_count
        and forwards == config.expected_forward_count
    )
    common["status"] = "complete" if complete else "failed"

    environment = environment_snapshot()
    git = environment.get("git") or {}

    return StudyRunManifest(
        **{key: value for key, value in common.items() if key != "diagnostics"},
        diagnostics=diagnostics,
        manifest_hash=compute_study_run_hash(common),
        config_path=_repo_relative(inputs.config_path),
        code_commit=git.get("commit"),
        code_branch=git.get("branch"),
        code_dirty=git.get("dirty"),
        environment=environment,
        provenance=provenance,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        notes=(
            "State-dependence execution run. Every candidate was applied to every prompt and "
            "every failure is recorded. An engineering smoke run validates the pipeline on real "
            "weights; it selects no ratio and no layer, and it is not a scientific result."
        ),
    )


def run_report(manifest: StudyRunManifest, directory: Path) -> dict[str, Any]:
    """The command's JSON output. Numbers come from the manifest, never recomputed here."""
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
        "norm_ratio": manifest.norm_ratio,
        "reference_norm": manifest.reference_norm,
        "global_alpha": manifest.global_alpha,
        "reference_norm_source": manifest.reference_norm_source,
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
        },
        "code_commit": manifest.code_commit,
        "code_branch": manifest.code_branch,
        "code_dirty": manifest.code_dirty,
        "notes": (
            "Engineering validation on real weights. Clean accuracy is descriptive over a "
            "handful of prompts and is not a capability measurement. Effect sizes are "
            "diagnostics and select no ratio and no layer. This is not a scientific result."
        ),
    }


def run_smoke(config_path: str | Path, run_id: str, force: bool = False) -> dict[str, Any]:
    """Execute the eight-prompt engineering smoke."""
    config = load_run_config(config_path)
    check_smoke_parameters(config)
    return execute_state_audit_run(config_path, run_id, force=force)


def manifest_content_bytes(manifest: StudyRunManifest) -> bytes:
    from ..schemas import study_run_payload

    return canonical_json_bytes(study_run_payload(manifest.model_dump(mode="json")))


__all__ = [
    "PERCENTILE_METHOD",
    "RUN_ALGORITHM_VERSION",
    "SMOKE_LAYER",
    "SMOKE_NORM_RATIO",
    "AppliedCandidate",
    "CleanPass",
    "RunInputs",
    "StateAuditRunError",
    "apply_candidate",
    "build_diagnostics",
    "check_smoke_parameters",
    "execute_state_audit_run",
    "guard_existing_run",
    "load_run_config",
    "load_study_run_manifest",
    "manifest_content_bytes",
    "resolve_run_inputs",
    "run_clean_pass",
    "run_report",
    "run_smoke",
    "trial_id_for",
]
