"""Fitting the predictors and committing the final-test forecasts.

Two commands' worth of work, in the order the preregistration fixes:

* **fit**: read the training run's observations, states, and clean pass; fit the transforms and
  the three ridges on the 96 training prompts only; write a `TransformFitRecord` per fitted
  object and a `StateAuditPredictorRecord` per method.
* **commit**: read the final-test clean run; build the 17 candidates per prompt; build the
  wrong-state pairing; predict every candidate under all sixteen method-and-condition
  combinations; commit all 512 records.

The fit boundary is enforced, not documented. Every fitted object records the exact prompt ids it
saw, `TransformFitRecord` refuses a fit role other than `training`, and the fitting entry point
refuses a run whose prompt role is not `training`.

Commitment happens before any final-test intervention exists, and the committer refuses to run at
all if an outcome artifact is present in the run directory. That refusal is the blinding: the
timestamps alone would look correct even if the forecasts had been written after the fact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import StateAuditRunConfig, load_config, repo_root
from ..forecasting.base import PUBLIC_FEATURE_KEYS, Forecaster, TrainingExample
from ..hashing import atomic_write_json, hash_file, hash_object, read_jsonl, write_jsonl
from ..interventions.direction_family import load_direction_family
from ..interventions.directions import DirectionStore
from ..logging_utils import info
from ..paths import (
    FORECAST_COMMITMENTS,
    FORECASTS,
    STATE_AUDIT_CANDIDATE_SETS,
    STATE_AUDIT_CLEAN_PASS,
    STATE_AUDIT_COMMITMENT_SUMMARY,
    STATE_AUDIT_OBSERVATIONS,
    STATE_AUDIT_PAIRING,
    STATE_AUDIT_PREDICTORS,
    STATE_AUDIT_TRANSFORM_FITS,
    directions_dir,
    ensure_run_dir,
    run_dir,
)
from ..schemas import (
    FeatureBlock,
    ForecastCandidate,
    ForecastRecord,
    PromptRole,
    StateAuditCandidateKind,
    StateAuditCandidateSet,
    StateAuditCleanPassRecord,
    StateAuditObservationRecord,
    StateAuditPredictorRecord,
    StateCondition,
    TransformFitRecord,
    WrongStatePairingRecord,
)
from ..tasks.loader import load_prepared_task
from ..trials.commitment import (
    commit_forecast,
    existing_outcome_artifacts,
    read_commitments,
    verify_commitment_ordering,
)
from ..trials.states import load_state
from .candidates import (
    DirectionRef,
    build_state_audit_candidate_set,
    public_view,
    selected_strength_templates,
)
from .features import (
    BLOCK_WIDTHS,
    METHOD_BLOCKS,
    FittedTransforms,
    PromptContext,
    StateAuditExample,
    build_feature_row,
    fit_transforms,
    visible_block_is_clean,
)
from .fit import FittedPredictor, fit_predictor
from .matching import build_pairing, donor_for
from .projection import load_projection_matrix, load_projection_record
from .run import load_state_audit_run_manifest, resolve_run_inputs

# The sixteen method-and-condition combinations committed per final-test prompt. Fixed here so
# the count is a property of the code rather than of whoever ran it.
DIAGNOSTIC_METHODS = ("constant", "prompt_lexical")
RIDGE_METHODS = ("intervention_only_ridge", "visible_information_ridge")
BILINEAR_METHOD = "state_bilinear_ridge"
PERMUTATION_CONDITIONS = 10

# Placeholders required by `ForecastCandidate`, recorded as placeholders. A ridge fitted on a
# continuous target estimates neither a bias-suppression probability nor a hidden-bias
# probability, and this arm has no model organism for either to be about.
PLACEHOLDER_PROBABILITY = 0.5

FORECAST_ALGORITHM_VERSION = "bluedot_state_audit_forecasts_v1.0"


class PredictError(RuntimeError):
    """Raised when predictors cannot be fitted or forecasts cannot be committed."""


# ---------------------------------------------------------------------------
# Reading a run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunBundle:
    """One executed run, read back into the objects the feature system needs."""

    run_id: str
    config: StateAuditRunConfig
    contexts: dict[str, PromptContext]
    observations: list[StateAuditObservationRecord]
    candidate_sets: dict[str, StateAuditCandidateSet]
    prompt_manifest_hash: str
    direction_family_hash: str
    layer: int
    norm_ratio: float
    global_alpha: float


def _prompt_texts(task_name: str) -> dict[str, str]:
    _, variants = load_prepared_task(task_name)
    return {variant.variant_id: variant.prompt_text for variant in variants}


def _read_clean_pass(run_id: str) -> list[StateAuditCleanPassRecord]:
    path = run_dir(run_id) / STATE_AUDIT_CLEAN_PASS
    if not path.exists():
        raise PredictError(f"no clean pass at {path}")
    return [StateAuditCleanPassRecord.model_validate(row) for row in read_jsonl(path)]


def load_run_bundle(config_path: str | Path, run_id: str, expect_role: PromptRole) -> RunBundle:
    """Read a completed run's artifacts into prompt contexts and observations."""
    inputs = resolve_run_inputs(config_path)
    config = inputs.config
    if config.prompt_role is not expect_role:
        raise PredictError(
            f"expected a {expect_role.value} run; the config declares {config.prompt_role.value}"
        )

    manifest = load_state_audit_run_manifest(run_id)
    if manifest.status != "complete":
        raise PredictError(
            f"run {run_id!r} is marked {manifest.status!r}; fitting on an incomplete run would "
            "fit on a biased subset"
        )
    if manifest.prompt_manifest_hash != inputs.manifest.manifest_hash:
        raise PredictError(f"run {run_id!r} cites a different prompt manifest than the config")

    clean_records = _read_clean_pass(run_id)
    texts = _prompt_texts(inputs.task_config.name)
    states_path = run_dir(run_id) / "state_audit_states.npz"
    if not states_path.exists():
        raise PredictError(f"no captured states at {states_path}")

    contexts: dict[str, PromptContext] = {}
    for record in clean_records:
        text = texts.get(record.variant_id)
        if text is None:
            raise PredictError(f"{record.variant_id} is not in the prepared task")
        state = load_state(states_path, record.state_id).numpy().astype(np.float64)
        contexts[record.variant_id] = PromptContext(
            variant_id=record.variant_id,
            group_id=record.group_id,
            trial_id=record.trial_id,
            prompt_text=text,
            prompt_token_count=record.prompt_token_count,
            clean_logits=dict(record.clean_logits),
            clean_preferred_label=record.clean_preferred_label,
            clean_top_margin=record.clean_top_margin,
            clean_entropy=record.clean_entropy,
            state=state,
        )

    observations: list[StateAuditObservationRecord] = []
    observations_path = run_dir(run_id) / STATE_AUDIT_OBSERVATIONS
    if observations_path.exists():
        observations = [
            StateAuditObservationRecord.model_validate(row) for row in read_jsonl(observations_path)
        ]

    candidate_sets: dict[str, StateAuditCandidateSet] = {}
    sets_path = run_dir(run_id) / STATE_AUDIT_CANDIDATE_SETS
    if sets_path.exists():
        for row in read_jsonl(sets_path):
            candidate_set = StateAuditCandidateSet.model_validate(row)
            candidate_sets[candidate_set.trial_id] = candidate_set

    strengths = _manifest_strength(manifest)
    return RunBundle(
        run_id=run_id,
        config=config,
        contexts=contexts,
        observations=observations,
        candidate_sets=candidate_sets,
        prompt_manifest_hash=manifest.prompt_manifest_hash,
        direction_family_hash=manifest.direction_family_hash,
        layer=manifest.layer,
        norm_ratio=strengths[0],
        global_alpha=strengths[1],
    )


def _manifest_strength(manifest: Any) -> tuple[float, float]:
    ratio = getattr(manifest, "norm_ratio", None)
    alpha = getattr(manifest, "global_alpha", None)
    if ratio is None or alpha is None:
        ratio = getattr(manifest, "selected_norm_ratio", None)
        alpha = getattr(manifest, "selected_global_alpha", None)
    if ratio is None or alpha is None:
        raise PredictError("the run manifest records no single selected strength")
    return float(ratio), float(alpha)


def direction_vectors(family_id: str) -> dict[str, np.ndarray]:
    """Load every direction, checked against the family manifest's content hashes."""
    from .run import vector_content_hash

    family = load_direction_family(family_id)
    store = DirectionStore(directions_dir())
    vectors: dict[str, np.ndarray] = {}
    for entry in family.directions:
        if not store.has(entry.opaque_id):
            raise PredictError(f"direction {entry.opaque_id!r} is not in {store.root}")
        array = store.load(entry.opaque_id).numpy().astype(np.float32)
        if vector_content_hash(array) != entry.vector_hash:
            raise PredictError(
                f"direction {entry.opaque_id!r} does not match the hash the family records"
            )
        vectors[entry.opaque_id] = array.astype(np.float64)
    return vectors


def signed_vector(
    vectors: dict[str, np.ndarray], direction_ref: str | None, strength: float, dim: int
) -> np.ndarray:
    """`v = sign * alpha * d`, and exactly zero for the no-op."""
    if direction_ref is None:
        return np.zeros(dim, dtype=np.float64)
    if direction_ref not in vectors:
        raise PredictError(f"unknown direction {direction_ref!r}")
    return float(strength) * vectors[direction_ref]


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FittedStudy:
    """Everything fitted on the training prompts, frozen together."""

    transforms: FittedTransforms
    predictors: dict[str, FittedPredictor]
    transform_records: list[TransformFitRecord]
    predictor_records: list[StateAuditPredictorRecord]
    training_run_id: str
    training_prompt_ids: list[str]


def build_examples(
    bundle: RunBundle, vectors: dict[str, np.ndarray], dim: int
) -> list[StateAuditExample]:
    """One row per observation, with the signed intervention vector reconstructed."""
    by_candidate: dict[tuple[str, str], Any] = {}
    for candidate_set in bundle.candidate_sets.values():
        for candidate in candidate_set.candidates:
            by_candidate[(candidate_set.trial_id, candidate.candidate_id)] = candidate

    examples: list[StateAuditExample] = []
    for record in bundle.observations:
        candidate = by_candidate.get((record.trial_id, record.candidate_id))
        if candidate is None:
            raise PredictError(
                f"observation {record.trial_id}/{record.candidate_id} names no frozen candidate"
            )
        examples.append(
            StateAuditExample(
                variant_id=record.variant_id,
                group_id=record.group_id,
                trial_id=record.trial_id,
                candidate_id=record.candidate_id,
                is_noop=record.is_noop,
                intervention_vector=signed_vector(
                    vectors, candidate.direction_ref, candidate.strength, dim
                ),
                observed_target=record.delta_clean_top_margin,
                observed_flip=record.answer_flip,
            )
        )
    return examples


def fit_study(
    training_config: str | Path,
    training_run_id: str,
    projection_id: str,
) -> FittedStudy:
    """Fit every transform and predictor on the training prompts only."""
    bundle = load_run_bundle(training_config, training_run_id, PromptRole.TRAINING)
    projection_record = load_projection_record(projection_id)
    projection = load_projection_matrix(projection_id)

    family_id = bundle.config.direction_family_id
    vectors = direction_vectors(family_id)
    dim = projection_record.hidden_dim
    examples = build_examples(bundle, vectors, dim)
    if not examples:
        raise PredictError(f"training run {training_run_id!r} holds no observations")

    contexts = list(bundle.contexts.values())
    transforms = fit_transforms(contexts, examples, projection)

    signed = [example for example in examples if not example.is_noop]
    if not signed:
        raise PredictError("no non-no-op training rows to fit on")

    targets = np.asarray([example.observed_target for example in signed], dtype=np.float64)
    flips = [example.observed_flip for example in signed]
    group_ids = [example.group_id for example in signed]

    # Build every block once per row, then slice per method. The intervention block is therefore
    # literally the same array for the visible and the state-conditioned models.
    rows_by_method: dict[str, list[np.ndarray]] = {method: [] for method in METHOD_BLOCKS}
    for example in signed:
        context = bundle.contexts[example.variant_id]
        for method, blocks in METHOD_BLOCKS.items():
            rows_by_method[method].append(
                build_feature_row(
                    transforms, context, example.intervention_vector, projection, blocks
                )
            )

    predictors: dict[str, FittedPredictor] = {}
    for method, blocks in METHOD_BLOCKS.items():
        features = np.stack(rows_by_method[method], axis=0)
        predictors[method] = fit_predictor(
            method_id=method,
            blocks=blocks,
            features=features,
            targets=targets,
            flips=flips,
            group_ids=group_ids,
            master_seed=bundle.config.master_seed,
        )

    transform_records = transforms.fit_records(
        study_id=bundle.config.study_id,
        master_seed=bundle.config.master_seed,
        prompt_manifest_hash=bundle.prompt_manifest_hash,
        fit_row_count=len(signed),
    )
    transform_ids = [record.transform_id for record in transform_records]

    predictor_records: list[StateAuditPredictorRecord] = []
    for method, predictor in predictors.items():
        payload = {
            "algorithm_version": FORECAST_ALGORITHM_VERSION,
            "method_id": method,
            "blocks": [block.value for block in predictor.blocks],
            "coefficient_hash": predictor.coefficient_hash(),
            "ridge_alpha": predictor.selection.selected_alpha,
            "training_rows": predictor.training_rows,
            "transform_ids": transform_ids,
            "projection_hash": projection_record.matrix_hash,
            "prompt_manifest_hash": bundle.prompt_manifest_hash,
            "direction_family_hash": bundle.direction_family_hash,
        }
        predictor_records.append(
            StateAuditPredictorRecord(
                predictor_id=f"{method}_v1",
                study_id=bundle.config.study_id,
                method_id=method,
                feature_blocks=list(predictor.blocks),
                block_widths={block.value: BLOCK_WIDTHS[block] for block in predictor.blocks},
                feature_dim=predictor.feature_dim,
                training_rows=predictor.training_rows,
                training_prompt_count=predictor.training_prompt_count,
                ridge_selection=predictor.selection,
                residual_q05=predictor.residual_q05,
                residual_q95=predictor.residual_q95,
                training_flip_base_rate=predictor.flip_base_rate,
                coefficient_hash=predictor.coefficient_hash(),
                transform_ids=transform_ids,
                prompt_manifest_hash=bundle.prompt_manifest_hash,
                direction_family_hash=bundle.direction_family_hash,
                projection_hash=projection_record.matrix_hash,
                training_run_id=training_run_id,
                predictor_hash=hash_object(payload),
            )
        )

    directory = run_dir(training_run_id)
    write_jsonl(directory / STATE_AUDIT_TRANSFORM_FITS, transform_records)
    write_jsonl(directory / STATE_AUDIT_PREDICTORS, predictor_records)

    info(
        "fitted state-audit predictors",
        run_id=training_run_id,
        rows=len(signed),
        prompts=len(contexts),
        methods=len(predictors),
    )
    return FittedStudy(
        transforms=transforms,
        predictors=predictors,
        transform_records=transform_records,
        predictor_records=predictor_records,
        training_run_id=training_run_id,
        training_prompt_ids=sorted(bundle.contexts),
    )


def fit_report(study: FittedStudy) -> dict[str, Any]:
    return {
        "training_run_id": study.training_run_id,
        "training_prompt_count": len(study.training_prompt_ids),
        "scientific_result": False,
        "transforms": [
            {
                "transform_id": record.transform_id,
                "kind": record.kind,
                "output_dim": record.output_dim,
                "fit_prompt_count": len(record.fit_prompt_ids),
                "fit_row_count": record.fit_row_count,
                "parameters": record.parameters,
                "transform_hash": record.transform_hash,
                "fit_prompt_identity_hash": record.fit_prompt_identity_hash,
            }
            for record in study.transform_records
        ],
        "predictors": [
            {
                "method_id": record.method_id,
                "feature_blocks": [block.value for block in record.feature_blocks],
                "block_widths": record.block_widths,
                "feature_dim": record.feature_dim,
                "training_rows": record.training_rows,
                "selected_ridge_alpha": record.ridge_selection.selected_alpha,
                "cv_mean_absolute_error": record.ridge_selection.selected_alpha_mae,
                "mean_absolute_error_by_alpha": record.ridge_selection.mean_absolute_error_by_alpha,
                "alpha_at_grid_edge": record.ridge_selection.alpha_at_grid_edge,
                "residual_q05": record.residual_q05,
                "residual_q95": record.residual_q95,
                "training_flip_base_rate": record.training_flip_base_rate,
                "coefficient_hash": record.coefficient_hash,
                "predictor_hash": record.predictor_hash,
            }
            for record in study.predictor_records
        ],
        "notes": (
            "Fitted on the 96 training prompts only. Cross-validated errors are training-fold "
            "diagnostics, not a result: no method has been scored on the final test, which has "
            "no outcomes."
        ),
    }


# ---------------------------------------------------------------------------
# Committing
# ---------------------------------------------------------------------------


def _candidate_sets_for(bundle: RunBundle, family_id: str) -> list[StateAuditCandidateSet]:
    """Build the 17 candidates per final-test prompt, deterministically.

    Frozen here, before commitment, because a forecast has to name the candidates it predicts.
    Resolution rebuilds them from the same inputs and gets the same ids, and the committed
    candidate-set file is the record either way.
    """
    family = load_direction_family(family_id)
    directions = [
        DirectionRef(opaque_id=entry.opaque_id, vector_hash=entry.vector_hash)
        for entry in family.directions
    ]
    templates = selected_strength_templates(directions, bundle.norm_ratio, bundle.global_alpha)
    return [
        build_state_audit_candidate_set(
            trial_id=context.trial_id,
            study_id=bundle.config.study_id,
            variant_id=context.variant_id,
            group_id=context.group_id,
            prompt_role=bundle.config.prompt_role,
            kind=StateAuditCandidateKind.SELECTED_STRENGTH,
            layer=bundle.layer,
            position_index=bundle.config.capture_position,
            direction_family_id=family.family_id,
            direction_family_hash=family.family_hash,
            direction_count=len(directions),
            templates=templates,
            master_seed=bundle.config.master_seed,
        )
        for context in sorted(bundle.contexts.values(), key=lambda c: c.variant_id)
    ]


def _training_examples_for_baselines(
    bundle: RunBundle, examples: Sequence[StateAuditExample]
) -> list[TrainingExample]:
    """The benchmark baselines' own example type, built from the study's observations.

    `TrainingExample` is unchanged and still fenced to public information. The public view of a
    study candidate carries the same four keys the baselines already consume, so they can be
    trained on this arm's outcomes without widening what they may read.
    """
    by_candidate = {
        (candidate_set.trial_id, candidate.candidate_id): candidate
        for candidate_set in bundle.candidate_sets.values()
        for candidate in candidate_set.candidates
    }
    rows: list[TrainingExample] = []
    for example in examples:
        if example.is_noop:
            continue
        candidate = by_candidate[(example.trial_id, example.candidate_id)]
        view = public_view(candidate)
        rows.append(
            TrainingExample(
                prompt_text=bundle.contexts[example.variant_id].prompt_text,
                public_features={key: view[key] for key in PUBLIC_FEATURE_KEYS},
                observed_delta=example.observed_target,
                observed_flip=example.observed_flip,
                group_id=example.group_id,
                split=PromptRole.TRAINING.value,
            )
        )
    return rows


def _forecast_candidates_from_ridge(
    predictor: FittedPredictor,
    transforms: FittedTransforms,
    context: PromptContext,
    candidate_set: StateAuditCandidateSet,
    vectors: dict[str, np.ndarray],
    projection: np.ndarray,
    dim: int,
    state_override: np.ndarray | None,
) -> list[ForecastCandidate]:
    predictions: list[ForecastCandidate] = []
    for candidate in candidate_set.candidates:
        vector = signed_vector(vectors, candidate.direction_ref, candidate.strength, dim)
        row = build_feature_row(
            transforms,
            context,
            vector,
            projection,
            predictor.blocks,
            state_override=state_override,
        )
        mean = predictor.predict(row)
        predictions.append(
            ForecastCandidate(
                intervention_id=candidate.candidate_id,
                delta_margin_mean=mean,
                delta_margin_q05=mean + predictor.residual_q05,
                delta_margin_q95=mean + predictor.residual_q95,
                p_answer_flip=predictor.flip_base_rate,
                p_bias_suppressed=PLACEHOLDER_PROBABILITY,
            )
        )
    return predictions


def _forecast_candidates_from_baseline(
    baseline: Forecaster, context: PromptContext, candidate_set: StateAuditCandidateSet
) -> list[ForecastCandidate]:
    predictions: list[ForecastCandidate] = []
    for candidate in candidate_set.candidates:
        view = public_view(candidate)
        features = {key: view[key] for key in PUBLIC_FEATURE_KEYS}
        prediction = baseline.predict_candidate(
            context.prompt_text, candidate.candidate_id, features
        )
        predictions.append(prediction.to_forecast_candidate())
    return predictions


def commit_final_test_forecasts(
    training_config: str | Path,
    training_run_id: str,
    final_test_config: str | Path,
    final_test_run_id: str,
    projection_id: str,
) -> dict[str, Any]:
    """Fit, build every condition, and commit all 512 final-test forecasts."""
    outcomes = existing_outcome_artifacts(final_test_run_id)
    if outcomes:
        raise PredictError(
            f"final-test run {final_test_run_id!r} already holds outcome artifacts {outcomes}. "
            "A forecast committed after an outcome exists is not a forecast."
        )
    if read_commitments(final_test_run_id):
        raise PredictError(
            f"final-test run {final_test_run_id!r} already holds commitments; committing twice "
            "would leave the second set unverifiable. Use a fresh run directory."
        )

    from ..forecasting import ConstantBaseline, PromptLexicalBaseline

    study = fit_study(training_config, training_run_id, projection_id)
    training = load_run_bundle(training_config, training_run_id, PromptRole.TRAINING)
    final = load_run_bundle(final_test_config, final_test_run_id, PromptRole.FINAL_TEST)

    projection_record = load_projection_record(projection_id)
    projection = load_projection_matrix(projection_id)
    dim = projection_record.hidden_dim
    vectors = direction_vectors(final.config.direction_family_id)

    if final.observations:
        raise PredictError(
            f"final-test run {final_test_run_id!r} holds {len(final.observations)} observations; "
            "the clean stage must not have applied an intervention"
        )

    candidate_sets = _candidate_sets_for(final, final.config.direction_family_id)
    directory = ensure_run_dir(final_test_run_id)
    write_jsonl(directory / STATE_AUDIT_CANDIDATE_SETS, candidate_sets)
    by_trial = {candidate_set.trial_id: candidate_set for candidate_set in candidate_sets}

    contexts = sorted(final.contexts.values(), key=lambda context: context.variant_id)
    pairing = build_pairing(
        pairing_id=f"{final_test_run_id}_pairing_v1",
        study_id=final.config.study_id,
        layer=final.layer,
        contexts=contexts,
        master_seed=final.config.master_seed,
        prompt_manifest_hash=final.prompt_manifest_hash,
        final_test_run_id=final_test_run_id,
    )
    atomic_write_json(directory / STATE_AUDIT_PAIRING, pairing.model_dump(mode="json"))

    # Train the two diagnostics on the same training outcomes, through their own fenced type.
    training_examples = build_examples(training, vectors, dim)
    baseline_rows = _training_examples_for_baselines(training, training_examples)
    baselines: dict[str, Forecaster] = {
        "constant": ConstantBaseline(),
        "prompt_lexical": PromptLexicalBaseline(),
    }
    for baseline in baselines.values():
        baseline.check_no_leakage()
        baseline.fit(baseline_rows)

    states = {context.variant_id: context.state for context in contexts}
    committed = 0
    per_condition: dict[str, int] = {}
    records_per_prompt = len(DIAGNOSTIC_METHODS) + len(RIDGE_METHODS) + 2 + PERMUTATION_CONDITIONS

    for context in contexts:
        candidate_set = by_trial[context.trial_id]
        plans: list[tuple[str, StateCondition, int, np.ndarray | None]] = [
            *[(name, StateCondition.NONE, 0, None) for name in DIAGNOSTIC_METHODS],
            *[(name, StateCondition.NONE, 0, None) for name in RIDGE_METHODS],
            (BILINEAR_METHOD, StateCondition.TRUE, 0, None),
            (
                BILINEAR_METHOD,
                StateCondition.WRONG_EXAMPLE,
                0,
                states[donor_for(pairing, context.variant_id, 0, shuffled=False)],
            ),
            *[
                (
                    BILINEAR_METHOD,
                    StateCondition.SHUFFLED,
                    index,
                    states[donor_for(pairing, context.variant_id, index, shuffled=True)],
                )
                for index in range(PERMUTATION_CONDITIONS)
            ],
        ]
        if len(plans) != records_per_prompt:
            raise PredictError(
                f"built {len(plans)} method-and-condition records for {context.variant_id}, "
                f"expected {records_per_prompt}"
            )

        for method_id, condition, index, override in plans:
            if method_id in baselines:
                predictions = _forecast_candidates_from_baseline(
                    baselines[method_id], context, candidate_set
                )
            else:
                predictions = _forecast_candidates_from_ridge(
                    study.predictors[method_id],
                    study.transforms,
                    context,
                    candidate_set,
                    vectors,
                    projection,
                    dim,
                    override,
                )
            forecast = ForecastRecord(
                trial_id=context.trial_id,
                method_id=method_id,
                candidate_forecasts=predictions,
                p_hidden_bias_active=PLACEHOLDER_PROBABILITY,
                state_condition=condition,
                condition_index=index,
            )
            commit_forecast(final_test_run_id, forecast)
            committed += 1
            key = f"{method_id}:{condition.value}"
            per_condition[key] = per_condition.get(key, 0) + 1

    summary = {
        "algorithm_version": FORECAST_ALGORITHM_VERSION,
        "final_test_run_id": final_test_run_id,
        "training_run_id": training_run_id,
        "projection_id": projection_id,
        "projection_hash": projection_record.matrix_hash,
        "prompt_manifest_hash": final.prompt_manifest_hash,
        "direction_family_hash": final.direction_family_hash,
        "pairing_hash": pairing.pairing_hash,
        "prompts": len(contexts),
        "candidates_per_prompt": len(candidate_sets[0].candidates),
        "records_per_prompt": records_per_prompt,
        "commitments": committed,
        "commitments_by_condition": dict(sorted(per_condition.items())),
        "candidate_sets_hash": hash_file(directory / STATE_AUDIT_CANDIDATE_SETS),
        "forecasts_hash": hash_file(directory / FORECASTS),
        "commitments_hash": hash_file(directory / FORECAST_COMMITMENTS),
        "scientific_result": False,
        "notes": (
            "Forecasts committed before any final-test intervention was applied. No outcome "
            "artifact exists in this run directory, and the committer refuses to run if one does."
        ),
    }
    atomic_write_json(directory / STATE_AUDIT_COMMITMENT_SUMMARY, summary)

    info(
        "committed final-test forecasts",
        run_id=final_test_run_id,
        prompts=len(contexts),
        commitments=committed,
    )
    report = dict(summary)
    report["fit"] = fit_report(study)
    report["pairing"] = _pairing_report(pairing)
    report["verification"] = verify_final_test_commitments(final_test_run_id)
    return report


def _pairing_report(pairing: WrongStatePairingRecord) -> dict[str, Any]:
    same_label = sum(1 for match in pairing.matches if match.same_clean_preferred_label)
    return {
        "pairing_id": pairing.pairing_id,
        "pairing_hash": pairing.pairing_hash,
        "matched_prompts": len(pairing.matches),
        "matches_with_same_clean_preferred_label": same_label,
        "permutations": len(pairing.permutations),
        "self_matches": sum(
            1 for match in pairing.matches if match.variant_id == match.donor_variant_id
        ),
        "max_margin_distance": max(match.margin_distance for match in pairing.matches),
        "median_margin_distance": float(
            np.median([match.margin_distance for match in pairing.matches])
        ),
    }


# ---------------------------------------------------------------------------
# Verification at the commitment checkpoint
# ---------------------------------------------------------------------------


def verify_final_test_commitments(run_id: str) -> dict[str, Any]:
    """Check the commitment checkpoint from artifacts. Loads no model.

    At this point no reveal exists, by design: the salts stay sealed until the interventions are
    resolved. So this does not call `verify_run_commitments`, which requires reveals. It checks
    what can be checked now: the key structure, that every commitment has a forecast and a salt,
    that the counts are right, and that no outcome exists.
    """
    directory = run_dir(run_id)
    failures: list[str] = []

    commitments = read_commitments(run_id)
    from ..trials.commitment import read_forecasts, record_key

    forecasts = {record_key(forecast): forecast for forecast in read_forecasts(run_id)}
    keys = [record_key(commitment) for commitment in commitments]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        failures.append(f"duplicate commitment keys: {duplicates[:3]}")

    salts_dir = directory / "private_payloads" / "salts"
    salt_count = len(list(salts_dir.glob("*.salt"))) if salts_dir.exists() else 0
    if salt_count != len(commitments):
        failures.append(f"{salt_count} salt files for {len(commitments)} commitments")

    missing_forecasts = [key for key in keys if key not in forecasts]
    if missing_forecasts:
        failures.append(f"{len(missing_forecasts)} commitments have no forecast record")

    for key, forecast in forecasts.items():
        for candidate in forecast.candidate_forecasts:
            if candidate.delta_margin_q05 > candidate.delta_margin_q95:
                failures.append(f"{key}: an interval is inverted")
                break

    ordering = verify_commitment_ordering(run_id)
    if not ordering["ordering_valid"]:
        failures.extend(ordering["failures"])
    if ordering["final_test_outcomes_exist"]:
        failures.append(
            f"outcome artifacts already exist in this run: {ordering['outcome_artifacts_present']}"
        )

    conditions = sorted({f"{k[1]}:{k[2]}:{k[3]}" for k in keys})
    return {
        "run_id": run_id,
        "commitments": len(commitments),
        "forecasts": len(forecasts),
        "salt_files": salt_count,
        "distinct_method_conditions": len(conditions),
        "reveals": 0,
        "final_test_outcomes_exist": ordering["final_test_outcomes_exist"],
        "outcome_artifacts_present": ordering["outcome_artifacts_present"],
        "latest_committed_at": ordering["latest_committed_at"],
        "valid": not failures,
        "failures": failures,
        "notes": (
            "Commitment checkpoint. No reveal exists yet and none should: the salts stay sealed "
            "until the final-test interventions are resolved, and the commitments are verifiable "
            "by a third party only after that."
        ),
    }


def check_feature_policies(
    study: FittedStudy, contexts: Sequence[PromptContext], projection: np.ndarray
) -> dict[str, Any]:
    """Check the two feature policies that carry the comparison.

    The visible model must not move when only the state changes, and a wrong-state substitution
    must change the state and interaction blocks and nothing else.
    """
    if len(contexts) < 2:
        raise PredictError("checking the feature policies needs at least two prompts")
    first, second = contexts[0], contexts[1]
    zero = np.zeros(projection.shape[0], dtype=np.float64)

    visible_clean = visible_block_is_clean(study.transforms, first, second.state)

    blocks = METHOD_BLOCKS[BILINEAR_METHOD]
    true_row = build_feature_row(study.transforms, first, zero, projection, blocks)
    swapped_row = build_feature_row(
        study.transforms, first, zero, projection, blocks, state_override=second.state
    )
    offset = 0
    changed: list[str] = []
    for block in blocks:
        width = BLOCK_WIDTHS[block]
        piece = slice(offset, offset + width)
        if not np.array_equal(true_row[piece], swapped_row[piece]):
            changed.append(block.value)
        offset += width

    expected = {FeatureBlock.STATE.value, FeatureBlock.STATE_INTERVENTION.value}
    # With a zero intervention vector the interaction block is identically zero under both
    # states, so it legitimately does not move; what must never move is I or V.
    forbidden = sorted(set(changed) - expected)
    return {
        "visible_block_ignores_the_state": visible_clean,
        "blocks_changed_by_substitution": sorted(changed),
        "blocks_allowed_to_change": sorted(expected),
        "substitution_touched_forbidden_blocks": forbidden,
        "valid": visible_clean and not forbidden,
    }


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root())).replace("\\", "/")
    except ValueError:
        return path.name


def load_run_config_for(path: str | Path) -> StateAuditRunConfig:
    return load_config(path, StateAuditRunConfig)


__all__ = [
    "BILINEAR_METHOD",
    "DIAGNOSTIC_METHODS",
    "FORECAST_ALGORITHM_VERSION",
    "PERMUTATION_CONDITIONS",
    "RIDGE_METHODS",
    "FittedStudy",
    "PredictError",
    "RunBundle",
    "build_examples",
    "check_feature_policies",
    "commit_final_test_forecasts",
    "direction_vectors",
    "fit_report",
    "fit_study",
    "load_run_bundle",
    "signed_vector",
    "verify_final_test_commitments",
]
