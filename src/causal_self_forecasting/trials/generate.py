"""Trial generation.

Produces the fixed, pre-forecast half of a run: the clean outputs, the captured states, and
the candidate sets. Nothing here depends on any forecast, and no forecaster is loaded, which
is what makes it safe to generate trials and forecast them in separate processes.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from ..config import InterventionConfig, ResolvedExperiment
from ..hashing import atomic_write_json, read_jsonl, write_jsonl
from ..interventions.directions import (
    DirectionStore,
    matched_random_direction,
    random_direction_seed,
)
from ..logging_utils import info, warn
from ..models.capture import capture_hidden_states, validate_layer
from ..models.loader import load_model
from ..models.scoring import resolve_label_token_ids, score_logits
from ..paths import (
    CANDIDATE_SETS,
    ENVIRONMENT,
    RUN_MANIFEST,
    TRIAL_MANIFEST,
    directions_dir,
    ensure_run_dir,
    run_dir,
)
from ..reproducibility import derive_seed, environment_snapshot, set_global_seed
from ..schemas import (
    CandidateSet,
    Mechanism,
    PromptVariant,
    RunManifest,
    TaskItem,
    TrialRecord,
)
from ..tasks.loader import TaskLoadError, load_prepared_task
from .candidates import build_candidate_set, candidate_set_hash, default_templates
from .states import STATES_FILENAME, StateShardWriter


class TrialGenerationError(RuntimeError):
    """Raised when trials cannot be generated as configured."""


def _load_processed(task_name: str) -> tuple[list[TaskItem], list[PromptVariant]]:
    """Read prepared data, reporting a missing dataset as a trial-generation failure."""
    try:
        return load_prepared_task(task_name)
    except TaskLoadError as error:
        raise TrialGenerationError(str(error)) from error


def _training_grid(
    interventions: list[InterventionConfig], heldout: list[Mechanism]
) -> list[tuple[int, float]]:
    """Build the (layer, strength) grid trials are drawn from.

    Held-out mechanisms are excluded here, not filtered later. A held-out mechanism that
    reaches trial generation would end up in the training data of any forecaster fit on that
    run, and the transfer test would be silently invalid.
    """
    grid: list[tuple[int, float]] = []
    for config in interventions:
        if config.mechanism in heldout:
            warn(
                "skipping a held-out mechanism during trial generation",
                mechanism=config.mechanism.value,
                config=config.name,
            )
            continue
        if config.mechanism is not Mechanism.RESIDUAL_ADD:
            continue
        for layer in config.layers:
            for strength in config.strengths:
                # Magnitude only. `default_templates` builds the positive and negative
                # candidates from it, so a signed grid would double every trial.
                if strength > 0:
                    grid.append((layer, float(strength)))
    if not grid:
        raise TrialGenerationError(
            "no residual_add intervention grid was found; trial generation needs at least one "
            "residual_add config with a positive strength"
        )
    return sorted(set(grid))


def _ensure_random_controls(
    store: DirectionStore,
    direction_id: str,
    count: int,
    run_seed: int,
    hidden_dim: int,
) -> list[str]:
    """Create norm-matched random control directions for this run.

    Derived from the run seed, so they are reproducible, and matched to the norm of the
    direction under test, so a difference in effect cannot be explained by a difference in
    magnitude.
    """
    reference = store.load(direction_id)
    if reference.shape[0] != hidden_dim:
        raise TrialGenerationError(
            f"direction {direction_id!r} has dimension {reference.shape[0]} but the model's "
            f"hidden dimension is {hidden_dim}; the direction was estimated on a different model"
        )

    control_ids: list[str] = []
    for index in range(count):
        control_id = f"{direction_id}__random_{index:02d}"
        if not store.has(control_id):
            seed = random_direction_seed(run_seed, direction_id, index)
            vector = matched_random_direction(reference, seed)
            store.save(
                control_id,
                vector,
                {
                    "method": "matched_random_control",
                    "reference_direction_id": direction_id,
                    "seed": seed,
                    "run_seed": run_seed,
                    "validated": False,
                    "note": "norm-matched random vector; a control, not a discovered direction",
                },
            )
        control_ids.append(control_id)
    return control_ids


def generate_trials(
    resolved: ResolvedExperiment,
    run_id: str | None = None,
    max_trials: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate a run's trials and write its artifacts."""
    experiment = resolved.experiment
    set_global_seed(experiment.seed)

    items, variants = _load_processed(resolved.task.name)
    answer_by_item = {item.item_id: item.answer_label for item in items}

    selected_splits = set(experiment.trial_splits)
    eligible = [variant for variant in variants if variant.split in selected_splits]
    if not eligible:
        raise TrialGenerationError(
            f"no prepared variants fall in splits {sorted(s.value for s in selected_splits)}"
        )
    eligible.sort(key=lambda variant: variant.variant_id)

    limit = max_trials if max_trials is not None else experiment.max_trials
    if limit is not None and limit < len(eligible):
        # Subsample with a seeded shuffle rather than truncating the sorted list. Truncation
        # would take every wrapper variant of the first few items, which in practice means a
        # capped run covers one split and one or two questions. That looks like a working run
        # and silently answers a different question than the config asks.
        picker = random.Random(derive_seed("trial_selection", experiment.seed, resolved.task.name))
        eligible = sorted(picker.sample(eligible, limit), key=lambda variant: variant.variant_id)

    grid = _training_grid(resolved.interventions, experiment.heldout_mechanisms)

    if dry_run:
        return {
            "dry_run": True,
            "trials": len(eligible),
            "grid": [{"layer": layer, "strength": strength} for layer, strength in grid],
            "splits": sorted({variant.split.value for variant in eligible}),
            "direction_id": experiment.direction_id,
        }

    model = load_model(resolved.model)
    for layer in experiment.capture_layers:
        validate_layer(layer, model)
    for layer, _ in grid:
        validate_layer(layer, model)

    store = DirectionStore(directions_dir())
    if not store.has(experiment.direction_id):
        raise TrialGenerationError(
            f"direction {experiment.direction_id!r} is not in {directions_dir()}; "
            "create it first. `csf directions synthetic` builds a seeded random vector for the "
            "smoke pipeline. Data-estimated direction discovery is not implemented yet, so a "
            "real experiment must supply its direction by another route and record how it was "
            "built in the direction metadata."
        )
    control_ids = _ensure_random_controls(
        store,
        experiment.direction_id,
        experiment.random_control_count,
        experiment.seed,
        model.hidden_dim,
    )

    label_token_ids = resolve_label_token_ids(
        model.tokenizer, resolved.task.answer_labels, prefix=resolved.task.label_prefix
    )
    info("resolved answer label tokens", tokens=label_token_ids)

    identifier = run_id or ""
    directory = ensure_run_dir(identifier)
    shard = StateShardWriter(directory / STATES_FILENAME)

    trial_records: list[TrialRecord] = []
    candidate_sets: list[CandidateSet] = []

    for index, variant in enumerate(eligible):
        trial_id = f"trial_{index:05d}"
        layer, strength = grid[index % len(grid)]

        capture = capture_hidden_states(
            model,
            variant.prompt_text,
            layers=sorted({*experiment.capture_layers, layer}),
            position_index=experiment.capture_position,
        )
        scores = score_logits(
            capture.next_token_logits, label_token_ids, answer_by_item[variant.item_id]
        )

        state_ids: dict[int, str] = {}
        for captured_layer, vector in capture.hidden_states.items():
            state_id = f"{variant.variant_id}.L{captured_layer}"
            shard.add(
                state_id=state_id,
                variant_id=variant.variant_id,
                model_variant=model.spec.variant,
                layer=captured_layer,
                position_index=capture.position_absolute,
                vector=vector,
            )
            state_ids[captured_layer] = state_id

        control_id = control_ids[
            derive_seed("control_pick", experiment.seed, trial_id) % len(control_ids)
        ]
        templates = default_templates(
            experiment.direction_id, control_id, layer, experiment.capture_position, strength
        )
        candidate_set = build_candidate_set(trial_id, experiment.seed, templates)
        candidate_sets.append(candidate_set)

        trial_records.append(
            TrialRecord(
                trial_id=trial_id,
                variant_id=variant.variant_id,
                item_id=variant.item_id,
                group_id=variant.group_id,
                model_variant=model.spec.variant,
                framing=variant.framing,
                split=variant.split,
                state_id=state_ids[layer],
                clean_logits=scores.logits,
                clean_margin=scores.margin,
                clean_predicted_label=scores.predicted_label,
                candidate_set_hash=candidate_set_hash(candidate_set),
            )
        )

    state_refs = shard.close()
    write_jsonl(directory / TRIAL_MANIFEST, trial_records)
    write_jsonl(directory / CANDIDATE_SETS, candidate_sets)
    write_jsonl(directory / "state_refs.jsonl", state_refs)

    environment = environment_snapshot()
    atomic_write_json(directory / ENVIRONMENT, environment)

    manifest = RunManifest(
        run_id=identifier,
        phase="trials_generate",
        config_path=resolved.experiment_path,
        config_hash=resolved.experiment_hash,
        seed=experiment.seed,
        models=[model.spec],
        counts={
            "trials": len(trial_records),
            "candidate_sets": len(candidate_sets),
            "states": len(state_refs),
            "grid_points": len(grid),
            "random_controls": len(control_ids),
        },
        environment=environment,
        status="complete",
        notes=("Trials only. No forecasts, observations, or scores exist in this run yet."),
    )
    atomic_write_json(directory / RUN_MANIFEST, manifest.model_dump(mode="json"))

    info(
        "generated trials",
        run_id=identifier,
        trials=len(trial_records),
        states=len(state_refs),
        path=str(directory),
    )
    return {
        "run_id": identifier,
        "trials": len(trial_records),
        "states": len(state_refs),
        "path": str(directory),
        "splits": _count_splits(trial_records),
    }


def _count_splits(trials: list[TrialRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trial in trials:
        counts[trial.split.value] = counts.get(trial.split.value, 0) + 1
    return counts


def read_trials(run_id: str) -> list[TrialRecord]:
    path = run_dir(run_id) / TRIAL_MANIFEST
    if not path.exists():
        raise TrialGenerationError(f"no trial manifest at {path}")
    return [TrialRecord.model_validate(row) for row in read_jsonl(path)]


def states_path(run_id: str) -> Path:
    return run_dir(run_id) / STATES_FILENAME


__all__ = ["TrialGenerationError", "generate_trials", "read_trials", "states_path"]
