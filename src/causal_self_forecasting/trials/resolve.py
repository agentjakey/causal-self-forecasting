"""Trial resolution: apply interventions and record observations.

Resolution is the second half of the protocol. Generation produced the clean outputs and the
candidate sets; resolution applies interventions and measures what happened. It never edits a
trial or a commitment record, because those are the committed-before-the-fact half of the
evidence. It only appends observations.

Two modes, chosen explicitly rather than inferred, because they answer different questions:

* Forecast mode (default). Requires valid committed forecasts. It reads or writes a selection
  seed after the commitments exist, selects one candidate per trial deterministically, applies
  it, observes it, then reveals the salt and verifies the commitment. This is the path a
  scientific forecast evaluation uses, and it produces one observation per trial tied to the
  selected candidate.

* Ground-truth mode (`--ground-truth`). Applies every candidate in every trial and observes
  each. It requires no forecasts and does no selection or reveal, so it produces the effect
  distribution used to train baselines and to validate the harness on real weights. It is
  never a scientific forecast evaluation and its manifest says so.

A failed intervention is recorded, never dropped. A run where a third of the interventions
silently vanished would report clean-looking numbers over a biased subset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import ResolvedExperiment, resolve_experiment
from ..hashing import (
    append_jsonl,
    atomic_write_json,
    hash_file,
    read_json,
)
from ..interventions.directions import DirectionStore
from ..interventions.tensor_ops import InterventionPayload, InterventionShapeError
from ..logging_utils import info, warn
from ..models.capture import CaptureError, capture_hidden_states, run_with_intervention
from ..models.loader import LoadedModel, ModelLoadError, load_model
from ..models.scoring import (
    LabelTokenError,
    entropy_from_label_logits,
    resolve_label_token_ids,
    score_logits,
)
from ..paths import (
    ARTIFACT_HASHES,
    OBSERVATION_RECORDS,
    RESOLUTION_FAILURES,
    RESOLUTION_MANIFEST,
    RUN_MANIFEST,
    directions_dir,
    run_dir,
)
from ..reproducibility import derive_seed
from ..schemas import (
    ArtifactHashRecord,
    CandidateSet,
    InterventionSpec,
    Mechanism,
    ModelVariant,
    ObservationRecord,
    TrialRecord,
)
from ..tasks.loader import TaskLoadError, load_prepared_task
from .commitment import (
    ProtocolOrderError,
    read_candidate_sets,
    read_forecasts,
    reveal_selection,
    select_candidate_index,
    write_selection_seed,
)
from .commitment import (
    read_commitments as _read_commitments,
)
from .commitment import (
    read_selection_seed as _read_selection_seed,
)
from .commitment import (
    selection_seed_exists as _selection_seed_exists,
)
from .generate import read_trials
from .states import STATES_FILENAME, load_state


class ResolutionError(RuntimeError):
    """Raised when a run cannot be resolved as requested."""


def _run_manifest(run_id: str) -> dict[str, Any]:
    path = run_dir(run_id) / RUN_MANIFEST
    if not path.exists():
        raise ResolutionError(
            f"run {run_id!r} has no run manifest at {path}; it is not a generated trial run"
        )
    return read_json(path)


def _guard_run_kind(run_id: str, manifest: dict[str, Any]) -> None:
    """Refuse runs that are not generated trial runs.

    A systems benchmark has no trials and no candidate sets. Resolving it would be
    meaningless, and letting it through would blur the line the whole project keeps between a
    benchmark and an experiment.
    """
    phase = manifest.get("phase")
    if phase == "systems_benchmark":
        raise ResolutionError(
            f"run {run_id!r} is a systems benchmark, not a trial run; it cannot be resolved"
        )
    if not (run_dir(run_id) / "trial_manifest.jsonl").exists():
        raise ResolutionError(
            f"run {run_id!r} has no trial manifest; generate trials before resolving"
        )


def _resolved_experiment(manifest: dict[str, Any]) -> ResolvedExperiment:
    config_path = manifest.get("config_path")
    if not config_path:
        raise ResolutionError("run manifest does not record the experiment config path")
    try:
        return resolve_experiment(config_path)
    except Exception as error:
        raise ResolutionError(
            f"could not resolve the experiment config {config_path!r} recorded in the run "
            f"manifest: {error}"
        ) from error


def _prompt_lookup(task_name: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return variant_id to prompt text, and item_id to correct label."""
    try:
        items, variants = load_prepared_task(task_name)
    except TaskLoadError as error:
        raise ResolutionError(str(error)) from error
    prompts = {variant.variant_id: variant.prompt_text for variant in variants}
    answers = {item.item_id: item.answer_label for item in items}
    return prompts, answers


def _build_payload(
    spec: InterventionSpec,
    model: LoadedModel,
    store: DirectionStore,
    states_file: Path,
) -> InterventionPayload:
    """Reconstruct the private payload for a candidate from its public spec.

    The candidate records only a direction id or a source state id, never the vector. The
    payload is rebuilt here from the direction store and the state shard, which is what keeps
    the committed candidate set free of the actual intervention contents.
    """
    if spec.mechanism is Mechanism.NOOP:
        return InterventionPayload()
    if spec.mechanism in (Mechanism.RESIDUAL_ADD, Mechanism.RANDOM_ADD, Mechanism.DIRECTION_ABLATE):
        if spec.direction_id is None:
            raise ResolutionError(f"candidate {spec.intervention_id} has no direction id")
        if not store.has(spec.direction_id):
            raise ResolutionError(
                f"direction {spec.direction_id!r} for candidate {spec.intervention_id} is not "
                f"in {store.root}"
            )
        vector = store.load(spec.direction_id, dtype=model.dtype).to(model.device)
        if vector.shape[0] != model.hidden_dim:
            raise ResolutionError(
                f"direction {spec.direction_id!r} has dimension {vector.shape[0]} but the model "
                f"hidden dimension is {model.hidden_dim}"
            )
        return InterventionPayload(direction=vector)
    if spec.mechanism is Mechanism.ACTIVATION_PATCH:
        if spec.source_state_id is None:
            raise ResolutionError(f"candidate {spec.intervention_id} has no source state id")
        source = load_state(states_file, spec.source_state_id, dtype=model.dtype).to(model.device)
        return InterventionPayload(source_activation=source)
    raise ResolutionError(f"unhandled mechanism {spec.mechanism}")


def _apply_and_observe(
    trial: TrialRecord,
    spec: InterventionSpec,
    prompt: str,
    correct_label: str,
    model: LoadedModel,
    label_token_ids: dict[str, int],
    store: DirectionStore,
    states_file: Path,
) -> ObservationRecord:
    """Apply one candidate to one trial and build its observation.

    The clean baseline comes from the committed trial record, not from a fresh clean forward.
    That is deliberate: the delta must be measured against exactly the clean output the
    forecast was made against, and the harness guarantees clean reruns are deterministic, so
    a fresh clean forward would only reproduce the committed one at extra cost.
    """
    payload = _build_payload(spec, model, store, states_file)
    result = run_with_intervention(model, prompt, spec, payload)
    post = score_logits(result.next_token_logits, label_token_ids, correct_label)

    diagnostics = result.diagnostics
    if diagnostics is None:
        raise ResolutionError(
            f"intervention {spec.intervention_id} produced no diagnostics; cannot record norms"
        )

    return ObservationRecord(
        trial_id=trial.trial_id,
        intervention_id=spec.intervention_id,
        mechanism=spec.mechanism,
        clean_logits=trial.clean_logits,
        post_logits=post.logits,
        clean_margin=trial.clean_margin,
        post_margin=post.margin,
        delta_margin=post.margin - trial.clean_margin,
        clean_predicted_label=trial.clean_predicted_label,
        post_predicted_label=post.predicted_label,
        answer_flip=trial.clean_predicted_label != post.predicted_label,
        clean_entropy=entropy_from_label_logits(trial.clean_logits),
        post_entropy=post.entropy,
        pre_norm=diagnostics.pre_norm,
        post_norm=diagnostics.post_norm,
    )


def _check_clean_reproduces(
    trial: TrialRecord,
    prompt: str,
    correct_label: str,
    model: LoadedModel,
    label_token_ids: dict[str, int],
    tolerance: float,
) -> None:
    """One-time guard that the resolution model matches the committed clean baseline.

    Run once, on the first trial, not per trial. A model loaded at a different precision or
    from a different revision would score every intervention on a different scale than the
    committed clean, turning every delta into noise while nothing looked wrong. This catches
    that gross mismatch cheaply.
    """
    clean = capture_hidden_states(model, prompt, layers=[])
    scored = score_logits(clean.next_token_logits, label_token_ids, correct_label)
    drift = abs(scored.margin - trial.clean_margin)
    if drift > tolerance:
        raise ResolutionError(
            f"the resolution model does not reproduce the committed clean baseline for trial "
            f"{trial.trial_id}: recomputed margin {scored.margin:.6f} versus committed "
            f"{trial.clean_margin:.6f} (drift {drift:.6f} exceeds tolerance {tolerance:.6f}). "
            "The run was generated with a different model, revision, or precision."
        )


def _selected_candidates(
    run_id: str,
    trials: list[TrialRecord],
    candidate_sets: dict[str, CandidateSet],
    selection_seed_file: str | Path | None,
    force_seed: bool,
) -> tuple[dict[str, InterventionSpec], str]:
    """Resolve the selection seed and pick one candidate per trial.

    The seed is read from a file when given, otherwise generated now. Generating it here is
    correct in forecast mode: commitments already exist, so a seed created at resolution time
    still post-dates them, which is the property the blinding depends on.
    """
    if selection_seed_file is not None:
        seed_hex = _read_selection_seed(run_id, selection_seed_file)
        if not _selection_seed_exists(run_id):
            write_selection_seed(run_id, seed_hex, force=force_seed)
    elif _selection_seed_exists(run_id):
        seed_hex = _read_selection_seed(run_id)
    else:
        seed_hex = derive_seed("resolution_selection", run_id).to_bytes(8, "big").hex()
        write_selection_seed(run_id, seed_hex, force=force_seed)

    selected: dict[str, InterventionSpec] = {}
    for trial in trials:
        candidate_set = candidate_sets[trial.trial_id]
        index = select_candidate_index(seed_hex, candidate_set)
        selected[trial.trial_id] = candidate_set.candidates[index]
    return selected, seed_hex


def resolve_run(
    run_id: str,
    selection_seed_file: str | Path | None = None,
    ground_truth: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Resolve a generated trial run and write observations."""
    directory = run_dir(run_id)
    manifest = _run_manifest(run_id)
    _guard_run_kind(run_id, manifest)

    observations_path = directory / OBSERVATION_RECORDS
    if observations_path.exists() and not force:
        raise ResolutionError(
            f"{observations_path} already exists; pass --force to resolve the run again"
        )

    trials = read_trials(run_id)
    if not trials:
        raise ResolutionError(f"run {run_id!r} has no trials to resolve")
    candidate_sets = {cs.trial_id: cs for cs in read_candidate_sets(run_id)}
    missing = [trial.trial_id for trial in trials if trial.trial_id not in candidate_sets]
    if missing:
        raise ResolutionError(f"trials are missing candidate sets: {missing[:5]}")

    resolved = _resolved_experiment(manifest)
    fixture_only = resolved.model.kind == "fixture"

    forecasts = read_forecasts(run_id)
    forecast_mode = not ground_truth
    if forecast_mode and not forecasts:
        raise ResolutionError(
            f"run {run_id!r} has no committed forecasts. Commit forecasts before resolving a "
            "scientific forecast run, or pass --ground-truth to apply every candidate and "
            "record observations without scoring a forecaster."
        )
    if forecast_mode:
        _validate_commitments(run_id, trials)

    model = _load_resolution_model(resolved, trials)
    store = DirectionStore(directions_dir())
    states_file = directory / STATES_FILENAME
    prompts, answers = _prompt_lookup(resolved.task.name)

    try:
        label_token_ids = resolve_label_token_ids(
            model.tokenizer, resolved.task.answer_labels, prefix=resolved.task.label_prefix
        )
    except LabelTokenError as error:
        raise ResolutionError(str(error)) from error

    # Guard against a model that does not reproduce the committed clean baseline, once.
    first = trials[0]
    _check_clean_reproduces(
        first,
        prompts[first.variant_id],
        answers[first.item_id],
        model,
        label_token_ids,
        resolved.experiment.rerun_tolerance,
    )

    if forecast_mode:
        selected, seed_hex = _selected_candidates(
            run_id, trials, candidate_sets, selection_seed_file, force
        )
    else:
        selected, seed_hex = {}, None

    observations: list[ObservationRecord] = []
    failures: list[dict[str, Any]] = []
    reveals_verified = 0
    reveals_total = 0

    # Fresh output files, so a forced re-resolution does not append to a stale run.
    for name in (OBSERVATION_RECORDS, RESOLUTION_FAILURES):
        (directory / name).unlink(missing_ok=True)

    forecast_by_trial = {f.trial_id: f for f in forecasts}
    commitments = {c.trial_id: c for c in _read_commitments(run_id)} if forecast_mode else {}

    for trial in trials:
        candidate_set = candidate_sets[trial.trial_id]
        specs = [selected[trial.trial_id]] if forecast_mode else list(candidate_set.candidates)

        for spec in specs:
            try:
                observation = _apply_and_observe(
                    trial,
                    spec,
                    prompts[trial.variant_id],
                    answers[trial.item_id],
                    model,
                    label_token_ids,
                    store,
                    states_file,
                )
            except (
                ResolutionError,
                CaptureError,
                InterventionShapeError,
                KeyError,
            ) as error:
                # Preserved, never dropped. A silently omitted failure would bias the run.
                failure = {
                    "trial_id": trial.trial_id,
                    "intervention_id": spec.intervention_id,
                    "mechanism": spec.mechanism.value,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failures.append(failure)
                append_jsonl(directory / RESOLUTION_FAILURES, failure)
                warn(
                    "intervention failed during resolution",
                    trial_id=trial.trial_id,
                    intervention_id=spec.intervention_id,
                    error=type(error).__name__,
                )
                continue

            observations.append(observation)
            append_jsonl(directory / OBSERVATION_RECORDS, observation)

        if forecast_mode and seed_hex is not None and trial.trial_id in commitments:
            forecast = forecast_by_trial.get(trial.trial_id)
            if forecast is not None:
                try:
                    reveal = reveal_selection(
                        run_id, candidate_set, commitments[trial.trial_id], forecast, seed_hex
                    )
                    reveals_total += 1
                    reveals_verified += int(reveal.verified)
                except ProtocolOrderError as error:
                    warn("could not reveal commitment", trial_id=trial.trial_id, error=str(error))

    return _write_resolution_manifest(
        run_id=run_id,
        directory=directory,
        resolved=resolved,
        model_spec_variant=model.spec.variant,
        fixture_only=fixture_only,
        forecast_mode=forecast_mode,
        trials=trials,
        observations=observations,
        failures=failures,
        reveals_total=reveals_total,
        reveals_verified=reveals_verified,
    )


def _validate_commitments(run_id: str, trials: list[TrialRecord]) -> None:
    """Refuse missing, malformed, duplicated, or mismatched commitments.

    Checked before any intervention runs, so a broken commitment set fails fast rather than
    after minutes of forward passes.
    """
    commitments = _read_commitments(run_id)
    forecasts = read_forecasts(run_id)

    seen: set[tuple[str, str]] = set()
    for commitment in commitments:
        key = (commitment.trial_id, commitment.method_id)
        if key in seen:
            raise ResolutionError(
                f"duplicate commitment for trial {commitment.trial_id} method "
                f"{commitment.method_id}"
            )
        seen.add(key)

    forecast_keys = {(f.trial_id, f.method_id) for f in forecasts}
    for commitment in commitments:
        if (commitment.trial_id, commitment.method_id) not in forecast_keys:
            raise ResolutionError(
                f"commitment for trial {commitment.trial_id} has no matching forecast record"
            )

    trial_ids = {trial.trial_id for trial in trials}
    committed_trials = {commitment.trial_id for commitment in commitments}
    uncommitted = sorted(trial_ids - committed_trials)
    if uncommitted:
        raise ResolutionError(
            f"these trials have no committed forecast: {uncommitted[:5]}"
            f"{' and more' if len(uncommitted) > 5 else ''}. Commit forecasts for every trial "
            "or resolve with --ground-truth."
        )


def _load_resolution_model(resolved: ResolvedExperiment, trials: list[TrialRecord]) -> LoadedModel:
    variant = trials[0].model_variant
    try:
        model = load_model(resolved.model)
    except ModelLoadError as error:
        raise ResolutionError(f"could not load the resolution model: {error}") from error
    if model.spec.variant is not variant:
        raise ResolutionError(
            f"trials were generated with the {variant.value} model but the config loads the "
            f"{model.spec.variant.value} model; resolution must use the same weights"
        )
    return model


def _write_resolution_manifest(
    run_id: str,
    directory: Path,
    resolved: ResolvedExperiment,
    model_spec_variant: ModelVariant,
    fixture_only: bool,
    forecast_mode: bool,
    trials: list[TrialRecord],
    observations: list[ObservationRecord],
    failures: list[dict[str, Any]],
    reveals_total: int,
    reveals_verified: int,
) -> dict[str, Any]:
    # Classification mirrors the benchmark's honesty rule. A fixture run is never scientific.
    # A ground-truth run is observation generation, not a forecast evaluation. Only a
    # forecast-mode run on a real model, with every commitment verified, is a scientific
    # forecast resolution, and even that is a resolution, not a published result.
    if fixture_only:
        classification = "fixture_resolution"
    elif not forecast_mode:
        classification = "ground_truth_resolution"
    else:
        classification = "scientific_forecast_resolution"

    scientific = classification == "scientific_forecast_resolution"
    commitments_verified = forecast_mode and reveals_total > 0 and reveals_verified == reveals_total

    provenance: list[dict[str, Any]] = []
    for name, kind in (
        (OBSERVATION_RECORDS, "observations"),
        (RESOLUTION_FAILURES, "resolution_failures"),
    ):
        path = directory / name
        if path.exists():
            provenance.append(
                ArtifactHashRecord(
                    path=name,
                    hash=hash_file(path),
                    size_bytes=path.stat().st_size,
                    kind=kind,
                ).model_dump(mode="json")
            )

    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "phase": "trials_resolve",
        "classification": classification,
        "scientific_result": False,
        "scientific_forecast_resolution": scientific,
        "fixture_only": fixture_only,
        "mode": "forecast" if forecast_mode else "ground_truth",
        "model": resolved.model.model_id,
        "model_variant": model_spec_variant.value,
        "counts": {
            "trials": len(trials),
            "observations": len(observations),
            "failures": len(failures),
            "reveals_total": reveals_total,
            "reveals_verified": reveals_verified,
        },
        "commitments_verified": commitments_verified,
        "provenance": provenance,
        "notes": (
            "Trial resolution. Observations only. This is not a CSF-Bench scientific result "
            "and the public exporter does not accept it."
        ),
    }
    atomic_write_json(directory / RESOLUTION_MANIFEST, manifest)

    # A resolution-scoped artifact hash file, named so it never clobbers the generation one.
    atomic_write_json(
        directory / f"resolution_{ARTIFACT_HASHES}",
        [
            ArtifactHashRecord(
                path=RESOLUTION_MANIFEST,
                hash=hash_file(directory / RESOLUTION_MANIFEST),
                size_bytes=(directory / RESOLUTION_MANIFEST).stat().st_size,
                kind="resolution_manifest",
            ).model_dump(mode="json"),
            *provenance,
        ],
    )

    info(
        "resolved trials",
        run_id=run_id,
        classification=classification,
        observations=len(observations),
        failures=len(failures),
        reveals_verified=f"{reveals_verified}/{reveals_total}",
    )
    result = dict(manifest)
    result["observations_path"] = str(directory / OBSERVATION_RECORDS)
    return result


def read_observations(run_id: str) -> list[ObservationRecord]:
    from ..hashing import read_jsonl

    path = run_dir(run_id) / OBSERVATION_RECORDS
    if not path.exists():
        return []
    return [ObservationRecord.model_validate(row) for row in read_jsonl(path)]


__all__ = ["ResolutionError", "read_observations", "resolve_run"]
