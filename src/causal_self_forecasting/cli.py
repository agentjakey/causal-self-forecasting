"""Command-line interface.

Commands are added as their phase lands. There are deliberately no placeholder commands for
unimplemented phases: a command that exists but does nothing is worse than one that is
absent, because it suggests the pipeline is further along than it is.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer

from . import SCHEMA_VERSION, __version__
from .config import (
    ConfigError,
    ExperimentConfig,
    ModelConfig,
    TaskConfig,
    config_hash,
    load_config,
    repo_root,
    resolve_experiment,
)
from .hashing import atomic_write_json
from .logging_utils import configure_logging, info
from .paths import RUN_LOG, ensure_run_dir, new_run_id, run_dir
from .schemas import Split
from .state_audit.bundle import BUNDLE_ID

app = typer.Typer(
    name="csf",
    help="CSF-Bench: precommitted forecasting of blinded internal interventions.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Prepare task datasets.", no_args_is_help=True)
prompts_app = typer.Typer(help="Freeze and inspect prompt-role manifests.", no_args_is_help=True)
calibration_app = typer.Typer(
    help="Plan, summarize, and decide intervention-strength calibration.", no_args_is_help=True
)
directions_app = typer.Typer(
    help="Create and inspect intervention directions.", no_args_is_help=True
)
state_audit_app = typer.Typer(
    help="Execute and verify BlueDot state-dependence runs.", no_args_is_help=True
)
interventions_app = typer.Typer(help="Validate the intervention harness.", no_args_is_help=True)
trials_app = typer.Typer(help="Generate and resolve trials.", no_args_is_help=True)
score_app = typer.Typer(help="Score resolved runs.", no_args_is_help=True)
verify_app = typer.Typer(help="Verify run artifacts.", no_args_is_help=True)

app.add_typer(data_app, name="data")
app.add_typer(prompts_app, name="prompts")
app.add_typer(calibration_app, name="calibration")
app.add_typer(directions_app, name="directions")
app.add_typer(state_audit_app, name="state-audit")
app.add_typer(interventions_app, name="interventions")
app.add_typer(trials_app, name="trials")
app.add_typer(score_app, name="score")
app.add_typer(verify_app, name="verify")


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    configure_logging("DEBUG" if verbose else "INFO")


@app.command()
def version() -> None:
    """Print the package and schema versions."""
    _echo_json({"version": __version__, "schema_version": SCHEMA_VERSION})


@app.command()
def doctor() -> None:
    """Report the environment and validate every config in the repository.

    Exits nonzero if a config does not validate. Everything else is reported rather than
    enforced: a missing GPU or missing model access is a fact about the machine, not a
    reason this command should fail.
    """
    from .models.device import available_devices
    from .reproducibility import environment_snapshot

    report: dict[str, Any] = {
        "version": __version__,
        "schema_version": SCHEMA_VERSION,
        "environment": environment_snapshot(),
        "devices": available_devices(),
    }

    from .config import (
        CalibrationPlanConfig,
        DirectionFamilyConfig,
        InterventionConfig,
        PromptManifestConfig,
        StateAuditRunConfig,
    )

    config_types: list[tuple[str, type]] = [
        ("configs/models", ModelConfig),
        ("configs/tasks", TaskConfig),
        ("configs/experiments", ExperimentConfig),
        ("configs/interventions", InterventionConfig),
        ("configs/prompts", PromptManifestConfig),
        ("configs/directions", DirectionFamilyConfig),
        ("configs/calibration", CalibrationPlanConfig),
        ("configs/state_audit", StateAuditRunConfig),
    ]

    results: dict[str, Any] = {}
    failures: list[str] = []
    for directory, config_type in config_types:
        for path in sorted((repo_root() / directory).glob("*.yaml")):
            relative = str(path.relative_to(repo_root())).replace("\\", "/")
            try:
                load_config(path, config_type)
                results[relative] = {"valid": True, "hash": config_hash(path)}
            except ConfigError as error:
                results[relative] = {"valid": False, "error": str(error)}
                failures.append(relative)
    report["configs"] = results

    devices = report["devices"]
    notes: list[str] = []
    if not devices["cuda"]:
        notes.append(
            "No CUDA device. The fixture-based smoke pipeline and the test suite run on CPU. "
            "Real Gemma trial sweeps and model-organism training are expected to need a GPU."
        )
    notes.append(
        "Gemma 3 weights are gated on Hugging Face. Accept the license on the model page and "
        "authenticate (`hf auth login`) before running configs/models/gemma3_1b_it.yaml."
    )
    report["notes"] = notes
    report["configs_valid"] = not failures

    _echo_json(report)
    if failures:
        typer.secho(f"invalid configs: {failures}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@data_app.command("prepare")
def data_prepare(
    config: Path = typer.Option(..., "--config", help="Path to a task config."),
    max_items: int | None = typer.Option(None, "--max-items", help="Limit the number of items."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report what would be written."),
) -> None:
    """Load a task, render wrapper variants, and write the processed dataset."""
    from .tasks.loader import load_task_items, prepare_task

    task_config = load_config(config, TaskConfig)

    if dry_run:
        items = load_task_items(task_config, max_items=max_items)
        _echo_json(
            {
                "dry_run": True,
                "task": task_config.name,
                "items": len(items),
                "variants": len(items) * len(task_config.wrappers),
                "wrappers": [w.wrapper_id for w in task_config.wrappers],
            }
        )
        return

    manifest = prepare_task(task_config, str(config), max_items=max_items)
    _echo_json(manifest)


@prompts_app.command("manifest")
def prompts_manifest(
    config: Path = typer.Option(..., "--config", help="Path to a prompt-manifest config."),
    seed: int | None = typer.Option(
        None, "--seed", help="Override the master seed. Changes the split, and the hash."
    ),
    force: bool = typer.Option(
        False, "--force", help="Replace an existing, different manifest at the same path."
    ),
) -> None:
    """Freeze a deterministic, role-labeled prompt split.

    Selection is a pure function of the master seed and the group ids. It never looks at model
    correctness, confidence, logits, hidden states, or any outcome, so the split cannot be
    chosen to suit a result. No model is loaded and no forward pass runs.

    Rerunning with identical inputs leaves an identical manifest untouched rather than
    rewriting it, so the file on disk stays byte-identical. A manifest that differs is refused
    unless force is passed, because a frozen split must not be replaced silently.
    """
    from .tasks.prompt_manifest import PromptManifestError, generate_prompt_manifest

    try:
        report = generate_prompt_manifest(config, master_seed=seed, force=force)
    except PromptManifestError as error:
        typer.secho(f"prompt manifest failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    _echo_json(report)

    if not report["task_artifacts_match"]:
        typer.secho(
            "the prepared task artifacts no longer match the ones this manifest was built "
            "against; the split refers to a pool that has changed",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@prompts_app.command("verify")
def prompts_verify(
    manifest_id: str = typer.Option(..., "--manifest-id", help="Manifest id to verify."),
) -> None:
    """Recheck a frozen manifest against the task artifacts currently on disk.

    Loading the manifest already recomputes its content hash, so an edited file fails here
    before anything else is checked.
    """
    from .tasks.prompt_manifest import (
        PromptManifestError,
        load_prompt_manifest,
        verify_prompt_manifest,
    )

    try:
        manifest = load_prompt_manifest(manifest_id)
    except PromptManifestError as error:
        typer.secho(f"prompt manifest failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    report = verify_prompt_manifest(manifest)
    report["total_prompts"] = len(manifest.assignments)
    report["role_counts"] = dict(manifest.role_counts)
    report["manifest_hash"] = manifest.manifest_hash
    _echo_json(report)

    if not report["valid"]:
        typer.secho(
            f"manifest {manifest_id} does not match the prepared task: {report['mismatches']}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@directions_app.command("synthetic")
def directions_synthetic(
    config: Path = typer.Option(..., "--config", help="Path to an experiment config."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing direction."),
) -> None:
    """Create a seeded random direction for the smoke pipeline.

    This is not a bias direction and has passed no causal validation. It exists so the
    plumbing can be exercised before any real direction exists. A direction discovered from
    data must be validated against matched random controls before it is called meaningful;
    that discovery pipeline is not implemented, and no command in this CLI produces one.
    """
    import torch

    from .interventions.directions import DirectionStore
    from .models.loader import load_model
    from .paths import directions_dir
    from .reproducibility import derive_seed

    resolved = resolve_experiment(config)
    store = DirectionStore(directions_dir())
    direction_id = resolved.experiment.direction_id

    if store.has(direction_id) and not force:
        _echo_json(
            {
                "direction_id": direction_id,
                "created": False,
                "reason": "already exists; pass --force to overwrite",
                "metadata": store.metadata(direction_id),
            }
        )
        return

    model = load_model(resolved.model)
    seed = derive_seed("synthetic_direction", resolved.experiment.seed, direction_id)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed % (2**63 - 1))
    vector = torch.randn(model.hidden_dim, generator=generator, dtype=torch.float32)
    vector = vector / torch.linalg.vector_norm(vector)

    path = store.save(
        direction_id,
        vector,
        {
            "method": "synthetic_random_unit",
            "seed": seed,
            "model_id": resolved.model.model_id,
            "model_kind": resolved.model.kind,
            "hidden_dim": model.hidden_dim,
            "validated": False,
            "warning": (
                "randomly generated unit vector for pipeline testing; not estimated from data "
                "and not causally validated. Never describe this as a bias direction."
            ),
        },
    )
    _echo_json(
        {
            "direction_id": direction_id,
            "created": True,
            "path": str(path),
            "dim": model.hidden_dim,
            "validated": False,
        }
    )


@calibration_app.command("plan")
def calibration_plan(
    config: Path = typer.Option(..., "--config", help="Path to a calibration-plan config."),
    force: bool = typer.Option(
        False, "--force", help="Replace an existing, different plan at the same path."
    ),
) -> None:
    """Freeze the calibration plan: thresholds, ratios, layers, and expected counts.

    Loads no model. It reads the frozen prompt manifest, the frozen direction family, and the
    pinned model config's identity, cross-checks them, and does arithmetic.

    This is calibration planning, not calibration. No prompt is run, no state norm is measured,
    no intervention is applied, and nothing it writes is a scientific result. Rerunning with
    identical inputs leaves an identical plan untouched; a different plan is refused unless
    force is passed, because thresholds chosen after seeing the numbers are not thresholds.
    """
    from .calibration.plan import CalibrationPlanError, plan_command

    try:
        report = plan_command(config, force=force)
    except CalibrationPlanError as error:
        typer.secho(f"calibration plan failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    _echo_json(report)

    if not report["verification"]["valid"]:
        typer.secho(
            f"the plan does not match the manifests it cites: {report['verification']['failures']}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@calibration_app.command("verify-plan")
def calibration_verify_plan(
    plan_id: str = typer.Option(..., "--plan-id", help="Calibration plan id to verify."),
) -> None:
    """Recheck a frozen plan against the manifests it cites. Loads no model.

    Loading the plan already recomputes its own content hash, so an edited file fails here
    before anything else is checked.
    """
    from .calibration.plan import (
        CalibrationPlanError,
        load_calibration_plan,
        plan_path,
        plan_report,
        verify_calibration_plan,
    )

    try:
        record = load_calibration_plan(plan_id)
    except CalibrationPlanError as error:
        typer.secho(f"calibration plan failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    report = plan_report(record, "verified", plan_path(plan_id))
    report["verification"] = verify_calibration_plan(record)
    _echo_json(report)

    if not report["verification"]["valid"]:
        typer.secho(
            f"calibration plan {plan_id} did not verify: {report['verification']['failures']}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@calibration_app.command("summarize")
def calibration_summarize(
    plan_id: str = typer.Option(..., "--plan-id", help="Calibration plan to judge against."),
    observations: Path = typer.Option(
        ..., "--observations", help="JSONL of state-audit observations to summarize."
    ),
    layer: int = typer.Option(..., "--layer", help="Which calibrated layer to summarize."),
    failures: Path | None = typer.Option(
        None, "--failures", help="JSONL of recorded intervention failures, if any."
    ),
    output: Path | None = typer.Option(None, "--output", help="Write the summaries to a file."),
) -> None:
    """Compute one ratio summary per preregistered ratio from supplied observations.

    Loads no model. Every observation revalidates its own target against its own logits as it is
    read, so a summary can only be built from records that already recompute.

    The summaries say whether each grid point met the preregistered conditions. They are
    calibration infrastructure, not a scientific result.
    """
    from .calibration.criteria import CriteriaError
    from .calibration.observations import (
        ObservationLoadError,
        read_failures,
        read_state_audit_observations,
        summarize_layer,
    )
    from .calibration.plan import CalibrationPlanError, load_calibration_plan
    from .calibration.strength import StrengthError

    try:
        plan = load_calibration_plan(plan_id)
        records = read_state_audit_observations(observations)
        recorded_failures = read_failures(failures)
        summaries = summarize_layer(records, plan, layer, failures=recorded_failures)
    except (CalibrationPlanError, ObservationLoadError, StrengthError, CriteriaError) as error:
        typer.secho(f"calibration summary failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    payload = {
        "plan_id": plan.plan_id,
        "plan_hash": plan.plan_hash,
        "layer": layer,
        "observations_read": len(records),
        "failures_read": len(recorded_failures),
        "summaries": [summary.model_dump(mode="json") for summary in summaries],
        "passing_ratios": [s.norm_ratio for s in summaries if s.passed],
        "scientific_result": False,
        "notes": (
            "Calibration infrastructure. These summaries describe an intervention-strength "
            "grid; they are not a measurement of the model's abilities."
        ),
    }
    _echo_json(payload)
    if output is not None:
        atomic_write_json(output, payload)


@calibration_app.command("select")
def calibration_select(
    plan_id: str = typer.Option(..., "--plan-id", help="Calibration plan to select under."),
    summaries: Path = typer.Option(
        ..., "--summaries", help="Summaries file for the primary layer."
    ),
    fallback_summaries: Path | None = typer.Option(
        None,
        "--fallback-summaries",
        help="Summaries for the fallback layer. Only valid when the primary layer produced none.",
    ),
    output: Path | None = typer.Option(None, "--output", help="Write the decision to a file."),
) -> None:
    """Choose the ratio mechanically, or report the fallback status.

    Smallest passing ratio in preregistered order. Not the largest effect, not the most flips,
    and not whatever a forecaster does best on: any of those would choose the stimulus using the
    outcome. A passing primary layer prohibits the fallback.

    Loads no model and reads no prompt. This is a calibration decision, not a scientific result.
    """
    from .calibration.observations import ObservationLoadError, read_ratio_summaries
    from .calibration.plan import (
        CalibrationPlanError,
        build_decision_record,
        load_calibration_plan,
    )
    from .calibration.selection import SelectionError, select_calibration_ratio

    try:
        plan = load_calibration_plan(plan_id)
        primary = read_ratio_summaries(summaries)
        fallback = read_ratio_summaries(fallback_summaries) if fallback_summaries else None
        selection = select_calibration_ratio(
            primary_layer=plan.primary_layer,
            fallback_layer=plan.fallback_layer,
            expected_ratios=plan.norm_ratios,
            primary_summaries=primary,
            fallback_summaries=fallback,
        )
        decision = build_decision_record(plan, selection)
    except (CalibrationPlanError, ObservationLoadError, SelectionError) as error:
        typer.secho(f"calibration selection failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    payload = decision.model_dump(mode="json")
    payload["notes"] = (
        "Calibration decision. Chooses an intervention strength; measures nothing about the "
        "model and is not a scientific result."
    )
    _echo_json(payload)
    if output is not None:
        atomic_write_json(output, decision.model_dump(mode="json"))

    if decision.status.value == "failed_all_layers":
        typer.secho(
            "no ratio passed at either preregistered layer; the study stops under this design",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@directions_app.command("build-family")
def directions_build_family(
    config: Path = typer.Option(..., "--config", help="Path to a direction-family config."),
    force: bool = typer.Option(
        False, "--force", help="Replace an existing, different family at the same manifest path."
    ),
) -> None:
    """Construct the study's direction family from the pinned model's output embedding.

    Reads the unembedding rows for the answer tokens and nothing else. No prompt is run, no
    state is captured, and no intervention is applied.

    This is construction, not causal validation. The directions are stimuli with a recorded
    recipe, and their artifacts are marked `validated: false`. Nothing here licenses describing
    any of them as meaningful or bias-related.

    Rerunning with identical inputs leaves an identical manifest untouched. A different family
    at the same path is refused unless force is passed.
    """
    from .interventions.direction_family import DirectionFamilyError, build_family_command

    try:
        report = build_family_command(config, force=force)
    except DirectionFamilyError as error:
        typer.secho(f"direction family failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    _echo_json(report)

    if not report["artifact_verification"]["valid"]:
        typer.secho(
            f"the stored artifacts did not verify: {report['artifact_verification']['failures']}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@directions_app.command("verify-family")
def directions_verify_family(
    manifest_id: str = typer.Option(..., "--manifest-id", help="Direction family id to verify."),
    regenerate: bool = typer.Option(
        False,
        "--regenerate",
        help="Also rebuild every direction from the pinned model and compare. Loads the model.",
    ),
    config: Path | None = typer.Option(
        None, "--config", help="Config to regenerate from. Defaults to the one the manifest cites."
    ),
) -> None:
    """Verify a built direction family.

    Without `--regenerate` this loads no model: it checks the manifest's own content hash, the
    presence and content hash of every stored vector, dimensions, norms, orthogonality, and
    family completeness. With `--regenerate` it additionally rebuilds all eight directions from
    the pinned weights and compares them, writing nothing.
    """
    from .interventions.direction_family import DirectionFamilyError, verify_family_command

    try:
        report = verify_family_command(manifest_id, regenerate=regenerate, config_path=config)
    except DirectionFamilyError as error:
        typer.secho(f"direction family failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    _echo_json(report)

    if not report["valid"]:
        typer.secho(
            f"direction family {manifest_id} did not verify",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@state_audit_app.command("smoke")
def state_audit_smoke(
    config: Path = typer.Option(..., "--config", help="Path to a state-audit run config."),
    run_id: str = typer.Option(..., "--run-id", help="Run id to write artifacts under."),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite a documented partial or failed run at this id. Never a completed one.",
    ),
) -> None:
    """Run the eight-prompt engineering smoke on the pinned weights.

    Loads the model. One clean forward per prompt capturing the layer-13 residual stream, then
    all 17 candidates per prompt: 144 forwards in total. The intervention strength is one global
    alpha, computed as the preregistered ratio times the median clean state norm across the smoke
    prompts, and applied unchanged to every prompt and every signed direction.

    This is engineering validation. It carries `scientific_result: false`, the ratio and the
    layer were both fixed in advance, and its effect sizes select nothing. A completed run at the
    same id is refused rather than rewritten, because its artifacts are the only record of what
    happened.
    """
    from .state_audit.run import StateAuditRunError, run_smoke

    directory = ensure_run_dir(run_id)
    configure_logging("INFO", log_file=directory / RUN_LOG)

    try:
        report = run_smoke(config, run_id, force=force)
    except StateAuditRunError as error:
        typer.secho(f"state-audit smoke failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    _echo_json(report)

    if report["status"] != "complete":
        typer.secho(
            f"run {run_id} did not complete: {report['counts']['failures']} failures were "
            "recorded and the manifest is marked failed",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@state_audit_app.command("calibrate")
def state_audit_calibrate(
    config: Path = typer.Option(..., "--config", help="Path to a calibration run config."),
    run_id: str = typer.Option(..., "--run-id", help="Run id to write artifacts under."),
    primary_run_id: str | None = typer.Option(
        None,
        "--primary-run-id",
        help="The layer-13 run this falls back from. Required only for the fallback layer.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite a documented partial or failed run at this id. Never a completed one.",
    ),
) -> None:
    """Run the preregistered calibration sweep at one layer and record the decision.

    Loads the model. Captures the clean state for all 32 calibration prompts, takes the median as
    the reference norm before any intervention runs, derives one global alpha per frozen ratio,
    then applies 8 directions x 2 signs x 5 ratios plus one shared no-op to every prompt: 2,624
    forwards. It then evaluates the six preregistered conditions per ratio and takes the
    **smallest** passing one.

    Calibration chooses an intervention strength. It measures nothing about the model's abilities
    and is not a scientific result. Thresholds, ratios, prompts, target, and layers all come from
    the frozen plan and are not adjustable here. The layer-20 fallback is refused unless the
    layer-13 run it names recorded `fallback_required`.
    """
    from .state_audit.calibrate import run_calibration
    from .state_audit.run import StateAuditRunError

    directory = ensure_run_dir(run_id)
    configure_logging("INFO", log_file=directory / RUN_LOG)

    try:
        report = run_calibration(config, run_id, primary_run_id=primary_run_id, force=force)
    except StateAuditRunError as error:
        typer.secho(f"calibration failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    _echo_json(report)

    if report["status"] != "complete":
        typer.secho(
            f"run {run_id} did not complete: {report['counts']['failures']} failures were "
            "recorded and the manifest is marked failed",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    status = report["decision"]["status"]
    if status == "failed_all_layers":
        typer.secho(
            "no ratio passed at either preregistered layer; the study stops under this design. "
            "Write it up as a negative engineering result rather than widening the grid.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    if status == "fallback_required":
        typer.secho(
            "no ratio passed at the primary layer; the preregistered layer-20 fallback is now "
            f"open. Run it with --primary-run-id {run_id}.",
            fg=typer.colors.YELLOW,
            err=True,
        )


@state_audit_app.command("train")
def state_audit_train(
    config: Path = typer.Option(..., "--config", help="Path to a training run config."),
    run_id: str = typer.Option(..., "--run-id", help="Run id to write artifacts under."),
    force: bool = typer.Option(False, "--force", help="Overwrite a partial or failed run."),
) -> None:
    """Run the 96-prompt training stage at the calibrated strength.

    Loads the model. One clean forward per prompt plus all 17 candidates: 1,728 forwards. The
    intervention strength is **inherited** from the calibration decision rather than recomputed,
    so the predictors are fitted on the same stimulus the final test will be scored on.

    This produces outcomes for the training prompts, which a predictor is allowed to learn from.
    It touches no final-test prompt.
    """
    from .state_audit.run import StateAuditRunError, run_training

    directory = ensure_run_dir(run_id)
    configure_logging("INFO", log_file=directory / RUN_LOG)

    try:
        report = run_training(config, run_id, force=force)
    except StateAuditRunError as error:
        typer.secho(f"training run failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    _echo_json(report)
    if report["status"] != "complete":
        typer.secho(
            f"run {run_id} did not complete: {report['counts']['failures']} failures were "
            "recorded and the manifest is marked failed",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@state_audit_app.command("projection")
def state_audit_projection(
    config: Path = typer.Option(..., "--config", help="Path to a state-audit run config."),
    projection_id: str = typer.Option(..., "--projection-id", help="Projection id to build."),
    force: bool = typer.Option(False, "--force", help="Replace a different projection."),
) -> None:
    """Build the fixed 16-dimensional intervention projection. Loads no model weights.

    Generated once from the master seed, stored, hashed, and cited by every forecast. It is never
    fitted, so it has no training-boundary exposure and is identical for calibration, training,
    and final test. Every method receives the same `P^T v`.
    """
    from .state_audit.predict import direction_vectors
    from .state_audit.projection import (
        ProjectionError,
        build_projection,
        verify_projection,
        write_projection,
    )
    from .state_audit.run import StateAuditRunError, resolve_run_inputs

    try:
        inputs = resolve_run_inputs(config)
        vectors = direction_vectors(inputs.config.direction_family_id)
        record, matrix = build_projection(
            projection_id=projection_id,
            study_id=inputs.config.study_id,
            hidden_dim=inputs.family.hidden_dim,
            components=16,
            master_seed=inputs.config.master_seed,
            direction_vectors=vectors,
            direction_family_id=inputs.family.family_id,
            direction_family_hash=inputs.family.family_hash,
        )
        path, status = write_projection(record, matrix, force=force)
        verification = verify_projection(projection_id)
    except (ProjectionError, StateAuditRunError) as error:
        typer.secho(f"projection failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    _echo_json(
        {
            "status": status,
            "manifest_path": str(path),
            "projection_id": record.projection_id,
            "matrix_hash": record.matrix_hash,
            "hidden_dim": record.hidden_dim,
            "components": record.components,
            "orthonormality_error": record.orthonormality_error,
            "injectivity_margin": record.injectivity_margin,
            "realized_vector_count": record.realized_vector_count,
            "derived_seed": record.derived_seed,
            "verification": verification,
            "scientific_result": False,
        }
    )
    if not verification["valid"]:
        typer.secho(
            f"the projection did not verify: {verification['failures']}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@state_audit_app.command("final-test-clean")
def state_audit_final_test_clean(
    config: Path = typer.Option(..., "--config", help="Path to a final-test clean config."),
    run_id: str = typer.Option(..., "--run-id", help="Run id to write artifacts under."),
    force: bool = typer.Option(False, "--force", help="Overwrite a partial or failed run."),
) -> None:
    """Capture clean logits and states for the final-test prompts. Applies no intervention.

    Loads the model for one forward per prompt. No candidate set is built and no intervention
    hook is registered, so the run produces no outcome; that is what lets a forecast committed
    afterwards still be a forecast. The manifest's `intervention_count` is a typed literal zero.
    """
    from .state_audit.run import StateAuditRunError, execute_clean_only

    directory = ensure_run_dir(run_id)
    configure_logging("INFO", log_file=directory / RUN_LOG)

    try:
        report = execute_clean_only(config, run_id, force=force)
    except StateAuditRunError as error:
        typer.secho(f"final-test clean stage failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    _echo_json(report)
    if report["status"] != "complete":
        typer.secho(
            f"run {run_id} did not complete: {report['counts']['failures']} failures",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@state_audit_app.command("commit-forecasts")
def state_audit_commit_forecasts(
    training_config: Path = typer.Option(..., "--training-config", help="Training run config."),
    training_run_id: str = typer.Option(..., "--training-run-id", help="Completed training run."),
    final_test_config: Path = typer.Option(
        ..., "--final-test-config", help="Final-test clean config."
    ),
    final_test_run_id: str = typer.Option(
        ..., "--final-test-run-id", help="Completed final-test clean run."
    ),
    projection_id: str = typer.Option(..., "--projection-id", help="Intervention projection id."),
) -> None:
    """Fit the predictors and commit every final-test forecast. Loads no model.

    Fits the transforms and the three ridges on the 96 training prompts only, builds the
    wrong-state pairing and the ten derangements, then predicts all 17 candidates for each of the
    32 final-test prompts under all 16 method-and-condition combinations and commits 512 records.

    It refuses to run if any outcome artifact exists in the final-test run directory. That refusal
    is the blinding: correct-looking timestamps prove nothing on their own.
    """
    from .state_audit.predict import PredictError, commit_final_test_forecasts
    from .state_audit.run import StateAuditRunError

    directory = ensure_run_dir(final_test_run_id)
    configure_logging("INFO", log_file=directory / RUN_LOG)

    try:
        report = commit_final_test_forecasts(
            training_config,
            training_run_id,
            final_test_config,
            final_test_run_id,
            projection_id,
        )
    except (PredictError, StateAuditRunError) as error:
        typer.secho(f"commitment failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    _echo_json(report)
    if not report["verification"]["valid"]:
        typer.secho(
            f"the commitment checkpoint did not verify: {report['verification']['failures']}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@state_audit_app.command("resolve-final-test")
def state_audit_resolve_final_test(
    config: Path = typer.Option(..., "--config", help="Path to the final-test config."),
    run_id: str = typer.Option(..., "--run-id", help="The final-test run at the checkpoint."),
    layer: int = typer.Option(..., "--layer", help="Required layer. Checked, not trusted."),
    norm_ratio: float = typer.Option(..., "--norm-ratio", help="Required calibrated ratio."),
    global_alpha: float = typer.Option(..., "--global-alpha", help="Required calibrated alpha."),
    yes_i_understand_this_is_irreversible: bool = typer.Option(
        False,
        "--yes-i-understand-this-is-irreversible",
        help="Required. Resolving spends the blinding and cannot be undone.",
    ),
) -> None:
    """Apply all 17 candidates to every final-test prompt, then reveal and verify. IRREVERSIBLE.

    Loads the model for 544 intervened forwards: 32 prompts x (16 signed + 1 no-op). It runs **no
    clean forward**; the clean logits and states come from the clean stage, so every delta is
    measured against exactly the baseline the forecasts were made against.

    Every guard runs before the weights are touched. It refuses a dirty working tree, a setting
    that differs from the verified calibration decision, a commitment count other than the
    preregistered 512, any pre-existing reveal, and any pre-existing outcome artifact. The commit
    that performs it is recorded inside the hashed manifest.

    Once this completes, the study's forecasts have been checked against outcomes and that cannot
    be undone. The confirmation flag is required for that reason.
    """
    from .state_audit.resolve import FinalTestResolutionError, resolve_final_test

    if not yes_i_understand_this_is_irreversible:
        typer.secho(
            "refusing to resolve the final test without --yes-i-understand-this-is-irreversible. "
            "Applying these interventions spends the study's blinding permanently.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    directory = ensure_run_dir(run_id)
    configure_logging("INFO", log_file=directory / RUN_LOG)

    try:
        report = resolve_final_test(config, run_id, layer, norm_ratio, global_alpha)
    except FinalTestResolutionError as error:
        typer.secho(f"final-test resolution refused: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    _echo_json(report)

    if not report["commitments_verified"]:
        typer.secho(
            "the commitments did not verify. This is recorded as evidence and must not be "
            "re-resolved away; report the run as a run that did not verify.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    if report["status"] != "complete":
        typer.secho(
            f"run {run_id} did not complete: {report['counts']['failures']} failures were recorded",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@state_audit_app.command("analyze-final-test")
def state_audit_analyze_final_test(
    run_id: str = typer.Option(..., "--run-id", help="The resolved final-test run."),
    training_run_id: str = typer.Option(
        ..., "--training-run-id", help="The training run the predictors were fitted on."
    ),
    force: bool = typer.Option(
        False, "--force", help="Rerun a completed analysis. Only to repair a documented bug."
    ),
) -> None:
    """Score the sealed forecasts and run the preregistered analysis. Loads no model.

    Nothing is fitted, refitted, tuned, or dropped. Absolute errors are averaged within each prompt
    first, then across the 32 prompts; the two primary paired differences are bootstrapped with
    10,000 paired resamples over prompt groups from the frozen seed; and the decision rule is
    applied mechanically. An interval crossing zero is reported as no detected difference.

    Writes machine-readable tables and two minimal figures. Runs once by default: rerunning after
    seeing the numbers is how a decision rule gets renegotiated.
    """
    from .state_audit.analyze import AnalysisError, analyze_final_test

    try:
        report = analyze_final_test(run_id, training_run_id, force=force)
    except AnalysisError as error:
        typer.secho(f"analysis failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    _echo_json(report)
    for comparison in report["primary_comparisons"]:
        colour = typer.colors.GREEN if comparison["supported"] else typer.colors.YELLOW
        typer.secho(f"{comparison['name']}: {comparison['interpretation']}", fg=colour, err=True)
    if not report["scientific_forecast_evaluation"]:
        typer.secho(
            "this analysis is not a scientific forecast evaluation: the commitments did not verify "
            "or the resolution did not complete",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@state_audit_app.command("replay-analysis")
def state_audit_replay_analysis(
    run_id: str = typer.Option(..., "--run-id", help="The analyzed final-test run."),
    output: Path | None = typer.Option(None, "--output", help="Write the report to a file."),
) -> None:
    """Recompute the analysis from artifacts and check it against the stored record.

    Loads no model. This is the check a third party runs: every method summary and both primary
    comparisons are recomputed from the forecasts and outcomes on disk, so a stored analysis that
    does not follow from its own inputs is detectable without any weights.
    """
    from .state_audit.analyze import AnalysisError, replay_analysis

    try:
        report = replay_analysis(run_id)
    except AnalysisError as error:
        typer.secho(f"replay failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    _echo_json(report)
    if output is not None:
        atomic_write_json(output, report)
    if not report["valid"]:
        typer.secho(
            f"the stored analysis does not recompute: {report['failures']}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@state_audit_app.command("publish-bundle")
def state_audit_publish_bundle(
    final_test_run_id: str = typer.Option(
        ..., "--final-test-run-id", help="The resolved and analyzed final-test run."
    ),
    training_run_id: str = typer.Option(
        ..., "--training-run-id", help="The training run holding the fit provenance."
    ),
    calibration_run_id: str = typer.Option(
        ..., "--calibration-run-id", help="The calibration run holding the strength decision."
    ),
    bundle_id: str = typer.Option(BUNDLE_ID, "--bundle-id", help="Bundle directory name."),
    output: Path | None = typer.Option(None, "--output", help="Write the report to a file."),
) -> None:
    """Copy the allowlisted artifacts into a public replay bundle. Loads no model.

    Publishes the analysis, resolution manifest, score tables, sealed forecasts, commitments,
    reveals, calibration decision, fit provenance, and figures, with a checksum manifest. Model
    weights, residual-stream arrays, and the pre-reveal salt files are never published; the
    post-reveal salts travel inside the reveals, because a commitment hash cannot be checked
    without its salt.
    """
    from .state_audit.bundle import BundleError, build_public_bundle

    try:
        report = build_public_bundle(
            {
                "final_test": final_test_run_id,
                "training": training_run_id,
                "calibration": calibration_run_id,
            },
            bundle_id=bundle_id,
        )
    except BundleError as error:
        typer.secho(f"could not build the bundle: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    _echo_json(report)
    if output is not None:
        atomic_write_json(output, report)


@state_audit_app.command("replay-bundle")
def state_audit_replay_bundle(
    bundle_path: Path = typer.Option(..., "--bundle", help="Path to the public bundle directory."),
    output: Path | None = typer.Option(None, "--output", help="Write the report to a file."),
) -> None:
    """Verify a public bundle's checksums and replay its analysis. Loads no model.

    The check a third party runs against the published bundle alone: nothing outside the bundle is
    read, and the recomputation goes through the same function the run directory uses.
    """
    from .state_audit.bundle import BundleError, replay_bundle

    try:
        report = replay_bundle(bundle_path)
    except BundleError as error:
        typer.secho(f"bundle replay failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    _echo_json(report)
    if output is not None:
        atomic_write_json(output, report)
    if not report["valid"]:
        typer.secho(
            f"the published bundle does not verify: {report['failures']}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@state_audit_app.command("verify-commitments")
def state_audit_verify_commitments(
    run_id: str = typer.Option(..., "--run-id", help="Final-test run id to verify."),
    output: Path | None = typer.Option(None, "--output", help="Write the report to a file."),
) -> None:
    """Verify the commitment checkpoint from artifacts. Loads no model.

    Checks the key structure, that every commitment has a forecast and its own salt, that no
    interval is inverted, and that no final-test outcome artifact exists. No reveal is expected
    here and none should exist: the salts stay sealed until the interventions are resolved.
    """
    from .state_audit.predict import verify_final_test_commitments

    report = verify_final_test_commitments(run_id)
    _echo_json(report)
    if output is not None:
        atomic_write_json(output, report)
    if not report["valid"]:
        typer.secho(
            f"run {run_id} did not verify: {report['failures']}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)


@state_audit_app.command("verify-run")
def state_audit_verify_run(
    run_id: str = typer.Option(..., "--run-id", help="State-audit run id to verify."),
    compare_run_id: str | None = typer.Option(
        None,
        "--compare-run-id",
        help="Second run of the same inputs. Compares every target to measure determinism.",
    ),
    tolerance: float = typer.Option(
        0.0, "--tolerance", help="Allowed absolute target difference when comparing two runs."
    ),
    output: Path | None = typer.Option(None, "--output", help="Write the report to a file."),
) -> None:
    """Verify a state-audit run from its artifacts. Loads no model.

    Recomputes the manifest's own content hash, every artifact hash, every observation's target
    from its own logits, the reference norm from the recorded clean state norms, and the single
    global alpha across every non-no-op observation. With `--compare-run-id` it also compares two
    runs of the same inputs row by row, which is how cross-process determinism is measured.
    """
    from .state_audit.run import StateAuditRunError
    from .state_audit.verify import compare_runs, verify_run

    try:
        report = verify_run(run_id)
        if compare_run_id is not None:
            report["determinism"] = compare_runs(run_id, compare_run_id, tolerance=tolerance)
    except StateAuditRunError as error:
        typer.secho(f"state-audit verification failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    _echo_json(report)
    if output is not None:
        atomic_write_json(output, report)

    determinism = report.get("determinism")
    if not report["valid"]:
        typer.secho(
            f"run {run_id} did not verify: {report['failures']}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)
    if determinism is not None and not determinism["deterministic"]:
        typer.secho(
            f"runs {run_id} and {compare_run_id} do not agree: {determinism['differences']}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@interventions_app.command("validate")
def interventions_validate(
    config: Path = typer.Option(..., "--config", help="Path to an experiment config."),
) -> None:
    """Run the intervention harness controls.

    Exits nonzero if any required control fails, because a failure means every number the
    harness would go on to produce is untrustworthy.
    """
    from .interventions.validate import validate_interventions

    resolved = resolve_experiment(config)
    report = validate_interventions(resolved)
    _echo_json(report)

    if not report["passed"]:
        typer.secho(
            f"required intervention controls failed: {report['required_failed']}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@app.command()
def benchmark(
    model_config: Path = typer.Option(
        ..., "--model-config", help="Path to a model config. Its pinned revision is used as is."
    ),
    task_config: Path = typer.Option(..., "--task-config", help="Path to a prepared task config."),
    split: str = typer.Option("test", "--split", help="Which split to draw prompts from."),
    max_items: int = typer.Option(1, "--max-items", min=1, help="How many prompts to score."),
    warmup_runs: int = typer.Option(1, "--warmup-runs", min=0, help="Untimed forwards first."),
    timed_runs: int = typer.Option(3, "--timed-runs", min=1, help="Timed forwards for latency."),
    capture_layer: int | None = typer.Option(
        None, "--capture-layer", help="Layer to verify capture at. Defaults to the middle layer."
    ),
    seed: int = typer.Option(12345, "--seed", help="Seed for deterministic prompt selection."),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Where to write artifacts. Defaults to a run directory."
    ),
    offline: bool = typer.Option(
        False, "--offline", help="Require locally cached weights and never reach the network."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite a non-empty output directory."),
) -> None:
    """Measure whether a real model runs here, how fast, and whether capture works.

    This is a systems and clean-model sanity benchmark, not a CSF-Bench result. It loads the
    pinned weights, scores a few multiple-choice items from their answer-token logits, times
    forward passes on this machine, and verifies that the hook-owned capture path reads back
    the point it intervened on.

    Nothing it writes is a scientific finding, and the artifacts are typed so they cannot be
    mistaken for one.
    """
    from .benchmark import BenchmarkError, run_benchmark
    from .models.scoring import LabelTokenError

    model_settings = load_config(model_config, ModelConfig)
    task_settings = load_config(task_config, TaskConfig)

    try:
        resolved_split = Split(split)
    except ValueError as error:
        raise typer.BadParameter(
            f"unknown split {split!r}; valid splits are {[item.value for item in Split]}"
        ) from error

    try:
        report = run_benchmark(
            model_config=model_settings,
            task_config=task_settings,
            model_config_path=model_config,
            task_config_path=task_config,
            split=resolved_split,
            max_items=max_items,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
            capture_layer=capture_layer,
            seed=seed,
            output_dir=output_dir,
            offline=offline,
            force=force,
        )
    except BenchmarkError as error:
        # The status is the whole point of the message: it says which of the several very
        # different problems this is, so the reader knows what to go and fix.
        typer.secho(
            f"benchmark failed [{error.status.value}]: {error}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1) from error
    except LabelTokenError as error:
        typer.secho(
            f"benchmark failed [answer_labels_unscoreable]: {error}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1) from error

    _echo_json(report)

    if not report["capture"]["capture_point_verified"]:
        typer.secho(
            "capture point verification failed: the layer read back does not match the layer "
            "patched, so intervention results from this model would not mean what they claim",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@trials_app.command("generate")
def trials_generate(
    config: Path = typer.Option(..., "--config", help="Path to an experiment config."),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Run id. Defaults to a timestamped id."
    ),
    max_trials: int | None = typer.Option(None, "--max-trials", help="Limit the number of trials."),
    seed: int | None = typer.Option(None, "--seed", help="Override the config seed."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report the plan without loading a model."
    ),
) -> None:
    """Generate trials: clean outputs, captured states, and candidate sets."""
    from .trials.generate import generate_trials

    resolved = resolve_experiment(config)
    if seed is not None:
        resolved = resolved.model_copy(
            update={"experiment": resolved.experiment.model_copy(update={"seed": seed})}
        )

    if dry_run:
        _echo_json(generate_trials(resolved, run_id=None, max_trials=max_trials, dry_run=True))
        return

    identifier = run_id or new_run_id(resolved.experiment.name)
    directory = ensure_run_dir(identifier)
    configure_logging("INFO", log_file=directory / RUN_LOG)
    info("starting run", run_id=identifier, config=str(config))

    _echo_json(generate_trials(resolved, run_id=identifier, max_trials=max_trials))


@trials_app.command("resolve")
def trials_resolve(
    run_id: str = typer.Option(..., "--run-id", help="Generated trial run to resolve."),
    selection_seed_file: Path | None = typer.Option(
        None,
        "--selection-seed-file",
        help="Read the selection seed from this file. Generated after commitment if omitted.",
    ),
    ground_truth: bool = typer.Option(
        False,
        "--ground-truth",
        help="Apply every candidate and record observations without a forecaster or scoring.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-resolve even if observations already exist."
    ),
) -> None:
    """Apply interventions and record observations for a generated trial run.

    Forecast mode (default) requires committed forecasts, selects one candidate per trial
    after commitment, applies it, and reveals and verifies the commitment. Ground-truth mode
    applies every candidate and records observations only, which is what trains baselines and
    validates the harness on real weights. Neither produces a scientific result.
    """
    from .trials.resolve import ResolutionError, resolve_run

    directory = run_dir(run_id)
    if not directory.exists():
        typer.secho(f"no such run: {directory}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    configure_logging("INFO", log_file=directory / RUN_LOG)

    try:
        report = resolve_run(
            run_id,
            selection_seed_file=selection_seed_file,
            ground_truth=ground_truth,
            force=force,
        )
    except ResolutionError as error:
        typer.secho(f"resolution failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    _echo_json(report)

    if report["counts"]["failures"]:
        typer.secho(
            f"{report['counts']['failures']} interventions failed and were recorded in "
            "resolution_failures.jsonl",
            fg=typer.colors.YELLOW,
            err=True,
        )


@score_app.command("run")
def score_run_command(
    run_id: str = typer.Option(..., "--run-id", help="Resolved run to score."),
    force: bool = typer.Option(False, "--force", help="Rescore even if scores already exist."),
) -> None:
    """Score committed forecasts in a resolved run against its observations.

    Headline metrics exclude no-op candidates; the no-op-inclusive numbers are reported
    separately. Every metric carries its sample count and a group-bootstrapped interval.
    Scores are written to the run directory and are never a public result.
    """
    from .scoring import ScoringError, score_run

    try:
        report = score_run(run_id, force=force)
    except ScoringError as error:
        typer.secho(f"scoring failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error

    _echo_json(report)


@verify_app.command("run")
def verify_run(
    run_id: str = typer.Option(..., "--run-id", help="Run id to verify."),
    output: Path | None = typer.Option(None, "--output", help="Write the report to a file."),
) -> None:
    """Recompute every commitment in a run from its revealed salt.

    This is the check a third party runs to confirm that the forecasts on disk are the ones
    that were committed to before the interventions were selected.
    """
    from .trials.commitment import verify_run_commitments

    directory = run_dir(run_id)
    if not directory.exists():
        typer.secho(f"no such run: {directory}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    report = verify_run_commitments(run_id)
    _echo_json(report)
    if output is not None:
        atomic_write_json(output, report)

    if not report["verified"]:
        typer.secho(
            "run did not verify; it must not be exported to the dashboard",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    sys.exit(app())
