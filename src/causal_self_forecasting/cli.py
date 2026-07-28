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

app = typer.Typer(
    name="csf",
    help="CSF-Bench: precommitted forecasting of blinded internal interventions.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Prepare task datasets.", no_args_is_help=True)
directions_app = typer.Typer(
    help="Create and inspect intervention directions.", no_args_is_help=True
)
interventions_app = typer.Typer(help="Validate the intervention harness.", no_args_is_help=True)
trials_app = typer.Typer(help="Generate and resolve trials.", no_args_is_help=True)
score_app = typer.Typer(help="Score resolved runs.", no_args_is_help=True)
verify_app = typer.Typer(help="Verify run artifacts.", no_args_is_help=True)

app.add_typer(data_app, name="data")
app.add_typer(directions_app, name="directions")
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

    config_types: list[tuple[str, type]] = [
        ("configs/models", ModelConfig),
        ("configs/tasks", TaskConfig),
        ("configs/experiments", ExperimentConfig),
    ]
    from .config import InterventionConfig

    config_types.append(("configs/interventions", InterventionConfig))

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
