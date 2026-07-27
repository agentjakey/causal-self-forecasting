"""Integration tests for trial resolution and the full generate-to-score pipeline.

The guard tests need no model: resolution refuses benchmark runs and validates commitments
before it ever loads weights. The rest run the real fixture model through a workspace that
redirects every directory into a temporary tree, so a complete pipeline (generate, resolve,
fit baselines, commit, resolve, score) runs offline and leaves nothing behind.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from causal_self_forecasting import paths
from causal_self_forecasting.config import ResolvedExperiment, resolve_experiment
from causal_self_forecasting.hashing import atomic_write_json, write_jsonl
from causal_self_forecasting.interventions.directions import DirectionStore
from causal_self_forecasting.paths import RUN_MANIFEST, run_dir
from causal_self_forecasting.schemas import Framing, PromptVariant, Split, TaskItem
from causal_self_forecasting.tasks import loader as task_loader
from causal_self_forecasting.trials import generate as generate_mod
from causal_self_forecasting.trials import resolve as resolve_mod
from causal_self_forecasting.trials.resolve import ResolutionError, read_observations, resolve_run

_WORDS = ["water", "energy", "rock", "heat", "light", "gas", "cell", "force", "sun", "earth"]


@dataclass
class Workspace:
    tmp: Path
    directions: Path
    experiment_path: Path
    direction_id: str

    def generate(self, run_id: str, max_trials: int = 2) -> dict[str, Any]:
        resolved = self.resolved()
        return generate_mod.generate_trials(resolved, run_id=run_id, max_trials=max_trials)

    def resolved(self) -> ResolvedExperiment:
        return resolve_experiment(self.experiment_path)


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Workspace:
    processed = tmp_path / "processed"
    manifests = tmp_path / "manifests"
    directions = tmp_path / "directions"
    runs = tmp_path / "runs"
    for directory in (processed, manifests, directions, runs):
        directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(task_loader, "processed_dir", lambda: processed)
    monkeypatch.setattr(task_loader, "manifests_dir", lambda: manifests)
    monkeypatch.setattr(generate_mod, "directions_dir", lambda: directions)
    monkeypatch.setattr(resolve_mod, "directions_dir", lambda: directions)
    monkeypatch.setattr(paths, "runs_dir", lambda: runs)

    task_name = "wstask"
    _write_prepared_task(processed, task_name)

    experiment_path = _write_configs(tmp_path, task_name)

    # A synthetic direction sized for the 4-block, 64-dim fixture model.
    torch.manual_seed(1234)
    vector = torch.randn(64)
    DirectionStore(directions).save(
        "ws_direction",
        vector / torch.linalg.vector_norm(vector),
        {"method": "synthetic_test", "validated": False},
    )

    return Workspace(tmp_path, directions, experiment_path, "ws_direction")


def _write_prepared_task(processed: Path, task_name: str) -> None:
    items: list[TaskItem] = []
    variants: list[PromptVariant] = []
    for index in range(6):
        item = TaskItem(
            item_id=f"q{index}",
            group_id=f"q{index}",
            source="allenai/ai2_arc",
            source_id=f"src{index}",
            subject="ARC-Challenge",
            question=" ".join(_WORDS[: 3 + index % 3]),
            choices=["water", "energy", "light", "heat"],
            answer_index=index % 4,
            split=Split.TEST if index % 2 == 0 else Split.TRAIN,
        )
        items.append(item)
        variant = PromptVariant(
            variant_id=f"{item.item_id}.neutral_a",
            item_id=item.item_id,
            group_id=item.group_id,
            wrapper_id="neutral_a",
            framing=Framing.NEUTRAL,
            prompt_text=f"answer the question {' '.join(_WORDS[:4])} A B C D",
            answer_labels=["A", "B", "C", "D"],
            split=item.split,
        )
        variants.append(variant)
    write_jsonl(processed / task_name / "items.jsonl", items)
    write_jsonl(processed / task_name / "variants.jsonl", variants)


def _write_configs(tmp_path: Path, task_name: str) -> Path:
    model_path = tmp_path / "model.yaml"
    task_path = tmp_path / "task.yaml"
    intervention_path = tmp_path / "residual.yaml"
    experiment_path = tmp_path / "experiment.yaml"

    model_path.write_text(
        "name: fixture_tiny\nkind: fixture\nmodel_id: tiny_llama_ws\nrevision: fixture-v1\n"
        "dtype: float32\ndevice: cpu\nfixture_hidden_size: 64\nfixture_num_layers: 4\n",
        encoding="utf-8",
    )
    wrappers = "".join(
        f"  - wrapper_id: {wid}\n    framing: {framing}\n"
        f"    template: |-\n      {{question}}\n      {{choices}}\n      Answer:\n"
        for wid, framing in (
            ("neutral_a", "neutral"),
            ("eval_a", "evaluation"),
            ("deploy_a", "deployment"),
        )
    )
    task_path.write_text(
        f"name: {task_name}\nsource: allenai/ai2_arc\nsource_config: ARC-Challenge\n"
        f"answer_labels: [A, B, C, D]\nlabel_prefix: ' '\nwrappers:\n{wrappers}",
        encoding="utf-8",
    )
    intervention_path.write_text(
        "name: ws_residual\nmechanism: residual_add\nmechanism_version: '1.0'\n"
        "layers: [2]\npositions: [-1]\nstrengths: [1.0]\n",
        encoding="utf-8",
    )
    experiment_path.write_text(
        "name: ws_experiment\nseed: 12345\n"
        f"model_ref: {model_path.as_posix()}\n"
        f"task_ref: {task_path.as_posix()}\n"
        f"intervention_refs: [{intervention_path.as_posix()}]\n"
        "capture_layers: [2]\ncapture_position: -1\ndirection_id: ws_direction\n"
        "random_control_count: 2\ncandidates_per_trial: 4\n"
        "trial_splits: [train, test]\nrerun_tolerance: 1.0e-4\n",
        encoding="utf-8",
    )
    return experiment_path


# ---------------------------------------------------------------------------
# Guards that need no model
# ---------------------------------------------------------------------------


def test_resolve_refuses_a_benchmark_run(isolated_runs: Path) -> None:
    directory = run_dir("bench-run")
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        directory / RUN_MANIFEST,
        {"run_id": "bench-run", "phase": "systems_benchmark", "config_path": "x"},
    )
    with pytest.raises(ResolutionError, match="systems benchmark"):
        resolve_run("bench-run")


def test_resolve_refuses_a_run_without_a_manifest(isolated_runs: Path) -> None:
    run_dir("empty-run").mkdir(parents=True, exist_ok=True)
    with pytest.raises(ResolutionError, match="no run manifest"):
        resolve_run("empty-run")


def test_validate_commitments_rejects_duplicates(isolated_runs: Path, monkeypatch) -> None:
    from causal_self_forecasting.schemas import ForecastCommitment
    from causal_self_forecasting.trials import resolve as resolve_module

    commitment = ForecastCommitment(
        trial_id="trial_00000",
        method_id="constant",
        commitment_hash="sha256:" + "0" * 64,
        forecast_ref="forecasts.jsonl",
    )
    monkeypatch.setattr(
        resolve_module, "_read_commitments", lambda run_id: [commitment, commitment]
    )
    monkeypatch.setattr(resolve_module, "read_forecasts", lambda run_id: [])

    class _Trial:
        trial_id = "trial_00000"

    with pytest.raises(ResolutionError, match="duplicate commitment"):
        resolve_module._validate_commitments("x", [_Trial()])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Ground-truth resolution
# ---------------------------------------------------------------------------


def test_ground_truth_resolves_all_candidates(workspace: Workspace) -> None:
    workspace.generate("gt", max_trials=2)
    report = resolve_run("gt", ground_truth=True)
    assert report["classification"] == "fixture_resolution"
    assert report["fixture_only"] is True
    assert report["scientific_result"] is False
    # Two trials, four candidates each.
    assert report["counts"]["observations"] == 8
    assert report["counts"]["failures"] == 0


def test_ground_truth_noop_reproduces_the_clean_output(workspace: Workspace) -> None:
    workspace.generate("gt", max_trials=2)
    resolve_run("gt", ground_truth=True)
    observations = read_observations("gt")
    noops = [obs for obs in observations if obs.mechanism.value == "noop"]
    assert noops
    for observation in noops:
        assert observation.delta_margin == pytest.approx(0.0, abs=1e-4)
        assert observation.answer_flip is False


def test_resolution_refuses_output_collision(workspace: Workspace) -> None:
    workspace.generate("gt", max_trials=2)
    resolve_run("gt", ground_truth=True)
    with pytest.raises(ResolutionError, match="--force"):
        resolve_run("gt", ground_truth=True)
    assert resolve_run("gt", ground_truth=True, force=True)["counts"]["observations"] == 8


def test_failed_intervention_is_preserved_not_dropped(workspace: Workspace) -> None:
    """A missing direction makes one candidate fail; it must be recorded, not omitted."""
    workspace.generate("gt", max_trials=2)
    # Remove a random-control direction so its candidates cannot be applied.
    controls = list(workspace.directions.glob("ws_direction__random_*.npz"))
    assert controls
    controls[0].unlink()

    report = resolve_run("gt", ground_truth=True)
    assert report["counts"]["failures"] >= 1
    failures_file = run_dir("gt") / "resolution_failures.jsonl"
    assert failures_file.exists()
    assert report["counts"]["observations"] + report["counts"]["failures"] == 8


# ---------------------------------------------------------------------------
# Forecast mode
# ---------------------------------------------------------------------------


def test_forecast_mode_requires_committed_forecasts(workspace: Workspace) -> None:
    workspace.generate("fc", max_trials=2)
    with pytest.raises(ResolutionError, match="no committed forecasts"):
        resolve_run("fc")


def test_forecast_mode_selects_reveals_and_verifies(workspace: Workspace) -> None:
    from causal_self_forecasting.forecasting import (
        ConstantBaseline,
        build_training_examples,
        commit_forecasts,
    )
    from causal_self_forecasting.trials.commitment import verify_run_commitments

    workspace.generate("train", max_trials=4)
    resolve_run("train", ground_truth=True)

    workspace.generate("fc", max_trials=2)
    baseline = ConstantBaseline()
    baseline.fit(build_training_examples("train", splits=("train", "test")))
    commit_forecasts("fc", baseline)

    report = resolve_run("fc")
    # One observation per trial in forecast mode: the selected candidate only.
    assert report["counts"]["observations"] == 2
    assert report["counts"]["reveals_verified"] == 2
    assert report["commitments_verified"] is True

    verification = verify_run_commitments("fc")
    assert verification["verified"] is True


def test_forecast_mode_selection_is_deterministic(workspace: Workspace) -> None:
    """Re-resolving the same run selects the same candidates.

    The selection seed is written once, after commitment, and persists, so a forced
    re-resolution reads the same seed and reselects identically. This is the property the
    protocol depends on: the selection cannot be re-rolled until a convenient candidate comes
    up.
    """
    from causal_self_forecasting.forecasting import (
        ConstantBaseline,
        build_training_examples,
        commit_forecasts,
    )

    workspace.generate("train", max_trials=4)
    resolve_run("train", ground_truth=True)
    examples = build_training_examples("train", splits=("train", "test"))

    workspace.generate("fc", max_trials=2)
    baseline = ConstantBaseline()
    baseline.fit(examples)
    commit_forecasts("fc", baseline)

    resolve_run("fc")
    first = sorted((obs.trial_id, obs.intervention_id) for obs in read_observations("fc"))
    resolve_run("fc", force=True)
    second = sorted((obs.trial_id, obs.intervention_id) for obs in read_observations("fc"))
    assert first == second


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def test_full_pipeline_generate_resolve_fit_commit_score(workspace: Workspace) -> None:
    from causal_self_forecasting.forecasting import (
        ConstantBaseline,
        PromptLexicalBaseline,
        build_training_examples,
        commit_forecasts,
    )
    from causal_self_forecasting.scoring import score_run

    workspace.generate("train", max_trials=4)
    resolve_run("train", ground_truth=True)
    examples = build_training_examples("train", splits=("train", "test"))
    assert examples

    workspace.generate("test", max_trials=2)
    for baseline in (ConstantBaseline(), PromptLexicalBaseline()):
        baseline.fit(examples)
        commit_forecasts("test", baseline)

    resolve_run("test", ground_truth=True)
    report = score_run("test")

    assert set(report["methods"]) == {"constant", "prompt_lexical"}
    assert report["scored_as_scientific"] is False
    assert report["fixture_only"] is True
    for method in report["methods"].values():
        assert method["headline_excludes_noop"]["n"] >= 1
        assert method["all_candidates_including_noop"]["n"] > method["headline_excludes_noop"]["n"]


def test_build_training_examples_isolates_the_requested_split(workspace: Workspace) -> None:
    """A forecaster must never train on outcomes from the split it will be scored on."""
    from causal_self_forecasting.forecasting import build_training_examples

    workspace.generate("run", max_trials=6)
    resolve_run("run", ground_truth=True)

    train_only = build_training_examples("run", splits=("train",))
    assert train_only
    assert all(example.split == "train" for example in train_only)

    test_only = build_training_examples("run", splits=("test",))
    assert all(example.split == "test" for example in test_only)

    # A group belongs entirely to one split, so the train and test example groups are disjoint.
    train_groups = {example.group_id for example in train_only}
    test_groups = {example.group_id for example in test_only}
    assert train_groups.isdisjoint(test_groups)
