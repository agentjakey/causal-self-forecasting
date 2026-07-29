"""End-to-end tests for the state-dependence execution path.

These run the whole pipeline against the local fixture model: a genuinely randomly initialized
Llama with 14 blocks and 64 hidden units, built on demand from a fixed seed. Real forward
passes, real hooks, real interventions, no network and no gated weights.

The fixture has no knowledge, so its logits are noise. Nothing measured here is a finding about
language models; these tests check that the plumbing does what the artifacts claim it did.

Every directory is redirected into tmp_path, so the suite can neither read nor replace the
study's real prompt manifest, direction family, calibration plan, or run artifacts.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from causal_self_forecasting import paths
from causal_self_forecasting.calibration import plan as plan_module
from causal_self_forecasting.cli import app
from causal_self_forecasting.hashing import hash_file, read_jsonl, write_jsonl
from causal_self_forecasting.interventions import direction_family as df
from causal_self_forecasting.paths import (
    STATE_AUDIT_CANDIDATE_SETS,
    STATE_AUDIT_CLEAN_PASS,
    STATE_AUDIT_FAILURES,
    STATE_AUDIT_OBSERVATIONS,
    STATE_AUDIT_RUN_MANIFEST,
    STATE_AUDIT_STATE_REFS,
    STATE_AUDIT_STATES,
    run_dir,
)
from causal_self_forecasting.schemas import Framing, PromptVariant, Split, TaskItem
from causal_self_forecasting.state_audit import run as run_module
from causal_self_forecasting.state_audit import verify as verify_module
from causal_self_forecasting.tasks import loader as task_loader
from causal_self_forecasting.tasks import prompt_manifest as pm

runner = CliRunner()

TASK_NAME = "satask"
MANIFEST_ID = "sa_manifest_v1"
FAMILY_ID = "sa_family_v1"
PLAN_ID = "sa_plan_v1"
RUN_ID = "sa-smoke"

HIDDEN = 64
LAYERS = 14
LAYER = 13

# Fixture-vocabulary words, so the word-level fixture tokenizer produces real tokens rather than
# a prompt made entirely of unknowns.
_WORDS = ["water", "energy", "rock", "heat", "light", "gas", "cell", "force", "sun", "earth"]


def _text(result) -> str:
    parts = [result.stdout or ""]
    with contextlib.suppress(ValueError):
        parts.append(result.stderr or "")
    return "".join(parts)


@dataclass
class Workspace:
    tmp: Path
    runs: Path
    directions: Path
    run_config: Path

    def smoke(self, run_id: str = RUN_ID, extra: list[str] | None = None):
        return runner.invoke(
            app,
            [
                "state-audit",
                "smoke",
                "--config",
                str(self.run_config),
                "--run-id",
                run_id,
                *(extra or []),
            ],
        )

    def verify(self, run_id: str = RUN_ID, extra: list[str] | None = None):
        return runner.invoke(app, ["state-audit", "verify-run", "--run-id", run_id, *(extra or [])])


def _write_prepared_task(processed: Path, manifests: Path) -> None:
    items: list[TaskItem] = []
    variants: list[PromptVariant] = []
    for index in range(12):
        item = TaskItem(
            item_id=f"sa{index:02d}",
            group_id=f"sa{index:02d}",
            source="allenai/ai2_arc",
            source_id=f"src{index}",
            subject="ARC-Challenge",
            question=" ".join(_WORDS[: 3 + index % 4]),
            choices=["water", "energy", "light", "heat"],
            answer_index=index % 4,
            split=Split.TEST if index % 2 == 0 else Split.TRAIN,
        )
        items.append(item)
        variants.append(
            PromptVariant(
                variant_id=f"{item.item_id}.neutral_a",
                item_id=item.item_id,
                group_id=item.group_id,
                wrapper_id="neutral_a",
                framing=Framing.NEUTRAL,
                prompt_text=(
                    "answer the question using one letter "
                    f"{item.question} A water B energy C light D heat Answer:"
                ),
                answer_labels=["A", "B", "C", "D"],
                split=item.split,
            )
        )

    items_path = write_jsonl(processed / TASK_NAME / "items.jsonl", items)
    variants_path = write_jsonl(processed / TASK_NAME / "variants.jsonl", variants)

    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / f"{TASK_NAME}.json").write_text(
        json.dumps(
            {
                "task_name": TASK_NAME,
                "config_path": "fixture",
                "config_hash": "sha256:" + "7" * 64,
                "source": "allenai/ai2_arc",
                "source_config": "ARC-Challenge",
                "source_split": "train",
                "item_count": len(items),
                "variant_count": len(variants),
                "group_count": len({item.group_id for item in items}),
                "item_splits": {"train": 6, "test": 6},
                "variant_splits": {"train": 6, "test": 6},
                "wrapper_ids": ["neutral_a", "eval_a", "deploy_a"],
                "heldout_wrapper_ids": [],
                "files": {
                    "items": {"path": "items.jsonl", "hash": hash_file(items_path)},
                    "variants": {"path": "variants.jsonl", "hash": hash_file(variants_path)},
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_configs(tmp_path: Path) -> dict[str, Path]:
    model = tmp_path / "model.yaml"
    task = tmp_path / "task.yaml"
    prompts = tmp_path / "prompts.yaml"
    family = tmp_path / "family.yaml"
    plan = tmp_path / "plan.yaml"
    run_config = tmp_path / "smoke.yaml"

    model.write_text(
        json.dumps(
            {
                "name": "fixture_state_audit",
                "kind": "fixture",
                "model_id": "tiny_llama_sa14",
                "revision": "fixture-v1",
                "dtype": "float32",
                "device": "cpu",
                "fixture_hidden_size": HIDDEN,
                "fixture_num_layers": LAYERS,
            }
        ),
        encoding="utf-8",
    )
    task.write_text(
        json.dumps(
            {
                "name": TASK_NAME,
                "source": "allenai/ai2_arc",
                "source_config": "ARC-Challenge",
                "answer_labels": ["A", "B", "C", "D"],
                "label_prefix": " ",
                "wrappers": [
                    {
                        "wrapper_id": "neutral_a",
                        "framing": "neutral",
                        "template": "{question}\n{choices}\nAnswer:",
                    },
                    {
                        "wrapper_id": "eval_a",
                        "framing": "evaluation",
                        "template": "{question}\n{choices}\nAnswer:",
                    },
                    {
                        "wrapper_id": "deploy_a",
                        "framing": "deployment",
                        "template": "{question}\n{choices}\nAnswer:",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    prompts.write_text(
        json.dumps(
            {
                "name": "sa_prompts",
                "manifest_id": MANIFEST_ID,
                "task_ref": task.as_posix(),
                "canonical_wrapper_id": "neutral_a",
                "master_seed": 20260727,
                "role_counts": {"smoke": 8, "calibration": 1, "training": 1, "final_test": 1},
            }
        ),
        encoding="utf-8",
    )
    family.write_text(
        json.dumps(
            {
                "name": "sa_family",
                "family_id": FAMILY_ID,
                "study_id": "sa_study",
                "model_ref": model.as_posix(),
                "direction_id_prefix": "sa1",
                "answer_labels": ["A", "B", "C", "D"],
                "label_prefix": " ",
                "master_seed": 20260727,
                "random_control_count": 4,
            }
        ),
        encoding="utf-8",
    )
    plan.write_text(
        json.dumps(
            {
                "name": "sa_plan",
                "plan_id": PLAN_ID,
                "study_id": "sa_study",
                "target": "delta_clean_top_margin",
                "model_ref": model.as_posix(),
                "prompt_manifest_id": MANIFEST_ID,
                "direction_family_id": FAMILY_ID,
                "primary_layer": 13,
                "fallback_layer": 20,
                "norm_ratios": [0.02, 0.05, 0.10, 0.20, 0.40],
                "thresholds": {
                    "min_large_effect_fraction": 0.15,
                    "large_effect_threshold": 0.10,
                    "min_median_abs_effect": 0.05,
                    "max_p95_abs_effect": 4.0,
                },
                "noop_tolerance": 1.0e-3,
                "percentile_method": "numpy.quantile(method='linear')",
                "median_method": "numpy.median",
                "expected_role_counts": {
                    "smoke": 8,
                    "calibration": 1,
                    "training": 1,
                    "final_test": 1,
                },
                "expected_direction_count": 8,
                "expected_signed_directions": 16,
                "master_seed": 20260727,
                "selection_algorithm_version": "bluedot_smallest_passing_ratio_v1.0",
            }
        ),
        encoding="utf-8",
    )
    run_config.write_text(
        json.dumps(_run_config_body(model, task)),
        encoding="utf-8",
    )
    return {
        "model": model,
        "task": task,
        "prompts": prompts,
        "family": family,
        "plan": plan,
        "run": run_config,
    }


def _run_config_body(model: Path, task: Path, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "sa_smoke",
        "study_id": "sa_study",
        "run_role": "engineering_smoke",
        "prompt_role": "smoke",
        "target": "delta_clean_top_margin",
        "model_ref": model.as_posix(),
        "task_ref": task.as_posix(),
        "prompt_manifest_id": MANIFEST_ID,
        "direction_family_id": FAMILY_ID,
        "calibration_plan_id": PLAN_ID,
        "layer": LAYER,
        "capture_position": -1,
        "norm_ratio": 0.10,
        "expected_prompt_count": 8,
        "expected_direction_count": 8,
        "expected_signed_directions": 16,
        "expected_hidden_dim": HIDDEN,
        "expected_role_counts": {"smoke": 8, "calibration": 1, "training": 1, "final_test": 1},
        "master_seed": 20260727,
        "noop_tolerance": 1.0e-3,
        "effect_report_threshold": 0.10,
    }
    body.update(overrides)
    return body


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Workspace:
    processed = tmp_path / "processed"
    manifests = tmp_path / "manifests"
    prompt_manifests = tmp_path / "prompt_manifests"
    directions = tmp_path / "directions"
    direction_manifests = tmp_path / "direction_manifests"
    plans = tmp_path / "calibration_plans"
    runs = tmp_path / "runs"
    for directory in (
        processed,
        manifests,
        prompt_manifests,
        directions,
        direction_manifests,
        plans,
        runs,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(task_loader, "processed_dir", lambda: processed)
    monkeypatch.setattr(pm, "manifests_dir", lambda: manifests)
    monkeypatch.setattr(pm, "prompt_manifest_path", lambda mid: prompt_manifests / f"{mid}.json")
    monkeypatch.setattr(df, "directions_dir", lambda: directions)
    monkeypatch.setattr(
        df, "direction_manifest_path", lambda fid: direction_manifests / f"{fid}.json"
    )
    monkeypatch.setattr(plan_module, "calibration_plan_path", lambda pid: plans / f"{pid}.json")
    monkeypatch.setattr(run_module, "directions_dir", lambda: directions)
    monkeypatch.setattr(paths, "runs_dir", lambda: runs)

    _write_prepared_task(processed, manifests)
    configs = _write_configs(tmp_path)

    assert (
        runner.invoke(app, ["prompts", "manifest", "--config", str(configs["prompts"])]).exit_code
        == 0
    )
    build = runner.invoke(app, ["directions", "build-family", "--config", str(configs["family"])])
    assert build.exit_code == 0, _text(build)
    planned = runner.invoke(app, ["calibration", "plan", "--config", str(configs["plan"])])
    assert planned.exit_code == 0, _text(planned)

    return Workspace(tmp=tmp_path, runs=runs, directions=directions, run_config=configs["run"])


# ---------------------------------------------------------------------------
# The full eight-prompt fixture smoke
# ---------------------------------------------------------------------------


def test_the_eight_prompt_fixture_smoke_runs_end_to_end(workspace: Workspace) -> None:
    result = workspace.smoke()
    assert result.exit_code == 0, _text(result)
    report = json.loads(result.stdout)

    assert report["status"] == "complete"
    assert report["run_role"] == "engineering_smoke"
    assert report["scientific_result"] is False
    assert report["target_name"] == "delta_clean_top_margin"
    assert report["layer"] == LAYER
    assert report["norm_ratio"] == 0.10

    counts = report["counts"]
    assert counts["observed_prompts"] == 8
    assert counts["observed_states"] == 8
    assert counts["observed_non_noop_observations"] == 128
    assert counts["observed_noop_observations"] == 8
    assert counts["observed_forwards"] == 144
    assert counts["expected_forwards"] == 144
    assert counts["failures"] == 0

    directory = run_dir(RUN_ID)
    for name in (
        STATE_AUDIT_RUN_MANIFEST,
        STATE_AUDIT_OBSERVATIONS,
        STATE_AUDIT_CANDIDATE_SETS,
        STATE_AUDIT_CLEAN_PASS,
        STATE_AUDIT_STATE_REFS,
        STATE_AUDIT_STATES,
    ):
        assert (directory / name).exists(), name
    assert not (directory / STATE_AUDIT_FAILURES).exists()

    observations = list(read_jsonl(directory / STATE_AUDIT_OBSERVATIONS))
    assert len(observations) == 136


def test_the_smoke_uses_one_global_alpha_for_every_prompt(workspace: Workspace) -> None:
    assert workspace.smoke().exit_code == 0
    report = json.loads(workspace.verify().stdout)

    alphas = {
        row["global_alpha"]
        for row in read_jsonl(run_dir(RUN_ID) / STATE_AUDIT_OBSERVATIONS)
        if not row["is_noop"]
    }
    assert len(alphas) == 1
    assert report["norm_ratios"] == [0.10]
    assert alphas == set(report["global_alphas"])
    assert report["global_alphas"][0] == pytest.approx(0.10 * report["reference_norm"])
    assert report["observation_checks"]["global_alphas_by_ratio"]["0.1"] == pytest.approx(
        report["global_alphas"][0]
    )


def test_the_reference_norm_is_the_median_of_the_recorded_clean_state_norms(
    workspace: Workspace,
) -> None:
    import numpy as np

    assert workspace.smoke().exit_code == 0
    clean = list(read_jsonl(run_dir(RUN_ID) / STATE_AUDIT_CLEAN_PASS))
    assert len(clean) == 8
    recomputed = float(np.median(np.asarray([row["state_norm"] for row in clean])))

    manifest = json.loads((run_dir(RUN_ID) / STATE_AUDIT_RUN_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["reference_norm"] == pytest.approx(recomputed)
    assert "not the calibration reference norm" in manifest["reference_norm_source"]


def test_only_smoke_role_prompts_are_run(workspace: Workspace) -> None:
    assert workspace.smoke().exit_code == 0
    manifest = json.loads(
        (workspace.tmp / "prompt_manifests" / f"{MANIFEST_ID}.json").read_text(encoding="utf-8")
    )
    smoke_groups = {a["group_id"] for a in manifest["assignments"] if a["role"] == "smoke"}
    other_groups = {a["group_id"] for a in manifest["assignments"] if a["role"] != "smoke"}
    assert len(smoke_groups) == 8

    observed = {row["group_id"] for row in read_jsonl(run_dir(RUN_ID) / STATE_AUDIT_OBSERVATIONS)}
    assert observed == smoke_groups
    assert observed.isdisjoint(other_groups)
    roles = {row["prompt_role"] for row in read_jsonl(run_dir(RUN_ID) / STATE_AUDIT_OBSERVATIONS)}
    assert roles == {"smoke"}


def test_every_noop_reproduces_the_clean_output_exactly(workspace: Workspace) -> None:
    assert workspace.smoke().exit_code == 0
    noops = [
        row for row in read_jsonl(run_dir(RUN_ID) / STATE_AUDIT_OBSERVATIONS) if row["is_noop"]
    ]
    assert len(noops) == 8
    for row in noops:
        assert abs(row["delta_clean_top_margin"]) <= 1.0e-3
        assert row["delta_norm"] == 0.0
        assert row["intervened_logits"] == row["clean_logits"]
        assert row["answer_flip"] is False


def test_the_capture_hooks_fired_and_the_interventions_reconstruct(workspace: Workspace) -> None:
    result = workspace.smoke()
    assert result.exit_code == 0
    diagnostics = json.loads(result.stdout)["diagnostics"]
    assert diagnostics["capture_hooks_fired"] == 8
    assert diagnostics["intervention_hooks_fired"] == 136
    assert diagnostics["state_dim"] == HIDDEN
    assert diagnostics["max_intervention_reconstruction_error"] == pytest.approx(0.0, abs=1e-5)
    assert diagnostics["max_abs_noop_delta_norm"] == 0.0
    assert diagnostics["min_state_norm"] > 0.0


def test_the_state_shard_is_cited_by_every_clean_pass_record(workspace: Workspace) -> None:
    assert workspace.smoke().exit_code == 0
    directory = run_dir(RUN_ID)
    shard_hash = hash_file(directory / STATE_AUDIT_STATES)

    refs = list(read_jsonl(directory / STATE_AUDIT_STATE_REFS))
    assert len(refs) == 8
    assert {ref["shard_hash"] for ref in refs} == {shard_hash}
    assert {ref["layer"] for ref in refs} == {LAYER}
    assert {ref["hidden_dim"] for ref in refs} == {HIDDEN}

    clean = list(read_jsonl(directory / STATE_AUDIT_CLEAN_PASS))
    assert {row["state_shard_hash"] for row in clean} == {shard_hash}
    assert {row["state_id"] for row in clean} == {ref["state_id"] for ref in refs}


def test_clean_accuracy_is_reported_as_a_descriptive_count(workspace: Workspace) -> None:
    result = workspace.smoke()
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["clean_scored_count"] == 8
    assert 0 <= report["clean_correct_count"] <= 8
    assert report["clean_accuracy_descriptive"] == pytest.approx(report["clean_correct_count"] / 8)
    assert "descriptive" in report["notes"]


def test_the_candidate_sets_are_frozen_and_cover_every_observation(workspace: Workspace) -> None:
    assert workspace.smoke().exit_code == 0
    sets = list(read_jsonl(run_dir(RUN_ID) / STATE_AUDIT_CANDIDATE_SETS))
    assert len(sets) == 8
    for candidate_set in sets:
        assert candidate_set["kind"] == "selected_strength"
        assert len(candidate_set["candidates"]) == 17
        assert sum(1 for c in candidate_set["candidates"] if c["is_noop"]) == 1

    expected = {
        (candidate_set["trial_id"], candidate["candidate_id"])
        for candidate_set in sets
        for candidate in candidate_set["candidates"]
    }
    observed = {
        (row["trial_id"], row["candidate_id"])
        for row in read_jsonl(run_dir(RUN_ID) / STATE_AUDIT_OBSERVATIONS)
    }
    assert observed == expected


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def test_verify_run_passes_without_loading_a_model(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert workspace.smoke().exit_code == 0

    from causal_self_forecasting.models import loader as loader_module

    def _refuse(*args, **kwargs):
        raise AssertionError("verification must not load a model")

    monkeypatch.setattr(loader_module, "load_model", _refuse)

    result = workspace.verify()
    assert result.exit_code == 0, _text(result)
    report = json.loads(result.stdout)
    assert report["valid"] is True
    assert report["failures"] == []
    assert report["scientific_result"] is False
    assert report["observations_read"] == 136
    assert report["state_dim"] == HIDDEN
    assert report["observation_checks"]["max_abs_noop_target"] <= 1.0e-3


def test_verify_run_detects_an_edited_observations_file(workspace: Workspace) -> None:
    assert workspace.smoke().exit_code == 0
    path = run_dir(RUN_ID) / STATE_AUDIT_OBSERVATIONS
    rows = list(read_jsonl(path))
    rows[0]["run_id"] = "somebody-elses-run"
    write_jsonl(path, rows)

    result = workspace.verify()
    assert result.exit_code == 1
    assert "hashes to" in _text(result)


def test_verify_run_rejects_an_observation_whose_target_was_edited(workspace: Workspace) -> None:
    assert workspace.smoke().exit_code == 0
    path = run_dir(RUN_ID) / STATE_AUDIT_OBSERVATIONS
    rows = list(read_jsonl(path))
    signed = next(index for index, row in enumerate(rows) if not row["is_noop"])
    rows[signed]["delta_clean_top_margin"] = 42.0
    write_jsonl(path, rows)

    result = workspace.verify()
    assert result.exit_code == 1
    assert "disagrees with its own" in _text(result)


def test_verify_run_reports_a_missing_prompt_manifest(workspace: Workspace) -> None:
    assert workspace.smoke().exit_code == 0
    (workspace.tmp / "prompt_manifests" / f"{MANIFEST_ID}.json").unlink()

    result = workspace.verify()
    assert result.exit_code == 1
    assert "prompt manifest" in _text(result)


def test_verify_run_reports_a_missing_run(workspace: Workspace) -> None:
    result = workspace.verify(run_id="never-happened")
    assert result.exit_code == 1
    assert "no state-audit run manifest" in _text(result)


# ---------------------------------------------------------------------------
# Determinism and rerun behaviour
# ---------------------------------------------------------------------------


def test_a_rerun_into_a_new_run_id_is_deterministic(workspace: Workspace) -> None:
    assert workspace.smoke(run_id="sa-smoke-a").exit_code == 0
    assert workspace.smoke(run_id="sa-smoke-b").exit_code == 0

    result = workspace.verify(run_id="sa-smoke-a", extra=["--compare-run-id", "sa-smoke-b"])
    assert result.exit_code == 0, _text(result)
    determinism = json.loads(result.stdout)["determinism"]
    assert determinism["deterministic"] is True
    assert determinism["compared_observations"] == 136
    assert determinism["max_abs_target_difference"] == 0.0
    assert determinism["max_abs_logit_difference"] == 0.0
    assert determinism["reference_norm_difference"] == 0.0
    assert determinism["max_abs_alpha_difference"] == 0.0


def test_rerunning_the_same_run_id_is_refused(workspace: Workspace) -> None:
    assert workspace.smoke().exit_code == 0
    second = workspace.smoke()
    assert second.exit_code == 1
    assert "already exists and completed" in _text(second)

    # The finished run is untouched by the refusal.
    manifest = run_dir(RUN_ID) / STATE_AUDIT_RUN_MANIFEST
    assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == "complete"


def test_a_run_id_holding_a_partial_attempt_is_refused(workspace: Workspace) -> None:
    """Artifacts without a manifest mean a previous attempt died; a new run id is the remedy."""
    assert workspace.smoke().exit_code == 0
    (run_dir(RUN_ID) / STATE_AUDIT_RUN_MANIFEST).unlink()

    second = workspace.smoke()
    assert second.exit_code == 1
    assert "did not finish" in _text(second)


def test_reusing_a_failed_run_id_for_different_inputs_is_refused(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from causal_self_forecasting.models.capture import CaptureError

    real = run_module.run_with_intervention
    state = {"calls": 0}

    def _fail_once(model, prompt, spec, payload, capture_layers=None):
        state["calls"] += 1
        if state["calls"] == 2:
            raise CaptureError("injected harness failure")
        return real(model, prompt, spec, payload, capture_layers=capture_layers)

    monkeypatch.setattr(run_module, "run_with_intervention", _fail_once)
    first = workspace.smoke(run_id="sa-reuse")
    assert first.exit_code == 1
    assert json.loads(first.stdout)["status"] == "failed"

    monkeypatch.setattr(run_module, "run_with_intervention", real)
    other = workspace.tmp / "smoke_reuse.yaml"
    body = _run_config_body(workspace.tmp / "model.yaml", workspace.tmp / "task.yaml")
    body["study_id"] = "sa_study_other"
    other.write_text(json.dumps(body), encoding="utf-8")

    second = runner.invoke(
        app, ["state-audit", "smoke", "--config", str(other), "--run-id", "sa-reuse"]
    )
    assert second.exit_code == 1
    assert "executed against different inputs" in _text(second)


def test_comparing_two_runs_with_different_inputs_is_refused(workspace: Workspace) -> None:
    assert workspace.smoke(run_id="sa-smoke-a").exit_code == 0

    other = workspace.tmp / "smoke_other.yaml"
    body = _run_config_body(workspace.tmp / "model.yaml", workspace.tmp / "task.yaml")
    body["study_id"] = "sa_study_other"
    other.write_text(json.dumps(body), encoding="utf-8")

    second = runner.invoke(
        app,
        ["state-audit", "smoke", "--config", str(other), "--run-id", "sa-smoke-c"],
    )
    assert second.exit_code == 0, _text(second)

    result = workspace.verify(run_id="sa-smoke-a", extra=["--compare-run-id", "sa-smoke-c"])
    assert result.exit_code == 1
    assert "not executed against the same inputs" in _text(result)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def _variant_config(workspace: Workspace, name: str, **overrides: Any) -> Path:
    path = workspace.tmp / name
    body = _run_config_body(workspace.tmp / "model.yaml", workspace.tmp / "task.yaml", **overrides)
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_the_smoke_refuses_the_fallback_layer(workspace: Workspace) -> None:
    config = _variant_config(workspace, "layer20.yaml", layer=20)
    result = runner.invoke(
        app, ["state-audit", "smoke", "--config", str(config), "--run-id", "sa-l20"]
    )
    assert result.exit_code == 1
    assert "fixed at layer 13" in _text(result)
    assert not (run_dir("sa-l20") / STATE_AUDIT_RUN_MANIFEST).exists()


def test_the_smoke_refuses_another_grid_ratio(workspace: Workspace) -> None:
    config = _variant_config(workspace, "ratio20.yaml", norm_ratio=0.20)
    result = runner.invoke(
        app, ["state-audit", "smoke", "--config", str(config), "--run-id", "sa-r20"]
    )
    assert result.exit_code == 1
    assert "chosen in advance" in _text(result)


def test_the_smoke_refuses_a_non_smoke_prompt_role(workspace: Workspace) -> None:
    config = _variant_config(
        workspace, "training.yaml", prompt_role="training", expected_prompt_count=1
    )
    result = runner.invoke(
        app, ["state-audit", "smoke", "--config", str(config), "--run-id", "sa-train"]
    )
    assert result.exit_code == 1
    assert "prompt role" in _text(result)


def test_the_smoke_refuses_wrong_role_counts(workspace: Workspace) -> None:
    config = _variant_config(
        workspace,
        "counts.yaml",
        expected_role_counts={"smoke": 8, "calibration": 2, "training": 1, "final_test": 1},
    )
    result = runner.invoke(
        app, ["state-audit", "smoke", "--config", str(config), "--run-id", "sa-counts"]
    )
    assert result.exit_code == 1
    assert "role counts" in _text(result)


def test_the_smoke_refuses_a_wrong_hidden_dimension(workspace: Workspace) -> None:
    config = _variant_config(workspace, "dim.yaml", expected_hidden_dim=1152)
    result = runner.invoke(
        app, ["state-audit", "smoke", "--config", str(config), "--run-id", "sa-dim"]
    )
    assert result.exit_code == 1
    assert "dimensional" in _text(result)


def test_the_smoke_refuses_a_dirty_direction_family(workspace: Workspace) -> None:
    stored = sorted(workspace.directions.glob("sa1.*.npz"))
    assert stored
    stored[0].unlink()

    result = workspace.smoke(run_id="sa-dirty")
    assert result.exit_code == 1
    assert "dirty direction family" in _text(result)
    assert not (run_dir("sa-dirty") / STATE_AUDIT_OBSERVATIONS).exists()


def test_the_smoke_refuses_a_prompt_manifest_that_no_longer_matches_the_task(
    workspace: Workspace,
) -> None:
    manifests = workspace.tmp / "manifests"
    payload = json.loads((manifests / f"{TASK_NAME}.json").read_text(encoding="utf-8"))
    payload["item_count"] = 99
    (manifests / f"{TASK_NAME}.json").write_text(json.dumps(payload, sort_keys=True), "utf-8")

    result = workspace.smoke(run_id="sa-stale")
    assert result.exit_code == 1
    assert "no longer matches the prepared task" in _text(result)


# ---------------------------------------------------------------------------
# Failure preservation
# ---------------------------------------------------------------------------


def test_a_failed_intervention_is_recorded_and_the_run_stays_failed(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A candidate that cannot be applied must survive as evidence, never vanish."""
    from causal_self_forecasting.models.capture import CaptureError

    real = run_module.run_with_intervention
    state = {"calls": 0}

    def _sometimes_fail(model, prompt, spec, payload, capture_layers=None):
        state["calls"] += 1
        if state["calls"] == 3:
            raise CaptureError("injected harness failure")
        return real(model, prompt, spec, payload, capture_layers=capture_layers)

    monkeypatch.setattr(run_module, "run_with_intervention", _sometimes_fail)

    result = workspace.smoke(run_id="sa-fail")
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["status"] == "failed"
    assert report["counts"]["failures"] == 1
    assert report["counts"]["observed_forwards"] == 143
    assert (
        report["counts"]["observed_non_noop_observations"]
        + report["counts"]["observed_noop_observations"]
        == 135
    )

    failures = list(read_jsonl(run_dir("sa-fail") / STATE_AUDIT_FAILURES))
    assert len(failures) == 1
    assert failures[0]["stage"] == "intervention"
    assert failures[0]["error_type"] == "CaptureError"
    assert failures[0]["candidate_id"]

    verification = workspace.verify(run_id="sa-fail")
    assert verification.exit_code == 1
    assert "is not a verified run" in _text(verification)


# ---------------------------------------------------------------------------
# Fences against reuse
# ---------------------------------------------------------------------------


def test_smoke_observations_cannot_be_summarized_for_a_calibration_decision(
    workspace: Workspace,
) -> None:
    """Smoke effect sizes must not choose the study's ratio."""
    assert workspace.smoke().exit_code == 0
    result = runner.invoke(
        app,
        [
            "calibration",
            "summarize",
            "--plan-id",
            PLAN_ID,
            "--observations",
            str(run_dir(RUN_ID) / STATE_AUDIT_OBSERVATIONS),
            "--layer",
            str(LAYER),
        ],
    )
    assert result.exit_code == 1
    assert "prompt roles" in _text(result)


def test_the_state_audit_command_group_is_listed() -> None:
    result = runner.invoke(app, ["state-audit", "--help"])
    assert result.exit_code == 0
    for command in ("smoke", "verify-run"):
        assert command in result.stdout


def test_the_verifier_module_exposes_no_model_loader() -> None:
    """A structural check: nothing in the verifier imports a model loading path."""
    source = Path(verify_module.__file__).read_text(encoding="utf-8")
    assert "load_model" not in source
    assert "capture_hidden_states" not in source
    assert "run_with_intervention" not in source
