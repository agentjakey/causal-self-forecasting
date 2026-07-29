"""End-to-end tests for the calibration sweep.

Same fixture workspace as the smoke tests: a locally built 14-block, 64-dim random Llama, every
directory redirected into tmp_path, no network and no gated weights. The fixture's logits are
noise, so nothing measured here is a finding; these tests check that the sweep applies the
preregistered procedure and that the decision falls mechanically out of the summaries.

The interesting cases are the refusals. A calibration run that could quietly widen the grid,
reach a third layer, judge itself against thresholds it chose, or open the fallback after a
layer-13 pass would defeat the point of freezing the plan.
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
from causal_self_forecasting.hashing import read_jsonl, write_jsonl
from causal_self_forecasting.interventions import direction_family as df
from causal_self_forecasting.paths import (
    STATE_AUDIT_CANDIDATE_SETS,
    STATE_AUDIT_CLEAN_PASS,
    STATE_AUDIT_DECISION,
    STATE_AUDIT_FAILURES,
    STATE_AUDIT_OBSERVATIONS,
    STATE_AUDIT_RATIO_SUMMARIES,
    STATE_AUDIT_REFERENCE_NORM,
    STATE_AUDIT_RUN_MANIFEST,
    run_dir,
)
from causal_self_forecasting.schemas import Framing, PromptVariant, Split, TaskItem
from causal_self_forecasting.state_audit import run as run_module
from causal_self_forecasting.tasks import loader as task_loader
from causal_self_forecasting.tasks import prompt_manifest as pm

runner = CliRunner()

TASK_NAME = "caltask"
MANIFEST_ID = "cal_manifest_v1"
FAMILY_ID = "cal_family_v1"
PLAN_ID = "cal_plan_v1"
RUN_ID = "cal-layer13"

HIDDEN = 64
# Deeper than the smoke fixture on purpose: the layer-20 fallback has to be reachable, and a
# 14-block model would refuse it for being out of range rather than for the preregistered reason.
LAYERS = 21
PRIMARY_LAYER = 13
FALLBACK_LAYER = 20
RATIOS = [0.02, 0.05, 0.10, 0.20, 0.40]

# Four calibration prompts rather than 32, so the whole sweep is 4 x 82 = 328 fixture forwards
# instead of 2,624. The shape under test is the grid and the decision, not the sample size.
CALIBRATION_PROMPTS = 4
ROLE_COUNTS = {"smoke": 1, "calibration": CALIBRATION_PROMPTS, "training": 1, "final_test": 1}
TOTAL_PROMPTS = sum(ROLE_COUNTS.values())

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
    model: Path
    task: Path
    primary_config: Path
    fallback_config: Path

    def calibrate(
        self, config: Path | None = None, run_id: str = RUN_ID, extra: list[str] | None = None
    ):
        return runner.invoke(
            app,
            [
                "state-audit",
                "calibrate",
                "--config",
                str(config or self.primary_config),
                "--run-id",
                run_id,
                *(extra or []),
            ],
        )

    def verify(self, run_id: str = RUN_ID):
        return runner.invoke(app, ["state-audit", "verify-run", "--run-id", run_id])


def _write_prepared_task(processed: Path, manifests: Path) -> None:
    from causal_self_forecasting.hashing import hash_file

    items: list[TaskItem] = []
    variants: list[PromptVariant] = []
    for index in range(TOTAL_PROMPTS + 2):
        item = TaskItem(
            item_id=f"cal{index:02d}",
            group_id=f"cal{index:02d}",
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
                "group_count": len(items),
                "item_splits": {"train": 4, "test": 5},
                "variant_splits": {"train": 4, "test": 5},
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


def _calibration_config_body(model: Path, task: Path, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "cal_run",
        "study_id": "cal_study",
        "run_role": "calibration",
        "prompt_role": "calibration",
        "candidate_kind": "calibration_grid",
        "target": "delta_clean_top_margin",
        "model_ref": model.as_posix(),
        "task_ref": task.as_posix(),
        "prompt_manifest_id": MANIFEST_ID,
        "direction_family_id": FAMILY_ID,
        "calibration_plan_id": PLAN_ID,
        "layer": PRIMARY_LAYER,
        "capture_position": -1,
        "norm_ratios": RATIOS,
        "expected_prompt_count": CALIBRATION_PROMPTS,
        "expected_direction_count": 8,
        "expected_signed_directions": 16,
        "expected_hidden_dim": HIDDEN,
        "expected_role_counts": dict(ROLE_COUNTS),
        "master_seed": 20260727,
        "noop_tolerance": 1.0e-3,
        "effect_report_threshold": 0.10,
    }
    body.update(overrides)
    return body


def _write_configs(tmp_path: Path) -> dict[str, Path]:
    paths_out: dict[str, Path] = {}
    model = tmp_path / "model.yaml"
    task = tmp_path / "task.yaml"

    model.write_text(
        json.dumps(
            {
                "name": "fixture_calibration",
                "kind": "fixture",
                "model_id": "tiny_llama_sa21",
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

    prompts = tmp_path / "prompts.yaml"
    prompts.write_text(
        json.dumps(
            {
                "name": "cal_prompts",
                "manifest_id": MANIFEST_ID,
                "task_ref": task.as_posix(),
                "canonical_wrapper_id": "neutral_a",
                "master_seed": 20260727,
                "role_counts": dict(ROLE_COUNTS),
            }
        ),
        encoding="utf-8",
    )
    family = tmp_path / "family.yaml"
    family.write_text(
        json.dumps(
            {
                "name": "cal_family",
                "family_id": FAMILY_ID,
                "study_id": "cal_study",
                "model_ref": model.as_posix(),
                "direction_id_prefix": "cl1",
                "answer_labels": ["A", "B", "C", "D"],
                "label_prefix": " ",
                "master_seed": 20260727,
                "random_control_count": 4,
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        json.dumps(
            {
                "name": "cal_plan",
                "plan_id": PLAN_ID,
                "study_id": "cal_study",
                "target": "delta_clean_top_margin",
                "model_ref": model.as_posix(),
                "prompt_manifest_id": MANIFEST_ID,
                "direction_family_id": FAMILY_ID,
                "primary_layer": PRIMARY_LAYER,
                "fallback_layer": FALLBACK_LAYER,
                "norm_ratios": RATIOS,
                "thresholds": {
                    "min_large_effect_fraction": 0.15,
                    "large_effect_threshold": 0.10,
                    "min_median_abs_effect": 0.05,
                    "max_p95_abs_effect": 4.0,
                },
                "noop_tolerance": 1.0e-3,
                "percentile_method": "numpy.quantile(method='linear')",
                "median_method": "numpy.median",
                "expected_role_counts": dict(ROLE_COUNTS),
                "expected_direction_count": 8,
                "expected_signed_directions": 16,
                "master_seed": 20260727,
                "selection_algorithm_version": "bluedot_smallest_passing_ratio_v1.0",
            }
        ),
        encoding="utf-8",
    )

    primary = tmp_path / "cal13.yaml"
    primary.write_text(json.dumps(_calibration_config_body(model, task)), encoding="utf-8")
    fallback = tmp_path / "cal20.yaml"
    fallback.write_text(
        json.dumps(_calibration_config_body(model, task, layer=FALLBACK_LAYER)), encoding="utf-8"
    )

    paths_out.update(
        model=model,
        task=task,
        prompts=prompts,
        family=family,
        plan=plan,
        primary=primary,
        fallback=fallback,
    )
    return paths_out


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

    return Workspace(
        tmp=tmp_path,
        runs=runs,
        directions=directions,
        model=configs["model"],
        task=configs["task"],
        primary_config=configs["primary"],
        fallback_config=configs["fallback"],
    )


def _variant_config(workspace: Workspace, name: str, **overrides: Any) -> Path:
    path = workspace.tmp / name
    path.write_text(
        json.dumps(_calibration_config_body(workspace.model, workspace.task, **overrides)),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


def test_the_calibration_sweep_runs_the_whole_grid(workspace: Workspace) -> None:
    result = workspace.calibrate()
    assert result.exit_code in (0, 1), _text(result)
    report = json.loads(result.stdout)

    assert report["status"] == "complete"
    assert report["run_role"] == "calibration"
    assert report["scientific_result"] is False
    assert report["layer"] == PRIMARY_LAYER
    assert report["norm_ratios"] == RATIOS

    counts = report["counts"]
    # 4 prompts x (1 clean + 81 candidates) = 328 forwards; 4 x 80 signed and 4 no-ops.
    assert counts["observed_prompts"] == CALIBRATION_PROMPTS
    assert counts["observed_states"] == CALIBRATION_PROMPTS
    assert counts["observed_non_noop_observations"] == CALIBRATION_PROMPTS * 80
    assert counts["observed_noop_observations"] == CALIBRATION_PROMPTS
    assert counts["observed_forwards"] == CALIBRATION_PROMPTS * 82
    assert counts["failures"] == 0

    directory = run_dir(RUN_ID)
    for name in (
        STATE_AUDIT_RUN_MANIFEST,
        STATE_AUDIT_OBSERVATIONS,
        STATE_AUDIT_CANDIDATE_SETS,
        STATE_AUDIT_CLEAN_PASS,
        STATE_AUDIT_REFERENCE_NORM,
        STATE_AUDIT_RATIO_SUMMARIES,
        STATE_AUDIT_DECISION,
    ):
        assert (directory / name).exists(), name
    assert not (directory / STATE_AUDIT_FAILURES).exists()

    observations = list(read_jsonl(directory / STATE_AUDIT_OBSERVATIONS))
    assert len(observations) == CALIBRATION_PROMPTS * 81


def test_one_alpha_per_ratio_derived_from_one_reference_norm(workspace: Workspace) -> None:
    import numpy as np

    assert workspace.calibrate().exit_code in (0, 1)
    report = json.loads((run_dir(RUN_ID) / STATE_AUDIT_RUN_MANIFEST).read_text(encoding="utf-8"))

    clean = list(read_jsonl(run_dir(RUN_ID) / STATE_AUDIT_CLEAN_PASS))
    recomputed = float(np.median(np.asarray([row["state_norm"] for row in clean])))
    assert report["reference_norm"] == pytest.approx(recomputed)
    assert "preregistered calibration reference norm" in report["reference_norm_source"]

    assert report["norm_ratios"] == RATIOS
    for ratio, alpha in zip(report["norm_ratios"], report["global_alphas"], strict=True):
        assert alpha == pytest.approx(ratio * report["reference_norm"])

    observed: dict[float, set[float]] = {}
    for row in read_jsonl(run_dir(RUN_ID) / STATE_AUDIT_OBSERVATIONS):
        if row["is_noop"]:
            continue
        observed.setdefault(row["norm_ratio"], set()).add(row["global_alpha"])
    assert sorted(observed) == RATIOS
    for ratio, alphas in observed.items():
        assert len(alphas) == 1, f"ratio {ratio} used more than one alpha"


def test_the_reference_norm_record_carries_every_norm_it_was_computed_from(
    workspace: Workspace,
) -> None:
    assert workspace.calibrate().exit_code in (0, 1)
    record = json.loads((run_dir(RUN_ID) / STATE_AUDIT_REFERENCE_NORM).read_text(encoding="utf-8"))
    assert record["scientific_result"] is False
    assert record["layer"] == PRIMARY_LAYER
    assert record["count"] == CALIBRATION_PROMPTS
    assert len(record["state_norms"]) == CALIBRATION_PROMPTS
    assert sorted(record["state_norms"]) == record["prompt_ids"]


def test_the_grid_carries_one_shared_noop_per_prompt(workspace: Workspace) -> None:
    assert workspace.calibrate().exit_code in (0, 1)
    sets = list(read_jsonl(run_dir(RUN_ID) / STATE_AUDIT_CANDIDATE_SETS))
    assert len(sets) == CALIBRATION_PROMPTS
    for candidate_set in sets:
        assert candidate_set["kind"] == "calibration_grid"
        assert len(candidate_set["candidates"]) == 81
        assert sum(1 for c in candidate_set["candidates"] if c["is_noop"]) == 1
        assert candidate_set["norm_ratios"] == RATIOS

    noops = [
        row for row in read_jsonl(run_dir(RUN_ID) / STATE_AUDIT_OBSERVATIONS) if row["is_noop"]
    ]
    assert len(noops) == CALIBRATION_PROMPTS
    for row in noops:
        assert abs(row["delta_clean_top_margin"]) <= 1.0e-3
        assert row["delta_norm"] == 0.0
        assert row["intervened_logits"] == row["clean_logits"]


def test_only_calibration_role_prompts_are_swept(workspace: Workspace) -> None:
    assert workspace.calibrate().exit_code in (0, 1)
    manifest = json.loads(
        (workspace.tmp / "prompt_manifests" / f"{MANIFEST_ID}.json").read_text(encoding="utf-8")
    )
    calibration_groups = {
        a["group_id"] for a in manifest["assignments"] if a["role"] == "calibration"
    }
    other = {a["group_id"] for a in manifest["assignments"] if a["role"] != "calibration"}

    observed = {row["group_id"] for row in read_jsonl(run_dir(RUN_ID) / STATE_AUDIT_OBSERVATIONS)}
    assert observed == calibration_groups
    assert observed.isdisjoint(other)


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


def test_the_decision_is_recorded_with_one_summary_per_ratio(workspace: Workspace) -> None:
    result = workspace.calibrate()
    report = json.loads(result.stdout)

    assert len(report["ratio_table"]) == len(RATIOS)
    assert [row["norm_ratio"] for row in report["ratio_table"]] == RATIOS
    for row in report["ratio_table"]:
        assert row["layer"] == PRIMARY_LAYER
        assert row["observed_non_noop_observations"] == CALIBRATION_PROMPTS * 16
        assert row["noop_count"] == CALIBRATION_PROMPTS
        assert row["max_abs_noop_target"] <= 1.0e-3
        assert row["finite_output_rate"] == 1.0

    decision = json.loads((run_dir(RUN_ID) / STATE_AUDIT_DECISION).read_text(encoding="utf-8"))
    assert decision["scientific_result"] is False
    assert decision["decision_hash"].startswith("sha256:")
    assert decision["status"] == report["decision"]["status"]


def test_the_selected_ratio_is_the_smallest_passing_one(workspace: Workspace) -> None:
    result = workspace.calibrate()
    report = json.loads(result.stdout)
    passing = [row["norm_ratio"] for row in report["ratio_table"] if row["passed"]]
    status = report["decision"]["status"]

    if passing:
        assert status == "passed_primary"
        assert report["decision"]["selected_norm_ratio"] == min(passing)
        assert "smallest passing ratio" in report["decision"]["rationale"]
        index = RATIOS.index(min(passing))
        assert report["decision"]["selected_global_alpha"] == pytest.approx(
            report["global_alphas"][index]
        )
        assert result.exit_code == 0
    else:
        assert status == "fallback_required"
        assert report["decision"]["selected_norm_ratio"] is None
        assert result.exit_code == 0


def test_a_calibration_run_verifies_without_a_model(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert workspace.calibrate().exit_code in (0, 1)

    from causal_self_forecasting.models import loader as loader_module

    def _refuse(*args, **kwargs):
        raise AssertionError("verification must not load a model")

    monkeypatch.setattr(loader_module, "load_model", _refuse)

    result = workspace.verify()
    assert result.exit_code == 0, _text(result)
    report = json.loads(result.stdout)
    assert report["valid"] is True
    assert report["failures"] == []
    assert report["norm_ratios"] == RATIOS
    assert len(report["global_alphas"]) == len(RATIOS)
    assert report["observations_read"] == CALIBRATION_PROMPTS * 81
    assert set(report["observation_checks"]["global_alphas_by_ratio"]) == {
        f"{ratio:g}" for ratio in RATIOS
    }
    assert "calibration_checks" in report


def test_verification_catches_an_edited_decision(workspace: Workspace) -> None:
    assert workspace.calibrate().exit_code in (0, 1)
    path = run_dir(RUN_ID) / STATE_AUDIT_DECISION
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selection_rationale"] = "because it looked good"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = workspace.verify()
    assert result.exit_code == 1
    assert "calibration decision" in _text(result)


def test_verification_catches_an_edited_summaries_file(workspace: Workspace) -> None:
    assert workspace.calibrate().exit_code in (0, 1)
    path = run_dir(RUN_ID) / STATE_AUDIT_RATIO_SUMMARIES
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["notes"] = "edited"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = workspace.verify()
    assert result.exit_code == 1
    assert STATE_AUDIT_RATIO_SUMMARIES in _text(result)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_widened_grid_is_refused(workspace: Workspace) -> None:
    config = _variant_config(workspace, "wide.yaml", norm_ratios=[*RATIOS, 0.80])
    result = workspace.calibrate(config=config, run_id="cal-wide")
    assert result.exit_code == 1
    assert "norm_ratios must be exactly" in _text(result)


def test_a_reordered_grid_is_refused(workspace: Workspace) -> None:
    config = _variant_config(workspace, "reordered.yaml", norm_ratios=list(reversed(RATIOS)))
    result = workspace.calibrate(config=config, run_id="cal-reordered")
    assert result.exit_code == 1
    assert "norm_ratios must be exactly" in _text(result)


def test_a_third_layer_is_refused(workspace: Workspace) -> None:
    config = _variant_config(workspace, "layer17.yaml", layer=17)
    result = workspace.calibrate(config=config, run_id="cal-l17")
    assert result.exit_code == 1
    assert "preregistered layers" in _text(result)


def test_a_non_calibration_prompt_role_is_refused(workspace: Workspace) -> None:
    config = _variant_config(
        workspace, "training.yaml", prompt_role="training", expected_prompt_count=1
    )
    result = workspace.calibrate(config=config, run_id="cal-training")
    assert result.exit_code == 1
    assert "prompt role" in _text(result)


def test_a_selected_strength_config_is_refused(workspace: Workspace) -> None:
    body = _calibration_config_body(workspace.model, workspace.task)
    body.pop("norm_ratios")
    body["candidate_kind"] = "selected_strength"
    body["norm_ratio"] = 0.10
    config = workspace.tmp / "single.yaml"
    config.write_text(json.dumps(body), encoding="utf-8")

    result = workspace.calibrate(config=config, run_id="cal-single")
    assert result.exit_code == 1
    assert "ratio grid" in _text(result)


def test_a_config_naming_both_a_ratio_and_a_grid_is_refused(workspace: Workspace) -> None:
    config = _variant_config(workspace, "both.yaml", norm_ratio=0.10)
    result = workspace.calibrate(config=config, run_id="cal-both")
    assert result.exit_code == 1
    assert "must not also name a single" in _text(result)


def test_a_tolerance_that_disagrees_with_the_plan_is_refused(workspace: Workspace) -> None:
    config = _variant_config(workspace, "tolerance.yaml", noop_tolerance=0.5)
    result = workspace.calibrate(config=config, run_id="cal-tolerance")
    assert result.exit_code == 1
    assert "no-op tolerance" in _text(result)


def test_the_smoke_command_refuses_a_calibration_config(workspace: Workspace) -> None:
    result = runner.invoke(
        app,
        [
            "state-audit",
            "smoke",
            "--config",
            str(workspace.primary_config),
            "--run-id",
            "cal-as-smoke",
        ],
    )
    assert result.exit_code == 1
    assert "engineering smoke only" in _text(result)


# ---------------------------------------------------------------------------
# The layer-20 fallback gate
# ---------------------------------------------------------------------------


def test_the_fallback_layer_refuses_to_run_without_a_primary_run(workspace: Workspace) -> None:
    result = workspace.calibrate(config=workspace.fallback_config, run_id="cal-l20")
    assert result.exit_code == 1
    assert "requires the primary run" in _text(result)
    assert not (run_dir("cal-l20") / STATE_AUDIT_OBSERVATIONS).exists()


def test_a_primary_run_id_is_refused_at_the_primary_layer(workspace: Workspace) -> None:
    result = workspace.calibrate(extra=["--primary-run-id", "whatever"])
    assert result.exit_code == 1
    assert "only meaningful when calibrating the fallback layer" in _text(result)


def test_the_fallback_refuses_a_primary_run_that_did_not_fail(workspace: Workspace) -> None:
    """The single most important gate: the fallback is not a second attempt."""
    first = workspace.calibrate()
    status = json.loads(first.stdout)["decision"]["status"]

    result = workspace.calibrate(
        config=workspace.fallback_config, run_id="cal-l20", extra=["--primary-run-id", RUN_ID]
    )
    if status == "passed_primary":
        assert result.exit_code == 1
        assert "reachable only from 'fallback_required'" in _text(result)
        assert not (run_dir("cal-l20") / STATE_AUDIT_OBSERVATIONS).exists()
    else:
        # The primary layer genuinely failed, so the fallback is open and must run.
        assert result.exit_code in (0, 1), _text(result)
        report = json.loads(result.stdout)
        assert report["layer"] == FALLBACK_LAYER
        assert report["decision"]["status"] in ("passed_fallback", "failed_all_layers")


def test_the_fallback_refuses_a_missing_primary_decision(workspace: Workspace) -> None:
    result = workspace.calibrate(
        config=workspace.fallback_config,
        run_id="cal-l20",
        extra=["--primary-run-id", "never-happened"],
    )
    assert result.exit_code == 1
    assert "no calibration decision" in _text(result)


# ---------------------------------------------------------------------------
# Rerun behaviour and failure preservation
# ---------------------------------------------------------------------------


def test_rerunning_a_completed_calibration_is_refused(workspace: Workspace) -> None:
    assert workspace.calibrate().exit_code in (0, 1)
    second = workspace.calibrate()
    assert second.exit_code == 1
    assert "already exists and completed" in _text(second)


def test_a_failed_intervention_is_preserved_and_the_run_stays_failed(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from causal_self_forecasting.models.capture import CaptureError

    real = run_module.run_with_intervention
    state = {"calls": 0}

    def _fail_once(model, prompt, spec, payload, capture_layers=None):
        state["calls"] += 1
        if state["calls"] == 5:
            raise CaptureError("injected harness failure")
        return real(model, prompt, spec, payload, capture_layers=capture_layers)

    monkeypatch.setattr(run_module, "run_with_intervention", _fail_once)

    result = workspace.calibrate(run_id="cal-fail")
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["status"] == "failed"
    assert report["counts"]["failures"] == 1

    failures = list(read_jsonl(run_dir("cal-fail") / STATE_AUDIT_FAILURES))
    assert len(failures) == 1
    assert failures[0]["stage"] == "intervention"
    assert failures[0]["norm_ratio"] in RATIOS

    # The failure is attributed to its own grid point rather than smeared across the grid.
    summaries = json.loads(
        (run_dir("cal-fail") / STATE_AUDIT_RATIO_SUMMARIES).read_text(encoding="utf-8")
    )["summaries"]
    charged = [s for s in summaries if s["failure_count"] == 1]
    assert len(charged) == 1
    assert charged[0]["norm_ratio"] == failures[0]["norm_ratio"]


def test_the_calibration_command_is_listed() -> None:
    result = runner.invoke(app, ["state-audit", "--help"])
    assert result.exit_code == 0
    for command in ("smoke", "calibrate", "verify-run"):
        assert command in result.stdout


def test_the_real_calibration_configs_validate() -> None:
    """The study's own configs, not a fixture: both layers, both frozen."""
    from causal_self_forecasting.config import StateAuditRunConfig, load_config
    from causal_self_forecasting.schemas import StateAuditCandidateKind

    primary = load_config(
        "configs/state_audit/bluedot_calibration_layer13.yaml", StateAuditRunConfig
    )
    fallback = load_config(
        "configs/state_audit/bluedot_calibration_layer20.yaml", StateAuditRunConfig
    )

    for config, layer in ((primary, 13), (fallback, 20)):
        assert config.layer == layer
        assert config.candidate_kind is StateAuditCandidateKind.CALIBRATION_GRID
        assert config.ratio_grid == (0.02, 0.05, 0.10, 0.20, 0.40)
        assert config.expected_prompt_count == 32
        assert config.expected_hidden_dim == 1152
        assert config.candidates_per_prompt == 81
        assert config.expected_forward_count == 2624
