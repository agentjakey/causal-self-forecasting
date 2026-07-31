"""End-to-end test of the training, clean, fit, and commitment stages on the fixture model.

Runs the whole precommitted-forecasting path offline: a training run at an inherited strength, a
final-test clean stage that applies no intervention, the fixed projection, the three ridges, the
wrong-state pairing, and the full set of commitments.

The counts are scaled down (20 training prompts, 6 final-test prompts) so the suite stays fast.
The shapes under test are the structure and the guarantees, not the sample size: sixteen records
per prompt, one salt each, the fit boundary, and the absence of any final-test outcome.
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
from causal_self_forecasting.calibration.plan import build_decision_record, load_calibration_plan
from causal_self_forecasting.calibration.selection import select_calibration_ratio
from causal_self_forecasting.cli import app
from causal_self_forecasting.hashing import (
    atomic_write_json,
    hash_file,
    hash_object,
    read_jsonl,
    write_jsonl,
)
from causal_self_forecasting.interventions import direction_family as df
from causal_self_forecasting.paths import (
    FORECAST_COMMITMENTS,
    FORECASTS,
    STATE_AUDIT_CANDIDATE_SETS,
    STATE_AUDIT_CLEAN_PASS,
    STATE_AUDIT_DECISION,
    STATE_AUDIT_OBSERVATIONS,
    STATE_AUDIT_PAIRING,
    STATE_AUDIT_PREDICTORS,
    STATE_AUDIT_RATIO_SUMMARIES,
    STATE_AUDIT_REFERENCE_NORM,
    STATE_AUDIT_RUN_MANIFEST,
    STATE_AUDIT_TRANSFORM_FITS,
    run_dir,
)
from causal_self_forecasting.schemas import (
    CalibrationCriterionResult,
    CalibrationRatioSummary,
    Framing,
    LayerReferenceNormRecord,
    PromptVariant,
    Split,
    TaskItem,
)
from causal_self_forecasting.state_audit import predict as predict_module
from causal_self_forecasting.state_audit import projection as projection_module
from causal_self_forecasting.state_audit import run as run_module
from causal_self_forecasting.state_audit.calibrate import MEDIAN_METHOD
from causal_self_forecasting.tasks import loader as task_loader
from causal_self_forecasting.tasks import prompt_manifest as pm

runner = CliRunner()

TASK_NAME = "fcttask"
MANIFEST_ID = "fct_manifest_v1"
FAMILY_ID = "fct_family_v1"
PLAN_ID = "fct_plan_v1"
PROJECTION_ID = "fct_projection_v1"
CAL_RUN_ID = "fct-cal"
TRAIN_RUN_ID = "fct-train"
FINAL_RUN_ID = "fct-final"

HIDDEN = 64
LAYERS = 21
LAYER = 13
RATIOS = [0.02, 0.05, 0.10, 0.20, 0.40]
SELECTED_RATIO = 0.02
REFERENCE_NORM = 40.0
SELECTED_ALPHA = SELECTED_RATIO * REFERENCE_NORM

SMOKE_N, CAL_N, TRAIN_N, FINAL_N = 1, 4, 20, 6
ROLE_COUNTS = {
    "smoke": SMOKE_N,
    "calibration": CAL_N,
    "training": TRAIN_N,
    "final_test": FINAL_N,
}
TOTAL = sum(ROLE_COUNTS.values())

# Sixteen records per prompt: four at state_condition none, one true, one matched, ten shuffled.
RECORDS_PER_PROMPT = 16
CANDIDATES_PER_PROMPT = 17

_WORDS = ["water", "energy", "rock", "heat", "light", "gas", "cell", "force", "sun", "earth"]


def _text(result) -> str:
    parts = [result.stdout or ""]
    with contextlib.suppress(ValueError):
        parts.append(result.stderr or "")
    return "".join(parts)


@dataclass
class Workspace:
    tmp: Path
    model: Path
    task: Path
    training_config: Path
    final_config: Path


def _write_prepared_task(processed: Path, manifests: Path) -> None:
    items: list[TaskItem] = []
    variants: list[PromptVariant] = []
    for index in range(TOTAL + 3):
        item = TaskItem(
            item_id=f"fct{index:02d}",
            group_id=f"fct{index:02d}",
            source="allenai/ai2_arc",
            source_id=f"src{index}",
            subject="ARC-Challenge",
            # Every question must be a distinct token sequence. Repeating text would give two
            # prompts byte-identical states on a deterministic model, the nearest-donor rule
            # would correctly pick the exact twin, and a wrong-state substitution would then
            # change nothing for those prompts.
            question=" ".join(
                _WORDS[: 3 + index % 6] + [_WORDS[index % len(_WORDS)]] * (index + 1)
            ),
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
                    f"answer the question using one letter {item.question} "
                    f"A water B energy C light D heat Answer:"
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
                "item_splits": {"train": 17, "test": 17},
                "variant_splits": {"train": 17, "test": 17},
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


def _run_config(model: Path, task: Path, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "fct_run",
        "study_id": "fct_study",
        "run_role": "training",
        "prompt_role": "training",
        "candidate_kind": "selected_strength",
        "target": "delta_clean_top_margin",
        "model_ref": model.as_posix(),
        "task_ref": task.as_posix(),
        "prompt_manifest_id": MANIFEST_ID,
        "direction_family_id": FAMILY_ID,
        "calibration_plan_id": PLAN_ID,
        "projection_id": PROJECTION_ID,
        "calibration_decision_run_id": CAL_RUN_ID,
        "layer": LAYER,
        "capture_position": -1,
        "norm_ratio": SELECTED_RATIO,
        "expected_prompt_count": TRAIN_N,
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


def _passing_summary(ratio: float, passed: bool) -> CalibrationRatioSummary:
    criteria = [
        CalibrationCriterionResult(
            name=name,
            passed=passed,
            observed=1.0 if passed else 0.0,
            threshold=0.5,
            comparison=">=",
        )
        for name in (
            "completeness",
            "finite_outputs",
            "noop_within_tolerance",
            "large_effect_fraction",
            "median_abs_effect",
            "p95_abs_effect",
        )
    ]
    return CalibrationRatioSummary(
        layer=LAYER,
        norm_ratio=ratio,
        global_alpha=ratio * REFERENCE_NORM,
        expected_non_noop_observations=CAL_N * 16,
        observed_non_noop_observations=CAL_N * 16,
        noop_count=CAL_N,
        failure_count=0,
        finite_output_rate=1.0,
        max_abs_noop_target=0.0,
        fraction_above_effect_threshold=0.5,
        median_abs_effect=0.3,
        p95_abs_effect=1.0,
        flip_count=3,
        criteria=criteria,
        passed=passed,
    )


def _write_calibration_decision() -> None:
    """A decision record for the training run to inherit from.

    Built through the real selector rather than hand-written, so the training run inherits a
    strength that came out of the same state machine the study uses. The summaries are synthetic;
    the fixture's own noise cannot be relied on to pass a preregistered condition, and what this
    test is about is the inheritance, not the calibration.
    """
    plan = load_calibration_plan(PLAN_ID)
    summaries = [_passing_summary(ratio, ratio == SELECTED_RATIO) for ratio in RATIOS]
    selection = select_calibration_ratio(
        primary_layer=plan.primary_layer,
        fallback_layer=plan.fallback_layer,
        expected_ratios=plan.norm_ratios,
        primary_summaries=summaries,
    )
    decision = build_decision_record(plan, selection)
    directory = run_dir(CAL_RUN_ID)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(directory / STATE_AUDIT_DECISION, decision.model_dump(mode="json"))

    # The evidence behind the decision. A real calibration run writes these beside it, and the
    # public bundle requires them, so the fixture writes them too rather than letting the bundle
    # quietly treat them as optional.
    atomic_write_json(
        directory / STATE_AUDIT_RATIO_SUMMARIES,
        {"summaries": [summary.model_dump(mode="json") for summary in summaries]},
    )
    prompt_ids = [f"{CAL_RUN_ID}_prompt_{index:02d}" for index in range(CAL_N)]
    reference = LayerReferenceNormRecord(
        layer=LAYER,
        prompt_ids=prompt_ids,
        prompt_identity_hash=hash_object(prompt_ids),
        count=len(prompt_ids),
        state_norms=dict.fromkeys(prompt_ids, REFERENCE_NORM),
        reference_norm=REFERENCE_NORM,
        median_method=MEDIAN_METHOD,
        prompt_manifest_hash=plan.prompt_manifest_hash,
        model_id=plan.model_id,
        model_revision=plan.model_revision,
    )
    atomic_write_json(directory / STATE_AUDIT_REFERENCE_NORM, reference.model_dump(mode="json"))


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Workspace:
    processed = tmp_path / "processed"
    manifests = tmp_path / "manifests"
    prompt_manifests = tmp_path / "prompt_manifests"
    directions = tmp_path / "directions"
    direction_manifests = tmp_path / "direction_manifests"
    plans = tmp_path / "calibration_plans"
    projections = tmp_path / "projections"
    runs = tmp_path / "runs"
    for directory in (
        processed,
        manifests,
        prompt_manifests,
        directions,
        direction_manifests,
        plans,
        projections,
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
    monkeypatch.setattr(predict_module, "directions_dir", lambda: directions)
    monkeypatch.setattr(
        projection_module, "projection_path", lambda pid: projections / f"{pid}.json"
    )
    monkeypatch.setattr(paths, "runs_dir", lambda: runs)

    _write_prepared_task(processed, manifests)

    model = tmp_path / "model.yaml"
    task = tmp_path / "task.yaml"
    model.write_text(
        json.dumps(
            {
                "name": "fixture_forecasts",
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
    wrappers = [
        {"wrapper_id": wid, "framing": framing, "template": "{question}\n{choices}\nAnswer:"}
        for wid, framing in (
            ("neutral_a", "neutral"),
            ("eval_a", "evaluation"),
            ("deploy_a", "deployment"),
        )
    ]
    task.write_text(
        json.dumps(
            {
                "name": TASK_NAME,
                "source": "allenai/ai2_arc",
                "source_config": "ARC-Challenge",
                "answer_labels": ["A", "B", "C", "D"],
                "label_prefix": " ",
                "wrappers": wrappers,
            }
        ),
        encoding="utf-8",
    )

    prompts = tmp_path / "prompts.yaml"
    prompts.write_text(
        json.dumps(
            {
                "name": "fct_prompts",
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
                "name": "fct_family",
                "family_id": FAMILY_ID,
                "study_id": "fct_study",
                "model_ref": model.as_posix(),
                "direction_id_prefix": "ft1",
                "answer_labels": ["A", "B", "C", "D"],
                "label_prefix": " ",
                "master_seed": 20260727,
                "random_control_count": 4,
            }
        ),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        json.dumps(
            {
                "name": "fct_plan",
                "plan_id": PLAN_ID,
                "study_id": "fct_study",
                "target": "delta_clean_top_margin",
                "model_ref": model.as_posix(),
                "prompt_manifest_id": MANIFEST_ID,
                "direction_family_id": FAMILY_ID,
                "primary_layer": LAYER,
                "fallback_layer": 20,
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

    training_config = tmp_path / "training.yaml"
    training_config.write_text(json.dumps(_run_config(model, task)), encoding="utf-8")
    final_config = tmp_path / "final.yaml"
    final_config.write_text(
        json.dumps(
            _run_config(
                model,
                task,
                name="fct_final",
                run_role="final_test_unresolved",
                prompt_role="final_test",
                expected_prompt_count=FINAL_N,
            )
        ),
        encoding="utf-8",
    )

    assert runner.invoke(app, ["prompts", "manifest", "--config", str(prompts)]).exit_code == 0
    build = runner.invoke(app, ["directions", "build-family", "--config", str(family)])
    assert build.exit_code == 0, _text(build)
    planned = runner.invoke(app, ["calibration", "plan", "--config", str(plan_path)])
    assert planned.exit_code == 0, _text(planned)
    _write_calibration_decision()

    projection = runner.invoke(
        app,
        [
            "state-audit",
            "projection",
            "--config",
            str(training_config),
            "--projection-id",
            PROJECTION_ID,
        ],
    )
    assert projection.exit_code == 0, _text(projection)

    return Workspace(
        tmp=tmp_path,
        model=model,
        task=task,
        training_config=training_config,
        final_config=final_config,
    )


# The helpers call the Python entry points rather than the CLI. Each CLI invocation attaches a
# fresh file log handler, and several invocations inside one pytest process end up writing to a
# capture stream pytest has already closed. The commands are thin wrappers over these functions,
# and the CLI surface itself is covered separately below.


def run_training(workspace: Workspace):
    """The training run goes through the same executor the smoke uses."""
    from causal_self_forecasting.state_audit.run import execute_state_audit_run

    return execute_state_audit_run(workspace.training_config, TRAIN_RUN_ID)


def run_final_clean(workspace: Workspace):
    from causal_self_forecasting.state_audit.run import execute_clean_only

    return execute_clean_only(workspace.final_config, FINAL_RUN_ID)


def commit(workspace: Workspace):
    from causal_self_forecasting.state_audit.predict import commit_final_test_forecasts

    return commit_final_test_forecasts(
        workspace.training_config,
        TRAIN_RUN_ID,
        workspace.final_config,
        FINAL_RUN_ID,
        PROJECTION_ID,
    )


# ---------------------------------------------------------------------------
# The projection
# ---------------------------------------------------------------------------


def test_the_projection_is_built_and_verifies(workspace: Workspace) -> None:
    manifest = json.loads(
        (workspace.tmp / "projections" / f"{PROJECTION_ID}.json").read_text(encoding="utf-8")
    )
    assert manifest["components"] == 16
    assert manifest["hidden_dim"] == HIDDEN
    assert manifest["realized_vector_count"] == 16
    assert manifest["injectivity_margin"] > 0.0
    # Rank preservation is the property; the retention says the projection kept the subspace
    # rather than that the direction family happens to be well conditioned.
    assert manifest["injectivity_retention"] > 1e-3
    assert manifest["scientific_result"] is False


# ---------------------------------------------------------------------------
# Training at the inherited strength
# ---------------------------------------------------------------------------


def test_the_training_run_inherits_the_calibrated_strength(workspace: Workspace) -> None:
    report = run_training(workspace)
    assert report["status"] == "complete"
    assert report["norm_ratio"] == SELECTED_RATIO
    assert report["global_alpha"] == pytest.approx(SELECTED_ALPHA)
    assert report["reference_norm"] == pytest.approx(REFERENCE_NORM)
    assert "inherited from the calibration decision" in report["reference_norm_source"]

    counts = report["counts"]
    assert counts["observed_prompts"] == TRAIN_N
    assert counts["observed_non_noop_observations"] == TRAIN_N * 16
    assert counts["observed_noop_observations"] == TRAIN_N
    assert counts["observed_forwards"] == TRAIN_N * 18
    assert counts["failures"] == 0

    alphas = {
        row["global_alpha"]
        for row in read_jsonl(run_dir(TRAIN_RUN_ID) / STATE_AUDIT_OBSERVATIONS)
        if not row["is_noop"]
    }
    assert len(alphas) == 1
    assert next(iter(alphas)) == pytest.approx(SELECTED_ALPHA)


def test_the_training_run_refuses_a_decision_for_another_layer(workspace: Workspace) -> None:
    from causal_self_forecasting.state_audit.run import StateAuditRunError, execute_state_audit_run

    bad = workspace.tmp / "training_l20.yaml"
    bad.write_text(
        json.dumps(_run_config(workspace.model, workspace.task, layer=20)), encoding="utf-8"
    )
    with pytest.raises(StateAuditRunError, match="calibration selected layer"):
        execute_state_audit_run(bad, "fct-train-l20")


def test_the_training_run_refuses_a_ratio_the_decision_did_not_select(
    workspace: Workspace,
) -> None:
    from causal_self_forecasting.state_audit.run import StateAuditRunError, execute_state_audit_run

    bad = workspace.tmp / "training_r10.yaml"
    bad.write_text(
        json.dumps(_run_config(workspace.model, workspace.task, norm_ratio=0.10)), encoding="utf-8"
    )
    with pytest.raises(StateAuditRunError, match="not adjustable"):
        execute_state_audit_run(bad, "fct-train-r10")


# ---------------------------------------------------------------------------
# The final-test clean stage
# ---------------------------------------------------------------------------


def test_the_final_test_clean_stage_applies_no_intervention(workspace: Workspace) -> None:
    run_training(workspace)
    report = run_final_clean(workspace)

    assert report["status"] == "complete"
    assert report["run_role"] == "final_test_unresolved"
    assert report["counts"]["observed_prompts"] == FINAL_N
    assert report["counts"]["observed_states"] == FINAL_N
    assert report["counts"]["clean_forwards"] == FINAL_N
    assert report["counts"]["interventions"] == 0
    assert report["selected_norm_ratio"] == SELECTED_RATIO
    assert report["selected_global_alpha"] == pytest.approx(SELECTED_ALPHA)

    directory = run_dir(FINAL_RUN_ID)
    assert (directory / STATE_AUDIT_CLEAN_PASS).exists()
    assert not (directory / STATE_AUDIT_OBSERVATIONS).exists()

    manifest = json.loads((directory / STATE_AUDIT_RUN_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["intervention_count"] == 0


# ---------------------------------------------------------------------------
# Fitting and committing
# ---------------------------------------------------------------------------


@pytest.fixture
def committed(workspace: Workspace):
    run_training(workspace)
    run_final_clean(workspace)
    return workspace, commit(workspace)


def test_the_commitment_count_is_prompts_times_sixteen(committed) -> None:
    _, report = committed
    assert report["prompts"] == FINAL_N
    assert report["records_per_prompt"] == RECORDS_PER_PROMPT
    assert report["candidates_per_prompt"] == CANDIDATES_PER_PROMPT
    assert report["commitments"] == FINAL_N * RECORDS_PER_PROMPT

    commitments = list(read_jsonl(run_dir(FINAL_RUN_ID) / FORECAST_COMMITMENTS))
    forecasts = list(read_jsonl(run_dir(FINAL_RUN_ID) / FORECASTS))
    assert len(commitments) == FINAL_N * RECORDS_PER_PROMPT
    assert len(forecasts) == FINAL_N * RECORDS_PER_PROMPT
    for forecast in forecasts:
        assert len(forecast["candidate_forecasts"]) == CANDIDATES_PER_PROMPT


def test_every_method_and_condition_is_committed_once_per_prompt(committed) -> None:
    _, report = committed
    by_condition = report["commitments_by_condition"]
    assert by_condition["constant:none"] == FINAL_N
    assert by_condition["prompt_lexical:none"] == FINAL_N
    assert by_condition["intervention_only_ridge:none"] == FINAL_N
    assert by_condition["visible_information_ridge:none"] == FINAL_N
    assert by_condition["state_bilinear_ridge:true"] == FINAL_N
    assert by_condition["state_bilinear_ridge:wrong_example"] == FINAL_N
    assert by_condition["state_bilinear_ridge:shuffled"] == FINAL_N * 10


def test_commitment_keys_are_unique_and_each_has_a_salt(committed) -> None:
    _, report = committed
    verification = report["verification"]
    assert verification["valid"] is True
    assert verification["failures"] == []
    assert verification["commitments"] == FINAL_N * RECORDS_PER_PROMPT
    assert verification["salt_files"] == FINAL_N * RECORDS_PER_PROMPT
    assert verification["reveals"] == 0


def test_no_final_test_outcome_exists_at_the_checkpoint(committed) -> None:
    _, report = committed
    assert report["verification"]["final_test_outcomes_exist"] is False
    assert report["verification"]["outcome_artifacts_present"] == []
    directory = run_dir(FINAL_RUN_ID)
    assert not (directory / STATE_AUDIT_OBSERVATIONS).exists()


def test_the_three_ridges_are_fitted_with_the_preregistered_widths(committed) -> None:
    _, report = committed
    predictors = {entry["method_id"]: entry for entry in report["fit"]["predictors"]}
    assert set(predictors) == {
        "intervention_only_ridge",
        "visible_information_ridge",
        "state_bilinear_ridge",
    }
    assert predictors["intervention_only_ridge"]["feature_dim"] == 16
    assert predictors["visible_information_ridge"]["feature_dim"] == 55
    assert predictors["state_bilinear_ridge"]["feature_dim"] == 327
    for entry in predictors.values():
        assert entry["selected_ridge_alpha"] in [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
        assert entry["training_rows"] == TRAIN_N * 16
        assert entry["coefficient_hash"].startswith("sha256:")
        assert entry["predictor_hash"].startswith("sha256:")


def test_the_transforms_name_only_the_training_prompts(committed) -> None:
    manifest = json.loads(
        (run_dir(TRAIN_RUN_ID) / STATE_AUDIT_RUN_MANIFEST).read_text(encoding="utf-8")
    )
    assert manifest["prompt_role"] == "training"

    records = list(read_jsonl(run_dir(TRAIN_RUN_ID) / STATE_AUDIT_TRANSFORM_FITS))
    assert len(records) == 3
    training_ids = {
        row["variant_id"] for row in read_jsonl(run_dir(TRAIN_RUN_ID) / STATE_AUDIT_CLEAN_PASS)
    }
    final_ids = {
        row["variant_id"] for row in read_jsonl(run_dir(FINAL_RUN_ID) / STATE_AUDIT_CLEAN_PASS)
    }
    for record in records:
        assert record["fit_prompt_role"] == "training"
        assert set(record["fit_prompt_ids"]) == training_ids
        assert set(record["fit_prompt_ids"]).isdisjoint(final_ids)
    assert (run_dir(TRAIN_RUN_ID) / STATE_AUDIT_PREDICTORS).exists()


def test_the_pairing_has_no_self_match_and_ten_derangements(committed) -> None:
    _, report = committed
    pairing = report["pairing"]
    assert pairing["matched_prompts"] == FINAL_N
    assert pairing["self_matches"] == 0
    assert pairing["permutations"] == 10
    assert pairing["pairing_hash"].startswith("sha256:")

    record = json.loads((run_dir(FINAL_RUN_ID) / STATE_AUDIT_PAIRING).read_text(encoding="utf-8"))
    assert record["prompt_role"] == "final_test"
    for match in record["matches"]:
        assert match["variant_id"] != match["donor_variant_id"]
    for permutation in record["permutations"]:
        assert all(key != value for key, value in permutation.items())


def test_the_true_and_matched_conditions_predict_differently(committed) -> None:
    """A substitution that changed nothing would make the control vacuous."""
    _, _report = committed
    forecasts = list(read_jsonl(run_dir(FINAL_RUN_ID) / FORECASTS))
    true_rows = {
        f["trial_id"]: f
        for f in forecasts
        if f["method_id"] == "state_bilinear_ridge" and f["state_condition"] == "true"
    }
    wrong_rows = {
        f["trial_id"]: f
        for f in forecasts
        if f["method_id"] == "state_bilinear_ridge" and f["state_condition"] == "wrong_example"
    }
    assert set(true_rows) == set(wrong_rows)
    differences = 0
    for trial_id, true_row in true_rows.items():
        true_means = [c["delta_margin_mean"] for c in true_row["candidate_forecasts"]]
        wrong_means = [c["delta_margin_mean"] for c in wrong_rows[trial_id]["candidate_forecasts"]]
        if true_means != wrong_means:
            differences += 1
    assert differences == len(true_rows)


def test_the_visible_model_is_identical_across_state_conditions(committed) -> None:
    """It has no state block, so no donor state can reach it. Committed once, at `none`."""
    _, _report = committed
    forecasts = [
        f
        for f in read_jsonl(run_dir(FINAL_RUN_ID) / FORECASTS)
        if f["method_id"] == "visible_information_ridge"
    ]
    assert len(forecasts) == FINAL_N
    assert {f["state_condition"] for f in forecasts} == {"none"}


def test_the_candidate_sets_are_frozen_before_commitment(committed) -> None:
    _, report = committed
    sets = list(read_jsonl(run_dir(FINAL_RUN_ID) / STATE_AUDIT_CANDIDATE_SETS))
    assert len(sets) == FINAL_N
    for candidate_set in sets:
        assert len(candidate_set["candidates"]) == CANDIDATES_PER_PROMPT
        assert sum(1 for c in candidate_set["candidates"] if c["is_noop"]) == 1
        assert candidate_set["norm_ratios"] == [SELECTED_RATIO]

    committed_ids = {
        candidate["intervention_id"]
        for forecast in read_jsonl(run_dir(FINAL_RUN_ID) / FORECASTS)
        for candidate in forecast["candidate_forecasts"]
    }
    frozen_ids = {candidate["candidate_id"] for cs in sets for candidate in cs["candidates"]}
    assert committed_ids == frozen_ids
    assert report["candidate_sets_hash"].startswith("sha256:")


def test_committing_twice_is_refused(committed) -> None:
    from causal_self_forecasting.state_audit.predict import PredictError

    workspace, _ = committed
    with pytest.raises(PredictError, match="already holds commitments"):
        commit(workspace)


def test_verify_commitments_passes_at_the_checkpoint(committed) -> None:
    _, _report = committed
    result = runner.invoke(app, ["state-audit", "verify-commitments", "--run-id", FINAL_RUN_ID])
    assert result.exit_code == 0, _text(result)
    report = json.loads(result.stdout)
    assert report["valid"] is True
    assert report["commitments"] == FINAL_N * RECORDS_PER_PROMPT
    assert report["final_test_outcomes_exist"] is False


def test_committing_is_refused_if_a_final_test_outcome_exists(workspace: Workspace) -> None:
    from causal_self_forecasting.state_audit.predict import PredictError

    run_training(workspace)
    run_final_clean(workspace)
    write_jsonl(run_dir(FINAL_RUN_ID) / STATE_AUDIT_OBSERVATIONS, [{"pretend": "outcome"}])

    with pytest.raises(PredictError, match="already holds outcome artifacts"):
        commit(workspace)


# ---------------------------------------------------------------------------
# Final-test resolution and analysis, end to end on the fixture
# ---------------------------------------------------------------------------


def _pretend_clean_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixture tree is dirty during development; the guard itself is unit-tested."""
    from causal_self_forecasting.state_audit import resolve as resolve_module

    monkeypatch.setattr(
        resolve_module,
        "git_state",
        lambda: {"commit": "0" * 40, "dirty": False, "branch": "test"},
    )


def resolve(workspace: Workspace):
    from causal_self_forecasting.state_audit.resolve import resolve_final_test

    return resolve_final_test(
        workspace.final_config, FINAL_RUN_ID, LAYER, SELECTED_RATIO, SELECTED_ALPHA
    )


@pytest.fixture
def resolved(committed, monkeypatch: pytest.MonkeyPatch):
    workspace, commit_report = committed
    _pretend_clean_tree(monkeypatch)
    return workspace, commit_report, resolve(workspace)


def test_resolution_applies_every_candidate_and_runs_no_clean_forward(resolved) -> None:
    _, _, report = resolved
    counts = report["counts"]
    assert report["status"] == "complete"
    assert report["run_role"] == "final_test_resolved"
    assert counts["observed_prompts"] == FINAL_N
    assert counts["observed_signed_observations"] == FINAL_N * 16
    assert counts["observed_noop_observations"] == FINAL_N
    assert counts["observed_intervened_forwards"] == FINAL_N * CANDIDATES_PER_PROMPT
    assert counts["clean_forwards"] == 0
    assert counts["failures"] == 0
    assert report["layer"] == LAYER
    assert report["norm_ratio"] == SELECTED_RATIO
    assert report["global_alpha"] == pytest.approx(SELECTED_ALPHA)


def test_resolution_reveals_every_commitment_without_selection(resolved) -> None:
    from causal_self_forecasting.trials.commitment import read_reveals

    _, _, report = resolved
    expected = FINAL_N * RECORDS_PER_PROMPT
    assert report["counts"]["reveals"] == expected
    assert report["counts"]["commitments_checked"] == expected
    assert report["commitments_verified"] is True

    reveals = read_reveals(FINAL_RUN_ID)
    assert len(reveals) == expected
    assert all(reveal.no_selection for reveal in reveals)
    assert all(reveal.selected_intervention_id is None for reveal in reveals)
    assert all(reveal.verified for reveal in reveals)


def test_every_commitment_predates_every_outcome_after_resolution(resolved) -> None:
    _, _, report = resolved
    ordering = report["ordering"]
    assert ordering["ordering_valid"] is True
    assert ordering["final_test_outcomes_exist"] is True
    assert ordering["latest_committed_at"] < ordering["earliest_observed_at"]


def test_resolution_integrity_is_exact(resolved) -> None:
    _, _, report = resolved
    integrity = report["integrity"]
    assert integrity["max_abs_noop_target"] == pytest.approx(0.0, abs=1e-3)
    assert integrity["max_abs_noop_delta_norm"] == 0.0
    assert integrity["max_intervention_reconstruction_error"] == pytest.approx(0.0, abs=1e-5)


def test_resolution_refuses_to_rerun_a_completed_test(resolved) -> None:
    from causal_self_forecasting.state_audit.resolve import FinalTestResolutionError

    workspace, _, _ = resolved
    with pytest.raises(FinalTestResolutionError, match="already holds outcome artifacts"):
        resolve(workspace)


def test_resolution_refuses_a_setting_that_differs_from_the_decision(
    committed, monkeypatch: pytest.MonkeyPatch
) -> None:
    from causal_self_forecasting.state_audit.resolve import (
        FinalTestResolutionError,
        resolve_final_test,
    )

    workspace, _ = committed
    _pretend_clean_tree(monkeypatch)
    with pytest.raises(FinalTestResolutionError, match="calibrated alpha"):
        resolve_final_test(workspace.final_config, FINAL_RUN_ID, LAYER, SELECTED_RATIO, 99.0)
    with pytest.raises(FinalTestResolutionError, match="calibrated ratio"):
        resolve_final_test(workspace.final_config, FINAL_RUN_ID, LAYER, 0.10, SELECTED_ALPHA)


def test_resolution_refuses_a_dirty_tree(committed, monkeypatch: pytest.MonkeyPatch) -> None:
    from causal_self_forecasting.state_audit import resolve as resolve_module
    from causal_self_forecasting.state_audit.resolve import (
        FinalTestResolutionError,
        resolve_final_test,
    )

    workspace, _ = committed
    monkeypatch.setattr(
        resolve_module, "git_state", lambda: {"commit": "abc", "dirty": True, "branch": "x"}
    )
    with pytest.raises(FinalTestResolutionError, match="uncommitted changes"):
        resolve_final_test(
            workspace.final_config, FINAL_RUN_ID, LAYER, SELECTED_RATIO, SELECTED_ALPHA
        )
    assert not (run_dir(FINAL_RUN_ID) / "state_audit_observations.jsonl").exists()


@pytest.fixture
def analyzed(resolved):
    from causal_self_forecasting.state_audit.analyze import analyze_final_test

    workspace, _commit_report, resolution = resolved
    return workspace, resolution, analyze_final_test(FINAL_RUN_ID, TRAIN_RUN_ID)


def test_the_analysis_scores_every_condition_prompt_first(analyzed) -> None:
    _, _, report = analyzed
    assert report["prompt_count"] == FINAL_N
    assert report["group_count"] == FINAL_N
    assert len(report["method_summaries"]) == RECORDS_PER_PROMPT
    assert report["scientific_forecast_evaluation"] is True
    assert report["commitments_verified"] is True
    for summary in report["method_summaries"]:
        assert summary["prompt_count"] == FINAL_N
        assert summary["signed_pair_count"] == FINAL_N * 16
        assert summary["noop_pair_count"] == FINAL_N


def test_the_analysis_reports_both_primary_comparisons(analyzed) -> None:
    _, _, report = analyzed
    comparisons = report["primary_comparisons"]
    assert len(comparisons) == 2
    assert comparisons[0]["left"] == "visible_information_ridge:none:0"
    assert comparisons[0]["right"] == "state_bilinear_ridge:true:0"
    assert comparisons[1]["left"] == "state_bilinear_ridge:wrong_example:0"
    assert comparisons[1]["right"] == "state_bilinear_ridge:true:0"
    for comparison in comparisons:
        assert comparison["prompt_count"] == FINAL_N
        assert comparison["group_count"] == FINAL_N
        assert comparison["ci_low"] <= comparison["ci_high"]
        # The decision rule is mechanical: support requires excluding zero.
        assert comparison["supported"] is (
            comparison["excludes_zero"] and comparison["point_estimate"] > 0
        )
        if not comparison["excludes_zero"]:
            assert "No detected difference" in comparison["interpretation"]


def test_the_analysis_uses_the_frozen_bootstrap_settings(analyzed) -> None:
    from causal_self_forecasting.reproducibility import derive_seed

    _, _, report = analyzed
    assert report["bootstrap"]["resamples"] == 10_000
    assert report["bootstrap"]["seed"] == derive_seed("bluedot.bootstrap", 20260727)


def test_the_analysis_writes_tables_and_figures(analyzed) -> None:
    _, _, report = analyzed
    directory = run_dir(FINAL_RUN_ID)
    for name in (
        "state_audit_analysis.json",
        "state_audit_method_summary.json",
        "state_audit_pair_scores.jsonl",
        "state_audit_prompt_scores.jsonl",
    ):
        assert (directory / name).exists(), name

    pairs = list(read_jsonl(directory / "state_audit_pair_scores.jsonl"))
    assert len(pairs) == RECORDS_PER_PROMPT * FINAL_N * CANDIDATES_PER_PROMPT
    prompts = list(read_jsonl(directory / "state_audit_prompt_scores.jsonl"))
    assert len(prompts) == RECORDS_PER_PROMPT * FINAL_N

    figures = sorted((directory / "figures").glob("*.png"))
    assert len(figures) == 2
    assert all(path.stat().st_size > 0 for path in figures)
    assert report["artifacts"]["figures"]


def test_the_permutations_are_a_band_not_ten_tests(analyzed) -> None:
    _, _, report = analyzed
    band = report["permutation_band"]
    assert band["condition_count"] == 10.0
    assert band["min_mae"] <= band["median_mae"] <= band["max_mae"]


def test_the_analysis_omits_brier_below_the_flip_floor(analyzed) -> None:
    _, _, report = analyzed
    for summary in report["method_summaries"]:
        if summary["observed_flip_count"] < summary["brier_min_flips"]:
            assert summary["brier_score"] is None
            assert summary["brier_omitted_reason"]
        else:
            assert summary["brier_score"] is not None


def test_the_analysis_runs_once(analyzed) -> None:
    from causal_self_forecasting.state_audit.analyze import AnalysisError, analyze_final_test

    with pytest.raises(AnalysisError, match="run once"):
        analyze_final_test(FINAL_RUN_ID, TRAIN_RUN_ID)


def test_the_model_free_replay_reproduces_the_stored_analysis(
    analyzed, monkeypatch: pytest.MonkeyPatch
) -> None:
    from causal_self_forecasting.models import loader as loader_module
    from causal_self_forecasting.state_audit.analyze import replay_analysis

    def _refuse(*args, **kwargs):
        raise AssertionError("the replay must not load a model")

    monkeypatch.setattr(loader_module, "load_model", _refuse)

    report = replay_analysis(FINAL_RUN_ID)
    assert report["valid"] is True
    assert report["failures"] == []
    assert report["conditions_checked"] == RECORDS_PER_PROMPT
    assert report["comparisons_checked"] == 2


def test_the_replay_detects_an_edited_analysis(analyzed) -> None:
    from causal_self_forecasting.state_audit.analyze import replay_analysis

    path = run_dir(FINAL_RUN_ID) / "state_audit_analysis.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["method_summaries"][0]["mae"] = 0.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    # The record recomputes its own content hash, so an edited analysis does not even load.
    from causal_self_forecasting.state_audit.analyze import AnalysisError

    with pytest.raises((AnalysisError, ValueError)):
        replay_analysis(FINAL_RUN_ID)


def test_the_replay_detects_edited_outcomes(analyzed) -> None:
    from causal_self_forecasting.state_audit.analyze import replay_analysis

    path = run_dir(FINAL_RUN_ID) / "state_audit_observations.jsonl"
    rows = list(read_jsonl(path))
    rows[0]["run_id"] = "somebody-elses-run"
    write_jsonl(path, rows)

    report = replay_analysis(FINAL_RUN_ID)
    assert report["valid"] is False
    assert any("outcomes" in failure for failure in report["failures"])


def test_the_command_group_lists_the_new_commands() -> None:
    result = runner.invoke(app, ["state-audit", "--help"])
    assert result.exit_code == 0
    for command in (
        "train",
        "projection",
        "final-test-clean",
        "commit-forecasts",
        "verify-commitments",
        "resolve-final-test",
        "analyze-final-test",
        "replay-analysis",
    ):
        assert command in result.stdout


def test_each_stage_command_refuses_another_stages_config(workspace: Workspace) -> None:
    """Every stage guard, checked at the entry point rather than trusted to the operator."""
    from causal_self_forecasting.state_audit.run import (
        StateAuditRunError,
        check_smoke_parameters,
        check_training_parameters,
        load_run_config,
    )

    training = load_run_config(workspace.training_config)
    final = load_run_config(workspace.final_config)

    with pytest.raises(StateAuditRunError, match="engineering smoke only"):
        check_smoke_parameters(training)
    with pytest.raises(StateAuditRunError, match="training stage only"):
        check_training_parameters(final)


def test_the_training_command_requires_an_inherited_strength(workspace: Workspace) -> None:
    from causal_self_forecasting.state_audit.run import (
        StateAuditRunError,
        check_training_parameters,
        load_run_config,
    )

    path = workspace.tmp / "training_no_decision.yaml"
    body = _run_config(workspace.model, workspace.task)
    body.pop("calibration_decision_run_id")
    path.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(StateAuditRunError, match="inherit its strength"):
        check_training_parameters(load_run_config(path))


def test_a_clean_only_run_is_refused_by_the_intervened_verifier(workspace: Workspace) -> None:
    from causal_self_forecasting.state_audit.run import StateAuditRunError
    from causal_self_forecasting.state_audit.verify import verify_run

    run_training(workspace)
    run_final_clean(workspace)
    with pytest.raises(StateAuditRunError, match="applied no intervention"):
        verify_run(FINAL_RUN_ID)


# -- the public replay bundle -------------------------------------------------


@pytest.fixture
def bundled(analyzed, tmp_path: Path):
    from causal_self_forecasting.state_audit.bundle import build_public_bundle

    workspace, resolution, analysis = analyzed
    destination = tmp_path / "public" / "fct-v0.1"
    report = build_public_bundle(
        {"final_test": FINAL_RUN_ID, "training": TRAIN_RUN_ID, "calibration": CAL_RUN_ID},
        bundle_id="fct-v0.1",
        destination=destination,
    )
    return workspace, resolution, analysis, destination, report


def test_the_bundle_replays_on_its_own(bundled) -> None:
    from causal_self_forecasting.state_audit.bundle import replay_bundle

    *_, destination, _ = bundled
    report = replay_bundle(destination)
    assert report["valid"] is True
    assert report["failures"] == []
    assert report["checksums_valid"] is True
    assert report["conditions_checked"] == RECORDS_PER_PROMPT
    assert report["comparisons_checked"] == 2
    assert report["bootstrap_resamples"] == 10_000


def test_the_bundle_replay_matches_the_run_directory_replay(bundled) -> None:
    """The published artifact must not be able to pass a weaker check than the run it came from."""
    from causal_self_forecasting.state_audit.analyze import replay_analysis
    from causal_self_forecasting.state_audit.bundle import replay_bundle

    *_, destination, _ = bundled
    from_bundle = replay_bundle(destination)
    from_run = replay_analysis(FINAL_RUN_ID)
    for field in ("analysis_hash", "conditions_checked", "comparisons_checked", "bootstrap_seed"):
        assert from_bundle[field] == from_run[field], field
    assert from_bundle["valid"] == from_run["valid"] is True


def test_the_bundle_excludes_states_weights_and_pre_reveal_salts(bundled) -> None:
    *_, destination, _ = bundled
    published = {path.name for path in destination.rglob("*") if path.is_file()}
    assert "state_audit_states.npz" not in published
    assert "state_audit_state_refs.jsonl" not in published
    assert not any(name.endswith(".salt") for name in published)
    assert not any(
        name.endswith((".npz", ".npy", ".pt", ".pth", ".ckpt", ".safetensors"))
        for name in published
    )
    assert not (destination / "private_payloads").exists()


def test_the_bundle_publishes_the_reveals_that_carry_the_salts(bundled) -> None:
    """Post-reveal salts must ship: a commitment hash cannot be checked without its salt."""
    *_, destination, _ = bundled
    rows = list(read_jsonl(destination / "selection_reveals.jsonl"))
    assert len(rows) == RECORDS_PER_PROMPT * FINAL_N
    assert all(row["salt_hex"] for row in rows)
    assert all(row["no_selection"] is True for row in rows)
    assert all(row["selected_intervention_id"] is None for row in rows)


def test_every_bundled_forecast_record_covers_all_candidates(bundled) -> None:
    *_, destination, _ = bundled
    rows = list(read_jsonl(destination / "forecasts.jsonl"))
    assert len(rows) == RECORDS_PER_PROMPT * FINAL_N
    for row in rows:
        assert len(row["candidate_forecasts"]) == CANDIDATES_PER_PROMPT


def test_an_edited_bundle_file_fails_the_checksums(bundled) -> None:
    from causal_self_forecasting.state_audit.bundle import replay_bundle, verify_bundle

    *_, destination, _ = bundled
    target = destination / "state_audit_observations.jsonl"
    rows = list(read_jsonl(target))
    rows[0]["delta_clean_top_margin"] = float(rows[0]["delta_clean_top_margin"]) + 1.0
    write_jsonl(target, rows)

    assert verify_bundle(destination)["valid"] is False
    report = replay_bundle(destination)
    assert report["valid"] is False
    assert report["checksums_valid"] is False


def test_a_file_added_to_the_bundle_is_detected(bundled) -> None:
    from causal_self_forecasting.state_audit.bundle import verify_bundle

    *_, destination, _ = bundled
    (destination / "extra_note.txt").write_text("added later", encoding="utf-8")
    report = verify_bundle(destination)
    assert report["valid"] is False
    assert any("extra_note.txt" in failure for failure in report["failures"])


def test_the_bundle_manifest_records_the_setting_and_the_exclusions(bundled) -> None:
    from causal_self_forecasting.hashing import read_json

    *_, destination, _ = bundled
    manifest = read_json(destination / "bundle_manifest.json")
    assert manifest["layer"] == LAYER
    assert manifest["norm_ratio"] == SELECTED_RATIO
    assert manifest["global_alpha"] == SELECTED_ALPHA
    assert manifest["scientific_result"] is True
    assert manifest["file_count"] == len(manifest["files"])
    excluded = {entry["path"] for entry in manifest["excluded"]}
    assert "state_audit_states.npz" in excluded
    assert "private_payloads/salts/" in excluded


def test_the_bundle_refuses_to_build_without_an_analysis(analyzed, tmp_path: Path) -> None:
    from causal_self_forecasting.state_audit.bundle import BundleError, build_public_bundle

    (run_dir(FINAL_RUN_ID) / "state_audit_analysis.json").unlink()
    with pytest.raises(BundleError, match="missing"):
        build_public_bundle(
            {"final_test": FINAL_RUN_ID, "training": TRAIN_RUN_ID, "calibration": CAL_RUN_ID},
            bundle_id="fct-v0.1",
            destination=tmp_path / "public" / "fct-v0.1",
        )
