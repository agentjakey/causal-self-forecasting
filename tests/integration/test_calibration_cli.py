"""CLI tests for the calibration group.

Every command here must work without a model, and the fixture below proves it rather than
assuming it: `load_model` is replaced with a function that raises, so any command that reached
for weights would fail loudly instead of quietly loading them.

The observation artifacts are synthetic. They exercise the summarizer and the selector; they are
not calibration results, and no number in them came from a model.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from causal_self_forecasting.calibration import plan as plan_module
from causal_self_forecasting.cli import app
from causal_self_forecasting.hashing import write_jsonl
from causal_self_forecasting.models import loader as loader_module
from causal_self_forecasting.schemas import PromptRole, StateAuditObservationRecord

runner = CliRunner()

REAL_CONFIG = "configs/calibration/bluedot_state_dependence.yaml"
PLAN_ID = "bluedot_state_dependence_calibration_v1"
RATIOS = (0.02, 0.05, 0.10, 0.20, 0.40)
REFERENCE_NORM = 32.0

CLEAN = {"A": 3.0, "B": 1.0, "C": 0.5, "D": 0.0}


def _text(result) -> str:
    parts = [result.stdout or ""]
    with contextlib.suppress(ValueError):
        parts.append(result.stderr or "")
    return "".join(parts)


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    plans = tmp_path / "calibration_plans"
    plans.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(plan_module, "calibration_plan_path", lambda pid: plans / f"{pid}.json")

    def _refuse(*args, **kwargs):
        raise AssertionError(
            "a calibration command tried to load a model; every S3 command must work from "
            "manifests and supplied artifacts alone"
        )

    monkeypatch.setattr(loader_module, "load_model", _refuse)
    return tmp_path


def _observation(
    prompt_index: int,
    candidate_index: int,
    ratio: float,
    delta: float,
    layer: int = 13,
    is_noop: bool = False,
) -> StateAuditObservationRecord:
    """One synthetic observation with a chosen target.

    `c_star` is A with a clean margin of 2.0, so shifting A by `delta` moves the target by
    exactly `delta` and leaves the arithmetic easy to read.
    """
    intervened = dict(CLEAN) | {"A": CLEAN["A"] + delta}
    return StateAuditObservationRecord(
        study_id="bluedot_state_dependence",
        run_id="fixture-run",
        trial_id=f"trial_{prompt_index:05d}",
        candidate_id=f"trial_{prompt_index:05d}.opaque_{candidate_index:02d}",
        group_id=f"g{prompt_index:04d}",
        variant_id=f"g{prompt_index:04d}.neutral_a",
        prompt_role=PromptRole.CALIBRATION,
        clean_preferred_label="A",
        clean_logits=dict(CLEAN),
        intervened_logits=intervened,
        clean_top_margin=2.0,
        intervened_top_margin=2.0 + delta,
        delta_clean_top_margin=delta,
        answer_flip=(2.0 + delta) < 0.0,
        is_noop=is_noop,
        direction_ref=f"bd1.{candidate_index:016x}",
        layer=layer,
        norm_ratio=0.0 if is_noop else ratio,
        global_alpha=0.0 if is_noop else ratio * REFERENCE_NORM,
        model_id="google/gemma-3-1b-it",
        model_revision="dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        prompt_manifest_hash="sha256:" + "1" * 64,
        direction_family_hash="sha256:" + "2" * 64,
        config_hash="sha256:" + "3" * 64,
    )


def _write_observations(
    path: Path,
    passing_ratios: set[float],
    layer: int = 13,
    prompts: int = 32,
    signed_directions: int = 16,
) -> Path:
    """A full calibration grid: 32 prompts x 16 signed directions x 5 ratios, plus 32 no-ops."""
    records: list[StateAuditObservationRecord] = []
    for ratio in RATIOS:
        # A passing ratio gets effects of 0.5 everywhere: median 0.5, all above 0.10, p95 0.5.
        # A failing ratio gets 0.001, which fails the median floor and the large-effect share.
        delta = 0.5 if ratio in passing_ratios else 0.001
        for prompt in range(prompts):
            for candidate in range(signed_directions):
                sign = 1.0 if candidate % 2 == 0 else -1.0
                records.append(_observation(prompt, candidate, ratio, sign * delta, layer=layer))
    for prompt in range(prompts):
        records.append(_observation(prompt, 99, 0.0, 0.0, layer=layer, is_noop=True))
    write_jsonl(path, records)
    return path


def _plan(workspace: Path) -> None:
    result = runner.invoke(app, ["calibration", "plan", "--config", REAL_CONFIG])
    assert result.exit_code == 0, _text(result)


# ---------------------------------------------------------------------------
# plan and verify-plan
# ---------------------------------------------------------------------------


def test_cli_freezes_the_plan(workspace: Path) -> None:
    result = runner.invoke(app, ["calibration", "plan", "--config", REAL_CONFIG])
    assert result.exit_code == 0, _text(result)

    report = json.loads(result.stdout)
    assert report["status"] == "written"
    assert report["scientific_result"] is False
    assert report["target_name"] == "delta_clean_top_margin"
    assert report["primary_layer"] == 13
    assert report["fallback_layer"] == 20
    assert report["norm_ratios"] == [0.02, 0.05, 0.10, 0.20, 0.40]
    assert report["calibration_prompt_count"] == 32
    assert report["direction_count"] == 8
    assert report["forward_counts"]["primary_total_forwards"] == 5072
    assert report["forward_counts"]["with_fallback_total_forwards"] == 7696
    assert report["verification"]["valid"] is True
    assert "no calibration result exists" in report["notes"]
    assert "No model was loaded" in report["notes"]

    written = workspace / "calibration_plans" / f"{PLAN_ID}.json"
    assert written.exists()


def test_cli_plan_rerun_preserves_bytes_and_mtime(workspace: Path) -> None:
    _plan(workspace)
    written = workspace / "calibration_plans" / f"{PLAN_ID}.json"
    original = written.read_bytes()
    mtime = written.stat().st_mtime_ns

    second = runner.invoke(app, ["calibration", "plan", "--config", REAL_CONFIG])
    assert second.exit_code == 0, _text(second)
    assert json.loads(second.stdout)["status"] == "unchanged"
    assert written.read_bytes() == original
    assert written.stat().st_mtime_ns == mtime


def test_cli_verify_plan(workspace: Path) -> None:
    _plan(workspace)
    result = runner.invoke(app, ["calibration", "verify-plan", "--plan-id", PLAN_ID])
    assert result.exit_code == 0, _text(result)
    report = json.loads(result.stdout)
    assert report["verification"]["valid"] is True
    assert report["plan_hash"].startswith("sha256:")
    assert report["prompt_manifest_hash"].startswith("sha256:")
    assert report["direction_family_hash"].startswith("sha256:")


def test_cli_verify_plan_reports_a_missing_plan(workspace: Path) -> None:
    result = runner.invoke(app, ["calibration", "verify-plan", "--plan-id", "nope"])
    assert result.exit_code == 1
    assert "no calibration plan" in _text(result)


def test_cli_verify_plan_rejects_an_edited_plan(workspace: Path) -> None:
    _plan(workspace)
    written = workspace / "calibration_plans" / f"{PLAN_ID}.json"
    payload = json.loads(written.read_text(encoding="utf-8"))
    payload["noop_tolerance"] = 0.5
    written.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(app, ["calibration", "verify-plan", "--plan-id", PLAN_ID])
    assert result.exit_code == 1
    assert "not a valid calibration plan" in _text(result)


# ---------------------------------------------------------------------------
# summarize and select, on synthetic artifacts
# ---------------------------------------------------------------------------


def test_cli_summarizes_a_fixture_grid(workspace: Path) -> None:
    _plan(workspace)
    observations = _write_observations(workspace / "obs.jsonl", passing_ratios={0.10, 0.20, 0.40})

    result = runner.invoke(
        app,
        [
            "calibration",
            "summarize",
            "--plan-id",
            PLAN_ID,
            "--observations",
            str(observations),
            "--layer",
            "13",
            "--output",
            str(workspace / "summaries.json"),
        ],
    )
    assert result.exit_code == 0, _text(result)
    payload = json.loads(result.stdout)
    assert payload["scientific_result"] is False
    assert payload["observations_read"] == 32 * 16 * 5 + 32
    assert len(payload["summaries"]) == 5
    assert payload["passing_ratios"] == [0.10, 0.20, 0.40]

    first = payload["summaries"][0]
    assert first["norm_ratio"] == 0.02
    assert first["expected_non_noop_observations"] == 512
    assert first["observed_non_noop_observations"] == 512
    assert first["noop_count"] == 32
    assert first["global_alpha"] == pytest.approx(0.02 * REFERENCE_NORM)
    assert first["passed"] is False

    chosen = payload["summaries"][2]
    assert chosen["norm_ratio"] == 0.10
    assert chosen["passed"] is True
    assert chosen["median_abs_effect"] == pytest.approx(0.5)
    assert chosen["max_abs_noop_target"] == 0.0

    assert (workspace / "summaries.json").exists()


def test_cli_selects_the_smallest_passing_ratio(workspace: Path) -> None:
    _plan(workspace)
    observations = _write_observations(workspace / "obs.jsonl", passing_ratios={0.10, 0.40})
    summaries = workspace / "summaries.json"
    assert (
        runner.invoke(
            app,
            [
                "calibration",
                "summarize",
                "--plan-id",
                PLAN_ID,
                "--observations",
                str(observations),
                "--layer",
                "13",
                "--output",
                str(summaries),
            ],
        ).exit_code
        == 0
    )

    result = runner.invoke(
        app, ["calibration", "select", "--plan-id", PLAN_ID, "--summaries", str(summaries)]
    )
    assert result.exit_code == 0, _text(result)
    decision = json.loads(result.stdout)
    assert decision["status"] == "passed_primary"
    assert decision["selected_layer"] == 13
    assert decision["selected_norm_ratio"] == 0.10
    assert decision["selected_global_alpha"] == pytest.approx(0.10 * REFERENCE_NORM)
    assert decision["scientific_result"] is False
    assert decision["decision_hash"].startswith("sha256:")
    assert "smallest passing ratio" in decision["selection_rationale"]


def test_cli_selection_reports_a_required_fallback(workspace: Path) -> None:
    _plan(workspace)
    observations = _write_observations(workspace / "obs.jsonl", passing_ratios=set())
    summaries = workspace / "summaries.json"
    runner.invoke(
        app,
        [
            "calibration",
            "summarize",
            "--plan-id",
            PLAN_ID,
            "--observations",
            str(observations),
            "--layer",
            "13",
            "--output",
            str(summaries),
        ],
    )

    result = runner.invoke(
        app, ["calibration", "select", "--plan-id", PLAN_ID, "--summaries", str(summaries)]
    )
    assert result.exit_code == 0, _text(result)
    decision = json.loads(result.stdout)
    assert decision["status"] == "fallback_required"
    assert decision["selected_layer"] is None


def test_cli_selection_refuses_a_fallback_after_a_passing_primary(workspace: Path) -> None:
    _plan(workspace)
    primary = workspace / "primary.json"
    fallback = workspace / "fallback.json"
    for path, passing, layer in ((primary, {0.05}, 13), (fallback, {0.02}, 20)):
        observations = _write_observations(
            workspace / f"obs_{layer}.jsonl", passing_ratios=passing, layer=layer
        )
        runner.invoke(
            app,
            [
                "calibration",
                "summarize",
                "--plan-id",
                PLAN_ID,
                "--observations",
                str(observations),
                "--layer",
                str(layer),
                "--output",
                str(path),
            ],
        )

    result = runner.invoke(
        app,
        [
            "calibration",
            "select",
            "--plan-id",
            PLAN_ID,
            "--summaries",
            str(primary),
            "--fallback-summaries",
            str(fallback),
        ],
    )
    assert result.exit_code == 1
    assert "must not be calibrated" in _text(result)


def test_cli_selection_fails_when_no_layer_passes(workspace: Path) -> None:
    _plan(workspace)
    primary = workspace / "primary.json"
    fallback = workspace / "fallback.json"
    for path, layer in ((primary, 13), (fallback, 20)):
        observations = _write_observations(
            workspace / f"obs_{layer}.jsonl", passing_ratios=set(), layer=layer
        )
        runner.invoke(
            app,
            [
                "calibration",
                "summarize",
                "--plan-id",
                PLAN_ID,
                "--observations",
                str(observations),
                "--layer",
                str(layer),
                "--output",
                str(path),
            ],
        )

    result = runner.invoke(
        app,
        [
            "calibration",
            "select",
            "--plan-id",
            PLAN_ID,
            "--summaries",
            str(primary),
            "--fallback-summaries",
            str(fallback),
        ],
    )
    assert result.exit_code == 1
    assert "stops under this design" in _text(result)


def test_cli_summarize_refuses_an_unsupported_layer(workspace: Path) -> None:
    _plan(workspace)
    observations = _write_observations(workspace / "obs.jsonl", passing_ratios={0.10})
    result = runner.invoke(
        app,
        [
            "calibration",
            "summarize",
            "--plan-id",
            PLAN_ID,
            "--observations",
            str(observations),
            "--layer",
            "17",
        ],
    )
    assert result.exit_code == 1
    assert "no other layer may be calibrated" in _text(result)


def test_cli_summarize_refuses_a_prompt_specific_alpha(workspace: Path) -> None:
    """The leak the global rule exists to prevent, refused at the artifact boundary."""
    _plan(workspace)
    observations = _write_observations(workspace / "obs.jsonl", passing_ratios={0.10})
    rows = [json.loads(line) for line in observations.read_text(encoding="utf-8").splitlines()]
    # One prompt at ratio 0.02 gets its own alpha, which is exactly the per-prompt strength the
    # global rule forbids.
    for row in rows:
        if not row["is_noop"] and row["norm_ratio"] == 0.02:
            row["global_alpha"] = 99.0
            break
    write_jsonl(workspace / "bad.jsonl", rows)

    result = runner.invoke(
        app,
        [
            "calibration",
            "summarize",
            "--plan-id",
            PLAN_ID,
            "--observations",
            str(workspace / "bad.jsonl"),
            "--layer",
            "13",
        ],
    )
    assert result.exit_code == 1
    assert "more than one alpha" in _text(result)


def test_cli_summarize_rejects_an_observation_whose_target_disagrees(workspace: Path) -> None:
    _plan(workspace)
    record = _observation(0, 0, 0.10, 0.5).model_dump(mode="json")
    record["delta_clean_top_margin"] = 9.0
    write_jsonl(workspace / "bad.jsonl", [record])

    result = runner.invoke(
        app,
        [
            "calibration",
            "summarize",
            "--plan-id",
            PLAN_ID,
            "--observations",
            str(workspace / "bad.jsonl"),
            "--layer",
            "13",
        ],
    )
    assert result.exit_code == 1
    assert "disagrees with its own" in _text(result)


def test_the_calibration_group_is_listed(workspace: Path) -> None:
    result = runner.invoke(app, ["calibration", "--help"])
    assert result.exit_code == 0
    for command in ("plan", "verify-plan", "summarize", "select"):
        assert command in result.stdout
