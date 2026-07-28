"""CLI tests for `csf directions build-family` and `csf directions verify-family`.

These build a real family against the local fixture model, which is randomly initialized and
built on demand. No pretrained weights and no network. Every output directory is redirected
into tmp_path, so the suite cannot create or replace the study's real family.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from causal_self_forecasting.cli import app
from causal_self_forecasting.interventions import direction_family as df
from causal_self_forecasting.interventions.directions import DirectionStore

runner = CliRunner()

FAMILY_ID = "cli_family_v1"


def _text(result) -> str:
    parts = [result.stdout or ""]
    with contextlib.suppress(ValueError):
        parts.append(result.stderr or "")
    return "".join(parts)


def _write_config(tmp_path: Path, family_id: str = FAMILY_ID, **overrides) -> Path:
    body = {
        "name": "cli_direction_family",
        "family_id": family_id,
        "study_id": "cli_study",
        "model_ref": "configs/models/fixture_tiny.yaml",
        "direction_id_prefix": "cf1",
        "answer_labels": ["A", "B", "C", "D"],
        "label_prefix": " ",
        "master_seed": 20260727,
        "random_control_count": 4,
    }
    body.update(overrides)
    path = tmp_path / f"{family_id}.yaml"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    directions = tmp_path / "directions"
    manifests = tmp_path / "direction_manifests"
    for directory in (directions, manifests):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(df, "directions_dir", lambda: directions)
    monkeypatch.setattr(df, "direction_manifest_path", lambda fid: manifests / f"{fid}.json")
    return tmp_path


def test_cli_builds_a_family(workspace: Path) -> None:
    config = _write_config(workspace)
    result = runner.invoke(app, ["directions", "build-family", "--config", str(config)])
    assert result.exit_code == 0, _text(result)

    report = json.loads(result.stdout)
    assert report["status"] == "written"
    assert report["direction_count"] == 8
    assert report["hidden_dim"] == 64
    assert report["answer_span_rank"] == 3
    assert report["output_embedding_source"] == "get_output_embeddings"
    assert report["tied_embeddings"] is False
    assert report["family_hash"].startswith("sha256:")
    assert len(report["opaque_direction_ids"]) == 8
    assert report["opaque_direction_ids"] == sorted(report["opaque_direction_ids"])
    assert report["artifact_verification"]["valid"] is True
    assert set(report["raw_answer_norms"]) == {"A", "B", "C", "D"}

    manifest = workspace / "direction_manifests" / f"{FAMILY_ID}.json"
    assert manifest.exists()
    assert len(list((workspace / "directions").glob("*.npz"))) == 8


def test_cli_rerun_is_unchanged_and_preserves_bytes_and_mtime(workspace: Path) -> None:
    config = _write_config(workspace)
    assert (
        runner.invoke(app, ["directions", "build-family", "--config", str(config)]).exit_code == 0
    )

    manifest = workspace / "direction_manifests" / f"{FAMILY_ID}.json"
    original = manifest.read_bytes()
    mtime = manifest.stat().st_mtime_ns

    second = runner.invoke(app, ["directions", "build-family", "--config", str(config)])
    assert second.exit_code == 0, _text(second)
    assert json.loads(second.stdout)["status"] == "unchanged"
    assert manifest.read_bytes() == original
    assert manifest.stat().st_mtime_ns == mtime


def test_cli_refuses_a_different_family_at_the_same_path(workspace: Path) -> None:
    config = _write_config(workspace)
    assert (
        runner.invoke(app, ["directions", "build-family", "--config", str(config)]).exit_code == 0
    )

    changed = _write_config(workspace, master_seed=999)
    result = runner.invoke(app, ["directions", "build-family", "--config", str(changed)])
    assert result.exit_code == 1
    assert "already holds a different direction family" in _text(result)


def test_cli_refuses_unexpected_token_ids(workspace: Path) -> None:
    config = _write_config(workspace, expected_token_ids={"A": 562, "B": 603, "C": 565, "D": 622})
    result = runner.invoke(app, ["directions", "build-family", "--config", str(config)])
    assert result.exit_code == 1
    assert "differ from the pinned expectations" in _text(result)
    assert not list((workspace / "direction_manifests").iterdir())


def test_cli_verifies_artifacts_without_a_model(workspace: Path) -> None:
    config = _write_config(workspace)
    assert (
        runner.invoke(app, ["directions", "build-family", "--config", str(config)]).exit_code == 0
    )

    result = runner.invoke(app, ["directions", "verify-family", "--manifest-id", FAMILY_ID])
    assert result.exit_code == 0, _text(result)
    report = json.loads(result.stdout)
    assert report["valid"] is True
    assert report["artifact_verification"]["level"] == "artifacts"
    assert report["artifact_verification"]["directions_checked"] == 8
    assert "regeneration_verification" not in report


def test_cli_regeneration_verification(workspace: Path) -> None:
    config = _write_config(workspace)
    assert (
        runner.invoke(app, ["directions", "build-family", "--config", str(config)]).exit_code == 0
    )

    result = runner.invoke(
        app,
        [
            "directions",
            "verify-family",
            "--manifest-id",
            FAMILY_ID,
            "--regenerate",
            "--config",
            str(config),
        ],
    )
    assert result.exit_code == 0, _text(result)
    report = json.loads(result.stdout)
    assert report["valid"] is True
    assert report["regeneration_verification"]["valid"] is True
    assert (
        report["regeneration_verification"]["rebuilt_family_hash"]
        == report["artifact_verification"]["family_hash"]
    )


def test_cli_verify_fails_on_a_missing_vector_artifact(workspace: Path) -> None:
    config = _write_config(workspace)
    build = runner.invoke(app, ["directions", "build-family", "--config", str(config)])
    assert build.exit_code == 0, _text(build)

    missing = json.loads(build.stdout)["opaque_direction_ids"][0]
    DirectionStore(workspace / "directions").path_for(missing).unlink()

    result = runner.invoke(app, ["directions", "verify-family", "--manifest-id", FAMILY_ID])
    assert result.exit_code == 1
    assert "did not verify" in _text(result)
    assert "no stored vector artifact" in _text(result)


def test_cli_verify_reports_a_missing_manifest(workspace: Path) -> None:
    result = runner.invoke(app, ["directions", "verify-family", "--manifest-id", "nope"])
    assert result.exit_code == 1
    assert "no direction family manifest" in _text(result)


def test_cli_regeneration_rejects_a_config_for_another_family(workspace: Path) -> None:
    config = _write_config(workspace)
    assert (
        runner.invoke(app, ["directions", "build-family", "--config", str(config)]).exit_code == 0
    )

    other = _write_config(workspace, family_id="some_other_family")
    result = runner.invoke(
        app,
        [
            "directions",
            "verify-family",
            "--manifest-id",
            FAMILY_ID,
            "--regenerate",
            "--config",
            str(other),
        ],
    )
    assert result.exit_code == 1
    assert "builds family" in _text(result)


def test_synthetic_direction_command_still_exists() -> None:
    """S2 adds commands; it must not remove or rename the existing one."""
    result = runner.invoke(app, ["directions", "--help"])
    assert result.exit_code == 0
    assert "synthetic" in result.stdout
    assert "build-family" in result.stdout
    assert "verify-family" in result.stdout
