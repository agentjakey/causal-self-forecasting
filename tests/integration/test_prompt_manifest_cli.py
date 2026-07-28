"""CLI tests for `csf prompts manifest` and `csf prompts verify`.

Offline and model-free: the command reads prepared task files and writes one JSON artifact.
Every directory is redirected into tmp_path, so running the suite cannot create or replace a
real frozen split.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from causal_self_forecasting.cli import app
from causal_self_forecasting.hashing import atomic_write_json, hash_file, hash_object, write_jsonl
from causal_self_forecasting.schemas import Framing, PromptVariant, Split, TaskItem
from causal_self_forecasting.tasks import loader as task_loader
from causal_self_forecasting.tasks import prompt_manifest as pm

TASK_NAME = "clitask"
WRAPPERS = ("neutral_a", "eval_a", "deploy_a")

runner = CliRunner()


def _text(result) -> str:
    """Stdout plus stderr, across click versions that separate them and ones that do not."""
    parts = [result.stdout or ""]
    with contextlib.suppress(ValueError):
        parts.append(result.stderr or "")
    return "".join(parts)


def _write_pool(processed: Path, manifests: Path, group_count: int) -> None:
    items: list[TaskItem] = []
    variants: list[PromptVariant] = []
    for index in range(group_count):
        group_id = f"c{index:04d}"
        split = (Split.TRAIN, Split.VAL, Split.TEST)[index % 3]
        items.append(
            TaskItem(
                item_id=group_id,
                group_id=group_id,
                source="allenai/ai2_arc",
                source_id=f"src{index}",
                subject="ARC-Challenge",
                question=f"question number {index}",
                choices=["alpha", "beta", "gamma", "delta"],
                answer_index=index % 4,
                split=split,
            )
        )
        for wrapper_id in WRAPPERS:
            variants.append(
                PromptVariant(
                    variant_id=f"{group_id}.{wrapper_id}",
                    item_id=group_id,
                    group_id=group_id,
                    wrapper_id=wrapper_id,
                    framing=Framing.NEUTRAL,
                    prompt_text=f"question number {index} [{wrapper_id}]",
                    answer_labels=["A", "B", "C", "D"],
                    split=split,
                )
            )

    items_path = write_jsonl(processed / TASK_NAME / "items.jsonl", items)
    variants_path = write_jsonl(processed / TASK_NAME / "variants.jsonl", variants)
    atomic_write_json(
        manifests / f"{TASK_NAME}.json",
        {
            "task_name": TASK_NAME,
            "config_path": f"configs/tasks/{TASK_NAME}.yaml",
            "config_hash": hash_object({"name": TASK_NAME}),
            "source": "allenai/ai2_arc",
            "source_config": "ARC-Challenge",
            "source_split": "train",
            "item_count": len(items),
            "variant_count": len(variants),
            "group_count": len(items),
            "item_splits": {"train": len(items)},
            "variant_splits": {"train": len(variants)},
            "wrapper_ids": list(WRAPPERS),
            "heldout_wrapper_ids": [],
            "files": {
                "items": {"path": "items.jsonl", "hash": hash_file(items_path)},
                "variants": {"path": "variants.jsonl", "hash": hash_file(variants_path)},
            },
        },
    )


def _write_configs(tmp_path: Path, manifest_id: str = "cli_v1", seed: int = 20260727) -> Path:
    task_path = tmp_path / "task.yaml"
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
        f"name: {TASK_NAME}\nsource: allenai/ai2_arc\nsource_config: ARC-Challenge\n"
        f"answer_labels: [A, B, C, D]\nlabel_prefix: ' '\nwrappers:\n{wrappers}",
        encoding="utf-8",
    )

    prompts_path = tmp_path / "prompts.yaml"
    prompts_path.write_text(
        f"name: cli_prompt_manifest\nmanifest_id: {manifest_id}\n"
        f"task_ref: {task_path.as_posix()}\ncanonical_wrapper_id: neutral_a\n"
        f"master_seed: {seed}\n"
        "role_counts:\n  smoke: 8\n  calibration: 32\n  training: 96\n  final_test: 32\n",
        encoding="utf-8",
    )
    return prompts_path


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    manifests = tmp_path / "manifests"
    out = tmp_path / "prompt_manifests"
    for directory in (processed, manifests, out):
        directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(task_loader, "processed_dir", lambda: processed)
    monkeypatch.setattr(task_loader, "manifests_dir", lambda: manifests)
    monkeypatch.setattr(pm, "manifests_dir", lambda: manifests)
    monkeypatch.setattr(pm, "prompt_manifest_path", lambda mid: out / f"{mid}.json")

    _write_pool(processed, manifests, 256)
    return tmp_path


def test_cli_freezes_a_manifest(workspace: Path) -> None:
    config = _write_configs(workspace)
    result = runner.invoke(app, ["prompts", "manifest", "--config", str(config)])
    assert result.exit_code == 0, _text(result)

    report = json.loads(result.stdout)
    assert report["status"] == "written"
    assert report["total_prompts"] == 168
    assert report["role_counts"] == {
        "smoke": 8,
        "calibration": 32,
        "training": 96,
        "final_test": 32,
    }
    assert report["canonical_wrapper_id"] == "neutral_a"
    assert report["master_seed"] == 20260727
    assert report["eligible_group_count"] == 256
    assert report["task_artifacts_match"] is True
    assert report["manifest_hash"].startswith("sha256:")
    assert report["task_manifest_hash"].startswith("sha256:")

    written = workspace / "prompt_manifests" / "cli_v1.json"
    assert written.exists()
    assert (
        json.loads(written.read_text(encoding="utf-8"))["manifest_hash"]
        == (report["manifest_hash"])
    )


def test_cli_rerun_is_byte_identical(workspace: Path) -> None:
    config = _write_configs(workspace)
    first = runner.invoke(app, ["prompts", "manifest", "--config", str(config)])
    assert first.exit_code == 0, _text(first)
    written = workspace / "prompt_manifests" / "cli_v1.json"
    original = written.read_bytes()

    second = runner.invoke(app, ["prompts", "manifest", "--config", str(config)])
    assert second.exit_code == 0, _text(second)
    assert json.loads(second.stdout)["status"] == "unchanged"
    assert written.read_bytes() == original


def test_cli_refuses_to_replace_a_different_manifest(workspace: Path) -> None:
    config = _write_configs(workspace)
    assert runner.invoke(app, ["prompts", "manifest", "--config", str(config)]).exit_code == 0

    changed = _write_configs(workspace, seed=999)
    result = runner.invoke(app, ["prompts", "manifest", "--config", str(changed)])
    assert result.exit_code == 1
    assert "already holds a different manifest" in _text(result)


def test_cli_force_replaces_a_different_manifest(workspace: Path) -> None:
    config = _write_configs(workspace)
    runner.invoke(app, ["prompts", "manifest", "--config", str(config)])
    changed = _write_configs(workspace, seed=999)
    result = runner.invoke(app, ["prompts", "manifest", "--config", str(changed), "--force"])
    assert result.exit_code == 0, _text(result)
    assert json.loads(result.stdout)["status"] == "overwritten"


def test_cli_seed_override_changes_the_hash(workspace: Path) -> None:
    config = _write_configs(workspace)
    baseline = runner.invoke(app, ["prompts", "manifest", "--config", str(config)])
    assert baseline.exit_code == 0, _text(baseline)

    other = _write_configs(workspace, manifest_id="cli_v2")
    overridden = runner.invoke(
        app, ["prompts", "manifest", "--config", str(other), "--seed", "424242"]
    )
    assert overridden.exit_code == 0, _text(overridden)
    assert json.loads(overridden.stdout)["master_seed"] == 424242
    assert (
        json.loads(overridden.stdout)["manifest_hash"]
        != json.loads(baseline.stdout)["manifest_hash"]
    )


def test_cli_fails_on_insufficient_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    manifests = tmp_path / "manifests"
    out = tmp_path / "prompt_manifests"
    for directory in (processed, manifests, out):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(task_loader, "processed_dir", lambda: processed)
    monkeypatch.setattr(task_loader, "manifests_dir", lambda: manifests)
    monkeypatch.setattr(pm, "manifests_dir", lambda: manifests)
    monkeypatch.setattr(pm, "prompt_manifest_path", lambda mid: out / f"{mid}.json")

    _write_pool(processed, manifests, 100)
    config = _write_configs(tmp_path)

    result = runner.invoke(app, ["prompts", "manifest", "--config", str(config)])
    assert result.exit_code == 1
    assert "100 eligible groups" in _text(result)
    assert not list(out.iterdir()), "a failed run must not leave a partial manifest behind"


def test_cli_verify_accepts_a_fresh_manifest(workspace: Path) -> None:
    config = _write_configs(workspace)
    assert runner.invoke(app, ["prompts", "manifest", "--config", str(config)]).exit_code == 0

    result = runner.invoke(app, ["prompts", "verify", "--manifest-id", "cli_v1"])
    assert result.exit_code == 0, _text(result)
    report = json.loads(result.stdout)
    assert report["valid"] is True
    assert report["total_prompts"] == 168


def test_cli_verify_rejects_a_manifest_whose_pool_moved(workspace: Path) -> None:
    config = _write_configs(workspace)
    assert runner.invoke(app, ["prompts", "manifest", "--config", str(config)]).exit_code == 0

    _write_pool(workspace / "processed", workspace / "manifests", 255)
    result = runner.invoke(app, ["prompts", "verify", "--manifest-id", "cli_v1"])
    assert result.exit_code == 1
    assert "does not match the prepared task" in _text(result)


def test_cli_verify_rejects_an_edited_manifest(workspace: Path) -> None:
    config = _write_configs(workspace)
    assert runner.invoke(app, ["prompts", "manifest", "--config", str(config)]).exit_code == 0

    written = workspace / "prompt_manifests" / "cli_v1.json"
    payload = json.loads(written.read_text(encoding="utf-8"))
    payload["assignments"][0]["role"] = "final_test"
    written.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(app, ["prompts", "verify", "--manifest-id", "cli_v1"])
    assert result.exit_code == 1
    assert "not a valid prompt manifest" in _text(result)
