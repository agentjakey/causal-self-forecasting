"""Tests for the systems benchmark.

These never download Gemma. The model is the local fixture and the task is a small prepared
dataset written into a temporary directory, so the whole file runs offline in CI.

The access-classification tests drive `check_model_access` with stubbed hub errors rather
than real network calls. What is being tested is that each distinct failure is reported as
itself: telling someone "model load failed" when their license is unaccepted sends them to
debug the wrong thing entirely.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError
from typer.testing import CliRunner

from causal_self_forecasting import benchmark as benchmark_module
from causal_self_forecasting.benchmark import (
    BenchmarkError,
    build_compute_estimate,
    check_model_access,
    read_process_memory,
    run_benchmark,
    select_variants,
    verify_capture_point,
)
from causal_self_forecasting.cli import app
from causal_self_forecasting.config import ModelConfig, TaskConfig, WrapperSpec
from causal_self_forecasting.hashing import write_jsonl
from causal_self_forecasting.models.loader import ModelLoadError, load_model
from causal_self_forecasting.models.scoring import LabelTokenError
from causal_self_forecasting.schemas import (
    AccessStatus,
    BenchmarkClassification,
    BenchmarkRecord,
    Framing,
    PromptVariant,
    PublicDashboardRecord,
    ResultStatus,
    Split,
    TaskItem,
)
from causal_self_forecasting.tasks import loader as task_loader

runner = CliRunner()

GEMMA_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"

# Words the fixture tokenizer knows. A prompt of unknown tokens would exercise nothing.
_QUESTION_WORDS = ["what", "is", "the", "energy", "of", "the", "sun"]


def _fixture_model_config(**overrides: Any) -> ModelConfig:
    base: dict[str, Any] = dict(
        name="fixture_tiny",
        kind="fixture",
        model_id="tiny_llama_bench",
        revision="fixture-v1",
        dtype="float32",
        device="cpu",
        fixture_hidden_size=64,
        fixture_num_layers=4,
    )
    base.update(overrides)
    return ModelConfig(**base)


def _gemma_model_config() -> ModelConfig:
    return ModelConfig(
        name="gemma3_1b_it",
        kind="huggingface",
        model_id="google/gemma-3-1b-it",
        revision=GEMMA_REVISION,
        dtype="float32",
        device="cpu",
    )


def _task_config() -> TaskConfig:
    return TaskConfig(
        name="bench_task",
        source="allenai/ai2_arc",
        source_config="ARC-Challenge",
        source_split="train",
        answer_labels=["A", "B", "C", "D"],
        label_prefix=" ",
        wrappers=[
            WrapperSpec(
                wrapper_id="eval_a",
                framing=Framing.EVALUATION,
                template="this is a test\n{question}\n{choices}\nanswer:",
            ),
            WrapperSpec(
                wrapper_id="deploy_a",
                framing=Framing.DEPLOYMENT,
                template="normal application environment\n{question}\n{choices}\nanswer:",
            ),
            WrapperSpec(
                wrapper_id="neutral_a",
                framing=Framing.NEUTRAL,
                template="answer the question\n{question}\n{choices}\nanswer:",
            ),
        ],
    )


@pytest.fixture
def prepared_task(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TaskConfig:
    """Write a small prepared task to disk, using only fixture-tokenizer vocabulary."""
    config = _task_config()
    processed = tmp_path / "processed"
    manifests = tmp_path / "manifests"
    # Patched on the loader module, not on `paths`. The loader imports these names directly,
    # so they are bound in its namespace at import time and patching `paths` would not reach
    # the call sites.
    monkeypatch.setattr(task_loader, "processed_dir", lambda: processed)
    monkeypatch.setattr(task_loader, "manifests_dir", lambda: manifests)

    items: list[TaskItem] = []
    variants: list[PromptVariant] = []
    for index in range(6):
        item = TaskItem(
            item_id=f"q{index}",
            group_id=f"q{index}",
            source="allenai/ai2_arc",
            source_id=f"src{index}",
            subject="ARC-Challenge",
            question=" ".join(_QUESTION_WORDS),
            choices=["water", "energy", "light", "heat"],
            answer_index=index % 4,
            split=Split.TEST if index % 2 == 0 else Split.TRAIN,
        )
        items.append(item)
        for wrapper in config.wrappers:
            variants.append(
                PromptVariant(
                    variant_id=f"{item.item_id}.{wrapper.wrapper_id}",
                    item_id=item.item_id,
                    group_id=item.group_id,
                    wrapper_id=wrapper.wrapper_id,
                    framing=wrapper.framing,
                    prompt_text=wrapper.template.format(
                        question=item.question,
                        choices="\n".join(
                            f"{label} . {choice}"
                            for label, choice in zip(
                                ["A", "B", "C", "D"], item.choices, strict=True
                            )
                        ),
                    ),
                    answer_labels=["A", "B", "C", "D"],
                    split=item.split,
                )
            )

    write_jsonl(processed / config.name / "items.jsonl", items)
    write_jsonl(processed / config.name / "variants.jsonl", variants)
    (manifests / f"{config.name}.json").parent.mkdir(parents=True, exist_ok=True)
    (manifests / f"{config.name}.json").write_text(
        json.dumps({"task_name": config.name, "item_count": len(items)}), encoding="utf-8"
    )
    return config


@pytest.fixture
def config_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Config files on disk, since the benchmark hashes them for provenance."""
    model_path = tmp_path / "model.yaml"
    task_path = tmp_path / "task.yaml"
    model_path.write_text("name: fixture_tiny\nkind: fixture\n", encoding="utf-8")
    task_path.write_text("name: bench_task\n", encoding="utf-8")
    return model_path, task_path


def _run(
    prepared: TaskConfig,
    config_paths: tuple[Path, Path],
    output_dir: Path,
    **overrides: Any,
) -> dict[str, Any]:
    model_path, task_path = config_paths
    settings: dict[str, Any] = dict(
        model_config=_fixture_model_config(),
        task_config=prepared,
        model_config_path=model_path,
        task_config_path=task_path,
        split=Split.TEST,
        max_items=2,
        warmup_runs=1,
        timed_runs=2,
        capture_layer=None,
        seed=12345,
        output_dir=output_dir,
        offline=False,
        force=False,
    )
    settings.update(overrides)
    return run_benchmark(**settings)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_benchmark_help_lists_the_supported_options() -> None:
    result = runner.invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0
    for option in (
        "--model-config",
        "--task-config",
        "--split",
        "--max-items",
        "--warmup-runs",
        "--timed-runs",
        "--capture-layer",
        "--seed",
        "--output-dir",
        "--offline",
        "--force",
    ):
        assert option in result.stdout


def test_benchmark_is_a_top_level_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "benchmark" in result.stdout


# ---------------------------------------------------------------------------
# A successful fixture benchmark
# ---------------------------------------------------------------------------


def test_fixture_benchmark_completes(prepared_task, config_paths, tmp_path: Path) -> None:
    report = _run(prepared_task, config_paths, tmp_path / "out")
    assert report["status"] == "completed"
    assert report["evaluation"]["scored_items"] == 2


def test_fixture_benchmark_writes_a_valid_record(
    prepared_task, config_paths, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    _run(prepared_task, config_paths, out)
    payload = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
    # Round-trips through the schema, so the artifact on disk is not merely dict-shaped.
    record = BenchmarkRecord.model_validate(payload)
    assert record.classification is BenchmarkClassification.FIXTURE_SYSTEMS_TEST


def test_fixture_run_is_marked_fixture_only(prepared_task, config_paths, tmp_path: Path) -> None:
    report = _run(prepared_task, config_paths, tmp_path / "out")
    assert report["fixture_only"] is True
    assert report["classification"] == "fixture_systems_test"
    assert report["scientific_result"] is False


def test_scientific_result_is_always_false(prepared_task, config_paths, tmp_path: Path) -> None:
    report = _run(prepared_task, config_paths, tmp_path / "out")
    assert report["scientific_result"] is False


def test_a_record_cannot_claim_to_be_scientific(
    prepared_task, config_paths, tmp_path: Path
) -> None:
    """Typed as Literal[False], so this is unconstructible rather than merely discouraged."""
    out = tmp_path / "out"
    _run(prepared_task, config_paths, out)
    payload = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
    payload["scientific_result"] = True
    with pytest.raises(ValidationError):
        BenchmarkRecord.model_validate(payload)


def test_a_real_run_is_not_marked_fixture(prepared_task, config_paths, tmp_path: Path) -> None:
    """The non-fixture labeling path, exercised without loading Gemma."""
    out = tmp_path / "out"
    _run(prepared_task, config_paths, out)
    payload = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
    payload["classification"] = "systems_benchmark"
    payload["fixture_only"] = False
    record = BenchmarkRecord.model_validate(payload)
    assert record.fixture_only is False
    assert record.scientific_result is False


def test_classification_and_fixture_flag_must_agree(
    prepared_task, config_paths, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    _run(prepared_task, config_paths, out)
    payload = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
    payload["classification"] = "systems_benchmark"  # left fixture_only True
    with pytest.raises(ValidationError, match="disagree"):
        BenchmarkRecord.model_validate(payload)


def test_benchmark_writes_item_level_results_privately(
    prepared_task, config_paths, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    _run(prepared_task, config_paths, out)
    items_file = out / "private_payloads" / "benchmark_items.jsonl"
    assert items_file.exists()
    rows = [json.loads(line) for line in items_file.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert sorted(rows[0]["label_logits"]) == ["A", "B", "C", "D"]
    assert rows[0]["expected_label"] in "ABCD"


def test_benchmark_records_all_four_label_logits(
    prepared_task, config_paths, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    _run(prepared_task, config_paths, out)
    payload = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
    assert sorted(payload["task"]["label_token_ids"]) == ["A", "B", "C", "D"]


def test_benchmark_writes_provenance_and_manifest(
    prepared_task, config_paths, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    _run(prepared_task, config_paths, out)
    for name in ("benchmark.json", "environment.json", "run_manifest.json", "artifact_hashes.json"):
        assert (out / name).exists(), name
    hashes = json.loads((out / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert all(entry["hash"].startswith("sha256:") for entry in hashes)


def test_run_manifest_marks_the_run_non_scientific(
    prepared_task, config_paths, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    _run(prepared_task, config_paths, out)
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["phase"] == "systems_benchmark"
    assert "not a CSF-Bench result" in (manifest["notes"] or "")


# ---------------------------------------------------------------------------
# Determinism and selection
# ---------------------------------------------------------------------------


def _variants(count: int = 8) -> list[PromptVariant]:
    return [
        PromptVariant(
            variant_id=f"q{index:02d}.neutral_a",
            item_id=f"q{index:02d}",
            group_id=f"q{index:02d}",
            wrapper_id="neutral_a",
            framing=Framing.NEUTRAL,
            prompt_text="text",
            answer_labels=["A", "B", "C", "D"],
            split=Split.TEST,
        )
        for index in range(count)
    ]


def test_item_selection_is_deterministic() -> None:
    first = select_variants(_variants(), Split.TEST, 3, seed=1, task_name="t")
    second = select_variants(_variants(), Split.TEST, 3, seed=1, task_name="t")
    assert [v.variant_id for v in first] == [v.variant_id for v in second]


def test_item_selection_changes_with_the_seed() -> None:
    first = select_variants(_variants(20), Split.TEST, 3, seed=1, task_name="t")
    second = select_variants(_variants(20), Split.TEST, 3, seed=2, task_name="t")
    assert [v.variant_id for v in first] != [v.variant_id for v in second]


def test_item_selection_respects_the_split() -> None:
    mixed = [
        *_variants(4),
        PromptVariant(
            variant_id="train01.neutral_a",
            item_id="train01",
            group_id="train01",
            wrapper_id="neutral_a",
            framing=Framing.NEUTRAL,
            prompt_text="text",
            answer_labels=["A", "B", "C", "D"],
            split=Split.TRAIN,
        ),
    ]
    selected = select_variants(mixed, Split.TEST, 10, seed=1, task_name="t")
    assert all(variant.split is Split.TEST for variant in selected)


def test_item_selection_does_not_truncate_a_sorted_list() -> None:
    """Truncation would take every wrapper of the first question instead of a spread."""
    selected = select_variants(_variants(20), Split.TEST, 3, seed=7, task_name="t")
    ids = [variant.variant_id for variant in selected]
    assert ids != [v.variant_id for v in _variants(20)[:3]]


def test_empty_split_is_reported_clearly() -> None:
    with pytest.raises(BenchmarkError, match="no prepared variants"):
        select_variants(_variants(), Split.VAL, 1, seed=1, task_name="t")


def test_benchmark_is_reproducible(prepared_task, config_paths, tmp_path: Path) -> None:
    first = _run(prepared_task, config_paths, tmp_path / "a")
    second = _run(prepared_task, config_paths, tmp_path / "b")
    assert first["evaluation"] == second["evaluation"]
    assert first["capture"]["shape"] == second["capture"]["shape"]


# ---------------------------------------------------------------------------
# Output collisions
# ---------------------------------------------------------------------------


def test_existing_output_directory_is_not_overwritten(
    prepared_task, config_paths, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    _run(prepared_task, config_paths, out)
    with pytest.raises(BenchmarkError, match="--force"):
        _run(prepared_task, config_paths, out)


def test_force_overwrites_an_existing_directory(
    prepared_task, config_paths, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    _run(prepared_task, config_paths, out)
    report = _run(prepared_task, config_paths, out, force=True)
    assert report["status"] == "completed"


def test_an_empty_output_directory_is_not_a_collision(
    prepared_task, config_paths, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    assert _run(prepared_task, config_paths, out)["status"] == "completed"


# ---------------------------------------------------------------------------
# Capture verification
# ---------------------------------------------------------------------------


def test_capture_uses_the_hook_owned_path(prepared_task, config_paths, tmp_path: Path) -> None:
    """The benchmark must not reintroduce output_hidden_states.

    Pinned because the hook-owned path exists to fix a bug where the framework's own
    hidden-state recording returned a value from a different point than the intervention.
    """
    source = Path(benchmark_module.__file__).read_text(encoding="utf-8")
    assert "output_hidden_states" not in source
    assert "capture_hidden_states" in source
    assert "run_with_intervention" in source


def test_capture_point_is_verified(prepared_task, config_paths, tmp_path: Path) -> None:
    report = _run(prepared_task, config_paths, tmp_path / "out")
    assert report["capture"]["capture_point_verified"] is True
    assert report["capture"]["hook_fired"] is True
    assert report["capture"]["max_abs_patch_error"] < 1e-4


def test_capture_records_shape_and_dtype(prepared_task, config_paths, tmp_path: Path) -> None:
    report = _run(prepared_task, config_paths, tmp_path / "out")
    assert report["capture"]["shape"] == [64]
    assert report["capture"]["dtype"] == "float32"


def test_capture_defaults_to_the_middle_layer(prepared_task, config_paths, tmp_path: Path) -> None:
    report = _run(prepared_task, config_paths, tmp_path / "out")
    assert report["capture"]["layer"] == 2


@pytest.mark.parametrize("layer", [0, 1, 4])
def test_capture_hook_fires_at_the_requested_layer(layer: int) -> None:
    model = load_model(_fixture_model_config(model_id="tiny_llama_layercheck"))
    record, seconds = verify_capture_point(model, "what is the energy of the sun", layer)
    assert record.layer == layer
    assert record.hook_fired
    assert record.capture_point_verified
    assert seconds >= 0.0


def test_capture_rejects_an_out_of_range_layer(prepared_task, config_paths, tmp_path: Path) -> None:
    from causal_self_forecasting.models.capture import CaptureError

    with pytest.raises(CaptureError, match="out of range"):
        _run(prepared_task, config_paths, tmp_path / "out", capture_layer=99)


# ---------------------------------------------------------------------------
# Answer-label validation
# ---------------------------------------------------------------------------


def test_unscoreable_answer_labels_fail_loudly(prepared_task, config_paths, tmp_path: Path) -> None:
    """Never silently switch to generated text when labels cannot be scored."""
    broken = prepared_task.model_copy(update={"answer_labels": ["A B", "B", "C", "D"]})
    with pytest.raises(LabelTokenError, match="single token"):
        _run(prepared_task, config_paths, tmp_path / "out", task_config=broken)


def test_label_token_ids_are_recorded(prepared_task, config_paths, tmp_path: Path) -> None:
    report = _run(prepared_task, config_paths, tmp_path / "out")
    ids = report["task"]["label_token_ids"]
    assert len(set(ids.values())) == 4


def test_scoring_format_is_recorded(prepared_task, config_paths, tmp_path: Path) -> None:
    report = _run(prepared_task, config_paths, tmp_path / "out")
    assert report["task"]["scoring_format"] == benchmark_module.SCORING_FORMAT


# ---------------------------------------------------------------------------
# Timing and memory
# ---------------------------------------------------------------------------


def test_timing_fields_are_finite_and_non_negative(
    prepared_task, config_paths, tmp_path: Path
) -> None:
    timing = _run(prepared_task, config_paths, tmp_path / "out")["timing"]
    for field in (
        "model_load_seconds",
        "tokenization_seconds_total",
        "forward_seconds_median",
        "forward_seconds_p90",
        "forward_seconds_min",
        "forward_seconds_max",
        "evaluation_seconds_total",
    ):
        value = timing[field]
        assert isinstance(value, float)
        assert value == value  # not NaN
        assert value >= 0.0
        assert value != float("inf")


def test_timing_ordering_is_coherent(prepared_task, config_paths, tmp_path: Path) -> None:
    timing = _run(prepared_task, config_paths, tmp_path / "out")["timing"]
    assert timing["forward_seconds_min"] <= timing["forward_seconds_median"]
    assert timing["forward_seconds_median"] <= timing["forward_seconds_max"]
    assert timing["forward_seconds_p90"] <= timing["forward_seconds_max"]


def test_model_load_time_is_not_counted_as_forward_latency(
    prepared_task, config_paths, tmp_path: Path
) -> None:
    timing = _run(prepared_task, config_paths, tmp_path / "out")["timing"]
    assert timing["model_load_seconds"] != timing["forward_seconds_median"]


def test_timing_records_run_counts(prepared_task, config_paths, tmp_path: Path) -> None:
    timing = _run(prepared_task, config_paths, tmp_path / "out", warmup_runs=2, timed_runs=3)[
        "timing"
    ]
    assert timing["warmup_runs"] == 2
    assert timing["timed_runs"] == 3
    assert timing["total_timed_tokens"] == timing["representative_prompt_tokens"] * 3


def test_memory_is_measured_or_explicitly_null(prepared_task, config_paths, tmp_path: Path) -> None:
    memory = _run(prepared_task, config_paths, tmp_path / "out")["memory"]
    assert memory["measurement"]
    for field in ("rss_bytes_after_load", "rss_bytes_peak"):
        value = memory[field]
        assert value is None or (isinstance(value, int) and value > 0)


def test_read_process_memory_reports_a_reason_when_unavailable() -> None:
    current, peak, note = read_process_memory()
    assert isinstance(note, str) and note
    if current is None:
        assert peak is None
    else:
        assert current > 0


# ---------------------------------------------------------------------------
# Access classification
# ---------------------------------------------------------------------------


def _hub_error(error_type: type[Exception], status: int) -> Exception:
    """Build a hub error the way the hub really builds one.

    These carry a real HTTP response, so constructing them by hand needs one too. Faking the
    exception type without its response would test a shape the library never raises.
    """
    import requests

    response = requests.Response()
    response.status_code = status
    response.url = "https://huggingface.co/google/gemma-3-1b-it"
    response.request = requests.Request(method="GET", url=response.url).prepare()
    return error_type("stubbed hub error", response=response)  # type: ignore[call-arg]


def _stub_hub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    error: Exception | None = None,
    token: str | None = "present",
    sha: str = GEMMA_REVISION,
    cached: bool = False,
    sibling_size: int | None = 2_000_000_000,
    free_bytes: int = 500_000_000_000,
) -> None:
    """Stand in for the hub so classification is tested without a network call."""
    import huggingface_hub

    class _Sibling:
        rfilename = "model.safetensors"
        size = sibling_size

    class _Info:
        def __init__(self) -> None:
            self.sha = sha
            self.siblings = [_Sibling()]

    class _Api:
        def model_info(self, *args: Any, **kwargs: Any) -> Any:
            if error is not None:
                raise error
            return _Info()

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: _Api())
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: token)
    monkeypatch.setattr(benchmark_module, "_weights_cached", lambda *a, **k: cached)
    monkeypatch.setattr(
        benchmark_module.shutil,
        "disk_usage",
        lambda *a, **k: type("U", (), {"free": free_bytes, "total": 0, "used": 0})(),
    )


def test_fixture_needs_no_hub_access() -> None:
    check = check_model_access(_fixture_model_config(), offline=False)
    assert check.status is AccessStatus.FIXTURE_LOCAL


def test_missing_authentication_is_classified(monkeypatch: pytest.MonkeyPatch) -> None:
    from huggingface_hub.errors import GatedRepoError

    _stub_hub(monkeypatch, error=_hub_error(GatedRepoError, 403), token=None)
    check = check_model_access(_gemma_model_config(), offline=False)
    assert check.status is AccessStatus.NO_AUTHENTICATION
    assert "hf auth login" in check.detail


def test_missing_gated_access_is_classified(monkeypatch: pytest.MonkeyPatch) -> None:
    """Authenticated but unlicensed is a different problem from unauthenticated."""
    from huggingface_hub.errors import GatedRepoError

    _stub_hub(monkeypatch, error=_hub_error(GatedRepoError, 403), token="present")
    check = check_model_access(_gemma_model_config(), offline=False)
    assert check.status is AccessStatus.GATED_ACCESS_DENIED
    assert "Accept the license" in check.detail


def test_unauthenticated_repository_not_found_reads_as_missing_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hub returns 401 for gated repos, so this is indistinguishable without a token."""
    from huggingface_hub.errors import RepositoryNotFoundError

    _stub_hub(monkeypatch, error=_hub_error(RepositoryNotFoundError, 401), token=None)
    assert check_model_access(_gemma_model_config(), offline=False).status is (
        AccessStatus.NO_AUTHENTICATION
    )


def test_unknown_revision_is_classified(monkeypatch: pytest.MonkeyPatch) -> None:
    from huggingface_hub.errors import RevisionNotFoundError

    _stub_hub(monkeypatch, error=_hub_error(RevisionNotFoundError, 404))
    check = check_model_access(_gemma_model_config(), offline=False)
    assert check.status is AccessStatus.REVISION_UNAVAILABLE


def test_a_revision_that_resolves_elsewhere_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never silently run a different revision than the pinned one."""
    _stub_hub(monkeypatch, sha="0" * 40)
    check = check_model_access(_gemma_model_config(), offline=False)
    assert check.status is AccessStatus.REVISION_UNAVAILABLE
    assert "refusing" in check.detail


def test_network_failure_is_classified(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_hub(monkeypatch, error=OSError("connection reset"))
    assert check_model_access(_gemma_model_config(), offline=False).status is (
        AccessStatus.NETWORK_FAILURE
    )


def test_httpx_transport_failure_is_classified(monkeypatch: pytest.MonkeyPatch) -> None:
    """httpx errors are not OSError, and huggingface_hub 1.x raises them.

    Pinned after a real dropped connection escaped as a raw traceback: the except clause
    only listed OSError, so the one failure most likely to happen mid-download was the one
    it could not report.
    """
    import httpx

    _stub_hub(monkeypatch, error=httpx.ConnectError("connection forcibly closed"))
    check = check_model_access(_gemma_model_config(), offline=False)
    assert check.status is AccessStatus.NETWORK_FAILURE
    assert "ConnectError" in check.detail


def test_a_dropped_download_reads_as_a_network_failure(
    monkeypatch: pytest.MonkeyPatch, prepared_task, config_paths, tmp_path: Path
) -> None:
    """A connection lost mid-download is not a broken model."""
    import httpx

    _stub_hub(monkeypatch, cached=False)

    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise ModelLoadError("could not load google/gemma-3-1b-it") from httpx.ConnectError(
            "connection forcibly closed"
        )

    monkeypatch.setattr(benchmark_module, "load_model", _fail)
    with pytest.raises(BenchmarkError) as caught:
        _run(prepared_task, config_paths, tmp_path / "out", model_config=_gemma_model_config())
    assert caught.value.status is AccessStatus.NETWORK_FAILURE


def test_a_genuine_load_failure_is_not_called_a_network_failure(
    monkeypatch: pytest.MonkeyPatch, prepared_task, config_paths, tmp_path: Path
) -> None:
    _stub_hub(monkeypatch, cached=True)

    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise ModelLoadError("unsupported architecture")

    monkeypatch.setattr(benchmark_module, "load_model", _fail)
    with pytest.raises(BenchmarkError) as caught:
        _run(prepared_task, config_paths, tmp_path / "out", model_config=_gemma_model_config())
    assert caught.value.status is AccessStatus.MODEL_LOAD_FAILURE


def test_insufficient_disk_space_is_classified(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_hub(monkeypatch, cached=False, sibling_size=10_000_000_000, free_bytes=1_000_000)
    check = check_model_access(_gemma_model_config(), offline=False)
    assert check.status is AccessStatus.INSUFFICIENT_DISK_SPACE
    assert check.free_bytes == 1_000_000


def test_cached_weights_are_classified_as_a_cached_load(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_hub(monkeypatch, cached=True)
    assert check_model_access(_gemma_model_config(), offline=False).status is (
        AccessStatus.CACHED_LOAD
    )


def test_uncached_weights_are_classified_as_a_download(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_hub(monkeypatch, cached=False)
    assert check_model_access(_gemma_model_config(), offline=False).status is (
        AccessStatus.REMOTE_DOWNLOAD
    )


def test_offline_cache_miss_is_classified(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_hub(monkeypatch, cached=False)
    check = check_model_access(_gemma_model_config(), offline=True)
    assert check.status is AccessStatus.OFFLINE_CACHE_MISS
    assert "--offline" in check.detail


def test_offline_uses_the_cache_without_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("offline mode must not contact the hub")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _explode)
    monkeypatch.setattr(benchmark_module, "_weights_cached", lambda *a, **k: True)
    check = check_model_access(_gemma_model_config(), offline=True)
    assert check.status is AccessStatus.CACHED_LOAD


def test_offline_cache_miss_fails_the_run(
    monkeypatch: pytest.MonkeyPatch, prepared_task, config_paths, tmp_path: Path
) -> None:
    _stub_hub(monkeypatch, cached=False)
    with pytest.raises(BenchmarkError) as caught:
        _run(
            prepared_task,
            config_paths,
            tmp_path / "out",
            model_config=_gemma_model_config(),
            offline=True,
        )
    assert caught.value.status is AccessStatus.OFFLINE_CACHE_MISS


def test_gated_access_stops_the_run_before_loading(
    monkeypatch: pytest.MonkeyPatch, prepared_task, config_paths, tmp_path: Path
) -> None:
    """A licensing problem must not surface as a loader traceback."""
    from huggingface_hub.errors import GatedRepoError

    _stub_hub(monkeypatch, error=_hub_error(GatedRepoError, 403), token="present")

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the model must not be loaded when access was denied")

    monkeypatch.setattr(benchmark_module, "load_model", _explode)
    with pytest.raises(BenchmarkError) as caught:
        _run(prepared_task, config_paths, tmp_path / "out", model_config=_gemma_model_config())
    assert caught.value.status is AccessStatus.GATED_ACCESS_DENIED


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


def test_no_token_appears_in_artifacts_or_logs(
    monkeypatch: pytest.MonkeyPatch, prepared_task, config_paths, tmp_path: Path, caplog
) -> None:
    """A token must never reach an artifact, a log line, or stdout."""
    secret = "hf_thisisafaketokenvalue123456"
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: secret)
    monkeypatch.setenv("HF_TOKEN", secret)

    out = tmp_path / "out"
    with caplog.at_level("DEBUG"):
        _run(prepared_task, config_paths, out)

    for path in out.rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(encoding="utf-8", errors="ignore"), path
    assert secret not in caplog.text


def test_benchmark_source_never_reads_a_token_value() -> None:
    source = Path(benchmark_module.__file__).read_text(encoding="utf-8")
    # Presence is all that is ever needed, so the value is never bound to a name.
    assert "get_token() is not None" in source


# ---------------------------------------------------------------------------
# Compute estimate
# ---------------------------------------------------------------------------


def test_compute_estimate_follows_the_documented_formula() -> None:
    estimate = build_compute_estimate(0.5)
    expected = 2000 * 2 * 4 * 1
    assert estimate.estimated_forward_count == expected
    assert estimate.estimated_cpu_seconds == pytest.approx(expected * 0.5)
    assert estimate.estimated_cpu_hours == pytest.approx(expected * 0.5 / 3600)


def test_compute_estimate_carries_its_assumptions() -> None:
    estimate = build_compute_estimate(0.5)
    for key in (
        "number_of_prompts",
        "model_states_per_prompt",
        "candidates_per_trial",
        "forward_passes_per_candidate",
        "median_forward_seconds",
        "formula",
    ):
        assert key in estimate.assumptions


def test_a_fast_model_is_judged_practical_on_cpu() -> None:
    estimate = build_compute_estimate(0.001)
    assert estimate.full_sweep_practical_on_cpu
    assert estimate.small_clean_validation_practical_on_cpu
    assert not estimate.gpu_rental_recommended


def test_a_slow_model_recommends_a_gpu() -> None:
    estimate = build_compute_estimate(10.0)
    assert not estimate.full_sweep_practical_on_cpu
    assert not estimate.lora_training_practical_on_cpu
    assert estimate.gpu_rental_recommended


def test_benchmark_includes_a_compute_estimate(prepared_task, config_paths, tmp_path: Path) -> None:
    report = _run(prepared_task, config_paths, tmp_path / "out")
    assert report["compute_estimate"]["estimated_forward_count"] > 0
    assert report["compute_estimate"]["notes"]


# ---------------------------------------------------------------------------
# The benchmark is not a scientific result
# ---------------------------------------------------------------------------


def test_verify_run_does_not_verify_a_benchmark_run(
    prepared_task, config_paths, tmp_path: Path, isolated_runs: Path
) -> None:
    """A benchmark directory has no commitments, so verification must refuse it."""
    from causal_self_forecasting.trials.commitment import verify_run_commitments

    out = isolated_runs / "benchmark-run"
    _run(prepared_task, config_paths, out)
    report = verify_run_commitments("benchmark-run")
    assert report["verified"] is False
    assert report["checked"] == 0


def test_public_exporter_refuses_a_benchmark_run(
    prepared_task, config_paths, tmp_path: Path
) -> None:
    """A systems benchmark can never be dressed up as a published result."""
    out = tmp_path / "out"
    report = _run(prepared_task, config_paths, out)
    with pytest.raises(ValidationError, match="did not verify"):
        PublicDashboardRecord(
            run_id=report["run_id"],
            status=ResultStatus.PRELIMINARY,
            commitments_verified=False,
        )


def test_benchmark_carries_its_limitations(prepared_task, config_paths, tmp_path: Path) -> None:
    report = _run(prepared_task, config_paths, tmp_path / "out")
    joined = " ".join(report["limitations"])
    assert "machine-specific" in joined
    assert "not a causal self-forecasting experiment" in joined


def test_benchmark_records_the_pinned_revision(prepared_task, config_paths, tmp_path: Path) -> None:
    report = _run(prepared_task, config_paths, tmp_path / "out")
    assert report["model"]["revision"] == "fixture-v1"
    assert report["model"]["tokenizer_revision"] == report["model"]["revision"]


def test_gemma_config_revision_is_unchanged() -> None:
    """The pinned revision is provenance. Nothing may quietly move it."""
    from causal_self_forecasting.config import load_config, repo_root

    config = load_config(repo_root() / "configs/models/gemma3_1b_it.yaml", ModelConfig)
    assert config.model_id == "google/gemma-3-1b-it"
    assert config.revision == GEMMA_REVISION


def test_accuracy_matches_the_recorded_items(prepared_task, config_paths, tmp_path: Path) -> None:
    out = tmp_path / "out"
    report = _run(prepared_task, config_paths, out, max_items=3)
    rows = [
        json.loads(line)
        for line in (out / "private_payloads" / "benchmark_items.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert report["evaluation"]["correct_items"] == sum(row["correct"] for row in rows)
    assert report["evaluation"]["accuracy"] == pytest.approx(
        report["evaluation"]["correct_items"] / 3
    )


def test_predicted_label_comes_from_label_logits(
    prepared_task, config_paths, tmp_path: Path
) -> None:
    """Scoring reads the four label logits directly, never generated text."""
    out = tmp_path / "out"
    _run(prepared_task, config_paths, out)
    rows = [
        json.loads(line)
        for line in (out / "private_payloads" / "benchmark_items.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    for row in rows:
        best = max(row["label_logits"], key=lambda label: row["label_logits"][label])
        assert row["predicted_label"] == best
        assert torch.isclose(
            torch.tensor(sum(row["label_probabilities"].values())), torch.tensor(1.0), atol=1e-6
        )
