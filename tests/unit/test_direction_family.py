"""Tests for the deterministic direction family.

Everything here runs offline against hand-built output-weight matrices or the local fixture
model. No pretrained weights, no network.

The failure this file is really guarding against is a family that looks well formed while its
controls are not actually orthogonal, its ids leak the construction role, or its vectors cannot
be regenerated. Those would all survive a smoke run and quietly weaken the study's central
control.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from causal_self_forecasting.config import DirectionFamilyConfig, ModelConfig, load_config
from causal_self_forecasting.interventions import direction_family as df
from causal_self_forecasting.interventions.direction_family import (
    TOLERANCES,
    DirectionFamilyError,
    build_direction_family,
    canonicalize_sign,
    centered_answer_directions,
    family_content_bytes,
    load_direction_family,
    modified_gram_schmidt,
    opaque_direction_id,
    orthogonal_random_controls,
    random_family_seed,
    regenerate_direction_family,
    vector_content_hash,
    verify_direction_family,
    write_direction_family,
)
from causal_self_forecasting.interventions.directions import DirectionStore
from causal_self_forecasting.models.loader import LoadedModel, ModelLoadError, load_model
from causal_self_forecasting.schemas import (
    DirectionConstructionRole,
    DirectionFamilyRecord,
    ModelSpec,
    compute_direction_family_hash,
)

LABELS = ["A", "B", "C", "D"]
STUB_TOKEN_IDS = {"A": 5, "B": 6, "C": 7, "D": 8}


# ---------------------------------------------------------------------------
# Offline stubs
# ---------------------------------------------------------------------------


class _Tokenizer:
    """The smallest tokenizer the scoring path accepts."""

    unk_token_id = 0

    def __init__(self, mapping: dict[str, int]) -> None:
        self._mapping = mapping

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        stripped = text.strip()
        if stripped not in self._mapping:
            return [self.unk_token_id]
        return [self._mapping[stripped]]


class _Embedding(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)


class _Config:
    def __init__(self, tie: bool | None) -> None:
        self.tie_word_embeddings = tie


class _Model(torch.nn.Module):
    """A stand-in exposing only the surface the accessor is allowed to touch."""

    def __init__(
        self,
        out_weight: torch.Tensor | None,
        in_weight: torch.Tensor | None = None,
        tie: bool | None = False,
        share: bool = False,
    ) -> None:
        super().__init__()
        self._out = None if out_weight is None else _Embedding(out_weight)
        if share and self._out is not None:
            self._in = self._out
        else:
            self._in = None if in_weight is None else _Embedding(in_weight)
        self.config = _Config(tie)

    def get_output_embeddings(self):
        return self._out

    def get_input_embeddings(self):
        return self._in


def _loaded(
    out_weight: torch.Tensor | None,
    hidden_dim: int,
    token_ids: dict[str, int] | None = None,
    in_weight: torch.Tensor | None = None,
    tie: bool | None = False,
    share: bool = False,
) -> LoadedModel:
    return LoadedModel(
        model=cast(Any, _Model(out_weight, in_weight=in_weight, tie=tie, share=share)),
        tokenizer=cast(Any, _Tokenizer(token_ids or STUB_TOKEN_IDS)),
        spec=ModelSpec(model_id="stub/model", revision="rev-abc123", dtype="float32", device="cpu"),
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_layers=2,
        hidden_dim=hidden_dim,
    )


def _stub_model(hidden_dim: int = 32, vocab: int = 16, seed: int = 11) -> LoadedModel:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(vocab, hidden_dim, generator=generator, dtype=torch.float32)
    return _loaded(weight, hidden_dim)


def _write_config(tmp_path: Path, **overrides: Any) -> Path:
    body: dict[str, Any] = {
        "name": "test_family",
        "family_id": "test_family_v1",
        "study_id": "test_study",
        "model_ref": "configs/models/fixture_tiny.yaml",
        "direction_id_prefix": "tf1",
        "answer_labels": LABELS,
        "label_prefix": " ",
        "master_seed": 20260727,
        "random_control_count": 4,
    }
    body.update(overrides)
    path = tmp_path / f"{body['family_id']}.yaml"
    path.write_text(json.dumps(body), encoding="utf-8")  # JSON is valid YAML
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


def _build(tmp_path: Path, model: LoadedModel, **overrides: Any) -> DirectionFamilyRecord:
    config_path = _write_config(tmp_path, **overrides)
    family_config = load_config(config_path, DirectionFamilyConfig)
    store = DirectionStore(tmp_path / "directions")
    return build_direction_family(family_config, model, store, config_path=config_path)


# ---------------------------------------------------------------------------
# Output-embedding accessor
# ---------------------------------------------------------------------------


def test_accessor_returns_the_output_embedding() -> None:
    model = _stub_model(hidden_dim=32, vocab=16)
    embedding = model.output_embedding(required_token_ids=[5, 6, 7, 8])
    assert embedding.source == "get_output_embeddings"
    assert embedding.weight.shape == (16, 32)
    assert embedding.vocab_size == 16
    assert embedding.hidden_dim == 32
    assert embedding.revision == "rev-abc123"
    assert embedding.tied is False


def test_accessor_falls_back_to_a_tied_input_embedding() -> None:
    weight = torch.randn(16, 32)
    model = _loaded(None, 32, in_weight=weight, tie=True)
    embedding = model.output_embedding()
    assert embedding.source == "tied_input_embeddings"
    assert embedding.tied is False, "tying needs both tensors present to be observed"
    assert embedding.config_tie_word_embeddings is True


def test_accessor_reports_tying_from_the_tensors_not_the_config() -> None:
    """A config flag and the weights in memory can disagree; the tensors decide."""
    weight = torch.randn(16, 32)
    shared = _loaded(weight, 32, tie=False, share=True)
    assert shared.output_embedding().tied is True
    assert shared.output_embedding().config_tie_word_embeddings is False

    separate = _loaded(weight, 32, in_weight=torch.randn(16, 32), tie=True)
    assert separate.output_embedding().tied is False


def test_accessor_rejects_a_missing_embedding() -> None:
    with pytest.raises(ModelLoadError, match="neither an output embedding"):
        _loaded(None, 32).output_embedding()


def test_accessor_rejects_a_non_two_dimensional_matrix() -> None:
    with pytest.raises(ModelLoadError, match="must be 2-D"):
        _loaded(torch.randn(4, 16, 32), 32).output_embedding()


def test_accessor_rejects_a_hidden_axis_mismatch() -> None:
    with pytest.raises(ModelLoadError, match="hidden axis"):
        _loaded(torch.randn(16, 31), 32).output_embedding()


def test_accessor_rejects_a_token_id_outside_the_vocabulary() -> None:
    with pytest.raises(ModelLoadError, match="outside the"):
        _stub_model(vocab=16).output_embedding(required_token_ids=[99])


def test_accessor_rejects_a_non_finite_answer_row() -> None:
    weight = torch.randn(16, 32)
    weight[5, 0] = float("nan")
    with pytest.raises(ModelLoadError, match="non-finite"):
        _loaded(weight, 32).output_embedding(required_token_ids=[5])


# ---------------------------------------------------------------------------
# Token-id resolution
# ---------------------------------------------------------------------------


def test_token_ids_come_from_the_scoring_path(tmp_path: Path) -> None:
    record = _build(tmp_path, _stub_model())
    assert record.answer_token_ids == STUB_TOKEN_IDS


def test_unexpected_token_ids_refuse_the_build(tmp_path: Path) -> None:
    with pytest.raises(DirectionFamilyError, match="differ from the pinned expectations"):
        _build(
            tmp_path,
            _stub_model(),
            expected_token_ids={"A": 562, "B": 603, "C": 565, "D": 622},
        )


def test_matching_expected_token_ids_are_accepted(tmp_path: Path) -> None:
    record = _build(tmp_path, _stub_model(), expected_token_ids=dict(STUB_TOKEN_IDS))
    assert record.answer_token_ids == STUB_TOKEN_IDS


def test_an_unexpected_hidden_dimension_refuses_the_build(tmp_path: Path) -> None:
    with pytest.raises(DirectionFamilyError, match="hidden dimension"):
        _build(tmp_path, _stub_model(hidden_dim=32), expected_hidden_dim=1152)


def test_unscoreable_labels_refuse_the_build(tmp_path: Path) -> None:
    model = _loaded(torch.randn(16, 32), 32, token_ids={"A": 5, "B": 6, "C": 7})
    with pytest.raises(DirectionFamilyError, match="not scoreable"):
        _build(tmp_path, model)


# ---------------------------------------------------------------------------
# Answer directions
# ---------------------------------------------------------------------------


def test_centered_formula_is_exact() -> None:
    rows = {
        "A": np.array([4.0, 0.0, 0.0]),
        "B": np.array([0.0, 4.0, 0.0]),
        "C": np.array([0.0, 0.0, 4.0]),
        "D": np.array([0.0, 0.0, 0.0]),
    }
    raw, unit, norms = centered_answer_directions(rows, LABELS)
    expected_a = rows["A"] - (rows["B"] + rows["C"] + rows["D"]) / 3.0
    assert np.allclose(raw["A"], expected_a)
    assert norms["A"] == pytest.approx(float(np.linalg.norm(expected_a)))
    assert np.allclose(unit["A"], expected_a / np.linalg.norm(expected_a))


def test_raw_centered_directions_sum_to_zero() -> None:
    generator = np.random.default_rng(5)
    rows = {label: generator.standard_normal(32) for label in LABELS}
    raw, _, _ = centered_answer_directions(rows, LABELS)
    total = sum(raw[label] for label in LABELS)
    assert float(np.max(np.abs(total))) < 1e-12


def test_raw_norms_are_nonzero_and_recorded(tmp_path: Path) -> None:
    record = _build(tmp_path, _stub_model())
    answers = record.by_role(DirectionConstructionRole.ANSWER_TOKEN_CENTERED)
    assert len(answers) == 4
    for entry in answers:
        assert entry.raw_norm is not None and entry.raw_norm > 0.0


def test_degenerate_answer_rows_are_refused() -> None:
    rows = {label: np.ones(8) for label in LABELS}
    with pytest.raises(DirectionFamilyError, match="degenerate"):
        centered_answer_directions(rows, LABELS)


def test_answer_directions_are_unit_norm(tmp_path: Path) -> None:
    record = _build(tmp_path, _stub_model())
    store = DirectionStore(tmp_path / "directions")
    for entry in record.directions:
        vector = store.load(entry.opaque_id).numpy().astype(np.float64)
        assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=TOLERANCES.norm_tolerance)


# ---------------------------------------------------------------------------
# Answer-span basis
# ---------------------------------------------------------------------------


def test_gram_schmidt_drops_a_dependent_candidate() -> None:
    vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
    ]
    basis, error = modified_gram_schmidt(vectors, TOLERANCES.rank_tolerance)
    assert len(basis) == 2
    assert error < 1e-12


def test_gram_schmidt_is_deterministic_and_order_dependent() -> None:
    generator = np.random.default_rng(3)
    vectors = [generator.standard_normal(16) for _ in range(4)]
    vectors = [vector / np.linalg.norm(vector) for vector in vectors]
    first, _ = modified_gram_schmidt(vectors, TOLERANCES.rank_tolerance)
    second, _ = modified_gram_schmidt(vectors, TOLERANCES.rank_tolerance)
    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))

    reversed_basis, _ = modified_gram_schmidt(list(reversed(vectors)), TOLERANCES.rank_tolerance)
    assert not np.array_equal(first[0], reversed_basis[0])


def test_the_answer_span_has_rank_three(tmp_path: Path) -> None:
    """Four centered directions that sum to zero span at most three dimensions."""
    record = _build(tmp_path, _stub_model())
    assert record.answer_span_rank == 3
    assert record.diagnostics.centered_sum_max_abs_residual < 1e-10
    assert record.diagnostics.answer_span_orthonormality_error < 1e-10


def test_answer_directions_are_not_required_to_be_orthogonal(tmp_path: Path) -> None:
    record = _build(tmp_path, _stub_model())
    assert record.diagnostics.max_answer_pairwise_abs_cosine > 0.01


# ---------------------------------------------------------------------------
# Random controls
# ---------------------------------------------------------------------------


def test_random_controls_are_orthogonal_to_the_answer_span(tmp_path: Path) -> None:
    record = _build(tmp_path, _stub_model())
    assert record.diagnostics.max_answer_to_random_abs_dot < TOLERANCES.orthogonality_tolerance


def test_random_controls_are_mutually_orthogonal(tmp_path: Path) -> None:
    record = _build(tmp_path, _stub_model())
    assert record.diagnostics.max_random_to_random_abs_dot < TOLERANCES.orthogonality_tolerance


def test_random_controls_are_deterministic(tmp_path: Path) -> None:
    first = _build(tmp_path, _stub_model())
    second = _build(tmp_path, _stub_model())
    assert [entry.vector_hash for entry in first.directions] == [
        entry.vector_hash for entry in second.directions
    ]


def test_a_different_master_seed_changes_the_controls(tmp_path: Path) -> None:
    baseline = _build(tmp_path, _stub_model())
    other = _build(tmp_path, _stub_model(), master_seed=1)
    baseline_controls = {
        entry.vector_hash
        for entry in baseline.by_role(DirectionConstructionRole.RANDOM_ORTHOGONAL_CONTROL)
    }
    other_controls = {
        entry.vector_hash
        for entry in other.by_role(DirectionConstructionRole.RANDOM_ORTHOGONAL_CONTROL)
    }
    assert not (baseline_controls & other_controls)


def test_the_seed_is_bound_to_the_model_revision() -> None:
    first, labels = random_family_seed(1, "s", "f", "m", "rev-a")
    second, _ = random_family_seed(1, "s", "f", "m", "rev-b")
    assert first != second
    assert "rev-a" in labels


def test_sign_canonicalization_makes_the_first_significant_component_positive() -> None:
    vector = np.array([0.0, -0.5, 0.3])
    fixed = canonicalize_sign(vector, TOLERANCES.sign_tolerance)
    assert fixed[1] > 0.0
    assert np.allclose(fixed, -vector)

    already = np.array([0.0, 0.5, -0.3])
    assert np.array_equal(canonicalize_sign(already, TOLERANCES.sign_tolerance), already)


def test_sign_canonicalization_rejects_an_all_zero_vector() -> None:
    with pytest.raises(DirectionFamilyError, match="sign tolerance"):
        canonicalize_sign(np.zeros(8), TOLERANCES.sign_tolerance)


def test_a_degenerate_random_draw_is_rejected_and_redrawn() -> None:
    """A draw that lies in the removed span must not be normalized into rounding noise."""
    basis = [np.eye(4)[0], np.eye(4)[1]]
    draws = iter(
        [
            np.array([1.0, 2.0, 0.0, 0.0]),  # entirely inside the answer span
            np.array([0.0, 0.0, 3.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 5.0]),
        ]
    )
    controls, redraws = orthogonal_random_controls(
        dim=4, count=2, answer_basis=basis, tolerances=TOLERANCES, draw=lambda: next(draws)
    )
    assert redraws == 1
    assert len(controls) == 2
    assert abs(float(np.dot(controls[0], controls[1]))) < 1e-12


def test_an_impossible_control_request_raises() -> None:
    basis = [np.eye(3)[0], np.eye(3)[1], np.eye(3)[2]]
    generator = np.random.default_rng(0)
    with pytest.raises(DirectionFamilyError, match="redrew"):
        orthogonal_random_controls(
            dim=3,
            count=1,
            answer_basis=basis,
            tolerances=TOLERANCES,
            draw=lambda: generator.standard_normal(3),
            max_redraws=4,
        )


# ---------------------------------------------------------------------------
# Identity, hashing, and storage
# ---------------------------------------------------------------------------


def test_vector_hash_depends_only_on_the_values() -> None:
    vector = np.arange(8, dtype=np.float32)
    assert vector_content_hash(vector) == vector_content_hash(np.array(vector, copy=True))
    other = vector.copy()
    other[0] += 1.0
    assert vector_content_hash(vector) != vector_content_hash(other)


def test_opaque_ids_are_stable_and_role_specific() -> None:
    first = opaque_direction_id(
        "p", "f", "rev", 1, DirectionConstructionRole.ANSWER_TOKEN_CENTERED, 0
    )
    assert first == opaque_direction_id(
        "p", "f", "rev", 1, DirectionConstructionRole.ANSWER_TOKEN_CENTERED, 0
    )
    assert first != opaque_direction_id(
        "p", "f", "rev", 1, DirectionConstructionRole.RANDOM_ORTHOGONAL_CONTROL, 0
    )
    assert first != opaque_direction_id(
        "p", "f", "rev2", 1, DirectionConstructionRole.ANSWER_TOKEN_CENTERED, 0
    )


def test_direction_store_round_trips_a_family_vector(tmp_path: Path) -> None:
    record = _build(tmp_path, _stub_model())
    store = DirectionStore(tmp_path / "directions")
    assert sorted(store.list_ids()) == sorted(entry.opaque_id for entry in record.directions)
    for entry in record.directions:
        loaded = store.load(entry.opaque_id).numpy().astype(np.float32)
        assert vector_content_hash(loaded) == entry.vector_hash
        assert loaded.shape == (record.hidden_dim,)


def test_family_hash_is_the_hash_of_its_own_payload(tmp_path: Path) -> None:
    record = _build(tmp_path, _stub_model())
    assert record.family_hash == compute_direction_family_hash(record.model_dump(mode="json"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_revision", "other-revision"),
        ("master_seed", 999),
        ("construction_algorithm_version", "v99"),
        ("answer_span_rank", 2),
    ],
)
def test_family_hash_is_sensitive_to_its_inputs(tmp_path: Path, field: str, value: object) -> None:
    dumped = _build(tmp_path, _stub_model()).model_dump(mode="json")
    before = compute_direction_family_hash(dumped)
    dumped[field] = value
    assert compute_direction_family_hash(dumped) != before


def test_family_hash_is_sensitive_to_a_changed_vector(tmp_path: Path) -> None:
    dumped = _build(tmp_path, _stub_model()).model_dump(mode="json")
    before = compute_direction_family_hash(dumped)
    dumped["directions"][0]["vector_hash"] = "sha256:" + "0" * 64
    assert compute_direction_family_hash(dumped) != before


def test_family_hash_is_sensitive_to_reordering(tmp_path: Path) -> None:
    dumped = _build(tmp_path, _stub_model()).model_dump(mode="json")
    before = compute_direction_family_hash(dumped)
    dumped["directions"].reverse()
    assert compute_direction_family_hash(dumped) != before


def test_family_hash_is_sensitive_to_a_tolerance(tmp_path: Path) -> None:
    dumped = _build(tmp_path, _stub_model()).model_dump(mode="json")
    before = compute_direction_family_hash(dumped)
    dumped["tolerances"]["rank_tolerance"] = 1e-7
    assert compute_direction_family_hash(dumped) != before


def test_family_hash_ignores_container_and_creation_metadata(tmp_path: Path) -> None:
    record = _build(tmp_path, _stub_model())
    dumped = record.model_dump(mode="json")
    dumped["created_at"] = "2020-01-01T00:00:00Z"
    dumped["config_path"] = "somewhere/else.yaml"
    dumped["environment"] = {"packages": {"numpy": "0.0.0"}}
    dumped["directions"][0]["artifact_hash"] = "sha256:" + "1" * 64
    assert compute_direction_family_hash(dumped) == record.family_hash


def test_family_content_bytes_are_stable(tmp_path: Path) -> None:
    assert family_content_bytes(_build(tmp_path, _stub_model())) == family_content_bytes(
        _build(tmp_path, _stub_model())
    )


# ---------------------------------------------------------------------------
# Schema round trip and tamper evidence
# ---------------------------------------------------------------------------


def test_family_round_trips_through_json(tmp_path: Path) -> None:
    record = _build(tmp_path, _stub_model())
    restored = DirectionFamilyRecord.model_validate(
        json.loads(json.dumps(record.model_dump(mode="json")))
    )
    assert restored.family_hash == record.family_hash
    assert restored.directions == record.directions


def test_a_tampered_family_hash_fails_to_load(tmp_path: Path) -> None:
    dumped = _build(tmp_path, _stub_model()).model_dump(mode="json")
    dumped["family_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError, match="does not match the family contents"):
        DirectionFamilyRecord.model_validate(dumped)


def test_directions_must_be_ordered_by_opaque_id(tmp_path: Path) -> None:
    dumped = _build(tmp_path, _stub_model()).model_dump(mode="json")
    dumped["directions"].reverse()
    with pytest.raises(ValidationError, match="ascending opaque-id order"):
        DirectionFamilyRecord.model_validate(dumped)


def test_an_unknown_field_is_rejected(tmp_path: Path) -> None:
    dumped = _build(tmp_path, _stub_model()).model_dump(mode="json")
    dumped["extra"] = 1
    with pytest.raises(ValidationError):
        DirectionFamilyRecord.model_validate(dumped)


def test_a_random_control_may_not_carry_a_label(tmp_path: Path) -> None:
    dumped = _build(tmp_path, _stub_model()).model_dump(mode="json")
    for entry in dumped["directions"]:
        if entry["construction_role"] == "random_orthogonal_control":
            entry["label"] = "A"
            break
    with pytest.raises(ValidationError, match="must not carry a label"):
        DirectionFamilyRecord.model_validate(dumped)


# ---------------------------------------------------------------------------
# Privacy boundary
# ---------------------------------------------------------------------------


_LEAKY_TERMS = (
    "answer",
    "random",
    "control",
    "orthogonal",
    "centered",
    "unembed",
    "label",
    "token",
    "role",
    "bias",
)


def test_opaque_ids_leak_no_construction_role(tmp_path: Path) -> None:
    record = _build(tmp_path, _stub_model())
    for entry in record.directions:
        lowered = entry.opaque_id.lower()
        for term in _LEAKY_TERMS:
            assert term not in lowered, f"{entry.opaque_id} leaks {term!r}"
        prefix, _, suffix = entry.opaque_id.partition(".")
        assert prefix == "tf1"
        assert len(suffix) == 16
        assert all(char in "0123456789abcdef" for char in suffix)
        for label in LABELS:
            assert f"_{label.lower()}" not in lowered


def test_stored_direction_metadata_carries_no_construction_role(tmp_path: Path) -> None:
    """The npz sits in the payload store that resolution reads; the mapping stays elsewhere."""
    record = _build(tmp_path, _stub_model())
    store = DirectionStore(tmp_path / "directions")
    for entry in record.directions:
        metadata = store.metadata(entry.opaque_id)
        serialized = json.dumps(metadata).lower()
        for term in _LEAKY_TERMS:
            assert term not in serialized, f"{entry.opaque_id} metadata leaks {term!r}"
        assert metadata["validated"] is False


def test_public_intervention_view_is_unchanged() -> None:
    """S2 must not widen what a forecaster can see."""
    from causal_self_forecasting.forecasting.base import PUBLIC_FEATURE_KEYS
    from causal_self_forecasting.schemas import InterventionSpec, Mechanism
    from causal_self_forecasting.trials.candidates import public_view

    assert PUBLIC_FEATURE_KEYS == ("operation", "layer", "position_index", "strength")
    spec = InterventionSpec(
        intervention_id="t1.opaque_00",
        mechanism=Mechanism.RESIDUAL_ADD,
        mechanism_version="1.0",
        layer=13,
        position_index=-1,
        strength=1.0,
        direction_id="tf1.deadbeefdeadbeef",
        analysis_role="direction_positive",
    )
    view = public_view(spec)
    assert set(view) == {"intervention_id", "operation", "layer", "position_index", "strength"}
    assert "direction_id" not in view
    assert "analysis_role" not in view
    assert "construction_role" not in view


def test_public_metadata_still_rejects_semantic_roles() -> None:
    from causal_self_forecasting.schemas import InterventionSpec, Mechanism

    with pytest.raises(ValidationError, match="would leak"):
        InterventionSpec(
            intervention_id="t1.opaque_00",
            mechanism=Mechanism.RESIDUAL_ADD,
            mechanism_version="1.0",
            layer=13,
            position_index=-1,
            strength=1.0,
            direction_id="tf1.deadbeefdeadbeef",
            public_metadata={"analysis_role": "random_control"},
        )


# ---------------------------------------------------------------------------
# Atomicity, storage, and verification
# ---------------------------------------------------------------------------


def test_a_failed_save_writes_no_manifest(workspace: Path, monkeypatch) -> None:
    config_path = _write_config(workspace)
    family_config = load_config(config_path, DirectionFamilyConfig)
    store = DirectionStore(workspace / "directions")

    calls = {"n": 0}
    original = store.save

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 5:
            raise OSError("disk full")
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "save", flaky)
    with pytest.raises(OSError, match="disk full"):
        build_direction_family(family_config, _stub_model(), store, config_path=config_path)

    assert not (workspace / "direction_manifests" / "test_family_v1.json").exists()


def test_write_then_verify(workspace: Path) -> None:
    record = _build(workspace, _stub_model())
    path, status = write_direction_family(record)
    assert status == "written"
    assert path.exists()

    report = verify_direction_family(record, DirectionStore(workspace / "directions"))
    assert report["valid"], report["failures"]
    assert report["directions_checked"] == 8
    assert report["answer_directions"] == 4
    assert report["random_controls"] == 4
    assert report["max_answer_to_random_abs_dot"] < TOLERANCES.orthogonality_tolerance


def test_verification_detects_a_missing_artifact(workspace: Path) -> None:
    record = _build(workspace, _stub_model())
    store = DirectionStore(workspace / "directions")
    store.path_for(record.directions[0].opaque_id).unlink()
    report = verify_direction_family(record, store)
    assert not report["valid"]
    assert "no stored vector artifact" in report["failures"][0]


def test_verification_detects_a_modified_vector(workspace: Path) -> None:
    record = _build(workspace, _stub_model())
    store = DirectionStore(workspace / "directions")
    target = record.directions[0]
    tampered = store.load(target.opaque_id).numpy().astype(np.float32)
    tampered[0] = float(tampered[0]) + 0.5
    store.save(target.opaque_id, tampered / np.linalg.norm(tampered), {"tampered": True})
    report = verify_direction_family(record, store)
    assert not report["valid"]
    assert any("does not match its recorded hash" in failure for failure in report["failures"])


def test_rewriting_an_identical_family_leaves_the_file_untouched(workspace: Path) -> None:
    record = _build(workspace, _stub_model())
    path, _ = write_direction_family(record)
    original = path.read_bytes()
    mtime = path.stat().st_mtime_ns

    _, status = write_direction_family(_build(workspace, _stub_model()))
    assert status == "unchanged"
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == mtime


def test_a_different_family_is_refused_at_the_same_path(workspace: Path) -> None:
    write_direction_family(_build(workspace, _stub_model()))
    with pytest.raises(DirectionFamilyError, match="already holds a different"):
        write_direction_family(_build(workspace, _stub_model(), master_seed=4242))


def test_force_replaces_a_different_family(workspace: Path) -> None:
    write_direction_family(_build(workspace, _stub_model()))
    replacement = _build(workspace, _stub_model(), master_seed=4242)
    _, status = write_direction_family(replacement, force=True)
    assert status == "overwritten"
    assert load_direction_family(replacement.family_id).family_hash == replacement.family_hash


def test_regeneration_reproduces_the_family(workspace: Path) -> None:
    config_path = _write_config(workspace)
    family_config = load_config(config_path, DirectionFamilyConfig)
    store = DirectionStore(workspace / "directions")
    model = _stub_model()
    record = build_direction_family(family_config, model, store, config_path=config_path)

    report = regenerate_direction_family(record, family_config, model, store, config_path)
    assert report["valid"], report["failures"]
    assert report["rebuilt_family_hash"] == record.family_hash


def test_regeneration_detects_different_weights(workspace: Path) -> None:
    config_path = _write_config(workspace)
    family_config = load_config(config_path, DirectionFamilyConfig)
    store = DirectionStore(workspace / "directions")
    record = build_direction_family(family_config, _stub_model(seed=11), store, config_path)

    report = regenerate_direction_family(
        record, family_config, _stub_model(seed=12), store, config_path
    )
    assert not report["valid"]
    assert any("differs from the recorded hash" in failure for failure in report["failures"])


# ---------------------------------------------------------------------------
# The real fixture model, still offline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_model() -> LoadedModel:
    return load_model(
        ModelConfig(
            name="fixture_tiny",
            kind="fixture",
            model_id="tiny_llama_directions",
            revision="fixture-v1",
            dtype="float32",
            device="cpu",
            fixture_hidden_size=64,
            fixture_num_layers=4,
        )
    )


def test_a_family_builds_from_the_fixture_model(workspace: Path, fixture_model) -> None:
    record = _build(workspace, fixture_model)
    assert record.hidden_dim == 64
    assert record.output_embedding_source == "get_output_embeddings"
    assert record.tied_embeddings is False
    assert record.config_tie_word_embeddings is False
    assert record.answer_span_rank == 3
    assert len(record.directions) == 8

    report = verify_direction_family(record, DirectionStore(workspace / "directions"))
    assert report["valid"], report["failures"]


def test_regeneration_from_the_fixture_model(workspace: Path, fixture_model) -> None:
    config_path = _write_config(workspace)
    family_config = load_config(config_path, DirectionFamilyConfig)
    store = DirectionStore(workspace / "directions")
    record = build_direction_family(family_config, fixture_model, store, config_path=config_path)
    report = regenerate_direction_family(record, family_config, fixture_model, store, config_path)
    assert report["valid"], report["failures"]
