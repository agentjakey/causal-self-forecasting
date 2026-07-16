"""Tests for the intervention mathematics.

These run on hand-built tensors with no model, which is exactly what makes them useful: the
mechanism can be wrong in ways that a full pipeline run would still report as plausible
numbers.
"""

from __future__ import annotations

import pytest
import torch

from causal_self_forecasting.interventions.directions import (
    DirectionNotFoundError,
    DirectionStore,
    matched_random_direction,
)
from causal_self_forecasting.interventions.tensor_ops import (
    InterventionPayload,
    InterventionShapeError,
    apply_activation_patch,
    apply_direction_ablate,
    apply_intervention,
    apply_noop,
    apply_residual_add,
)
from causal_self_forecasting.schemas import InterventionSpec, Mechanism


def _spec(
    mechanism: Mechanism, strength: float, direction_id: str | None = "d", **kwargs
) -> InterventionSpec:
    return InterventionSpec(
        intervention_id="t1.opaque_00",
        mechanism=mechanism,
        mechanism_version="1.0",
        layer=2,
        position_index=-1,
        strength=strength,
        direction_id=direction_id,
        **kwargs,
    )


def test_noop_returns_an_identical_vector() -> None:
    hidden = torch.randn(16)
    assert torch.equal(apply_noop(hidden), hidden)


def test_noop_returns_a_copy_not_an_alias() -> None:
    """A caller mutating the result must not corrupt the clean state it came from."""
    hidden = torch.randn(16)
    result = apply_noop(hidden)
    result[0] += 1.0
    assert not torch.equal(result, hidden)


def test_residual_add_at_zero_strength_is_exactly_the_identity() -> None:
    hidden = torch.randn(16)
    direction = torch.randn(16)
    assert torch.equal(apply_residual_add(hidden, direction, 0.0), hidden)


def test_residual_add_is_exact() -> None:
    hidden = torch.zeros(4)
    direction = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = apply_residual_add(hidden, direction, 2.0)
    assert torch.allclose(result, torch.tensor([2.0, 4.0, 6.0, 8.0]))


def test_residual_add_reverses_under_sign_flip() -> None:
    hidden = torch.randn(16)
    direction = torch.randn(16)
    positive = apply_residual_add(hidden, direction, 1.5) - hidden
    negative = apply_residual_add(hidden, direction, -1.5) - hidden
    assert torch.allclose(positive, -negative, atol=1e-6)


def test_ablation_removes_the_aligned_component() -> None:
    hidden = torch.tensor([3.0, 4.0, 0.0])
    direction = torch.tensor([1.0, 0.0, 0.0])
    result = apply_direction_ablate(hidden, direction)
    assert torch.allclose(result, torch.tensor([0.0, 4.0, 0.0]), atol=1e-6)


def test_ablation_leaves_an_orthogonal_state_untouched() -> None:
    hidden = torch.tensor([0.0, 5.0, 0.0])
    direction = torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(apply_direction_ablate(hidden, direction), hidden, atol=1e-6)


def test_ablation_output_is_orthogonal_to_the_direction() -> None:
    hidden = torch.randn(32)
    direction = torch.randn(32)
    result = apply_direction_ablate(hidden, direction)
    assert abs(float(torch.dot(result, direction / torch.linalg.vector_norm(direction)))) < 1e-5


def test_ablation_is_idempotent() -> None:
    """Removing a component that is already gone must do nothing."""
    hidden = torch.randn(32)
    direction = torch.randn(32)
    once = apply_direction_ablate(hidden, direction)
    twice = apply_direction_ablate(once, direction)
    assert torch.allclose(once, twice, atol=1e-5)


def test_ablation_does_not_depend_on_direction_scale() -> None:
    hidden = torch.randn(32)
    direction = torch.randn(32)
    assert torch.allclose(
        apply_direction_ablate(hidden, direction),
        apply_direction_ablate(hidden, direction * 17.0),
        atol=1e-5,
    )


def test_ablation_rejects_a_zero_direction() -> None:
    with pytest.raises(InterventionShapeError, match="undefined"):
        apply_direction_ablate(torch.randn(8), torch.zeros(8))


def test_activation_patch_replaces_the_state() -> None:
    source = torch.randn(16)
    assert torch.equal(apply_activation_patch(torch.randn(16), source), source)


def test_activation_patch_returns_a_copy() -> None:
    source = torch.randn(16)
    result = apply_activation_patch(torch.randn(16), source)
    result[0] += 1.0
    assert not torch.equal(result, source)


@pytest.mark.parametrize(
    "bad",
    [torch.ones(17), torch.ones(1), torch.ones(2, 16)],
)
def test_residual_add_rejects_mismatched_directions(bad: torch.Tensor) -> None:
    """Torch would broadcast some of these into a plausible, meaningless result."""
    with pytest.raises(InterventionShapeError):
        apply_residual_add(torch.randn(16), bad, 1.0)


def test_intervention_rejects_a_two_dimensional_hidden_state() -> None:
    with pytest.raises(InterventionShapeError, match="1-D"):
        apply_noop(torch.randn(2, 16))


def test_intervention_rejects_non_finite_input() -> None:
    hidden = torch.randn(8)
    hidden[0] = float("nan")
    with pytest.raises(InterventionShapeError, match="non-finite"):
        apply_noop(hidden)


def test_dispatch_reports_zero_delta_for_a_noop() -> None:
    hidden = torch.randn(16)
    result, diagnostics = apply_intervention(
        _spec(Mechanism.NOOP, 0.0, None), hidden, InterventionPayload()
    )
    assert torch.equal(result, hidden)
    assert diagnostics.delta_norm == 0.0
    assert diagnostics.pre_norm == pytest.approx(diagnostics.post_norm)


def test_dispatch_reports_the_expected_delta_norm() -> None:
    hidden = torch.zeros(8)
    direction = torch.zeros(8)
    direction[0] = 1.0
    _, diagnostics = apply_intervention(
        _spec(Mechanism.RESIDUAL_ADD, 3.0), hidden, InterventionPayload(direction=direction)
    )
    assert diagnostics.delta_norm == pytest.approx(3.0)


def test_dispatch_requires_a_direction_payload() -> None:
    with pytest.raises(InterventionShapeError, match="requires a direction"):
        apply_intervention(
            _spec(Mechanism.RESIDUAL_ADD, 1.0), torch.randn(8), InterventionPayload()
        )


def test_dispatch_requires_a_source_activation_for_patching() -> None:
    spec = _spec(Mechanism.ACTIVATION_PATCH, 1.0, direction_id=None, source_state_id="s1")
    with pytest.raises(InterventionShapeError, match="source_activation"):
        apply_intervention(spec, torch.randn(8), InterventionPayload())


def test_matched_random_direction_matches_the_reference_norm() -> None:
    """Norm matching is what makes this a control rather than a weaker intervention."""
    reference = torch.randn(64) * 3.7
    control = matched_random_direction(reference, seed=1)
    assert float(torch.linalg.vector_norm(control)) == pytest.approx(
        float(torch.linalg.vector_norm(reference)), rel=1e-5
    )


def test_matched_random_direction_is_reproducible_from_its_seed() -> None:
    reference = torch.randn(64)
    assert torch.allclose(
        matched_random_direction(reference, seed=7), matched_random_direction(reference, seed=7)
    )


def test_matched_random_directions_differ_across_seeds() -> None:
    reference = torch.randn(64)
    assert not torch.allclose(
        matched_random_direction(reference, seed=7), matched_random_direction(reference, seed=8)
    )


def test_matched_random_direction_is_not_aligned_with_the_reference() -> None:
    # In 512 dimensions two independent vectors are nearly orthogonal. A control that happened
    # to align with the reference would not be a control.
    reference = torch.randn(512)
    control = matched_random_direction(reference, seed=3)
    cosine = torch.dot(reference, control) / (
        torch.linalg.vector_norm(reference) * torch.linalg.vector_norm(control)
    )
    assert abs(float(cosine)) < 0.2


def test_direction_store_round_trips(tmp_path) -> None:
    store = DirectionStore(tmp_path)
    vector = torch.randn(32)
    store.save("d1", vector, {"method": "test"})
    assert torch.allclose(store.load("d1"), vector, atol=1e-6)
    assert store.list_ids() == ["d1"]


def test_direction_store_records_metadata(tmp_path) -> None:
    store = DirectionStore(tmp_path)
    store.save("d1", torch.randn(32), {"method": "clean_minus_adapted", "validated": False})
    metadata = store.metadata("d1")
    assert metadata["method"] == "clean_minus_adapted"
    assert metadata["validated"] is False
    assert metadata["dim"] == 32


def test_direction_store_rejects_a_zero_vector(tmp_path) -> None:
    with pytest.raises(InterventionShapeError, match="effectively zero"):
        DirectionStore(tmp_path).save("d1", torch.zeros(32), {})


def test_direction_store_rejects_non_finite_values(tmp_path) -> None:
    vector = torch.randn(32)
    vector[0] = float("inf")
    with pytest.raises(InterventionShapeError, match="non-finite"):
        DirectionStore(tmp_path).save("d1", vector, {})


def test_direction_store_reports_a_missing_direction(tmp_path) -> None:
    with pytest.raises(DirectionNotFoundError, match="not found"):
        DirectionStore(tmp_path).load("nope")
