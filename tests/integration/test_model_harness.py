"""Integration tests against a real (tiny) model.

These use the fixture model: a genuinely randomly initialized Llama with a real forward pass
and real hooks. It knows nothing, so nothing here is a claim about language models. What it
does test is the plumbing that every later claim rests on.
"""

from __future__ import annotations

import pytest
import torch

from causal_self_forecasting.config import ModelConfig
from causal_self_forecasting.interventions.tensor_ops import InterventionPayload
from causal_self_forecasting.models.capture import (
    CaptureError,
    capture_hidden_states,
    resolve_position,
    run_with_intervention,
)
from causal_self_forecasting.models.loader import LoadedModel, load_model
from causal_self_forecasting.models.scoring import (
    LabelTokenError,
    resolve_label_token_ids,
    score_logits,
    target_option_rate,
)
from causal_self_forecasting.schemas import InterventionSpec, Mechanism

PROMPT = "what is the question A B C D"


@pytest.fixture(scope="module")
def model() -> LoadedModel:
    config = ModelConfig(
        name="fixture_tiny",
        kind="fixture",
        model_id="tiny_llama_test",
        revision="fixture-v1",
        dtype="float32",
        device="cpu",
        fixture_hidden_size=64,
        fixture_num_layers=4,
    )
    return load_model(config)


def _spec(mechanism: Mechanism, layer: int, strength: float, **kwargs) -> InterventionSpec:
    return InterventionSpec(
        intervention_id="trial_00001.opaque_00",
        mechanism=mechanism,
        mechanism_version="1.0",
        layer=layer,
        position_index=-1,
        strength=strength,
        **kwargs,
    )


def test_fixture_model_reports_its_shape(model: LoadedModel) -> None:
    assert model.num_layers == 4
    assert model.hidden_dim == 64
    assert model.spec.variant.value == "clean"


def test_capture_returns_one_state_per_requested_layer(model: LoadedModel) -> None:
    result = capture_hidden_states(model, PROMPT, layers=[0, 2, 4])
    assert sorted(result.hidden_states) == [0, 2, 4]
    for vector in result.hidden_states.values():
        assert vector.shape == (model.hidden_dim,)


def test_capture_is_deterministic(model: LoadedModel) -> None:
    """If two clean runs disagreed, no measured effect could be separated from run noise."""
    first = capture_hidden_states(model, PROMPT, layers=[2])
    second = capture_hidden_states(model, PROMPT, layers=[2])
    assert torch.equal(first.next_token_logits, second.next_token_logits)
    assert torch.equal(first.hidden_states[2], second.hidden_states[2])


def test_layer_zero_is_the_embedding_output(model: LoadedModel) -> None:
    result = capture_hidden_states(model, PROMPT, layers=[0])
    inputs = model.tokenizer(PROMPT, return_tensors="pt")
    with torch.no_grad():
        embedded = model.embedding_module(inputs["input_ids"])
    assert torch.allclose(result.hidden_states[0], embedded[0, -1], atol=1e-6)


@pytest.mark.parametrize("layer", [0, 1, 2, 3, 4])
def test_capture_matches_the_intervention_point(model: LoadedModel, layer: int) -> None:
    """Pins the layer-indexing convention.

    Patching layer L with a known vector and then capturing layer L must return that exact
    vector. An off-by-one here would mean interventions land one block away from where the
    state was read, and every result would look plausible and be wrong.
    """
    source = torch.full((model.hidden_dim,), 0.123)
    spec = _spec(Mechanism.ACTIVATION_PATCH, layer, 1.0, source_state_id="s1")

    result = run_with_intervention(
        model, PROMPT, spec, InterventionPayload(source_activation=source), capture_layers=[layer]
    )
    assert torch.allclose(result.hidden_states[layer], source, atol=1e-5)


def test_noop_reproduces_the_clean_logits_exactly(model: LoadedModel) -> None:
    clean = capture_hidden_states(model, PROMPT, layers=[2])
    noop = run_with_intervention(
        model, PROMPT, _spec(Mechanism.NOOP, 2, 0.0), InterventionPayload()
    )
    assert torch.equal(clean.next_token_logits, noop.next_token_logits)
    assert noop.diagnostics is not None
    assert noop.diagnostics.delta_norm == 0.0


def test_zero_strength_addition_reproduces_the_clean_logits(model: LoadedModel) -> None:
    clean = capture_hidden_states(model, PROMPT, layers=[2])
    zero = run_with_intervention(
        model,
        PROMPT,
        _spec(Mechanism.RESIDUAL_ADD, 2, 0.0, direction_id="d1"),
        InterventionPayload(direction=torch.randn(model.hidden_dim)),
    )
    assert torch.equal(clean.next_token_logits, zero.next_token_logits)


def test_a_real_intervention_changes_the_output(model: LoadedModel) -> None:
    """The complement of the no-op test: the hook must be capable of doing something."""
    clean = capture_hidden_states(model, PROMPT, layers=[2])
    steered = run_with_intervention(
        model,
        PROMPT,
        _spec(Mechanism.RESIDUAL_ADD, 2, 25.0, direction_id="d1"),
        InterventionPayload(direction=torch.randn(model.hidden_dim)),
    )
    assert not torch.equal(clean.next_token_logits, steered.next_token_logits)


def test_intervention_diagnostics_report_the_applied_change(model: LoadedModel) -> None:
    direction = torch.zeros(model.hidden_dim)
    direction[0] = 1.0
    result = run_with_intervention(
        model,
        PROMPT,
        _spec(Mechanism.RESIDUAL_ADD, 2, 4.0, direction_id="d1"),
        InterventionPayload(direction=direction),
    )
    assert result.diagnostics is not None
    assert result.diagnostics.delta_norm == pytest.approx(4.0, rel=1e-4)


def test_intervening_leaves_no_hook_behind(model: LoadedModel) -> None:
    """A leaked hook would silently contaminate every later run in the process."""
    before = capture_hidden_states(model, PROMPT, layers=[2])
    run_with_intervention(
        model,
        PROMPT,
        _spec(Mechanism.RESIDUAL_ADD, 2, 30.0, direction_id="d1"),
        InterventionPayload(direction=torch.randn(model.hidden_dim)),
    )
    after = capture_hidden_states(model, PROMPT, layers=[2])
    assert torch.equal(before.next_token_logits, after.next_token_logits)


def test_out_of_range_layer_is_rejected(model: LoadedModel) -> None:
    with pytest.raises(CaptureError, match="out of range"):
        capture_hidden_states(model, PROMPT, layers=[model.num_layers + 1])


def test_out_of_range_position_is_rejected() -> None:
    with pytest.raises(CaptureError, match="out of range"):
        resolve_position(50, sequence_length=5)


def test_negative_positions_resolve_from_the_end() -> None:
    assert resolve_position(-1, sequence_length=5) == 4
    assert resolve_position(0, sequence_length=5) == 0


def test_label_tokens_resolve_to_single_ids(model: LoadedModel) -> None:
    resolved = resolve_label_token_ids(model.tokenizer, ["A", "B", "C", "D"], prefix=" ")
    assert sorted(resolved) == ["A", "B", "C", "D"]
    assert len(set(resolved.values())) == 4


def test_unscoreable_labels_are_rejected(model: LoadedModel) -> None:
    """A multi-token label cannot be read off one next-token distribution."""
    with pytest.raises(LabelTokenError, match="single token"):
        resolve_label_token_ids(model.tokenizer, ["A B", "B", "C", "D"], prefix=" ")


def test_labels_outside_the_vocabulary_are_rejected(model: LoadedModel) -> None:
    with pytest.raises(LabelTokenError, match="unknown token"):
        resolve_label_token_ids(model.tokenizer, ["zzz", "B", "C", "D"], prefix=" ")


def test_margin_is_positive_when_the_correct_label_leads() -> None:
    logits = torch.zeros(100)
    logits[10], logits[11], logits[12], logits[13] = 3.0, 1.0, 0.0, 0.0
    scores = score_logits(logits, {"A": 10, "B": 11, "C": 12, "D": 13}, "A")
    assert scores.margin == pytest.approx(2.0)
    assert scores.predicted_label == "A"
    assert scores.is_correct


def test_margin_is_negative_when_a_distractor_leads() -> None:
    logits = torch.zeros(100)
    logits[10], logits[11] = 1.0, 3.0
    scores = score_logits(logits, {"A": 10, "B": 11, "C": 12, "D": 13}, "A")
    assert scores.margin == pytest.approx(-2.0)
    assert not scores.is_correct


def test_probabilities_are_normalized_over_the_four_labels() -> None:
    logits = torch.randn(100)
    scores = score_logits(logits, {"A": 10, "B": 11, "C": 12, "D": 13}, "A")
    assert sum(scores.probabilities.values()) == pytest.approx(1.0)


def test_entropy_is_bounded_by_log_four() -> None:
    import math

    logits = torch.zeros(100)
    scores = score_logits(logits, {"A": 10, "B": 11, "C": 12, "D": 13}, "A")
    assert scores.entropy == pytest.approx(math.log(4))


def test_scoring_rejects_a_correct_label_it_was_not_given() -> None:
    with pytest.raises(LabelTokenError, match="not among"):
        score_logits(torch.zeros(100), {"A": 10, "B": 11, "C": 12, "D": 13}, "E")


def test_scoring_rejects_a_multi_position_tensor() -> None:
    with pytest.raises(ValueError, match="1-D"):
        score_logits(torch.zeros(3, 100), {"A": 10, "B": 11, "C": 12, "D": 13}, "A")


def test_target_option_rate_counts_the_target_position() -> None:
    assert target_option_rate(["C", "C", "A", "B"], "C") == pytest.approx(0.5)


def test_target_option_rate_rejects_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="zero predictions"):
        target_option_rate([], "C")
