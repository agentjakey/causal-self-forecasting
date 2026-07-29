"""Unit tests for the projection, the feature blocks, the ridges, and the wrong-state matching.

No model and no run artifacts. Everything here is arithmetic over synthetic states and vectors,
which is the right level for the properties that carry the comparison: block widths, what a
substitution is allowed to move, and whether the visible model can see the state.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_self_forecasting.schemas import FeatureBlock
from causal_self_forecasting.state_audit.features import (
    BLOCK_WIDTHS,
    INTERVENTION_WIDTH,
    METHOD_BLOCKS,
    STATE_WIDTH,
    FeatureError,
    PromptContext,
    StateAuditExample,
    build_feature_row,
    centered_answer_logits,
    fit_transforms,
    interaction_block,
    intervention_block,
    visible_block_is_clean,
)
from causal_self_forecasting.state_audit.fit import (
    CV_FOLD_COUNT,
    RIDGE_ALPHA_GRID,
    FitError,
    assign_cv_folds,
    fit_predictor,
    select_ridge_alpha,
)
from causal_self_forecasting.state_audit.matching import (
    MatchingError,
    build_derangement,
    build_matches,
    build_pairing,
    donor_for,
    nearest_matched_donor,
)
from causal_self_forecasting.state_audit.projection import (
    ProjectionError,
    build_projection_matrix,
    injectivity_margin,
    orthonormality_error,
    project,
    projection_seed,
)

HIDDEN = 64
MASTER_SEED = 20260727
WORDS = ["water", "energy", "rock", "heat", "light", "gas", "cell", "force"]


def make_projection(dim: int = HIDDEN) -> np.ndarray:
    return build_projection_matrix(dim, INTERVENTION_WIDTH, projection_seed(MASTER_SEED, "study"))


def make_context(index: int, dim: int = HIDDEN, rng: np.random.Generator | None = None):
    generator = rng or np.random.default_rng(1000 + index)
    logits = {
        "A": 3.0 + 0.1 * index,
        "B": 1.0 - 0.05 * index,
        "C": 0.5,
        "D": 0.0 + 0.02 * index,
    }
    top = max(logits.values())
    others = sorted(logits.values())[:-1]
    return PromptContext(
        variant_id=f"p{index:03d}.neutral_a",
        group_id=f"p{index:03d}",
        trial_id=f"sa_{index:05d}",
        prompt_text=" ".join(WORDS[: 3 + index % 5]) + f" question {index}",
        prompt_token_count=10 + index % 7,
        clean_logits=logits,
        clean_preferred_label="A",
        clean_top_margin=top - max(others),
        clean_entropy=0.5 + 0.01 * index,
        state=generator.standard_normal(dim),
    )


def make_examples(contexts, dim: int = HIDDEN, per_prompt: int = 16):
    generator = np.random.default_rng(7)
    directions = [generator.standard_normal(dim) for _ in range(per_prompt // 2)]
    examples = []
    for context in contexts:
        for index, direction in enumerate(directions):
            for sign in (1.0, -1.0):
                vector = sign * 2.0 * direction
                examples.append(
                    StateAuditExample(
                        variant_id=context.variant_id,
                        group_id=context.group_id,
                        trial_id=context.trial_id,
                        candidate_id=f"{context.trial_id}.opaque_{index:02d}_{sign:+.0f}",
                        is_noop=False,
                        intervention_vector=vector,
                        observed_target=float(vector[:4].sum() + context.clean_top_margin * 0.1),
                        observed_flip=bool(vector[0] > 1.0),
                    )
                )
    return examples


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def test_the_projection_has_orthonormal_columns() -> None:
    matrix = make_projection()
    assert matrix.shape == (HIDDEN, INTERVENTION_WIDTH)
    assert orthonormality_error(matrix) < 1e-5


def test_the_projection_is_deterministic_from_the_seed() -> None:
    assert np.array_equal(make_projection(), make_projection())


def test_a_different_seed_gives_a_different_projection() -> None:
    other = build_projection_matrix(HIDDEN, INTERVENTION_WIDTH, projection_seed(1, "study"))
    assert not np.array_equal(make_projection(), other)


def test_the_projection_refuses_more_columns_than_dimensions() -> None:
    with pytest.raises(ProjectionError, match="orthonormal columns"):
        build_projection_matrix(8, 16, 1)


def test_the_projection_is_injective_on_the_realized_signed_set() -> None:
    """The signed set is rank 8, not rank 16, because `+d` and `-d` are collinear."""
    matrix = make_projection()
    generator = np.random.default_rng(3)
    directions = [generator.standard_normal(HIDDEN) for _ in range(8)]
    vectors = [sign * d for d in directions for sign in (1.0, -1.0)]
    signed_margin = injectivity_margin(matrix, vectors)
    unsigned_margin = injectivity_margin(matrix, directions)
    assert signed_margin > 1e-6
    assert unsigned_margin > 1e-6
    # Adding the collinear negatives adds no dimension, so the rank is unchanged and only the
    # scale moves: stacking each row with its negation doubles the Gram matrix.
    assert signed_margin == pytest.approx(unsigned_margin * np.sqrt(2.0))


def test_the_projection_refuses_a_set_it_would_collapse() -> None:
    """A rank-loss check, not a formality: a 2-column projection cannot carry 3 dimensions."""
    narrow = build_projection_matrix(HIDDEN, 2, projection_seed(MASTER_SEED, "narrow"))
    generator = np.random.default_rng(11)
    vectors = [generator.standard_normal(HIDDEN) for _ in range(3)]
    with pytest.raises(ProjectionError, match="loses rank"):
        injectivity_margin(narrow, vectors)


def test_projecting_the_zero_vector_gives_zero() -> None:
    assert np.allclose(project(make_projection(), np.zeros(HIDDEN)), 0.0)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def test_the_preregistered_block_widths() -> None:
    assert BLOCK_WIDTHS[FeatureBlock.INTERVENTION] == 16
    assert BLOCK_WIDTHS[FeatureBlock.VISIBLE] == 39
    assert BLOCK_WIDTHS[FeatureBlock.STATE] == 16
    assert BLOCK_WIDTHS[FeatureBlock.STATE_INTERVENTION] == 256


def test_the_preregistered_method_widths() -> None:
    widths = {
        method: sum(BLOCK_WIDTHS[block] for block in blocks)
        for method, blocks in METHOD_BLOCKS.items()
    }
    assert widths == {
        "intervention_only_ridge": 16,
        "visible_information_ridge": 55,
        "state_bilinear_ridge": 327,
    }


def test_centered_logits_sum_to_zero() -> None:
    centered = centered_answer_logits({"A": 3.0, "B": 1.0, "C": 0.5, "D": 0.0})
    assert len(centered) == 4
    assert abs(sum(centered)) < 1e-12


def test_the_interaction_block_is_the_outer_product_in_a_fixed_order() -> None:
    state = np.arange(STATE_WIDTH, dtype=np.float64)
    intervention = np.arange(INTERVENTION_WIDTH, dtype=np.float64) + 1.0
    block = interaction_block(state, intervention)
    assert block.shape == (256,)
    for i in range(STATE_WIDTH):
        for j in range(INTERVENTION_WIDTH):
            assert block[i * INTERVENTION_WIDTH + j] == state[i] * intervention[j]


def test_the_intervention_block_refuses_a_wrong_shaped_vector() -> None:
    with pytest.raises(FeatureError, match="intervention vector has shape"):
        intervention_block(make_projection(), np.zeros(HIDDEN + 1))


# ---------------------------------------------------------------------------
# Fitted transforms and the substitution policy
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted():
    contexts = [make_context(i) for i in range(24)]
    examples = make_examples(contexts)
    projection = make_projection()
    return contexts, examples, projection, fit_transforms(contexts, examples, projection)


def test_the_intervention_block_is_shared_identically_by_every_method(fitted) -> None:
    """The comparison is about what the state adds, not about who sees the intervention better."""
    contexts, examples, projection, transforms = fitted
    context, example = contexts[0], examples[0]
    rows = {
        method: build_feature_row(
            transforms, context, example.intervention_vector, projection, blocks
        )
        for method, blocks in METHOD_BLOCKS.items()
    }
    shared = rows["intervention_only_ridge"][:INTERVENTION_WIDTH]
    for method, row in rows.items():
        assert np.array_equal(row[:INTERVENTION_WIDTH], shared), method


def test_the_visible_block_does_not_move_when_only_the_state_changes(fitted) -> None:
    contexts, _, _, transforms = fitted
    assert visible_block_is_clean(transforms, contexts[0], contexts[5].state)


def test_a_wrong_state_substitution_changes_only_the_state_and_interaction_blocks(fitted) -> None:
    contexts, examples, projection, transforms = fitted
    blocks = METHOD_BLOCKS["state_bilinear_ridge"]
    vector = examples[0].intervention_vector

    true_row = build_feature_row(transforms, contexts[0], vector, projection, blocks)
    swapped = build_feature_row(
        transforms, contexts[0], vector, projection, blocks, state_override=contexts[9].state
    )

    offset = 0
    moved = []
    for block in blocks:
        width = BLOCK_WIDTHS[block]
        piece = slice(offset, offset + width)
        if not np.array_equal(true_row[piece], swapped[piece]):
            moved.append(block.value)
        offset += width

    assert set(moved) == {FeatureBlock.STATE.value, FeatureBlock.STATE_INTERVENTION.value}


def test_the_visible_model_row_is_identical_under_a_substitution(fitted) -> None:
    """The visible model has no state block, so a donor state cannot reach it at all."""
    contexts, examples, projection, transforms = fitted
    blocks = METHOD_BLOCKS["visible_information_ridge"]
    vector = examples[0].intervention_vector
    baseline = build_feature_row(transforms, contexts[0], vector, projection, blocks)
    swapped = build_feature_row(
        transforms, contexts[0], vector, projection, blocks, state_override=contexts[3].state
    )
    assert np.array_equal(baseline, swapped)


def test_feature_rows_have_the_preregistered_width(fitted) -> None:
    contexts, examples, projection, transforms = fitted
    for method, blocks in METHOD_BLOCKS.items():
        row = build_feature_row(
            transforms, contexts[0], examples[0].intervention_vector, projection, blocks
        )
        assert row.shape == (sum(BLOCK_WIDTHS[b] for b in blocks),), method


def test_transform_fit_records_name_only_the_training_prompts(fitted) -> None:
    contexts, examples, _, transforms = fitted
    records = transforms.fit_records(
        study_id="study",
        master_seed=MASTER_SEED,
        prompt_manifest_hash="sha256:" + "1" * 64,
        fit_row_count=len(examples),
    )
    assert len(records) == 3
    expected = sorted(context.variant_id for context in contexts)
    for record in records:
        assert record.fit_prompt_role.value == "training"
        assert record.fit_prompt_ids == expected
        assert record.transform_hash.startswith("sha256:")


def test_fitting_refuses_too_few_states_for_a_sixteen_component_pca() -> None:
    contexts = [make_context(i) for i in range(8)]
    with pytest.raises(FeatureError, match="component PCA"):
        fit_transforms(contexts, make_examples(contexts), make_projection())


# ---------------------------------------------------------------------------
# Grouped cross-validation and the ridges
# ---------------------------------------------------------------------------


def test_folds_are_deterministic_and_cover_every_group() -> None:
    groups = [f"g{i:03d}" for i in range(96)]
    folds = assign_cv_folds(groups, MASTER_SEED)
    assert folds == assign_cv_folds(groups, MASTER_SEED)
    assert set(folds) == set(groups)
    assert sorted(set(folds.values())) == list(range(CV_FOLD_COUNT))


def test_ninety_six_groups_split_into_six_folds_of_sixteen() -> None:
    groups = [f"g{i:03d}" for i in range(96)]
    folds = assign_cv_folds(groups, MASTER_SEED)
    sizes = [sum(1 for f in folds.values() if f == fold) for fold in range(CV_FOLD_COUNT)]
    assert sizes == [16] * CV_FOLD_COUNT


def test_a_different_seed_gives_different_folds() -> None:
    groups = [f"g{i:03d}" for i in range(96)]
    assert assign_cv_folds(groups, MASTER_SEED) != assign_cv_folds(groups, 999)


def test_folds_refuse_too_few_groups() -> None:
    with pytest.raises(FitError, match="cannot be split"):
        assign_cv_folds(["a", "b"], MASTER_SEED)


def test_alpha_selection_reports_every_grid_point(fitted) -> None:
    contexts, examples, projection, transforms = fitted
    blocks = METHOD_BLOCKS["intervention_only_ridge"]
    features = np.stack(
        [
            build_feature_row(
                transforms,
                next(c for c in contexts if c.variant_id == e.variant_id),
                e.intervention_vector,
                projection,
                blocks,
            )
            for e in examples
        ]
    )
    targets = np.asarray([e.observed_target for e in examples])
    groups = [e.group_id for e in examples]

    selection = select_ridge_alpha("intervention_only_ridge", features, targets, groups, 20260727)
    assert selection.alpha_grid == list(RIDGE_ALPHA_GRID)
    assert set(selection.mean_absolute_error_by_alpha) == {f"{a:g}" for a in RIDGE_ALPHA_GRID}
    assert len(selection.fold_results) == CV_FOLD_COUNT * len(RIDGE_ALPHA_GRID)
    assert selection.selected_alpha in RIDGE_ALPHA_GRID
    assert selection.selected_alpha_mae == min(selection.mean_absolute_error_by_alpha.values())


def test_no_group_appears_on_both_sides_of_a_fold(fitted) -> None:
    contexts, examples, projection, transforms = fitted
    blocks = METHOD_BLOCKS["intervention_only_ridge"]
    features = np.stack(
        [
            build_feature_row(
                transforms,
                next(c for c in contexts if c.variant_id == e.variant_id),
                e.intervention_vector,
                projection,
                blocks,
            )
            for e in examples
        ]
    )
    targets = np.asarray([e.observed_target for e in examples])
    groups = [e.group_id for e in examples]
    folds = assign_cv_folds(groups, MASTER_SEED)
    for fold in range(CV_FOLD_COUNT):
        held = {g for g in groups if folds[g] == fold}
        fitted_groups = {g for g in groups if folds[g] != fold}
        assert held.isdisjoint(fitted_groups)
    # And the selector itself runs without tripping its own disjointness assertion.
    assert select_ridge_alpha("m", features, targets, groups, MASTER_SEED).fold_count == 6


def test_a_fitted_predictor_records_its_placeholders(fitted) -> None:
    contexts, examples, projection, transforms = fitted
    blocks = METHOD_BLOCKS["visible_information_ridge"]
    features = np.stack(
        [
            build_feature_row(
                transforms,
                next(c for c in contexts if c.variant_id == e.variant_id),
                e.intervention_vector,
                projection,
                blocks,
            )
            for e in examples
        ]
    )
    predictor = fit_predictor(
        "visible_information_ridge",
        blocks,
        features,
        np.asarray([e.observed_target for e in examples]),
        [e.observed_flip for e in examples],
        [e.group_id for e in examples],
        MASTER_SEED,
    )
    assert predictor.feature_dim == 55
    assert predictor.residual_q05 <= predictor.residual_q95
    assert 0.0 <= predictor.flip_base_rate <= 1.0
    assert predictor.coefficient_hash().startswith("sha256:")
    assert isinstance(predictor.predict(features[0]), float)


def test_a_predictor_refuses_a_row_of_the_wrong_width(fitted) -> None:
    contexts, examples, projection, transforms = fitted
    blocks = METHOD_BLOCKS["intervention_only_ridge"]
    features = np.stack(
        [
            build_feature_row(
                transforms,
                next(c for c in contexts if c.variant_id == e.variant_id),
                e.intervention_vector,
                projection,
                blocks,
            )
            for e in examples
        ]
    )
    predictor = fit_predictor(
        "intervention_only_ridge",
        blocks,
        features,
        np.asarray([e.observed_target for e in examples]),
        [e.observed_flip for e in examples],
        [e.group_id for e in examples],
        MASTER_SEED,
    )
    with pytest.raises(FitError, match="was fitted on"):
        predictor.predict(np.zeros(99))


# ---------------------------------------------------------------------------
# Wrong-state matching
# ---------------------------------------------------------------------------


def test_no_prompt_is_ever_its_own_donor() -> None:
    contexts = [make_context(i) for i in range(32)]
    for match in build_matches(contexts):
        assert match.variant_id != match.donor_variant_id


def test_matching_prefers_the_same_clean_preferred_answer() -> None:
    contexts = [make_context(i) for i in range(6)]
    odd = [
        PromptContext(**{**c.__dict__, "clean_preferred_label": "B"}) if index % 2 else c
        for index, c in enumerate(contexts)
    ]
    for match, target in zip(
        build_matches(odd), sorted(odd, key=lambda c: c.variant_id), strict=True
    ):
        donor = next(c for c in odd if c.variant_id == match.donor_variant_id)
        assert donor.clean_preferred_label == target.clean_preferred_label
        assert match.same_clean_preferred_label


def test_matching_minimizes_the_margin_distance() -> None:
    contexts = [make_context(i) for i in range(12)]
    target = contexts[0]
    donor_id, _, margin, _, _ = nearest_matched_donor(target, contexts)
    others = [c for c in contexts if c.variant_id != target.variant_id]
    best = min(abs(c.clean_top_margin - target.clean_top_margin) for c in others)
    assert margin == pytest.approx(best)
    assert donor_id != target.variant_id


def test_matching_is_deterministic() -> None:
    contexts = [make_context(i) for i in range(20)]
    assert build_matches(contexts) == build_matches(list(reversed(contexts)))


def test_matching_refuses_a_single_prompt() -> None:
    with pytest.raises(MatchingError, match="at least two"):
        build_matches([make_context(0)])


def test_a_derangement_has_no_fixed_point_and_is_a_bijection() -> None:
    ids = [f"p{i:03d}.neutral_a" for i in range(32)]
    for index in range(10):
        permutation = build_derangement(ids, MASTER_SEED, index)
        assert set(permutation) == set(ids)
        assert sorted(permutation.values()) == sorted(ids)
        assert all(key != value for key, value in permutation.items())


def test_derangements_are_deterministic_and_distinct() -> None:
    ids = [f"p{i:03d}.neutral_a" for i in range(32)]
    first = build_derangement(ids, MASTER_SEED, 0)
    assert first == build_derangement(ids, MASTER_SEED, 0)
    assert first != build_derangement(ids, MASTER_SEED, 1)


def test_the_pairing_record_carries_ten_permutations_and_a_hash() -> None:
    contexts = [make_context(i) for i in range(32)]
    pairing = build_pairing(
        pairing_id="pair_v1",
        study_id="study",
        layer=13,
        contexts=contexts,
        master_seed=MASTER_SEED,
        prompt_manifest_hash="sha256:" + "2" * 64,
        final_test_run_id="ft",
    )
    assert len(pairing.matches) == 32
    assert len(pairing.permutations) == 10
    assert pairing.pairing_hash.startswith("sha256:")
    assert pairing.prompt_role.value == "final_test"

    target = pairing.matches[0].variant_id
    assert donor_for(pairing, target, 0, shuffled=False) == pairing.matches[0].donor_variant_id
    assert donor_for(pairing, target, 3, shuffled=True) != target


def test_donor_lookup_refuses_an_out_of_range_permutation() -> None:
    contexts = [make_context(i) for i in range(8)]
    pairing = build_pairing(
        pairing_id="pair_v1",
        study_id="study",
        layer=13,
        contexts=contexts,
        master_seed=MASTER_SEED,
        prompt_manifest_hash="sha256:" + "2" * 64,
        final_test_run_id="ft",
    )
    with pytest.raises(MatchingError, match="outside the"):
        donor_for(pairing, pairing.matches[0].variant_id, 99, shuffled=True)
