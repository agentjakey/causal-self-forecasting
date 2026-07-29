"""The block-structured feature system for the state-dependence arm.

Four blocks, named so that a substitution can replace one and provably nothing else:

| Block | Width | Contents |
| `I` intervention | 16 | `P^T v`, the fixed projection of the signed intervention vector |
| `V` visible | 39 | centered clean logits (4), clean top margin (1), clean entropy (1),
  prompt length (1), TF-IDF reduced by truncated SVD (32) |
| `S` state | 16 | PCA of the 1152-dimensional clean residual state |
| `SxI` interaction | 256 | flattened outer product of S and I, in a fixed index order |


Three methods, differing only in which blocks they receive:

* `intervention_only_ridge`: I, 16 features
* `visible_information_ridge`: I + V, 55 features
* `state_bilinear_ridge`: I + V + S + SxI, 327 features

`forecasting.TrainingExample` is untouched. That type and its leakage fence belong to the
benchmark's baselines, which read a prompt and a public view and nothing else; widening it to
carry a hidden state would break the guarantee the audit test exists to hold. This arm gets its
own example type instead.

**The visible block contains no state-derived quantity.** Not the state norm, not a projection of
it, and not the intervention strength scaled by it. This is the whole reason the intervention
strength is a global constant: `public_view` publishes strength to every method, and a
prompt-relative strength would smuggle the prompt's state norm into V. `visible_block_is_clean`
checks the property directly rather than trusting the construction.

Every fitted object here is fitted on the 96 training prompts only, and each carries a
`TransformFitRecord` naming the exact prompt ids it saw.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from ..hashing import hash_object
from ..schemas import (
    FeatureBlock,
    PromptRole,
    StateAuditObservationRecord,
    TransformFitRecord,
)
from ..state_audit_target import ANSWER_LABELS, preferred_margin

# Preregistered widths. Asserted rather than inferred, because a block that silently changed
# width would change what the comparison is between.
INTERVENTION_WIDTH = 16
STATE_WIDTH = 16
SVD_COMPONENTS = 32
# 4 centered logits + top margin + entropy + prompt length + SVD components.
VISIBLE_WIDTH = 4 + 1 + 1 + 1 + SVD_COMPONENTS
INTERACTION_WIDTH = STATE_WIDTH * INTERVENTION_WIDTH

BLOCK_WIDTHS: dict[FeatureBlock, int] = {
    FeatureBlock.INTERVENTION: INTERVENTION_WIDTH,
    FeatureBlock.VISIBLE: VISIBLE_WIDTH,
    FeatureBlock.STATE: STATE_WIDTH,
    FeatureBlock.STATE_INTERVENTION: INTERACTION_WIDTH,
}

METHOD_BLOCKS: dict[str, tuple[FeatureBlock, ...]] = {
    "intervention_only_ridge": (FeatureBlock.INTERVENTION,),
    "visible_information_ridge": (FeatureBlock.INTERVENTION, FeatureBlock.VISIBLE),
    "state_bilinear_ridge": (
        FeatureBlock.INTERVENTION,
        FeatureBlock.VISIBLE,
        FeatureBlock.STATE,
        FeatureBlock.STATE_INTERVENTION,
    ),
}

TFIDF_TRANSFORM_ID = "bluedot_tfidf_svd_v1"
STATE_PCA_TRANSFORM_ID = "bluedot_state_pca_v1"
STANDARDIZER_TRANSFORM_ID = "bluedot_standardizer_v1"

# Below this a feature is constant across the training rows and standardizing it would divide by
# round-off. Constant features are centered and left at zero scale.
MIN_FEATURE_SCALE = 1e-12


class FeatureError(RuntimeError):
    """Raised when a feature block cannot be built as specified."""


@dataclass(frozen=True)
class PromptContext:
    """Everything about one prompt that features are built from, before any intervention.

    Assembled from the clean pass, so it exists for training and final-test prompts alike and is
    identical whether the prompt is about to be predicted under its own state or a donor's.
    """

    variant_id: str
    group_id: str
    trial_id: str
    prompt_text: str
    prompt_token_count: int
    clean_logits: dict[str, float]
    clean_preferred_label: str
    clean_top_margin: float
    clean_entropy: float
    state: np.ndarray


@dataclass(frozen=True)
class StateAuditExample:
    """One (prompt, candidate) row for the state-audit predictors.

    Deliberately separate from `forecasting.TrainingExample`. That type is fenced to public
    information and is what the benchmark's prompt-only baselines consume; this one carries a
    hidden state on purpose, and the two must not be confused.
    """

    variant_id: str
    group_id: str
    trial_id: str
    candidate_id: str
    is_noop: bool
    intervention_vector: np.ndarray
    observed_target: float
    observed_flip: bool


def centered_answer_logits(logits: dict[str, float]) -> list[float]:
    """The four label logits minus their own mean.

    Centering removes the overall scale of the logit vector, which carries no information about
    the answer's relative standing and would otherwise dominate a standardized feature.
    """
    values = [float(logits[label]) for label in ANSWER_LABELS]
    mean = sum(values) / len(values)
    return [value - mean for value in values]


def intervention_block(projection: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """`P^T v`. Identical for every method, by construction."""
    dense = np.asarray(projection, dtype=np.float64)
    operand = np.asarray(vector, dtype=np.float64)
    if operand.shape != (dense.shape[0],):
        raise FeatureError(
            f"intervention vector has shape {operand.shape}, expected {(dense.shape[0],)}"
        )
    block = dense.T @ operand
    if block.shape != (INTERVENTION_WIDTH,):
        raise FeatureError(f"intervention block has width {block.shape[0]}, expected 16")
    return block


def interaction_block(state: np.ndarray, intervention: np.ndarray) -> np.ndarray:
    """Flattened outer product of S and I, row-major: index `i * 16 + j` is `S[i] * I[j]`."""
    left = np.asarray(state, dtype=np.float64)
    right = np.asarray(intervention, dtype=np.float64)
    if left.shape != (STATE_WIDTH,) or right.shape != (INTERVENTION_WIDTH,):
        raise FeatureError(
            f"interaction needs a {STATE_WIDTH}-dim state and a {INTERVENTION_WIDTH}-dim "
            f"intervention, got {left.shape} and {right.shape}"
        )
    return np.outer(left, right).reshape(-1)


class FittedTransforms:
    """The train-only transforms, fitted once and frozen.

    Holds the TF-IDF vectorizer and its truncated SVD, the state PCA, and the feature
    standardizer. Nothing here is refitted at prediction time; `transform_*` only applies.
    """

    def __init__(
        self,
        tfidf: TfidfVectorizer,
        svd: TruncatedSVD,
        state_pca: PCA,
        feature_mean: dict[str, np.ndarray],
        feature_scale: dict[str, np.ndarray],
        fit_prompt_ids: list[str],
        realized_svd_components: int,
    ) -> None:
        self.tfidf = tfidf
        self.svd = svd
        self.state_pca = state_pca
        self.feature_mean = feature_mean
        self.feature_scale = feature_scale
        self.fit_prompt_ids = list(fit_prompt_ids)
        self.realized_svd_components = realized_svd_components

    # -- application ------------------------------------------------------

    def visible_block(self, context: PromptContext) -> np.ndarray:
        """The V block for one prompt. Contains no state-derived quantity."""
        text = self.svd.transform(self.tfidf.transform([context.prompt_text]))[0]
        padded = np.zeros(SVD_COMPONENTS, dtype=np.float64)
        padded[: text.shape[0]] = text
        block = np.concatenate(
            [
                np.asarray(centered_answer_logits(context.clean_logits), dtype=np.float64),
                np.asarray(
                    [
                        context.clean_top_margin,
                        context.clean_entropy,
                        float(context.prompt_token_count),
                    ],
                    dtype=np.float64,
                ),
                padded,
            ]
        )
        if block.shape != (VISIBLE_WIDTH,):
            raise FeatureError(
                f"visible block has width {block.shape[0]}, expected {VISIBLE_WIDTH}"
            )
        return block

    def state_block(self, state: np.ndarray) -> np.ndarray:
        """The S block: the train-fitted PCA of one 1152-dimensional clean state."""
        reduced = self.state_pca.transform(np.asarray(state, dtype=np.float64).reshape(1, -1))[0]
        if reduced.shape != (STATE_WIDTH,):
            raise FeatureError(f"state block has width {reduced.shape[0]}, expected {STATE_WIDTH}")
        return reduced

    def standardize(self, block: FeatureBlock, values: np.ndarray) -> np.ndarray:
        """Center and scale one block by the training mean and standard deviation."""
        mean = self.feature_mean[block.value]
        scale = self.feature_scale[block.value]
        return (np.asarray(values, dtype=np.float64) - mean) / scale

    # -- provenance -------------------------------------------------------

    def fit_records(
        self,
        study_id: str,
        master_seed: int,
        prompt_manifest_hash: str,
        fit_row_count: int,
    ) -> list[TransformFitRecord]:
        identity = hash_object(self.fit_prompt_ids)
        common: dict[str, Any] = {
            "study_id": study_id,
            "fit_prompt_role": PromptRole.TRAINING,
            "fit_prompt_ids": self.fit_prompt_ids,
            "fit_prompt_identity_hash": identity,
            "fit_row_count": fit_row_count,
            "master_seed": master_seed,
            "prompt_manifest_hash": prompt_manifest_hash,
        }
        vocabulary = sorted(self.tfidf.vocabulary_)
        tfidf_state = {
            "vocabulary": vocabulary,
            "idf": [float(v) for v in self.tfidf.idf_],
            "svd_components": np.asarray(self.svd.components_, dtype=np.float32).tolist(),
        }
        pca_state = {
            "mean": np.asarray(self.state_pca.mean_, dtype=np.float32).tolist(),
            "components": np.asarray(self.state_pca.components_, dtype=np.float32).tolist(),
        }
        standardizer_state = {
            "mean": {
                k: np.asarray(v, dtype=np.float32).tolist()
                for k, v in sorted(self.feature_mean.items())
            },
            "scale": {
                k: np.asarray(v, dtype=np.float32).tolist()
                for k, v in sorted(self.feature_scale.items())
            },
        }
        return [
            TransformFitRecord(
                transform_id=TFIDF_TRANSFORM_ID,
                kind="tfidf_truncated_svd",
                output_dim=SVD_COMPONENTS,
                parameters={
                    "vocabulary_size": len(vocabulary),
                    "requested_components": SVD_COMPONENTS,
                    "realized_components": self.realized_svd_components,
                },
                transform_hash=hash_object(tfidf_state),
                **common,
            ),
            TransformFitRecord(
                transform_id=STATE_PCA_TRANSFORM_ID,
                kind="state_pca",
                output_dim=STATE_WIDTH,
                parameters={
                    "components": STATE_WIDTH,
                    "input_dim": int(np.asarray(self.state_pca.components_).shape[1]),
                },
                transform_hash=hash_object(pca_state),
                **common,
            ),
            TransformFitRecord(
                transform_id=STANDARDIZER_TRANSFORM_ID,
                kind="feature_standardizer",
                output_dim=sum(BLOCK_WIDTHS.values()),
                parameters={"blocks": sorted(self.feature_mean)},
                transform_hash=hash_object(standardizer_state),
                **common,
            ),
        ]


def fit_transforms(
    contexts: Sequence[PromptContext],
    examples: Sequence[StateAuditExample],
    projection: np.ndarray,
) -> FittedTransforms:
    """Fit every transform on the training prompts.

    `k_svd = min(32, n_features - 1, 95)` is the preregistered rule; the realized value is
    recorded. When the vocabulary is small enough that fewer than 32 components exist, the block
    is zero-padded to keep the width fixed, because a comparison between two methods whose
    feature widths depend on the corpus is not a comparison.
    """
    if not contexts:
        raise FeatureError("cannot fit transforms with no training prompts")
    if not examples:
        raise FeatureError("cannot fit transforms with no training rows")

    prompt_ids = sorted(context.variant_id for context in contexts)
    if len(set(prompt_ids)) != len(prompt_ids):
        raise FeatureError("the training contexts repeat a prompt")

    texts = [context.prompt_text for context in contexts]
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(texts)
    n_features = int(cast(Any, matrix).shape[1])
    k_svd = min(SVD_COMPONENTS, max(1, n_features - 1), 95)
    svd = TruncatedSVD(n_components=k_svd, random_state=0)
    svd.fit(matrix)

    states = np.stack([np.asarray(c.state, dtype=np.float64) for c in contexts], axis=0)
    if states.shape[0] <= STATE_WIDTH:
        raise FeatureError(
            f"{states.shape[0]} training states cannot support a {STATE_WIDTH}-component PCA"
        )
    state_pca = PCA(n_components=STATE_WIDTH, random_state=0)
    state_pca.fit(states)

    fitted = FittedTransforms(
        tfidf=tfidf,
        svd=svd,
        state_pca=state_pca,
        feature_mean={},
        feature_scale={},
        fit_prompt_ids=prompt_ids,
        realized_svd_components=k_svd,
    )

    # Standardizer statistics come from the same rows the ridges are fitted on: the non-no-op
    # training candidates. The no-op is predicted but never fitted on, so an exact zero row
    # cannot pull the intercept or shrink a scale.
    by_variant = {context.variant_id: context for context in contexts}
    raw: dict[str, list[np.ndarray]] = {block.value: [] for block in BLOCK_WIDTHS}
    for example in examples:
        if example.is_noop:
            continue
        context = by_variant[example.variant_id]
        block_i = intervention_block(projection, example.intervention_vector)
        block_v = fitted.visible_block(context)
        block_s = fitted.state_block(context.state)
        raw[FeatureBlock.INTERVENTION.value].append(block_i)
        raw[FeatureBlock.VISIBLE.value].append(block_v)
        raw[FeatureBlock.STATE.value].append(block_s)
        raw[FeatureBlock.STATE_INTERVENTION.value].append(interaction_block(block_s, block_i))

    for name, rows in raw.items():
        if not rows:
            raise FeatureError(f"no non-no-op training rows produced a {name} block")
        stacked = np.stack(rows, axis=0)
        mean = stacked.mean(axis=0)
        scale = stacked.std(axis=0)
        scale = np.where(scale < MIN_FEATURE_SCALE, 1.0, scale)
        fitted.feature_mean[name] = mean
        fitted.feature_scale[name] = scale

    return fitted


def build_feature_row(
    transforms: FittedTransforms,
    context: PromptContext,
    intervention_vector: np.ndarray,
    projection: np.ndarray,
    blocks: Sequence[FeatureBlock],
    state_override: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble one standardized feature row.

    `state_override` is the wrong-state substitution. It replaces the state that feeds S and,
    through it, SxI. Every other block is built from `context` exactly as it would have been,
    which is the property `substitution_changes_only_state_blocks` checks.
    """
    block_i = transforms.standardize(
        FeatureBlock.INTERVENTION, intervention_block(projection, intervention_vector)
    )
    parts: list[np.ndarray] = []
    raw_state = context.state if state_override is None else state_override
    reduced_state: np.ndarray | None = None

    for block in blocks:
        if block is FeatureBlock.INTERVENTION:
            parts.append(block_i)
        elif block is FeatureBlock.VISIBLE:
            parts.append(
                transforms.standardize(FeatureBlock.VISIBLE, transforms.visible_block(context))
            )
        elif block is FeatureBlock.STATE:
            reduced_state = transforms.state_block(raw_state)
            parts.append(transforms.standardize(FeatureBlock.STATE, reduced_state))
        elif block is FeatureBlock.STATE_INTERVENTION:
            if reduced_state is None:
                reduced_state = transforms.state_block(raw_state)
            # Built from the standardized-in intervention's *raw* counterpart, so the outer
            # product is of the two blocks as they are defined rather than of two rescaled
            # copies. The product is then standardized as its own block.
            raw_i = intervention_block(projection, intervention_vector)
            parts.append(
                transforms.standardize(
                    FeatureBlock.STATE_INTERVENTION, interaction_block(reduced_state, raw_i)
                )
            )
        else:
            raise FeatureError(f"unhandled feature block {block}")

    row = np.concatenate(parts)
    expected = sum(BLOCK_WIDTHS[block] for block in blocks)
    if row.shape != (expected,):
        raise FeatureError(f"feature row has width {row.shape[0]}, expected {expected}")
    if not np.isfinite(row).all():
        raise FeatureError("a feature row contains non-finite values")
    return row


def visible_block_is_clean(
    transforms: FittedTransforms,
    context: PromptContext,
    other_state: np.ndarray,
) -> bool:
    """Confirm the V block does not move when only the state changes.

    A direct check of the property the whole design rests on. If any state-derived quantity had
    leaked into V, handing the same prompt a different state would change it.
    """
    baseline = transforms.visible_block(context)
    swapped = PromptContext(
        variant_id=context.variant_id,
        group_id=context.group_id,
        trial_id=context.trial_id,
        prompt_text=context.prompt_text,
        prompt_token_count=context.prompt_token_count,
        clean_logits=dict(context.clean_logits),
        clean_preferred_label=context.clean_preferred_label,
        clean_top_margin=context.clean_top_margin,
        clean_entropy=context.clean_entropy,
        state=np.asarray(other_state, dtype=np.float64),
    )
    return bool(np.array_equal(baseline, transforms.visible_block(swapped)))


def context_from_observation(
    record: StateAuditObservationRecord,
    prompt_text: str,
    prompt_token_count: int,
    clean_entropy: float,
    state: np.ndarray,
) -> PromptContext:
    """Build a prompt context from a stored observation and its clean pass."""
    margin = preferred_margin(record.clean_logits, record.clean_preferred_label)
    if not math.isclose(margin, record.clean_top_margin, abs_tol=1e-9):
        raise FeatureError(
            f"{record.variant_id}: the stored clean top margin does not recompute from its logits"
        )
    return PromptContext(
        variant_id=record.variant_id,
        group_id=record.group_id,
        trial_id=record.trial_id,
        prompt_text=prompt_text,
        prompt_token_count=prompt_token_count,
        clean_logits=dict(record.clean_logits),
        clean_preferred_label=record.clean_preferred_label,
        clean_top_margin=record.clean_top_margin,
        clean_entropy=clean_entropy,
        state=np.asarray(state, dtype=np.float64),
    )


__all__ = [
    "BLOCK_WIDTHS",
    "INTERACTION_WIDTH",
    "INTERVENTION_WIDTH",
    "METHOD_BLOCKS",
    "STANDARDIZER_TRANSFORM_ID",
    "STATE_PCA_TRANSFORM_ID",
    "STATE_WIDTH",
    "SVD_COMPONENTS",
    "TFIDF_TRANSFORM_ID",
    "VISIBLE_WIDTH",
    "FeatureError",
    "FittedTransforms",
    "PromptContext",
    "StateAuditExample",
    "build_feature_row",
    "centered_answer_logits",
    "context_from_observation",
    "fit_transforms",
    "interaction_block",
    "intervention_block",
    "visible_block_is_clean",
]
