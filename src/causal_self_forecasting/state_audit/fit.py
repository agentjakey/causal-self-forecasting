"""Fitting the three ridges, with grouped cross-validation over training prompt groups.

Ridge regression only. No MLP, no gradient method, no ensembling, no early stopping on any
held-out signal. That is a preregistered constraint, not a simplification: an unconstrained model
class would make "the state helped" indistinguishable from "we searched until something helped".

Regularization is chosen by six-fold grouped cross-validation over the 96 training prompt groups,
16 groups per fold, assigned deterministically from the master seed. No `group_id` appears in both
the fit and the held-out part of a fold, so a fold cannot score itself on a paraphrase of
something it trained on.

Rows are (prompt, candidate) pairs over the **16 non-no-op** candidates per training prompt:
96 x 16 = 1,536. The no-op is predicted and scored but never fitted on, because an exact zero row
repeated 96 times would pull the intercept toward the no-op and flatter every method equally.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

from ..hashing import hash_object
from ..reproducibility import derive_seed
from ..schemas import FeatureBlock, RidgeFoldResult, RidgeSelectionRecord

# The preregistered grid and fold count. Changing either after seeing training results is a
# researcher degree of freedom, so both are constants rather than parameters.
RIDGE_ALPHA_GRID: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
CV_FOLD_COUNT = 6
CV_SEED_LABEL = "bluedot.cv_folds"

FIT_ALGORITHM_VERSION = "bluedot_grouped_cv_ridge_v1.0"


class FitError(RuntimeError):
    """Raised when a predictor cannot be fitted as specified."""


def assign_cv_folds(group_ids: Sequence[str], master_seed: int) -> dict[str, int]:
    """Split the training groups into six folds, deterministically.

    Sorted first so the input to the seeded step is order-independent, then ordered by a derived
    per-group key, then dealt round-robin. Dealing rather than slicing keeps the folds within one
    group of each other in size when the count does not divide evenly.
    """
    unique = sorted(set(group_ids))
    if len(unique) < CV_FOLD_COUNT:
        raise FitError(f"{len(unique)} training groups cannot be split into {CV_FOLD_COUNT} folds")
    ordered = sorted(unique, key=lambda gid: (derive_seed(CV_SEED_LABEL, master_seed, gid), gid))
    return {group_id: index % CV_FOLD_COUNT for index, group_id in enumerate(ordered)}


def _fit_ridge(features: np.ndarray, targets: np.ndarray, alpha: float) -> Ridge:
    model = Ridge(alpha=alpha, fit_intercept=True, solver="auto", random_state=None)
    model.fit(features, targets)
    return model


@dataclass(frozen=True)
class FittedPredictor:
    """One fitted ridge, frozen with everything needed to describe and reuse it."""

    method_id: str
    blocks: tuple[FeatureBlock, ...]
    model: Ridge
    selection: RidgeSelectionRecord
    residual_q05: float
    residual_q95: float
    flip_base_rate: float
    training_rows: int
    training_prompt_count: int

    @property
    def feature_dim(self) -> int:
        return int(self.model.coef_.shape[0])

    def coefficient_hash(self) -> str:
        """Hash the fitted coefficients and intercept, in float32."""
        return hash_object(
            {
                "method_id": self.method_id,
                "algorithm_version": FIT_ALGORITHM_VERSION,
                "blocks": [block.value for block in self.blocks],
                "ridge_alpha": float(self.selection.selected_alpha),
                "coef": np.asarray(self.model.coef_, dtype=np.float32).tolist(),
                "intercept": float(np.asarray(self.model.intercept_, dtype=np.float32)),
            }
        )

    def predict(self, features: np.ndarray) -> float:
        row = np.asarray(features, dtype=np.float64).reshape(1, -1)
        if row.shape[1] != self.feature_dim:
            raise FitError(
                f"{self.method_id} was fitted on {self.feature_dim} features but received "
                f"{row.shape[1]}"
            )
        value = float(self.model.predict(row)[0])
        if not math.isfinite(value):
            raise FitError(f"{self.method_id} produced a non-finite prediction")
        return value


def select_ridge_alpha(
    method_id: str,
    features: np.ndarray,
    targets: np.ndarray,
    group_ids: Sequence[str],
    master_seed: int,
) -> RidgeSelectionRecord:
    """Choose the regularization strength by grouped cross-validation.

    Lowest mean absolute error across the six folds. Ties break toward the **stronger**
    regularizer, which is the conservative direction: given two settings that fit the training
    groups equally well, the more constrained one is less able to have memorized them.
    """
    folds = assign_cv_folds(group_ids, master_seed)
    assignments = np.asarray([folds[group_id] for group_id in group_ids])

    fold_results: list[RidgeFoldResult] = []
    mae_by_alpha: dict[str, float] = {}

    for alpha in RIDGE_ALPHA_GRID:
        fold_errors: list[float] = []
        for fold in range(CV_FOLD_COUNT):
            held_out = assignments == fold
            fit_mask = ~held_out
            if not held_out.any() or not fit_mask.any():
                raise FitError(f"fold {fold} is empty for method {method_id}")

            fit_groups = {g for g, m in zip(group_ids, fit_mask, strict=True) if m}
            out_groups = {g for g, m in zip(group_ids, held_out, strict=True) if m}
            overlap = fit_groups & out_groups
            if overlap:
                raise FitError(
                    f"fold {fold} has {len(overlap)} groups on both sides, for example "
                    f"{sorted(overlap)[:3]}"
                )

            model = _fit_ridge(features[fit_mask], targets[fit_mask], alpha)
            predicted = model.predict(features[held_out])
            error = float(np.mean(np.abs(predicted - targets[held_out])))
            fold_errors.append(error)
            fold_results.append(
                RidgeFoldResult(
                    fold_index=fold,
                    ridge_alpha=alpha,
                    held_out_groups=len(out_groups),
                    held_out_rows=int(held_out.sum()),
                    fit_rows=int(fit_mask.sum()),
                    mean_absolute_error=error,
                )
            )
        mae_by_alpha[f"{alpha:g}"] = float(np.mean(fold_errors))

    best = min(mae_by_alpha.values())
    # Ties toward the stronger regularizer: scan the ascending grid and keep the last match.
    selected = max(
        alpha for alpha in RIDGE_ALPHA_GRID if math.isclose(mae_by_alpha[f"{alpha:g}"], best)
    )

    return RidgeSelectionRecord(
        method_id=method_id,
        alpha_grid=list(RIDGE_ALPHA_GRID),
        fold_count=CV_FOLD_COUNT,
        fold_results=fold_results,
        mean_absolute_error_by_alpha=mae_by_alpha,
        selected_alpha=selected,
        selected_alpha_mae=best,
        alpha_at_grid_edge=selected in (min(RIDGE_ALPHA_GRID), max(RIDGE_ALPHA_GRID)),
    )


def fit_predictor(
    method_id: str,
    blocks: Sequence[FeatureBlock],
    features: np.ndarray,
    targets: np.ndarray,
    flips: Sequence[bool],
    group_ids: Sequence[str],
    master_seed: int,
) -> FittedPredictor:
    """Select alpha by grouped CV, then fit final coefficients on all training groups."""
    if features.ndim != 2:
        raise FitError(f"{method_id}: features must be 2-D, got shape {features.shape}")
    if features.shape[0] != targets.shape[0]:
        raise FitError(f"{method_id}: {features.shape[0]} rows against {targets.shape[0]} targets")
    if len(group_ids) != features.shape[0]:
        raise FitError(f"{method_id}: {len(group_ids)} group labels for {features.shape[0]} rows")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise FitError(f"{method_id}: the training matrix contains non-finite values")

    selection = select_ridge_alpha(method_id, features, targets, group_ids, master_seed)
    model = _fit_ridge(features, targets, selection.selected_alpha)

    residuals = targets - model.predict(features)
    # Placeholders, and labeled as such wherever they are used. A ridge fitted on a continuous
    # target estimates neither an interval nor a flip probability; these are the training-residual
    # spread and the training flip base rate, which is what the preregistration specifies.
    q05 = float(np.quantile(residuals, 0.05, method="linear"))
    q95 = float(np.quantile(residuals, 0.95, method="linear"))
    base_rate = float(np.mean([1.0 if flip else 0.0 for flip in flips])) if flips else 0.0

    return FittedPredictor(
        method_id=method_id,
        blocks=tuple(blocks),
        model=model,
        selection=selection,
        residual_q05=min(q05, q95),
        residual_q95=max(q05, q95),
        flip_base_rate=min(1.0, max(0.0, base_rate)),
        training_rows=int(features.shape[0]),
        training_prompt_count=len(set(group_ids)),
    )


__all__ = [
    "CV_FOLD_COUNT",
    "CV_SEED_LABEL",
    "FIT_ALGORITHM_VERSION",
    "RIDGE_ALPHA_GRID",
    "FitError",
    "FittedPredictor",
    "assign_cv_folds",
    "fit_predictor",
    "select_ridge_alpha",
]
