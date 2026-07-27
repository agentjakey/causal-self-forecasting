"""Constant baseline.

Predicts the average training-set effect for a public intervention group. It conditions only
on the public features a forecaster is allowed to see: the operation form, the layer, and the
strength. It does not condition on the true mechanism, because that is private, so a real
bias-direction steer and a matched random control fall in the same group and receive the same
prediction. Its inability to tell them apart is exactly the gap a state-aware method must beat.

It is the floor. It validates the scoring path and gives every other method something to be
better than.
"""

from __future__ import annotations

import statistics
from typing import Any

from ..schemas import StateCondition
from .base import CandidatePrediction, Forecaster, TrainingExample

# A wide default band for groups with too few samples to estimate a spread. Deliberately
# uninformative rather than falsely precise.
_DEFAULT_HALF_WIDTH = 1.0
_MIN_SAMPLES_FOR_QUANTILES = 5


def _strength_bucket(strength: float) -> str:
    """Coarsen strength to sign, so groups are not fragmented by tiny magnitude differences.

    Public strength is already visible, so this leaks nothing; it only decides how training
    observations are pooled.
    """
    if strength > 0:
        return "positive"
    if strength < 0:
        return "negative"
    return "zero"


def _group_key(features: dict[str, Any]) -> tuple[Any, ...]:
    return (
        features["operation"],
        int(features["layer"]),
        _strength_bucket(float(features["strength"])),
    )


class ConstantBaseline(Forecaster):
    method_id = "constant"

    def __init__(self) -> None:
        self._delta_by_group: dict[tuple[Any, ...], list[float]] = {}
        self._flip_by_group: dict[tuple[Any, ...], list[bool]] = {}
        self._global_delta: list[float] = []
        self._global_flip: list[bool] = []
        self._fitted = False

    def declared_inputs(self) -> dict[str, bool]:
        return {
            "prompt": False,
            "hidden_state": False,
            "adapter_identity": False,
            "clean_logits": False,
            "intervention_vector": False,
            "public_metadata": True,
            "observed_outcomes": False,
        }

    def state_condition(self) -> StateCondition:
        return StateCondition.NONE

    def fit(self, examples: list[TrainingExample]) -> None:
        self._delta_by_group.clear()
        self._flip_by_group.clear()
        self._global_delta.clear()
        self._global_flip.clear()
        for example in examples:
            key = _group_key(example.public_features)
            self._delta_by_group.setdefault(key, []).append(example.observed_delta)
            self._flip_by_group.setdefault(key, []).append(example.observed_flip)
            self._global_delta.append(example.observed_delta)
            self._global_flip.append(example.observed_flip)
        self._fitted = True

    def predict_candidate(
        self, prompt_text: str, intervention_id: str, public_features: dict[str, Any]
    ) -> CandidatePrediction:
        if not self._fitted:
            raise RuntimeError("constant baseline must be fit before predicting")

        key = _group_key(public_features)
        deltas = self._delta_by_group.get(key) or self._global_delta
        flips = self._flip_by_group.get(key) or self._global_flip

        mean_delta = statistics.fmean(deltas) if deltas else 0.0
        flip_rate = (sum(flips) / len(flips)) if flips else 0.5

        if len(deltas) >= _MIN_SAMPLES_FOR_QUANTILES:
            ordered = sorted(deltas)
            low = ordered[max(0, int(0.05 * len(ordered)) - 1)]
            high = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        else:
            low = mean_delta - _DEFAULT_HALF_WIDTH
            high = mean_delta + _DEFAULT_HALF_WIDTH

        return CandidatePrediction(
            intervention_id=intervention_id,
            delta_margin_mean=mean_delta,
            delta_margin_q05=low,
            delta_margin_q95=high,
            p_answer_flip=flip_rate,
        )
