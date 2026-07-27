"""Pure scoring metrics.

No file access, no model, no run state. Everything here takes numbers and returns numbers,
so it is testable in isolation and the same functions serve every method and every split.

Two conventions matter and are enforced by the callers, not hidden here:

* No-op candidates are excluded from headline metrics. A no-op is published as a zero-strength
  addition, and every method predicts it correctly, so pooling no-ops in would inflate every
  score and compress the differences between methods. The scorer computes headline metrics
  over non-no-op pairs and reports the no-op-inclusive numbers separately.
* Uncertainty is bootstrapped by group, not by pair. Pairs from the same task item share a
  question and are not independent; resampling pairs would produce intervals that are too
  narrow.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

_LOG_LOSS_EPS = 1e-12


@dataclass(frozen=True)
class PairScore:
    """The scored comparison of one forecast candidate against its observation."""

    trial_id: str
    method_id: str
    intervention_id: str
    group_id: str
    split: str
    mechanism: str
    is_noop: bool
    predicted_delta: float
    observed_delta: float
    predicted_flip_probability: float
    observed_flip: bool
    interval_low: float
    interval_high: float

    @property
    def absolute_error(self) -> float:
        return abs(self.predicted_delta - self.observed_delta)

    @property
    def squared_error(self) -> float:
        return (self.predicted_delta - self.observed_delta) ** 2

    @property
    def sign_correct(self) -> bool:
        return _same_sign(self.predicted_delta, self.observed_delta)

    @property
    def interval_covered(self) -> bool:
        return self.interval_low <= self.observed_delta <= self.interval_high

    @property
    def interval_width(self) -> float:
        return self.interval_high - self.interval_low

    @property
    def flip_brier(self) -> float:
        return (self.predicted_flip_probability - float(self.observed_flip)) ** 2

    @property
    def flip_log_loss(self) -> float:
        probability = min(max(self.predicted_flip_probability, _LOG_LOSS_EPS), 1 - _LOG_LOSS_EPS)
        if self.observed_flip:
            return -math.log(probability)
        return -math.log(1 - probability)


def _same_sign(left: float, right: float, tolerance: float = 1e-9) -> bool:
    """Sign agreement, treating near-zero as its own class.

    A prediction of no effect against an observed no effect counts as correct; a prediction
    of no effect against a real effect does not.
    """
    left_zero = abs(left) <= tolerance
    right_zero = abs(right) <= tolerance
    if left_zero or right_zero:
        return left_zero and right_zero
    return (left > 0) == (right > 0)


def mae(scores: Sequence[PairScore]) -> float:
    if not scores:
        raise ValueError("cannot compute MAE over zero pairs")
    return sum(score.absolute_error for score in scores) / len(scores)


def rmse(scores: Sequence[PairScore]) -> float:
    if not scores:
        raise ValueError("cannot compute RMSE over zero pairs")
    return math.sqrt(sum(score.squared_error for score in scores) / len(scores))


def sign_accuracy(scores: Sequence[PairScore]) -> float:
    if not scores:
        raise ValueError("cannot compute sign accuracy over zero pairs")
    return sum(int(score.sign_correct) for score in scores) / len(scores)


def brier_score(scores: Sequence[PairScore]) -> float:
    if not scores:
        raise ValueError("cannot compute a Brier score over zero pairs")
    return sum(score.flip_brier for score in scores) / len(scores)


def log_loss(scores: Sequence[PairScore]) -> float:
    if not scores:
        raise ValueError("cannot compute log loss over zero pairs")
    return sum(score.flip_log_loss for score in scores) / len(scores)


def interval_coverage(scores: Sequence[PairScore]) -> float:
    if not scores:
        raise ValueError("cannot compute interval coverage over zero pairs")
    return sum(int(score.interval_covered) for score in scores) / len(scores)


def top_effect_accuracy(trials: Sequence[Sequence[PairScore]]) -> float:
    """Fraction of trials whose predicted largest-effect candidate was the observed largest.

    Effect size is absolute delta margin. A trial only contributes if every candidate in it
    was both forecast and observed, so a run that observed only the selected candidate per
    trial contributes nothing here and the sample count reflects that.
    """
    usable = [group for group in trials if len(group) >= 2]
    if not usable:
        raise ValueError("no trials have enough observed candidates for a ranking metric")
    correct = 0
    for group in usable:
        predicted_top = max(group, key=lambda score: abs(score.predicted_delta))
        observed_top = max(group, key=lambda score: abs(score.observed_delta))
        correct += int(predicted_top.intervention_id == observed_top.intervention_id)
    return correct / len(usable)


def grouped_bootstrap_ci(
    scores: Sequence[PairScore],
    statistic: Callable[[Sequence[PairScore]], float],
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap a confidence interval by resampling groups, not pairs.

    Groups are resampled with replacement, so the interval reflects the fact that pairs from
    one task item are correlated. With a single group the interval collapses to the point
    estimate, which is honest: one group carries no information about between-group variation.
    """
    if not scores:
        raise ValueError("cannot bootstrap over zero pairs")

    by_group: dict[str, list[PairScore]] = {}
    for score in scores:
        by_group.setdefault(score.group_id, []).append(score)
    group_ids = sorted(by_group)

    if len(group_ids) < 2:
        point = statistic(scores)
        return point, point

    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        drawn: list[PairScore] = []
        for _ in group_ids:
            chosen = group_ids[rng.randrange(len(group_ids))]
            drawn.extend(by_group[chosen])
        try:
            estimates.append(statistic(drawn))
        except ValueError:
            continue
    if not estimates:
        point = statistic(scores)
        return point, point

    estimates.sort()
    tail = (1 - confidence) / 2
    low_index = max(0, math.floor(tail * len(estimates)))
    high_index = min(len(estimates) - 1, math.ceil((1 - tail) * len(estimates)) - 1)
    return estimates[low_index], estimates[high_index]
