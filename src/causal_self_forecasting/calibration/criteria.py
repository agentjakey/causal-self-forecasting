"""Calibration pass conditions for one (layer, ratio) grid point.

Six conditions, all frozen before any number exists, and a ratio passes only when every one of
them passes. They are deliberately about whether the intervention family produces a measurable,
non-degenerate effect distribution, and deliberately not about whether any forecaster does well
on it. Choosing a ratio by predictive performance would select the stimulus on the outcome.

| # | Condition |
| 1 | completeness: every expected observation is present, or its failure is recorded |
| 2 | every stored logit and target is finite |
| 3 | the worst absolute no-op target is within the harness no-op tolerance |
| 4 | at least 15 percent of non-no-op effects reach 0.10 in absolute value |
| 5 | the median absolute non-no-op effect is at least 0.05 |
| 6 | the 95th percentile absolute non-no-op effect is at most 4.0 |

Conditions 4, 5, and 6 are inclusive at the boundary: a ratio landing exactly on a threshold
passes. That is stated here because "at least" and "no greater than" are the preregistered
words, and a strict comparison would quietly move the grid.

Percentiles use `numpy.quantile(..., method="linear")`, NumPy's default, recorded in the plan
so a reader knows which of the nine common conventions produced the number.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..schemas import CalibrationCriterionResult, CalibrationRatioSummary, CalibrationThresholds

PERCENTILE_METHOD = "numpy.quantile(method='linear')"
SUMMARY_ALGORITHM_VERSION = "bluedot_calibration_criteria_v1.0"


class CriteriaError(ValueError):
    """Raised when a ratio summary cannot be computed from the supplied observations."""


@dataclass(frozen=True)
class EffectSample:
    """One observation reduced to what the pass conditions look at."""

    prompt_id: str
    candidate_id: str
    is_noop: bool
    target: float
    answer_flip: bool
    logits_finite: bool = True


def _quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="linear"))


def _criterion(
    name: str, passed: bool, observed: float, threshold: float, comparison: str
) -> CalibrationCriterionResult:
    return CalibrationCriterionResult(
        name=name,
        passed=passed,
        observed=float(observed),
        threshold=float(threshold),
        comparison=comparison,
    )


def summarize_ratio(
    samples: Sequence[EffectSample],
    layer: int,
    norm_ratio: float,
    global_alpha: float,
    expected_non_noop: int,
    expected_noop: int,
    thresholds: CalibrationThresholds,
    noop_tolerance: float,
    failure_count: int = 0,
    supporting_artifact_hashes: dict[str, str] | None = None,
) -> CalibrationRatioSummary:
    """Reduce one grid point's observations to a summary and a pass or fail."""
    if not samples:
        raise CriteriaError(
            f"no observations supplied for layer {layer} ratio {norm_ratio}; a grid point with "
            "no data cannot be judged, and treating it as a failure would hide the difference "
            "between a measured null and a missing run"
        )

    noop = [sample for sample in samples if sample.is_noop]
    non_noop = [sample for sample in samples if not sample.is_noop]

    finite_flags = [sample.logits_finite and math.isfinite(sample.target) for sample in samples]
    finite_rate = sum(finite_flags) / len(finite_flags)

    max_abs_noop = max((abs(sample.target) for sample in noop), default=0.0)

    # Non-finite targets would poison every order statistic, so the finiteness condition is
    # reported on its own and the statistics below are computed over what is finite. When
    # nothing finite remains, the order statistics are reported as 0.0 over an empty set rather
    # than as an invented number; the ratio still fails, because the finiteness condition and
    # the median floor both fail.
    finite_magnitudes = [abs(sample.target) for sample in non_noop if math.isfinite(sample.target)]
    if finite_magnitudes:
        fraction_large = sum(
            1 for value in finite_magnitudes if value >= thresholds.large_effect_threshold
        ) / len(finite_magnitudes)
        median_abs = float(np.median(np.asarray(finite_magnitudes, dtype=np.float64)))
        p95_abs = _quantile(finite_magnitudes, 0.95)
        p95_passed = p95_abs <= thresholds.max_p95_abs_effect
    else:
        fraction_large, median_abs, p95_abs = 0.0, 0.0, 0.0
        # An empty set has no 95th percentile. Reporting 0.0 keeps the artifact honest about
        # the arithmetic, and this condition is marked failed rather than vacuously passed.
        p95_passed = False

    accounted = len(non_noop) + failure_count
    criteria = [
        _criterion(
            "completeness",
            accounted == expected_non_noop and len(noop) == expected_noop,
            accounted,
            expected_non_noop,
            "observed non-noop plus recorded failures == expected, and the no-op count matches",
        ),
        _criterion("finite_outputs", finite_rate == 1.0, finite_rate, 1.0, "== 1.0"),
        _criterion(
            "noop_within_tolerance",
            max_abs_noop <= noop_tolerance,
            max_abs_noop,
            noop_tolerance,
            "<=",
        ),
        _criterion(
            "large_effect_fraction",
            fraction_large >= thresholds.min_large_effect_fraction,
            fraction_large,
            thresholds.min_large_effect_fraction,
            ">=",
        ),
        _criterion(
            "median_abs_effect",
            median_abs >= thresholds.min_median_abs_effect,
            median_abs,
            thresholds.min_median_abs_effect,
            ">=",
        ),
        _criterion(
            "p95_abs_effect",
            p95_passed,
            p95_abs,
            thresholds.max_p95_abs_effect,
            "<=",
        ),
    ]

    return CalibrationRatioSummary(
        layer=layer,
        norm_ratio=float(norm_ratio),
        global_alpha=float(global_alpha),
        expected_non_noop_observations=expected_non_noop,
        observed_non_noop_observations=len(non_noop),
        noop_count=len(noop),
        failure_count=failure_count,
        finite_output_rate=finite_rate,
        max_abs_noop_target=max_abs_noop,
        fraction_above_effect_threshold=fraction_large,
        median_abs_effect=median_abs,
        p95_abs_effect=p95_abs,
        flip_count=sum(1 for sample in non_noop if sample.answer_flip),
        criteria=criteria,
        passed=all(criterion.passed for criterion in criteria),
        supporting_artifact_hashes=dict(supporting_artifact_hashes or {}),
    )


__all__ = [
    "PERCENTILE_METHOD",
    "SUMMARY_ALGORITHM_VERSION",
    "CriteriaError",
    "EffectSample",
    "summarize_ratio",
]
