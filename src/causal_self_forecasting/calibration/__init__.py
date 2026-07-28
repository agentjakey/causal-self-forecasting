"""Calibration for the BlueDot state-dependence arm.

Calibration chooses one intervention strength. It measures nothing about what the model can do,
and no artifact it produces is a scientific result; the records are typed so they cannot claim
otherwise.

Nothing in this package loads a model. The strength rule, the pass conditions, and the layer
state machine are pure functions over supplied summaries, and the planner reads only the frozen
manifests and the pinned model's config.
"""

from .criteria import PERCENTILE_METHOD, CriteriaError, EffectSample, summarize_ratio
from .plan import (
    CalibrationPlanError,
    build_calibration_plan,
    build_decision_record,
    build_forward_counts,
    load_calibration_plan,
    plan_command,
    verify_calibration_plan,
    write_calibration_plan,
)
from .selection import (
    SELECTION_ALGORITHM_VERSION,
    CalibrationSelection,
    SelectionError,
    select_calibration_ratio,
)
from .strength import (
    MEDIAN_METHOD,
    NORM_RATIOS,
    StrengthError,
    alpha_for_ratio,
    alpha_table,
    check_global_alpha,
    reference_norm,
    validate_calibration_norms,
)

__all__ = [
    "MEDIAN_METHOD",
    "NORM_RATIOS",
    "PERCENTILE_METHOD",
    "SELECTION_ALGORITHM_VERSION",
    "CalibrationPlanError",
    "CalibrationSelection",
    "CriteriaError",
    "EffectSample",
    "SelectionError",
    "StrengthError",
    "alpha_for_ratio",
    "alpha_table",
    "build_calibration_plan",
    "build_decision_record",
    "build_forward_counts",
    "check_global_alpha",
    "load_calibration_plan",
    "plan_command",
    "reference_norm",
    "select_calibration_ratio",
    "summarize_ratio",
    "validate_calibration_norms",
    "verify_calibration_plan",
    "write_calibration_plan",
]
