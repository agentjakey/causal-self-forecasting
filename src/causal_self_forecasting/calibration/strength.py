"""Global intervention strength for one layer.

The preregistered rule: one absolute alpha per layer and ratio, applied unchanged to every
prompt.

```text
reference_norm = median(clean state norm over the 32 calibration prompts, at that layer)
alpha          = ratio * reference_norm
```

**There is deliberately no prompt-specific `ratio * ||h_prompt||` anywhere in this module, and
adding one would be a leak, not an improvement.** `trials/candidates.py:public_view` publishes
`strength` to every method, including the visible-information baseline the arm exists to beat.
A per-prompt alpha would put that prompt's state norm into the published strength, so the
"visible information only" method would silently receive a state-derived feature and the
headline comparison would be contaminated at the source. A global alpha depends on the 32
calibration prompts only, which are disjoint from training and final test, and is identical
across prompts, so it carries no prompt-specific state information at all.

`check_global_alpha` exists to catch a per-prompt alpha that arrives from somewhere else.

Arithmetic is float64. The median is NumPy's ordinary one: for an even sample it averages the
two central sorted values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

# NumPy's default. Recorded in every artifact so a reader knows which of the several "median"
# conventions produced the reference norm.
MEDIAN_METHOD = "numpy.median(linear interpolation; even samples average the two central values)"

# The preregistered grid, ascending. Ascending order is load-bearing: the selection rule is
# "smallest passing ratio", so reordering this changes which ratio wins.
NORM_RATIOS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.40)


class StrengthError(ValueError):
    """Raised when calibration state norms or a strength table are not usable."""


def validate_calibration_norms(
    state_norms: Mapping[str, float],
    expected_prompt_ids: Sequence[str],
) -> dict[str, float]:
    """Check the supplied norms against the prompts they are supposed to describe.

    Alignment is by prompt identity, never by position. A caller that hands over 32 numbers in
    a different order than the manifest lists them must still get the same reference norm, and
    a caller that hands over 32 numbers for the wrong 32 prompts must be refused rather than
    quietly averaged.
    """
    expected = list(expected_prompt_ids)
    if not expected:
        raise StrengthError("no calibration prompts were supplied to validate against")
    duplicates = sorted({p for p in expected if expected.count(p) > 1})
    if duplicates:
        raise StrengthError(f"the expected calibration prompts repeat: {duplicates[:5]}")

    supplied = set(state_norms)
    wanted = set(expected)
    if supplied != wanted:
        missing = sorted(wanted - supplied)
        unexpected = sorted(supplied - wanted)
        raise StrengthError(
            f"the supplied state norms do not match the calibration prompts: "
            f"{len(missing)} missing (for example {missing[:3]}), "
            f"{len(unexpected)} unexpected (for example {unexpected[:3]})"
        )

    validated: dict[str, float] = {}
    for prompt_id in sorted(expected):
        value = state_norms[prompt_id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StrengthError(f"the state norm for {prompt_id} is {value!r}, not a real number")
        as_float = float(value)
        if not math.isfinite(as_float):
            raise StrengthError(
                f"the state norm for {prompt_id} is {as_float!r}; it must be finite"
            )
        if as_float <= 0.0:
            raise StrengthError(
                f"the state norm for {prompt_id} is {as_float!r}; a captured state cannot have a "
                "zero or negative norm, so this is a capture failure rather than a small effect"
            )
        validated[prompt_id] = as_float
    return validated


def reference_norm(
    state_norms: Mapping[str, float],
    expected_prompt_ids: Sequence[str],
) -> float:
    """The median clean-state norm over the calibration prompts, at one layer."""
    validated = validate_calibration_norms(state_norms, expected_prompt_ids)
    # Sorted by prompt id so the input array is a deterministic function of the mapping, not of
    # its iteration order. The median does not depend on order, but the artifact should.
    values = np.asarray([validated[key] for key in sorted(validated)], dtype=np.float64)
    median = float(np.median(values))
    if not math.isfinite(median) or median <= 0.0:
        raise StrengthError(f"the reference norm came out as {median!r}, which cannot be used")
    return median


def alpha_for_ratio(reference: float, ratio: float) -> float:
    """`alpha = ratio * reference_norm`, one number for the whole layer."""
    if not math.isfinite(reference) or reference <= 0.0:
        raise StrengthError(f"the reference norm {reference!r} must be finite and positive")
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise StrengthError(f"the norm ratio {ratio!r} must be finite and positive")
    alpha = float(np.float64(ratio) * np.float64(reference))
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise StrengthError(f"alpha came out as {alpha!r} for ratio {ratio!r}")
    return alpha


def alpha_table(
    reference: float,
    ratios: Sequence[float] = NORM_RATIOS,
) -> list[tuple[float, float]]:
    """The ordered ratio-to-alpha table for one layer.

    Returned as a list of pairs rather than a mapping so that the preregistered order survives
    into the artifact, where it decides which ratio is tried first.
    """
    ordered = list(ratios)
    if not ordered:
        raise StrengthError("the ratio grid is empty")
    if ordered != sorted(ordered):
        raise StrengthError(f"the ratio grid must be ascending, got {ordered}")
    if len(set(ordered)) != len(ordered):
        raise StrengthError(f"the ratio grid repeats a ratio: {ordered}")
    return [(float(ratio), alpha_for_ratio(reference, ratio)) for ratio in ordered]


def check_global_alpha(
    observations: Sequence[tuple[str, float]],
    layer: int,
    ratio: float,
    tolerance: float = 0.0,
) -> float:
    """Confirm that every observation at one (layer, ratio) used the same alpha.

    `observations` is a sequence of `(label, alpha)`, one entry per observation. Every entry is
    compared, and entries are deliberately not collapsed by label: a prompt contributes many
    candidates, and folding them into a mapping would let a single deviant alpha be overwritten
    by its neighbours and vanish.

    The default tolerance is exact. A global alpha is one number copied everywhere, so any
    spread at all means a prompt-specific strength got in, which is the leak this arm's design
    exists to avoid.
    """
    entries = [(str(label), float(alpha)) for label, alpha in observations]
    if not entries:
        raise StrengthError(f"no observations supplied for layer {layer} ratio {ratio}")

    first_label, reference_alpha = entries[0]
    offenders = [
        (label, alpha) for label, alpha in entries if abs(alpha - reference_alpha) > tolerance
    ]
    if offenders:
        raise StrengthError(
            f"layer {layer} ratio {ratio} uses more than one alpha: {reference_alpha} for "
            f"{first_label!r} but {sorted(set(offenders))[:3]} elsewhere "
            f"({len(offenders)} of {len(entries)} observations disagree). The preregistered rule "
            "is one global alpha per layer and ratio; a prompt-specific strength would leak the "
            "prompt's state norm into the published strength."
        )
    return reference_alpha


__all__ = [
    "MEDIAN_METHOD",
    "NORM_RATIOS",
    "StrengthError",
    "alpha_for_ratio",
    "alpha_table",
    "check_global_alpha",
    "reference_norm",
    "validate_calibration_norms",
]
