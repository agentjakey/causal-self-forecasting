"""The BlueDot state-dependence target, `delta_clean_top_margin`.

This is a second target living beside the benchmark's existing one, not a replacement for it.
`ObservationRecord.delta_margin` still measures the margin around the **dataset-correct** answer
and keeps its current meaning and validator. The target here measures the margin around the
**model's own clean preferred** answer, which is a different quantity answering a different
question, and it is stored in its own record type so the two can never be confused on disk.

Definition, for clean and intervened logits over exactly A, B, C, D:

```text
c_star                 = argmax over the clean logits, ties broken in fixed order A, B, C, D
clean_top_margin       = clean_logit[c_star]      - max(clean_logit[c]      for c != c_star)
intervened_top_margin  = intervened_logit[c_star] - max(intervened_logit[c] for c != c_star)
delta_clean_top_margin = intervened_top_margin - clean_top_margin
answer_flip            = the intervened top label differs from c_star
```

`c_star` is fixed from the clean run and never recomputed against the intervened logits. That is
the whole point: the target measures what happened to the answer the model actually preferred,
not to whichever answer happens to lead afterwards.

`clean_top_margin` is non-negative by construction. `intervened_top_margin` goes negative exactly
when some other label strictly overtakes `c_star`. At an exact intervened tie the margin is zero
and the flip is decided by the fixed label order, so `flip` and `margin < 0` agree everywhere
except on that measure-zero boundary. `answer_flip` is defined by the recomputed top label, not
by the sign, so the two are never quietly assumed to be the same thing.

Arithmetic is float64 throughout. Python's `float` is IEEE-754 binary64, so plain float
arithmetic here is float64 arithmetic; the conversion at the boundary is explicit so that a
numpy scalar or a Decimal cannot slip in and change the rounding.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Fixed label order. Used both for tie-breaking and for the "exactly these four" check, so a
# reordering here would change which answer wins a tie and is a preregistration amendment.
ANSWER_LABELS: tuple[str, ...] = ("A", "B", "C", "D")

TARGET_NAME = "delta_clean_top_margin"

# How far a stored target may drift from a recomputation before the record is rejected. Tight,
# because both sides are float64 arithmetic over four numbers: anything larger than round-off
# means the stored value was not produced by this definition.
TARGET_TOLERANCE = 1e-9


class TargetError(ValueError):
    """Raised when logits or a stored target cannot support the study target."""


@dataclass(frozen=True)
class StateAuditTarget:
    """One computed target, with every intermediate a verifier would need."""

    clean_preferred_label: str
    clean_top_margin: float
    intervened_top_margin: float
    delta_clean_top_margin: float
    answer_flip: bool
    intervened_preferred_label: str


def logits_from_pairs(pairs: Sequence[tuple[str, float]]) -> dict[str, float]:
    """Build a logit mapping from label/value pairs, rejecting duplicates.

    A mapping cannot carry a duplicate label, so a caller assembling one from a sequence would
    silently keep the last value. This is the entry point that refuses instead.
    """
    seen: dict[str, float] = {}
    for label, value in pairs:
        if label in seen:
            raise TargetError(f"label {label!r} appears more than once in the answer logits")
        seen[label] = value
    return validate_answer_logits(seen)


def validate_answer_logits(
    logits: Mapping[str, float], name: str = "answer logits"
) -> dict[str, float]:
    """Check that a mapping is exactly the four answer labels with finite values."""
    keys = set(logits)
    expected = set(ANSWER_LABELS)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise TargetError(
            f"{name} must contain exactly {list(ANSWER_LABELS)}; "
            f"missing {missing}, unexpected {extra}"
        )

    validated: dict[str, float] = {}
    for label in ANSWER_LABELS:
        value = logits[label]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TargetError(f"{name}[{label!r}] is {value!r}, which is not a real number")
        as_float = float(value)
        if not math.isfinite(as_float):
            raise TargetError(f"{name}[{label!r}] is {as_float!r}; every logit must be finite")
        validated[label] = as_float
    return validated


def preferred_label(logits: Mapping[str, float]) -> str:
    """The highest-logit label, with exact ties broken in fixed A, B, C, D order."""
    validated = validate_answer_logits(logits)
    best = ANSWER_LABELS[0]
    for label in ANSWER_LABELS[1:]:
        if validated[label] > validated[best]:
            best = label
    return best


def preferred_margin(logits: Mapping[str, float], label: str) -> float:
    """`logit[label] - max(logit[other])`, for a label chosen by the caller.

    Takes the label rather than deriving it, because the intervened margin must be computed
    against the *clean* preference. A function that recomputed the argmax here would silently
    turn the study target into a different quantity.
    """
    validated = validate_answer_logits(logits)
    if label not in validated:
        raise TargetError(f"label {label!r} is not one of {list(ANSWER_LABELS)}")
    others = [validated[other] for other in ANSWER_LABELS if other != label]
    return float(validated[label] - max(others))


def state_audit_target(
    clean_logits: Mapping[str, float],
    intervened_logits: Mapping[str, float],
) -> StateAuditTarget:
    """Compute the complete study target from one clean and one intervened logit vector."""
    clean = validate_answer_logits(clean_logits, "clean logits")
    intervened = validate_answer_logits(intervened_logits, "intervened logits")

    c_star = preferred_label(clean)
    clean_top = preferred_margin(clean, c_star)
    intervened_top = preferred_margin(intervened, c_star)
    delta = float(intervened_top - clean_top)

    if not math.isfinite(delta):
        raise TargetError(f"{TARGET_NAME} is {delta!r}; the target must be finite")
    if clean_top < 0.0:
        raise TargetError(
            f"clean_top_margin {clean_top!r} is negative, which is impossible for the clean "
            "argmax; the preferred label and the logits disagree"
        )

    intervened_label = preferred_label(intervened)
    return StateAuditTarget(
        clean_preferred_label=c_star,
        clean_top_margin=clean_top,
        intervened_top_margin=intervened_top,
        delta_clean_top_margin=delta,
        answer_flip=intervened_label != c_star,
        intervened_preferred_label=intervened_label,
    )


def verify_state_audit_target(
    clean_logits: Mapping[str, float],
    intervened_logits: Mapping[str, float],
    stored_clean_preferred_label: str,
    stored_clean_top_margin: float,
    stored_intervened_top_margin: float,
    stored_delta: float,
    stored_answer_flip: bool,
    tolerance: float = TARGET_TOLERANCE,
) -> list[str]:
    """Recompute a stored target from its saved logits and report every disagreement.

    Returns a list of human-readable discrepancies rather than raising, so a caller can report
    all of them at once. An empty list means the stored values are exactly what this definition
    produces from the logits beside them.
    """
    recomputed = state_audit_target(clean_logits, intervened_logits)
    problems: list[str] = []

    if stored_clean_preferred_label != recomputed.clean_preferred_label:
        problems.append(
            f"clean_preferred_label {stored_clean_preferred_label!r} does not match the "
            f"recomputed {recomputed.clean_preferred_label!r}"
        )
    for name, stored, expected in (
        ("clean_top_margin", stored_clean_top_margin, recomputed.clean_top_margin),
        (
            "intervened_top_margin",
            stored_intervened_top_margin,
            recomputed.intervened_top_margin,
        ),
        (TARGET_NAME, stored_delta, recomputed.delta_clean_top_margin),
    ):
        if not math.isfinite(float(stored)):
            problems.append(f"{name} {stored!r} is not finite")
            continue
        if abs(float(stored) - expected) > tolerance:
            problems.append(
                f"{name} {stored!r} does not match the recomputed {expected!r} "
                f"(tolerance {tolerance:g})"
            )
    if bool(stored_answer_flip) != recomputed.answer_flip:
        problems.append(
            f"answer_flip {bool(stored_answer_flip)} does not match the recomputed "
            f"{recomputed.answer_flip}"
        )
    return problems


__all__ = [
    "ANSWER_LABELS",
    "TARGET_NAME",
    "TARGET_TOLERANCE",
    "StateAuditTarget",
    "TargetError",
    "logits_from_pairs",
    "preferred_label",
    "preferred_margin",
    "state_audit_target",
    "validate_answer_logits",
    "verify_state_audit_target",
]
