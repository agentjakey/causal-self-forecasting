"""The layer-fallback state machine.

Deterministic, and mechanical on purpose. It reads ratio summaries and nothing else: no model,
no prompts, no states, no forecaster. Given the same summaries it always returns the same
decision, so the choice of intervention strength cannot drift with whoever runs it.

```text
layer 13, five ratios
  any pass  -> passed_primary, select the smallest passing ratio; layer 20 is now prohibited
  none pass -> fallback_required
layer 20, five ratios, only reachable from fallback_required
  any pass  -> passed_fallback, select the smallest passing ratio
  none pass -> failed_all_layers
```

Two rules carry the weight. **Smallest passing ratio, in preregistered order** -- not the
largest effect, not the most flips, not whatever a forecaster does best on, because any of those
would choose the stimulus using the outcome. And **a passing primary layer prohibits the
fallback** -- once layer 13 has produced a usable ratio there is no legitimate reason to look at
layer 20, and offering it would turn a preregistered fallback into a second try.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..schemas import CalibrationRatioSummary, CalibrationStatus

SELECTION_ALGORITHM_VERSION = "bluedot_smallest_passing_ratio_v1.0"


class SelectionError(ValueError):
    """Raised when the supplied summaries cannot support a decision."""


class CalibrationSelection:
    """The outcome of the state machine, before it becomes a record."""

    def __init__(
        self,
        status: CalibrationStatus,
        rationale: str,
        summaries: list[CalibrationRatioSummary],
        selected: CalibrationRatioSummary | None = None,
    ) -> None:
        self.status = status
        self.rationale = rationale
        self.summaries = summaries
        self.selected = selected

    @property
    def selected_layer(self) -> int | None:
        return None if self.selected is None else self.selected.layer

    @property
    def selected_norm_ratio(self) -> float | None:
        return None if self.selected is None else self.selected.norm_ratio

    @property
    def selected_global_alpha(self) -> float | None:
        return None if self.selected is None else self.selected.global_alpha


def _check_layer_coverage(
    summaries: Sequence[CalibrationRatioSummary],
    layer: int,
    expected_ratios: Sequence[float],
) -> list[CalibrationRatioSummary]:
    """Require exactly one summary per preregistered ratio, and order them by ratio."""
    for summary in summaries:
        if summary.layer != layer:
            raise SelectionError(
                f"a summary for layer {summary.layer} was supplied among the layer-{layer} "
                "summaries"
            )
    ratios = [summary.norm_ratio for summary in summaries]
    duplicates = sorted({ratio for ratio in ratios if ratios.count(ratio) > 1})
    if duplicates:
        raise SelectionError(f"layer {layer} has more than one summary for ratios {duplicates}")

    wanted = [float(ratio) for ratio in expected_ratios]
    missing = [ratio for ratio in wanted if ratio not in ratios]
    extra = [ratio for ratio in ratios if ratio not in wanted]
    if missing or extra:
        raise SelectionError(
            f"layer {layer} must be summarized at exactly the preregistered ratios {wanted}; "
            f"missing {missing}, unexpected {extra}. A partial grid could make a larger ratio "
            "look like the smallest passing one."
        )
    by_ratio = {summary.norm_ratio: summary for summary in summaries}
    return [by_ratio[ratio] for ratio in wanted]


def _describe(layer: int, ordered: Sequence[CalibrationRatioSummary]) -> str:
    parts: list[str] = []
    for summary in ordered:
        if summary.passed:
            parts.append(f"ratio {summary.norm_ratio:g} passed every condition")
        else:
            failed = [c.name for c in summary.criteria if not c.passed]
            parts.append(f"ratio {summary.norm_ratio:g} failed {failed}")
    return f"layer {layer}: " + "; ".join(parts)


def select_calibration_ratio(
    primary_layer: int,
    fallback_layer: int,
    expected_ratios: Sequence[float],
    primary_summaries: Sequence[CalibrationRatioSummary],
    fallback_summaries: Sequence[CalibrationRatioSummary] | None = None,
) -> CalibrationSelection:
    """Run the state machine over supplied summaries."""
    if primary_layer == fallback_layer:
        raise SelectionError("the fallback layer must differ from the primary layer")

    primary = _check_layer_coverage(primary_summaries, primary_layer, expected_ratios)
    primary_passing = [summary for summary in primary if summary.passed]
    primary_note = _describe(primary_layer, primary)

    if primary_passing:
        if fallback_summaries:
            raise SelectionError(
                f"layer {primary_layer} has {len(primary_passing)} passing ratio(s), so layer "
                f"{fallback_layer} must not be calibrated. The fallback is reachable only when "
                "the primary layer produces no usable ratio; running it anyway would be a "
                "second attempt, not a preregistered fallback."
            )
        selected = primary_passing[0]
        return CalibrationSelection(
            status=CalibrationStatus.PASSED_PRIMARY,
            rationale=(
                f"{primary_note}. Selected the smallest passing ratio "
                f"{selected.norm_ratio:g} at layer {primary_layer} in preregistered order. "
                f"The layer-{fallback_layer} fallback is prohibited because the primary layer "
                "produced a usable ratio."
            ),
            summaries=list(primary),
            selected=selected,
        )

    if not fallback_summaries:
        return CalibrationSelection(
            status=CalibrationStatus.FALLBACK_REQUIRED,
            rationale=(
                f"{primary_note}. No layer-{primary_layer} ratio satisfied every condition, "
                f"which is the only trigger for the layer-{fallback_layer} fallback. Layer "
                f"{fallback_layer} may now be calibrated once."
            ),
            summaries=list(primary),
        )

    fallback = _check_layer_coverage(fallback_summaries, fallback_layer, expected_ratios)
    fallback_passing = [summary for summary in fallback if summary.passed]
    fallback_note = _describe(fallback_layer, fallback)
    combined = list(primary) + list(fallback)

    if fallback_passing:
        selected = fallback_passing[0]
        return CalibrationSelection(
            status=CalibrationStatus.PASSED_FALLBACK,
            rationale=(
                f"{primary_note}. {fallback_note}. Selected the smallest passing ratio "
                f"{selected.norm_ratio:g} at the fallback layer {fallback_layer} in "
                "preregistered order."
            ),
            summaries=combined,
            selected=selected,
        )

    return CalibrationSelection(
        status=CalibrationStatus.FAILED_ALL_LAYERS,
        rationale=(
            f"{primary_note}. {fallback_note}. No ratio satisfied every condition at either "
            "preregistered layer. The study stops under this design: no third layer is "
            "searched, the ratio grid is not widened, and the direction family is not changed."
        ),
        summaries=combined,
    )


__all__ = [
    "SELECTION_ALGORITHM_VERSION",
    "CalibrationSelection",
    "SelectionError",
    "select_calibration_ratio",
]
