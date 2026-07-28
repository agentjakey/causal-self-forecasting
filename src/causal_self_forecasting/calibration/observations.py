"""Reading supplied observation artifacts and turning them into ratio summaries.

The pure conditions live in `criteria.py` and the pure state machine in `selection.py`. This
module is the glue: it reads records off disk, groups them by grid point, and checks the
grouping assumptions that the pure code is entitled to assume.

One grouping decision is worth stating because it is not obvious. A calibration prompt carries
**one** no-op, not one per ratio: adding zero does the same thing whatever ratio is under test.
The no-op observations are therefore attributed to every ratio summary at that layer, as shared
evidence that the harness reproduced the clean output. They are counted once per ratio in
`noop_count`, and they never enter the effect statistics, which are computed over non-no-op
observations only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..hashing import read_jsonl
from ..schemas import (
    CalibrationPlanRecord,
    CalibrationRatioSummary,
    StateAuditObservationRecord,
)
from .criteria import EffectSample, summarize_ratio
from .strength import check_global_alpha


class ObservationLoadError(RuntimeError):
    """Raised when supplied observation artifacts cannot be used."""


def read_state_audit_observations(path: str | Path) -> list[StateAuditObservationRecord]:
    """Read a JSONL of state-audit observations, validating every record.

    Validation is not optional here: the record recomputes its own target from its own logits,
    so a file that loads is a file whose targets have already been independently checked.
    """
    target = Path(path)
    if not target.exists():
        raise ObservationLoadError(f"no observation file at {target}")
    records: list[StateAuditObservationRecord] = []
    for index, row in enumerate(read_jsonl(target), start=1):
        try:
            records.append(StateAuditObservationRecord.model_validate(row))
        except Exception as error:
            raise ObservationLoadError(f"{target}:{index}: {error}") from error
    if not records:
        raise ObservationLoadError(f"{target} contains no observations")
    return records


def read_failures(path: str | Path | None) -> list[dict[str, Any]]:
    """Read a JSONL of recorded intervention failures, if one was supplied."""
    if path is None:
        return []
    target = Path(path)
    if not target.exists():
        raise ObservationLoadError(f"no failures file at {target}")
    return list(read_jsonl(target))


def _failure_count(failures: Iterable[dict[str, Any]], layer: int, ratio: float) -> int:
    count = 0
    for failure in failures:
        if int(failure.get("layer", -1)) != layer:
            continue
        recorded = failure.get("norm_ratio")
        if recorded is None or float(recorded) != float(ratio):
            continue
        count += 1
    return count


def _sample(record: StateAuditObservationRecord) -> EffectSample:
    return EffectSample(
        prompt_id=record.group_id,
        candidate_id=record.candidate_id,
        is_noop=record.is_noop,
        target=record.delta_clean_top_margin,
        answer_flip=record.answer_flip,
        # A record that validated has finite logits by construction, so anything non-finite
        # would already have been rejected at load time.
        logits_finite=True,
    )


def summarize_layer(
    observations: Sequence[StateAuditObservationRecord],
    plan: CalibrationPlanRecord,
    layer: int,
    failures: Sequence[dict[str, Any]] = (),
) -> list[CalibrationRatioSummary]:
    """Compute one summary per preregistered ratio, for one layer."""
    if layer not in (plan.primary_layer, plan.fallback_layer):
        raise ObservationLoadError(
            f"layer {layer} is neither the plan's primary layer {plan.primary_layer} nor its "
            f"fallback layer {plan.fallback_layer}; no other layer may be calibrated"
        )

    at_layer = [record for record in observations if record.layer == layer]
    if not at_layer:
        raise ObservationLoadError(f"the supplied observations contain nothing at layer {layer}")

    wrong_target = {record.target_name for record in at_layer} - {plan.target_name}
    if wrong_target:
        raise ObservationLoadError(
            f"observations at layer {layer} carry targets {sorted(wrong_target)}, but the plan "
            f"calibrates {plan.target_name!r}"
        )

    noops = [record for record in at_layer if record.is_noop]
    expected_non_noop = plan.calibration_prompt_count * plan.forward_counts.signed_directions
    expected_noop = plan.calibration_prompt_count

    summaries: list[CalibrationRatioSummary] = []
    for ratio in plan.norm_ratios:
        non_noop = [
            record
            for record in at_layer
            if not record.is_noop and record.norm_ratio == float(ratio)
        ]
        if not non_noop:
            raise ObservationLoadError(
                f"no non-no-op observations at layer {layer} ratio {ratio}; a grid point with no "
                "data cannot be judged, and scoring it as a failure would hide the difference "
                "between a measured null and a run that never happened"
            )
        alpha = check_global_alpha(
            # Labelled per candidate, not per prompt: a prompt contributes 16 observations, and
            # collapsing them by prompt would let one deviant alpha be hidden by its neighbours.
            [(record.candidate_id, record.global_alpha) for record in non_noop],
            layer,
            float(ratio),
        )
        samples = [_sample(record) for record in (*non_noop, *noops)]
        summaries.append(
            summarize_ratio(
                samples=samples,
                layer=layer,
                norm_ratio=float(ratio),
                global_alpha=alpha,
                expected_non_noop=expected_non_noop,
                expected_noop=expected_noop,
                thresholds=plan.thresholds,
                noop_tolerance=plan.noop_tolerance,
                failure_count=_failure_count(failures, layer, float(ratio)),
                supporting_artifact_hashes={
                    "plan_hash": plan.plan_hash,
                    "prompt_manifest_hash": plan.prompt_manifest_hash,
                    "direction_family_hash": plan.direction_family_hash,
                },
            )
        )
    return summaries


def read_ratio_summaries(path: str | Path) -> list[CalibrationRatioSummary]:
    """Read summaries written by `csf calibration summarize`."""
    target = Path(path)
    if not target.exists():
        raise ObservationLoadError(f"no summaries file at {target}")
    import json

    payload = json.loads(target.read_text(encoding="utf-8"))
    rows = payload.get("summaries") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ObservationLoadError(f"{target} does not contain a non-empty list of summaries")
    try:
        return [CalibrationRatioSummary.model_validate(row) for row in rows]
    except Exception as error:
        raise ObservationLoadError(f"{target}: {error}") from error


__all__ = [
    "ObservationLoadError",
    "read_failures",
    "read_ratio_summaries",
    "read_state_audit_observations",
    "summarize_layer",
]
