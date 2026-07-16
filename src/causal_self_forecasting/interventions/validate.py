"""Intervention harness controls.

These checks exist because the failure they guard against is invisible. An intervention hook
that silently does nothing produces clean numbers for every mechanism, and every downstream
analysis would run happily and report that interventions have small effects.

Checks are split into two kinds:

* Required. A failure means the harness is broken and no result from it means anything.
  These make the command exit nonzero.
* Reported. Measured and printed, but not pass/fail, because their expected value depends on
  the model and the direction. Sign reversal on a randomly initialized fixture model, for
  example, carries no information at all.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from ..config import ResolvedExperiment
from ..hashing import read_jsonl
from ..interventions.directions import DirectionStore
from ..interventions.tensor_ops import (
    InterventionPayload,
    InterventionShapeError,
    apply_direction_ablate,
    apply_residual_add,
)
from ..logging_utils import info
from ..models.capture import capture_hidden_states, run_with_intervention
from ..models.loader import LoadedModel, load_model
from ..models.scoring import resolve_label_token_ids, score_logits
from ..paths import directions_dir, processed_dir
from ..reproducibility import derive_seed, set_global_seed
from ..schemas import InterventionSpec, Mechanism, PromptVariant, TaskItem


@dataclass
class CheckResult:
    """One control's outcome."""

    name: str
    required: bool
    passed: bool | None
    detail: str
    measured: dict[str, Any]


def _spec(
    mechanism: Mechanism, layer: int, position: int, strength: float, direction_id: str | None
) -> InterventionSpec:
    # Built with derive_seed rather than the builtin hash: Python randomizes string hashing
    # per process, so a builtin-hash id would differ between runs of the same check.
    suffix = derive_seed("validate_spec", mechanism.value, strength, layer) % 10000
    return InterventionSpec(
        intervention_id=f"validate.{mechanism.value}.{suffix:04d}",
        mechanism=mechanism,
        mechanism_version="1.0",
        layer=layer,
        position_index=position,
        strength=strength,
        direction_id=direction_id,
        analysis_role="harness_validation",
    )


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.to(torch.float64) - right.to(torch.float64)).abs().max())


def _first_variant(task_name: str) -> tuple[PromptVariant, str]:
    directory = processed_dir() / task_name
    variants_path = directory / "variants.jsonl"
    items_path = directory / "items.jsonl"
    if not variants_path.exists():
        raise RuntimeError(f"no prepared data for task {task_name!r}; run `csf data prepare` first")
    variants = [PromptVariant.model_validate(row) for row in read_jsonl(variants_path)]
    items = {
        item.item_id: item for item in (TaskItem.model_validate(r) for r in read_jsonl(items_path))
    }
    variant = sorted(variants, key=lambda v: v.variant_id)[0]
    return variant, items[variant.item_id].answer_label


def _check_noop_equality(
    model: LoadedModel, prompt: str, layer: int, position: int, tolerance: float
) -> CheckResult:
    """A no-op must reproduce the clean run.

    The strictest available check on the hook machinery: the hook fires, rewrites the
    residual stream with identical values, and the output must be unchanged.
    """
    clean = capture_hidden_states(model, prompt, layers=[layer], position_index=position)
    noop = run_with_intervention(
        model, prompt, _spec(Mechanism.NOOP, layer, position, 0.0, None), InterventionPayload()
    )
    difference = _max_abs_diff(clean.next_token_logits, noop.next_token_logits)
    delta_norm = noop.diagnostics.delta_norm if noop.diagnostics else float("nan")
    return CheckResult(
        name="noop_equality",
        required=True,
        passed=difference <= tolerance and delta_norm == 0.0,
        detail=(
            "no-op reproduces the clean logits and changes the residual stream by exactly zero"
            if difference <= tolerance and delta_norm == 0.0
            else "no-op did not reproduce the clean run; the hook machinery is itself "
            "altering the forward pass"
        ),
        measured={
            "max_abs_logit_diff": difference,
            "residual_delta_norm": delta_norm,
            "tolerance": tolerance,
        },
    )


def _check_zero_strength_equality(
    model: LoadedModel,
    prompt: str,
    layer: int,
    position: int,
    direction_id: str,
    direction: torch.Tensor,
    tolerance: float,
) -> CheckResult:
    """Residual addition at strength zero must equal the clean run."""
    clean = capture_hidden_states(model, prompt, layers=[layer], position_index=position)
    zero = run_with_intervention(
        model,
        prompt,
        _spec(Mechanism.RESIDUAL_ADD, layer, position, 0.0, direction_id),
        InterventionPayload(direction=direction),
    )
    difference = _max_abs_diff(clean.next_token_logits, zero.next_token_logits)
    return CheckResult(
        name="zero_strength_equality",
        required=True,
        passed=difference <= tolerance,
        detail=(
            "adding the direction with strength 0 leaves the output unchanged"
            if difference <= tolerance
            else "strength 0 changed the output, so the strength parameter is not being "
            "applied as written"
        ),
        measured={"max_abs_logit_diff": difference, "tolerance": tolerance},
    )


def _check_rerun_determinism(
    model: LoadedModel, prompt: str, layer: int, position: int, tolerance: float
) -> CheckResult:
    """Two clean runs of the same prompt must agree.

    If they do not, every measured delta is contaminated by run-to-run noise and no effect
    smaller than that noise can be claimed.
    """
    first = capture_hidden_states(model, prompt, layers=[layer], position_index=position)
    second = capture_hidden_states(model, prompt, layers=[layer], position_index=position)
    logit_diff = _max_abs_diff(first.next_token_logits, second.next_token_logits)
    state_diff = _max_abs_diff(first.hidden_states[layer], second.hidden_states[layer])
    return CheckResult(
        name="rerun_determinism",
        required=True,
        passed=logit_diff <= tolerance and state_diff <= tolerance,
        detail=(
            "repeated clean runs agree"
            if logit_diff <= tolerance
            else "repeated clean runs disagree; measured effects cannot be separated from run noise"
        ),
        measured={
            "max_abs_logit_diff": logit_diff,
            "max_abs_state_diff": state_diff,
            "tolerance": tolerance,
        },
    )


def _check_shape_validation(hidden_dim: int) -> CheckResult:
    """A mismatched direction must raise rather than broadcast.

    Torch would happily broadcast a length-1 vector across the hidden state and produce a
    number. That number would be meaningless, so the harness has to refuse it.
    """
    hidden = torch.zeros(hidden_dim)
    failures: list[str] = []

    for name, operand in (
        ("wrong_length", torch.ones(hidden_dim + 1)),
        ("scalar_broadcast", torch.ones(1)),
        ("two_dimensional", torch.ones(2, hidden_dim)),
    ):
        try:
            apply_residual_add(hidden, operand, 1.0)
        except InterventionShapeError:
            continue
        failures.append(name)

    try:
        apply_direction_ablate(hidden, torch.zeros(hidden_dim))
    except InterventionShapeError:
        pass
    else:
        failures.append("zero_direction_ablate")

    return CheckResult(
        name="shape_validation",
        required=True,
        passed=not failures,
        detail=(
            "mismatched and degenerate operands are rejected"
            if not failures
            else f"these invalid operands were silently accepted: {failures}"
        ),
        measured={"accepted_invalid": failures},
    )


def _check_hook_fires(
    model: LoadedModel,
    prompt: str,
    layer: int,
    position: int,
    direction_id: str,
    direction: torch.Tensor,
    strength: float,
) -> CheckResult:
    """A nonzero intervention must actually change the residual stream."""
    result = run_with_intervention(
        model,
        prompt,
        _spec(Mechanism.RESIDUAL_ADD, layer, position, strength, direction_id),
        InterventionPayload(direction=direction),
    )
    delta_norm = result.diagnostics.delta_norm if result.diagnostics else 0.0
    expected = float(strength * torch.linalg.vector_norm(direction))
    agrees = abs(delta_norm - abs(expected)) <= 1e-3 * max(1.0, abs(expected))
    return CheckResult(
        name="hook_applies_intervention",
        required=True,
        passed=delta_norm > 0.0 and agrees,
        detail=(
            "the residual stream moved by exactly the expected amount"
            if delta_norm > 0.0 and agrees
            else "the applied change does not match alpha times the direction norm"
        ),
        measured={"residual_delta_norm": delta_norm, "expected_delta_norm": abs(expected)},
    )


def _measure_sign_reversal(
    model: LoadedModel,
    prompt: str,
    layer: int,
    position: int,
    direction_id: str,
    direction: torch.Tensor,
    strength: float,
    label_token_ids: dict[str, int],
    correct_label: str,
) -> CheckResult:
    """Measure the margin change under positive and negative steering.

    Reported, not required. Whether the two deltas have opposite signs is a fact about the
    direction and the model, and on a random fixture model it means nothing. It becomes
    evidence only after a direction has been estimated on a real model, which is why this is
    a measurement here and a validation criterion in Phase 6.
    """
    clean = capture_hidden_states(model, prompt, layers=[layer], position_index=position)
    clean_margin = score_logits(clean.next_token_logits, label_token_ids, correct_label).margin

    deltas: dict[str, float] = {}
    for sign, name in ((1.0, "positive"), (-1.0, "negative")):
        result = run_with_intervention(
            model,
            prompt,
            _spec(Mechanism.RESIDUAL_ADD, layer, position, sign * strength, direction_id),
            InterventionPayload(direction=direction),
        )
        margin = score_logits(result.next_token_logits, label_token_ids, correct_label).margin
        deltas[name] = margin - clean_margin

    opposite = deltas["positive"] * deltas["negative"] < 0
    return CheckResult(
        name="sign_reversal_coherence",
        required=False,
        passed=None,
        detail=(
            "positive and negative steering moved the margin in opposite directions"
            if opposite
            else "positive and negative steering did not move the margin in opposite directions"
        ),
        measured={
            "clean_margin": clean_margin,
            "delta_positive": deltas["positive"],
            "delta_negative": deltas["negative"],
            "opposite_signs": opposite,
            "note": "informational; on a random fixture model this carries no meaning",
        },
    )


def validate_interventions(resolved: ResolvedExperiment) -> dict[str, Any]:
    """Run the intervention harness controls and return a report."""
    experiment = resolved.experiment
    set_global_seed(experiment.seed)

    variant, correct_label = _first_variant(resolved.task.name)
    model = load_model(resolved.model)
    label_token_ids = resolve_label_token_ids(
        model.tokenizer, resolved.task.answer_labels, prefix=resolved.task.label_prefix
    )

    layer = experiment.capture_layers[0]
    position = experiment.capture_position
    tolerance = experiment.rerun_tolerance

    store = DirectionStore(directions_dir())
    if not store.has(experiment.direction_id):
        raise RuntimeError(
            f"direction {experiment.direction_id!r} is not in {directions_dir()}; "
            "create it with `csf directions synthetic` (smoke) or "
            "`csf directions estimate` (real runs)"
        )
    direction = store.load(experiment.direction_id, dtype=model.dtype).to(model.device)
    if direction.shape[0] != model.hidden_dim:
        raise RuntimeError(
            f"direction {experiment.direction_id!r} has dimension {direction.shape[0]} but the "
            f"model hidden dimension is {model.hidden_dim}"
        )

    strength = 1.0
    checks = [
        _check_noop_equality(model, variant.prompt_text, layer, position, tolerance),
        _check_zero_strength_equality(
            model,
            variant.prompt_text,
            layer,
            position,
            experiment.direction_id,
            direction,
            tolerance,
        ),
        _check_rerun_determinism(model, variant.prompt_text, layer, position, tolerance),
        _check_shape_validation(model.hidden_dim),
        _check_hook_fires(
            model,
            variant.prompt_text,
            layer,
            position,
            experiment.direction_id,
            direction,
            strength,
        ),
        _measure_sign_reversal(
            model,
            variant.prompt_text,
            layer,
            position,
            experiment.direction_id,
            direction,
            strength,
            label_token_ids,
            correct_label,
        ),
    ]

    required_failed = [check.name for check in checks if check.required and not check.passed]
    for check in checks:
        info(
            "intervention control",
            check=check.name,
            required=check.required,
            passed=check.passed,
        )

    return {
        "experiment": experiment.name,
        "model": model.spec.model_id,
        "model_variant": model.spec.variant.value,
        "layer": layer,
        "position": position,
        "direction_id": experiment.direction_id,
        "checks": [asdict(check) for check in checks],
        "required_failed": required_failed,
        "passed": not required_failed,
    }
