"""The intervention mathematics, as pure tensor functions.

Deliberately separate from any model code. These functions take a hidden vector and return
a new one, which means the whole mechanism layer is testable with hand-built tensors and no
weights on disk. Every function validates shape and dtype before touching the data, because
a silently broadcast intervention would produce plausible numbers that mean nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..schemas import InterventionSpec, Mechanism

# Below this, a direction is numerically indistinguishable from zero and normalizing it
# would amplify float noise into a unit vector pointing in an arbitrary direction.
MIN_DIRECTION_NORM = 1e-8


class InterventionShapeError(ValueError):
    """Raised when an intervention's operands do not line up."""


@dataclass(frozen=True)
class InterventionPayload:
    """The private half of an intervention.

    Held separately from `InterventionSpec` so that the public record can be handed to a
    forecaster, logged, or published without exposing what the intervention actually does.
    """

    direction: torch.Tensor | None = None
    source_activation: torch.Tensor | None = None


@dataclass(frozen=True)
class InterventionDiagnostics:
    """Norm bookkeeping recorded for every applied intervention.

    `pre_norm` and `post_norm` are the cheapest available evidence that an intervention did
    what it claimed. A no-op with a nonzero `delta_norm` is a bug, and this is how it gets
    caught.
    """

    pre_norm: float
    post_norm: float
    delta_norm: float


def _validate_hidden(hidden: torch.Tensor) -> None:
    if hidden.ndim != 1:
        raise InterventionShapeError(
            f"hidden state must be a 1-D vector, got shape {tuple(hidden.shape)}; "
            "select the layer and token position before intervening"
        )
    if hidden.numel() == 0:
        raise InterventionShapeError("hidden state is empty")
    if not torch.isfinite(hidden).all():
        raise InterventionShapeError("hidden state contains non-finite values")


def _validate_operand(hidden: torch.Tensor, operand: torch.Tensor, name: str) -> None:
    if operand.ndim != 1:
        raise InterventionShapeError(
            f"{name} must be a 1-D vector, got shape {tuple(operand.shape)}"
        )
    if operand.shape != hidden.shape:
        raise InterventionShapeError(
            f"{name} shape {tuple(operand.shape)} does not match hidden state shape "
            f"{tuple(hidden.shape)}"
        )
    if not torch.isfinite(operand).all():
        raise InterventionShapeError(f"{name} contains non-finite values")


def _as_hidden_dtype(hidden: torch.Tensor, operand: torch.Tensor) -> torch.Tensor:
    return operand.to(device=hidden.device, dtype=hidden.dtype)


def apply_noop(hidden: torch.Tensor) -> torch.Tensor:
    """Return the hidden state unchanged.

    Returns a clone rather than the same object so that a caller mutating the result cannot
    corrupt the clean run it is supposed to be compared against.
    """
    _validate_hidden(hidden)
    return hidden.clone()


def apply_residual_add(
    hidden: torch.Tensor, direction: torch.Tensor, strength: float
) -> torch.Tensor:
    """h + alpha * v."""
    _validate_hidden(hidden)
    _validate_operand(hidden, direction, "direction")
    return hidden + float(strength) * _as_hidden_dtype(hidden, direction)


def apply_direction_ablate(hidden: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove the component of h aligned with v.

    The direction is normalized here rather than assumed to be unit length, so that a
    direction artifact estimated as a mean difference can be used without a separate
    normalization step that someone could forget.
    """
    _validate_hidden(hidden)
    _validate_operand(hidden, direction, "direction")
    operand = _as_hidden_dtype(hidden, direction)
    norm = torch.linalg.vector_norm(operand)
    if norm.item() < MIN_DIRECTION_NORM:
        raise InterventionShapeError(
            f"direction norm {norm.item():.3e} is below {MIN_DIRECTION_NORM:.0e}; "
            "ablating along a zero direction is undefined"
        )
    unit = operand / norm
    return hidden - torch.dot(hidden, unit) * unit


def apply_activation_patch(hidden: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Replace h with an activation captured from another run."""
    _validate_hidden(hidden)
    _validate_operand(hidden, source, "source_activation")
    return _as_hidden_dtype(hidden, source).clone()


def apply_intervention(
    spec: InterventionSpec,
    hidden: torch.Tensor,
    payload: InterventionPayload,
) -> tuple[torch.Tensor, InterventionDiagnostics]:
    """Dispatch one intervention and report its norm diagnostics."""
    if spec.mechanism is Mechanism.NOOP:
        result = apply_noop(hidden)
    elif spec.mechanism in (Mechanism.RESIDUAL_ADD, Mechanism.RANDOM_ADD):
        if payload.direction is None:
            raise InterventionShapeError(f"mechanism {spec.mechanism} requires a direction payload")
        result = apply_residual_add(hidden, payload.direction, spec.strength)
    elif spec.mechanism is Mechanism.DIRECTION_ABLATE:
        if payload.direction is None:
            raise InterventionShapeError("mechanism direction_ablate requires a direction payload")
        result = apply_direction_ablate(hidden, payload.direction)
    elif spec.mechanism is Mechanism.ACTIVATION_PATCH:
        if payload.source_activation is None:
            raise InterventionShapeError(
                "mechanism activation_patch requires a source_activation payload"
            )
        result = apply_activation_patch(hidden, payload.source_activation)
    else:
        raise InterventionShapeError(f"unhandled mechanism {spec.mechanism}")

    diagnostics = InterventionDiagnostics(
        pre_norm=float(torch.linalg.vector_norm(hidden)),
        post_norm=float(torch.linalg.vector_norm(result)),
        delta_norm=float(torch.linalg.vector_norm(result - hidden)),
    )
    return result, diagnostics
