"""Intervention mechanisms and their application to real activations."""

from .directions import DirectionStore, matched_random_direction
from .tensor_ops import (
    InterventionPayload,
    InterventionShapeError,
    apply_activation_patch,
    apply_direction_ablate,
    apply_intervention,
    apply_noop,
    apply_residual_add,
)

__all__ = [
    "DirectionStore",
    "InterventionPayload",
    "InterventionShapeError",
    "apply_activation_patch",
    "apply_direction_ablate",
    "apply_intervention",
    "apply_noop",
    "apply_residual_add",
    "matched_random_direction",
]
