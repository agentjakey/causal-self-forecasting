"""Hidden-state capture and intervened forward passes.

Layer indexing convention, used everywhere in this project:

* Layer `0` is the output of the embedding module.
* Layer `i` (for `i` in 1..num_layers) is the output of decoder block `i - 1`.

This is the residual stream before the model's final norm. Capture and intervention refer to
exactly the same tensor at the same point, which `test_capture_matches_the_intervention_point`
pins for every layer.

Capture uses this module's own forward hooks rather than the `output_hidden_states` flag.
That is a deliberate choice, not an accident of style. In transformers 5, `LlamaModel.forward`
does not assemble the hidden-state tuple inline; the framework records it with internal hooks
whose ordering relative to a user hook is an implementation detail. Measured on transformers
5.14, an intervention hook combined with `output_hidden_states=True` returned states that were
partly pre-intervention, so a patched state read back as its unpatched value while the forward
pass really had been modified. Nothing warned; the numbers simply meant something else.

Owning both hooks removes that coupling. The intervention hook is registered before the
capture hooks, and torch passes each hook's returned output on to the next, so a capture at the
intervened layer observes the post-intervention value by construction.

Batch size is fixed at one. Padding a batch would put the last real token at a different index
per row, and this project reads the final-token residual stream.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import torch

from ..interventions.tensor_ops import (
    InterventionDiagnostics,
    InterventionPayload,
    apply_intervention,
)
from ..schemas import InterventionSpec
from .loader import LoadedModel


class CaptureError(RuntimeError):
    """Raised when capture or intervention cannot be performed as requested."""


@dataclass(frozen=True)
class CaptureResult:
    """The result of one forward pass."""

    next_token_logits: torch.Tensor
    hidden_states: dict[int, torch.Tensor]
    sequence_length: int
    position_absolute: int
    diagnostics: InterventionDiagnostics | None = None


def resolve_position(position_index: int, sequence_length: int) -> int:
    """Convert a possibly negative position into an absolute index."""
    absolute = position_index if position_index >= 0 else sequence_length + position_index
    if not 0 <= absolute < sequence_length:
        raise CaptureError(
            f"position {position_index} is out of range for a sequence of length {sequence_length}"
        )
    return absolute


def validate_layer(layer: int, model: LoadedModel) -> int:
    if not 0 <= layer <= model.num_layers:
        raise CaptureError(
            f"layer {layer} is out of range; this model has {model.num_layers} blocks, so valid "
            f"layer indices are 0 (embeddings) through {model.num_layers}"
        )
    return layer


def _target_module(layer: int, model: LoadedModel) -> torch.nn.Module:
    """The module whose output is the residual stream at `layer`."""
    if layer == 0:
        return model.embedding_module
    return model.decoder_layers[layer - 1]


def _split_output(output):
    """Blocks return either a tensor or a tuple whose first element is the tensor."""
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _rebuild_output(hidden: torch.Tensor, rest):
    if rest is None:
        return hidden
    return (hidden, *rest)


def _tokenize(model: LoadedModel, prompt: str) -> dict[str, torch.Tensor]:
    encoded = model.tokenizer(prompt, return_tensors="pt")
    if encoded["input_ids"].shape[0] != 1:
        raise CaptureError("capture requires a batch size of one")
    if encoded["input_ids"].shape[1] == 0:
        raise CaptureError("prompt tokenized to zero tokens")
    return {key: value.to(model.device) for key, value in encoded.items()}


@contextmanager
def _capture_hook(
    model: LoadedModel,
    layer: int,
    position_index: int,
    sink: dict[int, torch.Tensor],
) -> Iterator[None]:
    """Read one position of the residual stream at `layer` without modifying it."""

    def hook(module, args, output):
        hidden, _ = _split_output(output)
        absolute = resolve_position(position_index, hidden.shape[1])
        sink[layer] = hidden[0, absolute].detach().clone()
        return None

    handle = _target_module(layer, model).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def _intervention_hook(
    model: LoadedModel,
    layer: int,
    position_index: int,
    transform: Callable[[torch.Tensor], torch.Tensor],
    record: dict,
) -> Iterator[None]:
    """Rewrite one position of the residual stream at `layer`."""

    def hook(module, args, output):
        hidden, rest = _split_output(output)
        absolute = resolve_position(position_index, hidden.shape[1])
        original = hidden[0, absolute]
        replacement = transform(original)
        if replacement.shape != original.shape:
            raise CaptureError(
                f"intervention returned shape {tuple(replacement.shape)} but the residual "
                f"stream at layer {layer} has shape {tuple(original.shape)}"
            )
        updated = hidden.clone()
        updated[0, absolute] = replacement.to(dtype=hidden.dtype, device=hidden.device)
        record["fired"] = True
        return _rebuild_output(updated, rest)

    handle = _target_module(layer, model).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def capture_hidden_states(
    model: LoadedModel,
    prompt: str,
    layers: list[int],
    position_index: int = -1,
) -> CaptureResult:
    """Run a clean forward pass and capture the residual stream at the requested layers."""
    for layer in layers:
        validate_layer(layer, model)

    inputs = _tokenize(model, prompt)
    sink: dict[int, torch.Tensor] = {}

    with ExitStack() as stack:
        for layer in sorted(set(layers)):
            stack.enter_context(_capture_hook(model, layer, position_index, sink))
        outputs = model.model(**inputs)

    missing = sorted(set(layers) - set(sink))
    if missing:
        raise CaptureError(f"capture hooks for layers {missing} never fired")

    sequence_length = int(inputs["input_ids"].shape[1])
    return CaptureResult(
        next_token_logits=outputs.logits[0, -1].detach().clone(),
        hidden_states=sink,
        sequence_length=sequence_length,
        position_absolute=resolve_position(position_index, sequence_length),
    )


@torch.no_grad()
def run_with_intervention(
    model: LoadedModel,
    prompt: str,
    spec: InterventionSpec,
    payload: InterventionPayload,
    capture_layers: list[int] | None = None,
) -> CaptureResult:
    """Run a forward pass with one intervention applied.

    Raises if the intervention hook never fires. A hook that silently does not run would
    return a clean result under an intervention's label, which is the one failure that every
    downstream analysis would accept without complaint.
    """
    capture_layers = capture_layers or []
    for layer in capture_layers:
        validate_layer(layer, model)
    validate_layer(spec.layer, model)

    diagnostics_box: dict[str, InterventionDiagnostics] = {}
    record: dict[str, bool] = {"fired": False}
    sink: dict[int, torch.Tensor] = {}

    def transform(hidden: torch.Tensor) -> torch.Tensor:
        result, diagnostics = apply_intervention(spec, hidden, payload)
        diagnostics_box["value"] = diagnostics
        return result

    inputs = _tokenize(model, prompt)

    with ExitStack() as stack:
        # Registered first, so that a capture at the same layer sees the post-intervention
        # value: torch feeds each hook's returned output to the hooks that follow it.
        stack.enter_context(
            _intervention_hook(model, spec.layer, spec.position_index, transform, record)
        )
        for layer in sorted(set(capture_layers)):
            stack.enter_context(_capture_hook(model, layer, spec.position_index, sink))
        outputs = model.model(**inputs)

    if not record["fired"]:
        raise CaptureError(
            f"the intervention hook at layer {spec.layer} never fired; the intervention was "
            "not applied and any result from this pass would be a clean run mislabeled"
        )

    sequence_length = int(inputs["input_ids"].shape[1])
    return CaptureResult(
        next_token_logits=outputs.logits[0, -1].detach().clone(),
        hidden_states=sink,
        sequence_length=sequence_length,
        position_absolute=resolve_position(spec.position_index, sequence_length),
        diagnostics=diagnostics_box.get("value"),
    )
