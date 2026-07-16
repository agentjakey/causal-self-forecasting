"""Model loading.

Loading is centralized so that the `ModelSpec` recorded in a run manifest is produced by the
same code that did the loading. A manifest that claims a revision the loader did not actually
use would be worse than no manifest at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ..config import ModelConfig
from ..hashing import hash_file
from ..logging_utils import info
from ..schemas import ModelSpec, ModelVariant
from .device import dtype_name, resolve_device, resolve_dtype
from .fixture import FIXTURE_REVISION, build_fixture, fixture_dir


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded as specified."""


@dataclass(frozen=True)
class LoadedModel:
    """A loaded model plus everything needed to describe it in an artifact."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    spec: ModelSpec
    device: torch.device
    dtype: torch.dtype
    num_layers: int
    hidden_dim: int

    @property
    def decoder_layers(self) -> torch.nn.ModuleList:
        """The residual-stream blocks, for hook registration.

        Reached through the documented `model.model.layers` path. If a future architecture
        does not expose it, that fails loudly here rather than producing an intervention that
        silently attaches to nothing.
        """
        inner = getattr(self.model, "model", None)
        layers = getattr(inner, "layers", None)
        if layers is None:
            raise ModelLoadError(
                f"{type(self.model).__name__} does not expose model.model.layers; "
                "the intervention harness needs an explicit mapping for this architecture"
            )
        return layers

    @property
    def embedding_module(self) -> torch.nn.Module:
        inner = getattr(self.model, "model", None)
        embed = getattr(inner, "embed_tokens", None)
        if embed is None:
            raise ModelLoadError(
                f"{type(self.model).__name__} does not expose model.model.embed_tokens"
            )
        return embed


def _read_shape(model: PreTrainedModel) -> tuple[int, int]:
    """Read the layer count and hidden dimension from a loaded model's own config.

    Taken from the model rather than from our YAML so that the recorded shape is what was
    actually loaded. A config that disagrees with the weights should surface as an error
    here, not as an out-of-range layer much later.
    """
    inner = cast(Any, model).config
    try:
        return int(inner.num_hidden_layers), int(inner.hidden_size)
    except AttributeError as error:
        raise ModelLoadError(
            f"{type(model).__name__} config does not expose num_hidden_layers and hidden_size; "
            "this architecture needs an explicit mapping in the harness"
        ) from error


def _resolve_source(config: ModelConfig) -> tuple[str | Path, str]:
    if config.kind == "fixture":
        path = build_fixture(
            name=config.model_id,
            hidden_size=config.fixture_hidden_size,
            num_layers=config.fixture_num_layers,
        )
        return path, FIXTURE_REVISION
    return config.model_id, config.revision


def load_model(config: ModelConfig, adapter_path: str | None = None) -> LoadedModel:
    """Load a model and tokenizer according to a config.

    An adapter path switches the recorded variant to `adapted`, so a state captured from the
    adapted model can never be mislabeled as clean downstream.
    """
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype, device)
    source, revision = _resolve_source(config)

    if config.kind == "fixture" and adapter_path is not None:
        raise ModelLoadError("the fixture model does not support adapters")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            revision=revision if config.kind != "fixture" else None,
            trust_remote_code=config.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            source,
            revision=revision if config.kind != "fixture" else None,
            dtype=dtype,
            trust_remote_code=config.trust_remote_code,
        )
    except Exception as error:
        raise ModelLoadError(f"could not load {source} at revision {revision}: {error}") from error

    variant = ModelVariant.CLEAN
    adapter_hash = None
    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as error:
            raise ModelLoadError(
                "peft is required to load an adapter; run `uv sync --extra torch`"
            ) from error
        # PeftModel is not a PreTrainedModel subclass, but it forwards the surface this
        # harness uses: __call__, .config, and .model.layers. The cast records that the
        # substitution is deliberate rather than unnoticed.
        model = cast(PreTrainedModel, PeftModel.from_pretrained(model, adapter_path))
        variant = ModelVariant.ADAPTED

        weights = sorted(Path(adapter_path).glob("*.safetensors"))
        if not weights:
            raise ModelLoadError(f"no adapter weights found in {adapter_path}")
        adapter_hash = hash_file(weights[0])

    model = cast(PreTrainedModel, cast(Any, model).to(device))
    model.eval()
    # Gradients are needed only by the gradient baseline, which enables them explicitly.
    model.requires_grad_(False)

    num_layers, hidden_dim = _read_shape(model)

    spec = ModelSpec(
        model_id=str(config.model_id),
        revision=revision,
        variant=variant,
        adapter_path=adapter_path,
        adapter_hash=adapter_hash,
        dtype=dtype_name(dtype),
        device=str(device),
        trust_remote_code=config.trust_remote_code,
    )
    info(
        "loaded model",
        model_id=config.model_id,
        revision=revision,
        variant=variant.value,
        device=str(device),
        dtype=dtype_name(dtype),
        layers=num_layers,
        hidden_dim=hidden_dim,
    )
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        spec=spec,
        device=device,
        dtype=dtype,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )


def fixture_is_built(name: str) -> bool:
    return fixture_dir(name).exists()
