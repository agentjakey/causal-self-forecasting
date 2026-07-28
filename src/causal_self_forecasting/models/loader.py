"""Model loading.

Loading is centralized so that the `ModelSpec` recorded in a run manifest is produced by the
same code that did the loading. A manifest that claims a revision the loader did not actually
use would be worse than no manifest at all.
"""

from __future__ import annotations

from collections.abc import Sequence
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
class OutputEmbedding:
    """The output-embedding matrix, with enough provenance to cite it.

    `revision` is carried so that a direction built from these rows can name the exact weights
    it came from. The matrix is not copied; it is the live parameter, so callers must not
    mutate it.
    """

    weight: torch.Tensor
    source: str
    tied: bool
    config_tie_word_embeddings: bool | None
    vocab_size: int
    hidden_dim: int
    model_id: str
    revision: str


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

    def output_embedding(self, required_token_ids: Sequence[int] = ()) -> OutputEmbedding:
        """The output-embedding weight matrix, validated.

        Reads the already-loaded model. There is deliberately no second loading path here: a
        direction built from weights loaded by different code than the run that uses it could
        cite a revision it never actually read.

        `get_output_embeddings()` is the documented accessor and is preferred. The tied input
        embedding is used only when the model exposes no output embedding at all, and tying is
        then reported as a fact read off the loaded tensors rather than trusted from the
        config, because a config flag and the weights actually in memory can disagree.

        Finiteness is checked on the rows in `required_token_ids` rather than on the whole
        matrix. A full check on a 262144 x 1152 matrix would allocate hundreds of megabytes to
        validate values no caller reads; the rows that are read are checked exactly.
        """
        inner = cast(Any, self.model)
        output_module = None
        getter = getattr(inner, "get_output_embeddings", None)
        if callable(getter):
            output_module = getter()

        input_module = None
        input_getter = getattr(inner, "get_input_embeddings", None)
        if callable(input_getter):
            input_module = input_getter()

        output_weight = getattr(output_module, "weight", None)
        input_weight = getattr(input_module, "weight", None)

        if output_weight is not None:
            weight = output_weight
            source = "get_output_embeddings"
        elif input_weight is not None:
            weight = input_weight
            source = "tied_input_embeddings"
        else:
            raise ModelLoadError(
                f"{type(self.model).__name__} exposes neither an output embedding with a "
                "weight matrix nor an input embedding to fall back on; this architecture needs "
                "an explicit mapping before answer-token directions can be built from it"
            )

        if not isinstance(weight, torch.Tensor):
            raise ModelLoadError(f"the {source} weight is a {type(weight).__name__}, not a tensor")
        if weight.ndim != 2:
            raise ModelLoadError(
                f"the {source} weight must be 2-D (vocab, hidden), got shape {tuple(weight.shape)}"
            )

        vocab_size, hidden = int(weight.shape[0]), int(weight.shape[1])
        if hidden != self.hidden_dim:
            raise ModelLoadError(
                f"the {source} weight has hidden axis {hidden} but the loaded model reports "
                f"hidden_dim {self.hidden_dim}; the matrix is not the unembedding for these "
                "weights, or the axes are transposed"
            )

        for token_id in required_token_ids:
            if not 0 <= int(token_id) < vocab_size:
                raise ModelLoadError(
                    f"token id {token_id} is outside the {source} vocabulary axis of {vocab_size}"
                )
            row = weight[int(token_id)]
            if not bool(torch.isfinite(row).all()):
                raise ModelLoadError(
                    f"the {source} row for token id {token_id} contains non-finite values"
                )

        # Read tying off the tensors, not off the config. Both are recorded so a disagreement
        # is visible rather than silently resolved in favour of whichever was consulted.
        tied = (
            output_weight is not None
            and input_weight is not None
            and output_weight.shape == input_weight.shape
            and output_weight.data_ptr() == input_weight.data_ptr()
        )
        declared = getattr(getattr(self.model, "config", None), "tie_word_embeddings", None)

        return OutputEmbedding(
            weight=weight,
            source=source,
            tied=tied,
            config_tie_word_embeddings=None if declared is None else bool(declared),
            vocab_size=vocab_size,
            hidden_dim=hidden,
            model_id=self.spec.model_id,
            revision=self.spec.revision,
        )


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
