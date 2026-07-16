"""A tiny, deterministic, locally built model.

This is real project code, not a test helper, because the smoke experiment depends on it.
It exists so that the whole pipeline (capture, intervene, score, commit, verify) can be
exercised on any machine with no network access, no Hugging Face authentication, and no
gated weights, in seconds rather than minutes.

What it is: a genuinely randomly initialized Llama-architecture causal LM with a word-level
tokenizer. Real forward pass, real residual stream, real hooks.

What it is not: a source of any result. A random model has no knowledge, so its answers are
noise. Nothing measured on the fixture is a finding about language models, and no number
produced from it may ever reach the dashboard. It tests the plumbing, not the science.
"""

from __future__ import annotations

import json
import string
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from ..logging_utils import info
from ..paths import artifacts_dir

FIXTURE_REVISION = "fixture-v1"
FIXTURE_SEED = 20260715

# Enough English to render the wrapper templates and the ARC-style prompts used in tests.
_BASE_VOCAB = [
    "<pad>",
    "<unk>",
    "<s>",
    "</s>",
    "this",
    "item",
    "is",
    "part",
    "of",
    "a",
    "controlled",
    "capability",
    "evaluation",
    "answer",
    "the",
    "question",
    "accurately",
    "using",
    "one",
    "letter",
    "being",
    "processed",
    "in",
    "normal",
    "application",
    "environment",
    "following",
    "what",
    "which",
    "why",
    "how",
    "and",
    "or",
    "to",
    "for",
    "with",
    "water",
    "energy",
    "plant",
    "animal",
    "earth",
    "sun",
    "heat",
    "light",
    "cell",
    "force",
    "gas",
    "solid",
    "liquid",
    "rock",
    "test",
    "hello",
    "world",
]


def fixture_dir(name: str = "tiny_llama") -> Path:
    return artifacts_dir() / "fixtures" / name


def _build_tokenizer() -> PreTrainedTokenizerFast:
    """Build a word-level tokenizer where each answer label is exactly one token.

    Single-token labels are a hard requirement of the scoring code, so the fixture
    tokenizer guarantees it by construction.
    """
    words = list(_BASE_VOCAB)
    words.extend(string.ascii_uppercase[:4])
    words.extend(str(digit) for digit in range(10))
    words.extend(".:,?()-")

    vocab = {word: index for index, word in enumerate(dict.fromkeys(words))}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )


def _special_token_id(tokenizer: PreTrainedTokenizerFast, attribute: str) -> int:
    """Read a special-token id, insisting it is a single resolved int.

    The tokenizer API types these loosely (a list, a string, or None are all possible), but
    the fixture defines all of them explicitly, so anything else means `_build_tokenizer` and
    this function have drifted apart.
    """
    value = getattr(tokenizer, attribute)
    if not isinstance(value, int):
        raise TypeError(
            f"fixture tokenizer {attribute} resolved to {value!r} ({type(value).__name__}), "
            "expected a single int"
        )
    return value


def build_fixture(
    name: str = "tiny_llama",
    hidden_size: int = 64,
    num_layers: int = 4,
    seed: int = FIXTURE_SEED,
    force: bool = False,
) -> Path:
    """Build the fixture model and save it. Idempotent unless `force` is set.

    Weights come from a fixed seed, so two machines that build the fixture get identical
    weights and therefore identical logits. That is what lets the no-op equality and rerun
    determinism tests assert exact numbers.
    """
    target = fixture_dir(name)
    if target.exists() and not force:
        return target

    tokenizer = _build_tokenizer()
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        bos_token_id=_special_token_id(tokenizer, "bos_token_id"),
        eos_token_id=_special_token_id(tokenizer, "eos_token_id"),
        pad_token_id=_special_token_id(tokenizer, "pad_token_id"),
        tie_word_embeddings=False,
    )

    torch.manual_seed(seed)
    model = LlamaForCausalLM(config)
    model.eval()

    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    (target / "fixture_info.json").write_text(
        json.dumps(
            {
                "name": name,
                "revision": FIXTURE_REVISION,
                "seed": seed,
                "hidden_size": hidden_size,
                "num_hidden_layers": num_layers,
                "vocab_size": tokenizer.vocab_size,
                "warning": "randomly initialized; produces no meaningful predictions",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    info("built fixture model", name=name, path=str(target), seed=seed)
    return target
