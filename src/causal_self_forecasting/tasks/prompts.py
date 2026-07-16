"""Prompt rendering.

Wrappers supply the framing text. This module supplies the question body and enforces that
every rendered variant of an item differs only in wrapper text, which is the precondition
for the same-prompt state-swap control later on.
"""

from __future__ import annotations

from ..config import TaskConfig, WrapperSpec
from ..schemas import PromptVariant, Split, TaskItem


def render_choices(item: TaskItem, labels: list[str]) -> str:
    """Render the four options, one per line, as `A. text`."""
    return "\n".join(
        f"{label}. {choice}" for label, choice in zip(labels, item.choices, strict=True)
    )


def render_prompt(item: TaskItem, wrapper: WrapperSpec, labels: list[str]) -> str:
    return wrapper.template.format(
        question=item.question.strip(), choices=render_choices(item, labels)
    )


def variant_split(item: TaskItem, wrapper: WrapperSpec) -> Split:
    """Resolve the split for one rendered variant.

    A held-out wrapper forces the variant into the wrapper-paraphrase split even when its
    item is a training item. The generalization claim being tested is about unseen phrasings,
    so the phrasing decides, not the question.
    """
    if wrapper.heldout:
        return Split.HELDOUT_WRAPPER
    return item.split


def render_variants(item: TaskItem, config: TaskConfig) -> list[PromptVariant]:
    """Render one item under every configured wrapper."""
    variants: list[PromptVariant] = []
    for wrapper in config.wrappers:
        variants.append(
            PromptVariant(
                variant_id=f"{item.item_id}.{wrapper.wrapper_id}",
                item_id=item.item_id,
                group_id=item.group_id,
                wrapper_id=wrapper.wrapper_id,
                framing=wrapper.framing,
                prompt_text=render_prompt(item, wrapper, config.answer_labels),
                answer_labels=list(config.answer_labels),
                split=variant_split(item, wrapper),
            )
        )
    return variants
