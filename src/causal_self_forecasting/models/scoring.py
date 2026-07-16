"""Answer-label scoring.

The measured quantity for the whole project is the correct-answer logit margin, so this
module is where the outcome variable is defined. Two decisions matter.

Single-token labels: a label whose text does not tokenize to exactly one token cannot be
scored from a single next-token distribution. Such a template is rejected rather than worked
around, because comparing a one-token label against a two-token label would compare
different quantities.

Margin definition: `margin = logit(correct) - max(logit(incorrect))`. Computed over the four
label logits only, not the whole vocabulary. It is positive when the model would answer
correctly among the four options, and its sign flips exactly when the answer changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class LabelTokenError(ValueError):
    """Raised when answer labels cannot be scored reliably."""


@dataclass(frozen=True)
class AnswerScores:
    """The scored next-token distribution restricted to the four answer labels."""

    logits: dict[str, float]
    probabilities: dict[str, float]
    predicted_label: str
    correct_label: str
    correct_logit: float
    best_incorrect_logit: float
    margin: float
    entropy: float

    @property
    def is_correct(self) -> bool:
        return self.predicted_label == self.correct_label


def resolve_label_token_ids(
    tokenizer,
    labels: list[str],
    prefix: str = "",
) -> dict[str, int]:
    """Map each answer label to exactly one token id.

    `prefix` handles tokenizers that treat a leading space as part of the token. If a label
    resolves to multiple tokens, or two labels collide on the same id, scoring is impossible
    and this raises.
    """
    resolved: dict[str, int] = {}
    for label in labels:
        text = f"{prefix}{label}"
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) != 1:
            raise LabelTokenError(
                f"answer label {text!r} tokenizes to {len(token_ids)} tokens ({token_ids}); "
                "a label must be a single token to be scored from one next-token distribution"
            )
        resolved[label] = int(token_ids[0])

    if len(set(resolved.values())) != len(resolved):
        collisions = {label: token_id for label, token_id in resolved.items()}
        raise LabelTokenError(
            f"answer labels collide on the same token ids: {collisions}; they cannot be told apart"
        )
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and unk_id in resolved.values():
        offenders = [label for label, token_id in resolved.items() if token_id == unk_id]
        raise LabelTokenError(
            f"answer labels {offenders} tokenize to the unknown token; "
            "this tokenizer cannot score them"
        )
    return resolved


def score_logits(
    next_token_logits: torch.Tensor,
    label_token_ids: dict[str, int],
    correct_label: str,
) -> AnswerScores:
    """Score one next-token logit vector against the answer labels.

    Probabilities are renormalized over the four labels. This is a conditional distribution
    given that the answer is one of the options, which is the quantity the constrained task
    format is asking about. Entropy is reported over the same four-way distribution, so it is
    bounded by log(4).
    """
    if next_token_logits.ndim != 1:
        raise ValueError(
            "expected a 1-D logit vector for one position, got shape "
            f"{tuple(next_token_logits.shape)}"
        )
    if correct_label not in label_token_ids:
        raise LabelTokenError(
            f"correct label {correct_label!r} is not among {sorted(label_token_ids)}"
        )

    # float64 on CPU: these logits get differenced and exponentiated, and the margins of
    # interest are small. The cost is negligible for a four-element vector.
    working = next_token_logits.detach().to(device="cpu", dtype=torch.float64)

    labels = sorted(label_token_ids)
    selected = torch.tensor(
        [working[label_token_ids[label]] for label in labels], dtype=torch.float64
    )
    logits = {label: float(value) for label, value in zip(labels, selected, strict=True)}

    probs_tensor = torch.softmax(selected, dim=0)
    probabilities = {label: float(value) for label, value in zip(labels, probs_tensor, strict=True)}

    predicted_label = max(logits, key=lambda label: logits[label])
    correct_logit = logits[correct_label]
    incorrect = [value for label, value in logits.items() if label != correct_label]
    best_incorrect_logit = max(incorrect)

    entropy = float(-(probs_tensor * torch.log(probs_tensor.clamp_min(1e-12))).sum())

    return AnswerScores(
        logits=logits,
        probabilities=probabilities,
        predicted_label=predicted_label,
        correct_label=correct_label,
        correct_logit=correct_logit,
        best_incorrect_logit=best_incorrect_logit,
        margin=correct_logit - best_incorrect_logit,
        entropy=entropy,
    )


def target_option_rate(predicted_labels: list[str], target_label: str) -> float:
    """Fraction of predictions equal to a target answer position.

    The model-organism metric. Reported alongside accuracy, never instead of it: a bias that
    only shows up as a shift in this rate while accuracy is unchanged is a different finding
    from one that costs accuracy.
    """
    if not predicted_labels:
        raise ValueError("cannot compute a target-option rate over zero predictions")
    return sum(1 for label in predicted_labels if label == target_label) / len(predicted_labels)
