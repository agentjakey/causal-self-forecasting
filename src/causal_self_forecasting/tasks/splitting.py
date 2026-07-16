"""Deterministic, group-aware dataset splitting.

Splitting is a pure function of `group_id` and the split seed. It does not depend on the
order items arrive in, on how many items there are, or on which subset was loaded. That
property matters more than it looks: it means running with `--max-items 20` puts an item in
exactly the same split as a full run would, so a smoke run and a real run never disagree
about what is training data.
"""

from __future__ import annotations

import hashlib

from ..config import SplitConfig
from ..schemas import Split


def _group_unit_interval(group_id: str, seed: int) -> float:
    """Map a group id to a stable value in [0, 1)."""
    material = f"{seed}|{group_id}".encode()
    digest = hashlib.sha256(material).digest()
    # 53 bits is the mantissa width of a float64, so this uses the full precision available
    # without introducing rounding bias.
    value = int.from_bytes(digest[:8], "big") >> 11
    return value / float(1 << 53)


def assign_split(group_id: str, subject: str, config: SplitConfig) -> Split:
    """Assign one group to a split.

    A held-out subject wins over the random assignment. Subject holdout is a claim about
    generalization to unseen task categories, so those items must not appear in training
    under any circumstance.
    """
    if subject in config.heldout_subjects:
        return Split.HELDOUT_SUBJECT

    position = _group_unit_interval(group_id, config.split_seed)
    if position < config.train_fraction:
        return Split.TRAIN
    if position < config.train_fraction + config.val_fraction:
        return Split.VAL
    return Split.TEST
