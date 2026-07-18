"""Load real multiple-choice datasets and normalize them into `TaskItem` records.

Only two sources are supported, because two are enough for v0.1 and each one needs its own
verified field mapping. Adding a third source means writing its mapping and its test, not
flipping a flag.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..config import TaskConfig
from ..hashing import atomic_write_json, hash_file, hash_object, read_jsonl, write_jsonl
from ..logging_utils import info, warn
from ..paths import manifests_dir, processed_dir
from ..schemas import PromptVariant, TaskItem
from .prompts import render_variants
from .splitting import assign_split

_ID_CLEAN = re.compile(r"[^A-Za-z0-9_.:-]")
CHOICES_PER_ITEM = 4


class TaskLoadError(RuntimeError):
    """Raised when a dataset cannot be loaded or normalized."""


def _clean_id(raw: str) -> str:
    cleaned = _ID_CLEAN.sub("-", str(raw)).strip("-")
    if not cleaned or not cleaned[0].isalnum():
        cleaned = f"i{cleaned}"
    return cleaned[:120]


def _load_hf_dataset(config: TaskConfig) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise TaskLoadError(
            "the `datasets` package is required for data preparation; run `uv sync`"
        ) from error
    try:
        return load_dataset(config.source, config.source_config, split=config.source_split)
    except Exception as error:
        raise TaskLoadError(
            f"could not load {config.source} (config={config.source_config}, "
            f"split={config.source_split}): {error}"
        ) from error


def _normalize_arc(row: dict[str, Any], index: int, config: TaskConfig) -> TaskItem | None:
    """Map one ai2_arc row.

    ARC mixes three-, four-, and five-option questions and labels them either `A`-`E` or
    `1`-`5`. Anything that is not exactly four options is dropped rather than padded,
    because padding would invent a distractor that the dataset authors never wrote.
    """
    choices = row.get("choices") or {}
    texts = list(choices.get("text") or [])
    labels = [str(label) for label in (choices.get("label") or [])]
    answer_key = str(row.get("answerKey", "")).strip()

    if len(texts) != CHOICES_PER_ITEM or len(labels) != CHOICES_PER_ITEM:
        return None
    if answer_key not in labels:
        return None

    item_id = _clean_id(row.get("id") or f"{config.name}-{index}")
    return TaskItem(
        item_id=item_id,
        group_id=item_id,
        source=config.source,
        source_id=str(row.get("id") or index),
        subject=config.source_config or "arc",
        question=str(row["question"]),
        choices=[str(text) for text in texts],
        answer_index=labels.index(answer_key),
        split=assign_split(item_id, config.source_config or "arc", config.splits),
    )


def _normalize_mmlu(row: dict[str, Any], index: int, config: TaskConfig) -> TaskItem | None:
    """Map one cais/mmlu row."""
    texts = list(row.get("choices") or [])
    answer = row.get("answer")
    if len(texts) != CHOICES_PER_ITEM or not isinstance(answer, int):
        return None
    if not 0 <= answer < CHOICES_PER_ITEM:
        return None

    subject = str(row.get("subject") or config.source_config or "mmlu")
    item_id = _clean_id(f"{subject}-{index}")
    return TaskItem(
        item_id=item_id,
        group_id=item_id,
        source=config.source,
        source_id=f"{subject}:{index}",
        subject=subject,
        question=str(row["question"]),
        choices=[str(text) for text in texts],
        answer_index=answer,
        split=assign_split(item_id, subject, config.splits),
    )


_NORMALIZERS = {
    "allenai/ai2_arc": _normalize_arc,
    "cais/mmlu": _normalize_mmlu,
}


def load_task_items(config: TaskConfig, max_items: int | None = None) -> list[TaskItem]:
    """Load and normalize a task into `TaskItem` records.

    Items are sorted by id before truncation so that `--max-items` selects the same items on
    every machine, whatever order the dataset happens to stream in.
    """
    normalizer = _NORMALIZERS.get(config.source)
    if normalizer is None:
        raise TaskLoadError(
            f"no verified field mapping for source {config.source!r}; "
            f"supported sources are {sorted(_NORMALIZERS)}"
        )

    dataset = _load_hf_dataset(config)
    items: list[TaskItem] = []
    skipped = 0
    for index, row in enumerate(dataset):
        item = normalizer(dict(row), index, config)
        if item is None:
            skipped += 1
            continue
        if config.subjects and item.subject not in config.subjects:
            continue
        items.append(item)

    if skipped:
        info(
            "dropped rows that are not four-choice questions",
            source=config.source,
            dropped=skipped,
            kept=len(items),
        )
    if not items:
        raise TaskLoadError(f"task {config.name} produced zero usable items")

    items.sort(key=lambda item: item.item_id)
    limit = max_items if max_items is not None else config.max_items
    if limit is not None:
        items = items[:limit]
    return items


def load_prepared_task(task_name: str) -> tuple[list[TaskItem], list[PromptVariant]]:
    """Read a task that `prepare_task` has already written.

    The single reader for prepared data, so that trial generation and the systems benchmark
    cannot drift into disagreeing about what is on disk.
    """
    directory = processed_dir() / task_name
    items_path = directory / "items.jsonl"
    variants_path = directory / "variants.jsonl"
    if not items_path.exists() or not variants_path.exists():
        raise TaskLoadError(
            f"no prepared data for task {task_name!r} at {directory}; "
            f"run `csf data prepare --config configs/tasks/{task_name}.yaml` first"
        )
    items = [TaskItem.model_validate(row) for row in read_jsonl(items_path)]
    variants = [PromptVariant.model_validate(row) for row in read_jsonl(variants_path)]
    return items, variants


def task_manifest_path(task_name: str) -> Path:
    return manifests_dir() / f"{task_name}.json"


def _split_counts(records: list[TaskItem] | list[PromptVariant]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.split.value] = counts.get(record.split.value, 0) + 1
    return counts


def prepare_task(
    config: TaskConfig,
    config_path: str,
    max_items: int | None = None,
) -> dict[str, Any]:
    """Load a task, render every wrapper variant, and write the processed dataset.

    Returns the manifest, which carries the counts and hashes that a run manifest later
    cites. Nothing here is measured behavior of a model; it is dataset bookkeeping only.
    """
    items = load_task_items(config, max_items=max_items)
    variants: list[PromptVariant] = []
    for item in items:
        variants.extend(render_variants(item, config))

    out_dir = processed_dir() / config.name
    items_path = write_jsonl(out_dir / "items.jsonl", items)
    variants_path = write_jsonl(out_dir / "variants.jsonl", variants)

    item_splits = _split_counts(items)
    for split_name in ("train", "val", "test"):
        if item_splits.get(split_name, 0) == 0:
            warn(
                "a split is empty at this item count",
                task=config.name,
                split=split_name,
                items=len(items),
            )

    manifest = {
        "task_name": config.name,
        "config_path": config_path,
        "config_hash": hash_object(config.model_dump(mode="json")),
        "source": config.source,
        "source_config": config.source_config,
        "source_split": config.source_split,
        "item_count": len(items),
        "variant_count": len(variants),
        "group_count": len({item.group_id for item in items}),
        "item_splits": item_splits,
        "variant_splits": _split_counts(variants),
        "wrapper_ids": [wrapper.wrapper_id for wrapper in config.wrappers],
        "heldout_wrapper_ids": [w.wrapper_id for w in config.wrappers if w.heldout],
        "files": {
            "items": {
                "path": str(items_path.relative_to(Path(processed_dir()).parents[1])),
                "hash": hash_file(items_path),
            },
            "variants": {
                "path": str(variants_path.relative_to(Path(processed_dir()).parents[1])),
                "hash": hash_file(variants_path),
            },
        },
    }
    atomic_write_json(manifests_dir() / f"{config.name}.json", manifest)
    return manifest
