"""Deterministic prompt-role manifests.

A prompt manifest freezes which prompts play which role in the BlueDot state-dependence arm.
It is built once, before any state is captured, and everything downstream cites it by hash.

Three properties do the work, and each is enforced rather than documented:

*Determinism.* Role assignment is a pure function of the master seed and the `group_id`. The
eligible groups are sorted before anything seeded happens, then ordered by a per-group derived
key with the group id as the tie-break, then cut into contiguous role blocks. Nothing depends
on the order rows arrived in, on how many rows were loaded, or on anything the model produced.
The same pool and the same seed give the same manifest on any machine.

*Blindness.* Selection sees a group id and a seed. It does not see clean correctness, label
logits, confidence, hidden states, intervention effects, gold-label balance, or question topic.
A split chosen with any of those in view would not be a split, it would be a result.

*Checkability.* The manifest carries a content hash over its own payload, and `PromptManifest`
recomputes it on load. Editing an assignment, reordering the list, or changing a role makes the
file fail to parse. The hash deliberately excludes the manifest's own filesystem path and its
creation timestamp, so the same selection hashes the same after the repository moves.

Nothing here loads a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import SCHEMA_VERSION
from ..config import ConfigError, PromptManifestConfig, TaskConfig, load_config, repo_root
from ..hashing import (
    atomic_write_json,
    canonical_json_bytes,
    hash_object,
    read_json,
    sha256_hex,
)
from ..logging_utils import info
from ..paths import manifests_dir, prompt_manifest_path
from ..reproducibility import derive_seed
from ..schemas import (
    PromptAssignment,
    PromptManifest,
    PromptRole,
    PromptVariant,
    TaskItem,
    compute_prompt_manifest_hash,
    prompt_manifest_payload,
)
from .loader import TaskLoadError, load_prepared_task

# Bump when the mapping from (pool, seed) to roles changes. A manifest built under a different
# algorithm version is not comparable to one built under this version, and the version travels
# inside the hashed payload so the two cannot be confused.
SELECTION_ALGORITHM = "seeded_group_permutation_contiguous_roles"
SELECTION_ALGORITHM_VERSION = "1.0"

# Namespaced so that changing another seeded step cannot shift this one.
SELECTION_SEED_LABEL = "bluedot.prompt_manifest"

# The order roles are cut from the permutation. Fixed, because reordering it would silently
# reassign every prompt while leaving counts, seed, and pool identical.
ROLE_ORDER: tuple[PromptRole, ...] = (
    PromptRole.SMOKE,
    PromptRole.CALIBRATION,
    PromptRole.TRAINING,
    PromptRole.FINAL_TEST,
)

# Task-manifest fields the prompt manifest anchors to. Path fields are excluded on purpose:
# `config_path` and `files.*.path` are written with the host's separators, so hashing them
# would make an otherwise identical manifest hash differently on Windows and Linux.
_TASK_MANIFEST_PORTABLE_KEYS = (
    "task_name",
    "source",
    "source_config",
    "source_split",
    "config_hash",
    "item_count",
    "variant_count",
    "group_count",
    "item_splits",
    "variant_splits",
    "wrapper_ids",
    "heldout_wrapper_ids",
)


class PromptManifestError(RuntimeError):
    """Raised when a prompt manifest cannot be built, written, or trusted."""


# ---------------------------------------------------------------------------
# Task-side hashing
# ---------------------------------------------------------------------------


def task_manifest_file(task_name: str) -> Path:
    return manifests_dir() / f"{task_name}.json"


def read_task_manifest(task_name: str) -> dict[str, Any]:
    path = task_manifest_file(task_name)
    if not path.exists():
        raise PromptManifestError(
            f"no task manifest at {path}; run `csf data prepare --config "
            f"configs/tasks/{task_name}.yaml` before freezing a prompt manifest"
        )
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise PromptManifestError(f"{path} is not a task manifest object")
    return raw


def task_manifest_hashes(task_name: str) -> tuple[str, str, str]:
    """Return the portable task-manifest hash and the item and variant file hashes.

    The first is a hash of the task manifest's portable fields, not of its bytes. It changes
    whenever the prepared data, the task config, or the counts change, and does not change when
    the repository is checked out somewhere else.
    """
    raw = read_task_manifest(task_name)
    files = raw.get("files") or {}
    items_hash = (files.get("items") or {}).get("hash")
    variants_hash = (files.get("variants") or {}).get("hash")
    if not isinstance(items_hash, str) or not isinstance(variants_hash, str):
        raise PromptManifestError(
            f"task manifest for {task_name!r} does not record item and variant file hashes; "
            "it was written by an older version and must be regenerated"
        )

    payload = {key: raw[key] for key in _TASK_MANIFEST_PORTABLE_KEYS if key in raw}
    payload["items_hash"] = items_hash
    payload["variants_hash"] = variants_hash
    return hash_object(payload), items_hash, variants_hash


# ---------------------------------------------------------------------------
# Pool integrity
# ---------------------------------------------------------------------------


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return sorted(repeated)


def eligible_groups(
    items: list[TaskItem],
    variants: list[PromptVariant],
    canonical_wrapper_id: str,
) -> tuple[list[str], dict[str, PromptVariant], dict[str, TaskItem]]:
    """Validate the prepared pool and return the groups that can be selected from.

    Every failure here is loud. A pool that quietly loses a group to a missing wrapper, or that
    reuses an item id across groups, would produce a manifest that looks complete and is not,
    and the whole point of this artifact is that it can be checked rather than trusted.
    """
    if not items:
        raise PromptManifestError("the prepared task has no items")

    item_ids = [item.item_id for item in items]
    repeated_items = _duplicates(item_ids)
    if repeated_items:
        raise PromptManifestError(
            f"the prepared pool has duplicate item ids: {repeated_items[:5]}"
            f"{' and more' if len(repeated_items) > 5 else ''}"
        )
    group_ids = [item.group_id for item in items]
    repeated_groups = _duplicates(group_ids)
    if repeated_groups:
        raise PromptManifestError(
            f"the prepared pool has duplicate group ids: {repeated_groups[:5]}"
            f"{' and more' if len(repeated_groups) > 5 else ''}"
        )

    item_by_group = {item.group_id: item for item in items}

    canonical: dict[str, list[PromptVariant]] = {}
    for variant in variants:
        if variant.wrapper_id != canonical_wrapper_id:
            continue
        canonical.setdefault(variant.group_id, []).append(variant)

    unknown_groups = sorted(set(canonical) - set(item_by_group))
    if unknown_groups:
        raise PromptManifestError(
            f"these canonical variants reference groups that are not in the item pool: "
            f"{unknown_groups[:5]}"
        )

    missing: list[str] = []
    ambiguous: list[str] = []
    for group_id in sorted(item_by_group):
        found = canonical.get(group_id, [])
        if not found:
            missing.append(group_id)
        elif len(found) > 1:
            ambiguous.append(group_id)
    if missing:
        raise PromptManifestError(
            f"{len(missing)} groups have no {canonical_wrapper_id!r} variant, for example "
            f"{missing[:5]}. Every group must render exactly one canonical prompt; dropping "
            "the ones that do not would silently change the pool the split is drawn from."
        )
    if ambiguous:
        raise PromptManifestError(
            f"{len(ambiguous)} groups have more than one {canonical_wrapper_id!r} variant, for "
            f"example {ambiguous[:5]}; the canonical prompt would be ambiguous"
        )

    mismatched = sorted(
        group_id
        for group_id, found in canonical.items()
        if found[0].item_id != item_by_group[group_id].item_id
    )
    if mismatched:
        raise PromptManifestError(
            f"these canonical variants disagree with their item's id: {mismatched[:5]}"
        )

    variant_by_group = {group_id: found[0] for group_id, found in canonical.items()}
    return sorted(item_by_group), variant_by_group, item_by_group


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def selection_key(group_id: str, master_seed: int) -> tuple[int, str]:
    """The sort key that orders the eligible pool.

    The group id is the tie-break, so two groups whose derived seeds collide still order
    deterministically instead of falling back on input order.
    """
    return derive_seed(SELECTION_SEED_LABEL, master_seed, group_id), group_id


def ordered_groups(candidates: list[str], master_seed: int) -> list[str]:
    """Permute the eligible groups deterministically.

    Sorted lexicographically first so the input to the seeded step is itself order-independent,
    then sorted by the derived key.
    """
    return sorted(sorted(candidates), key=lambda group_id: selection_key(group_id, master_seed))


def check_role_disjointness(assignments: list[PromptAssignment]) -> None:
    """Assert that no group and no item appears under two roles.

    A post-condition rather than a filter. The construction cannot produce an overlap, so this
    firing means an assumption broke, and it should say so loudly rather than be repaired.
    """
    groups_by_role: dict[str, set[str]] = {}
    items_by_role: dict[str, set[str]] = {}
    for assignment in assignments:
        groups_by_role.setdefault(assignment.role.value, set()).add(assignment.group_id)
        items_by_role.setdefault(assignment.role.value, set()).add(assignment.item_id)

    for label, by_role in (("group_id", groups_by_role), ("item_id", items_by_role)):
        roles = sorted(by_role)
        for index, left in enumerate(roles):
            for right in roles[index + 1 :]:
                shared = sorted(by_role[left] & by_role[right])
                if shared:
                    raise PromptManifestError(
                        f"roles {left!r} and {right!r} share {len(shared)} {label} values, for "
                        f"example {shared[:5]}; prompt roles must be disjoint"
                    )


def build_prompt_manifest(
    manifest_config: PromptManifestConfig,
    task_config: TaskConfig,
    master_seed: int | None = None,
) -> PromptManifest:
    """Build the frozen manifest. Loads no model and runs no forward pass."""
    wrapper_ids = [wrapper.wrapper_id for wrapper in task_config.wrappers]
    canonical = manifest_config.canonical_wrapper_id
    if canonical not in wrapper_ids:
        raise PromptManifestError(
            f"wrapper {canonical!r} is not defined by task {task_config.name!r}; "
            f"available wrappers are {wrapper_ids}"
        )
    wrapper = next(w for w in task_config.wrappers if w.wrapper_id == canonical)
    if wrapper.heldout:
        raise PromptManifestError(
            f"wrapper {canonical!r} is a held-out paraphrase and cannot be the canonical "
            "prompt; held-out wrappers are reserved for the unseen-phrasing test"
        )

    try:
        items, variants = load_prepared_task(task_config.name)
    except TaskLoadError as error:
        raise PromptManifestError(str(error)) from error

    task_hash, items_hash, variants_hash = task_manifest_hashes(task_config.name)

    candidates, variant_by_group, item_by_group = eligible_groups(items, variants, canonical)

    required = manifest_config.total_prompts
    if len(candidates) < required:
        raise PromptManifestError(
            f"the prepared pool has {len(candidates)} eligible groups but the manifest needs "
            f"{required}. Prepare more items (`csf data prepare --config "
            f"configs/tasks/{task_config.name}.yaml --max-items N`). Do not shrink a role, "
            "reuse a group, or relax disjointness to fit the pool."
        )

    seed = manifest_config.master_seed if master_seed is None else int(master_seed)
    permuted = ordered_groups(candidates, master_seed=seed)
    selected = permuted[:required]

    role_counts = {role.value: manifest_config.role_counts[role] for role in ROLE_ORDER}

    assignments: list[PromptAssignment] = []
    cursor = 0
    for role in ROLE_ORDER:
        count = manifest_config.role_counts[role]
        for group_id in selected[cursor : cursor + count]:
            variant = variant_by_group[group_id]
            assignments.append(
                PromptAssignment(
                    selection_index=len(assignments),
                    variant_id=variant.variant_id,
                    item_id=item_by_group[group_id].item_id,
                    group_id=group_id,
                    role=role,
                    wrapper_id=variant.wrapper_id,
                    split=variant.split,
                    prompt_hash=sha256_hex(variant.prompt_text.encode("utf-8")),
                )
            )
        cursor += count

    check_role_disjointness(assignments)

    manifest_path = task_manifest_file(task_config.name)
    try:
        relative = str(manifest_path.relative_to(repo_root())).replace("\\", "/")
    except ValueError:
        relative = manifest_path.name

    # The hash is computed from the same field values the record will carry, so the record's
    # validator recomputing it is a real check rather than a restatement.
    draft: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest_config.manifest_id,
        "task_name": task_config.name,
        "canonical_wrapper_id": canonical,
        "master_seed": seed,
        "selection_algorithm": SELECTION_ALGORITHM,
        "selection_algorithm_version": SELECTION_ALGORITHM_VERSION,
        "task_manifest_hash": task_hash,
        "items_hash": items_hash,
        "variants_hash": variants_hash,
        "eligible_group_count": len(candidates),
        "role_counts": role_counts,
        "assignments": [assignment.model_dump(mode="json") for assignment in assignments],
    }

    return PromptManifest(
        manifest_id=manifest_config.manifest_id,
        task_name=task_config.name,
        canonical_wrapper_id=canonical,
        master_seed=seed,
        selection_algorithm=SELECTION_ALGORITHM,
        selection_algorithm_version=SELECTION_ALGORITHM_VERSION,
        task_manifest_hash=task_hash,
        items_hash=items_hash,
        variants_hash=variants_hash,
        eligible_group_count=len(candidates),
        role_counts=role_counts,
        assignments=assignments,
        manifest_hash=compute_prompt_manifest_hash(draft),
        task_manifest_path=relative,
    )


# ---------------------------------------------------------------------------
# Storage and verification
# ---------------------------------------------------------------------------


def manifest_content_bytes(manifest: PromptManifest) -> bytes:
    """The canonical bytes the manifest hash is computed over.

    This, not the file, is the thing that must be byte-identical across rebuilds. The file
    additionally carries a timestamp and a path, which are provenance and are expected to vary.
    """
    return canonical_json_bytes(prompt_manifest_payload(manifest.model_dump(mode="json")))


def load_prompt_manifest(manifest_id: str) -> PromptManifest:
    path = prompt_manifest_path(manifest_id)
    if not path.exists():
        raise PromptManifestError(f"no prompt manifest at {path}")
    try:
        return PromptManifest.model_validate(read_json(path))
    except Exception as error:
        raise PromptManifestError(f"{path} is not a valid prompt manifest: {error}") from error


def write_prompt_manifest(manifest: PromptManifest, force: bool = False) -> tuple[Path, str]:
    """Write a manifest atomically, refusing to replace a different one.

    Returns the path and one of `written`, `unchanged`, or `overwritten`. An existing manifest
    with the same content hash is left exactly as it is: rewriting it would change its
    timestamp for no reason and break the byte-identical rerun guarantee.
    """
    path = prompt_manifest_path(manifest.manifest_id)
    if path.exists():
        existing = load_prompt_manifest(manifest.manifest_id)
        if existing.manifest_hash == manifest.manifest_hash:
            return path, "unchanged"
        if not force:
            raise PromptManifestError(
                f"{path} already holds a different manifest "
                f"(existing {existing.manifest_hash[:23]}, new {manifest.manifest_hash[:23]}). "
                "A frozen split must not be replaced silently. Pass force only if you intend "
                "to discard the existing split, and record why in docs/experiment_log.md."
            )
        atomic_write_json(path, manifest.model_dump(mode="json"))
        return path, "overwritten"

    atomic_write_json(path, manifest.model_dump(mode="json"))
    return path, "written"


def verify_prompt_manifest(manifest: PromptManifest) -> dict[str, Any]:
    """Check a manifest against the task artifacts currently on disk.

    The record's own validator already proved the manifest is internally consistent. This
    checks the other half: that the pool it was drawn from has not changed underneath it.
    """
    task_hash, items_hash, variants_hash = task_manifest_hashes(manifest.task_name)
    mismatches: list[str] = []
    if task_hash != manifest.task_manifest_hash:
        mismatches.append("task_manifest_hash")
    if items_hash != manifest.items_hash:
        mismatches.append("items_hash")
    if variants_hash != manifest.variants_hash:
        mismatches.append("variants_hash")
    return {
        "manifest_id": manifest.manifest_id,
        "valid": not mismatches,
        "mismatches": mismatches,
        "expected": {
            "task_manifest_hash": manifest.task_manifest_hash,
            "items_hash": manifest.items_hash,
            "variants_hash": manifest.variants_hash,
        },
        "observed": {
            "task_manifest_hash": task_hash,
            "items_hash": items_hash,
            "variants_hash": variants_hash,
        },
    }


def resolve_task_config(manifest_config: PromptManifestConfig) -> TaskConfig:
    try:
        return load_config(manifest_config.task_ref, TaskConfig)
    except ConfigError as error:
        raise PromptManifestError(
            f"prompt manifest config {manifest_config.name!r} references a task config that "
            f"does not load: {error}"
        ) from error


def generate_prompt_manifest(
    config_path: str | Path,
    master_seed: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build, validate, and write a manifest. The whole command, minus the printing."""
    try:
        manifest_config = load_config(config_path, PromptManifestConfig)
    except ConfigError as error:
        raise PromptManifestError(str(error)) from error

    task_config = resolve_task_config(manifest_config)
    manifest = build_prompt_manifest(manifest_config, task_config, master_seed=master_seed)
    path, status = write_prompt_manifest(manifest, force=force)
    verification = verify_prompt_manifest(manifest)

    info(
        "froze prompt manifest",
        manifest_id=manifest.manifest_id,
        status=status,
        prompts=len(manifest.assignments),
        manifest_hash=manifest.manifest_hash[:23],
    )

    try:
        reported_path = str(path.relative_to(repo_root())).replace("\\", "/")
    except ValueError:
        reported_path = str(path)

    return {
        "status": status,
        "manifest_path": reported_path,
        "manifest_id": manifest.manifest_id,
        "manifest_hash": manifest.manifest_hash,
        "task_name": manifest.task_name,
        "task_manifest_hash": manifest.task_manifest_hash,
        "items_hash": manifest.items_hash,
        "variants_hash": manifest.variants_hash,
        "master_seed": manifest.master_seed,
        "canonical_wrapper_id": manifest.canonical_wrapper_id,
        "selection_algorithm": manifest.selection_algorithm,
        "selection_algorithm_version": manifest.selection_algorithm_version,
        "eligible_group_count": manifest.eligible_group_count,
        "role_counts": dict(manifest.role_counts),
        "total_prompts": len(manifest.assignments),
        "task_artifacts_match": verification["valid"],
        "notes": (
            "Prompt-role manifest only. No model was loaded, no state was captured, and this "
            "is not a scientific result."
        ),
    }


__all__ = [
    "ROLE_ORDER",
    "SELECTION_ALGORITHM",
    "SELECTION_ALGORITHM_VERSION",
    "PromptManifestError",
    "build_prompt_manifest",
    "check_role_disjointness",
    "eligible_groups",
    "generate_prompt_manifest",
    "load_prompt_manifest",
    "manifest_content_bytes",
    "ordered_groups",
    "selection_key",
    "task_manifest_hashes",
    "verify_prompt_manifest",
    "write_prompt_manifest",
]
