"""Tests for the deterministic prompt-role manifest.

A prompt manifest is a frozen split. The failure that matters is not a crash: it is a manifest
that looks complete while a group leaked across roles, or that quietly changed when the pool
underneath it moved. So most of these tests are about what the builder refuses to do.

Everything here is offline and model-free. The pools are hand-built, so the counts and the
failure modes are exact rather than incidental.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from causal_self_forecasting.config import PromptManifestConfig, TaskConfig, WrapperSpec
from causal_self_forecasting.hashing import atomic_write_json, hash_file, hash_object, write_jsonl
from causal_self_forecasting.schemas import (
    Framing,
    PromptAssignment,
    PromptManifest,
    PromptRole,
    PromptVariant,
    Split,
    TaskItem,
    compute_prompt_manifest_hash,
)
from causal_self_forecasting.tasks import loader as task_loader
from causal_self_forecasting.tasks import prompt_manifest as pm
from causal_self_forecasting.tasks.prompt_manifest import (
    PromptManifestError,
    build_prompt_manifest,
    check_role_disjointness,
    load_prompt_manifest,
    manifest_content_bytes,
    ordered_groups,
    verify_prompt_manifest,
    write_prompt_manifest,
)

TASK_NAME = "pmtask"
PREREGISTERED_COUNTS = {"smoke": 8, "calibration": 32, "training": 96, "final_test": 32}
PREREGISTERED_TOTAL = 168
MASTER_SEED = 20260727

_WRAPPERS = ("neutral_a", "eval_a", "deploy_a")


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------


def _task_config(name: str = TASK_NAME) -> TaskConfig:
    body = "{question}\n{choices}\nAnswer:"
    return TaskConfig(
        name=name,
        source="allenai/ai2_arc",
        source_config="ARC-Challenge",
        wrappers=[
            WrapperSpec(wrapper_id="neutral_a", framing=Framing.NEUTRAL, template=body),
            WrapperSpec(wrapper_id="eval_a", framing=Framing.EVALUATION, template=body),
            WrapperSpec(wrapper_id="deploy_a", framing=Framing.DEPLOYMENT, template=body),
            WrapperSpec(
                wrapper_id="neutral_heldout",
                framing=Framing.NEUTRAL,
                template=body,
                heldout=True,
            ),
        ],
    )


def _manifest_config(
    seed: int = MASTER_SEED,
    manifest_id: str = "pm_v1",
    wrapper: str = "neutral_a",
    counts: dict[str, int] | None = None,
) -> PromptManifestConfig:
    return PromptManifestConfig(
        name="pm_test",
        manifest_id=manifest_id,
        task_ref="configs/tasks/arc_mcq.yaml",
        canonical_wrapper_id=wrapper,
        master_seed=seed,
        role_counts=dict(counts or PREREGISTERED_COUNTS),  # type: ignore[arg-type]
    )


class Pool:
    """A hand-built prepared task on disk, with the directories redirected into tmp_path."""

    def __init__(self, processed: Path, manifests: Path, manifest_out: Path) -> None:
        self.processed = processed
        self.manifests = manifests
        self.manifest_out = manifest_out

    def write(
        self,
        group_count: int,
        *,
        skip_canonical_for: tuple[str, ...] = (),
        duplicate_canonical_for: tuple[str, ...] = (),
        duplicate_group_id: bool = False,
        shared_item_id: tuple[str, str] | None = None,
        item_count_override: int | None = None,
    ) -> None:
        items: list[TaskItem] = []
        variants: list[PromptVariant] = []
        for index in range(group_count):
            group_id = f"g{index:04d}"
            item_id = group_id
            if shared_item_id is not None and group_id == shared_item_id[1]:
                item_id = shared_item_id[0]
            split = (Split.TRAIN, Split.VAL, Split.TEST)[index % 3]
            items.append(
                TaskItem(
                    item_id=item_id,
                    group_id="g0000" if duplicate_group_id and index == 1 else group_id,
                    source="allenai/ai2_arc",
                    source_id=f"src{index}",
                    subject="ARC-Challenge",
                    question=f"question number {index}",
                    choices=["alpha", "beta", "gamma", "delta"],
                    answer_index=index % 4,
                    split=split,
                )
            )
            for wrapper_id in _WRAPPERS:
                if wrapper_id == "neutral_a" and group_id in skip_canonical_for:
                    continue
                copies = (
                    2 if wrapper_id == "neutral_a" and group_id in duplicate_canonical_for else 1
                )
                for copy in range(copies):
                    suffix = "" if copy == 0 else f"_dup{copy}"
                    variants.append(
                        PromptVariant(
                            variant_id=f"{item_id}.{wrapper_id}{suffix}",
                            item_id=item_id,
                            group_id=group_id,
                            wrapper_id=wrapper_id,
                            framing=Framing.NEUTRAL,
                            prompt_text=f"question number {index} [{wrapper_id}]",
                            answer_labels=["A", "B", "C", "D"],
                            split=split,
                        )
                    )

        items_path = write_jsonl(self.processed / TASK_NAME / "items.jsonl", items)
        variants_path = write_jsonl(self.processed / TASK_NAME / "variants.jsonl", variants)
        self.write_task_manifest(
            item_count=item_count_override if item_count_override is not None else len(items),
            variant_count=len(variants),
            group_count=len({item.group_id for item in items}),
            items_hash=hash_file(items_path),
            variants_hash=hash_file(variants_path),
        )

    def write_task_manifest(
        self,
        item_count: int,
        variant_count: int,
        group_count: int,
        items_hash: str,
        variants_hash: str,
    ) -> None:
        payload: dict[str, Any] = {
            "task_name": TASK_NAME,
            "config_path": "configs\\tasks\\pmtask.yaml",
            "config_hash": hash_object({"name": TASK_NAME}),
            "source": "allenai/ai2_arc",
            "source_config": "ARC-Challenge",
            "source_split": "train",
            "item_count": item_count,
            "variant_count": variant_count,
            "group_count": group_count,
            "item_splits": {"train": item_count},
            "variant_splits": {"train": variant_count},
            "wrapper_ids": list(_WRAPPERS),
            "heldout_wrapper_ids": [],
            "files": {
                "items": {"path": "data\\processed\\pmtask\\items.jsonl", "hash": items_hash},
                "variants": {
                    "path": "data\\processed\\pmtask\\variants.jsonl",
                    "hash": variants_hash,
                },
            },
        }
        atomic_write_json(self.manifests / f"{TASK_NAME}.json", payload)


@pytest.fixture
def pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Pool:
    processed = tmp_path / "processed"
    manifests = tmp_path / "manifests"
    manifest_out = tmp_path / "prompt_manifests"
    for directory in (processed, manifests, manifest_out):
        directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(task_loader, "processed_dir", lambda: processed)
    monkeypatch.setattr(task_loader, "manifests_dir", lambda: manifests)
    monkeypatch.setattr(pm, "manifests_dir", lambda: manifests)
    monkeypatch.setattr(pm, "prompt_manifest_path", lambda mid: manifest_out / f"{mid}.json")

    return Pool(processed, manifests, manifest_out)


def _build(pool: Pool, **config_kwargs) -> PromptManifest:
    return build_prompt_manifest(_manifest_config(**config_kwargs), _task_config())


# ---------------------------------------------------------------------------
# Counts, coverage, and disjointness
# ---------------------------------------------------------------------------


def test_role_counts_are_exactly_the_preregistered_split(pool: Pool) -> None:
    pool.write(256)
    manifest = _build(pool)
    assert manifest.role_counts == PREREGISTERED_COUNTS
    assert len(manifest.by_role(PromptRole.SMOKE)) == 8
    assert len(manifest.by_role(PromptRole.CALIBRATION)) == 32
    assert len(manifest.by_role(PromptRole.TRAINING)) == 96
    assert len(manifest.by_role(PromptRole.FINAL_TEST)) == 32


def test_total_is_one_hundred_and_sixty_eight_unique_prompts(pool: Pool) -> None:
    pool.write(256)
    manifest = _build(pool)
    assert len(manifest.assignments) == PREREGISTERED_TOTAL
    assert len({a.variant_id for a in manifest.assignments}) == PREREGISTERED_TOTAL
    assert len({a.group_id for a in manifest.assignments}) == PREREGISTERED_TOTAL


def test_every_assignment_is_the_canonical_wrapper(pool: Pool) -> None:
    pool.write(256)
    manifest = _build(pool)
    assert {a.wrapper_id for a in manifest.assignments} == {"neutral_a"}


def test_exactly_one_canonical_variant_per_selected_group(pool: Pool) -> None:
    pool.write(256)
    manifest = _build(pool)
    per_group: dict[str, int] = {}
    for assignment in manifest.assignments:
        per_group[assignment.group_id] = per_group.get(assignment.group_id, 0) + 1
    assert set(per_group.values()) == {1}


def test_no_group_overlaps_across_roles(pool: Pool) -> None:
    pool.write(256)
    manifest = _build(pool)
    seen: set[str] = set()
    for role in PromptRole:
        groups = manifest.group_ids(role)
        assert not (groups & seen), f"{role.value} reuses a group from an earlier role"
        seen |= groups
    assert len(seen) == PREREGISTERED_TOTAL


def test_no_item_overlaps_across_roles(pool: Pool) -> None:
    pool.write(256)
    manifest = _build(pool)
    by_role: dict[PromptRole, set[str]] = {
        role: {a.item_id for a in manifest.by_role(role)} for role in PromptRole
    }
    roles = list(PromptRole)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            assert not (by_role[left] & by_role[right])


def test_check_role_disjointness_rejects_a_shared_item() -> None:
    """The post-condition is tested directly, not only through a pool that cannot violate it."""
    shared = [
        PromptAssignment(
            selection_index=0,
            variant_id="a.neutral_a",
            item_id="shared",
            group_id="a",
            role=PromptRole.TRAINING,
            wrapper_id="neutral_a",
            split=Split.TRAIN,
            prompt_hash="sha256:" + "0" * 64,
        ),
        PromptAssignment(
            selection_index=1,
            variant_id="b.neutral_a",
            item_id="shared",
            group_id="b",
            role=PromptRole.FINAL_TEST,
            wrapper_id="neutral_a",
            split=Split.TEST,
            prompt_hash="sha256:" + "1" * 64,
        ),
    ]
    with pytest.raises(PromptManifestError, match="item_id"):
        check_role_disjointness(shared)


def test_check_role_disjointness_rejects_a_shared_group() -> None:
    shared = [
        PromptAssignment(
            selection_index=0,
            variant_id="a.neutral_a",
            item_id="a",
            group_id="shared",
            role=PromptRole.TRAINING,
            wrapper_id="neutral_a",
            split=Split.TRAIN,
            prompt_hash="sha256:" + "0" * 64,
        ),
        PromptAssignment(
            selection_index=1,
            variant_id="b.neutral_a",
            item_id="b",
            group_id="shared",
            role=PromptRole.SMOKE,
            wrapper_id="neutral_a",
            split=Split.TEST,
            prompt_hash="sha256:" + "1" * 64,
        ),
    ]
    with pytest.raises(PromptManifestError, match="group_id"):
        check_role_disjointness(shared)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_and_pool_produce_byte_identical_content(pool: Pool) -> None:
    pool.write(256)
    first = _build(pool)
    second = _build(pool)
    assert manifest_content_bytes(first) == manifest_content_bytes(second)
    assert first.manifest_hash == second.manifest_hash


def test_selection_does_not_depend_on_input_order(pool: Pool) -> None:
    """Sorted before the seeded step, so the order rows arrive in cannot matter."""
    groups = [f"g{index:04d}" for index in range(200)]
    forward = ordered_groups(groups, MASTER_SEED)
    backward = ordered_groups(list(reversed(groups)), MASTER_SEED)
    assert forward == backward


def test_a_different_seed_changes_the_assignment(pool: Pool) -> None:
    pool.write(200)
    baseline = {a.group_id: a.role for a in _build(pool).assignments}
    other = {a.group_id: a.role for a in _build(pool, seed=MASTER_SEED + 1).assignments}
    assert baseline != other


def test_the_manifest_hash_changes_with_the_seed(pool: Pool) -> None:
    pool.write(256)
    assert _build(pool).manifest_hash != _build(pool, seed=1).manifest_hash


def test_the_manifest_hash_changes_with_the_canonical_wrapper(pool: Pool) -> None:
    pool.write(256)
    assert _build(pool).manifest_hash != _build(pool, wrapper="eval_a").manifest_hash


def test_the_manifest_hash_changes_with_the_role_counts(pool: Pool) -> None:
    pool.write(256)
    altered = dict(PREREGISTERED_COUNTS) | {"smoke": 9, "training": 95}
    assert _build(pool).manifest_hash != _build(pool, counts=altered).manifest_hash


def test_the_manifest_hash_changes_with_the_pool(pool: Pool) -> None:
    pool.write(256)
    before = _build(pool).manifest_hash
    pool.write(255)
    assert _build(pool).manifest_hash != before


def test_the_manifest_hash_is_the_hash_of_its_own_payload(pool: Pool) -> None:
    pool.write(256)
    manifest = _build(pool)
    assert manifest.manifest_hash == compute_prompt_manifest_hash(manifest.model_dump(mode="json"))


def test_the_manifest_hash_ignores_path_and_timestamp(pool: Pool) -> None:
    """Provenance must not move the hash, or the same split would hash differently per host."""
    pool.write(256)
    manifest = _build(pool)
    dumped = manifest.model_dump(mode="json")
    dumped["task_manifest_path"] = "somewhere/else/pmtask.json"
    dumped["created_at"] = "2020-01-01T00:00:00Z"
    assert compute_prompt_manifest_hash(dumped) == manifest.manifest_hash


# ---------------------------------------------------------------------------
# Selection blindness
# ---------------------------------------------------------------------------


def test_roles_do_not_track_the_gold_answer(pool: Pool) -> None:
    """Selection must not balance or sort on the answer key, so roles must not partition it."""
    pool.write(256)
    manifest = _build(pool)
    # The pool cycles answer_index and split by construction, so a selection that keyed on
    # either would produce a role whose members all share one value.
    by_role_split = {
        role: {a.split for a in manifest.by_role(role)}
        for role in PromptRole
        if manifest.by_role(role)
    }
    for role, splits in by_role_split.items():
        assert len(splits) > 1, f"{role.value} drew a single split; selection is not blind"


def test_the_split_enum_is_unchanged() -> None:
    """PromptRole is additive. Redefining Split would change what every old artifact means."""
    assert [split.value for split in Split] == [
        "train",
        "val",
        "test",
        "heldout_subject",
        "heldout_wrapper",
    ]


def test_prompt_role_and_split_share_no_values() -> None:
    assert not ({role.value for role in PromptRole} & {split.value for split in Split})


def test_existing_split_semantics_are_untouched(pool: Pool) -> None:
    """A manifest records the task pipeline's Split; it never rewrites it."""
    pool.write(256)
    manifest = _build(pool)
    _, variants = task_loader.load_prepared_task(TASK_NAME)
    split_by_variant = {variant.variant_id: variant.split for variant in variants}
    for assignment in manifest.assignments:
        assert assignment.split is split_by_variant[assignment.variant_id]


# ---------------------------------------------------------------------------
# Loud failures
# ---------------------------------------------------------------------------


def test_a_pool_that_is_too_small_raises(pool: Pool) -> None:
    pool.write(167)
    with pytest.raises(PromptManifestError, match="167 eligible groups"):
        _build(pool)


def test_a_pool_of_exactly_the_required_size_is_accepted(pool: Pool) -> None:
    pool.write(168)
    manifest = _build(pool)
    assert len(manifest.assignments) == PREREGISTERED_TOTAL
    assert manifest.eligible_group_count == PREREGISTERED_TOTAL


def test_a_missing_canonical_variant_raises(pool: Pool) -> None:
    pool.write(256, skip_canonical_for=("g0007",))
    with pytest.raises(PromptManifestError, match="no 'neutral_a' variant"):
        _build(pool)


def test_a_duplicated_canonical_variant_raises(pool: Pool) -> None:
    pool.write(256, duplicate_canonical_for=("g0011",))
    with pytest.raises(PromptManifestError, match="more than one 'neutral_a' variant"):
        _build(pool)


def test_duplicate_group_ids_raise(pool: Pool) -> None:
    pool.write(256, duplicate_group_id=True)
    with pytest.raises(PromptManifestError, match="duplicate group ids"):
        _build(pool)


def test_duplicate_item_ids_raise(pool: Pool) -> None:
    pool.write(256, shared_item_id=("g0000", "g0100"))
    with pytest.raises(PromptManifestError, match="duplicate item ids"):
        _build(pool)


def test_an_unsupported_wrapper_raises(pool: Pool) -> None:
    pool.write(256)
    with pytest.raises(PromptManifestError, match="is not defined by task"):
        _build(pool, wrapper="no_such_wrapper")


def test_a_heldout_wrapper_cannot_be_canonical(pool: Pool) -> None:
    pool.write(256)
    with pytest.raises(PromptManifestError, match="held-out paraphrase"):
        _build(pool, wrapper="neutral_heldout")


def test_an_unprepared_task_raises(pool: Pool) -> None:
    with pytest.raises(PromptManifestError, match="no prepared data"):
        _build(pool)


def test_a_missing_task_manifest_raises(pool: Pool) -> None:
    pool.write(256)
    (pool.manifests / f"{TASK_NAME}.json").unlink()
    with pytest.raises(PromptManifestError, match="no task manifest"):
        _build(pool)


def test_role_counts_must_name_every_role() -> None:
    with pytest.raises(ValidationError, match="must name every prompt role"):
        PromptManifestConfig(
            name="partial",
            manifest_id="partial",
            task_ref="configs/tasks/arc_mcq.yaml",
            canonical_wrapper_id="neutral_a",
            master_seed=1,
            role_counts={PromptRole.SMOKE: 8},
        )


# ---------------------------------------------------------------------------
# Task-artifact drift
# ---------------------------------------------------------------------------


def test_a_changed_task_manifest_invalidates_the_manifest(pool: Pool) -> None:
    pool.write(256)
    manifest = _build(pool)
    assert verify_prompt_manifest(manifest)["valid"]

    pool.write(256, item_count_override=255)
    report = verify_prompt_manifest(manifest)
    assert not report["valid"]
    assert "task_manifest_hash" in report["mismatches"]


def test_changed_prepared_data_invalidates_the_manifest(pool: Pool) -> None:
    pool.write(256)
    manifest = _build(pool)
    pool.write(255)
    report = verify_prompt_manifest(manifest)
    assert not report["valid"]
    assert "items_hash" in report["mismatches"]
    assert "variants_hash" in report["mismatches"]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def test_writing_is_atomic_and_leaves_no_partial_file(pool: Pool) -> None:
    pool.write(256)
    manifest = _build(pool)
    path, status = write_prompt_manifest(manifest)
    assert status == "written"
    assert path.exists()
    leftovers = [p.name for p in pool.manifest_out.iterdir() if p.suffix == ".tmp"]
    assert not leftovers, f"atomic write left temporary files behind: {leftovers}"
    assert load_prompt_manifest(manifest.manifest_id).manifest_hash == manifest.manifest_hash


def test_rewriting_an_identical_manifest_leaves_the_file_untouched(pool: Pool) -> None:
    pool.write(256)
    path, _ = write_prompt_manifest(_build(pool))
    original = path.read_bytes()
    _, status = write_prompt_manifest(_build(pool))
    assert status == "unchanged"
    assert path.read_bytes() == original


def test_a_different_manifest_is_refused_at_the_same_path(pool: Pool) -> None:
    pool.write(256)
    write_prompt_manifest(_build(pool))
    with pytest.raises(PromptManifestError, match="already holds a different manifest"):
        write_prompt_manifest(_build(pool, seed=999))


def test_force_replaces_a_different_manifest(pool: Pool) -> None:
    pool.write(256)
    write_prompt_manifest(_build(pool))
    replacement = _build(pool, seed=999)
    path, status = write_prompt_manifest(replacement, force=True)
    assert status == "overwritten"
    assert load_prompt_manifest(replacement.manifest_id).manifest_hash == replacement.manifest_hash
    assert path.exists()


# ---------------------------------------------------------------------------
# Schema round trip and tamper evidence
# ---------------------------------------------------------------------------


def test_manifest_round_trips_through_json(pool: Pool) -> None:
    pool.write(256)
    manifest = _build(pool)
    restored = PromptManifest.model_validate(
        json.loads(json.dumps(manifest.model_dump(mode="json")))
    )
    assert restored.manifest_hash == manifest.manifest_hash
    assert restored.assignments == manifest.assignments


def test_an_edited_assignment_fails_to_load(pool: Pool) -> None:
    pool.write(256)
    dumped = _build(pool).model_dump(mode="json")
    dumped["assignments"][0]["role"] = PromptRole.FINAL_TEST.value
    with pytest.raises(ValidationError):
        PromptManifest.model_validate(dumped)


def test_a_reordered_assignment_list_fails_to_load(pool: Pool) -> None:
    pool.write(256)
    dumped = _build(pool).model_dump(mode="json")
    dumped["assignments"][0], dumped["assignments"][1] = (
        dumped["assignments"][1],
        dumped["assignments"][0],
    )
    with pytest.raises(ValidationError, match="selection order"):
        PromptManifest.model_validate(dumped)


def test_a_tampered_hash_fails_to_load(pool: Pool) -> None:
    pool.write(256)
    dumped = _build(pool).model_dump(mode="json")
    dumped["manifest_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError, match="does not match the manifest contents"):
        PromptManifest.model_validate(dumped)


def test_role_counts_must_match_the_assignments(pool: Pool) -> None:
    pool.write(256)
    dumped = _build(pool).model_dump(mode="json")
    dumped["role_counts"]["smoke"] = 7
    with pytest.raises(ValidationError, match="does not match the assignments"):
        PromptManifest.model_validate(dumped)


def test_an_unknown_field_is_rejected(pool: Pool) -> None:
    pool.write(256)
    dumped = _build(pool).model_dump(mode="json")
    dumped["extra_field"] = 1
    with pytest.raises(ValidationError):
        PromptManifest.model_validate(dumped)
