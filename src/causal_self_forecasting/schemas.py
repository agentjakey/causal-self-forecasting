"""Pydantic record types for every artifact CSF-Bench writes.

Design rules:

* `extra="forbid"` everywhere. A typo in a config or a stale field in an old artifact
  should fail at load time, not silently vanish.
* Every record that reaches disk carries `schema_version`.
* Public and private information are separated at the type level. Anything a forecaster is
  allowed to see lives in `public_metadata`. Anything that would leak the answer or the
  intervention payload lives behind a reference, not inline.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import SCHEMA_VERSION
from .hashing import hash_object
from .state_audit_target import (
    ANSWER_LABELS,
    TARGET_TOLERANCE,
    TargetError,
    verify_state_audit_target,
)

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=_ID_PATTERN.pattern)]
HashString = Annotated[str, Field(pattern=_HASH_PATTERN.pattern)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(BaseModel):
    # `protected_namespaces` is cleared because the domain vocabulary here genuinely uses
    # `model_id` and `model_variant`, and pydantic would otherwise warn on every field.
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        protected_namespaces=(),
    )


class Versioned(Base):
    schema_version: str = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Framing(StrEnum):
    """Wrapper framing applied around a task item."""

    EVALUATION = "evaluation"
    DEPLOYMENT = "deployment"
    NEUTRAL = "neutral"


class ModelVariant(StrEnum):
    """Which set of weights produced a state or an output."""

    CLEAN = "clean"
    ADAPTED = "adapted"


class Mechanism(StrEnum):
    """Intervention mechanism families.

    `ACTIVATION_PATCH` is reserved as the held-out mechanism for the transfer test and must
    not appear in any forecaster's training split.
    """

    NOOP = "noop"
    RESIDUAL_ADD = "residual_add"
    DIRECTION_ABLATE = "direction_ablate"
    RANDOM_ADD = "random_add"
    ACTIVATION_PATCH = "activation_patch"


class Split(StrEnum):
    """Dataset and evaluation splits. Held-out splits are never trained on."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    HELDOUT_SUBJECT = "heldout_subject"
    HELDOUT_WRAPPER = "heldout_wrapper"


class StateCondition(StrEnum):
    """Which hidden state a forecaster is handed at inference time.

    The gap between `TRUE` and the wrong-state conditions is the self-specificity gap, which
    is the centerpiece control of the project.
    """

    TRUE = "true"
    SHUFFLED = "shuffled"
    WRONG_EXAMPLE = "wrong_example"
    WRONG_MODEL = "wrong_model"
    NONE = "none"


class ResultStatus(StrEnum):
    PRELIMINARY = "preliminary"
    REPLICATED = "replicated"


class PromptRole(StrEnum):
    """Which part of the BlueDot state-dependence arm a prompt belongs to.

    Deliberately separate from `Split`. `Split` carries the preregistered train/val/test policy
    of the original CSF-Bench study and is cited by `ScoreRecord` and by
    `docs/preregistration.md` section 7; redefining it to carry a second, unrelated partition
    would silently change what every existing artifact means. A prompt therefore has both: a
    `Split` inherited from the task pipeline, and a `PromptRole` assigned by the study manifest.
    """

    SMOKE = "smoke"
    CALIBRATION = "calibration"
    TRAINING = "training"
    FINAL_TEST = "final_test"


# ---------------------------------------------------------------------------
# Models and tasks
# ---------------------------------------------------------------------------


class ModelSpec(Versioned):
    """Identifies a set of weights precisely enough to load them again.

    `revision` is required and must not be a moving pointer. A run pinned to `main` is not
    reproducible, because the upstream repository can change under it.
    """

    model_id: str
    revision: str
    variant: ModelVariant = ModelVariant.CLEAN
    adapter_path: str | None = None
    adapter_hash: HashString | None = None
    dtype: str = "float32"
    device: str = "cpu"
    trust_remote_code: bool = False

    @model_validator(mode="after")
    def _check_revision_and_adapter(self) -> ModelSpec:
        if self.revision in {"main", "master", "HEAD", ""}:
            raise ValueError(
                f"revision {self.revision!r} is a moving pointer and is not reproducible; "
                "pin an explicit commit sha"
            )
        if self.variant is ModelVariant.ADAPTED and self.adapter_path is None:
            raise ValueError("an adapted ModelSpec requires adapter_path")
        if self.variant is ModelVariant.CLEAN and self.adapter_path is not None:
            raise ValueError("a clean ModelSpec must not carry an adapter_path")
        return self


class TaskItem(Versioned):
    """One four-choice question, before any wrapper is applied."""

    item_id: Identifier
    group_id: Identifier
    source: str
    source_id: str
    subject: str
    question: str
    choices: list[str] = Field(min_length=4, max_length=4)
    answer_index: int = Field(ge=0, le=3)
    split: Split

    @property
    def answer_label(self) -> str:
        return "ABCD"[self.answer_index]

    @model_validator(mode="after")
    def _check_choices(self) -> TaskItem:
        stripped = [choice.strip() for choice in self.choices]
        if any(not choice for choice in stripped):
            raise ValueError(f"item {self.item_id} has an empty choice")
        if len(set(stripped)) != len(stripped):
            raise ValueError(
                f"item {self.item_id} has duplicate choices, so the answer is ambiguous"
            )
        return self


class PromptVariant(Versioned):
    """A task item rendered under one wrapper framing.

    `group_id` is carried through from the item so that the split policy can keep every
    variant of the same underlying question on the same side of a split.
    """

    variant_id: Identifier
    item_id: Identifier
    group_id: Identifier
    wrapper_id: Identifier
    framing: Framing
    prompt_text: str
    answer_labels: list[str] = Field(min_length=4, max_length=4)
    split: Split


# ---------------------------------------------------------------------------
# Prompt manifests
#
# A prompt manifest is a frozen split. It says which prompts play which role in the BlueDot
# state-dependence arm, and it is written once, before any state is captured. Everything about
# it is arranged so that a reader can check it rather than trust it: the selection is a pure
# function of the master seed and the group ids, the assignments are ordered, and the record
# carries a content hash that the validator recomputes on load. An edited manifest does not
# parse.
# ---------------------------------------------------------------------------


# Fields covered by the manifest content hash. Everything outside this tuple is provenance or
# creation metadata: useful to record, but not part of what the manifest *is*. In particular
# `task_manifest_path` is excluded because it is a filesystem location, and a manifest that
# hashed differently after the repository moved would be worse than useless.
PROMPT_MANIFEST_HASHED_FIELDS = (
    "schema_version",
    "manifest_id",
    "task_name",
    "canonical_wrapper_id",
    "master_seed",
    "selection_algorithm",
    "selection_algorithm_version",
    "task_manifest_hash",
    "items_hash",
    "variants_hash",
    "eligible_group_count",
    "role_counts",
    "assignments",
)


def prompt_manifest_payload(dumped: dict[str, Any]) -> dict[str, Any]:
    """The exact object a manifest hash covers, taken from a JSON-mode model dump."""
    missing = [name for name in PROMPT_MANIFEST_HASHED_FIELDS if name not in dumped]
    if missing:
        raise ValueError(f"prompt manifest dump is missing hashed fields: {missing}")
    return {name: dumped[name] for name in PROMPT_MANIFEST_HASHED_FIELDS}


def compute_prompt_manifest_hash(dumped: dict[str, Any]) -> str:
    return hash_object(prompt_manifest_payload(dumped))


class PromptAssignment(Base):
    """One prompt and the role it plays.

    `selection_index` is the prompt's position in the seeded permutation the roles were cut
    from. It duplicates the list position on purpose: reordering the list without renumbering
    changes the manifest hash and fails validation, so the ordering is evidence rather than
    presentation.
    """

    selection_index: int = Field(ge=0)
    variant_id: Identifier
    item_id: Identifier
    group_id: Identifier
    role: PromptRole
    wrapper_id: Identifier
    split: Split
    prompt_hash: HashString


class PromptManifest(Versioned):
    """A frozen, hashed, role-labeled prompt set.

    The validator is the point of the type. It recomputes the content hash, checks that the
    declared role counts match the assignments, checks that the ordering is intact, and checks
    that no group or item appears under two roles. A manifest that fails any of those does not
    load, so a downstream run cannot quietly use a tampered split.
    """

    manifest_id: Identifier
    task_name: str
    canonical_wrapper_id: Identifier
    master_seed: int
    selection_algorithm: str
    selection_algorithm_version: str
    # Computed over the portable fields of the task manifest, excluding its path fields. See
    # `tasks.prompt_manifest.task_manifest_hash` for exactly what goes in.
    task_manifest_hash: HashString
    items_hash: HashString
    variants_hash: HashString
    eligible_group_count: int = Field(gt=0)
    role_counts: dict[str, int]
    assignments: list[PromptAssignment] = Field(min_length=1)
    manifest_hash: HashString
    # Provenance and creation metadata. Deliberately outside the hashed payload.
    task_manifest_path: str
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _check_manifest(self) -> PromptManifest:
        known_roles = {role.value for role in PromptRole}
        unknown = sorted(set(self.role_counts) - known_roles)
        if unknown:
            raise ValueError(f"role_counts names roles that do not exist: {unknown}")
        if any(count <= 0 for count in self.role_counts.values()):
            raise ValueError("every declared role count must be positive")

        observed: dict[str, int] = {}
        for assignment in self.assignments:
            observed[assignment.role.value] = observed.get(assignment.role.value, 0) + 1
        if observed != dict(self.role_counts):
            raise ValueError(
                f"role_counts {dict(sorted(self.role_counts.items()))} does not match the "
                f"assignments {dict(sorted(observed.items()))}"
            )
        if sum(self.role_counts.values()) != len(self.assignments):
            raise ValueError("role counts do not sum to the number of assignments")

        if [a.selection_index for a in self.assignments] != list(range(len(self.assignments))):
            raise ValueError(
                "assignments must be stored in selection order with contiguous "
                "selection_index values starting at 0"
            )

        for field_name, values in (
            ("variant_id", [a.variant_id for a in self.assignments]),
            ("group_id", [a.group_id for a in self.assignments]),
            ("item_id", [a.item_id for a in self.assignments]),
        ):
            if len(set(values)) != len(values):
                duplicates = sorted({v for v in values if values.count(v) > 1})
                raise ValueError(
                    f"a prompt manifest must be disjoint by {field_name}; these appear under "
                    f"more than one assignment: {duplicates[:5]}"
                )

        wrong_wrapper = sorted(
            {a.wrapper_id for a in self.assignments if a.wrapper_id != self.canonical_wrapper_id}
        )
        if wrong_wrapper:
            raise ValueError(
                f"every assignment must use the canonical wrapper "
                f"{self.canonical_wrapper_id!r}; found {wrong_wrapper}"
            )

        if len(self.assignments) > self.eligible_group_count:
            raise ValueError(
                f"{len(self.assignments)} assignments cannot come from an eligible pool of "
                f"{self.eligible_group_count} groups"
            )

        recomputed = compute_prompt_manifest_hash(self.model_dump(mode="json"))
        if recomputed != self.manifest_hash:
            raise ValueError(
                f"manifest_hash {self.manifest_hash} does not match the manifest contents "
                f"({recomputed}); the file has been edited since it was written"
            )
        return self

    def by_role(self, role: PromptRole) -> list[PromptAssignment]:
        return [assignment for assignment in self.assignments if assignment.role is role]

    def group_ids(self, role: PromptRole) -> set[str]:
        return {assignment.group_id for assignment in self.by_role(role)}


# ---------------------------------------------------------------------------
# Direction families
#
# A direction family is a set of intervention stimuli built once from model weights, before
# any prompt is run. The record below is private experimental provenance: it maps opaque
# direction ids to the construction role that produced them, which is exactly the mapping a
# forecaster must never see. It lives in its own file, outside `public_metadata` and outside
# every candidate view, so that the name of the thing it sits in tells the truth about who may
# read it.
#
# Building a direction from the unembedding is construction, not causal validation. Nothing
# here licenses describing a direction as meaningful, load-bearing, or bias-related.
# ---------------------------------------------------------------------------


class DirectionConstructionRole(StrEnum):
    """How a direction was built. Private, and never published to a forecaster."""

    ANSWER_TOKEN_CENTERED = "answer_token_centered"
    RANDOM_ORTHOGONAL_CONTROL = "random_orthogonal_control"


class DirectionTolerances(Base):
    """The frozen numerical thresholds the construction depends on.

    Part of the hashed payload: a family built under different tolerances is a different
    family, even when the arithmetic happens to land in the same place.
    """

    # float64 residual norm below which a Gram-Schmidt candidate is treated as dependent.
    rank_tolerance: float = Field(gt=0.0)
    # float64 residual norm below which a random draw is rejected and redrawn.
    redraw_tolerance: float = Field(gt=0.0)
    # float64 component magnitude that counts as the first significant component for sign
    # canonicalization.
    sign_tolerance: float = Field(gt=0.0)
    # Validation thresholds applied to the saved float32 vectors.
    norm_tolerance: float = Field(gt=0.0)
    orthogonality_tolerance: float = Field(gt=0.0)


class DirectionEntry(Base):
    """One direction in a family.

    `artifact_hash` is deliberately outside the hashed payload. It covers the `.npz` container,
    whose bytes depend on the archive writer, while `vector_hash` covers the float32 values
    themselves. The numbers are what the family is; the container is how they were stored.
    """

    opaque_id: Identifier
    construction_role: DirectionConstructionRole
    role_index: int = Field(ge=0)
    label: str | None = None
    token_id: int | None = None
    dim: int = Field(gt=0)
    vector_hash: HashString
    artifact_hash: HashString
    raw_norm: float | None = None
    norm: float

    @model_validator(mode="after")
    def _check_role_fields(self) -> DirectionEntry:
        if self.construction_role is DirectionConstructionRole.ANSWER_TOKEN_CENTERED:
            if self.label is None or self.token_id is None or self.raw_norm is None:
                raise ValueError(
                    f"answer-token direction {self.opaque_id} must record its label, token id, "
                    "and raw norm"
                )
            if self.raw_norm <= 0.0:
                raise ValueError(f"answer-token direction {self.opaque_id} has a zero raw norm")
        elif self.label is not None or self.token_id is not None or self.raw_norm is not None:
            raise ValueError(
                f"random control {self.opaque_id} must not carry a label, token id, or raw norm"
            )
        return self


class DirectionFamilyDiagnostics(Base):
    """Measurements about a built family. Reported, and outside the hashed payload.

    These are raw float64 quantities. Keeping them out of the content hash means a family
    stays identifiable by what it is rather than by the last bit of a diagnostic, while the
    vectors themselves remain pinned by their content hashes.
    """

    centered_sum_max_abs_residual: float
    answer_span_orthonormality_error: float
    max_answer_pairwise_abs_cosine: float
    max_norm_error: float
    max_answer_to_random_abs_dot: float
    max_random_to_random_abs_dot: float
    random_redraws: int = Field(ge=0)


class DirectionFamilyRecord(Versioned):
    """A built, hashed, verifiable set of intervention directions.

    The validator recomputes the family hash on load, so an edited manifest does not parse.
    """

    family_id: Identifier
    study_id: Identifier
    construction_algorithm_version: str
    basis_algorithm_version: str
    random_algorithm_version: str

    model_id: str
    model_revision: str
    tokenizer_revision: str
    output_embedding_source: str
    tied_embeddings: bool
    config_tie_word_embeddings: bool | None
    hidden_dim: int = Field(gt=0)
    vocab_size: int = Field(gt=0)

    answer_labels: list[str] = Field(min_length=4, max_length=4)
    answer_token_ids: dict[str, int]

    master_seed: int
    seed_derivation_labels: list[str] = Field(min_length=1)
    derived_random_seed: int
    random_generator: str

    answer_span_rank: int = Field(ge=1)
    tolerances: DirectionTolerances
    directions: list[DirectionEntry] = Field(min_length=1)
    diagnostics: DirectionFamilyDiagnostics
    config_hash: HashString
    family_hash: HashString

    # Provenance and creation metadata, outside the hashed payload.
    config_path: str
    environment: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    notes: str | None = None

    @model_validator(mode="after")
    def _check_family(self) -> DirectionFamilyRecord:
        ids = [entry.opaque_id for entry in self.directions]
        if len(set(ids)) != len(ids):
            raise ValueError("a direction family must not repeat an opaque id")
        if ids != sorted(ids):
            raise ValueError(
                "directions must be stored in ascending opaque-id order, so that position "
                "carries no information about construction role"
            )

        if set(self.answer_token_ids) != set(self.answer_labels):
            raise ValueError("answer_token_ids must name exactly the answer labels")
        if len(set(self.answer_token_ids.values())) != len(self.answer_token_ids):
            raise ValueError("two answer labels resolved to the same token id")

        answers = [
            entry
            for entry in self.directions
            if entry.construction_role is DirectionConstructionRole.ANSWER_TOKEN_CENTERED
        ]
        if len(answers) != len(self.answer_labels):
            raise ValueError(
                f"expected one answer-token direction per label, got {len(answers)} for "
                f"{len(self.answer_labels)} labels"
            )
        if sorted(entry.label or "" for entry in answers) != sorted(self.answer_labels):
            raise ValueError("the answer-token directions do not cover every answer label")
        for entry in answers:
            if entry.label is not None and entry.token_id != self.answer_token_ids[entry.label]:
                raise ValueError(
                    f"direction {entry.opaque_id} cites token id {entry.token_id} for label "
                    f"{entry.label}, but the family resolved {self.answer_token_ids[entry.label]}"
                )

        if any(entry.dim != self.hidden_dim for entry in self.directions):
            raise ValueError("every direction must have the family's hidden dimension")

        if self.answer_span_rank > len(answers):
            raise ValueError(
                f"answer_span_rank {self.answer_span_rank} exceeds the number of answer "
                f"directions ({len(answers)})"
            )

        recomputed = compute_direction_family_hash(self.model_dump(mode="json"))
        if recomputed != self.family_hash:
            raise ValueError(
                f"family_hash {self.family_hash} does not match the family contents "
                f"({recomputed}); the file has been edited since it was written"
            )
        return self

    def by_role(self, role: DirectionConstructionRole) -> list[DirectionEntry]:
        return [entry for entry in self.directions if entry.construction_role is role]


# Fields covered by the family content hash. Excluded: `family_hash` itself, the diagnostics,
# the per-direction float measurements, the artifact container hashes, the config path, the
# environment snapshot, and the creation timestamp.
DIRECTION_FAMILY_HASHED_FIELDS = (
    "schema_version",
    "family_id",
    "study_id",
    "construction_algorithm_version",
    "basis_algorithm_version",
    "random_algorithm_version",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "output_embedding_source",
    "tied_embeddings",
    "hidden_dim",
    "vocab_size",
    "answer_labels",
    "answer_token_ids",
    "master_seed",
    "seed_derivation_labels",
    "derived_random_seed",
    "random_generator",
    "answer_span_rank",
    "tolerances",
    "directions",
    "config_hash",
)

# Per-direction fields inside the hashed payload. Identity and content only, no raw floats.
DIRECTION_ENTRY_HASHED_FIELDS = (
    "opaque_id",
    "construction_role",
    "role_index",
    "label",
    "token_id",
    "dim",
    "vector_hash",
)


def direction_family_payload(dumped: dict[str, Any]) -> dict[str, Any]:
    """The exact object a family hash covers, taken from a JSON-mode model dump."""
    missing = [name for name in DIRECTION_FAMILY_HASHED_FIELDS if name not in dumped]
    if missing:
        raise ValueError(f"direction family dump is missing hashed fields: {missing}")
    payload = {name: dumped[name] for name in DIRECTION_FAMILY_HASHED_FIELDS}
    payload["directions"] = [
        {name: entry[name] for name in DIRECTION_ENTRY_HASHED_FIELDS}
        for entry in payload["directions"]
    ]
    return payload


def compute_direction_family_hash(dumped: dict[str, Any]) -> str:
    return hash_object(direction_family_payload(dumped))


# ---------------------------------------------------------------------------
# States and interventions
# ---------------------------------------------------------------------------


class ModelStateRef(Versioned):
    """A pointer to one captured hidden state.

    Activations are large, so they live in a shard on disk and this record references a row
    inside it. The hash covers the shard, which is what makes a state citable.
    """

    state_id: Identifier
    variant_id: Identifier
    model_variant: ModelVariant
    layer: int = Field(ge=0)
    position_index: int
    hidden_dim: int = Field(gt=0)
    dtype: str
    shard_path: str
    row_index: int = Field(ge=0)
    shard_hash: HashString


class InterventionSpec(Versioned):
    """One intervention a forecaster may be asked to predict the effect of.

    `intervention_id` is deliberately opaque. Names such as "bias_direction_positive" would
    hand any language-model forecaster the answer through the label alone, so the public
    identity of a candidate is a meaningless string and the semantics stay in
    `private_payload_ref`.
    """

    intervention_id: Identifier
    mechanism: Mechanism
    mechanism_version: str
    layer: int = Field(ge=0)
    position_index: int
    strength: float
    direction_id: Identifier | None = None
    source_state_id: Identifier | None = None
    private_payload_ref: str | None = None
    # Why the candidate is in the set (`direction_positive`, `random_control`, ...). Needed
    # to analyze results by role, and never shown to a forecaster. It lives in its own field
    # rather than in `public_metadata` so that the name of the field it sits in tells the
    # truth about who may read it.
    analysis_role: str = "unspecified"
    public_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_mechanism_requirements(self) -> InterventionSpec:
        needs_direction = {Mechanism.RESIDUAL_ADD, Mechanism.DIRECTION_ABLATE, Mechanism.RANDOM_ADD}
        if self.mechanism in needs_direction and self.direction_id is None:
            raise ValueError(f"mechanism {self.mechanism} requires direction_id")
        if self.mechanism is Mechanism.ACTIVATION_PATCH and self.source_state_id is None:
            raise ValueError("mechanism activation_patch requires source_state_id")
        if self.mechanism is Mechanism.NOOP and self.strength != 0.0:
            raise ValueError("a noop intervention must have strength 0.0")
        if self.mechanism is Mechanism.DIRECTION_ABLATE and self.strength != 1.0:
            raise ValueError(
                "direction_ablate removes the full aligned component and is defined at "
                "strength 1.0; partial ablation is a different mechanism"
            )
        forbidden = {
            "is_bias_direction",
            "answer",
            "answer_index",
            "observed",
            "delta_margin",
            "role",
            "analysis_role",
            "direction_id",
        }
        leaked = forbidden.intersection(self.public_metadata)
        if leaked:
            raise ValueError(f"public_metadata would leak {sorted(leaked)} to forecasters")
        return self


class CandidateSet(Versioned):
    """The candidates offered for one trial, in the order the forecaster sees them."""

    trial_id: Identifier
    candidates: list[InterventionSpec] = Field(min_length=2)
    order_seed: int
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _check_unique_ids(self) -> CandidateSet:
        ids = [candidate.intervention_id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError(f"trial {self.trial_id} has duplicate intervention ids")
        return self


class TrialRecord(Versioned):
    """Everything fixed about a trial before any forecast exists."""

    trial_id: Identifier
    variant_id: Identifier
    item_id: Identifier
    group_id: Identifier
    model_variant: ModelVariant
    framing: Framing
    split: Split
    state_id: Identifier
    clean_logits: dict[str, float]
    clean_margin: float
    clean_predicted_label: str
    candidate_set_hash: HashString
    created_at: datetime = Field(default_factory=utc_now)


# ---------------------------------------------------------------------------
# Forecasts and commitments
# ---------------------------------------------------------------------------


class ForecastCandidate(Base):
    """A forecast for one candidate intervention.

    Quantiles are the 90 percent central interval. They are checked for ordering here rather
    than at scoring time, so that an incoherent interval can never be committed to.
    """

    intervention_id: Identifier
    delta_margin_mean: float
    delta_margin_q05: float
    delta_margin_q95: float
    p_answer_flip: Probability
    p_bias_suppressed: Probability

    @model_validator(mode="after")
    def _check_interval(self) -> ForecastCandidate:
        if not self.delta_margin_q05 <= self.delta_margin_q95:
            raise ValueError(
                f"candidate {self.intervention_id}: q05 {self.delta_margin_q05} exceeds "
                f"q95 {self.delta_margin_q95}"
            )
        return self


class ForecastRecord(Versioned):
    """A complete forecast over every candidate in one trial.

    This is the object that gets canonicalized and committed. Nothing about the outcome may
    appear in it.
    """

    trial_id: Identifier
    method_id: Identifier
    candidate_forecasts: list[ForecastCandidate] = Field(min_length=1)
    p_hidden_bias_active: Probability
    state_condition: StateCondition = StateCondition.TRUE
    optional_report: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _check_unique_candidates(self) -> ForecastRecord:
        ids = [forecast.intervention_id for forecast in self.candidate_forecasts]
        if len(set(ids)) != len(ids):
            raise ValueError(f"trial {self.trial_id}: duplicate candidate forecasts")
        return self


class ForecastCommitment(Versioned):
    """A tamper-evident commitment to one forecast record.

    The salt is absent by construction. It is written to a separate reveal record only after
    the intervention has been selected and applied.
    """

    trial_id: Identifier
    method_id: Identifier
    commitment_hash: HashString
    forecast_ref: str
    committed_at: datetime = Field(default_factory=utc_now)


class SelectionReveal(Versioned):
    """The reveal half of the protocol.

    `verified` is recorded rather than asserted, so that a failed verification survives in
    the artifact instead of crashing the run and disappearing.
    """

    trial_id: Identifier
    method_id: Identifier
    selection_seed_hash: HashString
    selected_intervention_id: Identifier
    selected_index: int = Field(ge=0)
    salt_hex: str = Field(min_length=32)
    commitment_hash: HashString
    verified: bool
    revealed_at: datetime = Field(default_factory=utc_now)


# ---------------------------------------------------------------------------
# Observations and scores
# ---------------------------------------------------------------------------


class ObservationRecord(Versioned):
    """What actually happened when the selected intervention was applied."""

    trial_id: Identifier
    intervention_id: Identifier
    mechanism: Mechanism
    clean_logits: dict[str, float]
    post_logits: dict[str, float]
    clean_margin: float
    post_margin: float
    delta_margin: float
    clean_predicted_label: str
    post_predicted_label: str
    answer_flip: bool
    clean_entropy: float
    post_entropy: float
    pre_norm: float
    post_norm: float
    bias_suppressed: bool | None = None
    observed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _check_derived_fields(self) -> ObservationRecord:
        expected = self.post_margin - self.clean_margin
        if abs(self.delta_margin - expected) > 1e-6:
            raise ValueError(
                f"trial {self.trial_id}: delta_margin {self.delta_margin} does not match "
                f"post_margin minus clean_margin ({expected})"
            )
        if self.answer_flip != (self.clean_predicted_label != self.post_predicted_label):
            raise ValueError(
                f"trial {self.trial_id}: answer_flip disagrees with the recorded labels"
            )
        return self


class ScoreRecord(Versioned):
    """Per-trial scoring of one method against one observation."""

    trial_id: Identifier
    method_id: Identifier
    intervention_id: Identifier
    state_condition: StateCondition
    split: Split
    mechanism: Mechanism
    group_id: Identifier
    absolute_error: float = Field(ge=0.0)
    squared_error: float = Field(ge=0.0)
    sign_correct: bool
    interval_covered: bool
    interval_width: float = Field(ge=0.0)
    flip_brier: float = Field(ge=0.0, le=1.0)
    flip_log_loss: float = Field(ge=0.0)
    predicted_delta: float
    observed_delta: float
    predicted_flip_probability: Probability
    observed_flip: bool
    top_effect_candidate_correct: bool | None = None


# ---------------------------------------------------------------------------
# The BlueDot state-dependence arm: observations and calibration
#
# These records sit beside the benchmark's own, never on top of them. `ObservationRecord` keeps
# its dataset-correct-answer semantics and its validator untouched; the arm's observations live
# in `StateAuditObservationRecord` with their own target and their own checks, so an artifact on
# disk always says which quantity it holds.
#
# Every record here carries `scientific_result: Literal[False]` where it could plausibly be
# mistaken for a finding. Calibration chooses an intervention strength; it measures nothing
# about the model's abilities and cannot become a result.
# ---------------------------------------------------------------------------


# Terms that would name a direction's construction role or family. A direction reference handed
# to an observation must be opaque, so the same fence that guards `public_metadata` guards it.
_FORBIDDEN_DIRECTION_TERMS = (
    "answer_token",
    "random_orthogonal",
    "random_control",
    "direction_positive",
    "direction_negative",
    "noop_control",
    "construction_role",
    "analysis_role",
    "centered",
    "unembed",
)


class StateAuditObservationRecord(Versioned):
    """One applied intervention in the state-dependence arm, with its target.

    The validator recomputes the clean preferred label, both margins, the delta, and the flip
    from the logits stored alongside them. A record whose stored target disagrees with its own
    logits does not load, so the target can be checked by a third party without rerunning the
    model.
    """

    study_id: Identifier
    run_id: Identifier
    trial_id: Identifier
    candidate_id: Identifier
    group_id: Identifier
    variant_id: Identifier
    prompt_role: PromptRole

    target_name: Literal["delta_clean_top_margin"] = "delta_clean_top_margin"
    clean_preferred_label: str
    clean_logits: dict[str, float]
    intervened_logits: dict[str, float]
    clean_top_margin: float
    intervened_top_margin: float
    delta_clean_top_margin: float
    answer_flip: bool

    is_noop: bool
    # Opaque. Never a construction role or a family label.
    direction_ref: str
    direction_vector_hash: HashString | None = None
    layer: int = Field(ge=0)
    norm_ratio: float = Field(ge=0.0)
    global_alpha: float

    model_id: str
    model_revision: str
    prompt_manifest_hash: HashString
    direction_family_hash: HashString
    config_hash: HashString
    observed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _check_target(self) -> StateAuditObservationRecord:
        lowered = self.direction_ref.lower()
        leaked = [term for term in _FORBIDDEN_DIRECTION_TERMS if term in lowered]
        if leaked:
            raise ValueError(
                f"direction_ref {self.direction_ref!r} names {sorted(leaked)}, which would tell "
                "a reader the direction's construction role; the reference must be opaque"
            )
        if self.clean_preferred_label not in ANSWER_LABELS:
            raise ValueError(
                f"clean_preferred_label {self.clean_preferred_label!r} is not one of "
                f"{list(ANSWER_LABELS)}"
            )

        if self.is_noop:
            if self.global_alpha != 0.0 or self.norm_ratio != 0.0:
                raise ValueError(
                    "a no-op observation must record norm_ratio 0.0 and global_alpha 0.0, got "
                    f"ratio {self.norm_ratio} and alpha {self.global_alpha}"
                )
        elif self.global_alpha == 0.0:
            raise ValueError(
                f"candidate {self.candidate_id} is not a no-op but records a zero global_alpha"
            )

        try:
            problems = verify_state_audit_target(
                self.clean_logits,
                self.intervened_logits,
                self.clean_preferred_label,
                self.clean_top_margin,
                self.intervened_top_margin,
                self.delta_clean_top_margin,
                self.answer_flip,
                tolerance=TARGET_TOLERANCE,
            )
        except TargetError as error:
            raise ValueError(str(error)) from error
        if problems:
            raise ValueError(
                f"observation {self.trial_id}/{self.candidate_id} disagrees with its own "
                f"logits: {problems}"
            )
        return self


class CalibrationThresholds(Base):
    """The frozen pass conditions for one norm ratio.

    Recorded in the plan and covered by its hash, so a run cannot be re-judged against different
    conditions than the ones it was planned under.
    """

    # At least this fraction of non-no-op effects must reach `large_effect_threshold`.
    min_large_effect_fraction: float = Field(gt=0.0, le=1.0)
    large_effect_threshold: float = Field(gt=0.0)
    min_median_abs_effect: float = Field(gt=0.0)
    max_p95_abs_effect: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _check_ordering(self) -> CalibrationThresholds:
        if self.min_median_abs_effect > self.max_p95_abs_effect:
            raise ValueError(
                f"the median floor {self.min_median_abs_effect} exceeds the 95th-percentile "
                f"ceiling {self.max_p95_abs_effect}; no distribution could satisfy both"
            )
        if self.large_effect_threshold > self.max_p95_abs_effect:
            raise ValueError(
                "the large-effect threshold exceeds the 95th-percentile ceiling; no ratio "
                "could pass both conditions"
            )
        return self


class CalibrationForwardCounts(Base):
    """The planned forward-pass arithmetic, encoded rather than described.

    Only calibration sweeps the ratio grid, so it carries `1 + directions * ratios + 1` forwards
    per prompt while every other role carries `1 + directions + 1`. Getting that wrong is what
    produced the superseded estimate in the audit, so the arithmetic is validated here.
    """

    signed_directions: int = Field(gt=0)
    ratio_count: int = Field(gt=0)
    smoke_prompts: int = Field(ge=0)
    calibration_prompts: int = Field(gt=0)
    training_prompts: int = Field(ge=0)
    final_test_prompts: int = Field(ge=0)

    candidates_per_calibration_prompt: int = Field(gt=0)
    candidates_per_other_prompt: int = Field(gt=0)
    forwards_per_calibration_prompt: int = Field(gt=0)
    forwards_per_other_prompt: int = Field(gt=0)

    smoke_forwards: int = Field(ge=0)
    calibration_forwards_per_layer: int = Field(gt=0)
    training_forwards: int = Field(ge=0)
    final_test_forwards: int = Field(ge=0)
    primary_total_forwards: int = Field(gt=0)
    fallback_additional_forwards: int = Field(gt=0)
    with_fallback_total_forwards: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_arithmetic(self) -> CalibrationForwardCounts:
        expected_calibration_candidates = self.signed_directions * self.ratio_count + 1
        expected_other_candidates = self.signed_directions + 1
        checks: list[tuple[str, int, int]] = [
            (
                "candidates_per_calibration_prompt",
                self.candidates_per_calibration_prompt,
                expected_calibration_candidates,
            ),
            (
                "candidates_per_other_prompt",
                self.candidates_per_other_prompt,
                expected_other_candidates,
            ),
            (
                "forwards_per_calibration_prompt",
                self.forwards_per_calibration_prompt,
                1 + expected_calibration_candidates,
            ),
            (
                "forwards_per_other_prompt",
                self.forwards_per_other_prompt,
                1 + expected_other_candidates,
            ),
            (
                "smoke_forwards",
                self.smoke_forwards,
                self.smoke_prompts * (1 + expected_other_candidates),
            ),
            (
                "calibration_forwards_per_layer",
                self.calibration_forwards_per_layer,
                self.calibration_prompts * (1 + expected_calibration_candidates),
            ),
            (
                "training_forwards",
                self.training_forwards,
                self.training_prompts * (1 + expected_other_candidates),
            ),
            (
                "final_test_forwards",
                self.final_test_forwards,
                self.final_test_prompts * (1 + expected_other_candidates),
            ),
            (
                "primary_total_forwards",
                self.primary_total_forwards,
                self.smoke_forwards
                + self.calibration_forwards_per_layer
                + self.training_forwards
                + self.final_test_forwards,
            ),
            (
                "fallback_additional_forwards",
                self.fallback_additional_forwards,
                self.calibration_forwards_per_layer,
            ),
            (
                "with_fallback_total_forwards",
                self.with_fallback_total_forwards,
                self.primary_total_forwards + self.calibration_forwards_per_layer,
            ),
        ]
        wrong = [
            f"{name} is {actual}, expected {expected}"
            for name, actual, expected in checks
            if actual != expected
        ]
        if wrong:
            raise ValueError(f"the planned forward arithmetic does not add up: {wrong}")
        return self


class CalibrationPlanRecord(Versioned):
    """The frozen calibration plan. Written before any calibration runs."""

    plan_id: Identifier
    study_id: Identifier
    target_name: Literal["delta_clean_top_margin"] = "delta_clean_top_margin"

    model_id: str
    model_revision: str
    prompt_manifest_id: Identifier
    prompt_manifest_hash: HashString
    direction_family_id: Identifier
    direction_family_hash: HashString

    calibration_prompt_count: int = Field(gt=0)
    role_counts: dict[str, int]
    direction_count: int = Field(gt=0)

    primary_layer: int = Field(ge=0)
    fallback_layer: int = Field(ge=0)
    norm_ratios: list[float] = Field(min_length=1)

    thresholds: CalibrationThresholds
    noop_tolerance: float = Field(gt=0.0)
    percentile_method: str
    median_method: str
    selection_algorithm_version: str
    master_seed: int

    forward_counts: CalibrationForwardCounts
    config_hash: HashString
    plan_hash: HashString

    scientific_result: Literal[False] = False
    config_path: str
    created_at: datetime = Field(default_factory=utc_now)
    notes: str | None = None

    @model_validator(mode="after")
    def _check_plan(self) -> CalibrationPlanRecord:
        if self.primary_layer == self.fallback_layer:
            raise ValueError("the fallback layer must differ from the primary layer")
        if sorted(self.norm_ratios) != self.norm_ratios:
            raise ValueError(
                "norm_ratios must be stored in ascending order, because the selection rule is "
                "'smallest passing ratio' and reordering would change which ratio wins"
            )
        if len(set(self.norm_ratios)) != len(self.norm_ratios):
            raise ValueError("norm_ratios must not repeat a ratio")
        if any(ratio <= 0.0 for ratio in self.norm_ratios):
            raise ValueError("every norm ratio must be positive")
        if self.forward_counts.ratio_count != len(self.norm_ratios):
            raise ValueError(
                f"the forward arithmetic assumes {self.forward_counts.ratio_count} ratios but "
                f"the plan lists {len(self.norm_ratios)}"
            )
        if self.forward_counts.calibration_prompts != self.calibration_prompt_count:
            raise ValueError("the forward arithmetic and the plan disagree on the prompt count")

        recomputed = compute_calibration_plan_hash(self.model_dump(mode="json"))
        if recomputed != self.plan_hash:
            raise ValueError(
                f"plan_hash {self.plan_hash} does not match the plan contents ({recomputed}); "
                "the file has been edited since it was written"
            )
        return self


class LayerReferenceNormRecord(Versioned):
    """The median clean-state norm for one candidate layer.

    The individual norms travel with the record. They are 32 numbers, and a reference norm no
    one can recompute is a number to be taken on trust.
    """

    layer: int = Field(ge=0)
    prompt_ids: list[str] = Field(min_length=1)
    prompt_identity_hash: HashString
    count: int = Field(gt=0)
    state_norms: dict[str, float]
    reference_norm: float = Field(gt=0.0)
    median_method: str
    prompt_manifest_hash: HashString
    model_id: str
    model_revision: str
    scientific_result: Literal[False] = False
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _check_norms(self) -> LayerReferenceNormRecord:
        if len(set(self.prompt_ids)) != len(self.prompt_ids):
            raise ValueError("prompt_ids must not repeat")
        if self.prompt_ids != sorted(self.prompt_ids):
            raise ValueError("prompt_ids must be stored in sorted order for a stable identity")
        if set(self.state_norms) != set(self.prompt_ids):
            raise ValueError("state_norms must cover exactly the listed prompt ids")
        if self.count != len(self.prompt_ids):
            raise ValueError(f"count {self.count} does not match {len(self.prompt_ids)} prompts")
        for prompt_id, norm in self.state_norms.items():
            if not math.isfinite(norm) or norm <= 0.0:
                raise ValueError(
                    f"the state norm for {prompt_id} is {norm!r}; every norm must be finite and "
                    "strictly positive"
                )
        if hash_object(self.prompt_ids) != self.prompt_identity_hash:
            raise ValueError("prompt_identity_hash does not match the listed prompt ids")
        return self


class CalibrationCriterionResult(Base):
    """One pass condition, with the number it was judged on."""

    name: str
    passed: bool
    observed: float
    threshold: float
    comparison: str


class CalibrationRatioSummary(Versioned):
    """What one (layer, ratio) grid point looked like, and whether it passed."""

    layer: int = Field(ge=0)
    norm_ratio: float = Field(gt=0.0)
    global_alpha: float = Field(gt=0.0)

    expected_non_noop_observations: int = Field(ge=0)
    observed_non_noop_observations: int = Field(ge=0)
    noop_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)

    finite_output_rate: float = Field(ge=0.0, le=1.0)
    max_abs_noop_target: float = Field(ge=0.0)
    fraction_above_effect_threshold: float = Field(ge=0.0, le=1.0)
    median_abs_effect: float = Field(ge=0.0)
    p95_abs_effect: float = Field(ge=0.0)
    flip_count: int = Field(ge=0)

    criteria: list[CalibrationCriterionResult] = Field(min_length=1)
    passed: bool
    supporting_artifact_hashes: dict[str, str] = Field(default_factory=dict)
    scientific_result: Literal[False] = False

    @model_validator(mode="after")
    def _check_pass(self) -> CalibrationRatioSummary:
        expected = all(criterion.passed for criterion in self.criteria)
        if self.passed != expected:
            raise ValueError(
                f"passed is {self.passed} but the criteria say {expected}; a ratio passes only "
                "when every condition passes"
            )
        return self


class CalibrationStatus(StrEnum):
    """The outcome of the layer-fallback state machine."""

    PASSED_PRIMARY = "passed_primary"
    FALLBACK_REQUIRED = "fallback_required"
    PASSED_FALLBACK = "passed_fallback"
    FAILED_ALL_LAYERS = "failed_all_layers"


class CalibrationDecisionRecord(Versioned):
    """The mechanically derived calibration decision."""

    plan_id: Identifier
    study_id: Identifier
    status: CalibrationStatus
    selected_layer: int | None = None
    selected_norm_ratio: float | None = None
    selected_global_alpha: float | None = None

    primary_layer: int = Field(ge=0)
    fallback_layer: int = Field(ge=0)
    ratio_summaries: list[CalibrationRatioSummary] = Field(min_length=1)
    selection_rationale: str
    selection_algorithm_version: str

    plan_hash: HashString
    prompt_manifest_hash: HashString
    direction_family_hash: HashString
    environment: dict[str, Any] = Field(default_factory=dict)
    decision_hash: HashString
    scientific_result: Literal[False] = False
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _check_decision(self) -> CalibrationDecisionRecord:
        selected = (self.selected_layer, self.selected_norm_ratio, self.selected_global_alpha)
        succeeded = self.status in (
            CalibrationStatus.PASSED_PRIMARY,
            CalibrationStatus.PASSED_FALLBACK,
        )
        if succeeded and any(value is None for value in selected):
            raise ValueError(
                f"status {self.status.value} requires a selected layer, ratio, and alpha"
            )
        if not succeeded and any(value is not None for value in selected):
            raise ValueError(
                f"status {self.status.value} must not carry a selection; nothing was selected"
            )
        if (
            self.status is CalibrationStatus.PASSED_PRIMARY
            and self.selected_layer != self.primary_layer
        ):
            raise ValueError("a primary pass must select the primary layer")
        if (
            self.status is CalibrationStatus.PASSED_FALLBACK
            and self.selected_layer != self.fallback_layer
        ):
            raise ValueError("a fallback pass must select the fallback layer")

        recomputed = compute_calibration_decision_hash(self.model_dump(mode="json"))
        if recomputed != self.decision_hash:
            raise ValueError(
                f"decision_hash {self.decision_hash} does not match the decision contents "
                f"({recomputed}); the file has been edited since it was written"
            )
        return self


# Fields covered by the plan hash. Creation metadata and the config path are outside it, so the
# same plan hashes the same after the repository moves.
CALIBRATION_PLAN_HASHED_FIELDS = (
    "schema_version",
    "plan_id",
    "study_id",
    "target_name",
    "model_id",
    "model_revision",
    "prompt_manifest_id",
    "prompt_manifest_hash",
    "direction_family_id",
    "direction_family_hash",
    "calibration_prompt_count",
    "role_counts",
    "direction_count",
    "primary_layer",
    "fallback_layer",
    "norm_ratios",
    "thresholds",
    "noop_tolerance",
    "percentile_method",
    "median_method",
    "selection_algorithm_version",
    "master_seed",
    "forward_counts",
    "config_hash",
)

CALIBRATION_DECISION_HASHED_FIELDS = (
    "schema_version",
    "plan_id",
    "study_id",
    "status",
    "selected_layer",
    "selected_norm_ratio",
    "selected_global_alpha",
    "primary_layer",
    "fallback_layer",
    "ratio_summaries",
    "selection_rationale",
    "selection_algorithm_version",
    "plan_hash",
    "prompt_manifest_hash",
    "direction_family_hash",
)


def calibration_plan_payload(dumped: dict[str, Any]) -> dict[str, Any]:
    missing = [name for name in CALIBRATION_PLAN_HASHED_FIELDS if name not in dumped]
    if missing:
        raise ValueError(f"calibration plan dump is missing hashed fields: {missing}")
    return {name: dumped[name] for name in CALIBRATION_PLAN_HASHED_FIELDS}


def compute_calibration_plan_hash(dumped: dict[str, Any]) -> str:
    return hash_object(calibration_plan_payload(dumped))


def calibration_decision_payload(dumped: dict[str, Any]) -> dict[str, Any]:
    missing = [name for name in CALIBRATION_DECISION_HASHED_FIELDS if name not in dumped]
    if missing:
        raise ValueError(f"calibration decision dump is missing hashed fields: {missing}")
    return {name: dumped[name] for name in CALIBRATION_DECISION_HASHED_FIELDS}


def compute_calibration_decision_hash(dumped: dict[str, Any]) -> str:
    return hash_object(calibration_decision_payload(dumped))


# ---------------------------------------------------------------------------
# Run-level records
# ---------------------------------------------------------------------------


class ArtifactHashRecord(Versioned):
    """Provenance for one file produced by a run."""

    path: str
    hash: HashString
    size_bytes: int = Field(ge=0)
    kind: str
    created_at: datetime = Field(default_factory=utc_now)


class RunManifest(Versioned):
    """The record that makes a run citable."""

    run_id: Identifier
    phase: str
    config_path: str
    config_hash: HashString
    seed: int
    models: list[ModelSpec] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    status: str = "running"
    notes: str | None = None


class MetricValue(Base):
    """A measured number with the context needed to read it honestly.

    `n` and the bootstrap interval are required, not optional. A point estimate published
    without a sample count is not a result.
    """

    value: float
    n: int = Field(gt=0)
    ci_low: float
    ci_high: float
    n_groups: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_interval(self) -> MetricValue:
        if not self.ci_low <= self.ci_high:
            raise ValueError(f"ci_low {self.ci_low} exceeds ci_high {self.ci_high}")
        return self


class PublicDashboardRecord(Versioned):
    """The only shape the dashboard is allowed to read.

    The exporter refuses to build one of these from an unverified run, which is what keeps
    unverified numbers off the site.
    """

    run_id: Identifier
    status: ResultStatus
    exported_at: datetime = Field(default_factory=utc_now)
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    provenance: list[ArtifactHashRecord] = Field(default_factory=list)
    commitments_verified: bool
    notes: str | None = None

    @model_validator(mode="after")
    def _check_verified(self) -> PublicDashboardRecord:
        if not self.commitments_verified:
            raise ValueError(
                "refusing to export a run whose commitments did not verify; "
                "publishing it would misrepresent the protocol"
            )
        return self


# ---------------------------------------------------------------------------
# Systems benchmark
#
# A systems benchmark answers "does this model run here, how fast, and does the capture path
# work on real weights". It is not a CSF-Bench result and never becomes one. The records
# below are kept structurally incapable of claiming otherwise: `scientific_result` is typed
# `Literal[False]`, so a record asserting a scientific finding cannot be constructed at all,
# rather than merely being discouraged.
# ---------------------------------------------------------------------------


class BenchmarkClassification(StrEnum):
    SYSTEMS_BENCHMARK = "systems_benchmark"
    FIXTURE_SYSTEMS_TEST = "fixture_systems_test"


class AccessStatus(StrEnum):
    """How the weights were reached, or why they were not.

    These are kept distinct because the remedies are completely different. Telling a user
    "model load failed" when the real problem is an unaccepted license sends them to debug
    the wrong thing.
    """

    CACHED_LOAD = "cached_load"
    REMOTE_DOWNLOAD = "remote_download"
    NO_AUTHENTICATION = "no_authentication"
    GATED_ACCESS_DENIED = "gated_access_denied"
    REVISION_UNAVAILABLE = "revision_unavailable"
    NETWORK_FAILURE = "network_failure"
    OFFLINE_CACHE_MISS = "offline_cache_miss"
    INSUFFICIENT_DISK_SPACE = "insufficient_disk_space"
    MODEL_LOAD_FAILURE = "model_load_failure"
    FIXTURE_LOCAL = "fixture_local"


def _check_finite_non_negative(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


class BenchmarkModelInfo(Base):
    model_id: str
    revision: str
    tokenizer_revision: str
    device: str
    dtype: str
    num_layers: int = Field(gt=0)
    hidden_dim: int = Field(gt=0)
    access_status: AccessStatus


class BenchmarkTaskInfo(Base):
    dataset: str
    split: Split
    item_count: int = Field(ge=0)
    manifest_hash: HashString | None = None
    label_token_ids: dict[str, int] = Field(default_factory=dict)
    label_prefix: str = ""
    scoring_format: str


class BenchmarkTiming(Base):
    """Wall-clock costs.

    `model_load_seconds` covers the tokenizer as well, because the centralized loader builds
    both together and splitting it would mean duplicating that path just to time it.
    """

    model_load_seconds: float
    tokenization_seconds_total: float
    warmup_runs: int = Field(ge=0)
    timed_runs: int = Field(gt=0)
    forward_seconds_median: float
    forward_seconds_p90: float
    forward_seconds_min: float
    forward_seconds_max: float
    evaluation_seconds_total: float
    capture_overhead_seconds: float | None = None
    representative_prompt_tokens: int = Field(gt=0)
    total_timed_tokens: int = Field(ge=0)
    prefill_tokens_per_second: float | None = None

    @field_validator(
        "model_load_seconds",
        "tokenization_seconds_total",
        "forward_seconds_median",
        "forward_seconds_p90",
        "forward_seconds_min",
        "forward_seconds_max",
        "evaluation_seconds_total",
        "capture_overhead_seconds",
        "prefill_tokens_per_second",
    )
    @classmethod
    def _finite(cls, value: float | None, info) -> float | None:
        return _check_finite_non_negative(value, info.field_name)


class BenchmarkMemory(Base):
    """Process memory, or an explicit null with the reason.

    A number that could not be measured is reported as null. Substituting an estimate would
    make the artifact look complete while being fiction.
    """

    rss_bytes_after_load: int | None = Field(default=None, ge=0)
    rss_bytes_peak: int | None = Field(default=None, ge=0)
    measurement: str


class BenchmarkEvaluation(Base):
    """Clean multiple-choice accuracy.

    Accuracy over a handful of items is a smoke check that scoring is wired up, not a
    capability measurement. The sample count travels with it so it cannot be read as one.
    """

    scored_items: int = Field(ge=0)
    correct_items: int = Field(ge=0)
    accuracy: float | None = None

    @model_validator(mode="after")
    def _check_counts(self) -> BenchmarkEvaluation:
        if self.correct_items > self.scored_items:
            raise ValueError(
                f"correct_items {self.correct_items} exceeds scored_items {self.scored_items}"
            )
        if self.scored_items == 0:
            if self.accuracy is not None:
                raise ValueError("accuracy must be null when no items were scored")
            return self
        expected = self.correct_items / self.scored_items
        if self.accuracy is None or abs(self.accuracy - expected) > 1e-9:
            raise ValueError(
                f"accuracy {self.accuracy} does not match correct_items / scored_items ({expected})"
            )
        return self


class BenchmarkCapture(Base):
    """Evidence that the hook-owned capture path works on these weights.

    `capture_point_verified` is the important field. It is set by patching the layer with a
    known vector and reading it back, which is the check that caught the transformers 5
    hidden-state indexing problem. `hook_fired` alone would not: a hook can fire and still be
    read back from the wrong point.
    """

    layer: int = Field(ge=0)
    hook_fired: bool
    shape: list[int]
    dtype: str
    capture_point_verified: bool
    max_abs_patch_error: float | None = None


class ComputeEstimate(Base):
    """A planning estimate, with its assumptions attached.

    Deliberately not a prediction. It is arithmetic on one measured median forward time, and
    the assumptions are carried in the artifact so a reader can redo it with their own.
    """

    assumptions: dict[str, Any]
    estimated_forward_count: int = Field(ge=0)
    estimated_cpu_seconds: float = Field(ge=0.0)
    estimated_cpu_hours: float = Field(ge=0.0)
    lora_estimated_cpu_hours: float | None = Field(default=None, ge=0.0)
    small_clean_validation_practical_on_cpu: bool
    full_sweep_practical_on_cpu: bool
    lora_training_practical_on_cpu: bool
    gpu_rental_recommended: bool
    notes: list[str] = Field(default_factory=list)


class BenchmarkRecord(Versioned):
    """The aggregate systems-benchmark artifact."""

    run_id: Identifier
    status: str
    classification: BenchmarkClassification
    # Typed as a literal rather than a bool: this makes a benchmark record that claims to be
    # a scientific result unconstructible, instead of merely against policy.
    scientific_result: Literal[False] = False
    fixture_only: bool
    model: BenchmarkModelInfo
    task: BenchmarkTaskInfo
    timing: BenchmarkTiming
    memory: BenchmarkMemory
    evaluation: BenchmarkEvaluation
    capture: BenchmarkCapture
    compute_estimate: ComputeEstimate | None = None
    limitations: list[str] = Field(min_length=1)
    config_hashes: dict[str, str] = Field(default_factory=dict)
    provenance: list[ArtifactHashRecord] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _check_classification(self) -> BenchmarkRecord:
        expected_fixture = self.classification is BenchmarkClassification.FIXTURE_SYSTEMS_TEST
        if self.fixture_only != expected_fixture:
            raise ValueError(
                f"classification {self.classification.value} and fixture_only "
                f"{self.fixture_only} disagree; a fixture run must be labeled as one"
            )
        return self
