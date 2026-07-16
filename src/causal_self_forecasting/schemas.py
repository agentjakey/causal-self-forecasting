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

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import SCHEMA_VERSION

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
