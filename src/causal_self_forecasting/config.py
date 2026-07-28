"""YAML configuration loading and validation.

Configs are validated into Pydantic models and hashed. The hash of the raw parsed document
goes into the run manifest, so a result can always be traced back to the exact settings that
produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ConfigDict, Field, model_validator

from .hashing import hash_object
from .schemas import Base, Framing, Mechanism, PromptRole, Split, StudyRunRole


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or fails validation."""


class ModelConfig(Base):
    """How to load one set of weights.

    `kind: fixture` builds a tiny random model locally. It exists so the smoke experiment
    runs offline. Any config using it is plumbing, never a result.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    name: str
    kind: Literal["huggingface", "fixture"] = "huggingface"
    model_id: str
    revision: str
    dtype: str = "float32"
    device: str = "auto"
    trust_remote_code: bool = False
    hidden_dim: int | None = None
    num_layers: int | None = None
    fixture_hidden_size: int = Field(default=64, gt=0)
    fixture_num_layers: int = Field(default=4, gt=0)


class WrapperSpec(Base):
    """One framing paraphrase.

    `{question}` and `{choices}` are the only substitutions. Wrappers marked `heldout=True`
    are reserved for the wrapper-paraphrase generalization test and must never appear in a
    training split.
    """

    wrapper_id: str
    framing: Framing
    template: str
    heldout: bool = False

    @model_validator(mode="after")
    def _check_template(self) -> WrapperSpec:
        for placeholder in ("{question}", "{choices}"):
            if placeholder not in self.template:
                raise ValueError(f"wrapper {self.wrapper_id} template is missing {placeholder}")
        return self


class SplitConfig(Base):
    """Split policy. Fixed before any forecaster is trained.

    Splitting is by `group_id`, never by row, so that paraphrases of the same question can
    not straddle a split boundary.
    """

    train_fraction: float = Field(default=0.6, gt=0.0, lt=1.0)
    val_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_fraction: float = Field(default=0.25, gt=0.0, lt=1.0)
    heldout_subjects: list[str] = Field(default_factory=list)
    split_seed: int = 20260715

    @model_validator(mode="after")
    def _check_fractions(self) -> SplitConfig:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must sum to 1.0, got {total}")
        return self


class TaskConfig(Base):
    """Which questions to use and how to render them."""

    name: str
    source: str
    source_config: str | None = None
    source_split: str = "train"
    subjects: list[str] = Field(default_factory=list)
    max_items: int | None = Field(default=None, gt=0)
    answer_labels: list[str] = Field(default=["A", "B", "C", "D"], min_length=4, max_length=4)
    # Prepended to a label when resolving its token id. Most tokenizers encode " A" and "A"
    # as different tokens, and a prompt ending in "Answer:" is followed by the spaced form.
    label_prefix: str = " "
    wrappers: list[WrapperSpec] = Field(min_length=1)
    splits: SplitConfig = Field(default_factory=SplitConfig)

    @model_validator(mode="after")
    def _check_wrappers(self) -> TaskConfig:
        ids = [wrapper.wrapper_id for wrapper in self.wrappers]
        if len(set(ids)) != len(ids):
            raise ValueError("wrapper_id values must be unique")
        framings = {wrapper.framing for wrapper in self.wrappers}
        missing = set(Framing) - framings
        if missing:
            names = sorted(framing.value for framing in missing)
            raise ValueError(
                f"task {self.name} is missing wrappers for framings {names}; "
                "the evaluation and deployment comparison needs all three"
            )
        return self


class PromptManifestConfig(Base):
    """How to freeze a role-labeled prompt set.

    Separate from `ExperimentConfig` because it references no model, no intervention grid, and
    no direction: building a manifest loads no weights and runs no forward pass. Folding it
    into the experiment config would make a config that cannot be satisfied without a model
    stand in front of a step that does not need one.
    """

    name: str
    manifest_id: str
    task_ref: str
    canonical_wrapper_id: str
    master_seed: int
    role_counts: dict[PromptRole, int] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_role_counts(self) -> PromptManifestConfig:
        missing = sorted(role.value for role in PromptRole if role not in self.role_counts)
        if missing:
            raise ValueError(
                f"role_counts must name every prompt role; missing {missing}. A role omitted "
                "here would be silently absent from the frozen split."
            )
        bad = sorted(role.value for role, count in self.role_counts.items() if count <= 0)
        if bad:
            raise ValueError(f"these roles have a non-positive count: {bad}")
        return self

    @property
    def total_prompts(self) -> int:
        return sum(self.role_counts.values())


class DirectionFamilyConfig(Base):
    """How to construct one direction family from a pinned model's output embedding.

    `expected_token_ids` and `expected_hidden_dim` are assertions, not inputs. The token ids are
    always resolved through the tokenizer at build time; these values only decide whether the
    build is allowed to proceed. Writing them here rather than in code means a tokenizer change
    stops the build loudly instead of silently producing directions for different tokens.
    """

    name: str
    family_id: str
    study_id: str
    model_ref: str
    direction_id_prefix: str = Field(min_length=1, max_length=16)
    answer_labels: list[str] = Field(default=["A", "B", "C", "D"], min_length=4, max_length=4)
    label_prefix: str = " "
    expected_token_ids: dict[str, int] = Field(default_factory=dict)
    expected_hidden_dim: int | None = Field(default=None, gt=0)
    master_seed: int
    random_control_count: int = Field(default=4, ge=1, le=32)

    @model_validator(mode="after")
    def _check_expectations(self) -> DirectionFamilyConfig:
        if self.expected_token_ids:
            unknown = sorted(set(self.expected_token_ids) - set(self.answer_labels))
            if unknown:
                raise ValueError(f"expected_token_ids names labels that are not answers: {unknown}")
            missing = sorted(set(self.answer_labels) - set(self.expected_token_ids))
            if missing:
                raise ValueError(
                    f"expected_token_ids must cover every answer label or none; missing {missing}"
                )
            if len(set(self.expected_token_ids.values())) != len(self.expected_token_ids):
                raise ValueError("expected_token_ids assigns the same id to two labels")
        if len(set(self.answer_labels)) != len(self.answer_labels):
            raise ValueError("answer_labels must be unique")
        return self


# The only two layers this arm may calibrate at. A third layer is a researcher degree of
# freedom: trying one after seeing the first two fail is selection on the outcome, so the
# config refuses it rather than leaving it to discipline.
ALLOWED_CALIBRATION_LAYERS = (13, 20)

# The preregistered ratio grid, exactly. Ascending, because the selection rule is "smallest
# passing ratio" and a reordering would change which ratio wins.
FROZEN_NORM_RATIOS = (0.02, 0.05, 0.10, 0.20, 0.40)

STATE_AUDIT_TARGET_NAME = "delta_clean_top_margin"


class CalibrationThresholdConfig(Base):
    """Pass conditions for one norm ratio, as declared in the config."""

    min_large_effect_fraction: float = Field(gt=0.0, le=1.0)
    large_effect_threshold: float = Field(gt=0.0)
    min_median_abs_effect: float = Field(gt=0.0)
    max_p95_abs_effect: float = Field(gt=0.0)


class CalibrationPlanConfig(Base):
    """How to plan calibration for the state-dependence arm.

    References no weights: planning reads the frozen prompt and direction manifests plus the
    model config's pinned identity, and never loads a model. What this validator can check on
    its own it checks here; the cross-checks that need the manifests on disk (the calibration
    prompt count, the direction count, the model revision agreeing with the direction family)
    happen at plan-build time, where the manifests are actually read.
    """

    name: str
    plan_id: str
    study_id: str
    target: str
    model_ref: str
    prompt_manifest_id: str
    direction_family_id: str

    primary_layer: int
    fallback_layer: int
    norm_ratios: list[float] = Field(min_length=1)

    thresholds: CalibrationThresholdConfig
    noop_tolerance: float = Field(gt=0.0)
    percentile_method: str
    median_method: str

    expected_role_counts: dict[PromptRole, int] = Field(min_length=1)
    expected_direction_count: int = Field(gt=0)
    expected_signed_directions: int = Field(gt=0)
    master_seed: int
    selection_algorithm_version: str

    @model_validator(mode="after")
    def _check_plan_config(self) -> CalibrationPlanConfig:
        if self.target != STATE_AUDIT_TARGET_NAME:
            raise ValueError(
                f"target must be {STATE_AUDIT_TARGET_NAME!r} for this arm, got {self.target!r}. "
                "The benchmark's delta_margin is a different quantity and is not calibrated here."
            )

        if tuple(self.norm_ratios) != FROZEN_NORM_RATIOS:
            raise ValueError(
                f"norm_ratios must be exactly {list(FROZEN_NORM_RATIOS)} in that order, got "
                f"{self.norm_ratios}. Widening or reordering the grid after preregistration is "
                "an amendment, not a config change."
            )

        for name, layer in (
            ("primary_layer", self.primary_layer),
            ("fallback_layer", self.fallback_layer),
        ):
            if layer not in ALLOWED_CALIBRATION_LAYERS:
                raise ValueError(
                    f"{name} {layer} is not one of the preregistered layers "
                    f"{list(ALLOWED_CALIBRATION_LAYERS)}; no third layer may be calibrated"
                )
        if self.primary_layer == self.fallback_layer:
            raise ValueError("the fallback layer must differ from the primary layer")
        if self.primary_layer != ALLOWED_CALIBRATION_LAYERS[0]:
            raise ValueError(
                f"the primary layer is preregistered as {ALLOWED_CALIBRATION_LAYERS[0]}, got "
                f"{self.primary_layer}"
            )

        missing = sorted(role.value for role in PromptRole if role not in self.expected_role_counts)
        if missing:
            raise ValueError(f"expected_role_counts must name every prompt role; missing {missing}")
        if any(count <= 0 for count in self.expected_role_counts.values()):
            raise ValueError("every expected role count must be positive")

        if self.expected_signed_directions != 2 * self.expected_direction_count:
            raise ValueError(
                f"expected_signed_directions {self.expected_signed_directions} is not two per "
                f"direction for {self.expected_direction_count} directions"
            )
        return self

    @property
    def calibration_prompt_count(self) -> int:
        return self.expected_role_counts[PromptRole.CALIBRATION]


class StateAuditRunConfig(Base):
    """How to execute one state-dependence run against real weights.

    Deliberately separate from `ExperimentConfig`. That config describes the benchmark's
    four-candidate trial sweep, is read by `csf trials generate`, and carries a `direction_id`
    and an intervention grid this arm does not use. Reusing it would mean loosening validators
    the benchmark depends on.

    The `expected_*` fields are assertions, not inputs. Counts, dimensions, and token ids are
    always read from the frozen manifests and the loaded model; these values only decide whether
    the run is allowed to proceed, so a manifest or a tokenizer that drifted stops the run
    loudly instead of producing observations about something else.
    """

    name: str
    study_id: str
    run_role: StudyRunRole
    prompt_role: PromptRole
    target: str

    model_ref: str
    task_ref: str
    prompt_manifest_id: str
    direction_family_id: str
    calibration_plan_id: str

    layer: int
    capture_position: int = -1
    norm_ratio: float

    expected_prompt_count: int = Field(gt=0)
    expected_direction_count: int = Field(gt=0)
    expected_signed_directions: int = Field(gt=0)
    expected_hidden_dim: int = Field(gt=0)
    expected_role_counts: dict[PromptRole, int] = Field(min_length=1)

    master_seed: int
    noop_tolerance: float = Field(gt=0.0)
    # Reported in the run diagnostics so an engineering run says what it saw. It is not a pass
    # condition here: calibration, not a smoke run, decides which ratio the study uses.
    effect_report_threshold: float = Field(default=0.10, gt=0.0)

    @model_validator(mode="after")
    def _check_run_config(self) -> StateAuditRunConfig:
        if self.target != STATE_AUDIT_TARGET_NAME:
            raise ValueError(
                f"target must be {STATE_AUDIT_TARGET_NAME!r} for this arm, got {self.target!r}"
            )
        if self.layer not in ALLOWED_CALIBRATION_LAYERS:
            raise ValueError(
                f"layer {self.layer} is not one of the preregistered layers "
                f"{list(ALLOWED_CALIBRATION_LAYERS)}; no other layer may be run"
            )
        if float(self.norm_ratio) not in FROZEN_NORM_RATIOS:
            raise ValueError(
                f"norm_ratio {self.norm_ratio} is not on the frozen grid "
                f"{list(FROZEN_NORM_RATIOS)}; a strength off the grid is an amendment, not a "
                "config change"
            )
        if self.expected_signed_directions != 2 * self.expected_direction_count:
            raise ValueError(
                f"expected_signed_directions {self.expected_signed_directions} is not two per "
                f"direction for {self.expected_direction_count} directions"
            )

        missing = sorted(role.value for role in PromptRole if role not in self.expected_role_counts)
        if missing:
            raise ValueError(f"expected_role_counts must name every prompt role; missing {missing}")
        if any(count <= 0 for count in self.expected_role_counts.values()):
            raise ValueError("every expected role count must be positive")

        declared = self.expected_role_counts[self.prompt_role]
        if declared != self.expected_prompt_count:
            raise ValueError(
                f"the {self.prompt_role.value} role holds {declared} prompts but "
                f"expected_prompt_count is {self.expected_prompt_count}"
            )
        return self

    @property
    def candidates_per_prompt(self) -> int:
        """Signed directions plus one no-op. The selected-strength shape."""
        return self.expected_signed_directions + 1

    @property
    def expected_forward_count(self) -> int:
        return self.expected_prompt_count * (1 + self.candidates_per_prompt)


class InterventionConfig(Base):
    """A grid of interventions for one mechanism."""

    name: str
    mechanism: Mechanism
    mechanism_version: str = "1.0"
    layers: list[int] = Field(min_length=1)
    positions: list[int] = Field(default=[-1], min_length=1)
    strengths: list[float] = Field(default=[0.0], min_length=1)
    directions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_layers(self) -> InterventionConfig:
        if any(layer < 0 for layer in self.layers):
            raise ValueError(f"intervention {self.name} has a negative layer index")
        if self.mechanism is Mechanism.NOOP and self.strengths != [0.0]:
            raise ValueError("a noop intervention grid must have strengths [0.0]")
        return self


class ExperimentConfig(Base):
    """One runnable experiment."""

    name: str
    seed: int = 12345
    model_ref: str
    task_ref: str
    intervention_refs: list[str] = Field(min_length=1)
    candidates_per_trial: int = Field(default=4, ge=2, le=8)
    max_trials: int | None = Field(default=None, gt=0)
    capture_layers: list[int] = Field(min_length=1)
    capture_position: int = -1
    # The direction under test. It must already exist in the direction store, which is what
    # forces direction estimation to be a separate, auditable step rather than something a
    # trial-generation run quietly does to itself using its own test data.
    direction_id: str
    # Several matched random controls rather than one. A single random draw can be unlucky,
    # and "the bias direction beat one random vector" is a much weaker statement than "it beat
    # a set of them".
    random_control_count: int = Field(default=4, ge=1, le=32)
    heldout_mechanisms: list[Mechanism] = Field(default=[Mechanism.ACTIVATION_PATCH])
    trial_splits: list[Split] = Field(default=[Split.TRAIN, Split.VAL, Split.TEST], min_length=1)
    rerun_tolerance: float = Field(default=1e-4, gt=0.0)

    @model_validator(mode="after")
    def _check_heldout(self) -> ExperimentConfig:
        if Mechanism.NOOP in self.heldout_mechanisms:
            raise ValueError("noop is a control and cannot be held out")
        return self


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Parse a YAML document into a plain dict."""
    target = Path(path)
    if not target.is_absolute():
        target = repo_root() / target
    if not target.exists():
        raise ConfigError(f"config file not found: {target}")
    try:
        loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"{target}: invalid YAML: {error}") from error
    if loaded is None:
        raise ConfigError(f"{target}: file is empty")
    if not isinstance(loaded, dict):
        raise ConfigError(
            f"{target}: expected a mapping at the top level, got {type(loaded).__name__}"
        )
    return loaded


def load_config[ConfigT: Base](path: str | Path, config_type: type[ConfigT]) -> ConfigT:
    """Load and validate one config file into a typed model."""
    raw = load_yaml(path)
    try:
        return config_type.model_validate(raw)
    except Exception as error:
        raise ConfigError(f"{path}: does not satisfy {config_type.__name__}: {error}") from error


def config_hash(path: str | Path) -> str:
    """Hash the parsed config document.

    Hashing the parsed structure rather than the file bytes means that reformatting or
    recommenting a config does not invalidate a run's provenance, while any change to an
    actual setting does.
    """
    return hash_object(load_yaml(path))


class ResolvedExperiment(Base):
    """An experiment config with its referenced configs loaded."""

    experiment: ExperimentConfig
    model: ModelConfig
    task: TaskConfig
    interventions: list[InterventionConfig]
    experiment_path: str
    experiment_hash: str


def resolve_experiment(path: str | Path) -> ResolvedExperiment:
    """Load an experiment config together with every config it references."""
    experiment = load_config(path, ExperimentConfig)
    return ResolvedExperiment(
        experiment=experiment,
        model=load_config(experiment.model_ref, ModelConfig),
        task=load_config(experiment.task_ref, TaskConfig),
        interventions=[
            load_config(ref, InterventionConfig) for ref in experiment.intervention_refs
        ],
        experiment_path=str(path),
        experiment_hash=config_hash(path),
    )
