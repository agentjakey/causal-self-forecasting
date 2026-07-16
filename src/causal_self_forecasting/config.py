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
from .schemas import Base, Framing, Mechanism, Split


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
