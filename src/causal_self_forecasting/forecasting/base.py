"""The forecaster interface, training-data assembly, and the commitment driver.

The central design idea is that a baseline cannot leak what it never receives. A
`TrainingExample` and the candidate views passed at prediction time carry only the public
description of an intervention: the operation form, the layer, the position, and the strength.
They never carry a hidden state, an adapter identity, a direction id, a true mechanism, a
correct answer, or a post-intervention outcome for the set being predicted. A baseline built
on these types is leakage-free by construction, and the leakage audit test checks that the
types stay that way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any

from ..config import resolve_experiment
from ..hashing import read_json
from ..logging_utils import info
from ..paths import RUN_MANIFEST, run_dir
from ..schemas import (
    CandidateSet,
    ForecastCandidate,
    ForecastRecord,
    StateCondition,
)
from ..tasks.loader import load_prepared_task
from ..trials.candidates import public_view
from ..trials.commitment import commit_forecast, read_candidate_sets
from ..trials.generate import read_trials
from ..trials.resolve import read_observations

# Public intervention features a forecaster may condition on. Anything not here is either
# private (mechanism, direction) or an outcome (delta, flip) and must not reach a forecaster.
PUBLIC_FEATURE_KEYS = ("operation", "layer", "position_index", "strength")

# Fields a TrainingExample is forbidden from ever carrying. The audit test asserts none of
# these appear on the dataclass.
FORBIDDEN_FIELDS = frozenset(
    {
        "hidden_state",
        "state",
        "state_id",
        "adapter",
        "adapter_path",
        "model_variant",
        "mechanism",
        "direction_id",
        "direction",
        "answer",
        "answer_index",
        "correct_label",
        "analysis_role",
        "post_logits",
    }
)


class LeakageError(RuntimeError):
    """Raised when a forecaster would receive information it is not allowed to see."""


@dataclass(frozen=True)
class TrainingExample:
    """One (prompt, public intervention, observed outcome) example.

    The observed delta and flip are the training targets. They are outcomes from the training
    split, which a forecaster is allowed to learn from. Using outcomes from the split being
    predicted would be leakage, and the driver never assembles such examples.
    """

    prompt_text: str
    public_features: dict[str, Any]
    observed_delta: float
    observed_flip: bool
    group_id: str
    split: str


@dataclass(frozen=True)
class CandidatePrediction:
    """A forecaster's prediction for one candidate."""

    intervention_id: str
    delta_margin_mean: float
    delta_margin_q05: float
    delta_margin_q95: float
    p_answer_flip: float

    def to_forecast_candidate(self) -> ForecastCandidate:
        low = min(self.delta_margin_q05, self.delta_margin_q95)
        high = max(self.delta_margin_q05, self.delta_margin_q95)
        return ForecastCandidate(
            intervention_id=self.intervention_id,
            delta_margin_mean=self.delta_margin_mean,
            delta_margin_q05=low,
            delta_margin_q95=high,
            p_answer_flip=_clamp01(self.p_answer_flip),
            # Baselines that read only the prompt and public metadata cannot estimate whether a
            # hidden bias was suppressed, so they report the non-committal 0.5 rather than a
            # confident guess that would be scored as if it meant something.
            p_bias_suppressed=0.5,
        )


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


class Forecaster(ABC):
    """A method that predicts intervention effects.

    Every method declares what it reads. The declaration is the experiment, not paperwork: a
    prompt-only baseline that quietly read hidden states would invalidate the comparison it
    exists to anchor.
    """

    method_id: str
    produces_language: bool = False

    @abstractmethod
    def declared_inputs(self) -> dict[str, bool]:
        """Report access to each information source, all keys always present."""

    @abstractmethod
    def state_condition(self) -> StateCondition:
        """The state condition this method operates under."""

    @abstractmethod
    def fit(self, examples: list[TrainingExample]) -> None: ...

    @abstractmethod
    def predict_candidate(
        self, prompt_text: str, intervention_id: str, public_features: dict[str, Any]
    ) -> CandidatePrediction:
        """Predict one candidate from the prompt and its public features only.

        `intervention_id` is the opaque public id, passed so the prediction can be labeled. It
        carries no information about the candidate's role, so receiving it leaks nothing.
        """

    def check_no_leakage(self) -> None:
        """Assert the declared inputs are within what this method is allowed to read.

        A prompt-only method must not declare access to hidden states, adapter identity, or
        outcomes. The base class enforces the contract so a subclass cannot silently widen it.
        """
        declared = self.declared_inputs()
        forbidden_keys = (
            "hidden_state",
            "adapter_identity",
            "observed_outcomes",
            "intervention_vector",
        )
        forbidden_declared = [key for key in forbidden_keys if declared.get(key)]
        if forbidden_declared:
            raise LeakageError(
                f"{self.method_id} declares access to {forbidden_declared}, which the baselines "
                "in this module are not permitted to use"
            )


def _validate_example_type() -> None:
    """Fail loudly if a forbidden field is ever added to TrainingExample."""
    names = {field.name for field in fields(TrainingExample)}
    leaked = names & FORBIDDEN_FIELDS
    if leaked:
        raise LeakageError(f"TrainingExample carries forbidden fields: {sorted(leaked)}")


_validate_example_type()


def _public_features(candidate_set: CandidateSet, intervention_id: str) -> dict[str, Any]:
    for candidate in candidate_set.candidates:
        if candidate.intervention_id == intervention_id:
            view = public_view(candidate)
            return {key: view[key] for key in PUBLIC_FEATURE_KEYS}
    raise KeyError(
        f"intervention {intervention_id!r} not in candidate set {candidate_set.trial_id}"
    )


def _prompt_lookup(run_id: str) -> dict[str, str]:
    manifest = read_json(run_dir(run_id) / RUN_MANIFEST)
    resolved = resolve_experiment(manifest["config_path"])
    _, variants = load_prepared_task(resolved.task.name)
    return {variant.variant_id: variant.prompt_text for variant in variants}


def build_training_examples(
    run_id: str,
    splits: tuple[str, ...] = ("train",),
) -> list[TrainingExample]:
    """Assemble leakage-free training examples from a resolved run.

    Only observations whose trial falls in the requested splits are used, so a forecaster
    trained here never sees an outcome from the split it will be scored on. Each example joins
    the prompt text and the public features of the intervention to its observed outcome.
    """
    trials = {trial.trial_id: trial for trial in read_trials(run_id)}
    candidate_sets = {cs.trial_id: cs for cs in read_candidate_sets(run_id)}
    prompts = _prompt_lookup(run_id)
    observations = read_observations(run_id)

    examples: list[TrainingExample] = []
    for observation in observations:
        trial = trials.get(observation.trial_id)
        if trial is None or trial.split.value not in splits:
            continue
        candidate_set = candidate_sets.get(observation.trial_id)
        if candidate_set is None:
            continue
        features = _public_features(candidate_set, observation.intervention_id)
        examples.append(
            TrainingExample(
                prompt_text=prompts.get(trial.variant_id, ""),
                public_features=features,
                observed_delta=observation.delta_margin,
                observed_flip=observation.answer_flip,
                group_id=trial.group_id,
                split=trial.split.value,
            )
        )
    return examples


def commit_forecasts(run_id: str, forecaster: Forecaster) -> dict[str, Any]:
    """Predict every candidate in a run and commit the forecasts.

    The forecaster sees only prompts and public features here, the same contract as training.
    Commitment goes through the existing protocol, so a selection seed must not yet exist.
    """
    forecaster.check_no_leakage()
    trials = read_trials(run_id)
    candidate_sets = {cs.trial_id: cs for cs in read_candidate_sets(run_id)}
    prompts = _prompt_lookup(run_id)
    state_condition = forecaster.state_condition()

    committed = 0
    for trial in trials:
        candidate_set = candidate_sets[trial.trial_id]
        prompt = prompts.get(trial.variant_id, "")
        predictions: list[ForecastCandidate] = []
        for candidate in candidate_set.candidates:
            features = {key: public_view(candidate)[key] for key in PUBLIC_FEATURE_KEYS}
            prediction = forecaster.predict_candidate(prompt, candidate.intervention_id, features)
            if prediction.intervention_id != candidate.intervention_id:
                raise LeakageError(
                    f"{forecaster.method_id} returned a prediction for "
                    f"{prediction.intervention_id} but was asked about {candidate.intervention_id}"
                )
            predictions.append(prediction.to_forecast_candidate())

        forecast = ForecastRecord(
            trial_id=trial.trial_id,
            method_id=forecaster.method_id,
            candidate_forecasts=predictions,
            p_hidden_bias_active=0.5,
            state_condition=state_condition,
        )
        commit_forecast(run_id, forecast)
        committed += 1

    info("committed forecasts", run_id=run_id, method=forecaster.method_id, trials=committed)
    return {"run_id": run_id, "method_id": forecaster.method_id, "committed": committed}
