"""Forecasting methods and the driver that commits their predictions.

Only the two baselines that need no model organism live here for now: a constant predictor
and a prompt-only lexical model. The state MLP, the linear probe, the verbal reporter, and the
soft-token reporter are deliberately absent until the pieces they depend on exist.
"""

from .base import (
    CandidatePrediction,
    Forecaster,
    LeakageError,
    TrainingExample,
    build_training_examples,
    commit_forecasts,
)
from .constant import ConstantBaseline
from .lexical import PromptLexicalBaseline

__all__ = [
    "CandidatePrediction",
    "ConstantBaseline",
    "Forecaster",
    "LeakageError",
    "PromptLexicalBaseline",
    "TrainingExample",
    "build_training_examples",
    "commit_forecasts",
]
