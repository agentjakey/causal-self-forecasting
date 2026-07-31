"""Scoring: compare committed forecasts against measured observations.

`metrics` is pure Python and imports eagerly. `run` is not: it reaches `trials.resolve`, which
reaches the model loader, which imports torch. Importing this package therefore used to pull the
whole model stack into any process that only wanted `mae`.

That mattered, because the model-free paths depend on it. The public replay and the figure suite
both import `scoring.metrics`, and both are supposed to be provably free of weights;
`tests/test_bundle_plots.py` asserts exactly that in a fresh interpreter. So `run` is exposed
through PEP 562 lazy attribute access: `from causal_self_forecasting.scoring import score_run`
still works and still returns the same object, but nothing heavy is imported until it is asked for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .metrics import (
    PairScore,
    brier_score,
    grouped_bootstrap_ci,
    log_loss,
    mae,
    rmse,
    sign_accuracy,
    top_effect_accuracy,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .run import ScoringError, score_run

_LAZY = {"ScoringError": ".run", "score_run": ".run"}


def __getattr__(name: str) -> Any:
    """Import `run` only when one of its names is actually used."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "PairScore",
    "ScoringError",
    "brier_score",
    "grouped_bootstrap_ci",
    "log_loss",
    "mae",
    "rmse",
    "score_run",
    "sign_accuracy",
    "top_effect_accuracy",
]
