"""Prompt-only lexical baseline.

Reads the prompt text and the public intervention metadata, nothing else. It fits a TF-IDF
representation of the prompt alongside the public strength and layer, then a ridge regressor
for the delta margin and a logistic classifier for the answer flip.

This baseline exists to answer one question: how much of a forecast can be reproduced from the
visible prompt without any access to the model's internal state. If a state-conditioned method
cannot beat it on same-prompt state-swap trials, the state is not doing the work. So its inputs
are fenced hard: prompt text and public metadata only, never a hidden state, a mechanism, a
direction, a correct answer, or an outcome from the split being predicted.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..schemas import StateCondition
from .base import CandidatePrediction, Forecaster, TrainingExample

_MIN_FLIP_CLASSES = 2


class PromptLexicalBaseline(Forecaster):
    method_id = "prompt_lexical"

    def __init__(self, max_features: int = 512, seed: int = 0) -> None:
        self._max_features = max_features
        self._seed = seed
        self._vectorizer: Any = None
        self._delta_model: Any = None
        self._flip_model: Any = None
        self._flip_constant: bool | None = None
        self._fallback_delta = 0.0
        self._delta_spread = 1.0
        self._fitted = False

    def declared_inputs(self) -> dict[str, bool]:
        return {
            "prompt": True,
            "hidden_state": False,
            "adapter_identity": False,
            "clean_logits": False,
            "intervention_vector": False,
            "public_metadata": True,
            "observed_outcomes": False,
        }

    def state_condition(self) -> StateCondition:
        return StateCondition.NONE

    def _numeric(self, features: dict[str, Any]) -> list[float]:
        # Public metadata only: strength and layer. The mechanism is not here, by design.
        return [float(features["strength"]), float(features["layer"])]

    def fit(self, examples: list[TrainingExample]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression, Ridge

        if not examples:
            raise ValueError("prompt-only baseline needs at least one training example")

        prompts = [example.prompt_text for example in examples]
        numeric = np.array([self._numeric(example.public_features) for example in examples])
        deltas = np.array([example.observed_delta for example in examples], dtype=float)
        flips = np.array([int(example.observed_flip) for example in examples])

        self._vectorizer = TfidfVectorizer(max_features=self._max_features, lowercase=True)
        text_matrix = self._vectorizer.fit_transform(prompts).toarray()
        design = np.hstack([text_matrix, numeric])

        self._delta_model = Ridge(alpha=1.0, random_state=self._seed)
        self._delta_model.fit(design, deltas)
        self._fallback_delta = float(deltas.mean())
        self._delta_spread = float(np.std(deltas)) if len(deltas) > 1 else 1.0

        unique = set(flips.tolist())
        if len(unique) >= _MIN_FLIP_CLASSES:
            # `class_weight="balanced"` because answer flips are usually the minority class and
            # an unweighted fit would collapse toward predicting "no flip" for everything.
            self._flip_model = LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=self._seed
            )
            self._flip_model.fit(design, flips)
            self._flip_constant = None
        else:
            # Only one class in training. A classifier cannot be fit, so record the base rate.
            self._flip_model = None
            self._flip_constant = bool(unique.pop())

        self._fitted = True

    def predict_candidate(
        self, prompt_text: str, intervention_id: str, public_features: dict[str, Any]
    ) -> CandidatePrediction:
        if not self._fitted:
            raise RuntimeError("prompt-only baseline must be fit before predicting")

        text_matrix = self._vectorizer.transform([prompt_text]).toarray()
        numeric = np.array([self._numeric(public_features)])
        design = np.hstack([text_matrix, numeric])

        delta = float(self._delta_model.predict(design)[0])

        if self._flip_model is not None:
            flip_prob = float(self._flip_model.predict_proba(design)[0][1])
        else:
            flip_prob = 1.0 if self._flip_constant else 0.0

        half_width = 1.645 * max(self._delta_spread, 1e-6)
        return CandidatePrediction(
            intervention_id=intervention_id,
            delta_margin_mean=delta,
            delta_margin_q05=delta - half_width,
            delta_margin_q95=delta + half_width,
            p_answer_flip=flip_prob,
        )
