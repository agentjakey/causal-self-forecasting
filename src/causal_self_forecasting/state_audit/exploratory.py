"""Post hoc descriptive decomposition of the final-test outcomes. Loads no model.

**This analysis was not preregistered.** It was conceived after the preregistered result had been
observed, in order to describe where the variation in the final-test target actually sits. It
computes no test statistic, no interval, and no p-value, and nothing in the paper's conclusions
rests on it. Everything it reads is already published in the replay bundle.

The final test is a balanced 32 x 16 matrix: every prompt received every signed direction exactly
once. For a balanced two-way layout the additive decomposition

    y_pd = mu + a_p + b_d + r_pd

partitions the total sum of squares exactly, with no ambiguity about ordering:

    SS_total = SS_prompt + SS_direction + SS_residual

`a_p` is the prompt mean minus the grand mean, `b_d` is the signed-direction mean minus the grand
mean, and `r_pd` is what is left. The residual term here is prompt-by-direction structure, not
measurement noise: the harness is deterministic and the no-op target is exactly zero, so a repeated
run would reproduce every cell.

Population (biased) definitions are used throughout, because these are descriptions of the 512
outcomes that exist, not estimates of a wider population.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..hashing import atomic_write_json, hash_file, hash_object, read_json, read_jsonl
from ..logging_utils import info

EXPLORATORY_ALGORITHM_VERSION = "bluedot_direction_decomposition_v1.0"

# Reconstruction must be exact to floating-point noise; a balanced design leaves no slack.
RECONSTRUCTION_TOLERANCE = 1e-9

PROVENANCE_NOTE = (
    "Post hoc descriptive analysis, conceived after the preregistered result was observed. Not "
    "preregistered, not inferential, and not part of any hypothesis test. Computed only from "
    "final-test outcomes already published in the replay bundle."
)


class ExploratoryError(RuntimeError):
    """Raised when the decomposition cannot be computed from the published artifacts."""


@dataclass(frozen=True)
class DirectionSummary:
    """One signed direction, described across the 32 final-test prompts."""

    signed_direction_id: str
    direction_ref: str
    sign: int
    observed_mean_effect: float
    intervention_only_prediction: float
    within_direction_sd: float
    prompt_count: int
    construction_role: str


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _population_sd(values: list[float]) -> float:
    mu = _mean(values)
    return (sum((v - mu) ** 2 for v in values) / len(values)) ** 0.5


def decompose_direction_structure(bundle: Path) -> dict[str, Any]:
    """Partition the final-test outcome variation by prompt and signed direction.

    Descriptive only. Reads the bundle's own observations, candidate sets, and pair scores.
    """
    for name in (
        "state_audit_observations.jsonl",
        "state_audit_candidate_sets.jsonl",
        "state_audit_pair_scores.jsonl",
        "state_audit_analysis.json",
    ):
        if not (bundle / name).exists():
            raise ExploratoryError(f"the bundle is missing {name}")

    candidates: dict[str, tuple[str, int, bool]] = {}
    for row in read_jsonl(bundle / "state_audit_candidate_sets.jsonl"):
        for candidate in row["candidates"]:
            candidates[candidate["candidate_id"]] = (
                candidate["direction_ref"],
                int(candidate["sign"]),
                bool(candidate["is_noop"]),
            )

    cells: dict[tuple[str, str], float] = {}
    for row in read_jsonl(bundle / "state_audit_observations.jsonl"):
        direction, sign, is_noop = candidates[row["candidate_id"]]
        if is_noop:
            continue
        key = (row["group_id"], _signed_id(direction, sign))
        if key in cells:
            raise ExploratoryError(f"duplicate cell for {key}; the design should be balanced")
        cells[key] = float(row["delta_clean_top_margin"])

    prompts = sorted({p for p, _ in cells})
    directions = sorted({d for _, d in cells})
    expected = len(prompts) * len(directions)
    if len(cells) != expected:
        raise ExploratoryError(
            f"the design is not balanced: {len(cells)} cells for {len(prompts)} prompts x "
            f"{len(directions)} signed directions"
        )

    values = list(cells.values())
    grand_mean = _mean(values)
    prompt_means = {p: _mean([cells[(p, d)] for d in directions]) for p in prompts}
    direction_means = {d: _mean([cells[(p, d)] for p in prompts]) for d in directions}

    ss_total = sum((v - grand_mean) ** 2 for v in values)
    ss_prompt = len(directions) * sum((prompt_means[p] - grand_mean) ** 2 for p in prompts)
    ss_direction = len(prompts) * sum((direction_means[d] - grand_mean) ** 2 for d in directions)
    ss_residual = sum(
        (cells[(p, d)] - prompt_means[p] - direction_means[d] + grand_mean) ** 2
        for p in prompts
        for d in directions
    )

    reconstruction_error = abs(ss_total - (ss_prompt + ss_direction + ss_residual))
    if reconstruction_error > RECONSTRUCTION_TOLERANCE:
        raise ExploratoryError(
            f"the components do not reconstruct the total sum of squares: off by "
            f"{reconstruction_error:.3e}"
        )

    predictions: dict[str, float] = {}
    for row in read_jsonl(bundle / "state_audit_pair_scores.jsonl"):
        if row["method_condition"] != "intervention_only_ridge:none:0":
            continue
        direction, sign, is_noop = candidates[row["candidate_id"]]
        if is_noop:
            continue
        signed = _signed_id(direction, sign)
        value = float(row["predicted_delta"])
        seen = predictions.setdefault(signed, value)
        if abs(seen - value) > 1e-12:
            raise ExploratoryError(
                f"the intervention-only prediction is not constant within {signed}, so it is not "
                f"a per-direction table after all"
            )

    # Construction roles are joined only here. They never reach a predictor: this is a descriptive
    # summary written after the forecasts were scored, so the blinding it would have broken has
    # already served its purpose.
    family_path = bundle / "bluedot_state_dependence_directions_v1.json"
    roles: dict[str, str] = {}
    if family_path.exists():
        family = read_json(family_path)
        roles = {d["opaque_id"]: d["construction_role"] for d in family["directions"]}

    summaries: list[DirectionSummary] = []
    for signed in directions:
        direction, sign = _split_signed(signed)
        column = [cells[(p, signed)] for p in prompts]
        summaries.append(
            DirectionSummary(
                signed_direction_id=signed,
                direction_ref=direction,
                sign=sign,
                observed_mean_effect=direction_means[signed],
                intervention_only_prediction=predictions[signed],
                within_direction_sd=_population_sd(column),
                prompt_count=len(column),
                construction_role=roles.get(direction, "unknown"),
            )
        )

    by_role: dict[str, list[float]] = {}
    for summary in summaries:
        column = [cells[(p, summary.signed_direction_id)] for p in prompts]
        by_role.setdefault(summary.construction_role, []).extend(column)

    analysis = read_json(bundle / "state_audit_analysis.json")
    payload: dict[str, Any] = {
        "algorithm_version": EXPLORATORY_ALGORITHM_VERSION,
        "aggregation": (
            "balanced two-way additive decomposition over 32 prompts x 16 signed directions, "
            "population sums of squares"
        ),
        "cited_analysis_hash": analysis["analysis_hash"],
        "components": {
            "direction_main_effects": ss_direction,
            "prompt_main_effects": ss_prompt,
            "prompt_by_direction_residual": ss_residual,
            "total": ss_total,
        },
        "direction_count": len(directions),
        "direction_summaries": [
            {
                "construction_role": s.construction_role,
                "direction_ref": s.direction_ref,
                "intervention_only_prediction": s.intervention_only_prediction,
                "observed_mean_effect": s.observed_mean_effect,
                "prompt_count": s.prompt_count,
                "sign": s.sign,
                "signed_direction_id": s.signed_direction_id,
                "within_direction_sd": s.within_direction_sd,
            }
            for s in summaries
        ],
        "grand_mean": grand_mean,
        "inferential": False,
        "note": PROVENANCE_NOTE,
        "observation_count": len(cells),
        "preregistered": False,
        "prompt_count": len(prompts),
        "reconstruction_error": reconstruction_error,
        "role_summaries": {
            role: {
                "mean_effect": _mean(vals),
                "observation_count": len(vals),
                "sd_effect": _population_sd(vals),
            }
            for role, vals in sorted(by_role.items())
        },
        "schema_version": "1.0",
        "shares_of_total": {
            "direction_main_effects": ss_direction / ss_total,
            "prompt_main_effects": ss_prompt / ss_total,
            "prompt_by_direction_residual": ss_residual / ss_total,
        },
        "source_artifact_hashes": {
            name: hash_file(bundle / name)
            for name in sorted(
                (
                    "state_audit_analysis.json",
                    "state_audit_candidate_sets.jsonl",
                    "state_audit_observations.jsonl",
                    "state_audit_pair_scores.jsonl",
                )
            )
        },
        "target_name": analysis["target_name"],
    }
    payload["decomposition_hash"] = hash_object(payload)
    return payload


def _signed_id(direction: str, sign: int) -> str:
    return f"{direction}{'+' if sign > 0 else '-'}"


def _split_signed(signed: str) -> tuple[str, int]:
    return signed[:-1], 1 if signed.endswith("+") else -1


def write_decomposition(bundle: Path, destination: Path) -> Path:
    """Compute the decomposition and write it beside the paper. Loads no model."""
    payload = decompose_direction_structure(bundle)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, payload)
    info(
        "exploratory decomposition written",
        path=str(destination),
        direction_share=round(payload["shares_of_total"]["direction_main_effects"], 4),
    )
    return destination


def load_decomposition(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ExploratoryError(f"no decomposition at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "EXPLORATORY_ALGORITHM_VERSION",
    "PROVENANCE_NOTE",
    "RECONSTRUCTION_TOLERANCE",
    "DirectionSummary",
    "ExploratoryError",
    "decompose_direction_structure",
    "load_decomposition",
    "write_decomposition",
]
