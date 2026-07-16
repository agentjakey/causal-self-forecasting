"""Candidate-set construction and the public view of an intervention.

Two ideas do the work here.

Opaque identity: a candidate's public id carries no meaning. A name such as
`bias_direction_positive` would let a language-model forecaster score well by reading the
label, which is the intervention-label shortcut in the failure-mode taxonomy.

Indistinguishable public description: the public view describes the *form* of the operation
and hides which direction is involved. A matched random control and a real bias-direction
steer are both reported as a residual addition at some layer and strength, because that is
all they visibly are. If the public view said `random_add`, a forecaster could predict "no
effect" for controls without any access to the model's state, and the random-direction
control would stop being a control.

A no-op is published as a residual addition of strength zero, which is exactly what it is.
Any forecaster can tell that adding zero does nothing. That is intended: the no-op validates
the scoring path rather than discriminating between methods, so it is reported separately in
analysis instead of being pooled into headline numbers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ..hashing import hash_object
from ..reproducibility import derive_seed
from ..schemas import CandidateSet, InterventionSpec, Mechanism

# Mechanisms whose public description is identical. Keeping this mapping explicit, rather
# than inferring it, means adding a mechanism forces a decision about what it reveals.
_PUBLIC_OPERATION = {
    Mechanism.NOOP: "residual_add",
    Mechanism.RESIDUAL_ADD: "residual_add",
    Mechanism.RANDOM_ADD: "residual_add",
    Mechanism.DIRECTION_ABLATE: "direction_ablate",
    Mechanism.ACTIVATION_PATCH: "activation_patch",
}


@dataclass(frozen=True)
class CandidateTemplate:
    """A candidate before it is given an opaque id and an order."""

    mechanism: Mechanism
    layer: int
    position_index: int
    strength: float
    direction_id: str | None = None
    source_state_id: str | None = None
    mechanism_version: str = "1.0"
    private_payload_ref: str | None = None
    role: str = "unspecified"


def public_view(spec: InterventionSpec) -> dict[str, Any]:
    """What a forecaster without payload access is allowed to see.

    Everything identifying the direction is omitted. `direction_id` is not included, and
    neither is the mechanism, because `random_add` versus `residual_add` is precisely the
    distinction the forecaster is supposed to have to work out from the model's state.
    """
    return {
        "intervention_id": spec.intervention_id,
        "operation": _PUBLIC_OPERATION[spec.mechanism],
        "layer": spec.layer,
        "position_index": spec.position_index,
        "strength": spec.strength,
    }


def public_candidate_views(candidate_set: CandidateSet) -> list[dict[str, Any]]:
    return [public_view(candidate) for candidate in candidate_set.candidates]


def default_templates(
    direction_id: str,
    random_direction_id: str,
    layer: int,
    position_index: int,
    strength: float,
) -> list[CandidateTemplate]:
    """The standard four-candidate set.

    Positive and negative steering along a validated direction, a norm-matched random
    control, and a no-op. `role` is recorded for analysis and is never published.
    """
    magnitude = abs(strength)
    return [
        CandidateTemplate(
            mechanism=Mechanism.RESIDUAL_ADD,
            layer=layer,
            position_index=position_index,
            strength=magnitude,
            direction_id=direction_id,
            role="direction_positive",
        ),
        CandidateTemplate(
            mechanism=Mechanism.RESIDUAL_ADD,
            layer=layer,
            position_index=position_index,
            strength=-magnitude,
            direction_id=direction_id,
            role="direction_negative",
        ),
        CandidateTemplate(
            mechanism=Mechanism.RANDOM_ADD,
            layer=layer,
            position_index=position_index,
            strength=magnitude,
            direction_id=random_direction_id,
            role="random_control",
        ),
        CandidateTemplate(
            mechanism=Mechanism.NOOP,
            layer=layer,
            position_index=position_index,
            strength=0.0,
            role="noop_control",
        ),
    ]


def build_candidate_set(
    trial_id: str,
    run_seed: int,
    templates: list[CandidateTemplate],
) -> CandidateSet:
    """Shuffle templates into a candidate set with opaque ids.

    The order seed is derived from the run seed and the trial id, so the order is fixed and
    reproducible, but it is not something the forecaster can exploit: ids are assigned after
    the shuffle, so position in the list carries no information about role.
    """
    if len(templates) < 2:
        raise ValueError(f"trial {trial_id}: a candidate set needs at least two candidates")

    order_seed = derive_seed("candidate_order", run_seed, trial_id)
    shuffled = list(templates)
    random.Random(order_seed).shuffle(shuffled)

    candidates: list[InterventionSpec] = []
    for index, template in enumerate(shuffled):
        candidates.append(
            InterventionSpec(
                intervention_id=f"{trial_id}.opaque_{index:02d}",
                mechanism=template.mechanism,
                mechanism_version=template.mechanism_version,
                layer=template.layer,
                position_index=template.position_index,
                strength=template.strength,
                direction_id=template.direction_id,
                source_state_id=template.source_state_id,
                private_payload_ref=template.private_payload_ref,
                analysis_role=template.role,
            )
        )
    return CandidateSet(trial_id=trial_id, candidates=candidates, order_seed=order_seed)


def candidate_set_hash(candidate_set: CandidateSet) -> str:
    return hash_object(candidate_set.model_dump(mode="json"))
