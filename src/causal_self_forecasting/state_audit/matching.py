"""The wrong-state control: one deterministic matched donor per prompt, plus ten derangements.

The control that decides whether a state-conditioned advantage is *prompt-specific*. Handing the
predictor a different prompt's state, while changing nothing else, separates "some state helps"
from "this prompt's state helps". Only the second supports the claim.

**The donor pool is the final-test set itself.** Drawing donors from the same 32 prompts means the
marginal distribution of states is identical between the true-state and wrong-state conditions, so
a difference cannot be explained by the donor states being unusual. A donor pool from training
prompts would have been easier and would have confounded prompt specificity with distribution
shift.

Matching, applied in this order and fully determined at every step:

1. exclude the prompt itself, always;
2. prefer donors with the same clean preferred answer, if any exist;
3. minimize the absolute difference in clean top margin;
4. break ties on the absolute difference in clean entropy;
5. break remaining ties on lexicographic variant id.

Step 5 exists so the match never depends on dictionary ordering. Steps 2 through 4 make the donor
as similar as possible on everything the predictor can already see, so what remains different is
the state itself.

The ten permutations are unrestricted seeded derangements, reported as a robustness band around
the primary matched control rather than as ten separate tests.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from ..hashing import hash_object
from ..reproducibility import derive_seed
from ..schemas import PromptRole, WrongStateMatch, WrongStatePairingRecord
from .features import PromptContext

PERMUTATION_SEED_LABEL = "bluedot.wrong_state_permutation"
PERMUTATION_COUNT = 10
MATCHING_ALGORITHM_VERSION = "bluedot_nearest_matched_wrong_state_v1.0"

# A derangement of n items exists only for n >= 2, and rejection sampling needs a sane ceiling.
MAX_DERANGEMENT_DRAWS = 10_000


class MatchingError(RuntimeError):
    """Raised when the wrong-state pairing cannot be built as specified."""


def nearest_matched_donor(
    target: PromptContext, pool: Sequence[PromptContext]
) -> tuple[str, bool, float, float, int]:
    """The donor for one prompt, by the five-step rule.

    Returns the donor id, whether the same clean preferred answer was available, the realized
    margin and entropy distances, and how many donors were eligible after step 2.
    """
    others = [context for context in pool if context.variant_id != target.variant_id]
    if not others:
        raise MatchingError(
            f"{target.variant_id} has no possible donor; a wrong-state control needs at least two "
            "prompts"
        )

    same_label = [
        context
        for context in others
        if context.clean_preferred_label == target.clean_preferred_label
    ]
    eligible = same_label if same_label else others
    matched_on_label = bool(same_label)

    def key(candidate: PromptContext) -> tuple[float, float, str]:
        return (
            abs(candidate.clean_top_margin - target.clean_top_margin),
            abs(candidate.clean_entropy - target.clean_entropy),
            candidate.variant_id,
        )

    donor = min(eligible, key=key)
    margin_distance, entropy_distance, _ = key(donor)
    return donor.variant_id, matched_on_label, margin_distance, entropy_distance, len(eligible)


def build_matches(contexts: Sequence[PromptContext]) -> list[WrongStateMatch]:
    """The deterministic nearest match for every prompt."""
    if len(contexts) < 2:
        raise MatchingError("a wrong-state pairing needs at least two prompts")
    ordered = sorted(contexts, key=lambda context: context.variant_id)
    matches: list[WrongStateMatch] = []
    for target in ordered:
        donor_id, same_label, margin, entropy, eligible = nearest_matched_donor(target, ordered)
        matches.append(
            WrongStateMatch(
                variant_id=target.variant_id,
                donor_variant_id=donor_id,
                same_clean_preferred_label=same_label,
                margin_distance=margin,
                entropy_distance=entropy,
                eligible_donor_count=eligible,
            )
        )
    return matches


def build_derangement(variant_ids: Sequence[str], master_seed: int, index: int) -> dict[str, str]:
    """One seeded derangement: a bijection with no prompt matched to its own state.

    Rejection sampling on a seeded shuffle. For 32 items the probability a random permutation is
    a derangement is about 1/e, so this draws roughly three times; the ceiling exists so a
    degenerate input fails loudly rather than spinning.
    """
    ordered = sorted(variant_ids)
    if len(ordered) < 2:
        raise MatchingError("a derangement needs at least two prompts")
    if len(set(ordered)) != len(ordered):
        raise MatchingError("cannot derange a list with duplicates")

    generator = random.Random(derive_seed(PERMUTATION_SEED_LABEL, master_seed, index))
    for _ in range(MAX_DERANGEMENT_DRAWS):
        shuffled = list(ordered)
        generator.shuffle(shuffled)
        if all(a != b for a, b in zip(ordered, shuffled, strict=True)):
            return dict(zip(ordered, shuffled, strict=True))
    raise MatchingError(
        f"could not draw a derangement of {len(ordered)} prompts in {MAX_DERANGEMENT_DRAWS} "
        "attempts"
    )


def build_pairing(
    pairing_id: str,
    study_id: str,
    layer: int,
    contexts: Sequence[PromptContext],
    master_seed: int,
    prompt_manifest_hash: str,
    final_test_run_id: str,
) -> WrongStatePairingRecord:
    """The complete frozen pairing: the matched donors and the ten derangements."""
    matches = build_matches(contexts)
    variant_ids = sorted(context.variant_id for context in contexts)
    permutations = [
        build_derangement(variant_ids, master_seed, index) for index in range(PERMUTATION_COUNT)
    ]

    payload = {
        "algorithm_version": MATCHING_ALGORITHM_VERSION,
        "pairing_id": pairing_id,
        "study_id": study_id,
        "layer": layer,
        "matches": [match.model_dump(mode="json") for match in matches],
        "permutations": permutations,
        "master_seed": master_seed,
        "prompt_manifest_hash": prompt_manifest_hash,
    }

    return WrongStatePairingRecord(
        pairing_id=pairing_id,
        study_id=study_id,
        prompt_role=PromptRole.FINAL_TEST,
        layer=layer,
        matches=matches,
        permutations=permutations,
        permutation_seed_label=PERMUTATION_SEED_LABEL,
        master_seed=master_seed,
        prompt_manifest_hash=prompt_manifest_hash,
        final_test_run_id=final_test_run_id,
        pairing_hash=hash_object(payload),
    )


def donor_for(
    pairing: WrongStatePairingRecord, variant_id: str, condition_index: int, shuffled: bool
) -> str:
    """The donor a given condition supplies for one prompt."""
    if shuffled:
        if not 0 <= condition_index < len(pairing.permutations):
            raise MatchingError(
                f"permutation index {condition_index} is outside the {len(pairing.permutations)} "
                "recorded permutations"
            )
        permutation = pairing.permutations[condition_index]
        if variant_id not in permutation:
            raise MatchingError(f"{variant_id} is not in permutation {condition_index}")
        return permutation[variant_id]

    if condition_index != 0:
        raise MatchingError(
            f"the matched wrong-state control has a single condition index 0, got {condition_index}"
        )
    for match in pairing.matches:
        if match.variant_id == variant_id:
            return match.donor_variant_id
    raise MatchingError(f"{variant_id} has no matched donor in the pairing")


__all__ = [
    "MATCHING_ALGORITHM_VERSION",
    "PERMUTATION_COUNT",
    "PERMUTATION_SEED_LABEL",
    "MatchingError",
    "build_derangement",
    "build_matches",
    "build_pairing",
    "donor_for",
    "nearest_matched_donor",
]
