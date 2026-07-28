"""Deterministic construction of the BlueDot direction family.

Eight unit directions built once from a pinned model's output embedding: four centered
answer-token directions and four seeded Gaussian controls orthogonal to their span and to each
other. Nothing here runs a prompt, captures a state, or applies an intervention. The only thing
read from the model is the unembedding matrix.

**This is construction, not causal validation.** A direction built from the unembedding is a
stimulus with a known recipe. Nothing in this module licenses calling any direction meaningful,
load-bearing, or bias-related, and the artifacts it writes carry `validated: false`.

Design notes that are part of the contract rather than commentary:

*Opaque ids.* A direction's stored id is a hash prefix, and the eight directions are ordered by
that id. Neither the string nor the position tells a reader which family a direction came from.
The mapping to construction roles lives only in the private family manifest.

*Float64 construction, float32 storage.* Centering, normalizing, orthogonalizing, and the sign
convention all run in float64; the saved artifact is float32, matching the rest of the direction
store. Every reported tolerance says which of the two it applies to.

*Two orthogonalization passes.* One pass of Gram-Schmidt loses orthogonality when the inputs are
nearly dependent, and the four centered answer directions are exactly dependent by construction:
their raw forms sum to zero, so they span at most three dimensions. The second pass is what keeps
the residual at round-off rather than at the size of the dependence.

*No selection on behavior.* Directions are accepted or redrawn on numerical grounds only. No
prompt, hidden state, logit, or outcome participates.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import (
    ConfigError,
    DirectionFamilyConfig,
    ModelConfig,
    config_hash,
    load_config,
    repo_root,
)
from ..hashing import (
    atomic_write_json,
    canonical_json_bytes,
    hash_file,
    hash_object,
    read_json,
    sha256_hex,
)
from ..logging_utils import info
from ..models.loader import LoadedModel, ModelLoadError, load_model
from ..models.scoring import LabelTokenError, resolve_label_token_ids
from ..paths import direction_manifest_path, directions_dir
from ..reproducibility import derive_seed, package_versions
from ..schemas import (
    DirectionConstructionRole,
    DirectionEntry,
    DirectionFamilyDiagnostics,
    DirectionFamilyRecord,
    DirectionTolerances,
    compute_direction_family_hash,
    direction_family_payload,
)
from .directions import DirectionStore

CONSTRUCTION_ALGORITHM_VERSION = "answer_token_centered_plus_orthogonal_random_v1.0"
BASIS_ALGORITHM_VERSION = "modified_gram_schmidt_two_pass_v1.0"
RANDOM_ALGORITHM_VERSION = "pcg64_project_reject_sign_canonical_v1.0"
RANDOM_GENERATOR = "numpy.random.Generator(PCG64)"

SEED_LABEL = "bluedot.direction_family.random"
OPAQUE_ID_LABEL = "bluedot.direction_family.opaque_id"

# Frozen tolerances. Part of the hashed payload, so changing one produces a different family
# rather than silently different vectors under the same name.
#
# rank_tolerance   float64 residual norm below which a Gram-Schmidt candidate is dependent.
#                  Inputs are unit vectors, so this is effectively relative.
# redraw_tolerance float64 residual norm below which a random draw is rejected. A Gaussian of
#                  dimension 1152 has norm near 34 and loses at most seven dimensions here, so
#                  this never fires in practice; it exists so a degenerate draw can never be
#                  normalized into a unit vector made of rounding noise.
# sign_tolerance   float64 component magnitude that counts as the first significant component.
# norm_tolerance,
# orthogonality_tolerance
#                  validation thresholds applied to the saved float32 vectors, where a dot
#                  product over 1152 terms accumulates roughly 1e-7 per term of rounding.
TOLERANCES = DirectionTolerances(
    rank_tolerance=1e-8,
    redraw_tolerance=1e-6,
    sign_tolerance=1e-12,
    norm_tolerance=1e-5,
    orthogonality_tolerance=1e-5,
)

MAX_RANDOM_REDRAWS = 64

# One direction as it stands just before it is stored:
# (opaque_id, construction role, role index, answer label, token id, raw norm, float32 vector).
type PlannedDirection = tuple[
    str, "DirectionConstructionRole", int, str | None, int | None, float | None, np.ndarray
]


class DirectionFamilyError(RuntimeError):
    """Raised when a direction family cannot be built, stored, or trusted."""


# ---------------------------------------------------------------------------
# Numerical core
# ---------------------------------------------------------------------------


def vector_content_hash(vector: np.ndarray) -> str:
    """Hash the float32 values, independent of how they are stored.

    Little-endian float32, C-contiguous. The `.npz` container hash covers the archive; this
    covers the numbers, which is what a family is actually made of.
    """
    return sha256_hex(np.ascontiguousarray(vector, dtype="<f4").tobytes())


def centered_answer_directions(
    rows: dict[str, np.ndarray],
    labels: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    """Build `d_c = normalize(w_c - mean(w_j for j != c))` for every label.

    Returns the raw directions, the unit directions, and the raw norms. Signs are left exactly
    as the construction produced them: a direction toward answer `c` has a meaningful
    orientation, unlike a random control.
    """
    raw: dict[str, np.ndarray] = {}
    unit: dict[str, np.ndarray] = {}
    norms: dict[str, float] = {}

    for label in labels:
        others = [rows[other] for other in labels if other != label]
        if not others:
            raise DirectionFamilyError("centering needs at least two answer labels")
        centered = rows[label] - np.mean(np.stack(others, axis=0), axis=0)
        norm = float(np.linalg.norm(centered))
        if not math.isfinite(norm) or norm <= 0.0:
            raise DirectionFamilyError(
                f"the centered direction for label {label!r} has norm {norm!r}; the answer-token "
                "rows are degenerate and no direction can be built from them"
            )
        raw[label] = centered
        unit[label] = centered / norm
        norms[label] = norm

    return raw, unit, norms


def modified_gram_schmidt(
    vectors: Sequence[np.ndarray],
    rank_tolerance: float,
    passes: int = 2,
) -> tuple[list[np.ndarray], float]:
    """Orthonormalize in the given order, deterministically.

    Modified Gram-Schmidt rather than QR or SVD: those pick signs and, for a rank-deficient
    input, bases that can differ between library versions without any canonicalization step.
    The answer family is rank-deficient by construction, so that difference would be a real
    reproducibility hazard rather than a theoretical one.

    Returns the basis and the worst off-diagonal absolute inner product within it.
    """
    basis: list[np.ndarray] = []
    for candidate in vectors:
        residual = np.array(candidate, dtype=np.float64, copy=True)
        for _ in range(passes):
            for accepted in basis:
                residual = residual - float(np.dot(accepted, residual)) * accepted
        norm = float(np.linalg.norm(residual))
        if norm < rank_tolerance:
            # Dependent on what is already in the basis. Skipping is the whole point of
            # recording an effective rank below the candidate count.
            continue
        basis.append(residual / norm)

    error = 0.0
    for index, left in enumerate(basis):
        for right in basis[index + 1 :]:
            error = max(error, abs(float(np.dot(left, right))))
    return basis, error


def canonicalize_sign(vector: np.ndarray, sign_tolerance: float) -> np.ndarray:
    """Fix the sign so that the first significant component is positive.

    A random control's sign is arbitrary, so without this the same seed could produce either
    orientation depending on an unrelated implementation detail. Answer directions are not
    passed through here: their orientation is determined by the construction.
    """
    significant = np.flatnonzero(np.abs(vector) > sign_tolerance)
    if significant.size == 0:
        raise DirectionFamilyError(
            "cannot canonicalize the sign of a vector whose every component is below the sign "
            "tolerance"
        )
    return -vector if vector[int(significant[0])] < 0.0 else vector


def orthogonal_random_controls(
    dim: int,
    count: int,
    answer_basis: Sequence[np.ndarray],
    tolerances: DirectionTolerances,
    draw: Callable[[], np.ndarray],
    max_redraws: int = MAX_RANDOM_REDRAWS,
) -> tuple[list[np.ndarray], int]:
    """Draw unit controls orthogonal to the answer span and to each other.

    `draw` supplies the raw Gaussian vectors, so the generator stays outside the numerics and a
    degenerate draw can be exercised in a test without waiting for one to occur by chance.
    """
    accepted: list[np.ndarray] = []
    redraws = 0

    while len(accepted) < count:
        candidate = np.array(draw(), dtype=np.float64, copy=True)
        if candidate.shape != (dim,):
            raise DirectionFamilyError(
                f"a random draw has shape {candidate.shape}, expected {(dim,)}"
            )
        if not np.isfinite(candidate).all():
            raise DirectionFamilyError("a random draw contains non-finite values")

        residual = candidate
        for _ in range(2):
            for reference in (*answer_basis, *accepted):
                residual = residual - float(np.dot(reference, residual)) * reference

        norm = float(np.linalg.norm(residual))
        if norm < tolerances.redraw_tolerance:
            redraws += 1
            if redraws > max_redraws:
                raise DirectionFamilyError(
                    f"redrew {redraws} times without finding a control outside the span of the "
                    f"answer directions and the {len(accepted)} accepted controls; the ambient "
                    f"dimension {dim} may be too small for {count} controls"
                )
            continue

        accepted.append(canonicalize_sign(residual / norm, tolerances.sign_tolerance))

    return accepted, redraws


def random_family_seed(
    master_seed: int,
    study_id: str,
    family_id: str,
    model_id: str,
    revision: str,
) -> tuple[int, list[str]]:
    """Derive the control seed, returning it with the labels it was derived from.

    The labels are recorded so a reader can recompute the seed rather than take it on trust.
    Binding the model id and revision in means that pointing the same config at different
    weights cannot silently reuse the same controls.
    """
    labels = [
        SEED_LABEL,
        str(master_seed),
        study_id,
        family_id,
        model_id,
        revision,
        RANDOM_ALGORITHM_VERSION,
    ]
    return derive_seed(*labels), labels


def opaque_direction_id(
    prefix: str,
    family_id: str,
    model_revision: str,
    master_seed: int,
    role: DirectionConstructionRole,
    role_index: int,
) -> str:
    """A stable, meaningless id for one direction.

    Deterministic, so a rebuild reuses the same stored artifact, and opaque, so the id cannot
    be read as `answer_A` or `random_01` by anything downstream.
    """
    digest = hash_object(
        {
            "label": OPAQUE_ID_LABEL,
            "family_id": family_id,
            "model_revision": model_revision,
            "master_seed": master_seed,
            "construction_algorithm_version": CONSTRUCTION_ALGORITHM_VERSION,
            "role": role.value,
            "role_index": role_index,
        }
    )
    return f"{prefix}.{digest.split(':', 1)[1][:16]}"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def resolve_answer_rows(
    model: LoadedModel,
    family_config: DirectionFamilyConfig,
) -> tuple[dict[str, int], dict[str, np.ndarray], Any]:
    """Resolve the answer token ids and read their unembedding rows.

    Ids come from the existing scoring path, never from the config. The config's expectations
    are checked against what the tokenizer actually produced, and a disagreement stops the
    build rather than being reconciled.
    """
    try:
        token_ids = resolve_label_token_ids(
            model.tokenizer, list(family_config.answer_labels), prefix=family_config.label_prefix
        )
    except LabelTokenError as error:
        raise DirectionFamilyError(f"answer labels are not scoreable: {error}") from error

    expected = family_config.expected_token_ids
    if expected:
        mismatched = {
            label: (expected[label], token_ids[label])
            for label in family_config.answer_labels
            if expected[label] != token_ids[label]
        }
        if mismatched:
            raise DirectionFamilyError(
                "the tokenizer resolved answer token ids that differ from the pinned "
                f"expectations {mismatched} (label: expected, resolved). Refusing to build: the "
                "directions would point at different tokens than the study preregistered."
            )

    expected_dim = family_config.expected_hidden_dim
    if expected_dim is not None and model.hidden_dim != expected_dim:
        raise DirectionFamilyError(
            f"the loaded model has hidden dimension {model.hidden_dim} but the config "
            f"expects {expected_dim}"
        )

    try:
        embedding = model.output_embedding(
            required_token_ids=[token_ids[label] for label in family_config.answer_labels]
        )
    except ModelLoadError as error:
        raise DirectionFamilyError(str(error)) from error

    rows = {
        label: embedding.weight[token_ids[label]]
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy()
        .copy()
        for label in family_config.answer_labels
    }
    return token_ids, rows, embedding


def build_direction_family(
    family_config: DirectionFamilyConfig,
    model: LoadedModel,
    store: DirectionStore,
    config_path: str | Path,
    save: bool = True,
) -> DirectionFamilyRecord:
    """Construct, validate, store, and describe one direction family.

    Every vector is built and validated in memory before anything is written, so a failure in
    the numerics cannot leave a partially stored family behind. The manifest is written last
    and separately by `write_direction_family`, and a family with no manifest is not a family.
    """
    labels = list(family_config.answer_labels)
    token_ids, rows, embedding = resolve_answer_rows(model, family_config)
    dim = int(model.hidden_dim)

    raw, unit, raw_norms = centered_answer_directions(rows, labels)

    stacked_raw = np.stack([raw[label] for label in labels], axis=0)
    centered_sum_residual = float(np.max(np.abs(stacked_raw.sum(axis=0))))

    answer_basis, basis_error = modified_gram_schmidt(
        [unit[label] for label in labels], TOLERANCES.rank_tolerance
    )
    if not answer_basis:
        raise DirectionFamilyError(
            "the answer directions span no dimensions at all; the unembedding rows for the "
            "answer tokens are identical"
        )

    seed, seed_labels = random_family_seed(
        family_config.master_seed,
        family_config.study_id,
        family_config.family_id,
        embedding.model_id,
        embedding.revision,
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    controls, redraws = orthogonal_random_controls(
        dim=dim,
        count=family_config.random_control_count,
        answer_basis=answer_basis,
        tolerances=TOLERANCES,
        draw=lambda: generator.standard_normal(dim),
    )

    # Validate on the float32 values that will actually be stored and later loaded, not on the
    # float64 originals. A family that is orthogonal only before rounding is not orthogonal.
    answer_f32 = {label: unit[label].astype(np.float32) for label in labels}
    control_f32 = [vector.astype(np.float32) for vector in controls]

    max_norm_error = 0.0
    for vector in (*answer_f32.values(), *control_f32):
        if not np.isfinite(vector).all():
            raise DirectionFamilyError("a constructed direction contains non-finite values")
        max_norm_error = max(
            max_norm_error, abs(float(np.linalg.norm(vector.astype(np.float64))) - 1.0)
        )
    if max_norm_error > TOLERANCES.norm_tolerance:
        raise DirectionFamilyError(
            f"a stored direction is not unit norm: worst error {max_norm_error:.3e} exceeds "
            f"{TOLERANCES.norm_tolerance:.0e}"
        )

    max_answer_to_random = 0.0
    for label in labels:
        left = answer_f32[label].astype(np.float64)
        for control in control_f32:
            max_answer_to_random = max(
                max_answer_to_random, abs(float(np.dot(left, control.astype(np.float64))))
            )
    if max_answer_to_random > TOLERANCES.orthogonality_tolerance:
        raise DirectionFamilyError(
            f"a random control is not orthogonal to the answer directions: worst absolute dot "
            f"{max_answer_to_random:.3e} exceeds {TOLERANCES.orthogonality_tolerance:.0e}"
        )

    max_random_to_random = 0.0
    for index, left in enumerate(control_f32):
        for right in control_f32[index + 1 :]:
            max_random_to_random = max(
                max_random_to_random,
                abs(float(np.dot(left.astype(np.float64), right.astype(np.float64)))),
            )
    if max_random_to_random > TOLERANCES.orthogonality_tolerance:
        raise DirectionFamilyError(
            f"two random controls are not orthogonal: worst absolute dot "
            f"{max_random_to_random:.3e} exceeds {TOLERANCES.orthogonality_tolerance:.0e}"
        )

    # Reported, never enforced. The answer directions are linearly dependent by construction.
    max_answer_cosine = 0.0
    for index, label in enumerate(labels):
        for other in labels[index + 1 :]:
            max_answer_cosine = max(
                max_answer_cosine,
                abs(
                    float(
                        np.dot(
                            answer_f32[label].astype(np.float64),
                            answer_f32[other].astype(np.float64),
                        )
                    )
                ),
            )

    planned: list[PlannedDirection] = []
    for index, label in enumerate(labels):
        planned.append(
            (
                opaque_direction_id(
                    family_config.direction_id_prefix,
                    family_config.family_id,
                    embedding.revision,
                    family_config.master_seed,
                    DirectionConstructionRole.ANSWER_TOKEN_CENTERED,
                    index,
                ),
                DirectionConstructionRole.ANSWER_TOKEN_CENTERED,
                index,
                label,
                token_ids[label],
                raw_norms[label],
                answer_f32[label],
            )
        )
    for index, vector in enumerate(control_f32):
        planned.append(
            (
                opaque_direction_id(
                    family_config.direction_id_prefix,
                    family_config.family_id,
                    embedding.revision,
                    family_config.master_seed,
                    DirectionConstructionRole.RANDOM_ORTHOGONAL_CONTROL,
                    index,
                ),
                DirectionConstructionRole.RANDOM_ORTHOGONAL_CONTROL,
                index,
                None,
                None,
                None,
                vector,
            )
        )

    opaque_ids = [item[0] for item in planned]
    if len(set(opaque_ids)) != len(opaque_ids):
        raise DirectionFamilyError(f"opaque direction ids collided: {sorted(opaque_ids)}")

    # Ordered by opaque id, so position in the manifest carries no role information either.
    planned.sort(key=lambda item: item[0])

    entries: list[DirectionEntry] = []
    for opaque_id, role, role_index, label, token_id, raw_norm, vector in planned:
        if save:
            store.save(
                opaque_id,
                vector,
                # Role-neutral by construction. The npz sits in the payload store that
                # resolution reads from, so it carries no construction role, no answer label,
                # and not even the algorithm version, whose name says what the family is made
                # of. All of that lives in the manifest, which is private provenance.
                {
                    "family_id": family_config.family_id,
                    "model_id": embedding.model_id,
                    "model_revision": embedding.revision,
                    "hidden_dim": dim,
                    "validated": False,
                    "note": (
                        "constructed intervention stimulus; not estimated from data and not "
                        "causally validated"
                    ),
                },
            )
            artifact_hash = store.hash(opaque_id)
        else:
            artifact_hash = sha256_hex(b"")

        entries.append(
            DirectionEntry(
                opaque_id=opaque_id,
                construction_role=role,
                role_index=role_index,
                label=label,
                token_id=token_id,
                dim=dim,
                vector_hash=vector_content_hash(vector),
                artifact_hash=artifact_hash,
                raw_norm=raw_norm,
                norm=float(np.linalg.norm(vector.astype(np.float64))),
            )
        )

    diagnostics = DirectionFamilyDiagnostics(
        centered_sum_max_abs_residual=centered_sum_residual,
        answer_span_orthonormality_error=basis_error,
        max_answer_pairwise_abs_cosine=max_answer_cosine,
        max_norm_error=max_norm_error,
        max_answer_to_random_abs_dot=max_answer_to_random,
        max_random_to_random_abs_dot=max_random_to_random,
        random_redraws=redraws,
    )

    relative_config = _repo_relative(Path(config_path))
    draft: dict[str, Any] = {
        "schema_version": DirectionFamilyRecord.model_fields["schema_version"].default,
        "family_id": family_config.family_id,
        "study_id": family_config.study_id,
        "construction_algorithm_version": CONSTRUCTION_ALGORITHM_VERSION,
        "basis_algorithm_version": BASIS_ALGORITHM_VERSION,
        "random_algorithm_version": RANDOM_ALGORITHM_VERSION,
        "model_id": embedding.model_id,
        "model_revision": embedding.revision,
        "tokenizer_revision": embedding.revision,
        "output_embedding_source": embedding.source,
        "tied_embeddings": embedding.tied,
        "hidden_dim": dim,
        "vocab_size": embedding.vocab_size,
        "answer_labels": labels,
        "answer_token_ids": dict(token_ids),
        "master_seed": family_config.master_seed,
        "seed_derivation_labels": seed_labels,
        "derived_random_seed": seed,
        "random_generator": RANDOM_GENERATOR,
        "answer_span_rank": len(answer_basis),
        "tolerances": TOLERANCES.model_dump(mode="json"),
        "directions": [entry.model_dump(mode="json") for entry in entries],
        "config_hash": config_hash(config_path),
    }

    record = DirectionFamilyRecord(
        family_id=family_config.family_id,
        study_id=family_config.study_id,
        construction_algorithm_version=CONSTRUCTION_ALGORITHM_VERSION,
        basis_algorithm_version=BASIS_ALGORITHM_VERSION,
        random_algorithm_version=RANDOM_ALGORITHM_VERSION,
        model_id=embedding.model_id,
        model_revision=embedding.revision,
        tokenizer_revision=embedding.revision,
        output_embedding_source=embedding.source,
        tied_embeddings=embedding.tied,
        config_tie_word_embeddings=embedding.config_tie_word_embeddings,
        hidden_dim=dim,
        vocab_size=embedding.vocab_size,
        answer_labels=labels,
        answer_token_ids=dict(token_ids),
        master_seed=family_config.master_seed,
        seed_derivation_labels=seed_labels,
        derived_random_seed=seed,
        random_generator=RANDOM_GENERATOR,
        answer_span_rank=len(answer_basis),
        tolerances=TOLERANCES,
        directions=entries,
        diagnostics=diagnostics,
        config_hash=str(draft["config_hash"]),
        family_hash=compute_direction_family_hash(draft),
        config_path=relative_config,
        environment={"packages": package_versions()},
        notes=(
            "Constructed intervention stimuli. No prompt was run, no state was captured, and "
            "no intervention was applied. This is not causal validation."
        ),
    )
    return record


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root())).replace("\\", "/")
    except ValueError:
        return path.name


# ---------------------------------------------------------------------------
# Storage and verification
# ---------------------------------------------------------------------------


def family_content_bytes(record: DirectionFamilyRecord) -> bytes:
    return canonical_json_bytes(direction_family_payload(record.model_dump(mode="json")))


def load_direction_family(family_id: str) -> DirectionFamilyRecord:
    path = direction_manifest_path(family_id)
    if not path.exists():
        raise DirectionFamilyError(f"no direction family manifest at {path}")
    try:
        return DirectionFamilyRecord.model_validate(read_json(path))
    except Exception as error:
        raise DirectionFamilyError(f"{path} is not a valid direction family: {error}") from error


def write_direction_family(record: DirectionFamilyRecord, force: bool = False) -> tuple[Path, str]:
    """Write the manifest atomically, refusing to replace a different family."""
    path = direction_manifest_path(record.family_id)
    if path.exists():
        existing = load_direction_family(record.family_id)
        if existing.family_hash == record.family_hash:
            return path, "unchanged"
        if not force:
            raise DirectionFamilyError(
                f"{path} already holds a different direction family "
                f"(existing {existing.family_hash}, proposed {record.family_hash}). Replacing a "
                "built family silently would orphan every artifact that cites it. Pass force "
                "only when a documented implementation bug invalidates the previous family, and "
                "record why in docs/experiment_log.md."
            )
        atomic_write_json(path, record.model_dump(mode="json"))
        return path, "overwritten"

    atomic_write_json(path, record.model_dump(mode="json"))
    return path, "written"


def verify_direction_family(
    record: DirectionFamilyRecord,
    store: DirectionStore,
) -> dict[str, Any]:
    """Check the stored artifacts against the manifest, without loading a model."""
    failures: list[str] = []
    container_notes: list[str] = []
    vectors: dict[str, np.ndarray] = {}

    for entry in record.directions:
        if not store.has(entry.opaque_id):
            failures.append(f"{entry.opaque_id}: no stored vector artifact")
            continue
        vector = store.load(entry.opaque_id).numpy().astype(np.float32)
        if vector.ndim != 1 or vector.shape[0] != record.hidden_dim:
            failures.append(
                f"{entry.opaque_id}: stored shape {vector.shape} does not match hidden dim "
                f"{record.hidden_dim}"
            )
            continue
        if not np.isfinite(vector).all():
            failures.append(f"{entry.opaque_id}: stored vector contains non-finite values")
            continue
        if vector_content_hash(vector) != entry.vector_hash:
            failures.append(f"{entry.opaque_id}: stored vector does not match its recorded hash")
            continue
        norm_error = abs(float(np.linalg.norm(vector.astype(np.float64))) - 1.0)
        if norm_error > record.tolerances.norm_tolerance:
            failures.append(f"{entry.opaque_id}: norm error {norm_error:.3e} exceeds tolerance")
        if store.hash(entry.opaque_id) != entry.artifact_hash:
            # Non-fatal: the archive container can differ across writer versions while the
            # values inside it are identical, and the values are what the family is.
            container_notes.append(
                f"{entry.opaque_id}: the .npz container hash differs from the recorded one "
                "while the vector contents match"
            )
        vectors[entry.opaque_id] = vector

    answers = record.by_role(DirectionConstructionRole.ANSWER_TOKEN_CENTERED)
    controls = record.by_role(DirectionConstructionRole.RANDOM_ORTHOGONAL_CONTROL)
    if len(answers) != len(record.answer_labels):
        failures.append(
            f"expected {len(record.answer_labels)} answer directions, found {len(answers)}"
        )
    if not controls:
        failures.append("the family has no random controls")

    max_answer_to_random = 0.0
    max_random_to_random = 0.0
    if not failures:
        for answer in answers:
            left = vectors[answer.opaque_id].astype(np.float64)
            for control in controls:
                max_answer_to_random = max(
                    max_answer_to_random,
                    abs(float(np.dot(left, vectors[control.opaque_id].astype(np.float64)))),
                )
        for index, left_entry in enumerate(controls):
            for right_entry in controls[index + 1 :]:
                max_random_to_random = max(
                    max_random_to_random,
                    abs(
                        float(
                            np.dot(
                                vectors[left_entry.opaque_id].astype(np.float64),
                                vectors[right_entry.opaque_id].astype(np.float64),
                            )
                        )
                    ),
                )
        if max_answer_to_random > record.tolerances.orthogonality_tolerance:
            failures.append(
                f"answer-to-random orthogonality is broken: worst {max_answer_to_random:.3e}"
            )
        if max_random_to_random > record.tolerances.orthogonality_tolerance:
            failures.append(
                f"random-to-random orthogonality is broken: worst {max_random_to_random:.3e}"
            )

    return {
        "family_id": record.family_id,
        "family_hash": record.family_hash,
        "level": "artifacts",
        "valid": not failures,
        "failures": failures,
        "container_notes": container_notes,
        "directions_checked": len(vectors),
        "answer_directions": len(answers),
        "random_controls": len(controls),
        "answer_span_rank": record.answer_span_rank,
        "max_answer_to_random_abs_dot": max_answer_to_random,
        "max_random_to_random_abs_dot": max_random_to_random,
    }


def regenerate_direction_family(
    record: DirectionFamilyRecord,
    family_config: DirectionFamilyConfig,
    model: LoadedModel,
    store: DirectionStore,
    config_path: str | Path,
) -> dict[str, Any]:
    """Rebuild the family from the model and compare it to the manifest.

    Nothing is written. The rebuild runs with `save=False`, so a regeneration check can never
    quietly replace the artifacts it is supposed to be checking.
    """
    rebuilt = build_direction_family(
        family_config, model, store, config_path=config_path, save=False
    )

    failures: list[str] = []
    if rebuilt.model_revision != record.model_revision:
        failures.append(
            f"model revision differs: manifest {record.model_revision}, loaded "
            f"{rebuilt.model_revision}"
        )
    if rebuilt.tokenizer_revision != record.tokenizer_revision:
        failures.append("tokenizer revision differs from the manifest")
    if rebuilt.answer_token_ids != record.answer_token_ids:
        failures.append(
            f"answer token ids differ: manifest {record.answer_token_ids}, resolved "
            f"{rebuilt.answer_token_ids}"
        )

    recorded = {entry.opaque_id: entry for entry in record.directions}
    rebuilt_by_id = {entry.opaque_id: entry for entry in rebuilt.directions}
    if set(recorded) != set(rebuilt_by_id):
        failures.append("the rebuilt family has a different set of opaque ids")
    else:
        for opaque_id, entry in recorded.items():
            other = rebuilt_by_id[opaque_id]
            if other.vector_hash != entry.vector_hash:
                failures.append(f"{opaque_id}: rebuilt vector differs from the recorded hash")
            if other.construction_role is not entry.construction_role:
                failures.append(f"{opaque_id}: rebuilt construction role differs")
            if store.has(opaque_id):
                stored = store.load(opaque_id).numpy().astype(np.float32)
                if vector_content_hash(stored) != other.vector_hash:
                    failures.append(f"{opaque_id}: stored vector differs from the rebuilt vector")
            else:
                failures.append(f"{opaque_id}: no stored vector artifact to compare against")

    payload_matches = family_content_bytes(rebuilt) == family_content_bytes(record)
    if not payload_matches:
        failures.append("the rebuilt family content hash differs from the manifest")

    return {
        "family_id": record.family_id,
        "level": "regeneration",
        "valid": not failures,
        "failures": failures,
        "manifest_family_hash": record.family_hash,
        "rebuilt_family_hash": rebuilt.family_hash,
        "model_revision": rebuilt.model_revision,
        "answer_token_ids": dict(rebuilt.answer_token_ids),
    }


# ---------------------------------------------------------------------------
# Command entry points
# ---------------------------------------------------------------------------


def _load_family_config(config_path: str | Path) -> DirectionFamilyConfig:
    try:
        return load_config(config_path, DirectionFamilyConfig)
    except ConfigError as error:
        raise DirectionFamilyError(str(error)) from error


def _load_family_model(family_config: DirectionFamilyConfig) -> LoadedModel:
    try:
        model_config = load_config(family_config.model_ref, ModelConfig)
    except ConfigError as error:
        raise DirectionFamilyError(
            f"direction family config {family_config.name!r} references a model config that "
            f"does not load: {error}"
        ) from error
    try:
        return load_model(model_config)
    except ModelLoadError as error:
        raise DirectionFamilyError(f"could not load the pinned model: {error}") from error


def _report(record: DirectionFamilyRecord, status: str, path: Path) -> dict[str, Any]:
    return {
        "status": status,
        "manifest_path": _repo_relative(path),
        "family_id": record.family_id,
        "family_hash": record.family_hash,
        "study_id": record.study_id,
        "model_id": record.model_id,
        "model_revision": record.model_revision,
        "tokenizer_revision": record.tokenizer_revision,
        "output_embedding_source": record.output_embedding_source,
        "output_embedding_shape": [record.vocab_size, record.hidden_dim],
        "tied_embeddings": record.tied_embeddings,
        "config_tie_word_embeddings": record.config_tie_word_embeddings,
        "hidden_dim": record.hidden_dim,
        "answer_token_ids": dict(record.answer_token_ids),
        "master_seed": record.master_seed,
        "derived_random_seed": record.derived_random_seed,
        "answer_span_rank": record.answer_span_rank,
        "direction_count": len(record.directions),
        "opaque_direction_ids": [entry.opaque_id for entry in record.directions],
        "vector_hashes": {entry.opaque_id: entry.vector_hash for entry in record.directions},
        "raw_answer_norms": {
            entry.label: entry.raw_norm
            for entry in record.by_role(DirectionConstructionRole.ANSWER_TOKEN_CENTERED)
            if entry.label is not None
        },
        "diagnostics": record.diagnostics.model_dump(mode="json"),
        "tolerances": record.tolerances.model_dump(mode="json"),
        "notes": (
            "Direction construction only. No prompt was run, no state was captured, no "
            "intervention was applied, and this is not causal validation."
        ),
    }


def build_family_command(
    config_path: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    """Build the family, store the vectors, and write the manifest."""
    family_config = _load_family_config(config_path)
    store = DirectionStore(directions_dir())
    model = _load_family_model(family_config)

    record = build_direction_family(family_config, model, store, config_path=config_path)
    path, status = write_direction_family(record, force=force)
    verification = verify_direction_family(record, store)

    info(
        "built direction family",
        family_id=record.family_id,
        status=status,
        directions=len(record.directions),
        family_hash=record.family_hash[:23],
    )

    report = _report(record, status, path)
    report["artifact_verification"] = verification
    return report


def verify_family_command(
    family_id: str,
    regenerate: bool = False,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a stored family, optionally rebuilding it from the pinned model."""
    record = load_direction_family(family_id)
    store = DirectionStore(directions_dir())
    report = _report(record, "verified", direction_manifest_path(family_id))
    report["artifact_verification"] = verify_direction_family(record, store)

    if regenerate:
        resolved_config = config_path or record.config_path
        family_config = _load_family_config(resolved_config)
        if family_config.family_id != record.family_id:
            raise DirectionFamilyError(
                f"config {resolved_config} builds family {family_config.family_id!r}, not "
                f"{record.family_id!r}"
            )
        model = _load_family_model(family_config)
        report["regeneration_verification"] = regenerate_direction_family(
            record, family_config, model, store, config_path=resolved_config
        )

    report["valid"] = bool(report["artifact_verification"]["valid"]) and bool(
        report.get("regeneration_verification", {"valid": True})["valid"]
    )
    return report


def family_manifest_hash(family_id: str) -> str:
    return hash_file(direction_manifest_path(family_id))


__all__ = [
    "BASIS_ALGORITHM_VERSION",
    "CONSTRUCTION_ALGORITHM_VERSION",
    "RANDOM_ALGORITHM_VERSION",
    "TOLERANCES",
    "DirectionFamilyError",
    "build_direction_family",
    "build_family_command",
    "canonicalize_sign",
    "centered_answer_directions",
    "family_content_bytes",
    "load_direction_family",
    "modified_gram_schmidt",
    "opaque_direction_id",
    "orthogonal_random_controls",
    "random_family_seed",
    "regenerate_direction_family",
    "resolve_answer_rows",
    "vector_content_hash",
    "verify_direction_family",
    "verify_family_command",
    "write_direction_family",
]
