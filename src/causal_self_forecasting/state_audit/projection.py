"""The fixed intervention projection matrix.

Every primary method receives the identical 16-dimensional description of the intervention:

```text
x_intervention = P^T v
```

where `v` is the actual signed intervention vector, 1152-dimensional and exactly zero for the
no-op, and `P` is a fixed `1152 x 16` matrix with orthonormal columns.

Three properties are frozen in `docs/bluedot/preregistration_state_dependence.md` section 7.4 and
enforced here.

**`P` is never fitted.** It is generated once from the master seed, stored, hashed, and cited by
every run. It has no training-boundary exposure at all, so calibration, training, and final test
all use the same map and no comparison can be contaminated by when it was built.

**Every method gets the same `x_intervention`.** The intervention block is shared identically by
the intervention-only, visible-information, and state-conditioned models. What separates them is
what else they see, not how well they see the intervention.

**A random projection, not the direction bank itself.** Projecting onto the eight bank directions
would make coordinate index equal family identity, which is the intervention-label shortcut by
another route. The trade is stated rather than hidden: because `x_intervention` is a fixed linear
map of `v`, family membership is in principle linearly recoverable from it. That is intended. This
arm deliberately hands every method a *complete* description of the intervention; what is withheld
is the semantic label, not linear separability.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ..config import repo_root
from ..hashing import atomic_write_json, hash_file, read_json, sha256_hex
from ..paths import projection_path
from ..reproducibility import derive_seed
from ..schemas import InterventionProjectionRecord

PROJECTION_ALGORITHM_VERSION = "gaussian_qr_orthonormal_v1.0"
PROJECTION_SEED_LABEL = "bluedot.intervention_projection"
PROJECTION_GENERATOR = "numpy.random.Generator(PCG64)"

# Frozen tolerances, applied to the float32 matrix that is actually stored and loaded. A
# projection that is orthonormal only before rounding is not orthonormal.
ORTHONORMALITY_TOLERANCE = 1e-5
# Injectivity is a rank property, not a magnitude one, and the two are easy to confuse here.
#
# The answer-token family is near-degenerate *by construction*: the four raw centered directions
# sum to zero, so the four unit directions span four dimensions only barely, and the realized set's
# own smallest nonzero singular value is tiny before any projection is applied. An absolute floor
# on the projected singular value would therefore be measuring the direction family's conditioning
# and calling it a property of `P`.
#
# What `P` must not do is *lose* a dimension. So the gate is rank preservation, enforced by
# `injectivity_margin` raising, plus a relative check: the projected smallest nonzero singular
# value must retain this fraction of the source's. A well-conditioned orthonormal projection of a
# subspace retains essentially all of it.
INJECTIVITY_RETENTION_TOLERANCE = 1e-3


class ProjectionError(RuntimeError):
    """Raised when the projection matrix cannot be built, stored, or trusted."""


def projection_seed(master_seed: int, study_id: str) -> int:
    return derive_seed(PROJECTION_SEED_LABEL, master_seed, study_id, PROJECTION_ALGORITHM_VERSION)


def matrix_content_hash(matrix: np.ndarray) -> str:
    """Hash the float32 values, independent of how they are stored."""
    return sha256_hex(np.ascontiguousarray(matrix, dtype="<f4").tobytes())


def build_projection_matrix(hidden_dim: int, components: int, seed: int) -> np.ndarray:
    """Draw a Gaussian matrix and take the `Q` factor of its QR decomposition.

    Built in float64 and returned in float32, matching how directions are stored. The sign
    convention of `numpy.linalg.qr` is not canonicalized: unlike the direction family, `P` is
    never regenerated against a rank-deficient input, and the stored matrix is the artifact of
    record. Regeneration from the same seed reproduces it exactly on the same NumPy generator,
    which the verifier checks.
    """
    if components > hidden_dim:
        raise ProjectionError(
            f"cannot take {components} orthonormal columns in {hidden_dim} dimensions"
        )
    generator = np.random.Generator(np.random.PCG64(seed))
    gaussian = generator.standard_normal((hidden_dim, components))
    q, _ = np.linalg.qr(gaussian)
    if q.shape != (hidden_dim, components):
        raise ProjectionError(f"QR produced shape {q.shape}, expected {(hidden_dim, components)}")
    return np.ascontiguousarray(q, dtype=np.float32)


def orthonormality_error(matrix: np.ndarray) -> float:
    """Worst absolute deviation of `P^T P` from the identity, in float64."""
    dense = np.asarray(matrix, dtype=np.float64)
    gram = dense.T @ dense
    return float(np.max(np.abs(gram - np.eye(gram.shape[0]))))


def project(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """`P^T v`, in float64, for one signed intervention vector."""
    dense = np.asarray(matrix, dtype=np.float64)
    operand = np.asarray(vector, dtype=np.float64)
    if operand.shape != (dense.shape[0],):
        raise ProjectionError(
            f"cannot project a vector of shape {operand.shape} with a {dense.shape} matrix"
        )
    return dense.T @ operand


def injectivity_margin(matrix: np.ndarray, vectors: list[np.ndarray]) -> float:
    """How far the projection is from collapsing the realized intervention set.

    The realized signed vectors are `+d` and `-d` for each of the eight directions, so they are
    pairwise collinear and span at most eight dimensions, not sixteen. Asking for the sixteenth
    singular value of their projection would therefore always return zero and would say nothing
    about injectivity.

    The property that matters is that the map does not lose rank: `P^T` is injective on the span
    of a set exactly when `rank(P^T V) == rank(V)`, and the margin is then the smallest **nonzero**
    singular value, which is the distance to a collapse.
    """
    if not vectors:
        raise ProjectionError("injectivity needs at least one realized intervention vector")
    source = np.stack([np.asarray(v, dtype=np.float64) for v in vectors], axis=0)
    projected = np.stack([project(matrix, vector) for vector in vectors], axis=0)

    source_singular = np.linalg.svd(source, compute_uv=False)
    # Relative rank tolerance, the convention numpy.linalg.matrix_rank uses.
    tolerance = source_singular.max() * max(source.shape) * float(np.finfo(np.float64).eps)
    rank = int((source_singular > tolerance).sum())
    if rank == 0:
        raise ProjectionError("the realized intervention set is entirely zero")

    projected_singular = np.linalg.svd(projected, compute_uv=False)
    projected_rank = int((projected_singular > tolerance).sum())
    if projected_rank < rank:
        raise ProjectionError(
            f"the projection loses rank on the realized set: {rank} dimensions in, "
            f"{projected_rank} out"
        )
    return float(projected_singular[rank - 1])


def source_margin(vectors: list[np.ndarray]) -> float:
    """The smallest nonzero singular value of the realized set before any projection."""
    source = np.stack([np.asarray(v, dtype=np.float64) for v in vectors], axis=0)
    singular = np.linalg.svd(source, compute_uv=False)
    tolerance = singular.max() * max(source.shape) * float(np.finfo(np.float64).eps)
    rank = int((singular > tolerance).sum())
    if rank == 0:
        raise ProjectionError("the realized intervention set is entirely zero")
    return float(singular[rank - 1])


def injectivity_retention(matrix: np.ndarray, vectors: list[np.ndarray]) -> float:
    """What fraction of the set's smallest nonzero singular value survives the projection.

    One means the projection preserved the subspace exactly; near zero means it nearly collapsed
    a direction. This is the quantity that says something about `P`, as opposed to something
    about how well conditioned the direction family happens to be.
    """
    before = source_margin(vectors)
    if before <= 0.0:
        raise ProjectionError("the realized intervention set is rank zero")
    return injectivity_margin(matrix, vectors) / before


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root())).replace("\\", "/")
    except ValueError:
        return path.name


def build_projection(
    projection_id: str,
    study_id: str,
    hidden_dim: int,
    components: int,
    master_seed: int,
    direction_vectors: dict[str, np.ndarray] | None = None,
    direction_family_id: str = "",
    direction_family_hash: str = "",
) -> tuple[InterventionProjectionRecord, np.ndarray]:
    """Build and validate the projection matrix. Loads no model."""
    seed = projection_seed(master_seed, study_id)
    matrix = build_projection_matrix(hidden_dim, components, seed)

    if not np.isfinite(matrix).all():
        raise ProjectionError("the projection matrix contains non-finite values")
    error = orthonormality_error(matrix)
    if error > ORTHONORMALITY_TOLERANCE:
        raise ProjectionError(
            f"the projection columns are not orthonormal: worst deviation of P^T P from the "
            f"identity is {error:.3e}, above {ORTHONORMALITY_TOLERANCE:.0e}"
        )

    margin = 0.0
    retention = 0.0
    realized = 0
    if direction_vectors:
        # The realized set: every direction at both signs. The alpha scales every vector by the
        # same positive constant and cannot change whether the map is injective, so the unit
        # directions are the right thing to measure.
        vectors = [
            sign * np.asarray(vector, dtype=np.float64)
            for vector in direction_vectors.values()
            for sign in (1.0, -1.0)
        ]
        realized = len(vectors)
        # Raises if the projection loses a dimension, which is what injectivity means here.
        margin = injectivity_margin(matrix, vectors)
        retention = injectivity_retention(matrix, vectors)
        if not math.isfinite(retention) or retention < INJECTIVITY_RETENTION_TOLERANCE:
            raise ProjectionError(
                f"the projection nearly collapses the realized set: it retains {retention:.3e} of "
                f"the set's own smallest nonzero singular value, below "
                f"{INJECTIVITY_RETENTION_TOLERANCE:.0e}"
            )

    record = InterventionProjectionRecord(
        projection_id=projection_id,
        study_id=study_id,
        algorithm_version=PROJECTION_ALGORITHM_VERSION,
        random_generator=PROJECTION_GENERATOR,
        seed_label=PROJECTION_SEED_LABEL,
        master_seed=master_seed,
        derived_seed=seed,
        hidden_dim=hidden_dim,
        components=components,
        orthonormality_error=error,
        orthonormality_tolerance=ORTHONORMALITY_TOLERANCE,
        realized_vector_count=realized,
        injectivity_margin=margin,
        injectivity_retention=retention,
        direction_family_id=direction_family_id or None,
        direction_family_hash=direction_family_hash or None,
        matrix_hash=matrix_content_hash(matrix),
    )
    return record, matrix


def write_projection(
    record: InterventionProjectionRecord, matrix: np.ndarray, force: bool = False
) -> tuple[Path, str]:
    """Write the manifest and the matrix, refusing to replace a different projection."""
    manifest_path = projection_path(record.projection_id)
    matrix_path = manifest_path.with_suffix(".npz")
    # Captured before the write. Checking afterwards would always report "overwritten".
    existed = manifest_path.exists()

    if existed:
        existing = load_projection_record(record.projection_id)
        if existing.matrix_hash == record.matrix_hash and matrix_path.exists():
            return manifest_path, "unchanged"
        if not force:
            raise ProjectionError(
                f"{manifest_path} already holds a different projection (existing "
                f"{existing.matrix_hash}, proposed {record.matrix_hash}). Replacing it would "
                "orphan every forecast that cites it."
            )

    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(matrix_path, matrix=np.ascontiguousarray(matrix, dtype=np.float32))
    payload = record.model_dump(mode="json")
    payload["matrix_artifact_hash"] = hash_file(matrix_path)
    atomic_write_json(manifest_path, payload)
    return manifest_path, "overwritten" if existed else "written"


def load_projection_record(projection_id: str) -> InterventionProjectionRecord:
    path = projection_path(projection_id)
    if not path.exists():
        raise ProjectionError(f"no intervention projection at {path}")
    payload = read_json(path)
    if isinstance(payload, dict):
        payload = {k: v for k, v in payload.items() if k != "matrix_artifact_hash"}
    try:
        return InterventionProjectionRecord.model_validate(payload)
    except Exception as error:
        raise ProjectionError(f"{path} is not a valid intervention projection: {error}") from error


def load_projection_matrix(projection_id: str) -> np.ndarray:
    """Load the stored matrix and check it against the hash the manifest records."""
    record = load_projection_record(projection_id)
    matrix_path = projection_path(projection_id).with_suffix(".npz")
    if not matrix_path.exists():
        raise ProjectionError(f"no projection matrix at {matrix_path}")
    with np.load(matrix_path, allow_pickle=False) as handle:
        matrix = np.asarray(handle["matrix"], dtype=np.float32)
    if matrix.shape != (record.hidden_dim, record.components):
        raise ProjectionError(
            f"the stored matrix has shape {matrix.shape}, expected "
            f"{(record.hidden_dim, record.components)}"
        )
    if matrix_content_hash(matrix) != record.matrix_hash:
        raise ProjectionError(
            "the stored projection matrix does not match the hash its manifest records"
        )
    return matrix


def verify_projection(projection_id: str, regenerate: bool = True) -> dict[str, Any]:
    """Check the stored projection, optionally rebuilding it from the seed. Loads no model."""
    record = load_projection_record(projection_id)
    failures: list[str] = []

    try:
        matrix = load_projection_matrix(projection_id)
    except ProjectionError as error:
        return {
            "projection_id": projection_id,
            "valid": False,
            "failures": [str(error)],
            "matrix_hash": record.matrix_hash,
        }

    error_value = orthonormality_error(matrix)
    if error_value > record.orthonormality_tolerance:
        failures.append(
            f"orthonormality error {error_value:.3e} exceeds the recorded tolerance "
            f"{record.orthonormality_tolerance:.0e}"
        )
    if abs(error_value - record.orthonormality_error) > 1e-9:
        failures.append("the recorded orthonormality error does not match the stored matrix")

    rebuilt_hash = None
    if regenerate:
        rebuilt = build_projection_matrix(record.hidden_dim, record.components, record.derived_seed)
        rebuilt_hash = matrix_content_hash(rebuilt)
        if rebuilt_hash != record.matrix_hash:
            failures.append(
                "regenerating from the recorded seed produced a different matrix than the one "
                "stored"
            )

    return {
        "projection_id": projection_id,
        "valid": not failures,
        "failures": failures,
        "matrix_hash": record.matrix_hash,
        "rebuilt_matrix_hash": rebuilt_hash,
        "hidden_dim": record.hidden_dim,
        "components": record.components,
        "orthonormality_error": error_value,
        "injectivity_margin": record.injectivity_margin,
        "injectivity_retention": record.injectivity_retention,
        "realized_vector_count": record.realized_vector_count,
        "manifest_path": _repo_relative(projection_path(projection_id)),
        "scientific_result": False,
    }


__all__ = [
    "INJECTIVITY_RETENTION_TOLERANCE",
    "ORTHONORMALITY_TOLERANCE",
    "PROJECTION_ALGORITHM_VERSION",
    "PROJECTION_SEED_LABEL",
    "ProjectionError",
    "build_projection",
    "build_projection_matrix",
    "injectivity_margin",
    "injectivity_retention",
    "load_projection_matrix",
    "load_projection_record",
    "matrix_content_hash",
    "orthonormality_error",
    "project",
    "projection_seed",
    "verify_projection",
    "write_projection",
]
