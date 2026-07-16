"""Storage and generation of intervention directions.

Directions are artifacts, not incidental tensors. A direction that steers behavior is the
central object of the causal claims here, so each one is written to disk with the metadata
describing how it was estimated, and it is cited by hash wherever it is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..hashing import hash_file
from ..reproducibility import derive_seed
from .tensor_ops import MIN_DIRECTION_NORM, InterventionShapeError


class DirectionNotFoundError(KeyError):
    """Raised when a referenced direction is not in the store."""


def matched_random_direction(
    reference: torch.Tensor,
    seed: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw a random direction with the same norm as a reference direction.

    Norm matching is what makes this a control rather than a weaker intervention. An
    unmatched random vector would almost always produce a smaller effect simply because it
    is smaller, and the comparison would say nothing about whether the reference direction
    is special.
    """
    if reference.ndim != 1:
        raise InterventionShapeError(
            f"reference direction must be 1-D, got shape {tuple(reference.shape)}"
        )
    target_norm = torch.linalg.vector_norm(reference)
    if target_norm.item() < MIN_DIRECTION_NORM:
        raise InterventionShapeError(
            f"reference norm {target_norm.item():.3e} is too small to match"
        )

    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) % (2**63 - 1))

    # Drawn on CPU in float32 regardless of the reference dtype: the same seed then yields
    # the same control direction whether the run is on CPU or a GPU, in any precision.
    raw = torch.randn(reference.shape, generator=generator, dtype=torch.float32, device="cpu")
    raw_norm = torch.linalg.vector_norm(raw)
    unit = raw / raw_norm
    return (unit * target_norm.to(torch.float32).cpu()).to(
        device=reference.device, dtype=reference.dtype
    )


def random_direction_seed(run_seed: int, direction_id: str, index: int = 0) -> int:
    """Derive the seed for one control direction."""
    return derive_seed("random_direction", run_seed, direction_id, index)


class DirectionStore:
    """A directory of direction artifacts.

    One `.npz` per direction, holding the vector and its metadata. The format is boring on
    purpose: a reviewer can open it with numpy alone and read the vector without installing
    this package.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, direction_id: str) -> Path:
        return self.root / f"{direction_id}.npz"

    def has(self, direction_id: str) -> bool:
        return self.path_for(direction_id).exists()

    def list_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(path.stem for path in self.root.glob("*.npz"))

    def save(
        self,
        direction_id: str,
        vector: torch.Tensor | np.ndarray,
        metadata: dict[str, Any],
    ) -> Path:
        """Write a direction and its provenance metadata.

        Stored in float32 on CPU. Directions are estimated once and reused across runs that
        may use different compute dtypes, so the artifact keeps a single canonical precision
        rather than inheriting whatever the estimating run happened to use.
        """
        array = (
            vector.detach().cpu().to(torch.float32).numpy()
            if isinstance(vector, torch.Tensor)
            else np.asarray(vector, dtype=np.float32)
        )
        if array.ndim != 1:
            raise InterventionShapeError(
                f"direction {direction_id} must be 1-D, got shape {array.shape}"
            )
        if not np.isfinite(array).all():
            raise InterventionShapeError(f"direction {direction_id} contains non-finite values")
        norm = float(np.linalg.norm(array))
        if norm < MIN_DIRECTION_NORM:
            raise InterventionShapeError(
                f"direction {direction_id} has norm {norm:.3e} and is effectively zero"
            )

        enriched = dict(metadata)
        enriched.update({"direction_id": direction_id, "dim": int(array.shape[0]), "norm": norm})

        self.root.mkdir(parents=True, exist_ok=True)
        target = self.path_for(direction_id)
        np.savez(target, vector=array, metadata=json.dumps(enriched, sort_keys=True))
        return target

    def load(self, direction_id: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        path = self.path_for(direction_id)
        if not path.exists():
            available = self.list_ids()
            raise DirectionNotFoundError(
                f"direction {direction_id!r} not found in {self.root}; available: {available}"
            )
        with np.load(path, allow_pickle=False) as handle:
            array = handle["vector"]
        return torch.from_numpy(np.asarray(array, dtype=np.float32)).to(dtype)

    def metadata(self, direction_id: str) -> dict[str, Any]:
        path = self.path_for(direction_id)
        if not path.exists():
            raise DirectionNotFoundError(f"direction {direction_id!r} not found in {self.root}")
        with np.load(path, allow_pickle=False) as handle:
            raw = str(handle["metadata"])
        return json.loads(raw)

    def hash(self, direction_id: str) -> str:
        return hash_file(self.path_for(direction_id))
