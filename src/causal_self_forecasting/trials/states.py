"""Hidden-state shard storage.

Activations are too large for JSONL and too important to keep only in memory. They go into
one `.npz` per run, and every `ModelStateRef` cites that shard by hash. A state that cannot
be cited cannot be used as evidence for a state-swap claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from ..hashing import hash_file
from ..schemas import ModelStateRef, ModelVariant

STATES_FILENAME = "states.npz"

# np.savez takes array names as keyword arguments, so a state id equal to one of its own
# parameter names would be consumed as an option rather than saved.
_RESERVED_SHARD_KEYS = frozenset({"file", "args", "kwds", "allow_pickle"})


class StateShardWriter:
    """Accumulates captured states and writes them as one shard.

    Refs are returned without a shard hash until `close` is called, because the hash of a
    file that is still being written is meaningless. `close` fills the hash in and returns
    the completed refs.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._arrays: dict[str, np.ndarray] = {}
        self._pending: list[tuple[str, dict]] = []
        self._closed = False

    def add(
        self,
        state_id: str,
        variant_id: str,
        model_variant: ModelVariant,
        layer: int,
        position_index: int,
        vector: torch.Tensor,
    ) -> str:
        if self._closed:
            raise RuntimeError("cannot add states after the shard has been closed")
        if state_id in self._arrays:
            raise ValueError(f"duplicate state_id {state_id!r}")
        if state_id in _RESERVED_SHARD_KEYS:
            # State ids become keyword arguments to np.savez, where these names are its own
            # parameters and would be swallowed instead of stored.
            raise ValueError(f"state_id {state_id!r} collides with a reserved npz parameter")
        if vector.ndim != 1:
            raise ValueError(f"state {state_id} must be a 1-D vector, got {tuple(vector.shape)}")

        # Stored in float32 regardless of compute dtype, so a shard captured in bfloat16 on a
        # GPU and one captured in float32 on CPU are the same kind of object downstream.
        array = vector.detach().cpu().to(torch.float32).numpy()
        self._arrays[state_id] = array
        self._pending.append(
            (
                state_id,
                {
                    "state_id": state_id,
                    "variant_id": variant_id,
                    "model_variant": model_variant,
                    "layer": layer,
                    "position_index": position_index,
                    "hidden_dim": int(array.shape[0]),
                    "dtype": "float32",
                    "row_index": len(self._arrays) - 1,
                },
            )
        )
        return state_id

    def close(self) -> list[ModelStateRef]:
        if self._closed:
            raise RuntimeError("shard is already closed")
        if not self._arrays:
            raise RuntimeError("refusing to write an empty state shard")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # savez types its **kwds as its own options; `add` has already rejected any state id
        # that would actually collide with one.
        save = cast(Any, np.savez)
        save(self.path, **self._arrays)
        self._closed = True

        shard_hash = hash_file(self.path)
        relative = self.path.name
        return [
            ModelStateRef(shard_path=relative, shard_hash=shard_hash, **fields)
            for _, fields in self._pending
        ]


def load_state(
    shard_path: str | Path, state_id: str, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    with np.load(Path(shard_path), allow_pickle=False) as handle:
        if state_id not in handle:
            raise KeyError(f"state {state_id!r} is not in shard {shard_path}")
        array = handle[state_id]
    return torch.from_numpy(np.asarray(array, dtype=np.float32)).to(dtype)
