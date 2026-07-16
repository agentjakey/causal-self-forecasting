"""Canonical serialization, hashing, and atomic writes.

Everything the commitment protocol depends on lives here. The rules are deliberately
strict: a forecast that cannot be serialized to exactly one byte string cannot be
committed to, because the verifier would not be able to recompute the same hash later.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

SALT_BYTES = 32
_HASH_PREFIX = "sha256:"


class CanonicalizationError(ValueError):
    """Raised when a value cannot be canonically serialized."""


def _check_float(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        raise CanonicalizationError(
            f"non-finite float {value!r} cannot be canonically serialized; "
            "forecasts must contain finite numbers only"
        )
    return value


def _canonicalize(value: Any) -> Any:
    """Recursively convert a value into JSON-native types with a deterministic layout.

    Rejects anything whose serialization would be ambiguous or platform-dependent.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        # Must precede the int branch: bool is a subclass of int.
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _check_float(value)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(
                    f"mapping keys must be strings for canonical JSON, got {type(key).__name__}"
                )
            out[key] = _canonicalize(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]

    # numpy scalars and arrays show up constantly in this codebase. Convert them
    # explicitly rather than letting json fail with an unhelpful message.
    item_method = getattr(value, "item", None)
    tolist_method = getattr(value, "tolist", None)
    if tolist_method is not None and getattr(value, "ndim", None) is not None:
        if value.ndim == 0 and item_method is not None:
            return _canonicalize(item_method())
        return _canonicalize(tolist_method())

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _canonicalize(model_dump(mode="json"))

    raise CanonicalizationError(
        f"value of type {type(value).__name__} has no canonical JSON representation"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value to canonical JSON bytes.

    Canonical here means: keys sorted, no insignificant whitespace, ASCII-escaped, and no
    NaN or Infinity. The same logical value always produces the same bytes, on any platform,
    which is what makes a commitment hash verifiable by a third party.
    """
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_text(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Hash bytes, returning a prefixed digest such as `sha256:ab12...`."""
    return _HASH_PREFIX + hashlib.sha256(data).hexdigest()


def hash_object(value: Any) -> str:
    """Hash any canonically serializable value."""
    return sha256_hex(canonical_json_bytes(value))


def hash_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hash a file's contents without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return _HASH_PREFIX + digest.hexdigest()


def new_salt() -> bytes:
    """Generate a commitment salt from the OS cryptographic RNG.

    Never seeded, never derived from the experiment seed. A salt that a reader could
    predict from the run config would defeat the point of committing.
    """
    return secrets.token_bytes(SALT_BYTES)


def commitment_hash(payload: Any, salt: bytes) -> str:
    """Compute SHA256(canonical_payload_bytes + salt_bytes)."""
    if not isinstance(salt, (bytes, bytearray)):
        raise TypeError(f"salt must be bytes, got {type(salt).__name__}")
    if len(salt) < 16:
        raise ValueError(f"salt must be at least 16 bytes, got {len(salt)}")
    return sha256_hex(canonical_json_bytes(payload) + bytes(salt))


def verify_commitment(payload: Any, salt: bytes, expected_hash: str) -> bool:
    """Recompute a commitment and compare it in constant time."""
    return secrets.compare_digest(commitment_hash(payload, salt), expected_hash)


def salt_to_hex(salt: bytes) -> str:
    return salt.hex()


def salt_from_hex(text: str) -> bytes:
    return bytes.fromhex(text.strip())


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Write bytes so that readers never observe a partial file.

    Writes a temporary file in the destination directory, fsyncs it, then renames. The
    rename is atomic on both Windows and POSIX when source and destination share a volume,
    which is why the temporary file is not placed in the system temp directory.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return target


def atomic_write_text(path: str | Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, value: Any) -> Path:
    """Write canonical JSON with a trailing newline."""
    return atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def append_jsonl(path: str | Path, value: Any) -> Path:
    """Append one canonical JSON record as a line.

    Append-only by design. Records in these files are evidence, so the writer never
    rewrites an existing line.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("ab") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return target


def write_jsonl(path: str | Path, values: Iterable[Any]) -> Path:
    """Write a whole JSONL file atomically, replacing any existing file."""
    body = b"".join(canonical_json_bytes(value) + b"\n" for value in values)
    return atomic_write_bytes(path, body)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines and reporting the line number on failure."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
