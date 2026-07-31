"""The frozen protocol copies must stay byte-identical to the artifacts the code loads.

`protocol/` holds a second copy of the prompt manifest, direction family, and calibration plan so
the protocol reads as one directory. The code loads them from `data/`. Two copies drift; this test
is what stops that drift from being silent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / "protocol"

# protocol/ filename -> the canonical path the code actually loads.
COPIES = {
    "bluedot_state_dependence_v1.json": "data/prompt_manifests/bluedot_state_dependence_v1.json",
    "bluedot_state_dependence_directions_v1.json": (
        "data/direction_manifests/bluedot_state_dependence_directions_v1.json"
    ),
    "bluedot_state_dependence_calibration_v1.json": (
        "data/calibration_plans/bluedot_state_dependence_calibration_v1.json"
    ),
}

# The internal content hashes the final-test resolution manifest cites. If a manifest were edited,
# its own hash field would no longer be the value the run recorded against it.
EXPECTED_HASHES = {
    "bluedot_state_dependence_v1.json": (
        "manifest_hash",
        "sha256:bf351c9d73042fcb3d0cdcb18247ded3411413d73000e047f1ba6ba9ab5b25b9",
    ),
    "bluedot_state_dependence_directions_v1.json": (
        "family_hash",
        "sha256:809fbb5b033da740a01574ad5a0ca48f34baca38eef1504a0d66bba8e2fb9138",
    ),
    "bluedot_state_dependence_calibration_v1.json": (
        "plan_hash",
        "sha256:a212c6e80f96db55e1aeb0b1879fef441aa6d893a73b7fc43a94142913f97877",
    ),
}

FROZEN_DOCUMENTS = {
    "preregistration_state_dependence.md": (
        "fac50ab937011f09e7a74a89f4a99cefc4ad600669d2ba1fc53556ef7c62e97b"
    ),
    "execution_decision_tree.md": (
        "9fe078c70f84a2aa36c4e8e6e02764a4737b2dad36f2d684f08a62874abc5a84"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(("name", "canonical"), sorted(COPIES.items()))
def test_the_protocol_copy_matches_the_artifact_the_code_loads(name: str, canonical: str) -> None:
    copy = PROTOCOL / name
    source = REPO_ROOT / canonical
    assert copy.exists(), f"{copy} is missing from the frozen protocol"
    assert source.exists(), f"{source} is missing"
    assert copy.read_bytes() == source.read_bytes(), (
        f"protocol/{name} has drifted from {canonical}; the protocol copy is a record of what was "
        f"frozen and must never be edited independently"
    )


@pytest.mark.parametrize(("name", "expected"), sorted(EXPECTED_HASHES.items()))
def test_the_frozen_manifest_still_carries_the_hash_the_run_cited(
    name: str, expected: tuple[str, str]
) -> None:
    field, value = expected
    payload = json.loads((PROTOCOL / name).read_text(encoding="utf-8"))
    assert payload[field] == value, (
        f"protocol/{name} records {field}={payload[field]}, but the final-test resolution manifest "
        f"cites {value}"
    )


@pytest.mark.parametrize(("name", "digest"), sorted(FROZEN_DOCUMENTS.items()))
def test_the_recovered_documents_match_their_recorded_digests(name: str, digest: str) -> None:
    path = PROTOCOL / name
    assert path.exists(), f"{path} is missing from the frozen protocol"
    assert _sha256(path) == digest, (
        f"protocol/{name} no longer matches the digest recorded in protocol/README.md; it was "
        f"recovered byte for byte from git history and should not have changed"
    )


def test_the_protocol_directory_holds_exactly_the_five_frozen_artifacts() -> None:
    """A sixth file here would mean the protocol quietly grew after the fact."""
    present = {path.name for path in PROTOCOL.iterdir() if path.is_file()}
    expected = set(COPIES) | set(FROZEN_DOCUMENTS) | {"README.md"}
    assert present == expected, f"unexpected change to protocol/: {present ^ expected}"
