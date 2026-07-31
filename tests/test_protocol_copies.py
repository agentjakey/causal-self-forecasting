"""The frozen protocol must still be the thing the study was run against.

There is exactly one tracked copy of each frozen JSON manifest, at the canonical path the code
loads it from. These tests check that each still carries the content hash the final-test resolution
manifest cites, and that the two recovered documents under `protocol/` still match the digests
recorded in `protocol/README.md`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / "protocol"

# Canonical path -> (field carrying the artifact's own content hash, the value the run cited).
FROZEN_MANIFESTS = {
    "data/prompt_manifests/bluedot_state_dependence_v1.json": (
        "manifest_hash",
        "sha256:bf351c9d73042fcb3d0cdcb18247ded3411413d73000e047f1ba6ba9ab5b25b9",
    ),
    "data/direction_manifests/bluedot_state_dependence_directions_v1.json": (
        "family_hash",
        "sha256:809fbb5b033da740a01574ad5a0ca48f34baca38eef1504a0d66bba8e2fb9138",
    ),
    "data/calibration_plans/bluedot_state_dependence_calibration_v1.json": (
        "plan_hash",
        "sha256:a212c6e80f96db55e1aeb0b1879fef441aa6d893a73b7fc43a94142913f97877",
    ),
    # The resolution manifest calls this `projection_hash`; the record itself calls it
    # `matrix_hash`, because it is the hash of the matrix rather than of the record.
    "data/projections/bluedot_state_dependence_projection_v1.json": (
        "matrix_hash",
        "sha256:0e4207cd69e52567fa703e95811772b3d31dabac3622d1fb1324133bcdad0328",
    ),
}

# The projection matrix itself. Kept tracked because the frozen record records its hash and every
# forecast cites it; regenerating it is deterministic but this is the artifact that was used.
PROJECTION_MATRIX = "data/projections/bluedot_state_dependence_projection_v1.npz"
PROJECTION_MATRIX_HASH = "sha256:8f697cb84b34c6ffd9abb919a81bba6e35efe50782849767a1b8ee4b7dc68cbc"

# Recovered from git history; digests are recorded in protocol/README.md.
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


@pytest.mark.parametrize(("canonical", "expected"), sorted(FROZEN_MANIFESTS.items()))
def test_the_frozen_manifest_still_carries_the_hash_the_run_cited(
    canonical: str, expected: tuple[str, str]
) -> None:
    field, value = expected
    path = REPO_ROOT / canonical
    assert path.exists(), f"{canonical} is missing"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[field] == value, (
        f"{canonical} records {field}={payload[field]}, but the final-test resolution manifest "
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


def test_the_projection_matrix_matches_the_hash_its_record_carries() -> None:
    """The .npz matches an ignore pattern, so it is force-tracked. This is why it is worth it."""
    path = REPO_ROOT / PROJECTION_MATRIX
    assert path.exists(), f"{PROJECTION_MATRIX} is missing"
    assert f"sha256:{_sha256(path)}" == PROJECTION_MATRIX_HASH


def test_there_is_exactly_one_working_copy_of_each_frozen_manifest() -> None:
    """A second working copy is a thing to drift. The canonical path under data/ is the only one.

    `results/public/` is excluded on purpose. A published bundle is a distribution artifact whose
    whole point is standing alone, so it carries its own copy of the direction family; that copy is
    checksummed inside the bundle and is replaced wholesale whenever the bundle is rebuilt. What
    must not happen is a second *editable* copy in the working tree, which is what this checks.
    """
    ignored = {".venv", ".git", "public", ".pytest_cache", ".ruff_cache"}
    for canonical in FROZEN_MANIFESTS:
        name = Path(canonical).name
        copies = [p for p in REPO_ROOT.rglob(name) if ignored.isdisjoint(p.parts)]
        assert len(copies) == 1, f"{name} exists at {len(copies)} working paths: {copies}"


def test_the_bundle_copy_of_the_direction_family_is_byte_identical() -> None:
    """The bundle's distribution copy must be the same bytes as the canonical artifact."""
    bundle_copy = (
        REPO_ROOT / "results/public/bluedot-v0.1/bluedot_state_dependence_directions_v1.json"
    )
    canonical = REPO_ROOT / "data/direction_manifests/bluedot_state_dependence_directions_v1.json"
    if not bundle_copy.exists():
        pytest.skip("the published bundle is not present in this checkout")
    assert bundle_copy.read_bytes() == canonical.read_bytes()


def test_the_protocol_directory_holds_only_the_recovered_documents() -> None:
    """Anything else here means the protocol quietly grew after the fact."""
    present = {path.name for path in PROTOCOL.iterdir() if path.is_file()}
    assert present == set(FROZEN_DOCUMENTS) | {"README.md"}, (
        f"unexpected protocol/ contents: {present}"
    )
