"""Tests for canonical serialization and the hashing primitives.

The commitment protocol is only as good as these functions. If canonical JSON were not
deterministic, an honest forecast would fail to verify; if it ignored a field, a dishonest
edit would pass.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from causal_self_forecasting.hashing import (
    CanonicalizationError,
    append_jsonl,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    commitment_hash,
    hash_file,
    hash_object,
    new_salt,
    read_jsonl,
    salt_from_hex,
    salt_to_hex,
    verify_commitment,
    write_jsonl,
)


def test_canonical_json_is_key_order_independent() -> None:
    left = {"b": 1, "a": 2, "c": {"z": 1, "y": 2}}
    right = {"c": {"y": 2, "z": 1}, "a": 2, "b": 1}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)


def test_canonical_json_preserves_list_order() -> None:
    # Order is meaningful in a candidate list, so it must not be normalized away.
    assert canonical_json_bytes([1, 2, 3]) != canonical_json_bytes([3, 2, 1])


def test_canonical_json_has_no_incidental_whitespace() -> None:
    assert canonical_json_bytes({"a": 1, "b": [1, 2]}) == b'{"a":1,"b":[1,2]}'


def test_canonical_json_distinguishes_bool_from_int() -> None:
    # bool is a subclass of int in Python; conflating them would let True and 1 collide.
    assert canonical_json_bytes({"x": True}) != canonical_json_bytes({"x": 1})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_non_finite_floats(value: float) -> None:
    with pytest.raises(CanonicalizationError):
        canonical_json_bytes({"delta": value})


def test_canonical_json_rejects_non_string_keys() -> None:
    with pytest.raises(CanonicalizationError):
        canonical_json_bytes({1: "a"})


def test_canonical_json_rejects_unserializable_types() -> None:
    with pytest.raises(CanonicalizationError):
        canonical_json_bytes({"when": object()})


@settings(max_examples=100, deadline=None)
@given(
    st.recursive(
        st.one_of(
            st.none(),
            st.booleans(),
            st.integers(min_value=-(10**9), max_value=10**9),
            st.floats(allow_nan=False, allow_infinity=False, width=64),
            st.text(max_size=20),
        ),
        lambda children: st.one_of(
            st.lists(children, max_size=5),
            st.dictionaries(st.text(max_size=8), children, max_size=5),
        ),
        max_leaves=15,
    )
)
def test_canonical_json_is_stable_across_repeated_calls(value: object) -> None:
    assert canonical_json_bytes(value) == canonical_json_bytes(value)


def test_commitment_verifies_with_the_right_salt() -> None:
    payload = {"trial_id": "t1", "delta": -0.5}
    salt = new_salt()
    digest = commitment_hash(payload, salt)
    assert verify_commitment(payload, salt, digest)


def test_commitment_fails_when_the_payload_is_edited() -> None:
    """The whole point: an edited forecast must not verify against its commitment."""
    payload = {"trial_id": "t1", "delta_margin_mean": -0.5}
    salt = new_salt()
    digest = commitment_hash(payload, salt)

    tampered = {"trial_id": "t1", "delta_margin_mean": -0.6}
    assert not verify_commitment(tampered, salt, digest)


def test_commitment_fails_with_a_different_salt() -> None:
    payload = {"trial_id": "t1"}
    digest = commitment_hash(payload, new_salt())
    assert not verify_commitment(payload, new_salt(), digest)


def test_commitment_is_insensitive_to_key_order() -> None:
    salt = new_salt()
    left = commitment_hash({"a": 1, "b": 2}, salt)
    right = commitment_hash({"b": 2, "a": 1}, salt)
    assert left == right


def test_salts_are_unique() -> None:
    salts = {new_salt() for _ in range(50)}
    assert len(salts) == 50


def test_salt_round_trips_through_hex() -> None:
    salt = new_salt()
    assert salt_from_hex(salt_to_hex(salt)) == salt


def test_commitment_rejects_a_short_salt() -> None:
    with pytest.raises(ValueError, match="at least 16 bytes"):
        commitment_hash({"a": 1}, b"tooshort")


def test_commitment_rejects_a_non_bytes_salt() -> None:
    with pytest.raises(TypeError):
        commitment_hash({"a": 1}, "not bytes")  # type: ignore[arg-type]


def test_hash_object_is_prefixed_and_sized() -> None:
    digest = hash_object({"a": 1})
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64


def test_hash_file_matches_hash_of_its_bytes(tmp_path: Path) -> None:
    from causal_self_forecasting.hashing import sha256_hex

    target = tmp_path / "artifact.bin"
    payload = b"some bytes" * 1000
    target.write_bytes(payload)
    assert hash_file(target) == sha256_hex(payload)


def test_atomic_write_replaces_existing_content(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "file.json"
    atomic_write_json(target, {"version": 1})
    atomic_write_json(target, {"version": 2})
    assert target.read_bytes() == b'{"version":2}\n'


def test_atomic_write_leaves_no_temporary_files(tmp_path: Path) -> None:
    atomic_write_bytes(tmp_path / "a.bin", b"x")
    assert [path.name for path in tmp_path.iterdir()] == ["a.bin"]


def test_atomic_write_does_not_clobber_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "file.json"
    atomic_write_json(target, {"good": 1})
    with pytest.raises(CanonicalizationError):
        atomic_write_json(target, {"bad": float("nan")})
    # The original survives, and no partial file is left behind.
    assert target.read_bytes() == b'{"good":1}\n'
    assert [path.name for path in tmp_path.iterdir()] == ["file.json"]


def test_jsonl_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "records.jsonl"
    records = [{"i": index, "label": f"row{index}"} for index in range(3)]
    write_jsonl(target, records)
    assert list(read_jsonl(target)) == records


def test_append_jsonl_adds_without_rewriting(tmp_path: Path) -> None:
    target = tmp_path / "records.jsonl"
    append_jsonl(target, {"i": 0})
    append_jsonl(target, {"i": 1})
    assert list(read_jsonl(target)) == [{"i": 0}, {"i": 1}]


def test_read_jsonl_reports_the_failing_line(tmp_path: Path) -> None:
    target = tmp_path / "broken.jsonl"
    target.write_text('{"a":1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        list(read_jsonl(target))


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    target = tmp_path / "gaps.jsonl"
    target.write_text('{"a":1}\n\n{"a":2}\n', encoding="utf-8")
    assert list(read_jsonl(target)) == [{"a": 1}, {"a": 2}]


def test_float_precision_survives_canonicalization() -> None:
    value = {"delta": 0.1 + 0.2}
    restored = float(canonical_json_bytes(value).decode().split(":")[1].rstrip("}"))
    assert math.isclose(restored, 0.30000000000000004, rel_tol=0, abs_tol=0)
