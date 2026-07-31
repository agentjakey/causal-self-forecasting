"""The public model-free replay bundle. Loads no model.

A third party should be able to check the headline result without the weights, without the hidden
states, and without trusting this repository's run directories. That means the bundle has to be
self-sufficient: every file the replay reads is inside it, and the replay runs through exactly the
same function the run directory uses.

Two rules govern what goes in.

**Allowlist, not denylist.** Files are copied because they appear in `BUNDLE_CONTENTS`, not because
they failed to match an exclusion pattern. A new artifact appearing in a run directory therefore
stays out of the bundle until someone adds it deliberately.

**The denylist is a second gate, not the first one.** `_refuse_forbidden` re-checks every path on
the way out. It should never fire, because the allowlist already decided. It exists so that a
careless future edit to the allowlist fails loudly instead of publishing a hidden-state array.

On salts: the pre-reveal salt files under `private_payloads/salts/` are excluded. The salts
themselves are *not* secret after resolution and are published inside `selection_reveals.jsonl`,
because a commitment hash cannot be checked without its salt. What must never ship is the salt
directory as it stood *before* the reveal, since that is the state in which knowing a salt would
have allowed a forecast to be rewritten to match an outcome.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..hashing import atomic_write_json, hash_file, read_json
from ..logging_utils import info
from ..paths import (
    FORECAST_COMMITMENTS,
    FORECASTS,
    SELECTION_REVEALS,
    STATE_AUDIT_ANALYSIS,
    STATE_AUDIT_CANDIDATE_SETS,
    STATE_AUDIT_CLEAN_PASS,
    STATE_AUDIT_COMMITMENT_SUMMARY,
    STATE_AUDIT_DECISION,
    STATE_AUDIT_FIGURES,
    STATE_AUDIT_METHOD_SUMMARY,
    STATE_AUDIT_OBSERVATIONS,
    STATE_AUDIT_PAIR_SCORES,
    STATE_AUDIT_PAIRING,
    STATE_AUDIT_PREDICTORS,
    STATE_AUDIT_PROMPT_SCORES,
    STATE_AUDIT_RATIO_SUMMARIES,
    STATE_AUDIT_REFERENCE_NORM,
    STATE_AUDIT_RESOLUTION,
    STATE_AUDIT_TRANSFORM_FITS,
    public_dir,
    run_dir,
)
from .analyze import AnalysisError, replay_from_directory

BUNDLE_ID = "bluedot-v0.1"
BUNDLE_ALGORITHM_VERSION = "bluedot_public_bundle_v1.0"
CHECKSUM_FILE = "CHECKSUMS.sha256"
BUNDLE_MANIFEST = "bundle_manifest.json"

# Never publish, whatever the allowlist says. Model weights, raw residual streams, and the
# pre-reveal salt directory.
FORBIDDEN_SUFFIXES = (".npz", ".npy", ".pt", ".pth", ".ckpt", ".safetensors", ".salt")
FORBIDDEN_PARTS = ("private_payloads", "salts")


@dataclass(frozen=True)
class BundleEntry:
    """One file to publish: where it comes from, what it is called, and why it is here."""

    source_run: str
    filename: str
    category: str
    required: bool = True


# Every file in the bundle, and nothing else. `source_run` is a key into the run-id mapping passed
# to `build_public_bundle`, not a literal run id, so the same allowlist works for the fixture runs
# the tests build.
BUNDLE_CONTENTS: tuple[BundleEntry, ...] = (
    # The result of record and the manifest that proves how it was produced.
    BundleEntry("final_test", STATE_AUDIT_ANALYSIS, "analysis"),
    BundleEntry("final_test", STATE_AUDIT_RESOLUTION, "resolution"),
    # Score tables, aggregated and pair level.
    BundleEntry("final_test", STATE_AUDIT_METHOD_SUMMARY, "scores_aggregated"),
    BundleEntry("final_test", STATE_AUDIT_PROMPT_SCORES, "scores_prompt"),
    BundleEntry("final_test", STATE_AUDIT_PAIR_SCORES, "scores_pair"),
    # The sealed predictions, their commitments, and the reveals that open them.
    BundleEntry("final_test", FORECASTS, "forecasts"),
    BundleEntry("final_test", FORECAST_COMMITMENTS, "commitments"),
    BundleEntry("final_test", SELECTION_REVEALS, "reveals"),
    BundleEntry("final_test", STATE_AUDIT_COMMITMENT_SUMMARY, "commitments"),
    # The outcomes the forecasts are scored against. Required by the replay.
    BundleEntry("final_test", STATE_AUDIT_OBSERVATIONS, "outcomes"),
    # What was intervened, and the clean baseline every delta is measured against. Both carry
    # hashes and model outputs only, no residual streams.
    BundleEntry("final_test", STATE_AUDIT_CANDIDATE_SETS, "stimulus"),
    BundleEntry("final_test", STATE_AUDIT_CLEAN_PASS, "clean_baseline"),
    # How the wrong-state control and the ten derangements were built.
    BundleEntry("final_test", STATE_AUDIT_PAIRING, "fit_provenance"),
    # Fit provenance. These record which prompts each transform and predictor was fitted on and
    # the hash of the fitted object; they contain no coefficients and no state arrays.
    BundleEntry("training", STATE_AUDIT_TRANSFORM_FITS, "fit_provenance"),
    BundleEntry("training", STATE_AUDIT_PREDICTORS, "fit_provenance"),
    # The frozen stimulus decision and the evidence behind it.
    BundleEntry("calibration", STATE_AUDIT_DECISION, "calibration"),
    BundleEntry("calibration", STATE_AUDIT_RATIO_SUMMARIES, "calibration"),
    BundleEntry("calibration", STATE_AUDIT_REFERENCE_NORM, "calibration"),
)

EXCLUDED_AND_WHY: tuple[tuple[str, str], ...] = (
    ("state_audit_states.npz", "raw layer-13 residual streams"),
    ("state_audit_state_refs.jsonl", "pointers into the excluded state array"),
    (
        "private_payloads/salts/",
        "pre-reveal commitment salts; the post-reveal salts ship in the reveals",
    ),
    ("run.log.jsonl", "local execution log, not evidence"),
    ("model weights", "never redistributed; Gemma remains under Google's license"),
    ("other runs", "smoke, determinism, and benchmark runs are not part of this result"),
)


class BundleError(RuntimeError):
    """Raised when the bundle cannot be built or does not verify."""


def bundle_dir(bundle_id: str = BUNDLE_ID) -> Path:
    return public_dir() / bundle_id


def _refuse_forbidden(paths: Iterable[Path]) -> None:
    """Second gate. Should never fire, because the allowlist already decided."""
    for path in paths:
        parts = {part.lower() for part in path.parts}
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise BundleError(f"refusing to publish {path.name}: {path.suffix} is never published")
        for forbidden in FORBIDDEN_PARTS:
            if forbidden in parts:
                raise BundleError(f"refusing to publish {path}: it sits under '{forbidden}'")


def _checksum_lines(directory: Path) -> list[str]:
    """Every published file, hashed, in the format `sha256sum -c` accepts."""
    lines: list[str] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == CHECKSUM_FILE:
            continue
        digest = hash_file(path).removeprefix("sha256:")
        lines.append(f"{digest}  {path.relative_to(directory).as_posix()}")
    return lines


def build_public_bundle(
    run_ids: dict[str, str],
    bundle_id: str = BUNDLE_ID,
    destination: Path | None = None,
) -> dict[str, Any]:
    """Copy the allowlisted artifacts into a public bundle and checksum it. Loads no model.

    Nothing is recomputed and nothing is edited on the way through: every file is copied byte for
    byte, so a hash taken from the bundle is a hash of the artifact that was produced by the run.
    """
    target = destination if destination is not None else bundle_dir(bundle_id)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    published: list[dict[str, Any]] = []
    missing: list[str] = []
    for entry in BUNDLE_CONTENTS:
        run_id = run_ids.get(entry.source_run)
        if run_id is None:
            raise BundleError(f"no run id supplied for '{entry.source_run}'")
        source = run_dir(run_id) / entry.filename
        if not source.exists():
            if entry.required:
                missing.append(f"{run_id}/{entry.filename}")
            continue
        _refuse_forbidden([source])
        shutil.copy2(source, target / entry.filename)
        published.append(
            {
                "category": entry.category,
                "path": entry.filename,
                "sha256": hash_file(target / entry.filename),
                "source_run_id": run_id,
                "source_role": entry.source_run,
            }
        )

    if missing:
        raise BundleError(f"cannot build the bundle, these artifacts are missing: {missing}")

    figures_source = run_dir(run_ids["final_test"]) / STATE_AUDIT_FIGURES
    if figures_source.is_dir():
        figures_target = target / STATE_AUDIT_FIGURES
        figures_target.mkdir(parents=True, exist_ok=True)
        for figure in sorted(figures_source.glob("*.png")):
            _refuse_forbidden([figure])
            shutil.copy2(figure, figures_target / figure.name)
            published.append(
                {
                    "category": "figure",
                    "path": f"{STATE_AUDIT_FIGURES}/{figure.name}",
                    "sha256": hash_file(figures_target / figure.name),
                    "source_run_id": run_ids["final_test"],
                    "source_role": "final_test",
                }
            )

    _refuse_forbidden(p for p in target.rglob("*") if p.is_file())

    analysis = read_json(target / STATE_AUDIT_ANALYSIS)
    resolution = read_json(target / STATE_AUDIT_RESOLUTION)
    manifest: dict[str, Any] = {
        "algorithm_version": BUNDLE_ALGORITHM_VERSION,
        "analysis_hash": analysis["analysis_hash"],
        "bundle_id": bundle_id,
        "excluded": [{"path": path, "reason": reason} for path, reason in EXCLUDED_AND_WHY],
        # `file_count` counts the published artifacts listed in `files`. The checksum manifest
        # covers one more: this manifest itself, which is written after the list is built and so
        # cannot appear inside it.
        "checksummed_file_count": len(published) + 1,
        "file_count": len(published),
        "files": published,
        "global_alpha": resolution["global_alpha"],
        "layer": resolution["layer"],
        "model_id": resolution["model_id"],
        "model_revision": resolution["model_revision"],
        "norm_ratio": resolution["norm_ratio"],
        "notes": (
            "Model-free replay bundle. Every file was copied byte for byte from a verified run "
            "directory. No model weights, no residual-stream arrays, and no pre-reveal salt files "
            "are included; the post-reveal salts are inside the reveals, because a commitment "
            "hash cannot be checked without its salt."
        ),
        "resolution_manifest_hash": resolution["manifest_hash"],
        "run_ids": dict(sorted(run_ids.items())),
        "schema_version": "1.0",
        "scientific_result": True,
        "study_id": resolution["study_id"],
        "target_name": resolution["target_name"],
    }
    atomic_write_json(target / BUNDLE_MANIFEST, manifest)

    lines = _checksum_lines(target)
    (target / CHECKSUM_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    info("public bundle written", bundle_id=bundle_id, files=len(lines), path=str(target))
    return {
        "bundle_id": bundle_id,
        "path": str(target),
        "published_file_count": len(published),
        "checksummed_file_count": len(lines),
        "analysis_hash": manifest["analysis_hash"],
        "checksum_file": CHECKSUM_FILE,
    }


def verify_bundle(directory: Path) -> dict[str, Any]:
    """Recompute every checksum in the bundle and report mismatches. Loads no model."""
    checksum_path = directory / CHECKSUM_FILE
    if not checksum_path.exists():
        raise BundleError(f"no checksum manifest at {checksum_path}")

    recorded: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        recorded[name] = digest

    failures: list[str] = []
    for name, digest in sorted(recorded.items()):
        path = directory / name
        if not path.exists():
            failures.append(f"{name} is listed in {CHECKSUM_FILE} but missing from the bundle")
            continue
        actual = hash_file(path).removeprefix("sha256:")
        if actual != digest:
            failures.append(f"{name} hashes to {actual}, the manifest records {digest}")

    on_disk = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != CHECKSUM_FILE
    }
    for extra in sorted(on_disk - set(recorded)):
        failures.append(f"{extra} is in the bundle but not in {CHECKSUM_FILE}")

    return {"checked": len(recorded), "failures": failures, "valid": not failures}


def replay_bundle(directory: Path) -> dict[str, Any]:
    """Verify the bundle's checksums, then replay the analysis from it. Loads no model.

    Runs through `replay_from_directory`, the same function the run directory uses, so the bundle
    cannot pass a weaker check than the run it came from.
    """
    if not directory.is_dir():
        raise BundleError(f"no bundle at {directory}")
    integrity = verify_bundle(directory)
    if not integrity["valid"]:
        return {
            "bundle_path": str(directory),
            "checksums_checked": integrity["checked"],
            "checksums_valid": False,
            "valid": False,
            "failures": integrity["failures"],
            "notes": "Checksums failed, so the analysis was not replayed.",
        }

    try:
        report = replay_from_directory(directory, source=directory.name)
    except AnalysisError as error:
        raise BundleError(f"the bundle does not contain a replayable analysis: {error}") from error

    report["bundle_path"] = str(directory)
    report["checksums_checked"] = integrity["checked"]
    report["checksums_valid"] = True
    report["notes"] = (
        "Model-free replay from the public bundle. Every published file was checksummed, then "
        "every summary and both primary comparisons were recomputed from the bundle's own "
        "forecasts and outcomes and compared to the stored analysis. No model was loaded and no "
        "file outside the bundle was read."
    )
    return report


__all__ = [
    "BUNDLE_ALGORITHM_VERSION",
    "BUNDLE_CONTENTS",
    "BUNDLE_ID",
    "BUNDLE_MANIFEST",
    "CHECKSUM_FILE",
    "EXCLUDED_AND_WHY",
    "BundleEntry",
    "BundleError",
    "build_public_bundle",
    "bundle_dir",
    "replay_bundle",
    "verify_bundle",
]
