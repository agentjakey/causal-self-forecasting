"""Seed control, deterministic seed derivation, and environment capture."""

from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

# Packages whose exact versions can change measured numbers. Recorded in every run manifest.
TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "peft",
    "accelerate",
    "safetensors",
    "numpy",
    "scikit-learn",
    "pandas",
    "pyarrow",
    "pydantic",
)

_SEED_MODULUS = 2**32


def derive_seed(*parts: Any) -> int:
    """Derive a stable 32-bit seed from arbitrary labels.

    Used so that per-trial randomness (candidate order, random directions) is a pure
    function of the run seed and the trial identity. That makes any single trial
    reproducible on its own, without replaying the whole run in order.
    """
    material = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, numpy, and torch if it is installed.

    Global seeding is a blunt instrument. Prefer explicit generators seeded with
    `derive_seed` for anything whose value ends up in an artifact.
    """
    seed = int(seed) % _SEED_MODULUS
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch = _import_torch()
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _import_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def package_versions() -> dict[str, str]:
    """Report installed versions of the tracked packages."""
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def git_state(repo_root: Path | None = None) -> dict[str, Any]:
    """Report the current commit and whether the working tree is dirty.

    A dirty tree is recorded rather than rejected. Refusing to run would be worse: it would
    push people toward throwaway commits. The run manifest carries the fact forward so a
    reader knows the code was not exactly the recorded commit.
    """
    root = repo_root or Path(__file__).resolve().parents[2]
    state: dict[str, Any] = {"commit": None, "dirty": None, "branch": None}
    try:
        state["commit"] = _git(root, "rev-parse", "HEAD")
        state["branch"] = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
        state["dirty"] = bool(_git(root, "status", "--porcelain"))
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        # Running from a tarball or without git on PATH is allowed.
        pass
    return state


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return result.stdout.strip()


def torch_environment() -> dict[str, Any]:
    """Report what torch can see, without claiming support that has not been measured."""
    torch = _import_torch()
    if torch is None:
        return {"available": False}

    info: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "mps_available": bool(getattr(torch.backends, "mps", None))
        and torch.backends.mps.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        info["devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    return info


def environment_snapshot() -> dict[str, Any]:
    """Capture everything needed to explain why a number came out the way it did."""
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": package_versions(),
        "torch": torch_environment(),
        "git": git_state(),
    }
