"""Shared test fixtures.

Tests never touch the real `results/` or `artifacts/` directories. Everything that writes is
redirected into `tmp_path`, so running the suite cannot damage a real run or, worse, leave
behind artifacts that later get mistaken for one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from causal_self_forecasting import paths


@pytest.fixture
def isolated_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect run output into a temporary directory."""
    target = tmp_path / "runs"
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(paths, "runs_dir", lambda: target)
    return target


@pytest.fixture
def isolated_directions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    target = tmp_path / "directions"
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(paths, "directions_dir", lambda: target)
    return target
