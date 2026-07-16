"""Guards against fabricated results reaching anything public.

The risk this project has to design against is not a dramatic fraud. It is the ordinary path
where a placeholder number gets typed into a dashboard to see how the layout looks, the
layout is fine, and the number is still there six weeks later with nobody remembering it was
invented.

Three guards:

1. The dashboard must not contain hard-coded benchmark-looking numbers. Every number it shows
   has to arrive from a verified export at runtime.
2. Exports must be structurally incapable of being published unverified.
3. Until a verified export exists, the dashboard's data directory must be empty or explicitly
   marked as containing none.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from causal_self_forecasting.config import repo_root
from causal_self_forecasting.schemas import (
    MetricValue,
    PublicDashboardRecord,
    ResultStatus,
)

DASHBOARD_SOURCE_SUFFIXES = {".ts", ".tsx", ".js", ".jsx", ".svelte", ".vue"}

# Metric-shaped identifiers. A literal number assigned to one of these in dashboard source is
# almost certainly a benchmark value that should have come from an export.
_METRIC_NAMES = (
    "brier",
    "brier_skill",
    "brierSkill",
    "auroc",
    "auprc",
    "ece",
    "mae",
    "rmse",
    "accuracy",
    "pearson",
    "spearman",
    "kendall",
    "ndcg",
    "coverage",
    "self_specificity_gap",
    "selfSpecificityGap",
    "delta_margin",
    "deltaMargin",
    "flip_rate",
    "flipRate",
    "n_trials",
    "nTrials",
    "sample_count",
    "sampleCount",
)

_ASSIGNMENT = re.compile(
    r"""["']?(?P<name>{names})["']?\s*[:=]\s*(?P<value>-?\d+\.?\d*)""".format(
        names="|".join(_METRIC_NAMES)
    ),
    re.IGNORECASE,
)


def _dashboard_sources() -> list[Path]:
    dashboard = repo_root() / "dashboard"
    if not dashboard.exists():
        return []
    skip = {"node_modules", ".next", "out", "dist", "coverage", ".vercel"}
    return [
        path
        for path in dashboard.rglob("*")
        if path.is_file()
        and path.suffix in DASHBOARD_SOURCE_SUFFIXES
        and not any(part in skip for part in path.parts)
    ]


def _public_exports() -> list[Path]:
    public = repo_root() / "results" / "public"
    if not public.exists():
        return []
    return sorted(public.glob("*.json"))


def test_dashboard_has_no_hard_coded_metrics() -> None:
    """A metric literal in dashboard source is a fabricated result until proven otherwise."""
    offenders: list[str] = []
    for path in _dashboard_sources():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith(("//", "*", "/*")):
                continue
            match = _ASSIGNMENT.search(line)
            if match:
                relative = path.relative_to(repo_root())
                offenders.append(
                    f"{relative}:{number}: {match.group('name')} = {match.group('value')}"
                )

    assert not offenders, (
        "dashboard source contains hard-coded metric values. Every displayed number must come "
        "from a verified export at runtime:\n" + "\n".join(offenders)
    )


def test_dashboard_data_directory_holds_only_verified_exports() -> None:
    """Anything in the dashboard's data directory must be a valid, verified export."""
    data_dir = repo_root() / "dashboard" / "public" / "data"
    if not data_dir.exists():
        pytest.skip("no dashboard data directory yet")

    for path in sorted(data_dir.glob("*.json")):
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        record = PublicDashboardRecord.model_validate(payload)
        assert record.commitments_verified, f"{path} is published but not verified"


def test_public_exports_are_valid_and_verified() -> None:
    for path in _public_exports():
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        record = PublicDashboardRecord.model_validate(payload)
        assert record.commitments_verified, f"{path} was exported without verified commitments"


def test_export_schema_rejects_unverified_runs() -> None:
    """The structural guarantee: an unverified run cannot be represented as an export."""
    with pytest.raises(ValidationError, match="did not verify"):
        PublicDashboardRecord(
            run_id="run-1",
            status=ResultStatus.PRELIMINARY,
            commitments_verified=False,
        )


def test_metrics_cannot_be_published_without_a_sample_count() -> None:
    """A point estimate with no n behind it is not a result."""
    with pytest.raises(ValidationError):
        MetricValue(value=0.42, n=0, ci_low=0.4, ci_high=0.44)


def test_metrics_require_an_uncertainty_interval() -> None:
    with pytest.raises(ValidationError):
        MetricValue(value=0.42, n=100)  # type: ignore[call-arg]


def test_no_verified_experiment_is_claimed_yet() -> None:
    """A tripwire for this project's current, honest state.

    No real experiment has been run: there is no model organism, no estimated direction, and
    no forecaster. If a verified export appears, this test fails on purpose. Deleting it is
    then the correct action, and it forces a deliberate look at whether the export really is
    backed by a real run before any of it is described as a result.
    """
    exports = _public_exports()
    assert not exports, (
        "a public export now exists: "
        f"{[str(path.relative_to(repo_root())) for path in exports]}. "
        "Confirm it comes from a real, verified run, update the claim-boundary docs, then "
        "delete this test."
    )
