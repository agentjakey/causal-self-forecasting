"""The figure suite must be drawn from the verified bundle and from nothing else.

These run against the real published bundle, because that is the artifact the figures ship with and
the only one whose numbers the captions claim. They are cheap: the bundle is small and no model is
involved.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from causal_self_forecasting.state_audit.plots import (
    EXPECTED_ANALYSIS_HASH,
    FIGURE_NAMES,
    METHOD_LABELS,
    PlotError,
    load_bundle,
    plot_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = REPO_ROOT / "results" / "public" / "bluedot-v0.1"

pytestmark = pytest.mark.skipif(
    not BUNDLE.is_dir(), reason="the published bundle is not present in this checkout"
)


@pytest.fixture(scope="module")
def drawn(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    output = tmp_path_factory.mktemp("figures")
    return output, plot_bundle(BUNDLE, output)


def test_the_command_runs_from_the_public_bundle_alone(drawn) -> None:
    _, report = drawn
    assert report["bundle_verified"] is True
    assert report["analysis_hash"] == EXPECTED_ANALYSIS_HASH
    assert report["figure_count"] == len(FIGURE_NAMES)


def test_every_expected_figure_is_produced_in_both_formats(drawn) -> None:
    output, _ = drawn
    for name in FIGURE_NAMES:
        for suffix in (".png", ".pdf"):
            path = output / f"{name}{suffix}"
            assert path.exists(), f"{path.name} was not written"
            assert path.stat().st_size > 5_000, f"{path.name} is suspiciously small"


def test_plotting_is_refused_when_the_bundle_does_not_verify(tmp_path: Path) -> None:
    """Verification is a precondition, not a flag: a tampered bundle produces no figures."""
    import shutil

    copy = tmp_path / "bundle"
    shutil.copytree(BUNDLE, copy)
    target = copy / "state_audit_observations.jsonl"
    rows = target.read_text(encoding="utf-8").splitlines()
    payload = json.loads(rows[0])
    payload["delta_clean_top_margin"] = float(payload["delta_clean_top_margin"]) + 1.0
    rows[0] = json.dumps(payload, sort_keys=True)
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")

    output = tmp_path / "figures"
    with pytest.raises(PlotError, match="does not verify"):
        plot_bundle(copy, output)
    assert not output.exists() or not list(output.glob("*.png"))


def test_plotting_is_refused_for_a_different_analysis(tmp_path: Path) -> None:
    """A figure captioned with the wrong study is worse than no figure."""
    import shutil

    copy = tmp_path / "bundle"
    shutil.copytree(BUNDLE, copy)
    analysis_path = copy / "state_audit_analysis.json"
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    payload["analysis_hash"] = "sha256:" + "0" * 64
    analysis_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(PlotError, match="refusing to caption"):
        load_bundle(copy)


def test_the_annotated_numbers_match_the_stored_artifacts() -> None:
    """Every value a figure prints must be the value the analysis recorded."""
    data = load_bundle(BUNDLE)
    stored = json.loads((BUNDLE / "state_audit_analysis.json").read_text(encoding="utf-8"))

    for summary in stored["method_summaries"]:
        key = f"{summary['method_id']}:{summary['state_condition']}:{summary['condition_index']}"
        assert data.summary(key)["mae"] == pytest.approx(summary["mae"], abs=0.0)

    comparisons = {c["name"]: c for c in stored["primary_comparisons"]}
    for comparison in data.analysis["primary_comparisons"]:
        recorded = comparisons[comparison["name"]]
        for field in ("point_estimate", "ci_low", "ci_high"):
            assert comparison[field] == recorded[field]

    # The exact preregistered endpoints, unrounded.
    visible = comparisons["visible_information minus true_state"]
    assert visible["point_estimate"] == pytest.approx(-0.013725250359079288, abs=0.0)
    assert visible["ci_high"] == pytest.approx(2.8966819969343353e-05, abs=0.0)

    shuffled = data.shuffled_maes()
    assert len(shuffled) == 10
    assert min(shuffled) == pytest.approx(stored["permutation_band"]["min_mae"], abs=0.0)
    assert max(shuffled) == pytest.approx(stored["permutation_band"]["max_mae"], abs=0.0)


def test_the_calibration_panel_numbers_match_the_ratio_summaries() -> None:
    data = load_bundle(BUNDLE)
    rates = {
        row["norm_ratio"]: 100.0 * row["flip_count"] / row["observed_non_noop_observations"]
        for row in data.ratio_summaries
    }
    assert rates[0.02] == pytest.approx(3.7109375)
    assert rates[0.05] == pytest.approx(7.2265625)
    assert rates[0.10] == pytest.approx(21.6796875)
    assert rates[0.20] == pytest.approx(61.5234375)
    assert rates[0.40] == pytest.approx(79.8828125)
    assert data.decision["selected_norm_ratio"] == 0.02
    for row in data.ratio_summaries:
        assert row["passed"] is (row["norm_ratio"] in (0.02, 0.05))


def test_no_figure_label_leaks_a_raw_internal_name() -> None:
    for label in METHOD_LABELS.values():
        assert ":" not in label
        assert "_ridge" not in label
        assert "state_bilinear" not in label


def test_plotting_names_no_model_machinery() -> None:
    import causal_self_forecasting.state_audit.plots as plots

    source = Path(plots.__file__).read_text(encoding="utf-8")
    for forbidden in ("models.loader", "load_model", "transformers", "AutoModel"):
        assert forbidden not in source, f"plots.py references {forbidden}"


def test_plotting_loads_no_model_in_a_fresh_interpreter(tmp_path: Path) -> None:
    """Run the whole command in a clean process and assert the heavy stack never got imported.

    Checking `sys.modules` inside the test process would prove nothing: another test will already
    have imported torch. A subprocess is the only honest way to ask this question.
    """
    script = (
        "import sys, pathlib\n"
        "from causal_self_forecasting.state_audit.plots import plot_bundle\n"
        f"plot_bundle(pathlib.Path({str(BUNDLE)!r}), pathlib.Path({str(tmp_path)!r}))\n"
        "leaked = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m.split('.')[0] in {'torch', 'transformers', 'accelerate', 'peft'}\n"
        "    or m.endswith('causal_self_forecasting.models.loader')\n"
        ")\n"
        "print('LEAKED=' + ','.join(leaked))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    leaked_line = next(line for line in result.stdout.splitlines() if line.startswith("LEAKED="))
    leaked = [name for name in leaked_line.removeprefix("LEAKED=").split(",") if name]
    assert leaked == [], f"plotting imported model machinery: {leaked}"
    assert list(tmp_path.glob("*.png")), "the subprocess wrote no figures"
