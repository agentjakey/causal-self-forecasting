"""The post hoc decomposition must be exact, deterministic, and honestly labelled.

It is not preregistered and nothing in the paper rests on it, which is precisely why it needs
tests: an unplanned analysis that quietly drifted or acquired inferential language would be the
easiest way for this project to overstate itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causal_self_forecasting.state_audit.exploratory import (
    RECONSTRUCTION_TOLERANCE,
    ExploratoryError,
    decompose_direction_structure,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = REPO_ROOT / "results" / "public" / "bluedot-v0.1"
PUBLISHED = REPO_ROOT / "paper" / "data" / "exploratory_direction_decomposition.json"

pytestmark = pytest.mark.skipif(
    not BUNDLE.is_dir(), reason="the published bundle is not present in this checkout"
)


@pytest.fixture(scope="module")
def decomposition() -> dict:
    return decompose_direction_structure(BUNDLE)


def test_the_components_reconstruct_the_total_sum_of_squares(decomposition: dict) -> None:
    parts = decomposition["components"]
    rebuilt = (
        parts["direction_main_effects"]
        + parts["prompt_main_effects"]
        + parts["prompt_by_direction_residual"]
    )
    assert abs(rebuilt - parts["total"]) <= RECONSTRUCTION_TOLERANCE
    assert decomposition["reconstruction_error"] <= RECONSTRUCTION_TOLERANCE


def test_the_shares_sum_to_one(decomposition: dict) -> None:
    assert sum(decomposition["shares_of_total"].values()) == pytest.approx(1.0, abs=1e-12)


def test_the_decomposition_is_deterministic(decomposition: dict) -> None:
    again = decompose_direction_structure(BUNDLE)
    assert again == decomposition
    assert again["decomposition_hash"] == decomposition["decomposition_hash"]


def test_the_design_is_the_balanced_final_test_matrix(decomposition: dict) -> None:
    assert decomposition["prompt_count"] == 32
    assert decomposition["direction_count"] == 16
    assert decomposition["observation_count"] == 32 * 16
    assert decomposition["target_name"] == "delta_clean_top_margin"


def test_it_cites_the_frozen_analysis_and_its_sources(decomposition: dict) -> None:
    assert (
        decomposition["cited_analysis_hash"]
        == "sha256:46e88d3a629c78009787f88fadc5c821e8c318e3ee301923a236171cde1ece4d"
    )
    sources = decomposition["source_artifact_hashes"]
    assert set(sources) == {
        "state_audit_analysis.json",
        "state_audit_candidate_sets.jsonl",
        "state_audit_observations.jsonl",
        "state_audit_pair_scores.jsonl",
    }
    for name, digest in sources.items():
        assert digest.startswith("sha256:")
        assert (BUNDLE / name).exists()


def test_it_is_labelled_post_hoc_and_non_inferential(decomposition: dict) -> None:
    assert decomposition["preregistered"] is False
    assert decomposition["inferential"] is False
    assert "conceived after" in decomposition["note"].lower()


def test_it_carries_no_inferential_quantity(decomposition: dict) -> None:
    """No p-value, no interval, no test statistic may appear anywhere in the payload."""
    flat = json.dumps(decomposition).lower()
    for banned in ("p_value", "pvalue", "ci_low", "ci_high", "confidence", "significan", "f_stat"):
        assert banned not in flat, f"the descriptive payload contains {banned!r}"


def test_the_intervention_only_prediction_is_one_value_per_signed_direction(
    decomposition: dict,
) -> None:
    """The paper's claim that this method is a per-direction table, checked against artifacts."""
    summaries = decomposition["direction_summaries"]
    assert len(summaries) == 16
    assert len({s["signed_direction_id"] for s in summaries}) == 16
    assert len({s["direction_ref"] for s in summaries}) == 8
    assert sorted({s["sign"] for s in summaries}) == [-1, 1]
    for summary in summaries:
        assert summary["prompt_count"] == 32
    # decompose_direction_structure raises if a prediction varies within a signed direction, so
    # reaching here at all is the assertion; this pins the count the paper quotes.
    assert len({s["intervention_only_prediction"] for s in summaries}) == 16


def test_only_final_test_outcomes_are_read(tmp_path: Path) -> None:
    """Remove the final-test outcomes and the analysis must not fall back to anything else."""
    import shutil

    copy = tmp_path / "bundle"
    shutil.copytree(BUNDLE, copy)
    (copy / "state_audit_observations.jsonl").unlink()
    with pytest.raises(ExploratoryError, match="missing state_audit_observations"):
        decompose_direction_structure(copy)


def test_an_unbalanced_matrix_is_refused(tmp_path: Path) -> None:
    """The exact partition is only valid for a balanced design, so refuse anything else."""
    import shutil

    copy = tmp_path / "bundle"
    shutil.copytree(BUNDLE, copy)
    target = copy / "state_audit_observations.jsonl"
    rows = target.read_text(encoding="utf-8").splitlines()
    target.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ExploratoryError, match="not balanced"):
        decompose_direction_structure(copy)


@pytest.mark.skipif(not PUBLISHED.exists(), reason="decomposition has not been written yet")
def test_the_published_file_matches_a_fresh_computation(decomposition: dict) -> None:
    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    assert published == decomposition


@pytest.mark.skipif(not PUBLISHED.exists(), reason="decomposition has not been written yet")
def test_the_numbers_quoted_in_the_documents_match_the_artifact() -> None:
    """Guards against the write-ups drifting from the file they describe."""
    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    shares = published["shares_of_total"]
    quoted = {
        "direction_main_effects": 0.130,
        "prompt_main_effects": 0.030,
        "prompt_by_direction_residual": 0.840,
    }
    for key, value in quoted.items():
        assert shares[key] == pytest.approx(value, abs=0.0005), (
            f"the documents quote {value:.1%} for {key} but the artifact holds {shares[key]:.4%}"
        )


@pytest.mark.skipif(not PUBLISHED.exists(), reason="decomposition has not been written yet")
def test_the_paper_and_report_quote_the_same_decomposition() -> None:
    """The prose and the artifact must not drift apart in either direction."""
    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    parts = published["components"]
    paper = (REPO_ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    report = (REPO_ROOT / "REPORT.md").read_text(encoding="utf-8")

    for document, name in ((paper, "paper/main.tex"), (report, "REPORT.md")):
        assert "13.0" in document, f"{name} does not quote the direction share"
        assert "84.0" in document, f"{name} does not quote the residual share"
        assert "post hoc" in document.lower(), f"{name} does not label the analysis post hoc"

    for value in (
        parts["direction_main_effects"],
        parts["prompt_main_effects"],
        parts["prompt_by_direction_residual"],
        parts["total"],
    ):
        rendered = f"{value:.6f}"
        assert rendered in paper, f"paper/main.tex does not quote {rendered}"
        assert rendered in report, f"REPORT.md does not quote {rendered}"


@pytest.mark.skipif(not PUBLISHED.exists(), reason="decomposition has not been written yet")
def test_the_documents_state_the_decomposition_was_not_preregistered() -> None:
    """Each document must disclaim preregistration in so many words, not merely by omission."""
    for relative in ("paper/main.tex", "REPORT.md"):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8").lower()
        assert "post hoc" in text, f"{relative} never labels the analysis post hoc"
        assert "not preregistered" in text, (
            f"{relative} never states that the decomposition was not preregistered"
        )
        assert "conceived after" in text, f"{relative} does not say when the analysis was conceived"
