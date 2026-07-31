"""Publication figures, drawn from the published bundle. Loads no model.

Every number here is read from a file inside `results/public/bluedot-v0.1/`. Nothing is refitted,
resampled, or recomputed except the arithmetic needed to draw a point that is already implied by a
stored value, and the module refuses to draw anything until the bundle's checksums and its stored
analysis have both been verified.

Three rules the figures follow, because a figure is where a preregistered result is most easily
overstated:

* **No unplanned inference.** The only intervals drawn are the two preregistered paired bootstrap
  intervals, taken verbatim from the analysis record. Method rows get no error bars, because none
  were computed for them.
* **The shuffled-state controls are descriptive.** They are drawn as ten individual points and
  labelled as such, never as a distribution, a band with a tail, or a reference null.
* **Higher MAE is not "worse".** Axis labels and captions say point estimate. No figure marks a
  difference as significant that no test was run for.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter, NullLocator

from ..hashing import read_json, read_jsonl
from ..logging_utils import info

PLOT_ALGORITHM_VERSION = "bluedot_bundle_plots_v1.0"

# The analysis this figure suite describes. A bundle carrying a different analysis is refused
# rather than silently plotted, so a figure cannot end up captioned with the wrong study.
EXPECTED_ANALYSIS_HASH = "sha256:46e88d3a629c78009787f88fadc5c821e8c318e3ee301923a236171cde1ece4d"

DPI = 300

# Restrained and colour-blind safe. One accent for the true-state condition, one for controls,
# one neutral for everything else.
INK = "#1b1b1b"
MUTED = "#6f6f6f"
GRID = "#d9d9d9"
PRIMARY = "#31688e"
ACCENT = "#b8552f"
CONTROL = "#8c8c8c"
HIGHLIGHT = "#c9a227"

# Human labels. No raw `state_bilinear_ridge:true` reaches a figure.
METHOD_LABELS: dict[str, str] = {
    "intervention_only_ridge:none:0": "Intervention only",
    "visible_information_ridge:none:0": "Visible information",
    "state_bilinear_ridge:true:0": "True state",
    "state_bilinear_ridge:wrong_example:0": "Matched wrong state",
    "constant:none:0": "Constant",
    "prompt_lexical:none:0": "Prompt lexical",
}

# Deterministic drawing order wherever an order is not derived from the data.
CANONICAL_ORDER: tuple[str, ...] = (
    "intervention_only_ridge:none:0",
    "visible_information_ridge:none:0",
    "state_bilinear_ridge:true:0",
    "state_bilinear_ridge:wrong_example:0",
    "constant:none:0",
    "prompt_lexical:none:0",
)

COMPARISON_LABELS: dict[str, str] = {
    "visible_information minus true_state": "Visible information\nminus true state",
    "matched_wrong_state minus true_state": "Matched wrong state\nminus true state",
}

FIGURE_NAMES: tuple[str, ...] = (
    "study_overview",
    "calibration_dose_response",
    "prediction_vs_observed",
    "per_prompt_differences",
    "state_specificity",
    "final_test_mae_by_method",
    "final_test_primary_comparisons",
    "intervention_effects",
)


class PlotError(RuntimeError):
    """Raised when the bundle cannot be plotted from."""


@dataclass(frozen=True)
class BundleData:
    """Everything the figures read, loaded once from a verified bundle."""

    analysis: dict[str, Any]
    method_summary: dict[str, Any]
    prompt_scores: list[dict[str, Any]]
    pair_scores: list[dict[str, Any]]
    ratio_summaries: list[dict[str, Any]]
    decision: dict[str, Any]
    calibration_observations: list[dict[str, Any]]
    direction_roles: dict[str, str]
    calibration_signs: dict[tuple[str, str], int]

    def summary(self, key: str) -> dict[str, Any]:
        for row in self.method_summary["methods"]:
            if f"{row['method_id']}:{row['state_condition']}:{row['condition_index']}" == key:
                return row
        raise PlotError(f"no stored summary for {key}")

    def shuffled_maes(self) -> list[float]:
        """The ten deranged-state point estimates, in stored condition order."""
        rows = [
            row
            for row in self.method_summary["methods"]
            if row["state_condition"] == "shuffled" and row["method_id"] == "state_bilinear_ridge"
        ]
        return [row["mae"] for row in sorted(rows, key=lambda r: r["condition_index"])]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.titleweight": "bold",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "text.color": INK,
            "figure.autolayout": False,
        }
    )


def load_bundle(directory: Path) -> BundleData:
    """Read every artifact the figures need. Verification is the caller's job and is enforced."""
    from .bundle import CALIBRATION_CANDIDATE_SETS, CALIBRATION_OBSERVATIONS, DIRECTION_FAMILY

    def need(name: str) -> Path:
        path = directory / name
        if not path.exists():
            raise PlotError(f"the bundle is missing {name}, which the figures need")
        return path

    analysis = read_json(need("state_audit_analysis.json"))
    if analysis["analysis_hash"] != EXPECTED_ANALYSIS_HASH:
        raise PlotError(
            f"this bundle carries analysis {analysis['analysis_hash']}, but these figures describe "
            f"{EXPECTED_ANALYSIS_HASH}; refusing to caption one study's figures with another's"
        )

    ratio_payload = read_json(need("state_audit_ratio_summaries.json"))
    family = read_json(need(DIRECTION_FAMILY))
    # The observations record which direction was applied but not with which sign; the candidate
    # sets record the sign. Key on (trial, candidate) because a candidate id is unique per prompt.
    signs: dict[tuple[str, str], int] = {}
    for candidate_set in read_jsonl(need(CALIBRATION_CANDIDATE_SETS)):
        for candidate in candidate_set["candidates"]:
            trial = str(candidate["candidate_id"]).split(".", 1)[0]
            signs[(trial, candidate["candidate_id"])] = int(candidate["sign"])
    return BundleData(
        analysis=analysis,
        method_summary=read_json(need("state_audit_method_summary.json")),
        prompt_scores=list(read_jsonl(need("state_audit_prompt_scores.jsonl"))),
        pair_scores=list(read_jsonl(need("state_audit_pair_scores.jsonl"))),
        ratio_summaries=sorted(ratio_payload["summaries"], key=lambda r: r["norm_ratio"]),
        decision=read_json(need("state_audit_calibration_decision.json")),
        calibration_observations=list(read_jsonl(need(CALIBRATION_OBSERVATIONS))),
        direction_roles={d["opaque_id"]: d["construction_role"] for d in family["directions"]},
        calibration_signs=signs,
    )


def _save(fig: Figure, output: Path, name: str) -> list[Path]:
    written: list[Path] = []
    for suffix in (".png", ".pdf"):
        path = output / f"{name}{suffix}"
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    return written


def _footnote(fig: Figure, text: str) -> None:
    fig.text(0.5, -0.02, text, ha="center", va="top", fontsize=6.8, color=MUTED, wrap=True)


# -- panels -------------------------------------------------------------------
#
# Each takes an axes and the loaded data, so a panel can appear both on its own and inside the
# overview without the two versions drifting apart.


def _ratio_axis(ax: Any, ratios: Sequence[float]) -> None:
    """Log x-axis labelled only at the five preregistered ratios.

    Matplotlib's log locator adds minor ticks at 2x, 3x, 4x and so on, and their labels collide
    with the five values that actually mean something here.
    """
    ax.set_xscale("log")
    ax.set_xticks(list(ratios))
    ax.set_xticklabels([f"{r:g}" for r in ratios])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def _panel_flip_rate(ax: Any, data: BundleData) -> None:
    ratios = [row["norm_ratio"] for row in data.ratio_summaries]
    rates = [
        100.0 * row["flip_count"] / row["observed_non_noop_observations"]
        for row in data.ratio_summaries
    ]
    passed = [row["passed"] for row in data.ratio_summaries]
    colors = [PRIMARY if ok else CONTROL for ok in passed]

    ax.plot(ratios, rates, color=MUTED, linewidth=1.0, zorder=1)
    ax.scatter(ratios, rates, c=colors, s=34, zorder=2, edgecolors="white", linewidths=0.6)
    for ratio, rate in zip(ratios, rates, strict=True):
        ax.annotate(
            f"{rate:.2f}%",
            (ratio, rate),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=6.8,
            color=INK,
        )
    _ratio_axis(ax, ratios)
    ax.set_xlabel("intervention strength (ratio of median clean state norm)")
    ax.set_ylabel("answer flips (% of 512)")
    ax.set_title("Stronger interventions flip more answers")
    ax.set_ylim(-6, 100)


def _panel_effect_sizes(ax: Any, data: BundleData) -> None:
    ratios = [row["norm_ratio"] for row in data.ratio_summaries]
    medians = [row["median_abs_effect"] for row in data.ratio_summaries]
    p95s = [row["p95_abs_effect"] for row in data.ratio_summaries]
    ceiling = 4.0

    ax.plot(ratios, medians, color=PRIMARY, marker="o", markersize=4, linewidth=1.2, label="median")
    ax.plot(
        ratios,
        p95s,
        color=ACCENT,
        marker="s",
        markersize=4,
        linewidth=1.2,
        label="95th percentile",
    )
    ax.axhline(ceiling, color=ACCENT, linestyle=":", linewidth=1.0)
    ax.annotate(
        "preregistered p95 ceiling 4.0",
        (ratios[0], ceiling),
        textcoords="offset points",
        xytext=(2, 4),
        fontsize=6.8,
        color=ACCENT,
    )
    for row, p95 in zip(data.ratio_summaries, p95s, strict=True):
        if not row["passed"]:
            ax.scatter(
                [row["norm_ratio"]],
                [p95],
                s=90,
                facecolors="none",
                edgecolors=ACCENT,
                linewidths=1.1,
                zorder=3,
            )
    _ratio_axis(ax, ratios)
    ax.set_yscale("log")
    ax.set_xlabel("intervention strength (ratio)")
    ax.set_ylabel("|change in clean top margin|")
    ax.set_title("Ratios above 0.02 exceed the p95 ceiling")
    ax.legend(loc="upper left")


def _mark_selected(ax: Any, data: BundleData, label: bool = True) -> None:
    selected = data.decision["selected_norm_ratio"]
    ax.axvline(selected, color=HIGHLIGHT, linewidth=1.2, zorder=0)
    if label:
        ax.annotate(
            f"selected {selected:g}",
            (selected, ax.get_ylim()[1]),
            textcoords="offset points",
            xytext=(4, -9),
            fontsize=6.8,
            color="#8a6d1a",
        )


def _panel_mae(ax: Any, data: BundleData) -> None:
    rows = [(key, data.summary(key)["mae"]) for key in CANONICAL_ORDER]
    rows.sort(key=lambda item: item[1])
    labels = [METHOD_LABELS[key] for key, _ in rows]
    values = [mae for _, mae in rows]
    colors = [ACCENT if key == "state_bilinear_ridge:true:0" else PRIMARY for key, _ in rows]

    positions = range(len(rows))
    ax.barh(list(positions), values, color=colors, height=0.62, zorder=2)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for position, value in zip(positions, values, strict=True):
        ax.annotate(
            f"{value:.4f}",
            (value, position),
            textcoords="offset points",
            xytext=(4, 0),
            va="center",
            fontsize=7,
            color=INK,
        )
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel("prompt-aggregated MAE (point estimate, 32 prompts)")
    ax.set_title("Lowest error came from the intervention alone")
    ax.grid(axis="y", visible=False)


def _panel_comparisons(ax: Any, data: BundleData) -> None:
    comparisons = data.analysis["primary_comparisons"]
    positions = list(range(len(comparisons)))
    for position, comparison in zip(positions, comparisons, strict=True):
        low, high, point = comparison["ci_low"], comparison["ci_high"], comparison["point_estimate"]
        ax.plot([low, high], [position, position], color=PRIMARY, linewidth=1.6, zorder=2)
        for endpoint in (low, high):
            ax.plot([endpoint, endpoint], [position - 0.09, position + 0.09], color=PRIMARY, lw=1.2)
        ax.scatter([point], [position], s=42, color=ACCENT, zorder=3, edgecolors="white", lw=0.6)
        ax.annotate(
            f"{point:+.6f}   [{low:+.6f}, {high:+.5f}]",
            (point, position),
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            fontsize=6.8,
            color=INK,
        )
    ax.axvline(0.0, color=INK, linestyle="--", linewidth=0.9, zorder=1)
    ax.set_yticks(positions)
    ax.set_yticklabels([COMPARISON_LABELS[c["name"]] for c in comparisons])
    ax.set_ylim(-0.6, len(comparisons) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("paired difference in prompt-aggregated MAE")
    ax.set_title("Both preregistered intervals cross zero")
    ax.grid(axis="y", visible=False)


# -- figures ------------------------------------------------------------------


def figure_study_overview(data: BundleData) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4))
    _panel_flip_rate(axes[0][0], data)
    _mark_selected(axes[0][0], data)
    _panel_effect_sizes(axes[0][1], data)
    _mark_selected(axes[0][1], data, label=False)
    _panel_mae(axes[1][0], data)
    _panel_comparisons(axes[1][1], data)
    fig.suptitle(
        "Preregistered state-dependence audit on Gemma 3 1B: no detected improvement "
        "from hidden-state access",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.015, 1, 0.965))
    _footnote(
        fig,
        "Top row: calibration diagnostics that chose the stimulus, before any forecast existed. "
        "Bottom row: the final test. Bars are point estimates with no interval; only the two "
        "preregistered comparisons were tested.",
    )
    return fig


def figure_calibration_dose_response(data: BundleData) -> Figure:
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 6.8), sharex=True)
    _panel_flip_rate(axes[0], data)
    _panel_effect_sizes(axes[1], data)
    for ax in axes:
        _mark_selected(ax, data, label=ax is axes[0])
    for ax in axes:
        for row in data.ratio_summaries:
            if not row["passed"]:
                ax.axvspan(
                    row["norm_ratio"] * 0.86, row["norm_ratio"] * 1.16, color="#f2e4de", zorder=0
                )
    axes[0].set_xlabel("")
    axes[0].set_title("Flip rate by intervention strength")
    axes[1].set_title("Effect size by intervention strength")
    fig.suptitle(
        "Calibration dose response: 0.02 was the smallest passing strength",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.955))
    _footnote(
        fig,
        "Shaded strengths (0.10, 0.20, 0.40) failed the preregistered p95 ceiling of 4.0 and only "
        "that condition. 0.05 also passed; 0.02 was selected because the rule takes the smallest "
        "passing ratio, not the largest effect. Calibration prompts only, 512 interventions per "
        "strength.",
    )
    return fig


def figure_prediction_vs_observed(data: BundleData) -> Figure:
    keys = (
        "intervention_only_ridge:none:0",
        "visible_information_ridge:none:0",
        "state_bilinear_ridge:true:0",
        "state_bilinear_ridge:wrong_example:0",
    )
    by_method: dict[str, list[dict[str, Any]]] = {key: [] for key in keys}
    for row in data.pair_scores:
        if row["method_condition"] in by_method and not row["is_noop"]:
            by_method[row["method_condition"]].append(row)

    observed_all = [r["observed_delta"] for rows in by_method.values() for r in rows]
    predicted_all = [r["predicted_delta"] for rows in by_method.values() for r in rows]
    lo = min(min(observed_all), min(predicted_all))
    hi = max(max(observed_all), max(predicted_all))
    pad = 0.06 * (hi - lo)
    limits = (lo - pad, hi + pad)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.0), sharex=True, sharey=True)
    for ax, key in zip(axes.flat, keys, strict=True):
        rows = sorted(by_method[key], key=lambda r: (r["trial_id"], r["candidate_id"]))
        summary = data.summary(key)
        ax.plot(limits, limits, color=MUTED, linestyle="--", linewidth=0.9, zorder=1)
        ax.axhline(0, color=GRID, linewidth=0.6, zorder=0)
        ax.axvline(0, color=GRID, linewidth=0.6, zorder=0)
        ax.scatter(
            [r["observed_delta"] for r in rows],
            [r["predicted_delta"] for r in rows],
            s=11,
            alpha=0.32,
            color=ACCENT if key == "state_bilinear_ridge:true:0" else PRIMARY,
            edgecolors="none",
            zorder=2,
        )
        spearman = summary["spearman"]
        ax.set_title(METHOD_LABELS[key])
        ax.annotate(
            f"MAE {summary['mae']:.4f}\nSpearman {spearman:.3f}\n{len(rows)} interventions",
            (0.03, 0.97),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize=7,
            color=INK,
        )
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")

    for ax in axes[1]:
        ax.set_xlabel("observed change in clean top margin")
    for ax in axes[:, 0]:
        ax.set_ylabel("predicted change")
    fig.suptitle(
        "Predicted against observed intervention effect, 512 sealed forecasts per method",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.955))
    _footnote(
        fig,
        "Identical axes across panels; the dashed line is y = x. Each point is one sealed "
        "forecast against its outcome; the annotated MAE and Spearman are the stored "
        "prompt-aggregated values from the analysis record, not recomputed over the points "
        "shown. Predictions span a far narrower range than outcomes, which is what the "
        "near-horizontal spread is.",
    )
    return fig


def _prompt_mae(data: BundleData, key: str) -> dict[str, float]:
    return {
        row["group_id"]: row["mae"] for row in data.prompt_scores if row["method_condition"] == key
    }


def figure_per_prompt_differences(data: BundleData) -> Figure:
    true_state = _prompt_mae(data, "state_bilinear_ridge:true:0")
    plans = (
        (
            "visible_information_ridge:none:0",
            "visible_information minus true_state",
            "Visible information minus true state",
            "H-BD1: positive bars are prompts where the true state helped",
        ),
        (
            "state_bilinear_ridge:wrong_example:0",
            "matched_wrong_state minus true_state",
            "Matched wrong state minus true state",
            "H-BD2: positive bars are prompts where the wrong state hurt",
        ),
    )
    comparisons = {c["name"]: c for c in data.analysis["primary_comparisons"]}

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.2))
    for ax, (key, comparison_name, title, reading) in zip(axes, plans, strict=True):
        other = _prompt_mae(data, key)
        # Sorted independently in each panel, as a waterfall, then tie-broken by prompt id so the
        # ordering is deterministic rather than dependent on dict iteration.
        diffs = sorted(
            ((group, other[group] - true_state[group]) for group in true_state),
            key=lambda item: (item[1], item[0]),
        )
        values = [value for _, value in diffs]
        colors = [PRIMARY if value > 0 else CONTROL for value in values]
        ax.bar(range(len(values)), values, color=colors, width=0.78, zorder=2)
        ax.axhline(0.0, color=INK, linewidth=0.9, zorder=3)

        comparison = comparisons[comparison_name]
        point, low, high = (
            comparison["point_estimate"],
            comparison["ci_low"],
            comparison["ci_high"],
        )
        ax.axhline(point, color=ACCENT, linestyle="--", linewidth=1.0, zorder=3)
        positive = sum(1 for value in values if value > 0)
        ax.set_title(
            f"{title}\n"
            f"point estimate {point:+.6f}, 95% interval [{low:+.6f}, {high:+.5f}] "
            f"(crosses zero) | {positive} of {len(values)} prompts positive",
            fontsize=8.5,
        )
        ax.annotate(
            reading,
            (0.995, 0.04),
            xycoords="axes fraction",
            ha="right",
            fontsize=7,
            color=MUTED,
        )
        ax.set_ylabel("difference in prompt MAE")
        ax.set_xlim(-0.8, len(values) - 0.2)
        ax.set_xticks([])
        ax.grid(axis="x", visible=False)
    axes[1].set_xlabel("final-test prompts, sorted independently within each panel")
    fig.suptitle(
        "Per-prompt differences behind the two preregistered comparisons",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.955))
    _footnote(
        fig,
        "One bar per prompt, each the mean absolute error over that prompt's 16 non-no-op "
        "interventions. The dashed line is the overall point estimate; the interval in each title "
        "is the preregistered paired bootstrap over 32 prompt groups, 10,000 resamples.",
    )
    return fig


def figure_state_specificity(data: BundleData) -> Figure:
    shuffled = data.shuffled_maes()
    true_mae = data.summary("state_bilinear_ridge:true:0")["mae"]
    wrong_mae = data.summary("state_bilinear_ridge:wrong_example:0")["mae"]
    visible_mae = data.summary("visible_information_ridge:none:0")["mae"]

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.scatter(
        shuffled,
        [0.0] * len(shuffled),
        s=44,
        facecolors="none",
        edgecolors=CONTROL,
        linewidths=1.1,
        zorder=2,
        label=f"shuffled state, {len(shuffled)} seeded derangements (descriptive)",
    )
    ax.scatter([true_mae], [0.0], s=110, marker="D", color=ACCENT, zorder=4, label="true state")
    ax.scatter(
        [wrong_mae], [0.0], s=110, marker="s", color=PRIMARY, zorder=4, label="matched wrong state"
    )
    ax.scatter(
        [visible_mae],
        [0.0],
        s=130,
        marker="*",
        color=HIGHLIGHT,
        zorder=4,
        label="visible information (no state)",
    )
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("prompt-aggregated MAE (point estimate)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncols=2)
    fig.suptitle(
        "Where the true state sits among deliberately wrong states", fontsize=11, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    _footnote(
        fig,
        "The ten shuffled points are DESCRIPTIVE ONLY: ten point estimates over ten seeds, not a "
        "null distribution, not a confidence interval, and not a test. No conclusion rests on "
        "them. The only tested contrast involving the true state is H-BD2, whose interval crosses "
        "zero.",
    )
    return fig


def figure_intervention_effects(data: BundleData) -> Figure:
    selected = data.decision["selected_norm_ratio"]
    rows = [
        row
        for row in data.calibration_observations
        if not row["is_noop"] and math.isclose(row["norm_ratio"], selected, rel_tol=1e-12)
    ]
    if not rows:
        raise PlotError(f"no calibration observations at the selected ratio {selected}")
    signed = [row["delta_clean_top_margin"] for row in rows]
    absolute = [abs(value) for value in signed]

    def flip_rate(subset: Sequence[dict[str, Any]]) -> float:
        return 100.0 * sum(1 for r in subset if r["answer_flip"]) / len(subset) if subset else 0.0

    by_role: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        role = data.direction_roles.get(row["direction_ref"], "unknown")
        by_role.setdefault(role, []).append(row)
    role_labels = {
        "answer_token_centered": "Answer-token directions",
        "random_orthogonal_control": "Random orthogonal controls",
    }
    roles = sorted(by_role, key=lambda r: role_labels.get(r, r))
    by_sign: dict[int, list[dict[str, Any]]] = {1: [], -1: []}
    for row in rows:
        sign = data.calibration_signs.get((row["trial_id"], row["candidate_id"]))
        if sign in by_sign:
            by_sign[sign].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.6))

    axes[0][0].hist(signed, bins=48, color=PRIMARY, edgecolor="white", linewidth=0.3)
    axes[0][0].axvline(0.0, color=INK, linestyle="--", linewidth=0.9)
    axes[0][0].set_xlabel("change in clean top margin (signed)")
    axes[0][0].set_ylabel("interventions")
    axes[0][0].set_title("Signed effect is centred near zero")

    axes[0][1].hist(absolute, bins=48, color=PRIMARY, edgecolor="white", linewidth=0.3)
    # The stored median, not a recomputed one. numpy's median averages the two central values of an
    # even sample; picking the upper-middle element here would disagree with the frozen record in
    # the fifth decimal, and the figure must show the number the study reported.
    selected_summary = next(
        row
        for row in data.ratio_summaries
        if math.isclose(row["norm_ratio"], selected, rel_tol=1e-12)
    )
    median = selected_summary["median_abs_effect"]
    axes[0][1].axvline(median, color=ACCENT, linewidth=1.1)
    axes[0][1].annotate(
        f"median {median:.5f}",
        (median, axes[0][1].get_ylim()[1]),
        textcoords="offset points",
        xytext=(5, -10),
        fontsize=7,
        color=ACCENT,
    )
    axes[0][1].set_xlabel("|change in clean top margin|")
    axes[0][1].set_ylabel("interventions")
    axes[0][1].set_title("Most effects are small but measurable")

    for ax, groups, labels, title in (
        (
            axes[1][0],
            [by_role[role] for role in roles],
            [role_labels.get(role, role) for role in roles],
            "Answer-token directions flip no more often than controls",
        ),
        (
            axes[1][1],
            [by_sign[1], by_sign[-1]],
            ["Positive sign", "Negative sign"],
            "Flip rate by intervention sign",
        ),
    ):
        rates = [flip_rate(group) for group in groups]
        positions = range(len(groups))
        ax.bar(list(positions), rates, color=PRIMARY, width=0.5, zorder=2)
        for position, rate, group in zip(positions, rates, groups, strict=True):
            flips = sum(1 for r in group if r["answer_flip"])
            ax.annotate(
                f"{rate:.2f}%\n{flips} of {len(group)}",
                (position, rate),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=7,
                color=INK,
            )
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels)
        ax.set_ylabel("answer flips (%)")
        ax.set_ylim(0, max(rates) * 1.5 if max(rates) else 1)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)

    fig.suptitle(
        f"Calibration diagnostics at the selected strength (ratio {selected:g}) "
        "- not the forecasting result",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.955))
    _footnote(
        fig,
        "Calibration prompts only. These panels describe the stimulus that was chosen before any "
        "forecast existed; they say nothing about whether hidden states predict it. That the "
        "answer-token directions flip answers at about the control rate is a limitation of the "
        "intervention family, reported in the paper. The two lower panels show the same counts by "
        "coincidence: the role split and the sign split happen to give 10 and 9 flips each.",
    )
    return fig


def figure_mae_by_method(data: BundleData) -> Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    _panel_mae(ax, data)
    # The ten shuffled-state controls are deliberately not overlaid here. On this axis they
    # collapse into an illegible cluster against the bars, and `state_specificity` already shows
    # them at a scale where their spread is readable.
    fig.tight_layout(rect=(0, 0.04, 1, 1.0))
    _footnote(
        fig,
        "32 prompts, 512 signed pairs per method. Bars are point estimates and carry no interval, "
        "because none was computed for them; only the two preregistered comparisons were tested. "
        "The ten shuffled-state controls are shown separately in the state-specificity figure.",
    )
    return fig


def figure_primary_comparisons(data: BundleData) -> Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    _panel_comparisons(ax, data)
    analysis = data.analysis
    fig.tight_layout(rect=(0, 0.05, 1, 1.0))
    _footnote(
        fig,
        f"{analysis['bootstrap_resamples']:,} paired bootstrap resamples over "
        f"{analysis['group_count']} prompt groups, seed {analysis['bootstrap_seed']}. Under the "
        "preregistered rule a comparison is supported only if its interval excludes zero in the "
        "hypothesized direction. Both cross zero, so neither is supported; neither is a "
        "demonstration of equivalence.",
    )
    return fig


BUILDERS = {
    "study_overview": figure_study_overview,
    "calibration_dose_response": figure_calibration_dose_response,
    "prediction_vs_observed": figure_prediction_vs_observed,
    "per_prompt_differences": figure_per_prompt_differences,
    "state_specificity": figure_state_specificity,
    "final_test_mae_by_method": figure_mae_by_method,
    "final_test_primary_comparisons": figure_primary_comparisons,
    "intervention_effects": figure_intervention_effects,
}


def plot_bundle(bundle_path: Path, output: Path) -> dict[str, Any]:
    """Verify the bundle, then draw every figure from it. Loads no model.

    Verification is not optional and not a flag. If the bundle's checksums or its stored analysis
    do not hold up, no figure is written, because a figure is the artifact most likely to be
    reused without its provenance.
    """
    from .bundle import BundleError, replay_bundle

    try:
        replay = replay_bundle(bundle_path)
    except BundleError as error:
        raise PlotError(f"refusing to plot: {error}") from error
    if not replay["valid"]:
        raise PlotError(f"refusing to plot, the bundle does not verify: {replay['failures']}")

    data = load_bundle(bundle_path)
    output.mkdir(parents=True, exist_ok=True)
    _style()

    written: list[str] = []
    for name in FIGURE_NAMES:
        figure = BUILDERS[name](data)
        for path in _save(figure, output, name):
            written.append(path.name)
    info("figures written", count=len(written), output=str(output))

    return {
        "algorithm_version": PLOT_ALGORITHM_VERSION,
        "analysis_hash": data.analysis["analysis_hash"],
        "bundle_path": str(bundle_path),
        "bundle_verified": True,
        "checksums_checked": replay["checksums_checked"],
        "figures": sorted(written),
        "figure_count": len(FIGURE_NAMES),
        "output": str(output),
        "notes": (
            "Drawn from the verified public bundle. No model was loaded, nothing was refitted, "
            "and no test beyond the two preregistered comparisons is displayed. The shuffled-state "
            "controls are drawn as descriptive point estimates."
        ),
    }


__all__ = [
    "BUILDERS",
    "CANONICAL_ORDER",
    "EXPECTED_ANALYSIS_HASH",
    "FIGURE_NAMES",
    "METHOD_LABELS",
    "PLOT_ALGORITHM_VERSION",
    "BundleData",
    "PlotError",
    "load_bundle",
    "plot_bundle",
]
