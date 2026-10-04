"""Publication-friendly plots derived only from supplied predictions/results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .metrics import evaluate_predictions, oscr_curve, reliability_bins, validate_predictions


COLORS = {"known": "#416b91", "unknown": "#608d81", "background": "#7e8791"}


def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(Path(path).with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _axes():
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    return fig, ax


def plot_predictions(predictions, output_dir, threshold=None):
    """Save ROC, OSCR, reliability and uncertainty/evidence figures when defined."""
    data = validate_predictions(predictions)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = data["labels"]
    known = labels >= 0
    unknown = ~known
    uncertainty = data["uncertainty"]
    probabilities = data["probabilities"]
    metrics = evaluate_predictions(data, threshold)
    if known.any() and unknown.any():
        fig, ax = _axes()
        fpr, tpr, _ = roc_curve(unknown.astype(int), uncertainty, drop_intermediate=False)
        ax.plot(fpr, tpr, color=COLORS["known"], label=f"AUROC {metrics['auroc']:.2f}%")
        ax.plot([0, 1], [0, 1], "--", color="#a3a7ac", linewidth=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Known rejection rate", ylabel="Unknown rejection rate")
        ax.legend(frameon=False, loc="lower right")
        _finish(fig, output / "roc")

        fig, ax = _axes()
        pred = data.get("prediction", probabilities.argmax(axis=1))
        fpr, ccr, _ = oscr_curve(uncertainty[known], uncertainty[unknown], pred[known] == labels[known])
        ax.plot(fpr, ccr, color=COLORS["known"], label=f"OSCR {metrics['oscr']:.2f}%")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Unknown false acceptance rate", ylabel="Known correct acceptance rate")
        ax.legend(frameon=False, loc="lower right")
        _finish(fig, output / "oscr")

    if known.any():
        bins = reliability_bins(probabilities[known], labels[known])
        selected = bins["count"] > 0
        fig, ax = _axes()
        ax.bar(bins["lower"][selected], bins["accuracy"][selected], width=1 / 15,
               align="edge", color=COLORS["known"], alpha=0.75, edgecolor="white", label="Bin accuracy")
        ax.scatter(bins["confidence"][selected], bins["accuracy"][selected], s=20,
                   color=COLORS["unknown"], zorder=3, label="Mean confidence")
        ax.plot([0, 1], [0, 1], "--", color="#a3a7ac", linewidth=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Top-label confidence", ylabel="Known-class accuracy",
               title=f"ECE {metrics['ece']:.2f}% (all known images)")
        ax.legend(frameon=False, loc="upper left", fontsize=8)
        _finish(fig, output / "reliability")

    fig, ax = _axes()
    lo, hi = float(uncertainty.min()), float(uncertainty.max())
    if lo == hi:
        lo, hi = lo - 0.01, hi + 0.01
    edges = np.linspace(lo, hi, 31)
    for name, selected in (("known", known), ("unknown", unknown)):
        if selected.any():
            ax.hist(uncertainty[selected], bins=edges, weights=np.ones(selected.sum()) / selected.sum(),
                    color=COLORS[name], alpha=0.55, label=f"{name.capitalize()} (n={selected.sum()})")
    if threshold is not None:
        ax.axvline(threshold, color="#555b63", linestyle="--", linewidth=1, label=f"Threshold {threshold:.3g}")
    ax.set(xlabel="Unknown score / uncertainty", ylabel="Fraction of images")
    ax.legend(frameon=False, fontsize=8)
    _finish(fig, output / "uncertainty")

    if "evidence" in data:
        total = data["evidence"].sum(axis=1)
        quantities = {"known": total[known], "unknown": total[unknown]}
        if "background_evidence" in data and known.any():
            quantities["background"] = data["background_evidence"][known].sum(axis=1)
        nonempty = [values for values in quantities.values() if len(values)]
        max_evidence = max(float(values.max()) for values in nonempty)
        edges = np.linspace(0, max(max_evidence, 1e-6), 31)
        fig, ax = _axes()
        for name, values in quantities.items():
            if len(values):
                ax.hist(values, bins=edges, weights=np.ones(len(values)) / len(values), alpha=0.5,
                        color=COLORS[name], label=name.capitalize())
        ax.set(xlabel="Total known-class evidence E", ylabel="Fraction of images")
        ax.legend(frameon=False, fontsize=8)
        _finish(fig, output / "evidence")


def plot_method_comparison(records, output_dir):
    """Plot completed seed means with sample-SD bars, separately by dataset."""
    data = pd.DataFrame(records)
    if not {"dataset", "method"}.issubset(data):
        raise ValueError("Comparison data requires dataset and method columns.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = [name for name in ("known_accuracy", "auroc", "oscr", "ece", "fpr95")
               if name in data and data[name].notna().any()]
    if not metrics:
        return
    for dataset, subset in data.groupby("dataset", sort=True):
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), squeeze=False)
        for metric, ax in zip(metrics, axes[0]):
            summary = subset.groupby("method", sort=True)[metric].agg(["mean", "std", "count"])
            summary = summary[summary["mean"].notna()]
            positions = np.arange(len(summary))
            ax.bar(positions, summary["mean"], color=COLORS["known"], alpha=0.85)
            with_sd = summary["count"] > 1
            if with_sd.any():
                ax.errorbar(positions[with_sd], summary.loc[with_sd, "mean"],
                            yerr=summary.loc[with_sd, "std"], fmt="none", color="#3f464e", capsize=3)
            ax.set_xticks(positions, summary.index, rotation=55, ha="right", fontsize=8)
            ax.set(ylabel="Percent", title=f"{metric.replace('_', ' ').upper()} {'↓' if metric in ('ece', 'fpr95') else '↑'}")
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{dataset}: completed seed means; error bars are sample SD", fontsize=11)
        safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in str(dataset))
        _finish(fig, output / f"comparison_{safe_name}")
