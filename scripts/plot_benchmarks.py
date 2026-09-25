"""Rebuild the historical appendix curves without training or downloading data.

The archive keeps the original cohorts separate. Accuracy is stored as a
fraction and plotted as a percentage; bands are sample SD / sqrt(seed count).
These are training and validation curves, not per-epoch test evaluations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.plot_style import (  # noqa: E402
    BLUE,
    FOREST,
    SCHEDULE_COLORS,
    SCHEDULE_LINESTYLES,
    paper_style,
)
from utils.results import load_npz_result  # noqa: E402

PROFILE_LABELS = {
    "none_tuned": "No dropout (tuned)",
    "none": "No dropout",
    "uniform": "Uniform",
    "step_early": "Step early",
    "big_step": "Big step",
    "linear_early": "Linear decreasing",
    "linear_late": "Linear increasing",
    "step_late": "Step late",
    "quadratic_early": "Quadratic early",
    "quadratic_late": "Quadratic late",
    "quartic_early": "Quartic early",
    "quartic_late": "Quartic late",
}
PROFILE_COLORS = {
    **SCHEDULE_COLORS,
    "quadratic_early": FOREST,
    "quadratic_late": BLUE,
    "quartic_early": FOREST,
    "quartic_late": BLUE,
}
PROFILE_LINESTYLES = {
    **SCHEDULE_LINESTYLES,
    "quadratic_early": ":",
    "quadratic_late": ":",
    "quartic_early": "-.",
    "quartic_late": "-.",
}
METRICS = (
    ("train/loss", "Training loss"),
    ("validation/loss", "Validation loss"),
    ("train/accuracy", "Training accuracy (%)"),
    ("validation/accuracy", "Validation accuracy (%)"),
)


def curve_summary(arm, metric, epochs):
    """Check seed coverage before averaging an archived profile."""
    seeds = arm["seeds"]
    values = np.asarray(arm["curves"][metric], dtype=float)
    if (
        len(seeds) < 2
        or len(set(seeds)) != len(seeds)
        or values.shape != (len(seeds), len(epochs))
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"{metric}: expected finite curves for unique paired seeds")
    if metric.endswith("accuracy"):
        if np.any((values < 0) | (values > 1)):
            raise ValueError("Archived accuracy must be a fraction between 0 and 1")
        values = values * 100
    elif np.any(values < 0):
        raise ValueError("Loss values must be nonnegative")
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(seeds))


def plot_cohort(cohort, output):
    epochs = np.asarray(cohort["epoch"])
    if not np.array_equal(epochs, np.arange(len(epochs))):
        raise ValueError("Historical curves must cover every zero-based epoch")
    unknown = set(cohort["profiles"]) - set(PROFILE_LABELS)
    if unknown:
        raise ValueError(f"Unknown profiles: {sorted(unknown)}")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), sharex=True)
    for ax, (metric, title) in zip(axes.flat, METRICS):
        for profile, label in PROFILE_LABELS.items():
            if profile not in cohort["profiles"]:
                continue
            mean, sem = curve_summary(cohort["profiles"][profile], metric, epochs)
            color = PROFILE_COLORS[profile]
            ax.plot(
                epochs,
                mean,
                color=color,
                linestyle=PROFILE_LINESTYLES[profile],
                label=label,
            )
            lower = np.maximum(mean - sem, 1e-10) if metric == "train/loss" else mean - sem
            ax.fill_between(epochs, lower, mean + sem, color=color, alpha=0.14, linewidth=0)
        ax.set_title(title)
        ax.set_ylabel(title)
        if metric == "train/loss":
            ax.set_yscale("log")
        if ax in axes[1]:
            ax.set_xlabel("Epoch")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5 if len(labels) > 6 else 3,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(output, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results/benchmark_curves.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/figures/benchmarks")
    args = parser.parse_args()
    archive = load_npz_result(args.data)
    if archive.get("schema_version") != 1:
        raise ValueError("Unsupported historical curve archive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(paper_style()):
        for name, cohort in archive["figures"].items():
            plot_cohort(cohort, args.output_dir / f"{name}.pdf")
    print(f"Wrote {len(archive['figures'])} historical figures to {args.output_dir}")


if __name__ == "__main__":
    main()
