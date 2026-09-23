"""Plot the saved paper results without training or downloading data."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.results import load_json, load_npz_result  # noqa: E402

STYLES = {
    "none": ("No dropout", "#777777"),
    "constant": ("Uniform", "#222222"),
    "step": ("Step late", "#cc9a34"),
    "reverse_step": ("Step early", "#2b8c6d"),
    "linear": ("Linear increasing", "#4b79a1"),
    "reverse_linear": ("Linear decreasing", "#c96b36"),
    "big_step": ("Big step early", "#7561a4"),
    "double": ("Uniform, double budget", "#ba6b8d"),
    "triple": ("Uniform, triple budget", "#82774c"),
}
METRICS = (
    ("train_loss", "Training loss", "Cross-entropy"),
    ("test_loss", "Test loss", "Cross-entropy"),
    ("train_acc", "Training accuracy", "Accuracy (%)"),
    ("test_acc", "Test accuracy", "Accuracy (%)"),
)


def plot_curves(results, title, output, shared_baseline=False):
    """Each row in a saved metric is one seed; columns are training epochs."""
    arrays = {}
    shape = None
    for schedule, history in results.items():
        if schedule not in STYLES:
            raise ValueError(f"Unknown schedule: {schedule}")
        arrays[schedule] = {}
        for metric, _, _ in METRICS:
            values = np.asarray(history[metric], dtype=float)
            if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
                raise ValueError(f"{title}/{schedule}/{metric}: expected seeds × epochs")
            if not np.isfinite(values).all():
                raise ValueError(f"{title}/{schedule}/{metric}: non-finite curve")
            if shape is not None and values.shape != shape:
                raise ValueError(f"{title}: schedules and metrics have different shapes")
            shape = values.shape
            arrays[schedule][metric] = values
    if shape is None:
        raise ValueError(f"{title}: no results")

    seeds, epochs = shape
    x = np.arange(1, epochs + 1)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (metric, panel_title, ylabel) in zip(axes.flat, METRICS):
        for schedule, history in arrays.items():
            values = history[metric]
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / np.sqrt(seeds)
            label, color = STYLES[schedule]
            ax.plot(x, mean, color=color, label=label, linewidth=1.7)
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.14)
        ax.set_title(panel_title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, epochs)
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    note = f"Mean ± SEM across {seeds} seeds"
    if shared_baseline:
        note += "; no-dropout baseline shared across ablations"
    fig.suptitle(f"{title}\n{note}", fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = output.with_suffix(suffix)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "figures")
    parser.add_argument(
        "--sweeps", action="store_true", help="Include every saved MLP sweep condition"
    )
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    results = ROOT / "results"
    for name, title in (
        ("mlp_schedules", "CIFAR-10 MLP: dropout schedules"),
        ("mlp_budget_controls", "CIFAR-10 MLP: dropout budget controls"),
    ):
        saved = load_npz_result(results / f"{name}.npz")
        plot_curves(saved["results"], title, args.output / name)
    plot_curves(
        load_json(results / "vit_schedules.json"),
        "CIFAR-100 ViT: dropout schedules",
        args.output / "vit_schedules",
    )
    ablations = load_json(results / "vit_ablation.json")
    for mode, label in (
        ("attn_only", "attention only"),
        ("mlp_only", "MLP blocks only"),
        ("both", "attention and MLP blocks"),
    ):
        profiles = {"none": ablations["both_none"]}
        profiles.update(
            (row["schedule"], row)
            for row in ablations.values()
            if row["mode"] == mode and row["schedule"] != "none"
        )
        plot_curves(
            profiles,
            f"CIFAR-10 ViT: dropout in {label}",
            args.output / f"vit_ablation_{mode}",
            shared_baseline=True,
        )

    if args.sweeps:
        for name, title, parameter in (
            ("mlp_dropout_sweep", "CIFAR-10 ReLU MLP", "mean dropout field"),
            ("mlp_width_sweep", "CIFAR-10 ReLU MLP", "width"),
            ("mlp_gelu", "CIFAR-10 GELU MLP", "mean dropout field"),
        ):
            saved = load_npz_result(results / f"{name}.npz")
            conditions = {}
            for key, history in saved["all_results"].items():
                value, schedule = ast.literal_eval(key)
                conditions.setdefault(value, {})[schedule] = history
            for value, profiles in sorted(conditions.items()):
                plot_curves(
                    profiles,
                    f"{title}: {parameter} = {value:g}",
                    args.output / name / f"value_{value:g}".replace(".", "_"),
                )


if __name__ == "__main__":
    main()
