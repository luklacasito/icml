"""Regenerate the eight original experimental paper figures from saved arrays.

    python scripts/plot_paper_results.py
    python scripts/plot_paper_results.py --output /tmp/paper-experiments --png

This ports the historical experimental figure generator to the public result
filenames and shared slide colors. The historical cohorts, epoch axes, log
scales, panels, labels, and mean +/- sample SEM calculations are retained.
The sweep endpoints are each seed's maximum test accuracy, as in the paper;
seed-extension results are not pooled into these multi-schedule figures.
No model training, downloads, private dependencies, or notebook execution.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.results import load_json, load_npz_result  # noqa: E402
from utils.plot_style import (  # noqa: E402
    NO_DROPOUT_GRAY,
    SCHEDULE_COLORS,
    SCHEDULE_LINESTYLES,
    paper_style,
)

RESULTS = ROOT / "results"
OUT = ROOT / "manuscript" / "figures" / "experiments"
WRITE_PNG = False

# Preserve the wording and marker shapes in the historical paper figures.
DISPLAY_NAMES = {
    "none": "No dropout",
    "constant": "Constant",
    "step": "Step (late)",
    "reverse_step": "Step (early)",
    "linear": "Linear (late)",
    "reverse_linear": "Linear (early)",
    "big_step": "Big step (1/3)",
    "double": "Double (2h)",
    "triple": "Triple (3h)",
}
MARKERS = {"none": "s", "constant": "o", "reverse_step": "D", "big_step": "v"}


def curve_with_band(ax, curves, *, color, label, x=None, alpha=0.16, **kwargs):
    """Plot the historical seed mean and sample SEM without smoothing."""
    arr = np.asarray(curves, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or not np.isfinite(arr).all():
        raise ValueError("Expected finite seeds x epochs arrays with at least two seeds")
    if x is None:
        x = np.arange(arr.shape[1])
    mean = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
    ax.plot(x, mean, color=color, label=label, **kwargs)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=alpha, linewidth=0)


def best_per_seed(entry):
    return np.nanmax(np.asarray(entry["test_acc"], dtype=float), axis=1)


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if WRITE_PNG:
        fig.savefig(path.with_suffix(".png"))
    print(path)


def _entry(container: dict, key0, schedule: str) -> dict:
    return container[str((key0, schedule))]


def _display(schedule: str, theory: dict | None = None) -> str:
    name = DISPLAY_NAMES.get(schedule, schedule)
    if theory and schedule in theory:
        xi = theory[schedule].get("xi_eff")
        if xi is not None:
            xi_s = r"\infty" if math.isinf(xi) else f"{xi:.1f}"
            return rf"{name} ($\xi={xi_s}$)"
    return name


def _panel_label(ax, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="left")


def make_mlp_overfit() -> None:
    data = load_npz_result(RESULTS / "mlp_schedules.npz")
    results, theory = data["results"], data["theory"]
    order = ["constant", "reverse_step", "step", "reverse_linear", "linear", "none"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.1), sharex=True)
    panels = [
        ("train_loss", "Training loss", "Training loss", True),
        ("test_loss", "Test loss", "Test loss", True),
        ("train_acc", "Training accuracy", "Training accuracy (%)", False),
        ("test_acc", "Test accuracy", "Test accuracy (%)", False),
    ]
    for ax, (metric, title, ylabel, logy) in zip(axes.ravel(), panels):
        for sched in order:
            curve_with_band(
                ax,
                results[sched][metric],
                color=SCHEDULE_COLORS[sched],
                linestyle=SCHEDULE_LINESTYLES[sched],
                label=_display(sched, theory),
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        if ax in axes[1]:
            ax.set_xlabel("Epoch")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save_figure(fig, OUT / "mlp" / "dropout_schedules_overfit.pdf")
    plt.close(fig)


def make_mlp_budget() -> None:
    data = load_npz_result(RESULTS / "mlp_budget_controls.npz")
    results, theory = data["results"], data["theory"]
    order = ["none", "constant", "reverse_step", "big_step", "double", "triple"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharex=True)
    panels = [
        ("train_loss", "Training loss", "Training loss", True),
        ("test_loss", "Test loss", "Test loss", True),
        ("train_acc", "Training accuracy", "Training accuracy (%)", False),
        ("test_acc", "Test accuracy", "Test accuracy (%)", False),
    ]
    for ax, (metric, title, ylabel, logy) in zip(axes.ravel(), panels):
        for sched in order:
            curve_with_band(
                ax,
                results[sched][metric],
                color=SCHEDULE_COLORS[sched],
                linestyle=SCHEDULE_LINESTYLES[sched],
                label=_display(sched, theory),
            )
        _panel_label(ax, "abcd"[list(axes.ravel()).index(ax)], title)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        if ax in axes[1]:
            ax.set_xlabel("Epoch")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save_figure(fig, OUT / "mlp" / "dropout_budget_comparison.pdf")
    plt.close(fig)


def make_h_sweep() -> None:
    data = load_npz_result(RESULTS / "mlp_dropout_sweep.npz")
    all_results = data["all_results"]
    h_values = data["config"]["H_BAR_VALUES"]
    schedules = ["none", "constant", "reverse_step", "big_step"]
    fig, ax = plt.subplots(figsize=(6.3, 4.1))
    for sched in schedules:
        means, sems = [], []
        for h in h_values:
            vals = best_per_seed(_entry(all_results, h, sched))
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ax.errorbar(
            h_values,
            means,
            yerr=sems,
            color=SCHEDULE_COLORS[sched],
            linestyle=SCHEDULE_LINESTYLES[sched],
            marker=MARKERS[sched],
            capsize=3,
            label=DISPLAY_NAMES[sched],
        )
    ax.set_title(r"Dropout scheduling across $\bar h$ (MLP, CIFAR-10)")
    ax.set_xlabel(r"Mean dropout field $\bar h$")
    ax.set_ylabel("Best test accuracy (%)")
    ax.legend(loc="upper left")
    save_figure(fig, OUT / "sweeps" / "h_sweep_schedule_comparison.pdf")
    plt.close(fig)


def make_width_sweep() -> None:
    data = load_npz_result(RESULTS / "mlp_width_sweep.npz")
    all_results = data["all_results"]
    widths = data["config"]["WIDTH_VALUES"]
    fig, ax = plt.subplots(figsize=(6.3, 4.1))
    for sched in ["reverse_step", "big_step"]:
        means, sems = [], []
        for width in widths:
            vals = best_per_seed(_entry(all_results, width, sched)) - best_per_seed(
                _entry(all_results, width, "constant")
            )
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ax.errorbar(
            widths,
            means,
            yerr=sems,
            color=SCHEDULE_COLORS[sched],
            linestyle=SCHEDULE_LINESTYLES[sched],
            marker=MARKERS[sched],
            capsize=3,
            label=f"{DISPLAY_NAMES[sched]} - Constant",
        )
    ax.axhline(0, color=NO_DROPOUT_GRAY, ls="--", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_title(r"Scheduling advantage vs width at $\bar h=0.1$")
    ax.set_xlabel(r"Hidden width $N$")
    ax.set_ylabel(r"$\Delta$ best test accuracy (%)")
    ax.legend(loc="lower right")
    save_figure(fig, OUT / "sweeps" / "width_sweep_baseline_advantage.pdf")
    plt.close(fig)


def make_gelu_sweep() -> None:
    data = load_npz_result(RESULTS / "mlp_gelu.npz")
    all_results = data["all_results"]
    h_values = data["config"]["H_BAR_VALUES"]
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    for sched in ["reverse_step", "big_step"]:
        means, sems = [], []
        for h in h_values:
            vals = best_per_seed(_entry(all_results, h, sched)) - best_per_seed(
                _entry(all_results, h, "constant")
            )
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ax.errorbar(
            h_values,
            means,
            yerr=sems,
            color=SCHEDULE_COLORS[sched],
            linestyle=SCHEDULE_LINESTYLES[sched],
            marker=MARKERS[sched],
            capsize=3,
            label=f"{DISPLAY_NAMES[sched]} - Constant",
        )
    ax.axhline(0, color=NO_DROPOUT_GRAY, ls="--", lw=1)
    ax.set_title("Scheduling advantage relative to constant dropout")
    ax.set_xlabel(r"Mean dropout field $\bar h$")
    ax.set_ylabel(r"$\Delta$ best test accuracy (%)")
    ax.legend(loc="lower left")
    save_figure(fig, OUT / "sweeps" / "gelu_h_sweep_delta.pdf")
    plt.close(fig)


def _json_curve_array(values):
    return np.asarray(values, dtype=float)


def make_vit_curves(cropped: bool) -> None:
    results = load_json(RESULTS / "vit_schedules.json")
    order = ["none", "constant", "reverse_step", "reverse_linear"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), sharex=True)
    panels = [
        ("train_loss", "Training loss", "Training loss", True),
        ("test_loss", "Test loss", "Test loss", False),
        ("train_acc", "Training accuracy (%)", "Training accuracy (%)", False),
        ("test_acc", "Test accuracy (%)", "Test accuracy (%)", False),
    ]
    start = 20 if cropped else 0
    for ax, (metric, title, ylabel, logy) in zip(axes.ravel(), panels):
        for sched in order:
            arr = _json_curve_array(results[sched][metric])
            x = np.arange(arr.shape[1])
            curve_with_band(
                ax,
                arr[:, start:],
                x=x[start:],
                color=SCHEDULE_COLORS[sched],
                linestyle=SCHEDULE_LINESTYLES[sched],
                label=DISPLAY_NAMES[sched],
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        if ax in axes[1]:
            ax.set_xlabel("Epoch")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    name = "dropout_schedules_cropped.pdf" if cropped else "dropout_schedules_full.pdf"
    save_figure(fig, OUT / "transformer" / name)
    plt.close(fig)


def make_component_ablations() -> None:
    data = load_json(RESULTS / "vit_ablation.json")
    modes = [
        ("both", "Both (Attn + MLP)"),
        ("attn_only", "Attention only"),
        ("mlp_only", "MLP only"),
    ]
    schedules = ["none", "constant", "reverse_step"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    for ax, (mode, title) in zip(axes, modes):
        for sched in schedules:
            key = f"{mode}_{sched}"
            if key not in data:
                continue
            arr = _json_curve_array(data[key]["test_acc"])
            label = "No dropout" if sched == "none" else DISPLAY_NAMES[sched]
            curve_with_band(
                ax,
                arr,
                color=SCHEDULE_COLORS[sched],
                linestyle=SCHEDULE_LINESTYLES[sched],
                label=label,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test accuracy (%)")
        ax.legend(loc="lower right")
    fig.suptitle("Dropout scheduling component ablation", fontweight="bold", y=1.03)
    fig.tight_layout()
    save_figure(fig, OUT / "transformer" / "component_ablations.pdf")
    plt.close(fig)


def main():
    global OUT, WRITE_PNG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--png", action="store_true", help="Also export PNG previews")
    args = parser.parse_args()
    OUT, WRITE_PNG = args.output, args.png
    plt.rcParams.update(paper_style())
    # Keep the original panel typography and dimensions for the paper layout.
    plt.rcParams.update(
        {
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.labelsize": 12.5,
            "axes.titlesize": 13.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "lines.linewidth": 2.2,
            "lines.markersize": 5.4,
        }
    )
    make_mlp_overfit()
    make_mlp_budget()
    make_h_sweep()
    make_width_sweep()
    make_gelu_sweep()
    make_vit_curves(cropped=True)
    make_vit_curves(cropped=False)
    make_component_ablations()


if __name__ == "__main__":
    main()
