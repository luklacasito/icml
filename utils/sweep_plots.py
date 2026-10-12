"""Common figures and tables for the width, dropout, and GELU MLP sweeps."""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.stats import sem as standard_error

from .plotting import plot_mean_band, errorbar_mean_sem

DISPLAY_NAMES = {
    "none": "No dropout",
    "constant": "Constant",
    "reverse_step": "Step (early)",
    "big_step": "Big step (1/3)",
}
COLORS = {
    "none": "#4477AA",
    "constant": "#EE6677",
    "reverse_step": "#AA3377",
    "big_step": "#009988",
}
MARKERS = {
    "none": "s",
    "constant": "o",
    "reverse_step": "D",
    "big_step": "v",
}


ERROR_STYLE = dict(capsize=3, markeredgecolor="white", markeredgewidth=0.5)


def sweep_style():

    plt.style.use("seaborn-v0_8-whitegrid")

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "mathtext.fontset": "cm",
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
        }
    )


def sweep_accuracy(ax, best, values, schedules, *, x=None, capthick=None, **style):
    """Each seed contributes its maximum test accuracy, with sample SEM."""
    options = ERROR_STYLE | style
    if capthick is not None:
        options["capthick"] = capthick
    for sched in schedules:
        samples = [best[value, sched] for value in values]
        errorbar_mean_sem(
            ax,
            values if x is None else x,
            samples,
            fmt=MARKERS[sched] + "-",
            color=COLORS[sched],
            label=DISPLAY_NAMES[sched],
            markersize=6,
            **options,
        )


def sweep_advantage(ax, best, values):
    """Paired seed differences against constant dropout."""
    for sched in ("reverse_step", "big_step"):
        differences = [best[value, sched] - best[value, "constant"] for value in values]
        errorbar_mean_sem(
            ax,
            values,
            differences,
            fmt=MARKERS[sched] + "-",
            color=COLORS[sched],
            label=DISPLAY_NAMES[sched] + " - Constant",
            markersize=6,
            **ERROR_STYLE,
        )
    ax.axhline(0, color="gray", ls="--", lw=0.8)


def xi_accuracy(ax, best, theory, values):
    for sched in ("constant", "reverse_step", "big_step"):
        points = [
            (theory[value, sched]["xi_eff"], best[value, sched])
            for value in values
            if theory[value, sched]["xi_eff"] <= 1e6
        ]
        errorbar_mean_sem(
            ax,
            [xi for xi, sample in points],
            [sample for xi, sample in points],
            fmt=MARKERS[sched],
            color=COLORS[sched],
            label=DISPLAY_NAMES[sched],
            markersize=7,
            ls="none",
            **ERROR_STYLE,
        )
    reference = best[values[0], "none"].mean()
    ax.axhline(
        reference, color=COLORS["none"], ls="--", lw=1, label=f"No dropout ({reference:.1f}%)"
    )


def training_panels(results, value, schedules, epochs, title):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    lines, labels = [], []
    for sched in schedules:
        if (value, sched) not in results:
            continue
        for ax, metric in zip(axes.flat, ("train_loss", "test_loss", "train_acc", "test_acc")):
            loss = metric.endswith("_loss")
            if loss:
                ax.set_yscale("log")
            line = plot_mean_band(
                ax,
                results[value, sched][metric],
                x=np.arange(epochs),
                minimum=1e-6 if loss else None,
                color=COLORS[sched],
                lw=1.5,
            )
        lines.append(line)
        labels.append(DISPLAY_NAMES[sched])
    for ax, title_text, ylabel in zip(
        axes.flat,
        ("(a) Training loss", "(b) Test loss", "(c) Training accuracy", "(d) Test accuracy"),
        ("Training loss", "Test loss", "Training accuracy (%)", "Test accuracy (%)"),
    ):
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.set_title(title_text, loc="left")
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.legend(
        lines,
        labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=8,
        framealpha=0.95,
        bbox_to_anchor=(0.5, -0.04),
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


def all_training_curves(results, values, schedules, epochs, row_labels):
    fig, axes = plt.subplots(len(values), 2, figsize=(10, 2.4 * len(values)), sharex=True)
    lines, labels = [], []
    for row, value in enumerate(values):
        for sched in schedules:
            if (value, sched) not in results:
                continue
            for ax, metric in zip(axes[row], ("test_loss", "test_acc")):
                loss = metric == "test_loss"
                if loss:
                    ax.set_yscale("log")
                line = plot_mean_band(
                    ax,
                    results[value, sched][metric],
                    x=np.arange(epochs),
                    minimum=1e-6 if loss else None,
                    color=COLORS[sched],
                    lw=1.3,
                )
            if row == 0:
                lines.append(line)
                labels.append(DISPLAY_NAMES[sched])
        axes[row, 0].set_ylabel("Test loss")
        axes[row, 1].set_ylabel("Test acc (%)")
        for ax in axes[row]:
            ax.annotate(
                row_labels[row],
                xy=(0.97, 0.95),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
            )
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    axes[0, 0].set_title("Test loss", fontsize=11)
    axes[0, 1].set_title("Test accuracy", fontsize=11)
    fig.legend(
        lines,
        labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=9,
        framealpha=0.95,
        bbox_to_anchor=(0.5, -0.02),
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def sweep_table(best, values, schedules, *, caption, label, heading, value_format):
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{" + caption + "}")
    print(r"\label{" + label + "}")
    print(r"\begin{tabular}{c" + "c" * len(schedules) + "}")
    print(r"\toprule")
    print(heading + "".join(" & " + DISPLAY_NAMES[s] for s in schedules) + " \\\\")
    print(r"\midrule")
    for value in values:
        stats = [(best[value, s].mean(), standard_error(best[value, s])) for s in schedules]
        candidates = [(i, mean) for i, (mean, error) in enumerate(stats) if schedules[i] != "none"]
        winner = max(candidates, key=lambda item: item[1])[0] if candidates else -1
        row = format(value, value_format)
        for i, (mean, error) in enumerate(stats):
            cell = f"\\textbf{{{mean:.2f}}}" if i == winner else f"{mean:.2f}"
            row += f" & {cell} $\\pm$ {error:.2f}"
        print(row + " \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
