"""Compare measured CIFAR image-pair correlations with initialization theory."""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.plot_style import field_palette, paper_style  # noqa: E402


def kernel(c):
    c = np.clip(c, -1, 1)
    return (np.sqrt(1 - c * c) + (np.pi - np.arccos(c)) * c) / np.pi


def main():
    data = json.loads((ROOT / "results/cifar_correlations.json").read_text())
    config = data["config"]
    rates, seeds = config["rates"], config["seeds"]
    stages = ["initial", "best", "final"]
    records = {(r["rate"], r["seed"], r["stage"]): r for r in data["records"]}
    expected = {(p, seed, stage) for p in rates for seed in seeds for stage in stages}
    if set(records) != expected or len(records) != len(data["records"]):
        raise ValueError("Expected one correlation curve per rate, seed and stage")
    layers = np.arange(config["depth"] + 1)
    palette = field_palette(len(rates))
    with plt.rc_context(paper_style()):
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.075, right=0.99, bottom=0.31, top=0.89, wspace=0.12)
        for ax, stage, title in zip(
            axes, ["initial", "final"], ["(a) Initialization", "(b) After 35 epochs"]
        ):
            for p, color in zip(rates, palette):
                values = np.array(
                    [records[p, seed, stage]["mean_pair_correlation"] for seed in seeds]
                )
                if values.shape != (len(seeds), len(layers)) or not np.isfinite(values).all():
                    raise ValueError("Invalid correlation curve")
                mean = values.mean(0)
                half = t.ppf(0.975, len(seeds) - 1) * values.std(0, ddof=1) / np.sqrt(len(seeds))
                ax.fill_between(
                    layers, mean - half, mean + half, color=color, alpha=0.08, linewidth=0
                )
                ax.plot(layers, mean, color=color, lw=1.35, label=f"$p={p:g}$")
                if stage == "initial":
                    pairs = np.asarray(data["input_correlation_by_pair"], dtype=float)
                    prediction = [pairs.mean()]
                    for _ in range(config["depth"]):
                        pairs = (1 - p) * kernel(pairs)
                        prediction.append(pairs.mean())
                    fixed = brentq(lambda c: (1 - p) * kernel(c) - c, 0, 1, xtol=1e-14)
                    ax.plot(layers, prediction, color=color, ls=":", lw=1.1)
                    ax.axhline(fixed, color=color, ls=(0, (4, 4)), lw=0.65, alpha=0.5)
            ax.set_title(title, fontsize=9, pad=7)
            ax.set(
                xlim=(0, config["depth"]),
                ylim=(-0.025, 1.03),
                xticks=[0, 3, 6, 9, 12],
                xlabel="Layer",
            )
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Image-pair correlation", fontsize=10)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.53, 0.01),
            ncol=5,
            frameon=False,
            fontsize=8,
            columnspacing=1.35,
            handlelength=1.7,
        )
        output = ROOT / "manuscript/figures/experiments/mlp/cifar_correlations"
        for extension in ("pdf", "png"):
            fig.savefig(output.with_suffix(f".{extension}"))
        plt.close(fig)


if __name__ == "__main__":
    main()
