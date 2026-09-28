"""Plot final test performance as the fixed dropout block moves through the MLP."""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.plot_style import CLAY, FOREST, paper_style  # noqa: E402


def main():
    evidence = json.loads((ROOT / "results/sliding_block.json").read_text())
    seeds = evidence["seeds"]
    positions = evidence["block_starts_zero_based"]
    records = {(r["block_start"], r["seed"]): r for r in evidence["records"]}
    expected = {(position, seed) for position in positions for seed in seeds}
    if len(records) != len(evidence["records"]) or set(records) != expected:
        raise ValueError("Expected exactly one result per block position and seed")

    panels = [
        ("final_test_loss", "Final test loss (CE)", FOREST),
        ("final_test_accuracy_percent", "Final test accuracy (%)", CLAY),
    ]
    statistics = {}
    with plt.rc_context(paper_style()):
        fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.65))
        fig.subplots_adjust(left=0.09, right=0.985, bottom=0.22, top=0.91, wspace=0.34)
        for panel, (ax, (metric, label, color)) in enumerate(zip(axes, panels)):
            values = np.array(
                [[records[position, seed][metric] for position in positions] for seed in seeds]
            )
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {metric}")
            mean = values.mean(axis=0)
            half_width = (
                t.ppf(0.975, len(seeds) - 1) * values.std(axis=0, ddof=1) / np.sqrt(len(seeds))
            )
            for row in values:
                ax.plot(
                    positions,
                    row,
                    color=color,
                    alpha=0.25,
                    linewidth=0.8,
                    marker="o",
                    markersize=2.5,
                )
            ax.errorbar(
                positions,
                mean,
                yerr=half_width,
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=4,
                capsize=3,
                elinewidth=1,
                zorder=3,
            )
            ax.set(
                xticks=positions,
                xticklabels=[f"{i + 1}-{i + 3}" for i in positions],
                xlabel="Layers receiving dropout",
                ylabel=label,
                xlim=(-0.2, 3.2),
            )
            ax.text(0, 1.035, f"({chr(97 + panel)})", transform=ax.transAxes, fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x", visible=False)
            ax.margins(y=0.1)
            statistics[metric] = {
                "mean": mean.tolist(),
                "ci95_lower": (mean - half_width).tolist(),
                "ci95_upper": (mean + half_width).tolist(),
            }
        output = ROOT / "manuscript/figures/experiments/mlp/sliding_block_position"
        for extension in ("pdf", "png"):
            fig.savefig(output.with_suffix(f".{extension}"))
        plt.close(fig)
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
