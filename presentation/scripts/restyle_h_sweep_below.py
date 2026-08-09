#!/usr/bin/env python
"""h_bar dropout-scheduling figure, legend moved horizontally BELOW the plot
and a wider aspect so the plot fills the poster column. Writes straight to the
poster's asset path."""
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator, FixedFormatter, NullLocator

INK    = "#202121"
FOREST = "#0E3B2C"
GOLD   = "#D8AD62"
GILT   = "#9E7B38"
MIST   = "#C9CBC7"
MUTED  = "#60645F"

SCHED_COLOR = {
    "none":         "#8A8D88",
    "constant":     "#C99A3B",
    "reverse_step": FOREST,
    "big_step":     "#B04A2F",
}
SCHED_MARKER = {"none": "s", "constant": "o", "reverse_step": "D", "big_step": "v"}
SCHED_NAME   = {
    "none": "No dropout", "constant": "Constant",
    "reverse_step": "Step (early)", "big_step": "Big step (1/3)",
}
SCHED_ORDER  = ["none", "constant", "reverse_step", "big_step"]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 19,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 1.2,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 16,
    "figure.dpi": 300,
})

REPO_DIR = Path("/Users/mac/Documents/icml/icml26")
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)

data = pickle.load(open(REPO_DIR / "supplementary/h_sweep/h_bar_sweep_results.pkl", "rb"))
all_results = data["all_results"]
H = data["config"]["H_BAR_VALUES"]

fig, ax = plt.subplots(figsize=(9.2, 4.9))
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

ax.yaxis.grid(True, color=MIST, lw=0.8, alpha=0.7)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

def key(hb, sched):
    return f"({hb}, '{sched}')"

for sched in SCHED_ORDER:
    means, sems = [], []
    for hb in H:
        best = all_results[key(hb, sched)]["test_acc"].max(axis=1)
        means.append(best.mean())
        sems.append(best.std() / np.sqrt(len(best)))
    c = SCHED_COLOR[sched]
    ax.errorbar(
        H, means, yerr=sems,
        fmt=SCHED_MARKER[sched] + "-", color=c, label=SCHED_NAME[sched],
        lw=2.6, markersize=8.5, markeredgecolor="white", markeredgewidth=0.9,
        capsize=3.5, capthick=1.4, ecolor=c, alpha=0.95,
        zorder=4 if sched == "reverse_step" else 3,
    )

ax.set_xlabel(r"Mean dropout field  $\bar{h}$")
ax.set_ylabel("Best test accuracy (%)")
ax.set_xscale("log")
ax.set_xlim(min(H) * 0.82, max(H) * 1.18)
ax.xaxis.set_major_locator(FixedLocator(list(H)))
ax.xaxis.set_major_formatter(FixedFormatter([f"{h:g}" for h in H]))
ax.xaxis.set_minor_locator(NullLocator())

leg = ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    ncol=4,
    frameon=False,
    handlelength=1.6,
    columnspacing=1.6,
    labelcolor=INK,
)

plt.tight_layout()
out = Path("/Users/mac/Documents/icml/poster/assets/figures/h_sweep_schedule_styled.pdf")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.03)
print("saved", out)
