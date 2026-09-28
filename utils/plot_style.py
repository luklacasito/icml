"""Slide colors on a white paper canvas, shared by the figure exporters.

Use ``plt.rcParams.update(paper_style())`` or ``plt.rc_context(paper_style())``.
Schedule colors follow the presentation's test-loss and dropout-sweep figures.
Line styles distinguish additional schedules without changing those colors.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap, to_hex

FOREST = "#0E3B2C"
CLAY = "#B04A2F"
GOLD = "#C99A3B"
BLUE = "#3A6EA5"
INK = "#202121"
NO_DROPOUT_GRAY = "#8A8D88"
WHITE = "#FFFFFF"
MUTED = "#60645F"
GRID = "#DDD5C2"

SCHEDULE_COLORS = {
    "none": NO_DROPOUT_GRAY,
    "constant": GOLD,
    "reverse_step": FOREST,
    "step": BLUE,
    "big_step": CLAY,
    "reverse_linear": CLAY,
    "linear": BLUE,
    "double": "#9E7B38",
    "triple": "#6B503A",
}
SCHEDULE_LINESTYLES = {
    "none": (0, (3.5, 2)),
    "constant": "-",
    "reverse_step": "-",
    "step": "-",
    "big_step": "-",
    "reverse_linear": "--",
    "linear": "--",
    "double": "--",
    "triple": ":",
}
SCHEDULE_LABELS = {
    "none": "No dropout",
    "constant": "Uniform",
    "reverse_step": "Step early",
    "step": "Step late",
    "big_step": "Big step early",
    "reverse_linear": "Linear decreasing",
    "linear": "Linear increasing",
    "double": "Uniform, double budget",
    "triple": "Uniform, triple budget",
}

# The benchmark archive uses these names for the same schedule shapes.
for _alias, _original in {
    "none_tuned": "none",
    "uniform": "constant",
    "frontloaded": "reverse_step",
    "step_early": "reverse_step",
    "step_late": "step",
    "linear_early": "reverse_linear",
    "linear_late": "linear",
}.items():
    SCHEDULE_COLORS[_alias] = SCHEDULE_COLORS[_original]
    SCHEDULE_LINESTYLES[_alias] = SCHEDULE_LINESTYLES[_original]
    SCHEDULE_LABELS[_alias] = SCHEDULE_LABELS[_original]


def field_palette(n: int) -> list[str]:
    """Return the slide's ordered gold/clay/forest ramp for field values."""
    if n < 0:
        raise ValueError("The number of colors must be nonnegative")
    ramp = LinearSegmentedColormap.from_list("paper_field", ["#C79A4E", CLAY, FOREST])
    return [to_hex(ramp(i / max(n - 1, 1))) for i in range(n)]


def paper_style() -> dict:
    """Return portable Matplotlib settings without changing global state."""
    return {
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 10,
        "mathtext.fontset": "cm",
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.5,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.facecolor": WHITE,
        "legend.edgecolor": GRID,
        "legend.framealpha": 0.95,
        "lines.linewidth": 1.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
