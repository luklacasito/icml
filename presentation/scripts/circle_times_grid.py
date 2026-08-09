#!/usr/bin/env python3
"""
Static 3x3 snapshot grid for the circular manifold simulation.

Rows are the three mean-field regimes already used in the talk animation.
Columns are layers t = 1, T/2, T. Each panel shows the 2D projected circle,
with a small explained-variance inset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import manifold_folding_hq as mf


PAPER = mf.PAPER
INK = mf.INK
FOREST = mf.FOREST
MUTED = mf.MUTED
GOLD = mf.GOLD
CLAY = mf.CLAY
ORDERED_BLUE = mf.ORDERED_BLUE


def pca_full(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = state - state.mean(axis=0, keepdims=True)
    gram = x @ x.T
    ev, u = np.linalg.eigh(gram)
    ev = np.clip(ev, 0.0, None)
    order = np.argsort(ev)[::-1]
    ev = ev[order]
    u = u[:, order]
    coords = u[:, :3] * np.sqrt(ev[:3])
    fve = ev / (ev.sum() + 1e-12)
    return coords, fve


def build_snapshot_data(
    width: int,
    depth: int,
    n_theta: int,
    sb: float,
    seed: int,
    snapshot_layers: list[int],
):
    sb2 = sb**2
    targets = [(0.85, "ordered"), (1.00, "edge-of-chaos"), (1.16, "chaotic")]
    sigmas = [mf.solve_for_chi(target, sb2) for target, _ in targets]
    raw = [
        mf.propagate_loop(sw, sb, width, depth, n_theta, seed + k)
        for k, sw in enumerate(sigmas)
    ]

    theta = raw[0][1]
    base_colors = mf.cyclic_palette(theta)
    coords_by_regime = []
    fve_by_regime = []
    scales = []

    for states, _ in raw:
        previous = None
        regime_coords = {}
        regime_fve = {}
        max_abs = 0.0
        for layer in snapshot_layers:
            coords, fve = pca_full(states[layer])
            coords = mf.procrustes(coords, previous)
            previous = coords
            regime_coords[layer] = coords
            regime_fve[layer] = fve
            max_abs = max(max_abs, float(np.abs(coords).max()))
        coords_by_regime.append(regime_coords)
        fve_by_regime.append(regime_fve)
        scales.append(max_abs + 1e-9)

    return targets, coords_by_regime, fve_by_regime, scales, base_colors


def tinted_colors(base: np.ndarray, accent: str) -> np.ndarray:
    accent_rgb = np.array(matplotlib.colors.to_rgb(accent))
    paper_rgb = np.array(matplotlib.colors.to_rgb(PAPER))
    colors = 0.34 * base + 0.66 * accent_rgb
    return 0.86 * colors + 0.14 * paper_rgb


def projected_extent(coords: np.ndarray, rview: np.ndarray) -> float:
    p = coords @ rview.T
    return float(np.abs(p[:, :2]).max())


def style_2d_axis(ax):
    ax.set_facecolor(PAPER)
    ax.set_xlim(-1.16, 1.16)
    ax.set_ylim(-1.16, 1.16)
    ax.set_aspect("equal")
    ax.axis("off")


def add_fve_inset(ax, fve: np.ndarray, show_labels: bool = False):
    inset = ax.inset_axes([0.66, 0.70, 0.25, 0.22])
    n_bars = 5
    vals = np.zeros(n_bars)
    vals[: min(n_bars, len(fve))] = fve[:n_bars]
    x = np.arange(1, n_bars + 1)
    inset.bar(x, vals, color=INK, width=0.42)
    inset.set_ylim(0, 1.0)
    inset.set_xlim(0.45, n_bars + 0.55)
    inset.set_facecolor("#F8F4EA")
    for spine in inset.spines.values():
        spine.set_color("#D6D3C9")
        spine.set_linewidth(0.85)
    inset.tick_params(axis="both", colors=MUTED, labelsize=6, length=1.5, pad=1)
    if show_labels:
        inset.set_ylabel("FVE", color=INK, fontsize=7, labelpad=0.5)
        inset.set_xlabel("PC idx", color=INK, fontsize=7, labelpad=0.5)
        inset.set_yticks([0, 0.5, 1.0])
        inset.set_xticks([1, n_bars])
    else:
        inset.set_yticks([])
        inset.set_xticks([])


def render(
    out_png: Path,
    out_pdf: Path | None,
    width: int,
    depth: int,
    n_theta: int,
    sb: float,
    seed: int,
    show_insets: bool,
):
    accents = [ORDERED_BLUE, FOREST, CLAY]
    columns = [(1, r"$t=1$"), (depth // 2, r"$t=T/2$"), (depth, r"$t=T$")]
    snapshot_layers = [layer for layer, _ in columns]
    targets, coords, fves, _scales, base_colors = build_snapshot_data(
        width=width,
        depth=depth,
        n_theta=n_theta,
        sb=sb,
        seed=seed,
        snapshot_layers=snapshot_layers,
    )
    rview = mf.view_matrix(18, 35)
    column_scales = {
        layer: max(projected_extent(coords[row][layer], rview) for row in range(len(targets)))
        + 1e-9
        for layer, _ in columns
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.titleweight": "bold",
        }
    )

    fig = plt.figure(figsize=(12.8, 7.2), dpi=220)
    fig.patch.set_facecolor(PAPER)
    grid = fig.add_gridspec(
        3,
        3,
        left=0.075,
        right=0.985,
        top=0.92,
        bottom=0.045,
        wspace=0.035,
        hspace=0.055,
    )

    for row, ((chi, name), accent) in enumerate(zip(targets, accents)):
        colors = tinted_colors(base_colors, accent)
        for col, (layer, col_title) in enumerate(columns):
            ax = fig.add_subplot(grid[row, col])
            style_2d_axis(ax)
            mf.draw_loop(
                ax,
                coords[row][layer],
                colors,
                rview,
                column_scales[layer],
                lw=2.65,
                smooth_keep=22,
            )
            if show_insets:
                add_fve_inset(ax, fves[row][layer], show_labels=(row == 0 and col == 0))
            if row == 0:
                ax.set_title(col_title, color=INK, fontsize=15, pad=4)
            if col == 0:
                ax.text(
                    -0.13,
                    0.52,
                    f"{name}\n$\\chi_1={chi:.2f}$",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color=accent,
                    fontweight="bold",
                    linespacing=1.18,
                )

    fig.text(
        0.03,
        0.95,
        "A",
        ha="left",
        va="center",
        fontsize=17,
        color=INK,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=PAPER, bbox_inches="tight", pad_inches=0.04)
    if out_pdf is not None:
        fig.savefig(out_pdf, facecolor=PAPER, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main():
    asset_dir = Path(__file__).resolve().parents[1] / "assets"
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(asset_dir / "circle_times_grid.png"))
    parser.add_argument("--pdf", default=str(asset_dir / "circle_times_grid.pdf"))
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--ntheta", type=int, default=120)
    parser.add_argument("--sb", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--show-insets", action="store_true")
    args = parser.parse_args()

    out_png = Path(args.out)
    out_pdf = Path(args.pdf) if args.pdf else None
    render(
        out_png=out_png,
        out_pdf=out_pdf,
        width=args.N,
        depth=args.depth,
        n_theta=args.ntheta,
        sb=args.sb,
        seed=args.seed,
        show_insets=args.show_insets,
    )
    print(f"saved {out_png}")
    if out_pdf is not None:
        print(f"saved {out_pdf}")


if __name__ == "__main__":
    main()
