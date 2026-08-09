#!/usr/bin/env python3
"""
Styled mean-field manifold folding animation for the ICML talk.

This is a self-contained version of the Poole et al. transient-chaos picture:
a circular input loop is propagated through fixed random tanh networks in three
initialization regimes, projected to 2-D, and rendered in the paper palette.
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from numpy.polynomial.hermite_e import hermegauss


INK = "#202121"
FOREST = "#0E3B2C"
SAGE = "#6F7F70"
GOLD = "#D8AD62"
GILT = "#9E7B38"
PAPER = "#F4EFE3"
MUTED = "#60645F"
CLAY = "#B04A2F"
ORDERED_BLUE = "#3A6EA5"
ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"

_NODES, _WEIGHTS = hermegauss(100)
_GNORM = np.sqrt(2.0 * np.pi)


def gaussian_expect(f):
    return np.sum(_WEIGHTS * f(_NODES)) / _GNORM


def q_star(sw2, sb2, iters=400, tol=1e-11):
    q = 1.0
    for _ in range(iters):
        s = np.sqrt(max(q, 1e-12))
        nq = sw2 * gaussian_expect(lambda z: np.tanh(s * z) ** 2) + sb2
        if abs(nq - q) < tol:
            break
        q = nq
    return max(q, 1e-12)


def chi1(sw2, sb2):
    q = q_star(sw2, sb2)
    s = np.sqrt(q)
    return sw2 * gaussian_expect(lambda z: (1.0 / np.cosh(s * z) ** 2) ** 2)


def solve_for_chi(target, sb2, lo=0.05, hi=10.0, iters=70):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if chi1(mid, sb2) < target:
            lo = mid
        else:
            hi = mid
    return np.sqrt(0.5 * (lo + hi))


def propagate_loop(sw, sb, width, depth, n_theta, seed):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    # Smooth closed curve embedded in width dimensions. We deliberately avoid
    # adding random high-dimensional fuzz to the input; the point of the slide is
    # the evolution of one clean manifold, not finite-sample texture.
    x = np.zeros((n_theta, width))
    x[:, 0] = np.cos(theta)
    x[:, 1] = np.sin(theta)
    x[:, 2] = 0.35 * np.cos(2 * theta)
    x[:, 3] = 0.35 * np.sin(2 * theta)

    states = [x]
    h = x
    for _ in range(depth):
        w = rng.standard_normal((width, width)) * (sw / np.sqrt(width))
        b = rng.standard_normal(width) * sb
        h = np.tanh(h @ w + b)
        states.append(h)
    return np.asarray(states), theta


def pca3(state):
    x = state - state.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:3].T
    return coords, s[:3], s[:3] ** 2


def fixed_pca_basis(states):
    x = states.reshape(-1, states.shape[-1])
    mean = x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    return mean, vt[:3]


def procrustes(coords, previous):
    if previous is None:
        return coords
    m = coords.T @ previous
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return coords @ r


def view_matrix(elev_deg, azim_deg):
    e = np.radians(elev_deg)
    a = np.radians(azim_deg)
    ry = np.array(
        [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    )
    rx = np.array(
        [[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]]
    )
    return rx @ ry


def cyclic_palette(theta):
    phase = theta / (2 * np.pi)
    gold = np.array([216, 173, 98]) / 255.0
    forest = np.array([14, 59, 44]) / 255.0
    sage = np.array([111, 127, 112]) / 255.0
    ink = np.array([32, 33, 33]) / 255.0
    colors = []
    for p in phase:
        if p < 1 / 3:
            a = p * 3
            c = (1 - a) * gold + a * forest
        elif p < 2 / 3:
            a = (p - 1 / 3) * 3
            c = (1 - a) * forest + a * sage
        else:
            a = (p - 2 / 3) * 3
            c = (1 - a) * sage + a * gold
        colors.append(0.88 * c + 0.12 * ink)
    return np.asarray(colors)


def lowpass_periodic(x, keep):
    coeff = np.fft.rfft(x, axis=0)
    coeff[keep + 1 :] = 0
    return np.fft.irfft(coeff, n=len(x), axis=0)


def draw_loop(ax, coords3d, base_rgb, rview, scale, lw=2.35, upsample=4, smooth_keep=18):
    p = coords3d @ rview.T
    if upsample > 1:
        n0 = len(p)
        ext = np.vstack([p[-2:], p, p[:2]])
        u0 = np.arange(-2, n0 + 2)
        uf = np.linspace(0, n0, n0 * upsample, endpoint=False)
        p = np.stack([np.interp(uf, u0, ext[:, d]) for d in range(3)], axis=1)
        cu = np.linspace(0, 1, len(base_rgb), endpoint=False)
        ufn = (uf % n0) / n0
        base_rgb = np.stack(
            [np.interp(ufn, cu, base_rgb[:, d], period=1.0) for d in range(3)],
            axis=1,
        )
    p = lowpass_periodic(p, keep=smooth_keep)

    xy = p[:, :2]
    n = len(xy)
    idx = np.arange(n)
    seg_i = np.stack([idx, np.roll(idx, -1)], axis=1)
    segs = np.stack([xy[seg_i[:, 0]], xy[seg_i[:, 1]]], axis=1) / scale
    seg_c = 0.5 * (base_rgb[seg_i[:, 0]] + base_rgb[seg_i[:, 1]])

    paper_rgb = np.array([244, 239, 227]) / 255.0
    col = 0.84 * seg_c + 0.16 * paper_rgb

    glow = LineCollection(
        segs,
        colors=np.clip(col, 0, 1),
        linewidths=lw * 2.0,
        alpha=0.09,
        capstyle="round",
    )
    ax.add_collection(glow)
    rgba = np.concatenate([np.clip(col, 0, 1), np.full((n, 1), 0.92)], axis=1)
    main = LineCollection(
        segs, colors=rgba, linewidths=lw, capstyle="round", joinstyle="round"
    )
    ax.add_collection(main)


def effective_dim(state):
    x = state - state.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(x, compute_uv=False)
    ev = sv**2
    return (ev.sum() ** 2) / (np.sum(ev**2) + 1e-12)


def build_states(width, depth, n_theta, sb, seed, projection="layer"):
    sb2 = sb**2
    targets = [(0.85, "ordered"), (1.00, "edge-of-chaos"), (1.16, "chaotic")]
    sigmas = [solve_for_chi(t, sb2) for t, _ in targets]
    raw = [propagate_loop(sw, sb, width, depth, n_theta, seed + k) for k, sw in enumerate(sigmas)]
    theta = raw[0][1]
    colors = cyclic_palette(theta)

    coords_cache = []
    scales = []
    edims = []
    for states, _ in raw:
        previous = None
        tmp = []
        maxr = 0.0
        mean = basis = None
        if projection == "fixed":
            mean, basis = fixed_pca_basis(states)
        for layer in range(depth + 1):
            if projection == "fixed":
                coords = (states[layer] - mean) @ basis.T
            else:
                coords, _, _ = pca3(states[layer])
                coords = procrustes(coords, previous)
                previous = coords
            tmp.append(coords)
            maxr = max(maxr, np.abs(coords).max())
        coords_cache.append(tmp)
        scales.append(maxr + 1e-9)
        edims.append([effective_dim(states[layer]) for layer in range(depth + 1)])
    return targets, sigmas, coords_cache, scales, edims, colors


def coords_at_layer(coords_for_regime, layer_value):
    layer0 = int(np.floor(layer_value))
    layer1 = min(layer0 + 1, len(coords_for_regime) - 1)
    alpha = layer_value - layer0
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return (1.0 - alpha) * coords_for_regime[layer0] + alpha * coords_for_regime[layer1]


def render_frame(fig, axes, layer_value, targets, sigmas, coords, scales, edims, colors, rview):
    accents = [ORDERED_BLUE, FOREST, CLAY]
    ordered_rgb = np.array([58, 110, 165]) / 255.0
    forest_rgb = np.array([14, 59, 44]) / 255.0
    clay_rgb = np.array([176, 74, 47]) / 255.0
    for k, ax in enumerate(axes):
        ax.cla()
        ax.set_facecolor(PAPER)
        ax.set_xlim(-1.12, 1.12)
        ax.set_ylim(-1.12, 1.12)
        ax.set_aspect("equal")
        ax.axis("off")
        if k == 0:
            panel_colors = 0.28 * colors + 0.72 * ordered_rgb
        elif k == 1:
            panel_colors = 0.28 * colors + 0.72 * forest_rgb
        elif k == 2:
            panel_colors = 0.28 * colors + 0.72 * clay_rgb
        else:
            panel_colors = colors
        draw_loop(ax, coords_at_layer(coords[k], layer_value), panel_colors, rview, scales[k])
        chi, name = targets[k]
        ax.text(
            0,
            1.22,
            name,
            color=accents[k],
            fontsize=10.8,
            ha="center",
            va="center",
            fontweight="bold",
            transform=ax.transData,
        )
    fig.text(
        0.5,
        0.075,
        f"layer {int(round(layer_value))}",
        ha="center",
        va="center",
        fontsize=9,
        color=MUTED,
    )


def main(
    out=None,
    frame_dir=None,
    width=420,
    depth=32,
    n_theta=240,
    hold=3,
    fps=18,
    sb=0.3,
    seed=3,
    elev=18,
    azim=35,
    projection="layer",
    interp=1,
):
    if out is None:
        out = str(ASSET_DIR / "manifold_folding_paper.gif")
    if frame_dir is None:
        frame_dir = str(ASSET_DIR / "manifold_frames")
    targets, sigmas, coords, scales, edims, colors = build_states(
        width, depth, n_theta, sb, seed, projection=projection
    )
    rview = view_matrix(elev, azim)
    layer_seq = []
    for layer in range(depth):
        for step in range(interp):
            layer_seq.append(layer + step / interp)
    layer_seq.append(float(depth))
    layer_seq = np.repeat(np.asarray(layer_seq), hold)

    fig = plt.figure(figsize=(8.9, 3.05), dpi=150)
    fig.patch.set_facecolor(PAPER)
    axes = [
        fig.add_axes([0.025 + k * 0.325, 0.10, 0.30, 0.78], facecolor=PAPER)
        for k in range(3)
    ]

    frame_path = Path(frame_dir)
    frame_path.mkdir(parents=True, exist_ok=True)

    def draw_frame(frame_idx):
        layer = float(layer_seq[frame_idx])
        fig.texts.clear()
        render_frame(fig, axes, layer, targets, sigmas, coords, scales, edims, colors, rview)
        return []

    for i in range(len(layer_seq)):
        draw_frame(i)
        fig.savefig(frame_path / f"manifold_{i}.png", facecolor=PAPER)

    anim = FuncAnimation(fig, draw_frame, frames=len(layer_seq), interval=1000 / fps, blit=False)
    try:
        anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=4200), savefig_kwargs={"facecolor": PAPER})
        print(f"saved {out}")
    except Exception as exc:
        gif = os.path.splitext(out)[0] + ".gif"
        print(f"ffmpeg unavailable ({exc}); saving gif instead")
        anim.save(gif, writer=PillowWriter(fps=fps), savefig_kwargs={"facecolor": PAPER})
        print(f"saved {gif}")
    plt.close(fig)
    print(f"saved {len(layer_seq)} frames to {frame_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--frame-dir", default=None)
    parser.add_argument("--N", type=int, default=420)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--ntheta", type=int, default=240)
    parser.add_argument("--hold", type=int, default=3)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--sb", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--projection", choices=["layer", "fixed"], default="layer")
    parser.add_argument("--interp", type=int, default=1, help="visual in-between frames per layer")
    args = parser.parse_args()
    main(
        out=args.out,
        frame_dir=args.frame_dir,
        width=args.N,
        depth=args.depth,
        n_theta=args.ntheta,
        hold=args.hold,
        fps=args.fps,
        sb=args.sb,
        seed=args.seed,
        projection=args.projection,
        interp=max(1, args.interp),
    )
