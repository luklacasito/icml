#!/usr/bin/env python3
"""
mft_propagation_video.py
========================
Animate a signal image propagating through three tanh mean-field regimes:
ordered, critical, and chaotic.

This renders the mean-field survival fraction directly rather than applying a
dense random matrix to pixels. The goal is a clean visual explanation of deep
information propagation: ordered networks erase the signal, critical networks
preserve it, and chaotic networks drown it in noise.

Usage:
    python mft_propagation_video.py
    python mft_propagation_video.py --image path/to.png
    python mft_propagation_video.py --out out.mp4 --depth 40 --frames 90
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from numpy.polynomial.hermite_e import hermegauss


# ----------------------------------------------------------------------------
# Mean-field theory
# ----------------------------------------------------------------------------
_NODES, _WEIGHTS = hermegauss(120)
_GNORM = np.sqrt(2.0 * np.pi)


def gaussian_expect(f):
    """E_{z~N(0,1)}[f(z)] via probabilist's Gauss-Hermite quadrature."""
    return np.sum(_WEIGHTS * f(_NODES)) / _GNORM


def q_star(sw2, sb2, iters=500, tol=1e-12):
    """Fixed point q* = sw2 E[tanh(sqrt(q*) z)^2] + sb2."""
    q = 1.0
    for _ in range(iters):
        s = np.sqrt(max(q, 1e-12))
        nq = sw2 * gaussian_expect(lambda z: np.tanh(s * z) ** 2) + sb2
        if abs(nq - q) < tol:
            break
        q = nq
    return max(q, 1e-12)


def chi1(sw2, sb2):
    """Per-layer correlation multiplier chi1 = sw2 E[sech(sqrt(q*) z)^4]."""
    q = q_star(sw2, sb2)
    s = np.sqrt(q)
    return sw2 * gaussian_expect(lambda z: (1.0 / np.cosh(s * z) ** 2) ** 2)


def xi_of(sw2, sb2):
    """Correlation length in layers."""
    c = chi1(sw2, sb2)
    ln = np.log(c)
    xi = np.inf if abs(ln) < 2e-3 else 1.0 / abs(ln)
    return xi, c


def solve_critical(sb2, lo=0.5, hi=6.0, iters=60):
    """Bisection for chi1 = 1."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if chi1(mid, sb2) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------
# Signal image
# ----------------------------------------------------------------------------
def load_signal(image_path=None, width=360, height=180):
    """Return a standardized HxW signal. White background is positive."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("L").resize((width, height))
        g = np.asarray(img, dtype=float) / 255.0
    else:
        img = Image.new("L", (width, height), 246)
        draw = ImageDraw.Draw(img)

        scale_x = width / 240
        scale_y = height / 120

        def sx(v):
            return int(round(v * scale_x))

        def sy(v):
            return int(round(v * scale_y))

        blocks = [
            (12, 32, 18),
            (32, 16, 18),
            (32, 40, 22),
            (54, 24, 16),
            (16, 60, 20),
            (40, 60, 18),
            (60, 52, 16),
        ]
        for x, y, s in blocks:
            draw.rounded_rectangle(
                [sx(x), sy(y), sx(x + s), sy(y + s)],
                radius=max(2, sx(2)),
                fill=22,
            )

        draw.ellipse([sx(78), sy(28), sx(118), sy(92)], fill=22)
        draw.ellipse([sx(85), sy(37), sx(111), sy(83)], fill=246)
        try:
            font = ImageFont.truetype("DejaVuSerif-Bold.ttf", sy(56))
        except Exception:
            font = ImageFont.load_default()
        draw.text((sx(124), sy(30)), "ICML", fill=22, font=font)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.15))
        g = np.asarray(img, dtype=float) / 255.0

    return (g - g.mean()) / (g.std() + 1e-8)


def vignette(shape, strength=0.10):
    """Subtle edge falloff for a less clinical frame."""
    h, w = shape
    yy, xx = np.mgrid[-1:1 : complex(h), -1:1 : complex(w)]
    rr = np.sqrt(xx * xx + yy * yy)
    return np.clip(1.0 - strength * rr, 0.0, 1.0)


def survived_image(z, sw2, sb2, depth_l, noise, paper_tone):
    """Display array in [0,1] after depth_l layers."""
    xi, c = xi_of(sw2, sb2)
    r = np.exp(-depth_l / xi) if np.isfinite(xi) else 1.0
    out = r * z
    if c > 1.0 and depth_l > 0:
        noise_gain = np.sqrt(max(0.0, 1.0 - r * r))
        out = out + 1.10 * noise_gain * noise

    disp = np.clip(0.53 + 0.23 * out, 0.0, 1.0)
    disp = 0.88 * disp + 0.12 * paper_tone
    return np.clip(disp, 0.0, 1.0), r


def make_video(
    image_path=None,
    out="mft_propagation.mp4",
    sb2=0.05,
    depth=40,
    frames=90,
    fps=22,
    seed=0,
):
    rng = np.random.default_rng(seed)
    z = load_signal(image_path)
    noise = rng.standard_normal(z.shape)
    tone = vignette(z.shape)

    sw_crit = solve_critical(sb2)
    configs = [
        ("Ordered", sw_crit * 0.45, "#2F6CA3", "signal fades"),
        ("Critical", sw_crit, "#18836B", "signal persists"),
        ("Chaotic", sw_crit * 1.80, "#B2572A", "noise dominates"),
    ]
    stats = [(name, sw2, col, tag) + xi_of(sw2, sb2) for name, sw2, col, tag in configs]

    for name, sw2, _, _, xi, c in stats:
        xs = "inf" if not np.isfinite(xi) else f"{xi:.2f}"
        print(f"{name:9s} sw2={sw2:.3f}  chi1={c:.3f}  xi={xs} layers")

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.15), dpi=150)
    fig.patch.set_facecolor("#F6F2EA")
    panel_bg = "#FBF8F1"

    image_objects = []
    label_objects = []
    for ax, (name, sw2, color, tag, xi, c) in zip(axes, stats):
        disp0, _ = survived_image(z, sw2, sb2, 0.0, noise, tone)
        obj = ax.imshow(disp0, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        ax.set_facecolor(panel_bg)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.6)
            spine.set_color(color)

        xi_text = r"\infty" if not np.isfinite(xi) else f"{xi:.1f}"
        ax.text(
            0.04,
            0.93,
            name,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
            fontweight="bold",
            color=color,
        )
        ax.text(
            0.04,
            0.80,
            rf"$\chi_1={c:.2f}$   $\xi={xi_text}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            color="#2D3340",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#F6F2EA", edgecolor="none", alpha=0.82),
        )
        label = ax.text(
            0.04,
            0.08,
            tag,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            color="#2D3340",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F6F2EA", edgecolor="none", alpha=0.85),
        )
        image_objects.append(obj)
        label_objects.append(label)

    depth_text = fig.text(
        0.5,
        0.955,
        "depth  $\\ell = 0$",
        ha="center",
        va="center",
        fontsize=13,
        color="#2D3340",
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.045,
        "Mean-field signal propagation: ordered / critical / chaotic",
        ha="center",
        va="center",
        fontsize=11,
        color="#58606D",
    )
    plt.subplots_adjust(left=0.035, right=0.965, top=0.885, bottom=0.135, wspace=0.055)

    def update(frame):
        depth_l = depth * frame / max(frames - 1, 1)
        for obj, (name, sw2, color, tag, xi, c) in zip(image_objects, stats):
            disp, _ = survived_image(z, sw2, sb2, depth_l, noise, tone)
            obj.set_data(disp)
        depth_text.set_text(rf"depth  $\ell = {depth_l:0.1f}$")
        return image_objects + label_objects + [depth_text]

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)

    try:
        anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=4800))
        print(f"saved {out}")
    except Exception as exc:
        gif = os.path.splitext(out)[0] + ".gif"
        print(f"ffmpeg unavailable ({exc}); saving gif instead")
        anim.save(gif, writer=PillowWriter(fps=fps))
        print(f"saved {gif}")
    plt.close(fig)


def forward_conv(z, sw2, sb2, n_layers, rng, k=3):
    """Optional literal structure-preserving random conv forward pass."""
    from numpy.lib.stride_tricks import sliding_window_view

    x = z.copy()
    fan_in = k * k
    for _ in range(n_layers):
        pad = np.pad(x, k // 2, mode="wrap")
        patches = sliding_window_view(pad, (k, k))
        w = rng.standard_normal((k, k)) * np.sqrt(sw2 / fan_in)
        b = rng.standard_normal() * np.sqrt(sb2)
        pre = np.einsum("hwij,ij->hw", patches, w) + b
        x = np.tanh(pre)
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", default=None, help="input image path")
    parser.add_argument("--out", default="mft_propagation.mp4", help="output video path")
    parser.add_argument("--sb2", type=float, default=0.05, help="bias variance sigma_b^2")
    parser.add_argument("--depth", type=float, default=40, help="max depth in layers")
    parser.add_argument("--frames", type=int, default=90, help="number of frames")
    parser.add_argument("--fps", type=int, default=22, help="frames per second")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    make_video(
        image_path=args.image,
        out=args.out,
        sb2=args.sb2,
        depth=args.depth,
        frames=args.frames,
        fps=args.fps,
        seed=args.seed,
    )
