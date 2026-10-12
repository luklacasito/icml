"""Mean-field figure shared by the notebook and manuscript exporter."""

import matplotlib.pyplot as plt
import numpy as np

from .mean_field import line_from_logfit
from .plot_style import FOREST, GOLD, INK, GRID


def style_log_axes(ax, legend_loc=None):
    """Standard grid and legend styling for log plots."""
    ax.grid(True, which="major", alpha=0.30, linestyle="-", linewidth=0.8, color=GRID)
    ax.grid(True, which="minor", alpha=0.15, linestyle=":", linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)
    if legend_loc is not None:
        ax.legend(loc=legend_loc, frameon=True, fancybox=False, framealpha=0.95)


def exponent_figure(critical, dropout):
    """Five panels with the notebook's original styles and fit labels."""
    point_style = dict(markersize=6, markeredgewidth=0.5, markeredgecolor="white", zorder=3)
    fit_style = dict(linewidth=2.5, zorder=2)
    fig = plt.figure(figsize=(22, 12))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])
    ax4 = fig.add_subplot(gs[1, 0:3])
    ax5 = fig.add_subplot(gs[1, 3:6])
    t_abs = np.abs(critical.tvals_below)
    ax1.loglog(t_abs, critical.xi, "o", color=FOREST, label="Data", **point_style)
    ax1.loglog(
        t_abs,
        critical.xi_fit,
        "--",
        color=FOREST,
        label=f"$\\nu = {critical.nu:.3f} \\pm {critical.se_nu:.3f}$",
        **fit_style,
    )
    ax1.set(xlabel="$|t| = |\\,\\chi - 1\\,|$", ylabel="$\\xi = -1/\\ln(\\chi)$")
    style_log_axes(ax1, legend_loc="lower left")
    ax2.loglog(
        critical.tvals_t, critical.m_t, "o", color=FOREST, label="Smooth tanh", **point_style
    )
    ax2.loglog(critical.tvals_r, critical.m_r, "s", color=GOLD, label="Kinked ReLU", **point_style)
    ax2.loglog(
        critical.tline,
        critical.m_fit_t,
        "--",
        color=FOREST,
        label=f"$\\beta_\\mathrm{{tanh}} = {critical.beta_t:.3f} \\pm {critical.se_beta_t:.3f}$",
        **fit_style,
    )
    ax2.loglog(
        critical.tline_r,
        critical.m_fit_r,
        "--",
        color=INK,
        label=f"$\\beta_\\mathrm{{ReLU}} = {critical.beta_r:.3f} \\pm {critical.se_beta_r:.3f}$",
        **fit_style,
    )
    ax2.set(xlabel="$t = \\chi - 1$", ylabel="$m_* = 1 - c_*$")
    style_log_axes(ax2, legend_loc="upper left")
    ax3.loglog(
        critical.ells,
        critical.m_l_t,
        color=FOREST,
        linewidth=1.5,
        alpha=0.7,
        zorder=2,
        label="Smooth tanh",
    )
    ax3.loglog(
        critical.ells,
        critical.m_l_r,
        color=GOLD,
        linewidth=1.5,
        alpha=0.7,
        zorder=2,
        label="Kinked ReLU",
    )
    ax3.loglog(
        critical.ells,
        critical.mfit_t,
        "--",
        color=FOREST,
        linewidth=2.5,
        zorder=3,
        label=f"$\\theta_{{\\mathrm{{rel,tanh}}}} = {critical.p_t:.3f} \\pm {critical.se_p_t:.3f}$",
    )
    ax3.loglog(
        critical.ells,
        critical.mfit_r,
        "--",
        color=INK,
        linewidth=2.5,
        zorder=3,
        label=f"$\\theta_{{\\mathrm{{rel}},\\mathrm{{ReLU}}}} = {critical.p_r:.3f} \\pm {critical.se_p_r:.3f}$",
    )
    ax3.set(xlabel="Layer $\\ell$", ylabel="$m_\\ell = 1 - c_\\ell$")
    style_log_axes(ax3, legend_loc="upper right")
    ax4.loglog(dropout.h_s, dropout.m_s, "o", color=FOREST, label="Smooth tanh", **point_style)
    ax4.loglog(dropout.h_k, dropout.m_k, "s", color=GOLD, label="Kinked ReLU", **point_style)
    hline = np.geomspace(
        min(dropout.h_k.min(), dropout.h_s.min()), max(dropout.h_k.max(), dropout.h_s.max()), 200
    )
    ax4.loglog(
        hline,
        line_from_logfit(hline, dropout.a_ms, dropout.b_ms),
        "--",
        color=FOREST,
        label=f"$1/\\delta_\\mathrm{{tanh}} = {dropout.a_ms:.3f} \\pm {dropout.se_a_ms:.3f}$",
        **fit_style,
    )
    ax4.loglog(
        hline,
        line_from_logfit(hline, dropout.a_mk, dropout.b_mk),
        "--",
        color=INK,
        label=f"$1/\\delta_\\mathrm{{ReLU}} = {dropout.a_mk:.3f} \\pm {dropout.se_a_mk:.3f}$",
        **fit_style,
    )
    ax4.set(xlabel="Dropout field $h = 1-\\bar{F}_\\rho(1)$", ylabel="$m_* = 1-c_*$")
    style_log_axes(ax4, legend_loc="upper left")
    ax5.loglog(dropout.h_s, dropout.xi_s, "o", color=FOREST, label="Smooth tanh", **point_style)
    ax5.loglog(dropout.h_k, dropout.xi_k, "s", color=GOLD, label="Kinked ReLU", **point_style)
    ax5.loglog(
        hline,
        line_from_logfit(hline, dropout.a_xs, dropout.b_xs),
        "--",
        color=FOREST,
        label=f"$\\nu_{{\\rho,\\mathrm{{tanh}}}} = {-dropout.a_xs:.3f} \\pm {dropout.se_nu_rho_s:.3f}$",
        **fit_style,
    )
    ax5.loglog(
        hline,
        line_from_logfit(hline, dropout.a_xk, dropout.b_xk),
        "--",
        color=INK,
        label=f"$\\nu_{{\\rho,\\mathrm{{ReLU}}}} = {-dropout.a_xk:.3f} \\pm {dropout.se_nu_rho_k:.3f}$",
        **fit_style,
    )
    ax5.set(
        xlabel="Dropout field $h$", ylabel="$\\xi = -1/\\ln\\lambda,\\ \\lambda=\\bar{F}'(c_*)$"
    )
    style_log_axes(ax5, legend_loc="upper right")
    return fig
