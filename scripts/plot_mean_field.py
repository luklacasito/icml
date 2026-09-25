"""Export the critical-exponent, scaling-collapse, and Hermite figures.

The notebook remains the source of the equations and fitting windows. This
command runs only the cells used by the figure, on CPU, without Jupyter.
"""

from __future__ import annotations

import argparse
from contextlib import chdir
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.plot_style import FOREST, GOLD, INK, field_palette, paper_style  # noqa: E402
from utils.scaling import RELU_KAPPA, kinked_scaling, kinked_universal  # noqa: E402

NOTEBOOK = ROOT / "notebooks/mean_field.ipynb"
# Check the cell contents as well as their positions, so notebook edits cannot
# silently make this command execute an unrelated analysis.
CELLS = {
    2: "def gh_expect_1d",
    3: "def F_tanh",
    4: "def loglog_fit",
    6: "sigw2_c = brentq",
    8: "def tune_sigw2_for_chi_tanh",
    10: "# Combine the first two figures",
}
FITS = {
    "nu": ("nu", "se_nu"),
    "beta_tanh": ("beta_t", "se_beta_t"),
    "beta_relu": ("beta_r", "se_beta_r"),
    "p_tanh": ("p_t", "se_p_t"),
    "p_relu": ("p_r", "se_p_r"),
    "inverse_delta_tanh": ("a_ms", "se_a_ms"),
    "inverse_delta_relu": ("a_mk", "se_a_mk"),
    "nu_dropout_tanh": ("nu_rho_s", "se_nu_rho_s"),
    "nu_dropout_relu": ("nu_rho_k", "se_nu_rho_k"),
}
ARRAYS = (
    "tvals_below",
    "xi",
    "sigw2_grid",
    "tvals_t",
    "m_t",
    "tvals_r",
    "m_r",
    "ells",
    "m_l_t",
    "m_l_r",
    "fit_win",
    "p_list",
    "rho_list",
    "h_s",
    "m_s",
    "xi_s",
    "fit_s",
    "h_k",
    "m_k",
    "xi_k",
    "fit_k",
)


def hermite_coefficients(nmax=36):
    """Coefficients of ReLU(z) and tanh(z) in He_n(z) / sqrt(n!).

    ReLU uses its closed form; quadrature across its kink converges slowly.
    The smooth tanh coefficients retain the figure's 220-point quadrature.
    """
    if not isinstance(nmax, int) or nmax < 0:
        raise ValueError("The maximum Hermite degree must be a nonnegative integer")
    relu = np.zeros(nmax + 1)
    relu[0] = 1 / math.sqrt(2 * math.pi)
    if nmax >= 1:
        relu[1] = 0.5
    if nmax >= 2:
        relu[2] = relu[0] / math.sqrt(2)
    for n in range(2, nmax - 1, 2):
        relu[n + 2] = -relu[n] * (n - 1) / math.sqrt((n + 1) * (n + 2))

    nodes, weights = np.polynomial.hermite.hermgauss(220)
    z = math.sqrt(2) * nodes
    weights = weights / math.sqrt(math.pi)
    basis = [np.ones_like(z)]
    if nmax >= 1:
        basis.append(z)
    for n in range(1, nmax):
        basis.append((z * basis[-1] - math.sqrt(n) * basis[-2]) / math.sqrt(n + 1))
    tanh = np.array([np.sum(weights * np.tanh(z) * term) for term in basis])
    tanh[::2] = 0  # Odd parity gives exactly zero even coefficients.
    return relu, tanh


def plot_hermite(output):
    """Write the coefficient figure and its signed numerical values."""
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    relu, tanh = hermite_coefficients()
    degrees = np.arange(len(relu))
    style = {
        **paper_style(),
        "axes.labelsize": 12.5,
        "axes.titlesize": 13.5,
        "axes.titleweight": "regular",
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2.2,
        "lines.markersize": 5.4,
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(6.2, 3.7))
        for values, marker, color, label in (
            (relu, "s-", GOLD, "ReLU"),
            (tanh, "o-", FOREST, r"$\tanh$"),
        ):
            visible = np.abs(values) > 1e-13
            ax.semilogy(
                degrees[visible],
                np.abs(values[visible]),
                marker,
                color=color,
                label=label,
            )
        ax.set_xlabel("Hermite degree n")
        ax.set_ylabel(r"$|a_n|$")
        ax.set_title("Hermite coefficient decay")
        ax.legend()
        fig.savefig(output / "hermite_decomposition.pdf", bbox_inches="tight")
        fig.savefig(output / "hermite_decomposition.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    report = {
        "basis": "He_n(z) / sqrt(n!), z standard normal",
        "relu_method": "Exact closed-form coefficients, evaluated by recurrence",
        "tanh_method": "220-point Gauss-Hermite quadrature; even coefficients zero by parity",
        "degrees": degrees.tolist(),
        "relu": relu.tolist(),
        "tanh": tanh.tolist(),
    }
    (output / "hermite_coefficients.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


def plot_scaling_collapses(namespace, source, output):
    """Export full-map fixed points with the paper's critical scaling convention."""
    if "# Universal scaling functions" not in source:
        raise ValueError("Notebook collapse cell changed; review the export script")
    exec(compile(source, "mean_field.ipynb:cell12", "exec"), namespace)
    plt = namespace["plt"]
    plt.close(namespace["fig"])
    for kind in ("smooth", "kinked"):
        smooth = kind == "smooth"
        data = namespace["smooth" if smooth else "kink"]
        with plt.rc_context(
            {
                **paper_style(),
                "axes.labelsize": 14 if smooth else 12,
                "xtick.labelsize": 12 if smooth else 10,
                "ytick.labelsize": 12 if smooth else 10,
                "legend.fontsize": 10.5 if smooth else 9,
            }
        ):
            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4) if smooth else (10.4, 3.65))
            xmin = -1 if smooth else -1.25
            ymax = 0.0
            for proxy, color in zip(namespace["h_list"], field_palette(len(namespace["h_list"]))):
                rows = data[np.isclose(data[:, 0], proxy)]
                t, h = rows[:, 1], rows[:, 2]
                # Both scans fix rho. ReLU labels its critical-point field h0.
                legend_value = proxy / (1 + proxy) if smooth else proxy
                mantissa, exponent = f"{legend_value:.1e}".split("e")
                label = rf"${mantissa}\times10^{{{int(exponent)}}}$"
                if smooth:
                    g, m = rows[:, 3], rows[:, 4]
                    scaled_t = -t / np.sqrt(2 * g * h)
                    scaled_m = m * np.sqrt(g / (2 * h))
                else:
                    m = rows[:, 3]
                    scaled_t, scaled_m = kinked_scaling(t, h, m)
                visible = (scaled_t >= xmin) & (scaled_t <= 2)
                ymax = max(ymax, float(np.max(scaled_m[visible])))
                for ax, x, y in ((axes[0], t, m), (axes[1], scaled_t, scaled_m)):
                    ax.plot(
                        x,
                        y,
                        "o" if smooth else "s",
                        color=color,
                        markersize=3.5,
                        label=label,
                    )
            u = np.linspace(xmin, 2, 400)
            theory = np.sqrt(1 + u * u) - u if smooth else kinked_universal(u)
            axes[1].plot(u, theory, color=INK, linewidth=2.4, label="Theory")
            axes[0].set_xlabel(r"$t=\chi-1$")
            axes[0].set_ylabel(r"$m=1-c_\ast$")
            axes[1].set_xlabel(
                r"$\tilde t=-t/\sqrt{2g_\rho h}$"
                if smooth
                else r"$-u=(1-\chi)/(\kappa^{2/3}h^{1/3})$"
            )
            axes[1].set_ylabel(
                r"$\tilde m=m\sqrt{g_\rho/(2h)}$" if smooth else r"$m/(h/\kappa)^{2/3}$"
            )
            axes[1].set_xlim(xmin, 2)
            # Retain the original ReLU figure's near-critical display window.
            axes[1].set_ylim(0, np.ceil(4.2 * max(ymax, theory.max())) / 4 if smooth else 2)
            for ax, location in zip(axes, ("upper left", "upper right")):
                namespace["style_log_axes"](ax)
                ax.legend(
                    loc=location,
                    ncol=2,
                    columnspacing=0.7,
                    handletextpad=0.3,
                    title=r"$p=1-\rho$" if smooth else r"$h_0=1/\rho-1$",
                )
            fig.tight_layout(pad=1.0, w_pad=2.0)
            for suffix in ("pdf", "png"):
                fig.savefig(
                    output / f"{kind}_scaling_collapse.{suffix}",
                    dpi=150,
                    bbox_inches="tight",
                )
            plt.close(fig)
    np.savez_compressed(
        output / "scaling_collapse.npz",
        smooth=namespace["smooth"],
        kinked=namespace["kink"],
        kinked_kappa=np.full(len(namespace["kink"]), RELU_KAPPA),
    )
    report = {
        "source": "notebooks/mean_field.ipynb:cell12",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "smooth_columns": ["proxy", "t", "h", "g", "m"],
        "kinked_columns": ["h0", "t", "h", "m"],
        "fixed_parameter": {
            "smooth": "Each curve fixes rho=1/(1+proxy); legend p=1-rho; exact h varies.",
            "kinked": "Each curve fixes rho=1/(1+h0); h0 is the field at chi=1; actual h=chi*h0 varies.",
        },
        "kinked_kappa": "Critical coefficient 2*sqrt(2)/(3*pi), constant across the scan.",
        "horizontal_axes": "Smooth: tilde_t=-t/sqrt(2*g*h). Kinked: -u=-t/(kappa**(2/3)*h**(1/3)), with the paper's critical kappa.",
        "kinked_display_window": {"x": [-1.25, 2], "y": [0, 2]},
        "scope": "Smooth uses tanh mean-field fixed points. Kinked solves the full formal map F(c)=chi*K(c)+1-chi/rho at fixed rho; points are not normal-form roots. Finite-field and finite-t deviations remain, including the full map's chi-dependent kink coefficient.",
        "solver_sha256": hashlib.sha256((ROOT / "utils/scaling.py").read_bytes()).hexdigest(),
    }
    (output / "scaling_collapse.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/mean_field")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    notebook = json.loads(NOTEBOOK.read_text())
    sources = {}
    for index, marker in CELLS.items():
        cell = notebook["cells"][index]
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or marker not in source:
            raise ValueError(f"Notebook cell {index} changed; review the export script")
        sources[index] = source

    namespace = {"__name__": "mean_field"}
    with chdir(ROOT):
        for index, source in sources.items():
            exec(compile(source, f"mean_field.ipynb:cell{index}", "exec"), namespace)
            if index == 2:
                namespace["RUN_DIR"] = output

    figure = namespace["fig"]
    figure.axes[0].lines[0].set_label("Exact map relation")
    namespace["style_log_axes"](figure.axes[0], legend_loc="lower left")
    # Retain small regression errors instead of rounding them to zero.
    for line, symbol, value, error in (
        (figure.axes[2].lines[2], r"p_{\mathrm{tanh}}", "p_t", "se_p_t"),
        (figure.axes[2].lines[3], r"p_{\mathrm{ReLU}}", "p_r", "se_p_r"),
        (figure.axes[3].lines[3], r"1/\delta_{\mathrm{ReLU}}", "a_mk", "se_a_mk"),
    ):
        line.set_label(rf"${symbol} = {namespace[value]:.5f} \pm {namespace[error]:.5f}$")
    namespace["style_log_axes"](figure.axes[2], legend_loc="upper right")
    namespace["style_log_axes"](figure.axes[3], legend_loc="upper left")
    # At manuscript width this gives approximately 7–8 pt labels and legends.
    figure.set_size_inches(11.5, 7.4)
    figure.subplots_adjust(left=0.08, right=0.99, bottom=0.10, top=0.98, hspace=0.55, wspace=0.75)
    figure.axes[0].get_subplotspec().get_gridspec().update(wspace=0.75, hspace=0.50)
    for ax in figure.axes:
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        ax.tick_params(axis="both", labelsize=12)
        for label in ax.get_legend().get_texts():
            label.set_fontsize(12)
    figure.savefig(output / "critical_exponents.pdf", bbox_inches="tight")
    figure.savefig(output / "critical_exponents.png", dpi=150, bbox_inches="tight")
    namespace["plt"].close(figure)
    np.savez_compressed(output / "mean_field.npz", **{name: namespace[name] for name in ARRAYS})
    report = {
        "source": "notebooks/mean_field.ipynb",
        "code_cells": list(CELLS),
        "source_sha256": hashlib.sha256("\n".join(sources.values()).encode()).hexdigest(),
        "method": "Deterministic mean-field recursions with 60-point Gauss-Hermite quadrature for tanh.",
        "uncertainty": "Ordinary least-squares log-log fit standard errors; not seed confidence intervals or numerical discretization errors.",
        "relu_scope": "The tunable ReLU map is a formal local family; chi > 1 and chi = 1 with dropout need not have nonnegative bias variance.",
        "fit_windows": {
            "nu": "All 20 |chi - 1| values from 0.01 to 0.1.",
            "beta": "The first 10 (smallest chi - 1) values per activation.",
            "p": "Layers 501 through 7999.",
            "dropout": "Fields h <= 10 times the smallest field across the two activations.",
        },
        "critical_tanh": {
            name: float(namespace[name]) for name in ("sigw2_c", "q_c", "g_c", "h_c")
        },
        "fits": {
            name: {
                "estimate": float(namespace[value]),
                "fit_se": float(namespace[error]),
            }
            for name, (value, error) in FITS.items()
        },
    }
    (output / "mean_field.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot_hermite(output)
    plot_scaling_collapses(namespace, "".join(notebook["cells"][12]["source"]), output)
    print(f"Figure, raw curves, and fit report: {output}")


if __name__ == "__main__":
    main()
