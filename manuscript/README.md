# Paper source

**[Read the paper](../paper.pdf).**

This is the source for the revised September 29, 2026 paper, including the
additional experiments, 94 additional fits for the retained comparisons, and
confidence intervals with separate seed counts for the models chosen by
validation loss and the final models. The original camera-ready
version is on [arXiv](https://arxiv.org/pdf/2605.21648v2).

## Build

With a LaTeX distribution and `latexmk` installed, run from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The output is `build/main.pdf`; the PDF at the repository root was built from
this source. The bibliography, ICML style files, and figures are all included.

## Reading the source

`main.tex` sets up the document and loads the files in `sections/`, following
the argument from signal propagation to dropout allocation and then to the
experiments.

| Files | Contents |
|---|---|
| `introduction.tex`, `background.tex`, `theory.tex` | The question, mean-field assumptions, critical scaling, and scheduling argument |
| `experiments.tex`, `discussion.tex` | Main comparisons, uncertainty, conclusions, and limitations |
| `original_results_table.tex`, `frontloaded_table.tex` | Tables computed from paired seed measurements, with each endpoint defined separately |
| `appendix_mean_field.tex`, `appendix_critical_scaling.tex`, `appendix_hermite.tex` | Dropout recursions, critical-exponent derivations, and the Hermite spectral interpretation |
| `appendix_original_experiments.tex` | CIFAR experiments and numerical fits |
| `experimental_appendix.tex`, `benchmark_methods.tex`, `fi2010_comparison_table.tex` | Additional datasets, reproducible protocols, and the standalone FI-2010 comparison |

## Rebuild figures and tables

Run these commands from the repository root:

| Command | Output |
|---|---|
| `python scripts/confidence_intervals.py` | Main tables and paired intervals in `runs/confidence/` |
| `python scripts/validation_cifar_results.py` | Separate CIFAR validation comparisons in `runs/confidence/` |
| `python scripts/plot_paper_results.py` | CIFAR curves and sweeps in `manuscript/figures/experiments/` |
| `python scripts/plot_sliding_block.py` | Block-position figure in `manuscript/figures/experiments/mlp/` |
| `python scripts/plot_cifar_correlations.py` | Correlation figure in `manuscript/figures/experiments/mlp/` |
| `python scripts/plot_benchmarks.py` | Additional-dataset curves in `runs/figures/benchmarks/` |
| `python scripts/plot_mean_field.py` | Critical exponents, scaling collapse, and Hermite figures in `runs/mean_field/` |

The [paired measurements](../results/confidence_seed_metrics.json) supply both
main tables. The benchmark exporter groups MLP and transformer panels by dataset;
use `--individual` for separate cohorts. It retains the original seed counts in
the archived curves; the main tables include the later seed extensions.
All figure exporters share `utils/plot_style.py`.

The two calculations carry different kinds of uncertainty: training seeds vary
from run to run, while the mean-field curves are deterministic and their fit
errors measure scatter around a fitted power law. The captions keep that
distinction explicit.
