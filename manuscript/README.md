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

## Figures and tables

The [figure guide](FIGURES.md) lists every figure, its plotting command, input
measurements and training or calculation code. From the repository root:

```bash
python scripts/build_figures.py
python scripts/confidence_intervals.py
```

Figures go into `runs/figures/`; tables and intervals go into `runs/confidence/`.
To replace the manuscript's figure files deliberately, use
`python scripts/build_figures.py --output-dir manuscript/figures` and rebuild
the PDF. The supplied river and regularization-reach illustrations are retained.

Training intervals describe variation across seeds. The mean-field curves
are deterministic; their regression errors measure scatter around the fitted
power law. The captions distinguish these quantities.
