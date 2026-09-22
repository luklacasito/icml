# Paper source

**[Read the paper](../paper.pdf).**

This is the source for the expanded September 22, 2026 version, including the
additional experiments, the 114-fit seed extension, and confidence intervals
with separate checkpoint/final seed counts. The original camera-ready
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
| `original_results_table.tex`, `frontloaded_table.tex` | Tables computed from paired seed measurements |
| `appendix_mean_field.tex`, `appendix_critical_scaling.tex`, `appendix_hermite.tex` | Dropout recursions, critical-exponent derivations, and the Hermite spectral interpretation |
| `appendix_original_experiments.tex` | CIFAR experiments and numerical fits |
| `experimental_appendix.tex`, `benchmark_methods.tex`, `benchmark_table.tex`, `benchmark_discussion.tex` | Additional datasets, reproducible protocols, and validation-selected comparisons |

The [confidence-interval script](../scripts/confidence_intervals.py) computes
both main tables from the [paired seed measurements](../results/confidence_seed_metrics.json).
For the critical-exponent, scaling-collapse, and Hermite figures, run
`python scripts/plot_mean_field.py` from the repository root.

The two calculations carry different kinds of uncertainty: training seeds vary
from run to run, while the mean-field curves are deterministic and their fit
errors measure scatter around a fitted power law. The captions keep that
distinction explicit.
