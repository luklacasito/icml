# Paper source

[Read the PDF](../paper.pdf).

This is the source for the expanded September 21, 2026 paper, including the
additional experimental appendix and confidence intervals in the two main
results tables. It includes the bibliography, ICML style files, and every
referenced figure. The original camera-ready release remains on
[arXiv](https://arxiv.org/pdf/2605.21648v2).

With a LaTeX distribution and `latexmk` installed, run from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The output is `build/main.pdf`. The checked-in PDF at the repository root was
built from this source.

The [confidence-interval script](../scripts/confidence_intervals.py) regenerates
the main gain tables from [paired seed metrics](../results/confidence_seed_metrics.json).
The original notebooks and saved results are in [this repository](../README.md).

## Source map

`main.tex` contains the preamble and document order. The text is in `sections/`:

| Files | Contents |
|---|---|
| `introduction.tex`, `background.tex`, `theory.tex` | Motivation, assumptions, scaling, and scheduling |
| `experiments.tex`, `discussion.tex` | Main results, uncertainty, conclusions, and limitations |
| `original_results_table.tex`, `frontloaded_table.tex` | Tables generated from the paired seed metrics |
| `appendix_mean_field.tex`, `appendix_critical_scaling.tex`, `appendix_hermite.tex` | Derivations |
| `appendix_original_experiments.tex` | Original CIFAR experiments and numerical fits |
| `experimental_appendix.tex`, `benchmark_table.tex`, `benchmark_discussion.tex` | Additional datasets and validation-selected comparisons |

The critical-exponent, scaling-collapse, and Hermite figures can be recomputed with
`python scripts/plot_mean_field.py` from the repository root. Regression errors
on deterministic numerical curves are distinct from uncertainty across training
seeds. The paper states this distinction in the corresponding captions.
