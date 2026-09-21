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
