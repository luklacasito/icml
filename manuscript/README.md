# Camera-ready paper

[Read the PDF](../paper.pdf).

This is the source for the May 29, 2026 camera-ready release. It includes the
bibliography, ICML style files, and all 13 figures used in the paper.

With a LaTeX distribution and `latexmk` installed, run from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The output is `build/main.pdf`. The checked-in PDF at the repository root is
the original release; building this source reproduces its text and layout.

The paper's frozen code link refers to an older private archive. The cleaned
notebooks and saved results are in [this repository](../README.md).
