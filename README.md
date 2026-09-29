# Dropout Universality

**[Read the paper](paper.pdf)** · [LaTeX source](manuscript/) · [Figures and code](manuscript/FIGURES.md) · [Results](results/README.md)

Lucas Fernandez Sarmiento · ICML 2026

Keeping average dropout fixed, where should we place it across a network?
The paper develops a mean-field account of how dropout changes signal
propagation and tests the resulting schedules on vision, speech and financial
time series. Gains are most consistent in MLPs; transformer results are mixed.

The PDF and source contain the September 29, 2026 revision.

## Start here

| Directory | Contents |
|---|---|
| [manuscript/](manuscript/) | Paper source, figures and build instructions |
| [notebooks/](notebooks/) | Seven notebooks covering mean-field calculations, CIFAR schedules and sweeps |
| [benchmarks/](benchmarks/README.md) | Training code, dataset preparation and experiment settings |
| [scripts/](scripts/) | Figure and table generation |
| [results/](results/README.md) | Saved measurements, confidence intervals and their definitions |
| [tests/](tests/) | Numerical, training and result checks |
| [utils/](utils/) | Shared schedules, training functions and plotting style |

## Quick start

Use Python 3.11, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build_figures.py
```

Plots go into `runs/figures/`, using saved measurements and CPU mean-field
calculations. No training or dataset download is needed. The two supplied
vector illustrations are copied alongside the generated plots. The
[figure guide](manuscript/FIGURES.md) maps every figure to its code and data.
Run `jupyter lab` to explore the notebooks.

For new training runs, see the [benchmark guide](benchmarks/README.md),
[CIFAR notebooks and seed extensions](benchmarks/original/README.md), or
[sliding-block and correlation experiments](benchmarks/CIFAR.md).

Loss at the final epoch, minimum recorded test loss, and test loss at the epoch
chosen by validation are different measurements. The [results guide](results/README.md)
explains the comparisons and their limitations. Additional comparisons outside
the paper are indexed under [supplementary results](results/supplementary/README.md).

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check .
ruff format --check .
```

[Citation](CITATION.cff) · [MIT license](LICENSE)
