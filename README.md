# Dropout Universality

**[Read the paper](paper.pdf)** · [LaTeX source](manuscript/) · [Results and confidence intervals](results/confidence_intervals.md)

*Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos*

Lucas Fernandez Sarmiento · ICML 2026

Dropout perturbs a signal as it travels through a network, so where we apply it
should matter. I study this through mean-field theory, treating dropout as a
field that can vary across depth, and ask how a fixed budget should be
allocated to preserve information while still regularizing the model.

An early mask affects many later layers, giving a concrete reason to try
frontloaded schedules. The paper develops this intuition through a propagation
approximation and a regularization-reach argument, then tests the schedules in
MLPs and transformers. The gains depend on the task and on when we measure
them: final loss, lowest recorded loss, and loss at a validation-selected
checkpoint can tell quite different stories.

The PDF is the revised September 22, 2026 paper, with the additional experiments,
confidence intervals, and corrections to the numerical figures and theoretical
claims. The original [camera-ready version is on arXiv](https://arxiv.org/pdf/2605.21648v2).

## Finding things

The [paper](paper.pdf) is at the top level. Its [source](manuscript/) is split by
section, with the experimental details and learning curves in the appendices.
[Saved results](results/) let you inspect the comparisons without training
anything; [scripts](scripts/) reproduce the plots and confidence intervals,
and [utils](utils/) holds the shared schedules, training loops, and loaders.

## Calculations and experiments

| Notebook | Question |
|---|---|
| [Mean field](notebooks/mean_field.ipynb) | How does dropout change critical scaling for smooth and kinked activations? |
| [MLP schedules](notebooks/mlp_schedules.ipynb) | Where should we put dropout across depth on CIFAR-10? |
| [Dropout-budget sweep](notebooks/mlp_dropout_sweep.ipynb) | How does the comparison change as we increase dropout? |
| [Width sweep](notebooks/mlp_width_sweep.ipynb) | What survives as the MLP width goes from 64 to 1024? |
| [GELU sweep](notebooks/mlp_gelu.ipynb) | Does the scheduling idea carry over to a smooth activation? |
| [ViT schedules](notebooks/vit_schedules.ipynb) | How do depth profiles compare in a CIFAR-100 transformer? |
| [ViT ablations](notebooks/vit_ablation.ipynb) | Should dropout act on attention, the MLP branch, or both? |

The MLP budget controls compare early concentration with simply increasing
uniform dropout; their [saved runs](results/mlp_budget_controls.npz) and
[figure](manuscript/figures/experiments/mlp/dropout_budget_comparison.pdf) are
included too. The additional datasets are in the paper's experimental appendix,
with paired test measurements in [results](results/).

## Running the code

Use Python 3.11 and run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

The mean-field calculations run on CPU; a CUDA GPU is preferable for training.
Each notebook explains which cells to run for the saved results and which train
new models. CIFAR downloads go into `data/`, reruns into `runs/`, and W&B logging
is off by default.

For the saved learning curves:

```bash
python scripts/plot_results.py
```

This writes to `runs/figures/`. Add `--sweeps` for every saved dropout strength
and width. To recompute the critical exponents, scaling collapse, and Hermite
coefficients:

```bash
python scripts/plot_mean_field.py
```

The numerical inputs, fits, and figures go into `runs/mean_field/`. The figures
used in the paper are in [manuscript/figures](manuscript/figures/).

## Additional datasets

The [benchmark guide](benchmarks/README.md) covers Speech Commands, Jannis,
Tiny ImageNet, and FI-2010. It includes data preparation, the retained models,
and [frozen configurations and split identifiers](benchmarks/protocols.json)
for the reported profiles. Training saves the validation-selected and final
checkpoints separately.

The [standalone financial MLP guide](benchmarks/VANILLA.md) reproduces its
separate data pipeline and learning-rate search. Its historical selected
learning rates were not preserved; the guide distinguishes a new search from
reproducing the archived results.

## Reading the comparisons

I report loss reductions relative to uniform dropout and accuracy gains in
percentage points, keeping the same schedule across the endpoints in each row.
The [results table](results/confidence_intervals.md) gives paired 95% intervals:
Fieller intervals for relative loss reductions and Student-t intervals for
accuracy gains. They describe variation across seeds on the recorded split,
conditional on the selected schedules and hyperparameters.

The endpoint matters. Taking the lowest recorded test loss uses the test set to
choose an epoch, whereas a validation-selected checkpoint is chosen before its
test evaluation. The paper keeps these comparisons separate. The
[results guide](results/README.md) also records gaps in the archives, including
the missing historical configuration for the CIFAR-100 ViT runs.

To reproduce the intervals and both main paper tables:

```bash
python scripts/confidence_intervals.py
```

Outputs go into `runs/confidence/`.

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

[CITATION.cff](CITATION.cff) contains the citation details. Code is under the
[MIT license](LICENSE).
