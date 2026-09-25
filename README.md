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

The PDF is the revised September 22, 2026 paper, including 114 additional fits
with the selected schedules held fixed. The headline comparisons now have at
least ten paired seeds; some final-epoch measurements cover only the five new
pairs because the historical runs did not save that endpoint. The
[updated tables](results/confidence_intervals.md) give the counts and confidence
intervals. The original [camera-ready version is on arXiv](https://arxiv.org/pdf/2605.21648v2).

## Finding things

| Path | Purpose |
|---|---|
| [paper.pdf](paper.pdf), [manuscript/](manuscript/) | Paper and LaTeX source |
| [notebooks/](notebooks/) | Original calculations, experiments, and plots |
| [benchmarks/](benchmarks/README.md) | Command-line runners for additional datasets and seed extensions |
| [utils/](utils/) | Shared notebook schedules, training, and result loading |
| [scripts/](scripts/) | Rebuild figures and confidence intervals from saved measurements |
| [results/](results/README.md) | Archived measurements and their limitations |
| [tests/](tests/) | Numerical, checkpoint, and provenance checks |

New runs go into `runs/`; archived measurements stay in `results/`.

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

To reproduce figures and tables without training:

```bash
python scripts/plot_results.py          # Saved learning curves; add --sweeps for all
python scripts/plot_paper_results.py    # Rebuild the eight original paper figures
python scripts/plot_benchmarks.py       # Historical benchmark appendix curves
python scripts/plot_mean_field.py       # Critical exponents and scaling collapse
python scripts/confidence_intervals.py  # Paired intervals and both paper tables
```

The original-paper exporter updates `manuscript/figures/experiments/`; the other
commands write to `runs/figures/`, `runs/mean_field/`, and `runs/confidence/`.
All exporters share the presentation's colors through `utils/plot_style.py`.
The paper's exported figures are in [manuscript/figures/](manuscript/figures/).

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

The [seed-extension guide](results/seed_extension.md) records which comparisons
received new seeds and links their saved learning curves and exact settings.
Historical multi-profile figures retain their original run counts; the main
tables combine those measurements with the fixed-profile extensions.

The endpoint matters. Taking the lowest recorded test loss uses the test set to
choose an epoch, whereas a validation-selected checkpoint is chosen before its
test evaluation. The paper keeps these comparisons separate. The
[results guide](results/README.md) also records gaps in the archives, including
the missing historical configuration for the CIFAR-100 ViT runs.

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check .
ruff format --check .
```

When editing code, run `ruff format .` before these checks. Source hashes identify
the exact version of a run, so resume existing runs with their original checkout.

[CITATION.cff](CITATION.cff) contains the citation details. Code is under the
[MIT license](LICENSE).
